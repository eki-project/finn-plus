"""Map FINN+'s heterogeneous failure signals onto :class:`FailureCategory`.

FINN+ reports incompatibilities in several ways: typed ``FINNUserError`` s, plain
``Exception`` s with a recognizable message, bare ``assert`` s inside custom ops, and
warnings that are only logged. This module turns any of these into a structured
:class:`Failure` so the NAS loop gets a stable, machine-readable signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from finn.nas.result import FailureCategory
from finn.util.exception import FINNConfigurationError, FINNInternalError, FINNUserError

if TYPE_CHECKING:
    from onnx import NodeProto
    from qonnx.core.modelwrapper import ModelWrapper

#: Op types that FINN's streamlining is supposed to absorb/remove. If one of these survives
#: in between HW layers, the topology is not streamlinable rather than "unsupported".
STREAMLINABLE_OPS = frozenset(
    {
        "Mul",
        "Add",
        "Sub",
        "Div",
        "BatchNormalization",
        "Transpose",
        "Reshape",
        "Flatten",
        "Squeeze",
        "Unsqueeze",
        "Identity",
    }
)

#: Quantizers. If one survives the conversion or is named by a conversion error, FINN rejected
#: its configuration (bit width above the multithreshold limit, signedness, ...), which is a
#: property of the datatype and not of the topology.
QUANTIZER_OPS = frozenset({"Quant", "BipolarQuant", "BinaryQuant", "Trunc", "MultiThreshold"})

#: Op types that have an ``Infer*`` conversion to a HW layer. If one of these stays unmapped,
#: the cause is usually upstream (e.g. an unquantized input produced by an unsupported op), so
#: they are only blamed if nothing better can be found. A plain ``AveragePool`` is not listed:
#: only its quantized form (``QuantAvgPool2d``) is converted.
CONVERTIBLE_OPS = frozenset(
    {
        "MatMul",
        "Conv",
        "Im2Col",
        "MaxPool",
        "MaxPoolNHWC",
        "QuantAvgPool2d",
        "GlobalAveragePool",
        "Pad",
        "Upsample",
        "Resize",
        "Concat",
        "Split",
        "TopK",
        "Gather",
        "Softmax",
        "LayerNormalization",
        "Slice",
    }
)

#: Op types whose second input holds the weights.
WEIGHT_OPS = frozenset({"MatMul", "Conv"})

#: Ops (standard and HW) that only move data around. They are skipped when looking for the
#: producer of a tensor, and at the graph boundary they are no host-side computation worth
#: rejecting a candidate for (e.g. the NCHW -> NHWC ``Transpose`` in front of every CNN).
LAYOUT_OPS = frozenset(
    {
        "Transpose",
        "Reshape",
        "Flatten",
        "Squeeze",
        "Unsqueeze",
        "Identity",
        "Shuffle",
        "DuplicateStreams",
    }
)

#: Scale/shift ops with a constant operand: as standard ops, and as the elementwise HW layers
#: FINN converts them to. With a non-integer constant they are what streamlining failed to
#: absorb into a following threshold.
AFFINE_OPS = frozenset(
    {
        "Mul",
        "Add",
        "Sub",
        "Div",
        "ElementwiseMul",
        "ElementwiseAdd",
        "ElementwiseSub",
        "ElementwiseDiv",
    }
)

HW_DOMAIN = "finn.custom_op.fpgadataflow"

FRONTEND_STAGES = frozenset({"step_qonnx_to_finn", "step_tidy_up", "step_streamline"})
#: Stages that query the folded shapes of the HW layers for the first time. Several custom ops
#: only check their PE/SIMD constraints there, so that is where an illegal folding surfaces.
POST_FOLDING_STAGES = frozenset({"step_minimize_bit_width", "estimate"})

_RE_UNMAPPED_START = re.compile(r"starting with node (\S+)\.")
_RE_UNMAPPED_LIST = re.compile(r"Complete list of unmapped nodes: (.*)$", re.DOTALL)
_RE_NO_HW_VARIANT = re.compile(r"Node (\S+) with optype (\S+) has no hw implementation variant")
_RE_BAD_IMPL_STYLE = re.compile(r"Invalid value for attribute preferred_impl_style")
_RE_SETFOLDING_UNHANDLED = re.compile(r"SetFolding doesn't know how to handle op_type (\S+)")
_RE_LEADING_NODE = re.compile(r"^(\S+): ")
_RE_FOLDING_ATTR = re.compile(r"\b(PE|SIMD)\b")
_RE_BOUNDARY_UNMAPPED = re.compile(
    r"nodes at the start/end of the graph will not be mapped to the accelerator.*?: (.*)$",
    re.DOTALL,
)


@dataclass
class Failure:
    """A classified failure."""

    category: FailureCategory
    stage: str
    node: str | None = None
    op_type: str | None = None
    message: str | None = None


def first_line(exc: BaseException) -> str:
    """Return a compact one-line description of an exception."""
    text = str(exc).strip() or exc.__class__.__name__
    return " ".join(text.split())[:500]


def lookup_node(model: ModelWrapper | None, name: str | None) -> tuple[str, str] | None:
    """Return ``(op_type, domain)`` of the node called ``name`` in ``model``, if any."""
    if model is None or name is None:
        return None
    for node in model.graph.node:
        if node.name == name:
            return node.op_type, node.domain
    return None


def node_from_traceback(exc: BaseException) -> tuple[str | None, str | None]:
    """Walk the traceback for the innermost frame that belongs to a custom op or holds a node.

    Returns ``(node_name, op_type)`` or ``(None, None)``.
    """
    from onnx import NodeProto

    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

    found: tuple[str | None, str | None] = (None, None)
    tb = exc.__traceback__
    while tb is not None:
        loc = tb.tb_frame.f_locals
        inst = loc.get("self")
        if isinstance(inst, HWCustomOp):
            found = (inst.onnx_node.name, inst.onnx_node.op_type)
        else:
            for key in ("node", "n"):
                cand = loc.get(key)
                if isinstance(cand, NodeProto):
                    found = (cand.name, cand.op_type)
                    break
        tb = tb.tb_next
    return found


def unhandled_folding_ops(messages: list[str]) -> list[str]:
    """Extract op types that SetFolding reported it cannot handle."""
    ops: list[str] = []
    for msg in messages:
        m = _RE_SETFOLDING_UNHANDLED.search(msg)
        if m and m.group(1) not in ops:
            ops.append(m.group(1))
    return ops


def boundary_unmapped_nodes(messages: list[str], model: ModelWrapper | None) -> list[list[str]]:
    """Extract ``[name, op_type]`` of boundary nodes FINN reported as staying on the host."""
    result: list[list[str]] = []
    for msg in messages:
        m = _RE_BOUNDARY_UNMAPPED.search(msg)
        if not m:
            continue
        for name in (n.strip() for n in m.group(1).split(",")):
            if not name:
                continue
            info = lookup_node(model, name)
            result.append([name, info[0] if info else "?"])
    return result


def _is_float(model: ModelWrapper, tensor: str) -> bool:
    """Return whether ``tensor`` carries a non-integer datatype."""
    return not model.get_tensor_datatype(tensor).is_integer()


def _reaches_hw(model: ModelWrapper, node: NodeProto, upstream: bool) -> bool:
    """Return whether a HW layer lies upstream (or else downstream) of ``node``."""
    step = model.find_direct_predecessors if upstream else model.find_direct_successors
    seen: set[str] = set()
    stack = [node]
    while stack:
        for nxt in step(stack.pop()) or []:
            if nxt.domain.startswith(HW_DOMAIN):
                return True
            if nxt.output[0] not in seen:
                seen.add(nxt.output[0])
                stack.append(nxt)
    return False


def _float_input(model: ModelWrapper, node: NodeProto) -> tuple[bool, NodeProto | None]:
    """Look for a non-integer dynamic input of ``node`` and the node responsible for it.

    Returns whether there is such an input, and its nearest producer that is not a mere
    layout op (``None`` if the data comes straight from the graph input).
    """
    for inp in node.input:
        # Empty names are omitted optional inputs.
        if not inp or model.get_initializer(inp) is not None or not _is_float(model, inp):
            continue
        producer = model.find_producer(inp)
        while producer is not None and producer.op_type in LAYOUT_OPS:
            producer = model.find_producer(producer.input[0])
        return True, producer
    return False, None


def is_unabsorbed_affine(model: ModelWrapper, node: NodeProto) -> bool:
    """Return whether ``node`` scales/shifts by a non-integer constant.

    That is what streamlining leaves behind when no threshold can absorb a scale or bias.
    FINN turns it into a float elementwise HW layer instead of leaving it unmapped, so it
    has to be recognized by its datatype.
    """
    return node.op_type in AFFINE_OPS and any(
        model.get_initializer(i) is not None and _is_float(model, i) for i in node.input
    )


def _quant_bits(model: ModelWrapper, node: NodeProto) -> str:
    """Return ``", N bit"`` for a ``Quant`` node with a constant bit width, else ``""``."""
    if node.op_type != "Quant" or len(node.input) < 4:
        return ""
    bits = model.get_initializer(node.input[3])
    return f", {int(bits.item())} bit" if bits is not None and bits.size == 1 else ""


def _float_cause(stage: str, model: ModelWrapper, node: NodeProto, msg: str) -> Failure | None:
    """Explain why the convertible ``node`` stayed unmapped, if non-integer data is the reason.

    Convertible layers only need integer operands. Unquantized weights are the layer's own
    fault; a non-integer input is the fault of whatever produced it.
    """
    desc = f"{node.name} ({node.op_type})"
    if (
        node.op_type in WEIGHT_OPS
        and len(node.input) > 1
        and model.get_initializer(node.input[1]) is not None
        and _is_float(model, node.input[1])
    ):
        return Failure(
            FailureCategory.DATATYPE_UNSUPPORTED,
            stage,
            node.name,
            node.op_type,
            f"{desc} has non-integer (unquantized) weights. [{msg}]",
        )
    has_float, producer = _float_input(model, node)
    if not has_float:
        return None
    if producer is None:
        return Failure(
            FailureCategory.DATATYPE_UNSUPPORTED,
            stage,
            node.name,
            node.op_type,
            f"{desc} receives the unquantized graph input. [{msg}]",
        )
    category = (
        FailureCategory.STREAMLINE_INCOMPLETE
        if is_unabsorbed_affine(model, producer)
        else FailureCategory.DATATYPE_UNSUPPORTED
    )
    return Failure(
        category,
        stage,
        producer.name,
        producer.op_type,
        f"{producer.name} ({producer.op_type}) feeds non-integer data into {desc}, which "
        f"therefore cannot be mapped. [{msg}]",
    )


def classify_unmapped(
    stage: str,
    names: list[str],
    msg: str,
    model: ModelWrapper | None,
    interior: bool = True,
) -> Failure:
    """Pick the most likely root cause among the unmapped nodes ``names`` and categorize it.

    FINN lists every unmapped node: harmless ones at the graph boundary and layers that
    only stayed unmapped as a consequence of the culprit included. With ``interior`` only
    nodes between HW layers are considered (if any). Priority: an op FINN has no conversion
    for, a quantizer FINN did not convert, a convertible layer starved of integer operands,
    an op streamlining should have absorbed, and finally any convertible or other node.
    """
    wanted = set(names)
    nodes = [n for n in model.graph.node if n.name in wanted] if model is not None else []
    if model is None or not nodes:
        first = names[0] if names else None
        return Failure(FailureCategory.UNMAPPED_LAYER, stage, first, None, msg)
    if interior:
        between = [n for n in nodes if _reaches_hw(model, n, True) and _reaches_hw(model, n, False)]
        nodes = between or nodes

    def blame(category: FailureCategory, node: NodeProto, reason: str) -> Failure:
        """Build the failure blaming ``node``."""
        return Failure(category, stage, node.name, node.op_type, f"{reason} [{msg}]")

    known = STREAMLINABLE_OPS | CONVERTIBLE_OPS | QUANTIZER_OPS
    for node in nodes:
        if not node.domain.startswith("finn.") and node.op_type not in known:
            return blame(
                FailureCategory.UNSUPPORTED_OP,
                node,
                f"FINN has no hardware conversion for {node.name} ({node.op_type}).",
            )
    for node in nodes:
        if node.op_type in QUANTIZER_OPS:
            return blame(
                FailureCategory.DATATYPE_UNSUPPORTED,
                node,
                f"Quantizer {node.name} ({node.op_type}{_quant_bits(model, node)}) was not "
                "converted to a hardware layer.",
            )
    for node in nodes:
        if node.op_type in CONVERTIBLE_OPS:
            failure = _float_cause(stage, model, node, msg)
            if failure is not None:
                return failure
    for node in nodes:
        if node.op_type in STREAMLINABLE_OPS:
            return blame(
                FailureCategory.STREAMLINE_INCOMPLETE,
                node,
                f"Streamlining left {node.name} ({node.op_type}) between hardware layers.",
            )
    node = next((n for n in nodes if n.op_type in CONVERTIBLE_OPS), nodes[0])
    return blame(
        FailureCategory.UNMAPPED_LAYER,
        node,
        f"{node.name} ({node.op_type}) was not converted to a hardware layer.",
    )


def classify_interior_float(stage: str, floats: list[list[str]], model: ModelWrapper) -> Failure:
    """Classify non-integer datatypes on the interior HW nodes ``floats`` (``[name, op]``).

    The first node in graph order is where the data stops being integer. If that is a
    scale/shift by a float constant, streamlining could not absorb it (e.g. residual
    branches with different scales); anything else is a float operation proper.
    """
    name, op_type = floats[0]
    node = next((n for n in model.graph.node if n.name == name), None)
    listed = ", ".join(n for n, _ in floats)
    if node is not None and is_unabsorbed_affine(model, node):
        return Failure(
            FailureCategory.STREAMLINE_INCOMPLETE,
            stage,
            name,
            op_type,
            f"Unabsorbed float scale/shift {name}; non-integer interior HW node(s): {listed}",
        )
    return Failure(
        FailureCategory.DATATYPE_UNSUPPORTED,
        stage,
        name,
        op_type,
        f"Non-integer datatype on interior HW node(s): {listed}",
    )


def _classify_partition_error(
    stage: str, msg: str, last_model: ModelWrapper | None
) -> Failure | None:
    """Classify the unmapped-layer error of the partitioning step, or return ``None``."""
    m = _RE_UNMAPPED_START.search(msg)
    if not m:
        return None
    names = [m.group(1)]
    m_list = _RE_UNMAPPED_LIST.search(msg)
    if m_list:
        listed = [n.strip() for n in m_list.group(1).split(",") if n.strip()]
        names = listed + [n for n in names if n not in listed]
    return classify_unmapped(stage, names, msg, last_model)


def classify_failure(
    stage: str,
    exc: BaseException,
    last_model: ModelWrapper | None = None,
) -> Failure:
    """Classify an exception raised by build stage ``stage``.

    Parameters
    ----------
    stage
        Name of the build step / oracle stage that raised.
    exc
        The exception.
    last_model
        The model as it was *before* the failing stage, used to look up op types of nodes
        that are only named in the exception message.
    """
    msg = first_line(exc)
    tb_node, tb_op = node_from_traceback(exc)

    # 1. Typed FINN exceptions carrying node information (future upstream change).
    node_attr = getattr(exc, "node_name", None)
    op_attr = getattr(exc, "op_type", None)
    if isinstance(exc, FINNConfigurationError):
        return Failure(FailureCategory.CONFIG_ERROR, stage, node_attr, op_attr, msg)

    # 2. Message patterns.
    m = _RE_NO_HW_VARIANT.search(msg)
    if m:
        return Failure(FailureCategory.UNSUPPORTED_OP, stage, m.group(1), m.group(2), msg)
    if _RE_BAD_IMPL_STYLE.search(msg):
        return Failure(FailureCategory.CONFIG_ERROR, stage, None, None, msg)
    partition = _classify_partition_error(stage, msg, last_model)
    if partition is not None:
        return partition
    if "Only a single StreamingDataflowPartition" in msg:
        return Failure(FailureCategory.UNMAPPED_LAYER, stage, None, None, msg)

    # 3. Stage defaults. FINN's "<node name>: <problem>" messages name the offending node
    # more precisely than the traceback, which only knows the node being processed.
    node, op_type = (node_attr or tb_node), (op_attr or tb_op)
    m = _RE_LEADING_NODE.match(msg)
    named = lookup_node(last_model, m.group(1)) if m else None
    if m and named is not None and node_attr is None:
        node, op_type = m.group(1), named[0]
    if stage == "export":
        return Failure(FailureCategory.EXPORT_FAILED, stage, None, None, msg)
    if stage in FRONTEND_STAGES:
        return Failure(FailureCategory.FRONTEND_FAILED, stage, node, op_type, msg)
    if stage == "step_convert_to_hw":
        if op_type in QUANTIZER_OPS:
            cat = FailureCategory.DATATYPE_UNSUPPORTED
        else:
            cat = FailureCategory.UNSUPPORTED_OP if node else FailureCategory.INTERNAL_ERROR
        return Failure(cat, stage, node, op_type, msg)
    if stage == "step_specialize_layers":
        return Failure(FailureCategory.UNSUPPORTED_OP, stage, node, op_type, msg)
    if stage == "folding" or (stage in POST_FOLDING_STAGES and _RE_FOLDING_ATTR.search(msg)):
        return Failure(FailureCategory.FOLDING_INFEASIBLE, stage, node, op_type, msg)
    if stage in ("step_minimize_bit_width", "step_minimize_bit_width_initial"):
        return Failure(FailureCategory.DATATYPE_UNSUPPORTED, stage, node, op_type, msg)
    if stage == "step_hw_codegen":
        # Custom ops reject a configuration with a bare assert or a FINNUserError; anything
        # else (missing files, KeyErrors, ...) is not the candidate's fault.
        cat = (
            FailureCategory.CODEGEN_ASSERT
            if isinstance(exc, (AssertionError, FINNUserError))
            else FailureCategory.INTERNAL_ERROR
        )
        return Failure(cat, stage, node, op_type, msg)
    if isinstance(exc, FINNInternalError):
        return Failure(FailureCategory.INTERNAL_ERROR, stage, node, op_type, msg)
    return Failure(FailureCategory.INTERNAL_ERROR, stage, node, op_type, msg)

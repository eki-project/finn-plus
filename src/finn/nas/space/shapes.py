"""Analytic shape inference, parameter counting and validation of genotypes (no torch).

The rules here are exactly the ones that make a network well-formed in PyTorch; nothing in
this module knows about FINN. ``apply_terminal`` is the single source of truth for whether an
operation is applicable to a shape, and is used by validation and by the sampler alike.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any

from finn.nas.space.config import AGGREGATIONS, HEAD_KINDS
from finn.nas.space.genotype import (
    VERSION,
    Genotype,
    InvalidGenotypeError,
    Node,
    Path,
    get_node,
    is_inner,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from finn.nas.space.config import SpaceConfig

#: ``(C, H, W)`` in image mode or ``(F,)`` in vector mode; the batch dimension is implicit.
Shape = tuple[int, ...]

#: Hyperparameter fields per terminal kind and their type. ``out_channels`` of a conv is
#: handled separately because it only exists for non-depthwise convolutions.
TERMINAL_FIELDS: dict[str, dict[str, type]] = {
    "conv": {"kernel": int, "stride": int, "padding": str, "depthwise": bool, "weight_bits": int},
    "linear": {"out_features": int, "weight_bits": int},
    "quant_act": {"bits": int},
    "relu_float": {},
    "batchnorm": {},
    "maxpool": {"kernel": int, "stride": int},
    "avgpool": {"kernel": int},
    "flatten": {},
    "identity": {},
}
PADDINGS = ("same", "valid")


@dataclass(frozen=True)
class NodeInfo:
    """Derived facts about one body node (and, for inner nodes, its whole subtree)."""

    in_shape: Shape
    out_shape: Shape
    #: Trainable parameters in the subtree.
    params: int
    #: Number of tree nodes in the subtree, including the node itself.
    size: int


@dataclass(frozen=True)
class GenotypeInfo:
    """Result of shape inference: per-node shapes plus the totals the budgets refer to."""

    input_shape: Shape
    #: Indexed by path; ``()`` is the body root.
    nodes: dict[Path, NodeInfo]
    #: Input features of the head's linear layer.
    head_features: int
    head_params: int
    output_shape: Shape

    @property
    def body_out(self) -> Shape:
        """Output shape of the body."""
        return self.nodes[()].out_shape

    @property
    def params(self) -> int:
        """Trainable parameters of body and head."""
        return self.nodes[()].params + self.head_params

    @property
    def num_nodes(self) -> int:
        """Number of tree nodes of the body."""
        return self.nodes[()].size

    @property
    def depth(self) -> int:
        """Nesting depth: length of the longest path."""
        return max(len(path) for path in self.nodes)


def is_image(shape: Shape) -> bool:
    """Return whether ``shape`` is a feature map (as opposed to a vector)."""
    return len(shape) == 3


def same_padding(kernel: int) -> int:
    """Return the per-side padding used for ``"same"`` convolutions.

    PyTorch's own ``padding="same"`` rejects strided convolutions, so "same" is defined as the
    symmetric padding ``(kernel - 1) // 2``; the output size is ``ceil(size / stride)`` for odd
    kernels.
    """
    return (kernel - 1) // 2


def _check_values(node: Node, fields: dict[str, type], what: str) -> None:
    """Require the ``fields`` of ``node`` to exist with the given type; ints must be >= 1."""
    for name, expected in fields.items():
        value = node.get(name)
        # bool is a subclass of int, so compare the exact type.
        if type(value) is not expected:
            raise InvalidGenotypeError(f"{what}: {name} must be {expected.__name__}, got {value!r}")
        if expected is int and value < 1:
            raise InvalidGenotypeError(f"{what}: {name} must be >= 1, got {value}")


def _check_fields(node: Node, fields: dict[str, type], what: str) -> None:
    """Require ``node`` to have exactly ``fields`` besides ``kind``, with valid values.

    Exact key sets keep the canonical JSON (and thus the hash) of a network unique.
    """
    keys = set(node) - {"kind"}
    if keys != set(fields):
        raise InvalidGenotypeError(f"{what} must have fields {sorted(fields)}, got {sorted(keys)}")
    _check_values(node, fields, what)


def _require_image(shape: Shape, what: str) -> None:
    """Raise unless ``shape`` is a feature map."""
    if not is_image(shape):
        raise InvalidGenotypeError(f"{what} needs a feature map, got vector {shape}")


def _window_out(size: int, kernel: int, stride: int, pad: int, what: str) -> int:
    """Return the output size of a sliding window along one dimension."""
    if size + 2 * pad < kernel:
        raise InvalidGenotypeError(f"{what}: kernel {kernel} does not fit size {size} (pad {pad})")
    return (size + 2 * pad - kernel) // stride + 1


def _conv(node: Node, shape: Shape) -> tuple[Shape, int]:
    """Shape and weight count of a convolution without bias."""
    _require_image(shape, "conv")
    channels, height, width = shape
    if node["padding"] not in PADDINGS:
        raise InvalidGenotypeError(f"conv: padding must be one of {PADDINGS}")
    kernel, stride = node["kernel"], node["stride"]
    pad = same_padding(kernel) if node["padding"] == "same" else 0
    out_h = _window_out(height, kernel, stride, pad, "conv")
    out_w = _window_out(width, kernel, stride, pad, "conv")
    if node["depthwise"]:
        return (channels, out_h, out_w), kernel * kernel * channels
    out_channels = node["out_channels"]
    return (out_channels, out_h, out_w), kernel * kernel * channels * out_channels


def _linear(node: Node, shape: Shape) -> tuple[Shape, int]:
    """Shape and weight count of a fully connected layer without bias."""
    if is_image(shape):
        raise InvalidGenotypeError(f"linear needs a vector, got feature map {shape}")
    return (node["out_features"],), shape[0] * node["out_features"]


def _maxpool(node: Node, shape: Shape) -> tuple[Shape, int]:
    """Shape of an unpadded max pooling."""
    _require_image(shape, "maxpool")
    channels, height, width = shape
    out_h = _window_out(height, node["kernel"], node["stride"], 0, "maxpool")
    out_w = _window_out(width, node["kernel"], node["stride"], 0, "maxpool")
    return (channels, out_h, out_w), 0


def _avgpool(node: Node, shape: Shape) -> tuple[Shape, int]:
    """Shape of an unpadded average pooling with stride = kernel."""
    _require_image(shape, "avgpool")
    channels, height, width = shape
    out_h = _window_out(height, node["kernel"], node["kernel"], 0, "avgpool")
    out_w = _window_out(width, node["kernel"], node["kernel"], 0, "avgpool")
    return (channels, out_h, out_w), 0


def _flatten(_node: Node, shape: Shape) -> tuple[Shape, int]:
    """Shape after flattening a feature map."""
    _require_image(shape, "flatten")
    return (prod(shape),), 0


def _batchnorm(_node: Node, shape: Shape) -> tuple[Shape, int]:
    """Batch normalization keeps the shape and has a scale and a shift per channel/feature."""
    return shape, 2 * shape[0]


def _unchanged(_node: Node, shape: Shape) -> tuple[Shape, int]:
    """Parameter-free operations that keep the shape."""
    return shape, 0


_TERMINALS: dict[str, Callable[[Node, Shape], tuple[Shape, int]]] = {
    "conv": _conv,
    "linear": _linear,
    "quant_act": _unchanged,
    "relu_float": _unchanged,
    "batchnorm": _batchnorm,
    "maxpool": _maxpool,
    "avgpool": _avgpool,
    "flatten": _flatten,
    "identity": _unchanged,
}


def apply_terminal(node: Node, shape: Shape) -> tuple[Shape, int]:
    """Return output shape and trainable parameter count of terminal ``node`` on ``shape``.

    Raises :class:`InvalidGenotypeError` if the node is malformed or not applicable.
    """
    kind = node.get("kind")
    if kind not in _TERMINALS:
        raise InvalidGenotypeError(f"unknown node kind {kind!r}")
    fields = TERMINAL_FIELDS[kind]
    if kind == "conv" and node.get("depthwise") is False:
        fields = {**fields, "out_channels": int}
    _check_fields(node, fields, kind)
    return _TERMINALS[kind](node, shape)


def aggregate(agg: str, shapes: list[Shape]) -> Shape:
    """Return the shape after merging branch outputs ``shapes`` with ``agg``."""
    first = shapes[0]
    if agg == "add":
        if any(shape != first for shape in shapes):
            raise InvalidGenotypeError(f"add needs equal branch shapes, got {shapes}")
        return first
    if agg == "concat":
        if any(len(shape) != len(first) or shape[1:] != first[1:] for shape in shapes):
            raise InvalidGenotypeError(f"concat needs equal mode and H, W, got {shapes}")
        return (sum(shape[0] for shape in shapes), *first[1:])
    raise InvalidGenotypeError(f"unknown aggregation {agg!r}; known: {AGGREGATIONS}")


@contextmanager
def _located(path: Path) -> Iterator[None]:
    """Prefix errors raised inside the block with the path of the offending node."""
    try:
        yield
    except InvalidGenotypeError as exc:
        raise InvalidGenotypeError(f"node {list(path)}: {exc}") from None


def _children(node: Node, fields: dict[str, type], minimum: int, path: Path) -> list[Any]:
    """Return the children of an inner node after checking the node's own fields."""
    with _located(path):
        _check_fields(node, fields, node["kind"])
        if len(node["children"]) < minimum:
            raise InvalidGenotypeError(f"{node['kind']} needs at least {minimum} children")
    return node["children"]


def _walk(node: Any, shape: Shape, path: Path, nodes: dict[Path, NodeInfo]) -> NodeInfo:
    """Infer the subtree at ``path`` for input ``shape`` and record every node in ``nodes``."""
    if not isinstance(node, dict):
        raise InvalidGenotypeError(f"node {list(path)}: not a dict")
    kind = node.get("kind")
    if kind == "sequential":
        out, params, size = shape, 0, 1
        for index, child in enumerate(_children(node, {"children": list}, 1, path)):
            info = _walk(child, out, (*path, index), nodes)
            out, params, size = info.out_shape, params + info.params, size + info.size
    elif kind == "branching":
        children = _children(node, {"children": list, "agg": str}, 2, path)
        infos = [_walk(child, shape, (*path, index), nodes) for index, child in enumerate(children)]
        with _located(path):
            out = aggregate(node["agg"], [info.out_shape for info in infos])
        params = sum(info.params for info in infos)
        size = 1 + sum(info.size for info in infos)
    else:
        with _located(path):
            out, params = apply_terminal(node, shape)
        size = 1
    nodes[path] = NodeInfo(shape, out, params, size)
    return nodes[path]


def infer_shapes(genotype: Genotype) -> GenotypeInfo:
    """Compute the shapes and parameter counts of every node of a self-contained genotype.

    Raises :class:`InvalidGenotypeError` (with the reason and the node path) if the genotype
    is ill-formed. Budgets and choice lists of a config are not checked, see :func:`validate`.
    """
    if not isinstance(genotype, dict):
        raise InvalidGenotypeError("genotype is not a dict")
    expected = {"version", "input_shape", "num_classes", "input_bits", "body", "head"}
    if set(genotype) != expected:
        raise InvalidGenotypeError(f"genotype must have keys {sorted(expected)}")
    if genotype["version"] != VERSION:
        raise InvalidGenotypeError(f"unsupported genotype version {genotype['version']!r}")
    shape = genotype["input_shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(type(dim) is not int or dim < 1 for dim in shape)
    ):
        raise InvalidGenotypeError(f"input_shape must be [C, H, W], got {shape!r}")
    _check_values(genotype, {"num_classes": int, "input_bits": int}, "genotype")
    head = genotype["head"]
    if not isinstance(head, dict) or head.get("kind") not in HEAD_KINDS:
        raise InvalidGenotypeError(f"head must be a dict with kind in {HEAD_KINDS}")
    _check_fields(head, {"weight_bits": int}, "head")

    nodes: dict[Path, NodeInfo] = {}
    body_out = _walk(genotype["body"], tuple(shape), (), nodes).out_shape
    # Global average pooling only exists for feature maps; a vector is fed to the head as is.
    features = body_out[0] if is_image(body_out) and head["kind"] == "gap" else prod(body_out)
    num_classes = genotype["num_classes"]
    return GenotypeInfo(tuple(shape), nodes, features, features * num_classes, (num_classes,))


def _check_membership(genotype: Genotype, info: GenotypeInfo, config: SpaceConfig) -> None:
    """Require every gene to be a value the config's choice lists can produce."""
    wrapper = {
        "input_bits": genotype["input_bits"],
        "head.kind": genotype["head"]["kind"],
        "head.weight_bits": genotype["head"]["weight_bits"],
    }
    for name, allowed in config.wrapper_choices().items():
        if wrapper[name] not in allowed:
            raise InvalidGenotypeError(f"{name}={wrapper[name]!r} is not in the space {allowed}")
    for path in info.nodes:
        node = get_node(genotype, path)
        if is_inner(node):
            continue
        if config.terminal_weights.get(node["kind"], 0) <= 0:
            raise InvalidGenotypeError(f"node {list(path)}: {node['kind']} is not in the space")
        for name, allowed in config.choices(node["kind"]).items():
            if name in node and node[name] not in allowed:
                raise InvalidGenotypeError(
                    f"node {list(path)}: {name}={node[name]!r} is not in the space {allowed}"
                )


def validate(genotype: Genotype, config: SpaceConfig | None = None) -> GenotypeInfo:
    """Check that ``genotype`` is well-formed and, given ``config``, a member of that space.

    Membership means: same task, within the depth/node/parameter budgets, and every terminal
    and hyperparameter value is one the config can sample. Returns the shape information.
    """
    info = infer_shapes(genotype)
    if config is None:
        return info
    if info.input_shape != tuple(config.input_shape) or info.output_shape != (config.num_classes,):
        raise InvalidGenotypeError(
            f"task mismatch: genotype maps {info.input_shape} -> {info.output_shape}"
        )
    for name, value, limit in (
        ("depth", info.depth, config.max_depth),
        ("nodes", info.num_nodes, config.max_nodes),
        ("params", info.params, config.max_params),
    ):
        if value > limit:
            raise InvalidGenotypeError(f"budget exceeded: {name} {value} > {limit}")
    _check_membership(genotype, info, config)
    return info


def count_params(genotype: Genotype) -> int:
    """Return the number of trainable parameters of the network described by ``genotype``.

    Counted are the weights of conv/linear layers (there are no biases), the scale and shift
    of batch normalization, and the head's linear layer. Quantizers contribute nothing: all
    scales are constants.
    """
    return infer_shapes(genotype).params

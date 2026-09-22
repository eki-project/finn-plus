"""Utilities for the simulation-based verification steps of the dataflow builder.

The verification steps compare the top-level output of a cppsim/rtlsim execution against the
expected output. When that comparison fails (or deviates), it does not tell *which* node
introduced the deviation. The helpers in this module execute the same folded graph with the
Python reference implementation of every layer and compare the two executions node by node.
"""

import numpy as np
from copy import deepcopy
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper

from finn.core.onnx_exec import execute_onnx
from finn.util.exception import FINNUserError
from finn.util.logging import log

#: Domains of the specialized (backend-specific) custom op variants
SPECIALIZED_DOMAINS = ("finn.custom_op.fpgadataflow.hls", "finn.custom_op.fpgadataflow.rtl")
#: Domain of the hardware abstraction layer custom ops, which execute in Python
HW_DOMAIN = "finn.custom_op.fpgadataflow"


def despecialize_model(model: ModelWrapper) -> ModelWrapper:
    """Return a copy of the model in which every HLS/RTL-specialized node is replaced by its
    hardware abstraction layer counterpart (e.g. ``MVAU_hls`` -> ``MVAU``).

    The abstraction layer ops implement ``execute_node`` in Python, so the returned model can
    be executed with :func:`execute_onnx` to obtain a golden reference for every tensor of the
    folded graph. Node and tensor names are left untouched, which allows a direct, name-based
    comparison with a cppsim/rtlsim execution of the original model.

    Raises:
        FINNUserError: if a specialized node has no abstraction layer counterpart.
    """
    from qonnx.custom_op.registry import getCustomOp

    from finn.custom_op.fpgadataflow import custom_op as hw_custom_ops

    model = deepcopy(model)
    # a model-level exec_mode ("rtlsim") would route execution to the stitched IP
    model.set_metadata_prop("exec_mode", "")
    for node in model.graph.node:
        if node.domain not in SPECIALIZED_DOMAINS:
            continue
        base_op_type = node.op_type.rsplit("_", 1)[0]
        if base_op_type not in hw_custom_ops:
            raise FINNUserError(
                f"Node {node.name} ({node.op_type}) has no Python-executable hardware "
                f"abstraction layer counterpart ({base_op_type})"
            )
        node.op_type = base_op_type
        node.domain = HW_DOMAIN
        # The node-level exec_mode ("cppsim"/"rtlsim") was set for the simulation and is kept
        # by the attribute copy. Ops whose execute_node dispatches on it (ReplicateStream,
        # attention, ...) would then simulate instead of executing in Python, or fail on
        # missing code generation directories: select the Python mode where the op offers
        # one, and clear the attribute otherwise (those ops always execute in Python).
        inst = getCustomOp(node)
        allowed_modes = inst.get_nodeattr_types().get("exec_mode", (None, None, None, set()))[3]
        if "python" in allowed_modes:
            inst.set_nodeattr("exec_mode", "python")
        else:
            for attr in list(node.attribute):
                if attr.name == "exec_mode":
                    node.attribute.remove(attr)
    return model


def compare_execution_contexts(
    model: ModelWrapper,
    sim_ctx: dict[str, np.ndarray],
    ref_ctx: dict[str, np.ndarray],
    sim_prefix: str = "",
    atol: float = 1e-3,
    rtol: float = 1e-5,
    sim_name_map: dict[str, str] | None = None,
) -> list[dict]:
    """Compare the output tensors of every node of ``model`` between two execution contexts.

    Args:
        model: The executed model, its node order defines the order of the result.
        sim_ctx: Execution context of the simulated (cppsim/rtlsim) execution.
        ref_ctx: Execution context of the reference (Python) execution.
        sim_prefix: Prefix of the tensor names in ``sim_ctx`` (the parent model's
            StreamingDataflowPartition node prefixes the tensors of its child model).
        atol: Absolute tolerance per element (as in ``np.isclose``).
        rtol: Relative tolerance per element (as in ``np.isclose``).
        sim_name_map: Explicit tensor name -> ``sim_ctx`` key mapping that takes precedence
            over ``sim_prefix`` (the child model's outputs appear under the parent's names).

    Returns:
        One dict per node output tensor, in graph order, with the keys ``node``, ``op_type``,
        ``tensor``, ``shape``, ``num_elements``, ``num_mismatch``, ``max_abs_err``,
        ``mean_abs_err`` and ``status`` (``ok``, ``deviates``, ``missing`` or ``shape``).
    """
    rows = []
    for node in model.graph.node:
        for tensor in node.output:
            row = {
                "node": node.name,
                "op_type": node.op_type,
                "tensor": tensor,
                "shape": None,
                "num_elements": 0,
                "num_mismatch": 0,
                "max_abs_err": 0.0,
                "mean_abs_err": 0.0,
                "status": "missing",
            }
            rows.append(row)
            sim_name = (sim_name_map or {}).get(tensor, sim_prefix + tensor)
            if sim_name not in sim_ctx or tensor not in ref_ctx:
                continue
            sim = np.asarray(sim_ctx[sim_name], dtype=np.float64)
            ref = np.asarray(ref_ctx[tensor], dtype=np.float64)
            row["shape"] = list(ref.shape)
            if sim.shape != ref.shape:
                if sim.size != ref.size:
                    row["status"] = "shape"
                    continue
                sim = sim.reshape(ref.shape)
            abs_err = np.abs(sim - ref)
            row["num_elements"] = int(ref.size)
            row["num_mismatch"] = int(np.count_nonzero(~np.isclose(sim, ref, atol=atol, rtol=rtol)))
            row["max_abs_err"] = float(abs_err.max()) if ref.size else 0.0
            row["mean_abs_err"] = float(abs_err.mean()) if ref.size else 0.0
            row["status"] = "deviates" if row["num_mismatch"] > 0 else "ok"
    return rows


def first_deviation(rows: list[dict]) -> dict | None:
    """Return the first row (in graph order) whose tensor deviates, or None."""
    for row in rows:
        if row["status"] in ("deviates", "shape"):
            return row
    return None


def write_nodewise_report(path: "str | Path", rows: list[dict], header: str = "") -> None:
    """Write the result of :func:`compare_execution_contexts` as a human-readable table."""
    lines = []
    if header:
        lines.append(header)
    lines.append(
        f"{'Node':<40} {'Op':<32} {'Tensor':<28} {'Elements':>10} {'Mismatch':>10} "
        f"{'MaxAbsErr':>12} {'MeanAbsErr':>12}  Status"
    )
    lines.append("-" * 160)
    for row in rows:
        lines.append(
            f"{row['node'][:39]:<40} {row['op_type'][:31]:<32} {row['tensor'][:27]:<28} "
            f"{row['num_elements']:>10} {row['num_mismatch']:>10} "
            f"{row['max_abs_err']:>12.4e} {row['mean_abs_err']:>12.4e}  {row['status']}"
        )
    lines.append("")
    deviation = first_deviation(rows)
    num_deviating = sum(1 for row in rows if row["status"] in ("deviates", "shape"))
    num_missing = sum(1 for row in rows if row["status"] == "missing")
    lines.append(f"Deviating tensors:            {num_deviating}/{len(rows)}")
    lines.append(f"Tensors not captured:         {num_missing}/{len(rows)}")
    if deviation is None:
        lines.append("First deviating node:         (none)")
    else:
        lines.append(
            f"First deviating node:         {deviation['node']} ({deviation['op_type']}), "
            f"tensor {deviation['tensor']}, {deviation['num_mismatch']} of "
            f"{deviation['num_elements']} elements, max abs err {deviation['max_abs_err']:.4e}"
        )
    with Path(path).open("w") as f:
        f.write("\n".join(lines) + "\n")


def nodewise_verification(
    child_model: ModelWrapper,
    child_inputs: dict[str, np.ndarray],
    sim_ctx: dict[str, np.ndarray],
    sim_prefix: str,
    report_path: "str | Path",
    atol: float,
    rtol: float,
    header: str = "",
    sim_name_map: dict[str, str] | None = None,
) -> list[dict] | None:
    """Execute the Python reference of ``child_model`` and compare it node by node against the
    simulated execution context, writing the table to ``report_path``.

    Returns the comparison rows, or None if no Python reference could be obtained (a warning is
    logged in that case, the verification step itself is unaffected).
    """
    try:
        ref_model = despecialize_model(child_model)
        ref_ctx = execute_onnx(ref_model, child_inputs, return_full_exec_context=True)
    except Exception as e:
        log.warning(f"Skipping node-wise verification report, Python reference failed: {e}")
        return None
    rows = compare_execution_contexts(
        child_model, sim_ctx, ref_ctx, sim_prefix, atol, rtol, sim_name_map
    )
    write_nodewise_report(report_path, rows, header)
    deviation = first_deviation(rows)
    if deviation is None:
        log.info(f"Node-wise verification: no deviation from Python reference ({report_path})")
    else:
        log.warning(
            f"Node-wise verification: first deviation at node {deviation['node']} "
            f"({deviation['op_type']}), {deviation['num_mismatch']}/{deviation['num_elements']} "
            f"elements, see {report_path}"
        )
    return rows

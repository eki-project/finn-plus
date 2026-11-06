"""In-process execution of the cheap FINN+ build steps for one candidate.

This is deliberately *not* going through ``build_dataflow_cfg``: that entry point swallows
exceptions into an exit code, requires ``XILINX_VIVADO`` even for estimate-only builds and
writes reports to disk. Here the step functions are called directly (the same way
``tests/fpgadataflow/multifpga/utils.py`` does it), every exception is classified, and the
estimates are collected from the analysis passes in memory.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from typing import TYPE_CHECKING, Any

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.op_and_param_counts import aggregate_dict_keys, op_and_param_counts
from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.builder.build_dataflow import resolve_build_steps
from finn.builder.build_dataflow_config import DataflowBuildConfig, estimate_only_dataflow_steps
from finn.nas.classify import (
    LAYOUT_OPS,
    boundary_unmapped_nodes,
    classify_failure,
    classify_interior_float,
    classify_unmapped,
    first_line,
)
from finn.nas.folding import make_policy
from finn.nas.hashing import finn_version
from finn.nas.logcapture import CapturedLogs, capture_finn_logs
from finn.nas.result import EvalResult, FailureCategory
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.util.exception import FINNUserError
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.util.platforms import platforms
from finn.util.resources import available_resources_on_platform
from finn.util.settings import get_settings, initialize_dummy_settings

if TYPE_CHECKING:
    from finn.nas.config import OracleConfig

#: Build steps executed before folding, in order: the builder's estimate-only flow up to the
#: folding step. Deriving the list keeps the oracle in sync when steps are added to the regular
#: flow (as happened with ``step_minimize_bit_width_initial``).
FRONTEND_STEPS = list(
    estimate_only_dataflow_steps[
        : estimate_only_dataflow_steps.index("step_target_fps_parallelization")
    ]
)

#: Node attributes captured as the "folding" of a candidate (mirrors the build steps).
FOLDING_ATTRS = [
    "PE",
    "SIMD",
    "EmbFold",
    "SeqFold",
    "parallel_window",
    "ram_style",
    "ram_style_thresholds",
    "ram_style_mask",
    "depth",
    "impl_style",
    "resType",
    "mac_resource",
    "mem_mode",
    "runtime_writeable_weights",
    "depth_trigger_uram",
    "depth_trigger_bram",
]

StageCallback = Callable[[str], None]


def make_build_config(oc: OracleConfig, cand_dir: Path) -> DataflowBuildConfig:
    """Create the ``DataflowBuildConfig`` used for one evaluation."""
    cfg = DataflowBuildConfig(
        output_dir=str(cand_dir / "build"),
        fpga_part=oc.fpga_part,
        board=oc.board,
        synth_clk_period_ns=oc.synth_clk_period_ns,
        target_fps=None,
        generate_outputs=[],
        save_intermediate_models=False,
        enable_exception_snapshots=False,
        mvau_wwidth_max=oc.mvau_wwidth_max,
        folding_two_pass_relaxation=oc.folding_two_pass_relaxation,
        minimize_bit_width=oc.minimize_bit_width,
        standalone_thresholds=oc.standalone_thresholds,
        max_multithreshold_bit_width=oc.max_multithreshold_bit_width,
        steps=list(FRONTEND_STEPS),
    )
    for key, value in oc.dataflow_cfg_overrides.items():
        if not hasattr(cfg, key):
            raise FINNUserError(f"Unknown DataflowBuildConfig attribute {key!r} in overrides.")
        setattr(cfg, key, value)
    # Fail early and with the right category if the part cannot be resolved.
    cfg._resolve_fpga_part()  # noqa: SLF001
    return cfg


def _use_build_dir(build_dir: Path) -> Callable[[], None]:
    """Point FINN's global build dir at ``build_dir``; return a function restoring the old one."""
    try:
        settings = get_settings()
    except FINNUserError:
        initialize_dummy_settings()
        settings = get_settings()
    old = settings.finn_build_dir
    build_dir.mkdir(parents=True, exist_ok=True)
    settings.finn_build_dir = build_dir

    def restore() -> None:
        """Point FINN's build dir back at the previous location."""
        settings.finn_build_dir = old

    return restore


def _is_hw(node: Any) -> bool:
    """Return whether ``node`` is a specialized (HLS or RTL) hardware layer."""
    return is_hls_node(node) or is_rtl_node(node)


def interior_float_nodes(model: ModelWrapper) -> list[list[str]]:
    """Return ``[name, op_type]`` of HW nodes that are neither first nor last and use FLOAT32."""
    graph_inputs = {vi.name for vi in model.graph.input}
    graph_outputs = {vi.name for vi in model.graph.output}
    found: list[list[str]] = []
    for node in model.graph.node:
        if not node.domain.startswith("finn.custom_op.fpgadataflow"):
            continue
        if any(i in graph_inputs for i in node.input) or any(
            o in graph_outputs for o in node.output
        ):
            continue
        inst = getCustomOp(node)
        is_float = False
        for i in range(len(node.input)):
            try:
                if inst.get_input_datatype(i) == DataType["FLOAT32"]:
                    is_float = True
            except Exception:  # some inputs have no datatype yet
                break
        for o in range(len(node.output)):
            try:
                if inst.get_output_datatype(o) == DataType["FLOAT32"]:
                    is_float = True
            except Exception:
                break
        if is_float:
            found.append([node.name, node.op_type])
    return found


def folding_snapshot(model: ModelWrapper) -> dict[str, dict[str, Any]]:
    """Collect the folding-related attributes of every HW node."""
    snapshot: dict[str, dict[str, Any]] = {}
    for node in model.graph.node:
        if not _is_hw(node):
            continue
        inst = getCustomOp(node)
        attrs: dict[str, Any] = {}
        for attr in FOLDING_ATTRS:
            try:
                value = inst.get_nodeattr(attr)
            except Exception:
                continue
            if hasattr(value, "item"):
                value = value.item()
            attrs[attr] = value
        snapshot[node.name] = attrs
    return snapshot


def device_fit(oc: OracleConfig, totals: dict[str, float]) -> dict[str, float]:
    """Fraction of the configured board's resources the estimates would occupy."""
    if oc.board is None or oc.board not in platforms:
        return {}
    platform = platforms[oc.board]()
    available = available_resources_on_platform(platform, list(oc.considered_resources))
    fit: dict[str, float] = {}
    for res in oc.considered_resources:
        capacity = available.get(res, 0) * oc.max_utilization
        if capacity > 0:
            fit[res] = float(totals.get(res, 0.0)) / capacity
    return fit


def collect_estimates(
    model: ModelWrapper, cfg: DataflowBuildConfig, oc: OracleConfig, result: EvalResult
) -> ModelWrapper:
    """Fill the estimate fields of ``result`` (what ``step_generate_estimate_reports`` writes)."""
    fpga_part = cfg._resolve_fpga_part()  # noqa: SLF001
    ops = model.analysis(op_and_param_counts)
    result.op_and_param_counts = {k: float(v) for k, v in ops.get("total", {}).items()}
    result.cycles_per_node = {k: int(v) for k, v in model.analysis(exp_cycles_per_layer).items()}
    per_node = model.analysis(partial(res_estimation, fpgapart=fpga_part))
    result.resources_per_node = {
        n: {k: float(v) for k, v in r.items()} for n, r in per_node.items()
    }
    result.resources_total = aggregate_dict_keys(per_node)
    model = model.transform(AnnotateCycles())
    perf = model.analysis(dataflow_performance)
    result.max_cycles = int(perf["max_cycles"])
    result.max_cycles_node = str(perf["max_cycles_node_name"])
    result.critical_path_cycles = int(perf["critical_path_cycles"])
    clk_ns = float(cfg.synth_clk_period_ns)
    if result.max_cycles > 0:
        result.est_fps = (1e9 / clk_ns) / result.max_cycles
    result.est_latency_ns = result.critical_path_cycles * clk_ns
    result.device_fit = device_fit(oc, result.resources_total)
    result.folding = folding_snapshot(model)
    return model


def run_pipeline(
    onnx_path: Path,
    oc: OracleConfig,
    cand_dir: Path,
    on_stage: StageCallback | None = None,
) -> EvalResult:
    """Evaluate the QONNX model at ``onnx_path`` and return an :class:`EvalResult`.

    ``on_stage`` is called with the stage name right before each stage starts (used by the
    worker to report progress for timeout diagnostics). The function never raises for
    problems with the candidate; only ``KeyboardInterrupt``/``SystemExit`` propagate.
    """
    cand_dir = Path(cand_dir)
    result = EvalResult(finn_version=finn_version())
    timings: dict[str, float] = {}
    result.timings_s = timings
    restore_build_dir = _use_build_dir(cand_dir / "tmp")

    def notify(stage: str) -> None:
        """Report the stage that is about to start."""
        if on_stage is not None:
            on_stage(stage)

    def fail(stage: str, exc: BaseException, last_model: ModelWrapper | None) -> EvalResult:
        """Record a failure caused by an exception."""
        f = classify_failure(stage, exc, last_model)
        result.feasible = False
        result.failure_stage = f.stage
        result.failure_category = f.category
        result.failure_node = f.node
        result.failure_op_type = f.op_type
        result.failure_message = f.message
        return result

    def fail_plain(
        stage: str,
        category: FailureCategory,
        message: str,
        node: str | None = None,
        op_type: str | None = None,
    ) -> EvalResult:
        """Record a failure detected by the oracle's own checks."""
        result.feasible = False
        result.failure_stage = stage
        result.failure_category = category
        result.failure_message = message
        result.failure_node = node
        result.failure_op_type = op_type
        return result

    t_start = time.perf_counter()
    try:
        with capture_finn_logs() as logs:
            outcome = _run_stages(onnx_path, oc, cand_dir, logs, notify, timings, result)
            if outcome is not None:
                stage, exc, last_model, category, message, node, op_type = outcome
                if exc is not None:
                    fail(stage, exc, last_model)
                else:
                    assert category is not None and message is not None
                    fail_plain(stage, category, message, node, op_type)
            else:
                result.feasible = True
        result.warnings = logs.messages()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        fail_plain("oracle", FailureCategory.INTERNAL_ERROR, first_line(exc))
    finally:
        timings["total"] = time.perf_counter() - t_start
        restore_build_dir()
    return result


_Outcome = tuple[
    str,
    BaseException | None,
    ModelWrapper | None,
    FailureCategory | None,
    str | None,
    str | None,
    str | None,
]


def _run_stages(
    onnx_path: Path,
    oc: OracleConfig,
    cand_dir: Path,
    logs: CapturedLogs,
    notify: StageCallback,
    timings: dict[str, float],
    result: EvalResult,
) -> _Outcome | None:
    """Run all stages; return ``None`` on success or a description of the first failure."""
    notify("config")
    try:
        cfg = make_build_config(oc, cand_dir)
        policy = make_policy(oc.folding)
    except Exception as exc:
        return ("config", exc, None, None, None, None, None)

    model = ModelWrapper(str(onnx_path))
    steps = resolve_build_steps(cfg, list_as_input=list(FRONTEND_STEPS))
    for name, step in zip(FRONTEND_STEPS, steps, strict=True):
        notify(name)
        t0 = time.perf_counter()
        mark = logs.mark()
        last_model = model
        try:
            model = step(model, cfg)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # bare asserts, KeyErrors, ... included
            timings[name] = time.perf_counter() - t0
            return (name, exc, last_model, None, None, None, None)
        timings[name] = time.perf_counter() - t0
        if name == "step_create_dataflow_partition":
            result.unmapped_boundary_nodes = boundary_unmapped_nodes(logs.since(mark), last_model)
            # Layout ops (the NCHW -> NHWC Transpose in front of every CNN) cost nothing on the
            # host; only real computation left there makes the estimates incomplete.
            host = [n for n, op in result.unmapped_boundary_nodes if op not in LAYOUT_OPS]
            if host and oc.boundary_unmapped_is_failure:
                f = classify_unmapped(
                    name,
                    host,
                    "Nodes left to the host at the graph boundary: " + ", ".join(host),
                    last_model,
                    interior=False,
                )
                return (name, None, None, f.category, f.message, f.node, f.op_type)
            floats = interior_float_nodes(model)
            if floats and oc.float_interior_is_failure:
                f = classify_interior_float(name, floats, model)
                return (name, None, None, f.category, f.message, f.node, f.op_type)

    notify("folding")
    t0 = time.perf_counter()
    try:
        outcome = policy.apply(model, cfg, logs)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        timings["folding"] = time.perf_counter() - t0
        return ("folding", exc, model, None, None, None, None)
    timings["folding"] = time.perf_counter() - t0
    model = outcome.model
    result.throughput_target_missed = not outcome.target_met
    if outcome.unhandled_op_types and policy.fail_on_unhandled_op:
        op_type = outcome.unhandled_op_types[0]
        node = next((n.name for n in model.graph.node if n.op_type == op_type), None)
        return (
            "folding",
            None,
            None,
            FailureCategory.FOLDING_INFEASIBLE,
            "SetFolding cannot fold op type(s): " + ", ".join(outcome.unhandled_op_types),
            node,
            op_type,
        )

    name = "step_minimize_bit_width"
    notify(name)
    t0 = time.perf_counter()
    try:
        model = resolve_build_steps(cfg, list_as_input=[name])[0](model, cfg)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        timings[name] = time.perf_counter() - t0
        return (name, exc, model, None, None, None, None)
    timings[name] = time.perf_counter() - t0

    notify("estimate")
    t0 = time.perf_counter()
    try:
        model = collect_estimates(model, cfg, oc, result)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        timings["estimate"] = time.perf_counter() - t0
        return ("estimate", exc, model, None, None, None, None)
    timings["estimate"] = time.perf_counter() - t0

    if result.throughput_target_missed and policy.require_target_met:
        return (
            "folding",
            None,
            None,
            FailureCategory.THROUGHPUT_TARGET_MISSED,
            (
                f"Throughput target not reachable; slowest node {result.max_cycles_node} "
                f"needs {result.max_cycles} cycles/frame (target {outcome.target_cycles})."
            ),
            result.max_cycles_node,
            None,
        )
    over = [r for r, f in result.device_fit.items() if f > 1.0]
    if over:
        return (
            "estimate",
            None,
            None,
            FailureCategory.RESOURCE_BUDGET_EXCEEDED,
            "Estimated resources exceed the board budget for: "
            + ", ".join(f"{r} ({result.device_fit[r]:.2f}x)" for r in over),
            None,
            None,
        )

    if oc.strict_codegen:
        name = "step_hw_codegen"
        notify(name)
        t0 = time.perf_counter()
        try:
            model = resolve_build_steps(cfg, list_as_input=[name])[0](model, cfg)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            timings[name] = time.perf_counter() - t0
            return (name, exc, model, None, None, None, None)
        timings[name] = time.perf_counter() - t0

    if oc.keep_artifacts != "never":
        model.save(str(cand_dir / "final.onnx"))
    return None

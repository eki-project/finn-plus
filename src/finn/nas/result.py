"""Result types returned by the FINN+ NAS oracle."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from typing import Any


class FailureCategory(StrEnum):
    """Machine-readable reason why a candidate network was rejected.

    The categories are coarse on purpose: they are meant to be consumed by a search algorithm
    (as a constraint signal and for learning what FINN+ accepts), not by a human debugging a
    single build. ``failure_message`` on the result carries the raw detail.
    """

    NONE = "none"
    #: Brevitas/QONNX export of the PyTorch module raised.
    EXPORT_FAILED = "export_failed"
    #: qonnx_to_finn, tidy_up or streamline raised.
    FRONTEND_FAILED = "frontend_failed"
    #: An affine/float op (Mul, Add, BatchNorm, ...) remained between HW layers: streamlining
    #: could not absorb it, i.e. the topology is not "streamlinable".
    STREAMLINE_INCOMPLETE = "streamline_incomplete"
    #: An op type with no HW inference or no HLS/RTL implementation variant.
    UNSUPPORTED_OP = "unsupported_op"
    #: A node between HW layers stayed unmapped and no finer classification applies.
    UNMAPPED_LAYER = "unmapped_layer"
    #: SetFolding raised, or does not know how to fold an op type in the graph.
    FOLDING_INFEASIBLE = "folding_infeasible"
    #: The throughput target could not be reached (only a failure if configured so).
    THROUGHPUT_TARGET_MISSED = "throughput_target_missed"
    #: Float datatypes on an interior HW node, or bit-width minimization failed.
    DATATYPE_UNSUPPORTED = "datatype_unsupported"
    #: Strict mode only: HLS/RTL code generation tripped an assertion (e.g. folding
    #: divisibility that is only checked at code generation time).
    CODEGEN_ASSERT = "codegen_assert"
    #: The estimated resources do not fit the configured budget/device.
    RESOURCE_BUDGET_EXCEEDED = "resource_budget_exceeded"
    #: The oracle itself is misconfigured (not the candidate's fault).
    CONFIG_ERROR = "config_error"
    #: The evaluation did not finish within the configured time limit.
    TIMEOUT = "timeout"
    #: Anything unexpected (FINN internal errors, worker crashes, ...).
    INTERNAL_ERROR = "internal_error"


#: Failure categories that describe a transient/environmental problem rather than a property
#: of the candidate. Results with these categories are never cached.
TRANSIENT_FAILURES = frozenset(
    {FailureCategory.TIMEOUT, FailureCategory.INTERNAL_ERROR, FailureCategory.CONFIG_ERROR}
)


@dataclass
class EvalResult:
    """Everything the oracle knows about one candidate after evaluation.

    All fields are plain JSON-serializable Python types so the result can cross process
    boundaries and be stored in a cache or an optimizer's trial attributes.
    """

    feasible: bool = False
    failure_stage: str | None = None
    failure_category: FailureCategory = FailureCategory.NONE
    failure_node: str | None = None
    failure_op_type: str | None = None
    failure_message: str | None = None

    #: Summed estimates over all HW nodes, e.g. ``{"LUT": 1234.0, "BRAM_18K": 2.0, ...}``.
    resources_total: dict[str, float] = field(default_factory=dict)
    #: Per-node estimates as returned by ``res_estimation``.
    resources_per_node: dict[str, dict[str, float]] = field(default_factory=dict)
    #: Per-node expected cycles per frame.
    cycles_per_node: dict[str, int] = field(default_factory=dict)
    max_cycles: int | None = None
    max_cycles_node: str | None = None
    critical_path_cycles: int | None = None
    est_fps: float | None = None
    est_latency_ns: float | None = None
    #: ``resources_total[r] / (device[r] * max_utilization)`` per resource; > 1 means the
    #: candidate does not fit the configured board. Empty if no board is configured.
    device_fit: dict[str, float] = field(default_factory=dict)
    #: Aggregated op and parameter counts (``op_mac``, ``param_weight``, ...).
    op_and_param_counts: dict[str, float] = field(default_factory=dict)
    #: Per-node folding/implementation attributes (PE, SIMD, ram_style, ...).
    folding: dict[str, dict[str, Any]] = field(default_factory=dict)
    throughput_target_missed: bool = False
    #: ``[name, op_type]`` of nodes at the graph boundary that stay unmapped (host-side).
    unmapped_boundary_nodes: list[list[str]] = field(default_factory=list)
    #: FINN warnings emitted during the evaluation.
    warnings: list[str] = field(default_factory=list)
    #: Wall-clock seconds per stage plus ``"total"``.
    timings_s: dict[str, float] = field(default_factory=dict)

    candidate_hash: str | None = None
    from_cache: bool = False
    finn_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        d = asdict(self)
        d["failure_category"] = self.failure_category.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalResult:
        """Reconstruct a result from :meth:`to_dict` output, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["failure_category"] = FailureCategory(kwargs.get("failure_category", "none"))
        return cls(**kwargs)

    @classmethod
    def failed(
        cls,
        stage: str,
        category: FailureCategory,
        message: str | None = None,
        node: str | None = None,
        op_type: str | None = None,
    ) -> EvalResult:
        """Construct an infeasible result."""
        return cls(
            feasible=False,
            failure_stage=stage,
            failure_category=category,
            failure_message=message,
            failure_node=node,
            failure_op_type=op_type,
        )

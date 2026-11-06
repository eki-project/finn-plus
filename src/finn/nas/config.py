"""Configuration of the FINN+ NAS oracle."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal


@dataclass
class FoldingSpec:
    """How the oracle chooses the parallelization (folding) of a candidate.

    ``kind="target_fps"`` runs FINN's ``SetFolding`` towards ``target_fps`` (as the regular
    build flow does). ``kind="fixed"`` applies a folding JSON (``folding_config_file``), which
    is useful to reproduce an oracle folding in a real synthesis run.
    """

    kind: Literal["target_fps", "fixed"] = "target_fps"
    target_fps: int | None = 1000
    folding_config_file: str | None = None
    #: Treat a missed throughput target as infeasible instead of only flagging it.
    require_target_met: bool = False
    #: Treat "SetFolding doesn't know how to handle op_type X" as infeasible. If False the node
    #: silently keeps PE=SIMD=1, which makes the estimates for it unreliable.
    fail_on_unhandled_op: bool = True


@dataclass
class OracleConfig:
    """Static configuration shared by all evaluations of one :class:`FinnOracle`.

    Everything is JSON-serializable so the config can be shipped to worker processes.
    """

    #: Either ``fpga_part`` or ``board`` must be given. ``board`` additionally enables the
    #: device-fit metric when the board is known to ``finn.util.platforms``.
    fpga_part: str | None = None
    board: str | None = None
    synth_clk_period_ns: float = 5.0
    folding: FoldingSpec = field(default_factory=FoldingSpec)
    #: Also run ``step_hw_codegen`` (writes HLS/RTL sources, no synthesis) to trigger the
    #: assertions that FINN only checks at code generation time. Considerably slower.
    strict_codegen: bool = False
    #: ``"process"`` evaluates every candidate in a fresh spawned interpreter with a timeout,
    #: ``"inprocess"`` runs in the calling process (real tracebacks, no timeout).
    isolation: Literal["process", "inprocess"] = "process"
    timeout_s: float = 300.0
    #: Directory for per-candidate build directories. Defaults to a temp directory.
    work_dir: str | None = None
    keep_artifacts: Literal["never", "on_failure", "always"] = "never"
    #: SQLite file (or directory, for a JSON-per-key cache) to memoize results in.
    cache_path: str | None = None
    #: Include weight values in the candidate hash. Off by default because NAS candidates
    #: are re-instantiated with random weights; note that bit-width minimization makes the
    #: estimates weakly weight-dependent.
    cache_include_weights: bool = False

    # Knobs forwarded to DataflowBuildConfig.
    mvau_wwidth_max: int = 36
    folding_two_pass_relaxation: bool = True
    minimize_bit_width: bool = True
    standalone_thresholds: bool = False
    max_multithreshold_bit_width: int = 8
    #: Any further ``DataflowBuildConfig`` attribute to override, by name.
    dataflow_cfg_overrides: dict[str, Any] = field(default_factory=dict)

    considered_resources: tuple[str, ...] = ("LUT", "BRAM_18K", "URAM", "DSP")
    max_utilization: float = 0.7
    #: Unmapped nodes at the graph start/end (which FINN leaves to the host) count as
    #: infeasible, except pure layout ops such as the NCHW->NHWC input transpose. Off, a network
    #: without input quantizer passes with its entire first layer missing from the estimates.
    boundary_unmapped_is_failure: bool = True
    #: FLOAT32 datatypes on an interior HW node count as infeasible.
    float_interior_is_failure: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        d = asdict(self)
        d["considered_resources"] = list(self.considered_resources)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OracleConfig:
        """Reconstruct a config from :meth:`to_dict` output."""
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        if isinstance(kwargs.get("folding"), dict):
            kwargs["folding"] = FoldingSpec(**kwargs["folding"])
        if "considered_resources" in kwargs:
            kwargs["considered_resources"] = tuple(kwargs["considered_resources"])
        return cls(**kwargs)

    def hash_relevant(self) -> dict[str, Any]:
        """Return the subset of settings that influences the result (used for cache keys)."""
        return {
            "fpga_part": self.fpga_part,
            "board": self.board,
            "synth_clk_period_ns": self.synth_clk_period_ns,
            "folding": asdict(self.folding),
            "strict_codegen": self.strict_codegen,
            "mvau_wwidth_max": self.mvau_wwidth_max,
            "folding_two_pass_relaxation": self.folding_two_pass_relaxation,
            "minimize_bit_width": self.minimize_bit_width,
            "standalone_thresholds": self.standalone_thresholds,
            "max_multithreshold_bit_width": self.max_multithreshold_bit_width,
            "dataflow_cfg_overrides": self.dataflow_cfg_overrides,
            "considered_resources": list(self.considered_resources),
            "max_utilization": self.max_utilization,
            "boundary_unmapped_is_failure": self.boundary_unmapped_is_failure,
            "float_interior_is_failure": self.float_interior_is_failure,
        }

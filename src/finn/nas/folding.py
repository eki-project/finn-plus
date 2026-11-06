"""Folding (parallelization) policies used by the oracle."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from finn.nas.classify import unhandled_folding_ops
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.transformation.general import ApplyConfig
from finn.transformation.qonnx.give_unique_node_names_recursive import GiveUniqueNodeNamesRecursive
from finn.util.exception import FINNConfigurationError

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

    from finn.builder.build_dataflow_config import DataflowBuildConfig
    from finn.nas.config import FoldingSpec
    from finn.nas.logcapture import CapturedLogs


@dataclass
class FoldingOutcome:
    """What a folding policy produced."""

    model: ModelWrapper
    target_cycles: int | None = None
    target_met: bool = True
    unhandled_op_types: list[str] = field(default_factory=list)


class FoldingPolicy(Protocol):
    """Chooses PE/SIMD/... attributes for all HW nodes of a model."""

    require_target_met: bool
    fail_on_unhandled_op: bool

    def apply(
        self, model: ModelWrapper, cfg: DataflowBuildConfig, logs: CapturedLogs
    ) -> FoldingOutcome:
        """Set folding attributes on ``model`` and report the outcome."""
        ...


@dataclass
class TargetFpsFolding:
    """Run FINN's ``SetFolding`` towards a frames-per-second target (the default flow)."""

    target_fps: int
    require_target_met: bool = False
    fail_on_unhandled_op: bool = True

    def apply(
        self, model: ModelWrapper, cfg: DataflowBuildConfig, logs: CapturedLogs
    ) -> FoldingOutcome:
        """Mirror ``step_target_fps_parallelization`` but keep the SetFolding instance."""
        cfg.target_fps = self.target_fps
        target_cycles = cfg._resolve_cycles_per_frame()  # noqa: SLF001
        assert target_cycles is not None
        mark = logs.mark()
        transform = SetFolding(
            target_cycles,
            mvau_wwidth_max=cfg.mvau_wwidth_max,
            two_pass_relaxation=cfg.folding_two_pass_relaxation,
        )
        model = model.transform(transform, apply_to_subgraphs=True)
        model = model.transform(GiveUniqueNodeNamesRecursive())
        # The flag is set during the first pass; the relaxed second pass (if any) targets the
        # achievable throughput, so it reflects whether *our* target was reachable.
        target_met = not getattr(transform, "any_throughput_target_missed", False)
        return FoldingOutcome(
            model=model,
            target_cycles=target_cycles,
            target_met=target_met,
            unhandled_op_types=unhandled_folding_ops(logs.since(mark)),
        )


@dataclass
class FixedFolding:
    """Apply a folding configuration JSON (as ``step_apply_folding_config`` does)."""

    folding_config_file: Path
    require_target_met: bool = False
    fail_on_unhandled_op: bool = False

    def apply(
        self, model: ModelWrapper, cfg: DataflowBuildConfig, logs: CapturedLogs  # noqa: ARG002
    ) -> FoldingOutcome:
        """Apply the folding JSON onto the model."""
        if not self.folding_config_file.exists():
            raise FINNConfigurationError(
                f"Folding config file {self.folding_config_file} does not exist."
            )
        model = model.transform(GiveUniqueNodeNamesRecursive())
        model = model.transform(ApplyConfig(str(self.folding_config_file)), apply_to_subgraphs=True)
        return FoldingOutcome(model=model)


def make_policy(spec: FoldingSpec) -> FoldingPolicy:
    """Instantiate the policy described by a :class:`FoldingSpec`."""
    if spec.kind == "target_fps":
        if spec.target_fps is None:
            raise FINNConfigurationError("FoldingSpec(kind='target_fps') needs target_fps.")
        return TargetFpsFolding(
            target_fps=int(spec.target_fps),
            require_target_met=spec.require_target_met,
            fail_on_unhandled_op=spec.fail_on_unhandled_op,
        )
    if spec.kind == "fixed":
        if spec.folding_config_file is None:
            raise FINNConfigurationError("FoldingSpec(kind='fixed') needs folding_config_file.")
        return FixedFolding(
            folding_config_file=Path(spec.folding_config_file),
            require_target_met=spec.require_target_met,
            fail_on_unhandled_op=spec.fail_on_unhandled_op,
        )
    raise FINNConfigurationError(f"Unknown folding kind {spec.kind!r}.")

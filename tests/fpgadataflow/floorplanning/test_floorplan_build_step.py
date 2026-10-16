"""Tests for the Phase-A build step and the floorplan_slr -> slr bridge."""

import pytest

from fpgadataflow.floorplanning.util import U55C_PART, attr, make_chain

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    FloorplanGranularity,
    PartitioningConfiguration,
    ShellFlowType,
    default_build_dataflow_steps,
)
from finn.builder.build_dataflow_steps import (
    build_dataflow_step_lookup,
    step_generate_floorplan_assignment,
)
from finn.transformation.fpgadataflow.floorplanning.bridge import BridgeFloorplanToSLR
from finn.transformation.fpgadataflow.floorplanning.grouping import ComputeFloorplanGroups
from finn.transformation.fpgadataflow.floorplanning.slr_assignment import AssignFloorplanSLRs
from finn.util.basic import make_build_dir
from finn.util.exception import FINNInternalError
from finn.util.platforms import platforms


def _cfg(**overrides: object) -> DataflowBuildConfig:
    """Build a minimal U55C build config with the floorplanner enabled."""
    kwargs: dict = {
        "output_dir": str(make_build_dir("test_floorplan_step_")),
        "synth_clk_period_ns": 5.0,
        "board": "U55C",
        "shell_flow_type": ShellFlowType.VITIS_ALVEO,
        "steps": [],
        "target_fps": 1000,
        "enable_auto_floorplanning": True,
    }
    kwargs.update(overrides)
    return DataflowBuildConfig(**kwargs)


# ------------------------------------------------------------------------ the bridge


@pytest.mark.fpgadataflow
def test_bridge_copies_the_assignment_onto_the_legacy_attr() -> None:
    """The legacy attr is only written by the explicit bridge, never implicitly."""
    model = make_chain(3)
    model = model.transform(ComputeFloorplanGroups())
    model = model.transform(AssignFloorplanSLRs(platforms["U55C"](), U55C_PART, "U55C"))
    assert all(attr(model, n.name, "slr") == -1 for n in model.graph.node)

    model = model.transform(BridgeFloorplanToSLR())
    for node in model.graph.node:
        assert attr(model, node.name, "slr") == attr(model, node.name, "floorplan_slr")
        # The floorplanner's own record survives, so it stays readable after
        # Floorplan has rewritten slr and partition_id.
        assert attr(model, node.name, "floorplan_slr") >= 0


@pytest.mark.fpgadataflow
def test_bridge_can_keep_an_existing_assignment() -> None:
    """With overwrite off, a pre-existing slr value is left alone."""
    from qonnx.custom_op.registry import getCustomOp

    model = make_chain(3)
    model = model.transform(ComputeFloorplanGroups())
    model = model.transform(AssignFloorplanSLRs(platforms["U55C"](), U55C_PART, "U55C"))
    getCustomOp(model.graph.node[0]).set_nodeattr("slr", 2)

    model = model.transform(BridgeFloorplanToSLR(overwrite_existing=False))
    assert attr(model, model.graph.node[0].name, "slr") == 2
    assert attr(model, model.graph.node[1].name, "slr") == attr(
        model, model.graph.node[1].name, "floorplan_slr"
    )


@pytest.mark.fpgadataflow
def test_bridge_before_assignment_is_an_internal_error() -> None:
    """Bridging nothing means the pipeline ran out of order."""
    model = make_chain(3)
    with pytest.raises(FINNInternalError, match="AssignFloorplanSLRs must run"):
        model.transform(BridgeFloorplanToSLR())


# -------------------------------------------------------------------------- the step


@pytest.mark.fpgadataflow
def test_step_is_registered_and_in_the_default_pipeline() -> None:
    """The step must be resolvable by name and run after hardware generation."""
    assert "step_generate_floorplan_assignment" in build_dataflow_step_lookup
    steps = default_build_dataflow_steps
    assert steps.index("step_generate_floorplan_assignment") > steps.index("step_generate_hardware")
    assert steps.index("step_generate_floorplan_assignment") < steps.index(
        "step_create_stitched_ip"
    )


@pytest.mark.fpgadataflow
def test_step_is_a_noop_when_disabled() -> None:
    """Nothing is assigned unless the feature is explicitly enabled."""
    model = make_chain(3)
    model = step_generate_floorplan_assignment(model, _cfg(enable_auto_floorplanning=False))
    assert all(attr(model, n.name, "floorplan_group_id") == -1 for n in model.graph.node)
    assert all(attr(model, n.name, "floorplan_slr") == -1 for n in model.graph.node)


@pytest.mark.fpgadataflow
def test_step_assigns_groups_and_slrs() -> None:
    """Enabled, the step produces a complete assignment and bridges it by default."""
    model = make_chain(4)
    model = step_generate_floorplan_assignment(model, _cfg())
    for node in model.graph.node:
        assert attr(model, node.name, "floorplan_group_id") >= 0
        assert attr(model, node.name, "floorplan_slr") >= 0
        assert attr(model, node.name, "slr") == attr(model, node.name, "floorplan_slr")


@pytest.mark.fpgadataflow
def test_step_can_skip_the_bridge() -> None:
    """With the bridge off, the legacy attr is untouched for isolated A/B testing."""
    model = make_chain(4)
    model = step_generate_floorplan_assignment(model, _cfg(floorplan_bridge_to_slr=False))
    assert all(attr(model, n.name, "floorplan_slr") >= 0 for n in model.graph.node)
    assert all(attr(model, n.name, "slr") == -1 for n in model.graph.node)


@pytest.mark.fpgadataflow
def test_step_honours_the_granularity_setting() -> None:
    """The granularity config field reaches the grouping transform."""
    model = make_chain(4)
    model = step_generate_floorplan_assignment(
        model, _cfg(floorplan_granularity=FloorplanGranularity.PER_NODE)
    )
    groups = {attr(model, n.name, "floorplan_group_id") for n in model.graph.node}
    assert len(groups) == len(model.graph.node)


@pytest.mark.fpgadataflow
@pytest.mark.parametrize(
    "conflict",
    [
        {"vitis_floorplan_file": "/tmp/floorplan.json"},
        {"partitioning_configuration": PartitioningConfiguration()},
    ],
)
def test_explicit_config_wins_over_auto_floorplanning(
    conflict: dict, caplog: pytest.LogCaptureFixture
) -> None:
    """An explicit floorplan or partitioning config skips auto-assignment, with a warning."""
    model = make_chain(3)
    with caplog.at_level("WARNING"):
        model = step_generate_floorplan_assignment(model, _cfg(**conflict))
    assert all(attr(model, n.name, "floorplan_slr") == -1 for n in model.graph.node)
    assert "takes precedence" in caplog.text
    assert next(iter(conflict)) in caplog.text


@pytest.mark.fpgadataflow
def test_board_without_a_platform_model_is_skipped(caplog: pytest.LogCaptureFixture) -> None:
    """Without per-SLR resource data there is nothing to assign against."""
    model = make_chain(3)
    with caplog.at_level("WARNING"):
        model = step_generate_floorplan_assignment(
            model, _cfg(board=None, fpga_part=U55C_PART, shell_flow_type=None)
        )
    assert all(attr(model, n.name, "floorplan_slr") == -1 for n in model.graph.node)
    assert "platform model" in caplog.text


@pytest.mark.fpgadataflow
def test_step_is_idempotent() -> None:
    """Re-running must recompute rather than layer onto the previous result.

    The Alveo flow calls the step twice: once where the plan places it, and again
    in step_prepare_synthesis, because IODMAs, DWCs and layer specialization only
    happen there and the earlier assignment covers a stale node set.
    """
    model = make_chain(4)
    cfg = _cfg()
    model = step_generate_floorplan_assignment(model, cfg)
    first = {n.name: attr(model, n.name, "floorplan_slr") for n in model.graph.node}
    groups_first = {n.name: attr(model, n.name, "floorplan_group_id") for n in model.graph.node}

    model = step_generate_floorplan_assignment(model, cfg)
    assert {n.name: attr(model, n.name, "floorplan_slr") for n in model.graph.node} == first
    assert {
        n.name: attr(model, n.name, "floorplan_group_id") for n in model.graph.node
    } == groups_first


@pytest.mark.fpgadataflow
def test_rerunning_covers_nodes_added_in_between() -> None:
    """A node inserted after the first run must be grouped and assigned by the second."""
    from fpgadataflow.floorplanning.util import make_fifo, make_model, make_mvau

    model = make_model(
        [
            make_mvau("mvau0", "inp", "t0"),
            make_mvau("mvau1", "t0", "outp"),
        ]
    )
    cfg = _cfg(floorplan_granularity=FloorplanGranularity.PER_NODE)
    model = step_generate_floorplan_assignment(model, cfg)

    # Stand in for the IODMA/DWC insertion the Alveo flow does after the first run.
    model = make_model(
        [
            make_mvau("mvau0", "inp", "t0"),
            make_fifo("fifo0", "t0", "t1"),
            make_mvau("mvau1", "t1", "outp"),
        ]
    )
    model = step_generate_floorplan_assignment(model, cfg)
    for node in model.graph.node:
        assert attr(model, node.name, "floorplan_group_id") >= 0
        assert attr(model, node.name, "floorplan_slr") >= 0

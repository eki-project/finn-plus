"""Tests for Level-2 placement unit computation."""

import pytest

from fpgadataflow.floorplanning.util import (
    attr,
    make_dwc,
    make_fifo,
    make_model,
    make_mvau,
    set_device_ids,
)
from qonnx.core.modelwrapper import ModelWrapper

from finn.builder.build_dataflow_config import FloorplanGranularity
from finn.transformation.fpgadataflow.floorplanning.grouping import (
    ComputeFloorplanGroups,
    compute_placement_units,
    get_placement_units,
)
from finn.util.exception import FINNInternalError


def _mvau_fifo_dwc_mvau() -> ModelWrapper:
    """MVAU -> FIFO -> DWC -> MVAU, i.e. one compute node with two glue nodes behind it."""
    return make_model(
        [
            make_mvau("mvau0", "inp", "t0"),
            make_fifo("fifo0", "t0", "t1"),
            make_dwc("dwc0", "t1", "t2"),
            make_mvau("mvau1", "t2", "outp"),
        ]
    )


@pytest.mark.fpgadataflow
def test_per_node_gives_one_unit_per_node() -> None:
    """PER_NODE granularity must not merge anything, glue included."""
    model = _mvau_fifo_dwc_mvau()
    units = compute_placement_units(model, FloorplanGranularity.PER_NODE)
    assert len(units) == len(model.graph.node)
    assert all(len(names) == 1 for names in units.values())


@pytest.mark.fpgadataflow
def test_glue_attaches_to_producer() -> None:
    """FIFOs and DWCs join the unit of the compute node feeding them."""
    model = _mvau_fifo_dwc_mvau()
    units = compute_placement_units(model, FloorplanGranularity.NODE_WITH_ADJACENT_GLUE)
    assert len(units) == 2
    assert units[0] == ["MVAU_hls_0", "StreamingFIFO_rtl_0", "StreamingDataWidthConverter_rtl_0"]
    assert units[1] == ["MVAU_hls_1"]


@pytest.mark.fpgadataflow
def test_dwc_grouping_can_be_disabled() -> None:
    """With group_dwcs off, only the FIFO merges and the DWC keeps its own unit."""
    model = _mvau_fifo_dwc_mvau()
    units = compute_placement_units(
        model, FloorplanGranularity.NODE_WITH_ADJACENT_GLUE, group_dwcs=False
    )
    assert len(units) == 3
    assert units[0] == ["MVAU_hls_0", "StreamingFIFO_rtl_0"]
    assert units[1] == ["StreamingDataWidthConverter_rtl_0"]
    assert units[2] == ["MVAU_hls_1"]


@pytest.mark.fpgadataflow
def test_leading_glue_attaches_to_consumer() -> None:
    """Glue with no producer falls back to joining its consumer's unit."""
    model = make_model(
        [
            make_fifo("fifo0", "inp", "t0"),
            make_mvau("mvau0", "t0", "outp"),
        ]
    )
    units = compute_placement_units(model, FloorplanGranularity.NODE_WITH_ADJACENT_GLUE)
    assert len(units) == 1
    assert sorted(next(iter(units.values()))) == ["MVAU_hls_0", "StreamingFIFO_rtl_0"]


@pytest.mark.fpgadataflow
def test_glue_never_merges_across_a_device_boundary() -> None:
    """A neighbour on another device is a separate chip, so it is not eligible for glue."""
    model = _mvau_fifo_dwc_mvau()
    # Put the trailing glue and the second MVAU on a second device, as a prior
    # multi-FPGA partitioning pass would have.
    set_device_ids(
        model,
        {
            "StreamingFIFO_rtl_0": 1,
            "StreamingDataWidthConverter_rtl_0": 1,
            "MVAU_hls_1": 1,
        },
    )
    units = compute_placement_units(model, FloorplanGranularity.NODE_WITH_ADJACENT_GLUE)
    assert units[0] == ["MVAU_hls_0"]
    # The glue may only merge with the node it shares a device with.
    assert units[1] == [
        "StreamingFIFO_rtl_0",
        "StreamingDataWidthConverter_rtl_0",
        "MVAU_hls_1",
    ]


@pytest.mark.fpgadataflow
def test_unit_ids_are_dense_and_topological() -> None:
    """Ids must be 0..n-1 and ascend with the position of each unit's earliest member."""
    model = _mvau_fifo_dwc_mvau()
    units = compute_placement_units(model, FloorplanGranularity.NODE_WITH_ADJACENT_GLUE)
    assert sorted(units) == list(range(len(units)))
    positions = {node.name: i for i, node in enumerate(model.graph.node)}
    firsts = [min(positions[name] for name in units[unit]) for unit in sorted(units)]
    assert firsts == sorted(firsts)


@pytest.mark.fpgadataflow
def test_transformation_writes_and_reads_back_group_ids() -> None:
    """The transformation must round-trip through the nodeattr it writes."""
    model = _mvau_fifo_dwc_mvau()
    expected = compute_placement_units(model, FloorplanGranularity.NODE_WITH_ADJACENT_GLUE)
    model = model.transform(ComputeFloorplanGroups(FloorplanGranularity.NODE_WITH_ADJACENT_GLUE))
    assert get_placement_units(model) == expected
    assert attr(model, "MVAU_hls_0", "floorplan_group_id") == 0


@pytest.mark.fpgadataflow
def test_regrouping_overwrites_previous_assignment() -> None:
    """Re-running at a different granularity must not leave stale ids behind."""
    model = _mvau_fifo_dwc_mvau()
    model = model.transform(ComputeFloorplanGroups(FloorplanGranularity.NODE_WITH_ADJACENT_GLUE))
    model = model.transform(ComputeFloorplanGroups(FloorplanGranularity.PER_NODE))
    assert len(get_placement_units(model)) == len(model.graph.node)


@pytest.mark.fpgadataflow
def test_reading_units_before_grouping_is_an_error() -> None:
    """Consuming placement units without grouping first must fail loudly."""
    model = _mvau_fifo_dwc_mvau()
    with pytest.raises(FINNInternalError, match="ComputeFloorplanGroups"):
        get_placement_units(model)


@pytest.mark.fpgadataflow
def test_structural_and_abstract_nodes_are_not_grouped() -> None:
    """Only nodes with an HLS or RTL backend get a placement unit.

    A StreamingDataflowPartition passes is_fpgadataflow_node, but it is not an
    HWCustomOp and has no resource estimate: writing a group id to one raises
    outright, and grouping one would build a unit that cannot be sized. Same
    reasoning for an abstract layer SpecializeLayers has not lowered yet.
    """
    from onnx import helper as oh

    model = _mvau_fifo_dwc_mvau()
    # Splice in a structural node and an unspecialized one.
    model.graph.node.append(
        oh.make_node(
            "StreamingDataflowPartition",
            ["outp"],
            ["sdp_out"],
            name="sdp0",
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            model="/nonexistent.onnx",
        )
    )
    model.graph.node.append(
        oh.make_node(
            "MVAU",
            ["sdp_out", "abstract_w"],
            ["abstract_out"],
            name="abstract_mvau",
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            MW=4,
            MH=4,
            SIMD=1,
            PE=1,
            inputDataType="INT2",
            weightDataType="INT2",
            outputDataType="INT2",
            ActVal=0,
            binaryXnorMode=0,
            noActivation=1,
        )
    )

    units = compute_placement_units(model, FloorplanGranularity.PER_NODE)
    grouped = {name for names in units.values() for name in names}
    assert "sdp0" not in grouped
    assert "abstract_mvau" not in grouped

    # Applying the transformation must not touch them either. An SDP is not even an
    # HWCustomOp, so it has no floorplan_group_id to write -- attempting one raises.
    model = model.transform(ComputeFloorplanGroups(FloorplanGranularity.PER_NODE))
    with pytest.raises(AttributeError):
        attr(model, "sdp0", "floorplan_group_id")
    # The abstract layer does have the attr; it just stays unassigned.
    assert attr(model, "abstract_mvau", "floorplan_group_id") == -1
    # Reading units back must agree, rather than raising over the skipped nodes.
    assert get_placement_units(model) == units

"""Tests for the floorplan report and its SVG diagram."""

import pytest

import json
import re
from fpgadataflow.floorplanning.util import U55C_PART, make_chain
from pathlib import Path

from finn.builder.build_dataflow_steps import _generate_pblock_svg
from finn.transformation.fpgadataflow.floorplanning.binding import collect_floorplan_units
from finn.transformation.fpgadataflow.floorplanning.grouping import ComputeFloorplanGroups
from finn.transformation.fpgadataflow.floorplanning.report import (
    enrich_floorplan_report,
    render_floorplan_svg,
    write_floorplan_report,
)
from finn.transformation.fpgadataflow.floorplanning.slr_assignment import AssignFloorplanSLRs
from finn.util.basic import make_build_dir
from finn.util.platforms import platforms

#: A report shaped like the one the Vivado-side engine writes.
ENGINE_REPORT: dict = {
    "device_slice_max_x": 232,
    "device_slice_max_y": 719,
    "units": {
        "pblock_finn_unit_0": {
            "slr": 0,
            "round": 1,
            "satisfied": True,
            "grid_ranges": "RAMB36_X0Y0:RAMB36_X1Y23 SLICE_X0Y0:SLICE_X20Y119",
            "pblock_capacity": {"luts": 20160, "brams": 48},
        },
        "pblock_finn_unit_1": {
            "slr": 2,
            "round": 1,
            "satisfied": True,
            "grid_ranges": "SLICE_X0Y480:SLICE_X32Y599",
            "pblock_capacity": {"luts": 30720, "brams": 0},
        },
    },
}

#: A report shaped like the DFX flow's, which carries no SLR information.
PR_REPORT: dict = {
    "device_slice_max_x": 100,
    "device_slice_max_y": 200,
    "pr_regions": {
        "pblock_Hier_sdp_0": {
            "grid_ranges": "SLICE_X0Y0:SLICE_X49Y99",
            "pblock_capacity": {"luts": 100},
        }
    },
}


def _rects(svg: str) -> list[tuple[int, int, int, int]]:
    """Return every (x, y, width, height) rect in an SVG."""
    return [
        (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
        for m in re.finditer(r'<rect x="(\d+)" y="(\d+)" width="(\d+)" height="(\d+)"', svg)
    ]


@pytest.mark.fpgadataflow
def test_engine_report_renders_with_one_rect_per_slice_range() -> None:
    """Every SLICE range is drawn, not just the first one Vivado happened to emit."""
    svg = render_floorplan_svg(ENGINE_REPORT)
    assert svg is not None
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    # One device background rect, one per unit SLICE range, two legend swatches.
    assert len(_rects(svg)) == 1 + 2 + 2


@pytest.mark.fpgadataflow
def test_units_are_coloured_and_labelled_by_slr() -> None:
    """Which SLR a layer landed in is the thing worth seeing at a glance."""
    svg = render_floorplan_svg(ENGINE_REPORT)
    assert svg is not None
    assert ">SLR0<" in svg
    assert ">SLR2<" in svg
    assert ">SLR1<" not in svg


@pytest.mark.fpgadataflow
def test_the_dfx_report_shape_still_renders() -> None:
    """The renderer was generalized, not replaced: the PR report must still work."""
    svg = render_floorplan_svg(PR_REPORT)
    assert svg is not None
    # No SLR data, so no legend: background plus the one region.
    assert len(_rects(svg)) == 2
    assert "SLR" not in svg
    assert ">sdp_0<" in svg


@pytest.mark.fpgadataflow
def test_a_report_with_nothing_drawable_renders_nothing() -> None:
    """A manual-pblock build has no SLICE ranges to draw and must not produce a stub."""
    assert render_floorplan_svg({"units": {}}) is None
    assert render_floorplan_svg({"pr_regions": {"a": {"grid_ranges": ""}}}) is None


@pytest.mark.fpgadataflow
def test_y_axis_is_flipped() -> None:
    """FPGA Y=0 is the bottom of the die and must be the bottom of the image."""
    svg = render_floorplan_svg(ENGINE_REPORT)
    assert svg is not None
    rects = _rects(svg)[1:3]
    # Both units are 120 SLICE rows tall, so tell them apart by width:
    # the SLR0 unit spans X0..20, the SLR2 one X0..32.
    low_y_rect = next(r for r in rects if r[2] == (20 - 0 + 1) * 3)
    high_y_rect = next(r for r in rects if r[2] == (32 - 0 + 1) * 3)
    assert low_y_rect[1] > high_y_rect[1], "the SLR0 unit must sit below the SLR2 one"


@pytest.mark.fpgadataflow
def test_svg_writer_matches_the_renderer() -> None:
    """The DFX entry point is a thin wrapper and must not drift from the renderer."""
    work_dir = Path(make_build_dir("floorplan_svg_"))
    report_path = work_dir / "report.json"
    report_path.write_text(json.dumps(PR_REPORT))
    svg_path = work_dir / "out.svg"
    _generate_pblock_svg(report_path, svg_path)
    assert svg_path.read_text() == render_floorplan_svg(PR_REPORT)


@pytest.mark.fpgadataflow
def test_report_is_enriched_with_the_nodes_each_pblock_holds() -> None:
    """The report must say which nodes a pblock holds, without touching the rest.

    In particular the engine's own `resource_source` must survive: it records how
    the geometry was sized, which only the engine knows. Overwriting it from the
    model once made every measured pblock report itself as estimate-sized.
    """
    model = make_chain(4)
    model = model.transform(ComputeFloorplanGroups())
    model = model.transform(AssignFloorplanSLRs(platforms["U55C"](), U55C_PART, "U55C"))
    specs = collect_floorplan_units(model, U55C_PART)

    engine_report = {
        "device_slice_max_x": 232,
        "device_slice_max_y": 719,
        "units": {
            spec.name: {
                "slr": spec.slr,
                "grid_ranges": "SLICE_X0Y0:SLICE_X20Y119",
                "resource_source": "post_synth",
            }
            for spec in specs
        },
    }
    enriched = enrich_floorplan_report(model, engine_report)

    covered = {node for unit in enriched["units"].values() for node in unit["nodes"]}
    assert covered == {node.name for node in model.graph.node}
    for unit in enriched["units"].values():
        assert unit["resource_source"] == "post_synth", "the engine's own value must survive"
        assert unit["slr"] >= 0


@pytest.mark.fpgadataflow
def test_write_report_produces_json_and_svg() -> None:
    """The build-side entry point writes both artefacts and reports that it did."""
    work_dir = Path(make_build_dir("floorplan_write_"))
    engine_path = work_dir / "engine.json"
    engine_path.write_text(json.dumps(ENGINE_REPORT))

    json_path = work_dir / "floorplan.json"
    svg_path = work_dir / "floorplan.svg"
    assert write_floorplan_report(make_chain(2), engine_path, json_path, svg_path)
    assert json.loads(json_path.read_text())["units"]
    assert svg_path.read_text().startswith("<svg")


@pytest.mark.fpgadataflow
def test_write_report_is_a_noop_without_an_engine_report() -> None:
    """A build that did not floorplan must not leave an empty report behind."""
    work_dir = Path(make_build_dir("floorplan_missing_"))
    json_path = work_dir / "floorplan.json"
    assert not write_floorplan_report(make_chain(2), work_dir / "absent.json", json_path)
    assert not json_path.exists()


@pytest.mark.fpgadataflow
def test_malformed_engine_report_is_reported_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A truncated Vivado report must not take the whole build down at reporting time."""
    work_dir = Path(make_build_dir("floorplan_bad_"))
    engine_path = work_dir / "engine.json"
    engine_path.write_text("{ this is not json")
    with caplog.at_level("WARNING"):
        assert not write_floorplan_report(make_chain(2), engine_path, work_dir / "floorplan.json")
    assert "not valid JSON" in caplog.text

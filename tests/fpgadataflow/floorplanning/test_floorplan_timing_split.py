"""Tests for the intra/inter-unit timing split (pure Python, no Vivado)."""

import pytest

import json
from pathlib import Path

from finn.transformation.fpgadataflow.floorplanning.timing_split import (
    Box,
    align_cells,
    analyse,
    classify,
    instance_name,
    load_units,
    op_of,
    remap_units,
    render_markdown,
    slice_box,
    summarize,
    unit_cells,
    write_roots_file,
)
from finn.util.basic import make_build_dir

PREFIX = "level0_i/ulp/StreamingDataflowPartition_1/StreamingDataflowPartition_1_i/"
MVAU0 = "StreamingDataflowPartition_1_MVAU_rtl_0"
FIFO0 = "StreamingDataflowPartition_1_StreamingFIFO_rtl_0"
MVAU1 = "StreamingDataflowPartition_1_MVAU_rtl_1"

#: Two layers and the FIFO between them, as the hook's artefacts describe them.
UNITS = [
    {"name": "pblock_finn_unit_0", "slr": 0, "cells": [MVAU0]},
    {"name": "pblock_finn_unit_1", "slr": 0, "cells": [FIFO0]},
    {"name": "pblock_finn_unit_2", "slr": 0, "cells": [MVAU1]},
]
REPORT = {
    "units": {
        "pblock_finn_unit_0": {"grid_ranges": "SLICE_X0Y0:SLICE_X39Y9 DSP48E2_X0Y0:DSP48E2_X0Y3"},
        "pblock_finn_unit_1": {"grid_ranges": "SLICE_X40Y0:SLICE_X41Y9"},
        "pblock_finn_unit_2": {"grid_ranges": "SLICE_X100Y0:SLICE_X109Y9"},
    }
}


def _unit(root: str, cells: int, box: tuple[int, int, int, int]) -> dict:
    """One unit entry of the Vivado export: its resolved root and placed box."""
    return {
        "roots": [PREFIX + root],
        "cells": cells,
        "x0": box[0],
        "y0": box[1],
        "x1": box[2],
        "y1": box[3],
    }


def _path(
    slack: float,
    logic: float,
    route: float,
    levels: int,
    start: str,
    end: str,
    s: tuple[int, int],
    e: tuple[int, int],
) -> dict:
    """One exported path from ``start`` to ``end`` with endpoint tile coordinates."""
    return {
        "slack": slack, "requirement": 10.0, "datapath": logic + route, "logic": logic,
        "route": route, "levels": levels, "start_cell": start, "end_cell": end,
        "start_site": "SLICE_X0Y0", "end_site": "SLICE_X0Y0",
        "sx": s[0], "sy": s[1], "ex": e[0], "ey": e[1],
    }  # fmt: skip


EXPORT = {
    "clock": "clk_kernel_00_unbuffered_net",
    "units": {
        "pblock_finn_unit_0": _unit(MVAU0, 100, (0, 0, 80, 10)),
        "pblock_finn_unit_1": _unit(FIFO0, 5, (81, 0, 83, 10)),
        "pblock_finn_unit_2": _unit(MVAU1, 100, (200, 0, 220, 10)),
    },
    "paths": [
        # intra unit 0, long and wire-bound
        _path(
            1.0, 1.0, 7.0, 2, f"{PREFIX}{MVAU0}/a/reg", f"{PREFIX}{MVAU0}/b/reg", (0, 0), (80, 0)
        ),
        # inter: unit 0 -> the adjacent FIFO
        _path(
            2.0, 2.0, 5.0, 1, f"{PREFIX}{MVAU0}/o/reg", f"{PREFIX}{FIFO0}/f/reg", (79, 0), (82, 2)
        ),
        # inter: FIFO -> the far unit 2
        _path(
            0.5, 1.0, 8.0, 1, f"{PREFIX}{FIFO0}/f/oreg", f"{PREFIX}{MVAU1}/i/reg", (83, 0), (200, 0)
        ),
        # other: from the DMA
        _path(
            3.0,
            2.0,
            3.0,
            1,
            "level0_i/ulp/idma_0/x/reg",
            f"{PREFIX}{MVAU0}/i/reg",
            (10, 10),
            (2, 2),
        ),
    ],
}


def _artefacts(tmp: Path) -> Path:
    """Write the fixture artefacts the way a link directory holds them."""
    (tmp / "finn_floorplan_units.json").write_text(json.dumps(UNITS))
    (tmp / "finn_floorplan.json").write_text(json.dumps(REPORT))
    (tmp / "export.json").write_text(json.dumps(EXPORT))
    return tmp


@pytest.mark.fpgadataflow
def test_box_geometry() -> None:
    """Aspect ratio, span, touching and gap follow the inclusive rectangle semantics."""
    a = Box(0, 0, 39, 9)
    b = Box(40, 0, 41, 9)
    c = Box(100, 0, 109, 9)
    assert a.width == 40 and a.height == 10 and a.aspect == 4.0 and a.half_perimeter == 50
    assert a.touches(b) and b.touches(a)
    assert not a.touches(c)
    assert a.gap(b) == 0
    assert b.gap(c) == 58
    assert slice_box(REPORT["units"]["pblock_finn_unit_0"]["grid_ranges"]) == a
    assert slice_box("DSP48E2_X0Y0:DSP48E2_X0Y3") is None


@pytest.mark.fpgadataflow
def test_op_of_reads_the_finn_op_type() -> None:
    """Unit cells and hierarchical cell names both yield the op type."""
    assert op_of("StreamingDataflowPartition_1_MVAU_rtl_3") == "MVAU_rtl"
    assert op_of(PREFIX + FIFO0) == "StreamingFIFO_rtl"
    assert op_of("level0_i/ulp/idma_0") == "idma"


@pytest.mark.fpgadataflow
def test_classification_and_summary() -> None:
    """Paths are split by the units of their endpoints; adjacency comes from placement."""
    tmp = _artefacts(Path(make_build_dir("timing_split_")))
    units = load_units(tmp)
    assert units["pblock_finn_unit_0"].pblock == Box(0, 0, 39, 9)
    records = classify(json.loads((tmp / "export.json").read_text()), units)
    assert [r.kind for r in records] == ["intra", "inter", "inter", "other"]
    assert records[3].start_unit == "idma_0"
    assert records[1].start_unit == "pblock_finn_unit_0"
    assert records[1].end_unit == "pblock_finn_unit_1"
    assert records[0].distance == 80 and records[2].distance == 117
    assert records[0].route_share == pytest.approx(7 / 8)

    summary = summarize(records, units, "test")
    assert summary["classes"]["intra"]["paths"] == 1
    assert summary["classes"]["inter"]["slack"]["min"] == 0.5
    assert summary["inter_adjacent"]["paths"] == 1 and summary["inter_apart"]["paths"] == 1
    # unit 0's placed box is 81 x 11 -> aspect 7.4, bucket 4-8; its pblock is 40 x 10
    assert list(summary["intra_by_placed_aspect"]) == ["4-8"]
    assert list(summary["intra_by_pblock_aspect"]) == ["2-4"]
    assert summary["worst_inter"][0]["from_op"] == "StreamingFIFO_rtl"
    assert summary["worst_inter"][0]["to_op"] == "MVAU_rtl"
    assert summary["worst_inter"][0]["paths"] == 1
    assert summary["worst_other"][0]["from"] == "idma_0"

    md = render_markdown([summary])
    assert "## Per class" in md and "test: worst inter-unit unit pairs" in md


@pytest.mark.fpgadataflow
def test_analyse_and_roots_file() -> None:
    """The driver pieces agree with the direct calls and the roots file lists every unit cell."""
    tmp = _artefacts(Path(make_build_dir("timing_split_")))
    summary, records = analyse(tmp / "export.json", tmp, "fp")
    assert summary["paths"] == 4 and len(records) == 4
    write_roots_file(load_units(tmp), tmp / "roots.txt")
    lines = (tmp / "roots.txt").read_text().splitlines()
    assert lines[0] == f"pblock_finn_unit_0 {MVAU0}"
    assert len(lines) == 3


@pytest.mark.fpgadataflow
def test_remap_units_follows_the_node_map() -> None:
    """Cells are renamed into the other build's names; cells without a counterpart vanish."""
    tmp = _artefacts(Path(make_build_dir("timing_split_")))
    units = load_units(tmp)
    mapping = {
        MVAU0: "StreamingDataflowPartition_1_MVAU_rtl_7",
        MVAU1: "StreamingDataflowPartition_1_MVAU_rtl_8",
    }
    other = remap_units(units, mapping)
    assert other["pblock_finn_unit_0"].cells == ["StreamingDataflowPartition_1_MVAU_rtl_7"]
    assert other["pblock_finn_unit_1"].cells == []
    assert other["pblock_finn_unit_0"].pblock == units["pblock_finn_unit_0"].pblock
    assert units["pblock_finn_unit_0"].cells == [MVAU0]


@pytest.mark.fpgadataflow
def test_align_cells_skips_inserted_nodes() -> None:
    """A FIFO the floorplanner inserted at an SLR crossing has no counterpart; the rest align."""
    here = [
        ("P1_MVAU_0", "MVAU_rtl"),
        ("P1_FIFO_0", "StreamingFIFO_rtl"),
        ("P2_FIFO_0", "StreamingFIFO_rtl"),  # inserted at the crossing
        ("P2_MVAU_0", "MVAU_rtl"),
        ("P2_FIFO_1", "StreamingFIFO_rtl"),
    ]
    there = [
        ("P1_MVAU_0", "MVAU_rtl"),
        ("P1_FIFO_0", "StreamingFIFO_rtl"),
        ("P1_MVAU_1", "MVAU_rtl"),
        ("P1_FIFO_1", "StreamingFIFO_rtl"),
    ]
    mapping = align_cells(here, there)
    # Two back-to-back FIFOs cannot be told apart by op type: one of them is the
    # unmatched one, every other node maps to its counterpart.
    assert mapping["P1_MVAU_0"] == "P1_MVAU_0"
    assert mapping["P2_MVAU_0"] == "P1_MVAU_1"
    assert mapping["P2_FIFO_1"] == "P1_FIFO_1"
    fifo0 = [c for c in ("P1_FIFO_0", "P2_FIFO_0") if c in mapping]
    assert len(fifo0) == 1 and mapping[fifo0[0]] == "P1_FIFO_0"
    assert len(mapping) == 4


@pytest.mark.fpgadataflow
def test_slack_is_restated_against_the_kernel_period() -> None:
    """A path under a tighter max_delay gets its slack measured against the period."""
    export = json.loads(json.dumps(EXPORT))
    export["paths"][0]["requirement"] = 5.0  # a 5 ns max_delay hop with 1.0 ns of slack
    tmp = _artefacts(Path(make_build_dir("timing_split_")))
    units = load_units(tmp)
    records = classify(export, units)
    assert records[0].slack == pytest.approx(6.0)
    assert records[1].slack == pytest.approx(2.0)


@pytest.mark.fpgadataflow
def test_a_ready_round_trip_is_an_inter_unit_path() -> None:
    """A path that leaves a unit and returns through its consumer is inter-unit to that unit."""
    export = json.loads(json.dumps(EXPORT))
    export["paths"][0]["cells"] = [f"{PREFIX}{MVAU1}/ready_lut", f"{PREFIX}{MVAU0}/x_lut"]
    tmp = _artefacts(Path(make_build_dir("timing_split_")))
    units = load_units(tmp)
    records = classify(export, units)
    assert records[0].kind == "inter"
    assert records[0].start_unit == "pblock_finn_unit_0"
    assert records[0].end_unit == "pblock_finn_unit_2"
    assert records[0].via == ("pblock_finn_unit_2",)
    assert records[1].via == () and records[1].kind == "inter"
    summary = summarize(records, units, "test")
    assert summary["classes"]["inter"]["round_trips"] == 1
    assert "1 round trips" in render_markdown([summary])


@pytest.mark.fpgadataflow
def test_instance_names_carry_the_partition_prefix_once() -> None:
    """Child node names with or without the partition prefix give the kernel's cell name."""
    assert instance_name("StreamingDataflowPartition_1", "StreamingFIFO_rtl_0") == FIFO0
    assert instance_name("StreamingDataflowPartition_1", FIFO0) == FIFO0


@pytest.mark.fpgadataflow
def test_unit_cells_are_the_design_side_of_the_alignment() -> None:
    """The units' own cells, in file order, with their op types."""
    tmp = _artefacts(Path(make_build_dir("timing_split_")))
    assert unit_cells(load_units(tmp)) == [
        (MVAU0, "MVAU_rtl"),
        (FIFO0, "StreamingFIFO_rtl"),
        (MVAU1, "MVAU_rtl"),
    ]

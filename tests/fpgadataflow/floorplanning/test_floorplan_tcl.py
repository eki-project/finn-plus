"""Tests for the Vivado-side geometry engine and the Tcl it is handed."""

import pytest

import json
import re
import subprocess
import sys
from pathlib import Path

from finn.transformation.fpgadataflow.floorplanning.tcl import (
    TCL_DEMAND_FIELDS,
    FloorplanUnitSpec,
    demands_from_estimate,
    floorplan_tcl_path,
    render_floorplan_tcl,
)
from finn.util.basic import make_build_dir, which
from finn.util.exception import FINNInternalError

#: A three-SLR Alveo part and a single-SLR Zynq part, so the engine is exercised
#: both with and without an SLR dimension to bound the search.
U55C_PART: str = "xcu55c-fsvh2892-2L-e"
ZYNQ_PART: str = "xczu3eg-sbva484-1-e"

#: How long a batch Vivado run over an empty design may take.
VIVADO_TIMEOUT_S: int = 1800


# ------------------------------------------------------------------- Tcl rendering


@pytest.mark.fpgadataflow
def test_demands_cover_every_field() -> None:
    """Every field the engine reads must be present, defaulting to 0."""
    demands = demands_from_estimate({"LUT": 100})
    assert set(demands) == set(TCL_DEMAND_FIELDS)
    # FINN has no analytical FF or CARRY8 model, so those stay zero here.
    assert demands["ffs"] == 0
    assert demands["carries"] == 0
    assert demands["luts"] == 100


@pytest.mark.fpgadataflow
def test_bram18k_is_converted_to_ramb36_sites() -> None:
    """The packer counts RAMB36 sites, so 18K estimates halve and round up."""
    assert demands_from_estimate({"BRAM_18K": 0})["brams"] == 0
    assert demands_from_estimate({"BRAM_18K": 1})["brams"] == 1
    assert demands_from_estimate({"BRAM_18K": 4})["brams"] == 2
    assert demands_from_estimate({"BRAM_18K": 5})["brams"] == 3


@pytest.mark.fpgadataflow
def test_unit_renders_as_a_tcl_dict() -> None:
    """A unit renders to the dict literal the engine's foreach expects."""
    unit = FloorplanUnitSpec(
        name="pblock_u0",
        slr=2,
        demands=demands_from_estimate({"LUT": 10, "DSP": 3}),
        cells=["top_i/Hier_a", "top_i/Hier_b"],
    )
    rendered = unit.to_tcl()
    assert rendered.startswith("{name pblock_u0 cells {top_i/Hier_a top_i/Hier_b} slr 2 demands {")
    assert "luts 10" in rendered
    assert "dsps 3" in rendered
    assert rendered.count("{") == rendered.count("}")


@pytest.mark.fpgadataflow
def test_render_sources_the_engine_and_calls_it() -> None:
    """The rendered script sources the engine before invoking it."""
    units = [FloorplanUnitSpec("pblock_u0", 0, demands_from_estimate({"LUT": 10}))]
    lines = render_floorplan_tcl(units, report_file="/tmp/report.json")
    # The interpreter and a whitelisted environment come first, so the shell can
    # run the packer under exactly the Python FINN runs in, not whatever Vivado
    # leaves on PATH.
    assert lines[0] == f"set FINN_FP_PYTHON {{{sys.executable}}}"
    assert lines[1].startswith("set FINN_FP_PACKER {") and lines[1].endswith("packer.py}")
    assert lines[2].startswith("set FINN_FP_PYTHON_ENV [list ")
    assert "{PATH}" in lines[2] and "AWS_SECRET" not in lines[2]
    assert lines[4] == "set FINN_FP_SLICEM_UTIL 0.8"
    assert lines[5] == "set FINN_FP_LOGIC_UTIL 0.75"
    assert lines[6] == "set FINN_FP_WORK_DIR {/tmp}"
    # No shell closure map passed: the list is empty and the hook prohibits nothing.
    assert lines[7] == "set FINN_FP_CLOSED_SITES [list ]"
    assert lines[8] == "set FINN_FP_CLOSED_SITES_VIVADO {}"
    assert lines[9] == "set FINN_FP_CLOSED_SITES_SOURCE {}"
    assert lines[10] == f"source {{{floorplan_tcl_path()}}}"
    assert lines[11].startswith("finn_fp_floorplan_units {")
    # units, soft, measure, margin -- measuring by default, since every real flow
    # runs the engine against a synthesized design.
    assert lines[11].split("}}} ")[1] == "1 1 1.1"
    assert lines[12] == "finn_fp_write_floorplan_report {/tmp/report.json}"


@pytest.mark.fpgadataflow
def test_no_units_renders_nothing() -> None:
    """With nothing to place, the caller can splice in the result unconditionally."""
    assert render_floorplan_tcl([]) == []


@pytest.mark.fpgadataflow
def test_duplicate_pblock_names_are_rejected() -> None:
    """Two units sharing a pblock name would silently overwrite each other."""
    demands = demands_from_estimate({"LUT": 1})
    units = [
        FloorplanUnitSpec("pblock_u0", 0, demands),
        FloorplanUnitSpec("pblock_u0", 1, demands),
    ]
    with pytest.raises(FINNInternalError, match="Duplicate"):
        render_floorplan_tcl(units)


@pytest.mark.fpgadataflow
def test_names_that_would_break_the_tcl_are_rejected() -> None:
    """A brace or space in a name must fail here, not produce a mis-parsed script."""
    demands = demands_from_estimate({"LUT": 1})
    with pytest.raises(FINNInternalError, match="Tcl list literal"):
        FloorplanUnitSpec("pblock u0", 0, demands).to_tcl()
    with pytest.raises(FINNInternalError, match="Tcl list literal"):
        FloorplanUnitSpec("pblock_u0", 0, demands, cells=["top_i/a b"]).to_tcl()


@pytest.mark.fpgadataflow
def test_missing_demand_field_is_an_internal_error() -> None:
    """A hand-built demand dict that is short a field must not reach Tcl."""
    with pytest.raises(FINNInternalError, match="missing demand field"):
        FloorplanUnitSpec("pblock_u0", 0, {"luts": 1}).to_tcl()


# ------------------------------------------------- engine helpers (no Vivado)


def _tcl_eval(body: str) -> str:
    """Source the engine in a bare tclsh and evaluate `body`, returning its stdout.

    The procs exercised here are pure list/dict logic, so they need neither a
    design nor a device -- running them under tclsh keeps the test fast and lets
    it run on a machine with no Vivado.
    """
    if which("tclsh") is None:
        pytest.skip("tclsh is not on PATH")
    script = f"source {floorplan_tcl_path()}\n{body}\n"
    result = subprocess.run(
        ["tclsh"], input=script, capture_output=True, text=True, timeout=60, check=False
    )
    if result.returncode != 0:
        pytest.fail(f"tclsh failed (rc={result.returncode}):\n{result.stderr[-2000:]}")
    return result.stdout.strip()


@pytest.mark.fpgadataflow
def test_site_intersection_keeps_only_shared_sites() -> None:
    """The census intersects an SLR's sites with its parent pblocks' sites."""
    assert _tcl_eval("puts [finn_fp_intersect_sites {a b c d} {b d e}]") == "b d"
    assert _tcl_eval("puts [finn_fp_intersect_sites {a b} {c}]") == ""
    # The first list's order is the column order the packer walks, so it must survive.
    assert _tcl_eval("puts [finn_fp_intersect_sites {d c b a} {a b c d}]") == "d c b a"


#: A census for a one-SLR device that has general logic, block RAM and DSP, but
#: no URAM at all -- enough to tell "the range was clipped" from "the demand names
#: a site this fabric does not have".
@pytest.mark.fpgadataflow
def test_region_scopes_are_empty_without_a_reconfigurable_partition() -> None:
    """With no dynamic-region pblock the whole SLR is usable, so nothing is clipped.

    get_pblocks is a Vivado command, so stub it to return nothing the way a design
    with no reconfigurable partition (a Zynq flow) would.
    """
    stub = "proc get_pblocks {args} { return {} }\n"
    assert _tcl_eval(stub + "puts '[finn_fp_region_scopes 0]'") == "''"


@pytest.mark.fpgadataflow
def test_region_scopes_prefer_the_per_slr_pblock_then_the_whole_region() -> None:
    """On a DFX shell both parents clip the child, innermost (per-SLR) first."""
    stub = "proc get_pblocks {args} { return [lindex $args end] }\n"
    out = _tcl_eval(stub + "puts [finn_fp_region_scopes 2]")
    assert out == "pblock_dynamic_SLR2 pblock_dynamic_region"


@pytest.mark.fpgadataflow
def test_site_filter_asks_for_available_sites_not_merely_present_ones() -> None:
    """The engine must count sites the placer can use, not sites that exist.

    A site occupied by static logic, explicitly prohibited, or prohibited because
    persisted static routing crosses it cannot take a cell. On a DFX shell the
    last of those is routine -- the platform's routing runs through the region the
    kernels live in -- and counting those sites as free is what made pblocks look
    big enough while the placer found them short.
    """
    out = _tcl_eval('puts [finn_fp_site_filter {SITE_TYPE == "DSP48E2"}]')
    assert 'SITE_TYPE == "DSP48E2"' in out
    for condition in ("IS_USED == 0", "PROHIBIT == 0", "PROHIBIT_FROM_PERSIST == 0"):
        assert condition in out, f"{condition} must gate the census"


@pytest.mark.fpgadataflow
def test_every_global_read_inside_a_proc_is_declared() -> None:
    """A Tcl proc sees a global only if it says so, and nothing else catches this.

    `tclsh` parses a proc body without evaluating it, so a missing `global` is not
    a syntax error; and the paths that read these are reachable only on a DFX
    flow, which no test here can stand up. A build died at
    "can't read FINN_FP_MAX_FIT_ATTEMPTS: no such variable" after four hours for
    exactly this, so check it statically instead.
    """
    source = Path(floorplan_tcl_path()).read_text()
    parts = re.split(r"^proc (\w+) ", source, flags=re.M)
    missing: dict[str, set[str]] = {}
    for name, body in zip(parts[1::2], parts[2::2], strict=True):
        body = body.split("\n################")[0]
        declared: set[str] = set()
        for decl in re.finditer(r"^\s*global ([^\n]+)", body, re.M):
            declared.update(decl.group(1).split())
        used = set(re.findall(r"\$(FINN_FP_\w+)", body))
        if used - declared:
            missing[name] = used - declared
    assert not missing, f"procs reading undeclared globals: {missing}"


@pytest.mark.fpgadataflow
def test_chain_packing_in_the_shell_matches_the_packer() -> None:
    """The shell re-checks chains on real site runs with the same best-fit rule.

    mobilenetv1's last MVAU carries 17 block RAM chains of up to 8; an
    11-column, 12-row window held the longest chain and the placer still
    rejected it ([Constraints 18-11842]). Every chain needs its own segment.
    """
    assert _tcl_eval("puts [finn_fp_chains_pack {12} {8 8}]") == "0"
    assert _tcl_eval("puts [finn_fp_chains_pack {12 12} {8 8}]") == "1"
    assert _tcl_eval("puts [finn_fp_chains_pack {12} {6 6}]") == "1", "two 6s stack in 12 rows"
    assert _tcl_eval("puts [finn_fp_chains_pack {} {2}]") == "0"


@pytest.mark.fpgadataflow
def test_ramb18_chains_ride_on_ramb36_columns() -> None:
    """RAMB18 chains share the block RAM columns, two RAMB18 rows per RAMB36 row."""
    assert _tcl_eval("puts [finn_fp_chain_lens_for {RAMB18 {15 4}} RAMB36]") == "8 2"
    assert _tcl_eval("puts [finn_fp_chain_lens_for {RAMB36 {5} RAMB18 {4}} RAMB36]") == "5 2"


@pytest.mark.fpgadataflow
def test_runs_collapse_rows_and_json_strings_are_escaped() -> None:
    """Census export helpers: available rows become runs, strings survive quoting."""
    assert _tcl_eval("puts [finn_fp_runs {5 1 2 3 9 10}]") == "{1 3} {5 5} {9 10}"
    assert _tcl_eval("puts [finn_fp_json_runs {{1 3} {9 10}}]") == "[[1,3],[9,10]]"
    assert _tcl_eval(r'puts [finn_fp_json_str {a"b}]') == r'"a\"b"'


@pytest.mark.fpgadataflow
def test_units_export_carries_demands_and_chains(tmp_path: Path) -> None:
    """What the shell hands the packer: every demand key, the chains as lists."""
    out = tmp_path / "units.json"
    body = (
        "set units {{name pblock_a cells {x} slr 1 demands {luts 10 ffs 2 carries 0 brams 3 "
        "dsps 0 urams 0 lutrams 0 srls 0} chains {RAMB36 {8 8}}}}\n"
        f"finn_fp_export_units $units {{{out}}}"
    )
    _tcl_eval(body)
    data = json.loads(out.read_text())
    assert data == [
        {
            "name": "pblock_a",
            "slr": 1,
            "order": 0,
            "demands": {
                "luts": 10,
                "ffs": 2,
                "carries": 0,
                "brams": 3,
                "dsps": 0,
                "urams": 0,
                "lutrams": 0,
                "srls": 0,
                "ram_macros": 0,
            },
            "chains": {"RAMB36": [8, 8]},
            "cells": ["x"],
        }
    ]


def _run_engine(part: str, units: list[FloorplanUnitSpec], work_dir: Path) -> dict:
    """Run the geometry engine over an empty design on `part` and return its report.

    An empty design is enough: the packer's input is the demand numbers and the
    device's own fabric layout, neither of which needs a netlist. Cells are left
    unresolved, which the engine warns about and otherwise ignores.
    """
    report_file = work_dir / "floorplan.json"
    script = work_dir / "drive.tcl"
    lines = [f"link_design -part {part}"]
    lines += render_floorplan_tcl(units, report_file=report_file)
    script.write_text("\n".join(lines) + "\n")

    result = subprocess.run(
        ["vivado", "-mode", "batch", "-nojournal", "-nolog", "-source", str(script)],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=VIVADO_TIMEOUT_S,
        check=False,
    )
    if result.returncode != 0 or not report_file.is_file():
        pytest.fail(f"Vivado run failed (rc={result.returncode}):\n{result.stdout[-4000:]}")
    return json.loads(report_file.read_text())


def _slice_y_ranges(grid: str) -> list[tuple[int, int]]:
    """Return the (y0, y1) of every SLICE range in a GRID_RANGES string."""
    return [
        (int(m.group(1)), int(m.group(2)))
        for m in re.finditer(r"SLICE_X\d+Y(\d+):SLICE_X\d+Y(\d+)", grid)
    ]


def _sparse_ranges(grid: str, kind: str) -> list[tuple[int, int, int, int]]:
    """Return the (x0, y0, x1, y1) of every range of a sparse site kind."""
    return [
        (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
        for m in re.finditer(rf"{kind}_X(\d+)Y(\d+):{kind}_X(\d+)Y(\d+)", grid)
    ]


@pytest.mark.vivado
@pytest.mark.slow
@pytest.mark.fpgadataflow
@pytest.mark.parametrize("part", [U55C_PART, ZYNQ_PART])
def test_engine_places_every_unit_inside_its_slr(part: str) -> None:
    """Each unit gets a satisfied pblock, and never leaves the SLR it was assigned."""
    if which("vivado") is None:
        pytest.skip("vivado is not on PATH")

    multi_slr = part == U55C_PART
    if multi_slr:
        estimates = [
            {"LUT": 20000, "BRAM_18K": 80, "DSP": 60, "URAM": 8},
            {"LUT": 15000, "BRAM_18K": 40, "DSP": 30},
            {"LUT": 9000, "BRAM_18K": 20, "DSP": 10, "URAM": 4},
            {"LUT": 4000},
        ]
    else:
        # The ZU3EG is far smaller and, unlike the Alveo parts, has no URAM at
        # all -- asking for some would (correctly) come back unsatisfiable.
        estimates = [
            {"LUT": 5000, "BRAM_18K": 20, "DSP": 15},
            {"LUT": 3000, "BRAM_18K": 8, "DSP": 8},
            {"LUT": 2000, "BRAM_18K": 4, "DSP": 2},
            {"LUT": 1000},
        ]

    units = [
        FloorplanUnitSpec(
            name=f"pblock_unit_{i}",
            slr=(i % 3) if multi_slr else 0,
            demands=demands_from_estimate(est),
        )
        for i, est in enumerate(estimates)
    ]
    report = _run_engine(part, units, Path(make_build_dir("floorplan_engine_")))

    assert set(report["units"]) == {unit.name for unit in units}
    for unit in units:
        entry = report["units"][unit.name]
        assert entry["satisfied"], f"{unit.name} did not get the area it asked for"
        assert entry["slr"] == unit.slr
        capacity = entry["pblock_capacity"]
        for field_name in ("luts", "brams", "dsps", "urams"):
            assert capacity[field_name] >= unit.demands[field_name]

    if not multi_slr:
        return

    # SLR bounding: units of different SLRs must not share any SLICE row.
    rows_by_slr: dict[int, set[int]] = {}
    for unit in units:
        entry = report["units"][unit.name]
        for y0, y1 in _slice_y_ranges(entry["grid_ranges"]):
            rows_by_slr.setdefault(entry["slr"], set()).update(range(y0, y1 + 1))
    for slr_a, rows_a in rows_by_slr.items():
        for slr_b, rows_b in rows_by_slr.items():
            if slr_a < slr_b:
                assert not (rows_a & rows_b), f"SLR{slr_a} and SLR{slr_b} pblocks overlap in Y"


@pytest.mark.vivado
@pytest.mark.slow
@pytest.mark.fpgadataflow
def test_engine_does_not_hand_one_sparse_column_to_two_units() -> None:
    """BRAM/DSP/URAM sites are scarce: two units must never be given the same ones.

    Handing a site to several units at once looks fine to a packer that counts
    capacity per unit -- and then they compete for it during placement, which is
    exactly the congestion the floorplanner is meant to avoid.
    """
    if which("vivado") is None:
        pytest.skip("vivado is not on PATH")

    units = [
        FloorplanUnitSpec(
            name=f"pblock_unit_{i}",
            slr=0,
            demands=demands_from_estimate({"LUT": 8000, "BRAM_18K": 40, "DSP": 20, "URAM": 4}),
        )
        for i in range(4)
    ]
    report = _run_engine(U55C_PART, units, Path(make_build_dir("floorplan_columns_")))

    # Two units may share a column at different rows -- the packer stacks
    # rectangles in Y -- but never the same sites.
    for kind in ("RAMB36", "DSP48E2", "URAM288"):
        claimed: dict[tuple[int, int], str] = {}
        for name, entry in report["units"].items():
            for x0, y0, x1, y1 in _sparse_ranges(entry["grid_ranges"], kind):
                for site in ((x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)):
                    assert (
                        site not in claimed
                    ), f"{kind} X{site[0]}Y{site[1]} given to both {claimed[site]} and {name}"
                    claimed[site] = name


@pytest.mark.vivado
@pytest.mark.slow
@pytest.mark.fpgadataflow
def test_unsatisfiable_unit_degrades_instead_of_failing() -> None:
    """A demand the device cannot meet is reported, not raised.

    A demand the fabric cannot honour anywhere must leave that unit unconstrained
    and say so -- failing the whole build over it would be worse than placing it
    freely, and a hard pblock that is too small is a UTLZ-1 error, not a hint.
    """
    if which("vivado") is None:
        pytest.skip("vivado is not on PATH")

    units = [
        FloorplanUnitSpec("pblock_fits", 0, demands_from_estimate({"LUT": 5000})),
        # The ZU3EG has no URAM at all, so this can never be satisfied.
        FloorplanUnitSpec("pblock_impossible", 0, demands_from_estimate({"LUT": 100, "URAM": 4})),
    ]
    report = _run_engine(ZYNQ_PART, units, Path(make_build_dir("floorplan_degrade_")))

    assert report["units"]["pblock_fits"]["satisfied"]
    # Hard pblocks cannot be "a bit short" (UTLZ-1), so the unit is left free and
    # reported as unplaced rather than given a region it cannot fit.
    assert "pblock_impossible" not in report["units"]


@pytest.mark.vivado
@pytest.mark.slow
@pytest.mark.fpgadataflow
def test_engine_creates_soft_pblocks_without_pr_properties() -> None:
    """Static pblocks must be soft and must not carry the DFX-only properties."""
    if which("vivado") is None:
        pytest.skip("vivado is not on PATH")

    work_dir = Path(make_build_dir("floorplan_soft_"))
    units = [FloorplanUnitSpec("pblock_unit_0", 0, demands_from_estimate({"LUT": 5000}))]
    script = work_dir / "drive.tcl"
    lines = [f"link_design -part {U55C_PART}"]
    lines += render_floorplan_tcl(units)
    lines += [
        'puts "PROBE IS_SOFT=[get_property IS_SOFT [get_pblocks pblock_unit_0]]"',
        (
            'puts "PROBE CONTAIN_ROUTING='
            '[get_property -quiet CONTAIN_ROUTING [get_pblocks pblock_unit_0]]"'
        ),
    ]
    script.write_text("\n".join(lines) + "\n")

    result = subprocess.run(
        ["vivado", "-mode", "batch", "-nojournal", "-nolog", "-source", str(script)],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=VIVADO_TIMEOUT_S,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:]
    assert "PROBE IS_SOFT=1" in result.stdout
    # CONTAIN_ROUTING is a DFX property; on a static pblock it must be unset/0.
    assert "PROBE CONTAIN_ROUTING=1" not in result.stdout


# ------------------------------------------------- against a synthesized netlist

#: A small synthesizable design whose instance names mimic FINN node names and sit
#: one level down, so bare-name resolution has something to search for.
PROBE_DESIGN: Path = Path(__file__).parent / "floorplan_probe.v"

#: What Vivado reports for the probe design. Checked against the real numbers
#: rather than only self-consistency, so a parser that silently returns zeros fails.
PROBE_EXPECTED: dict[str, dict[str, int]] = {
    "MVAU_hls_0": {"luts": 48, "ffs": 65, "brams": 0, "dsps": 2},
    "MVAU_hls_1": {"luts": 48, "ffs": 65, "brams": 0, "dsps": 2},
    "StreamingFIFO_rtl_0": {"luts": 0, "ffs": 0, "brams": 2, "dsps": 0},
}


@pytest.mark.vivado
@pytest.mark.slow
@pytest.mark.fpgadataflow
def test_engine_reads_utilisation_and_binds_real_cells() -> None:
    """Query a synthesized netlist and bind pblocks to the cells it reports.

    The empty-design tests above cover geometry, but not the two things that only
    exist once a netlist does: reading per-cell utilisation, and resolving a FINN
    node name to wherever Vivado actually put that cell.
    """
    if which("vivado") is None:
        pytest.skip("vivado is not on PATH")

    work_dir = Path(make_build_dir("floorplan_netlist_"))
    cells = list(PROBE_EXPECTED)
    script = work_dir / "drive.tcl"
    script.write_text(
        "\n".join(
            [
                f"set FINN_FP_PYTHON {{{sys.executable}}}",
                f"source {{{floorplan_tcl_path()}}}",
                f"read_verilog {{{PROBE_DESIGN}}}",
                f"synth_design -top top -part {U55C_PART} -mode out_of_context",
                f"set util [finn_fp_query_cells_batch {{{' '.join(cells)}}}]",
                "dict for {cell res} $util {",
                (
                    '    puts "UTIL $cell [dict get $res luts] [dict get $res ffs]'
                    ' [dict get $res brams] [dict get $res dsps]"'
                ),
                "}",
                "set units {}",
                "set i 0",
                f"foreach cell {{{' '.join(cells)}}} {{",
                "    set res [dict get $util $cell]",
                "    lappend units [dict create name pblock_probe_$i cells [list $cell] slr 0 \\",
                "        demands [dict create luts [expr {[dict get $res luts] + 200}] \\",
                "            ffs [expr {[dict get $res ffs] + 200}] carries 0 \\",
                "            brams [expr {[dict get $res brams] + 1}] \\",
                "            dsps [expr {[dict get $res dsps] + 1}] urams 0]]",
                "    incr i",
                "}",
                "finn_fp_floorplan_units $units 1",
                "foreach pb [lsort [get_pblocks -quiet pblock_probe_*]] {",
                '    puts "BOUND $pb [get_cells -quiet -of_objects $pb]"',
                "}",
            ]
        )
        + "\n"
    )

    result = subprocess.run(
        ["vivado", "-mode", "batch", "-nojournal", "-nolog", "-source", str(script)],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=VIVADO_TIMEOUT_S,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:]

    reported = {}
    for line in result.stdout.splitlines():
        if line.startswith("UTIL "):
            _, cell, luts, ffs, brams, dsps = line.split()
            reported[cell] = {
                "luts": int(luts),
                "ffs": int(ffs),
                "brams": int(brams),
                "dsps": int(dsps),
            }
    assert reported == PROBE_EXPECTED

    # The nested cells must resolve to their real paths and actually land in a pblock.
    bound = dict(
        line.split(maxsplit=2)[1:]
        for line in result.stdout.splitlines()
        if line.startswith("BOUND")
    )
    assert bound["pblock_probe_0"].strip() == "hier/MVAU_hls_0"
    assert bound["pblock_probe_1"].strip() == "hier/MVAU_hls_1"
    assert bound["pblock_probe_2"].strip() == "StreamingFIFO_rtl_0"


def _run_probe_script(body: list[str], work_dir: Path) -> str:
    """Synthesize the probe design and run `body` against it, returning stdout."""
    script = work_dir / "drive.tcl"
    script.write_text(
        "\n".join(
            [
                f"set FINN_FP_PYTHON {{{sys.executable}}}",
                f"source {{{floorplan_tcl_path()}}}",
                f"read_verilog {{{PROBE_DESIGN}}}",
                f"synth_design -top top -part {U55C_PART} -mode out_of_context",
                *body,
            ]
        )
        + "\n"
    )
    result = subprocess.run(
        ["vivado", "-mode", "batch", "-nojournal", "-nolog", "-source", str(script)],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=VIVADO_TIMEOUT_S,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:]
    # Vivado echoes every line of a sourced script, comment-prefixed, so the text
    # inside a `puts` shows up whether or not that puts ever ran. Drop the echo,
    # otherwise asserting on output really asserts on the script's own source.
    return "\n".join(
        line for line in result.stdout.splitlines() if not line.lstrip().startswith("#")
    )


def _units_tcl(luts: int) -> str:
    """Return a Tcl unit list for the probe design's MVAU, claiming `luts` LUTs."""
    return (
        "set units [list [dict create name pblock_a cells {MVAU_hls_0} slr 0 "
        f"demands [dict create luts {luts} ffs 200 carries 0 brams 0 dsps 3 urams 0]]]"
    )


@pytest.mark.vivado
@pytest.mark.slow
@pytest.mark.fpgadataflow
def test_measurement_sizes_a_unit_from_what_it_really_uses() -> None:
    """With measurement on, the estimate does not size the pblock at all.

    The probe MVAU really uses 48 LUTs. Claiming 5000 must not make its region any
    bigger: the engine runs on a synthesized design, so it reads the real number
    before packing rather than packing from the claim and correcting afterwards.
    """
    if which("vivado") is None:
        pytest.skip("vivado is not on PATH")

    stdout = _run_probe_script(
        [
            _units_tcl(5000),
            # measure off: the 5000-LUT claim is all the engine has to go on.
            "finn_fp_floorplan_units $units 1 0",
            'puts "WIDE [get_property GRID_RANGES [get_pblocks pblock_a]]"',
            # measure on: sized from the 48 LUTs the cell actually uses.
            "finn_fp_floorplan_units $units 1 1 1.1",
            'puts "SOURCE $::finn_fp_resource_source"',
            'puts "NARROW [get_property GRID_RANGES [get_pblocks pblock_a]]"',
        ],
        Path(make_build_dir("floorplan_measure_")),
    )

    assert "Measuring utilisation for" in stdout
    assert "SOURCE post_synth" in stdout
    wide = next(line for line in stdout.splitlines() if line.startswith("WIDE "))
    narrow = next(line for line in stdout.splitlines() if line.startswith("NARROW "))

    def slice_sites(line: str) -> int:
        """Return the SLICE sites over every SLICE range in a GRID_RANGES line.

        The packer claims columns selectively, so one pblock may carry several
        SLICE ranges; the size that matters is their total, not the first span.
        """
        total = 0
        for m in re.finditer(r"SLICE_X(\d+)Y(\d+):SLICE_X(\d+)Y(\d+)", line):
            x0, y0, x1, y1 = (int(v) for v in m.groups())
            total += (x1 - x0 + 1) * (y1 - y0 + 1)
        assert total > 0
        return total

    assert slice_sites(narrow) < slice_sites(wide), "the over-estimated unit must shrink"


@pytest.mark.vivado
@pytest.mark.slow
@pytest.mark.fpgadataflow
def test_measurement_packs_only_once() -> None:
    """Sizing from measurements must not leave an estimate-sized pblock behind.

    The engine used to pack from the estimates and then re-pack from measurements,
    which meant a unit the second pass could not place kept the pblock the first
    pass had given it -- hard, undersized, and a guaranteed UTLZ-1 inside a
    reconfigurable partition. Measuring first removes that window entirely, so
    there must be exactly one packing pass.
    """
    if which("vivado") is None:
        pytest.skip("vivado is not on PATH")

    stdout = _run_probe_script(
        [
            _units_tcl(5000),
            "finn_fp_floorplan_units $units 1 1 1.1",
        ],
        Path(make_build_dir("floorplan_onepass_")),
    )
    assert stdout.count("FINN+ automatic pblock floorplanning") == 1
    assert stdout.count("Floorplanning done:") == 1
    assert "Re-packing from measured utilisation" not in stdout


@pytest.mark.vivado
@pytest.mark.slow
@pytest.mark.fpgadataflow
def test_every_pblock_ranges_every_primitive_type() -> None:
    """A pblock must range all five site types, even ones the unit did not ask for.

    On Alveo the whole user region is a reconfigurable partition, so every cell in
    it is HD.RECONFIGURABLE and Vivado enforces "a reconfigurable Pblock must range
    all of its cell primitive types" (HDPR-18). A unit whose estimate said 0 BRAM
    can still contain BRAM primitives -- an MVAU weight streamer is the usual case
    -- and omitting the range fails DRC at v++ link time, after hours of build.
    """
    if which("vivado") is None:
        pytest.skip("vivado is not on PATH")

    units = [
        # Deliberately mixed: some units demand nothing but LUTs.
        FloorplanUnitSpec("pblock_all_0", 0, demands_from_estimate({"LUT": 9000, "BRAM_18K": 20})),
        FloorplanUnitSpec("pblock_all_1", 0, demands_from_estimate({"LUT": 9000})),
        FloorplanUnitSpec("pblock_all_2", 0, demands_from_estimate({"LUT": 4000})),
    ]
    report = _run_engine(U55C_PART, units, Path(make_build_dir("floorplan_hdpr_")))

    required = ("SLICE", "RAMB36", "RAMB18", "DSP48E2", "URAM288")
    for name in (unit.name for unit in units):
        grid = report["units"][name]["grid_ranges"]
        missing = [kind for kind in required if f"{kind}_X" not in grid]
        assert not missing, f"{name} has no {missing} range: {grid}"


@pytest.mark.vivado
@pytest.mark.slow
@pytest.mark.fpgadataflow
def test_closed_sites_are_prohibited_before_the_census(tmp_path: Path) -> None:
    """The shell's closed sites, and the RAMB18 halves of a closed RAMB36, leave the census.

    The census is what the packer sizes from and what the verify step counts, so a
    closed site that reached it would be handed to a unit as capacity the placer
    never grants.
    """
    if which("vivado") is None:
        pytest.skip("vivado is not on PATH")
    census = tmp_path / "census.json"
    out = tmp_path / "prohibit.txt"
    script = tmp_path / "drive.tcl"
    script.write_text(
        "\n".join(
            [
                f"link_design -part {U55C_PART}",
                "set FINN_FP_CLOSED_SITES [list RAMB36_X12Y0 DSP48E2_X29Y6 NO_SUCH_SITE_X0Y0]",
                "set FINN_FP_CLOSED_SITES_VIVADO {1999.1}",
                "set FINN_FP_CLOSED_SITES_SOURCE {test}",
                f"source {{{floorplan_tcl_path()}}}",
                "set n [finn_fp_prohibit_closed_sites]",
                f"set fh [open {{{out}}} w]",
                'puts $fh "n $n"',
                (
                    "foreach s {RAMB36_X12Y0 RAMB18_X12Y0 RAMB18_X12Y1 RAMB18_X12Y2 DSP48E2_X29Y6"
                    ' DSP48E2_X29Y7} { puts $fh "$s [get_property PROHIBIT [get_sites $s]]" }'
                ),
                "close $fh",
                f"finn_fp_export_census {{{census}}}",
            ]
        )
        + "\n"
    )
    result = subprocess.run(
        ["vivado", "-mode", "batch", "-nojournal", "-nolog", "-source", str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=VIVADO_TIMEOUT_S,
        check=False,
    )
    if result.returncode != 0 or not out.is_file():
        pytest.fail(f"Vivado run failed (rc={result.returncode}):\n{result.stdout[-4000:]}")
    assert "measured under Vivado 1999.1" in result.stdout
    assert "1 closed site(s) of the shell closure map do not exist" in result.stdout
    got = dict(line.split() for line in out.read_text().splitlines())
    assert got == {
        "n": "4",
        "RAMB36_X12Y0": "1",
        "RAMB18_X12Y0": "1",
        "RAMB18_X12Y1": "1",
        "RAMB18_X12Y2": "0",
        "DSP48E2_X29Y6": "1",
        "DSP48E2_X29Y7": "0",
    }
    data = json.loads(census.read_text())
    slr0 = data["columns"]["0"]
    bram12 = next(c for c in slr0["RAMB36"] if c["x"] == 12)
    assert bram12["runs"][0][0] == 1, "row 0 of RAMB36_X12 is gone from the census"
    dsp29 = next(c for c in slr0["DSP48E2"] if c["x"] == 29)
    assert all(not (a <= 6 <= b) for a, b in dsp29["runs"]), dsp29["runs"]


@pytest.mark.fpgadataflow
def test_pblocks_below_the_programmable_unit_minimum_are_read_from_the_drc() -> None:
    """Vivado's HDPRA-65 count decides which pblocks grow, and by how much.

    A Programmable Unit has no fixed slice size (on the U55C a 5-slice and a
    10-slice pblock both count 3), so the hook asks Vivado instead of guessing.
    """
    report = (
        "HDPRA-65#1 Critical Warning\\n"
        "Small nested Pblock inside Reconfigurable Pblock: Reconfigurable Pblock "
        "'pblock_dynamic_region' contains a nested Pblock 'pblock_finn_unit_105' that is too "
        "small (3 Programmable Units). This may lead to unroute during implementation.\\n"
        "HDPRA-65#2 Critical Warning\\n"
        "... contains a nested Pblock 'pblock_finn_unit_197' that is too small "
        "(4 Programmable Units). ...\\n"
        "... contains a nested Pblock 'pblock_other' that is too small (1 Programmable Units).\\n"
    )
    out = _tcl_eval(
        f'proc report_drc {{args}} {{ return "{report}" }}\n'
        'dict for {n g} [finn_fp_small_pblocks] { puts "$n [format %.3f $g]" }'
    )
    assert out.split("\n") == ["pblock_finn_unit_105 0.667", "pblock_finn_unit_197 0.250"]

"""Tests for the Python packer: shapes, chains, feedback rounds, and the real fixtures.

Everything here is `quick`-tier: the packer is pure arithmetic over JSON, and
the fixtures are a U55C device census exported by the Tcl shell plus the unit
demands measured in the Milestone-5 validation runs of 2 Oct 2026.
"""

import pytest

import json
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

from finn.transformation.fpgadataflow.floorplanning.packer import (
    DEFAULT_LOGIC_UTILIZATION,
    SLICE_LUTS,
    Census,
    Column,
    Placement,
    SlrPacker,
    Unit,
    footprint,
    load_previous,
    pack,
    pack_chains,
    slr_budget,
    summarize,
    write_result,
)
from finn.transformation.fpgadataflow.floorplanning.tcl import floorplan_packer_path

FIXTURES = Path(__file__).parent / "fixtures"


def _census(slice_cols: int = 12, bram_cols: int = 2, dsp_cols: int = 1, rows: int = 120) -> Census:
    """Build a small synthetic SLR: SLICE columns with a BRAM and a DSP column interleaved.

    Rows are two clock regions of 60 SLICE rows; block RAM has 12 rows per region,
    DSP 24. Tile columns interleave the kinds the way the U55C does.
    """
    cols: dict[str, list[Column]] = {
        "SLICE": [],
        "RAMB36": [],
        "RAMB18": [],
        "DSP48E2": [],
        "URAM288": [],
    }
    tile = 0
    b = d = 0
    for x in range(slice_cols):
        cols["SLICE"].append(Column("SLICE", x, tile, [(0, rows - 1)], slicem=(x % 2 == 1)))
        tile += 1
        if x % 4 == 1 and b < bram_cols:
            cols["RAMB36"].append(Column("RAMB36", b, tile, [(0, rows // 5 - 1)]))
            cols["RAMB18"].append(Column("RAMB18", b, tile, [(0, 2 * rows // 5 - 1)]))
            b += 1
            tile += 1
        if x % 4 == 3 and d < dsp_cols:
            cols["DSP48E2"].append(Column("DSP48E2", d, tile, [(0, 2 * rows // 5 - 1)]))
            d += 1
            tile += 1
    n_bands = rows // 60
    bands = {
        "SLICE": [(60 * i, 60 * i + 59) for i in range(n_bands)],
        "RAMB36": [(12 * i, 12 * i + 11) for i in range(n_bands)],
        "RAMB18": [(24 * i, 24 * i + 23) for i in range(n_bands)],
        "DSP48E2": [(24 * i, 24 * i + 23) for i in range(n_bands)],
    }
    return Census(slrs=[0], bands=bands, columns={0: cols})


def _unit(
    name: str,
    luts: int = 0,
    brams: int = 0,
    dsps: int = 0,
    chains: dict[str, list[int]] | None = None,
    order: int = 0,
) -> Unit:
    """Build a unit with the given demands (other fields zero)."""
    demands = dict.fromkeys(
        ("luts", "ffs", "carries", "brams", "dsps", "urams", "lutrams", "srls"), 0
    )
    demands.update(luts=luts, brams=brams, dsps=dsps)
    return Unit(name=name, slr=0, order=order, demands=demands, chains=chains or {})


@pytest.mark.fpgadataflow
def test_chain_packing_is_best_fit_decreasing() -> None:
    """Every chain needs its own segment; short chains may stack in one."""
    assert pack_chains([12], [8, 8]) is False
    assert pack_chains([12, 12], [8, 8]) is True
    assert pack_chains([12], [6, 6]) is True
    assert pack_chains([8, 4, 4], [4, 4, 4, 4]) is True
    assert pack_chains([], [2]) is False


@pytest.mark.fpgadataflow
def test_rows_map_through_matching_clock_region_bands() -> None:
    """SLICE rows 60..119 are the second region, so they map to RAMB36 rows 12..23."""
    census = _census()
    assert census.map_rows("RAMB36", 60, 119) == (12, 23)
    assert census.map_rows("RAMB36", 0, 4) == (0, 0), "five SLICE rows are one block RAM row"
    assert census.map_rows("DSP48E2", 0, 59) == (0, 23)
    assert census.map_rows("SLICE", 7, 9) == (7, 9)


@pytest.mark.fpgadataflow
def test_a_small_unit_gets_a_small_rectangle_with_every_kind_ranged() -> None:
    """Forty LUTs get a small rectangle: two columns, five rows -- not a shelf of 21.

    At the logic ceiling (0.75) forty LUTs need seven slices, so one column of
    five rows no longer holds them; the packer takes a second column.

    Inside a reconfigurable partition every primitive type must be ranged
    (HDPR-18), so the rectangle borrows the nearest block RAM and DSP column for
    the kinds it does not use.
    """
    result = pack(_census(), [_unit("u", luts=40)])
    p = result["u"]
    assert p.status == "placed"
    assert p.y1 - p.y0 + 1 == 5
    assert p.capacity["luts"] == 80
    assert p.capacity["luts"] * DEFAULT_LOGIC_UTILIZATION >= 40
    kinds = {r.split("_X")[0] for r in p.ranges}
    assert {"SLICE", "RAMB36", "RAMB18", "DSP48E2"} <= kinds


@pytest.mark.fpgadataflow
def test_claimed_ranges_stop_at_the_census_rows() -> None:
    """A shell-clipped block RAM column is ranged over its free rows only.

    Job 3551006 (3 Oct 2026): SLR0's two rightmost block RAM columns keep 12 of
    48 rows under the static shell; the packer claimed them for a 240-row unit
    and ranged RAMB36 Y0:Y47, so Vivado counted 96 sites against a pblock the
    census had sized at 24. Claimed ranges follow the column's runs; a column
    with a gap gets one range per run.
    """
    census = _census(slice_cols=4, bram_cols=1, dsp_cols=0, rows=240)
    bram = census.columns[0]["RAMB36"][0]
    bram.runs = [(0, 11)]
    census.columns[0]["RAMB18"][0].runs = [(0, 23)]
    result = pack(census, [_unit("u", luts=4000, brams=10)])
    p = result["u"]
    assert p.status == "placed"
    assert p.y1 - p.y0 + 1 >= 60, "ten block RAMs over 12 rows need the whole column"
    ramb = [r for r in p.ranges if r.startswith("RAMB36")]
    assert ramb == ["RAMB36_X0Y0:RAMB36_X0Y11"], ramb
    assert p.capacity["brams"] == 12

    bram.runs = [(0, 5), (8, 11)]
    result = pack(census, [_unit("u", luts=4000, brams=8)])
    ramb = sorted(r for r in result["u"].ranges if r.startswith("RAMB36"))
    assert ramb == ["RAMB36_X0Y0:RAMB36_X0Y5", "RAMB36_X0Y8:RAMB36_X0Y11"], ramb


@pytest.mark.fpgadataflow
def test_dsp_heavy_unit_grows_tall_instead_of_wide() -> None:
    """A unit needing most of the DSP column takes rows, not the whole SLR width.

    The synthetic SLR has one DSP column with 48 sites; 40 DSPs need 40 of its
    rows, i.e. 100 SLICE rows. The shelf engine would have taken a full-width
    strip; the packer takes the DSP column and the SLICEs beside it.
    """
    result = pack(_census(), [_unit("mvau", luts=400, dsps=40)])
    p = result["mvau"]
    assert p.status == "placed"
    assert p.y1 - p.y0 + 1 >= 100
    assert len(p.columns) <= 3, f"claimed {len(p.columns)} columns for one DSP column plus SLICEs"
    assert p.capacity["dsps"] >= 40


@pytest.mark.fpgadataflow
def test_every_chain_gets_its_own_segment_inside_one_clock_region() -> None:
    """Two 8-deep block RAM chains need two column-band segments of eight rows.

    One block RAM column has 12 rows per region, enough for one 8-chain per
    region; with two columns a 60-row rectangle suffices, as does a 120-row one
    on a single column. A unit with three such chains and two columns must grow
    into the second region.
    """
    two = pack(_census(), [_unit("two", luts=100, brams=16, chains={"RAMB36": [8, 8]})])["two"]
    assert two.status == "placed"
    three = pack(_census(), [_unit("three", luts=100, brams=24, chains={"RAMB36": [8, 8, 8]})])[
        "three"
    ]
    assert three.status == "placed"
    assert three.y1 - three.y0 + 1 >= 120, "three 8-chains on two columns span two regions"


@pytest.mark.fpgadataflow
def test_memory_luts_are_sized_against_the_slicem_ceiling() -> None:
    """Distributed RAM without a macro count fills only a fraction of the SLICEM LUTs.

    With the 0.8 ceiling a unit of 600 LUTRAM LUTs needs 94 SLICEM slices, not 75;
    with the ceiling at 1.0 the plain count rule is back.
    """

    def ram() -> Unit:
        unit = _unit("ram", luts=600)
        unit.demands["lutrams"] = 600
        return unit

    tight = pack(_census(), [ram()], slicem_util=1.0)["ram"]
    roomy = pack(_census(), [ram()], slicem_util=0.8)["ram"]
    assert tight.status == roomy.status == "placed"
    assert roomy.capacity["lutrams"] / 0.8 / 8 > tight.capacity["lutrams"] / 8
    assert ram().slicem_slices(0.8) == 94 and ram().slicem_slices(1.0) == 75


@pytest.mark.fpgadataflow
def test_shift_registers_pack_at_their_measured_density() -> None:
    """A FINN SRL FIFO lane does not share a slice the way its LUT count suggests.

    1,120 SRLC32E (224 deep x 160 wide) failed in hard regions of 280, 300 and 320
    SLICEM slices, placed in 340, and then occupied exactly 320 slices (U55C,
    2 Oct 2026); at eight LUTs per slice they would have "fit" in 140. The packer
    sizes shift registers at the measured 3.5 per slice, independent of the
    LUTRAM ceiling.
    """
    unit = _unit("fifo", luts=1500)
    unit.demands["srls"] = 1120
    assert unit.slicem_slices() == 320
    assert unit.slicem_slices(slicem_util=1.0) == 320, "the ceiling is not applied twice"
    assert unit.slicem_slices(srl_per_slice=8.0) == 140


@pytest.mark.fpgadataflow
def test_ram_macros_count_as_slices_under_the_ceiling() -> None:
    """Each distributed-RAM macro takes a SLICEM slice, plus the utilisation ceiling.

    An input generator with 32 RAM64M8 and 32 RAM64X1D (320 memory LUTs) occupies
    exactly 64 slices once placed, but out of context a region with 64 SLICEM
    never placed it, and one with 80 (64 / 0.8) did (3 Oct 2026).
    """
    unit = _unit("swg", luts=652)
    unit.demands["lutrams"] = 320
    assert unit.slicem_slices() == 50
    unit.demands["ram_macros"] = 64
    assert unit.slicem_slices() == 80
    assert unit.slicem_slices(slicem_util=1.0) == 64
    placed = pack(_census(), [unit])["swg"]
    assert placed.status == "placed" and placed.capacity["lutrams"] / 0.8 / 8 >= 80


@pytest.mark.fpgadataflow
def test_memory_slices_and_logic_slices_add_up() -> None:
    """A slice holding a macro holds no logic, so the unit needs both counts.

    Unit 2 of mobilenetv1 in job 3551390 (input generator: 718 LUTs of which 352
    distributed RAM in 64 macros) was given 90 slices because 718 LUTs "fit" at
    eight per slice; it needs 80 SLICEM for its macros and 46 slices for its 366
    logic LUTs. Out of context it failed in its link region and placed in
    80 SLICEM + 50 SLICEL. Logic fills its slices up to the logic ceiling: 366
    LUTs at 0.75 of eight per slice are 61 slices.
    """
    unit = _unit("swg", luts=718)
    unit.demands["lutrams"] = 352
    unit.demands["ram_macros"] = 64
    assert unit.logic_luts() == 366
    assert unit.slices_needed(logic_util=1.0) == 80 + 46
    assert unit.slices_needed() == 80 + 61
    assert unit.needed_slices() == 141
    p = pack(_census(), [unit])["swg"]
    assert p.status == "placed"
    assert p.capacity["luts"] / SLICE_LUTS >= 141, p.capacity
    assert p.capacity["lutrams"] / 0.8 / SLICE_LUTS >= 80, p.capacity


@pytest.mark.fpgadataflow
def test_units_are_packed_around_each_other_on_the_skyline() -> None:
    """Later units land beside or above earlier ones, never overlapping them."""
    units = [_unit(f"u{i}", luts=200 + 50 * i, order=i) for i in range(12)]
    result = pack(_census(), units)
    placed = [p for p in result.values() if p.status == "placed"]
    assert len(placed) == 12
    for a in placed:
        for b in placed:
            if a is b:
                continue
            shared = set(a.columns) & set(b.columns)
            overlap_y = a.y0 <= b.y1 and b.y0 <= a.y1
            assert not (shared and overlap_y), f"{a.name} shares columns {shared} with {b.name}"


@pytest.mark.fpgadataflow
def test_feedback_round_repacks_only_the_rejected_unit() -> None:
    """A verify shortfall raises that unit's demand; the others keep their rectangles."""
    units = [
        _unit("a", luts=400, order=0),
        _unit("b", luts=400, order=1),
        _unit("c", luts=400, order=2),
    ]
    first = pack(_census(), units)
    second = pack(
        _census(),
        [
            _unit("a", luts=400, order=0),
            _unit("b", luts=400, order=1),
            _unit("c", luts=400, order=2),
        ],
        previous=first,
        feedback={"b": {"luts": 300}},
        round_no=2,
    )
    assert second["a"] == first["a"] and second["c"] == first["c"]
    assert second["b"].round == 2
    assert second["b"].capacity["luts"] >= 700
    # and it still does not overlap the pinned ones
    for other in ("a", "c"):
        o = second[other]
        b = second["b"]
        assert not (set(o.columns) & set(b.columns) and o.y0 <= b.y1 and b.y0 <= o.y1)


@pytest.mark.fpgadataflow
def test_impossible_unit_is_reported_not_raised() -> None:
    """A demand the SLR cannot hold anywhere comes back unplaced with the starved kind."""
    result = pack(_census(), [_unit("huge", luts=100, dsps=1000)])
    assert result["huge"].status == "unplaced"
    assert "dsps" in result["huge"].reason


@pytest.mark.fpgadataflow
def test_area_order_puts_the_big_units_first() -> None:
    """Footprint counts the die-ratio cost of sparse kinds, not just LUTs."""
    small = _unit("s", luts=5000)
    dsp = _unit("d", luts=900, dsps=317)
    assert footprint(dsp) > footprint(small)


@pytest.mark.fpgadataflow
@pytest.mark.parametrize(
    "fixture", ["mobilenetv1_attempt1", "resnet18_attempt1", "resnet18_measured"]
)
def test_real_fixture_places_every_unit_with_little_lut_waste(fixture: str) -> None:
    """The Milestone-5 designs on the measured U55C census: full coverage, low waste.

    The shelf engine constrained 258 of 289 resnet18 units at a median LUT
    over-claim of 5x. Here every unit must get a region, the median over-claim
    must stay below 1.6x, and no region may leave its SLR. `resnet18_measured`
    is the full post-opt measurement of the first new-engine link (2 Oct 2026):
    flip-flops, LUTRAM and shift registers included, with three 122k-LUTRAM
    weight memories among the units.
    """
    census = Census.from_dict(json.loads((FIXTURES / "census_u55c_device.json").read_text()))
    units = [
        Unit.from_dict(d) for d in json.loads((FIXTURES / f"units_{fixture}.json").read_text())
    ]
    # Coverage and waste only; the gap refinement is measured in
    # test_refinement_removes_the_long_jumps_on_a_real_fixture (slow tier).
    result = pack(census, units, refine_gaps=False)
    summary = summarize(result, units, census)
    # An SLR whose SLICEM demand exceeds 90 % of its slices (resnet18's
    # distributed-RAM weights) is the SLR assignment's problem, reported as such;
    # coverage is only asserted where the budget allows it.
    budget = slr_budget(census, units)
    if any(b["slicem"] > 0.9 for b in budget.values()):
        assert summary["oversubscribed"], budget
    else:
        assert summary["placed"] == len(units), summary["reasons"]
    by_name = {u.name: u for u in units}
    # Waste is claimed SLICEs over the SLICEs the unit's LUTs *and* flip-flops need;
    # a LUT-only ratio would call a unit of 18 LUTs and 700 FFs 30x oversized.
    ratios = sorted(
        (p.capacity["luts"] / SLICE_LUTS) / by_name[n].needed_slices()
        for n, p in result.items()
        if p.status == "placed"
    )
    # The smallest region is one column by five rows (one block RAM tile), i.e.
    # five SLICEs; a unit that needs three gets 1.67x. The measured fixture is
    # dominated by such units, so its median sits at that quantisation.
    assert ratios[len(ratios) // 2] <= (1.8 if fixture == "resnet18_measured" else 1.6)
    for name, p in result.items():
        if p.status != "placed":
            continue
        y0, y1 = census.slice_bounds(p.slr)
        assert y0 <= p.y0 <= p.y1 <= y1, f"{name} leaves SLR{p.slr}"
        assert by_name[name].slr == p.slr


@pytest.mark.fpgadataflow
def test_result_round_trips_through_json_and_tcl(tmp_path: Path) -> None:
    """The JSON result reloads as placements and the Tcl dict sources cleanly."""
    result = pack(_census(), [_unit("u", luts=40), _unit("v", luts=80, order=1)])
    out, tcl = tmp_path / "r.json", tmp_path / "r.tcl"
    write_result(out, tcl, result, summarize(result, []))
    again = load_previous(out)
    assert again["u"].ranges == result["u"].ranges
    text = tcl.read_text()
    assert text.startswith("set ::finn_fp_result [dict create]")
    assert "dict set ::finn_fp_result u {" in text and "status placed" in text


@pytest.mark.fpgadataflow
def test_packer_runs_as_a_script_without_importing_finn(tmp_path: Path) -> None:
    """The hook runs the packer by path in a scrubbed environment; it must not need FINN."""
    census = tmp_path / "c.json"
    units = tmp_path / "u.json"
    census.write_text(
        json.dumps(
            {
                "version": 1,
                "slrs": [0],
                "bands": {
                    "SLICE": [[0, 59]],
                    "RAMB36": [[0, 11]],
                    "RAMB18": [[0, 23]],
                    "DSP48E2": [[0, 23]],
                },
                "columns": {
                    "0": {
                        "SLICE": [{"x": 0, "tile_col": 0, "slicem": False, "runs": [[0, 59]]}],
                        "RAMB36": [{"x": 0, "tile_col": 1, "runs": [[0, 11]]}],
                        "RAMB18": [{"x": 0, "tile_col": 1, "runs": [[0, 23]]}],
                        "DSP48E2": [{"x": 0, "tile_col": 2, "runs": [[0, 23]]}],
                        "URAM288": [],
                    }
                },
            }
        )
    )
    unit = {"name": "u", "slr": 0, "order": 0, "demands": {"luts": 16}, "chains": {}}
    units.write_text(json.dumps([unit]))
    out, tcl = tmp_path / "r.json", tmp_path / "r.tcl"
    proc = subprocess.run(
        [
            sys.executable,
            str(floorplan_packer_path()),
            "--census",
            str(census),
            "--units",
            str(units),
            "--out",
            str(out),
            "--out-tcl",
            str(tcl),
        ],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 of 1 unit(s) placed" in proc.stdout
    assert tcl.is_file()


@pytest.mark.fpgadataflow
def test_oversubscribed_slr_is_reported_as_an_assignment_problem() -> None:
    """With the shell's sites excluded, resnet18's memory LUTs exceed SLR1's SLICEM ceiling.

    The census the hook exported from the real link (`census_u55c_ulp.json`) has
    84-91 % of the device's SLICEM rows; against it the measured resnet18 units
    put 96 %, 103 % and 95 % of each SLR's memory-LUT ceiling in demand. The
    packer must say so up front -- it is the SLR assignment that has to change --
    rather than report a hundred starved FIFOs as its own failure.
    """
    census = Census.from_dict(json.loads((FIXTURES / "census_u55c_ulp.json").read_text()))
    units = [
        Unit.from_dict(d)
        for d in json.loads((FIXTURES / "units_resnet18_measured.json").read_text())
    ]
    budget = slr_budget(census, units)
    assert budget[1]["slicem"] > 1.0, budget[1]
    assert all(b["slicem"] > 0.9 for b in budget.values())
    assert all(b["luts"] < b["slicem"] for b in budget.values()), "plain LUTs are not the problem"
    summary = summarize(pack(census, units, refine_gaps=False), units, census)
    assert set(summary["oversubscribed"]) == {"SLR0", "SLR1", "SLR2"}
    assert all("slicem" in v for v in summary["oversubscribed"].values())


@pytest.mark.fpgadataflow
def test_units_get_compact_rectangles_not_strips() -> None:
    """The half-perimeter objective keeps a plain logic unit near square.

    Ranking by waste alone took the first zero-waste shape at the lowest height, a
    strip the width of the SLR; on mobilenetv1 that gave a mean aspect ratio of 13:1
    and 575 of the 617 worst intra-unit paths. Here 3000 LUTs (375 SLICEs at full
    fill, 500 under the logic ceiling) must come out as a rectangle, not a strip.
    The census is wider than the default one: with twelve SLICE columns, of which
    plain logic may take only the SLICEL ones, 500 SLICEs cannot be anything but
    a few full-height columns.
    """
    unit = _unit("logic", luts=3000)
    unit.demands["ffs"] = 3000
    result = pack(_census(slice_cols=48, rows=240), [unit])
    p = result["logic"]
    assert p.status == "placed"
    width, height = p.c1 - p.c0 + 1, p.y1 - p.y0 + 1
    assert max(width, height) / min(width, height) <= 3, (width, height)


@pytest.mark.fpgadataflow
def test_dataflow_neighbours_are_packed_next_to_each_other() -> None:
    """In dataflow order every unit settles beside or on top of its predecessor."""
    units = [_unit(f"u{i}", luts=600, order=i) for i in range(8)]
    for u in units:
        u.demands["ffs"] = 600
    result = pack(_census(slice_cols=48, rows=240), units, order="dataflow")
    assert all(p.status == "placed" for p in result.values())
    for a, b in pairwise(units):
        pa, pb = result[a.name], result[b.name]
        dx = max(0, pb.c0 - pa.c1 - 1, pa.c0 - pb.c1 - 1)
        dy = max(0, pb.y0 - pa.y1 - 1, pa.y0 - pb.y1 - 1)
        assert dx + dy <= 3, f"{b.name} is {dx + dy} tiles from {a.name}"
    summary = summarize(result, units)
    assert summary["placed"] == 8 and summary["aspect_mean"] <= 3
    assert summary["half_perimeter_mean"] > 0 and summary["aspect_max"] >= 1


@pytest.mark.fpgadataflow
def test_a_unit_settles_between_its_placed_neighbours() -> None:
    """A unit packed after its consumer (a chain carrier goes first) lands next to it too."""
    units = [_unit(f"u{i}", luts=600, order=i) for i in range(6)]
    for u in units:
        u.demands["ffs"] = 600
    # u4 carries a chain, so dataflow order packs it first; u0..u3 follow in order and
    # u3 feeds it: u3 must end up next to u4, not only next to u2.
    units[4].chains = {"RAMB36": [2]}
    units[4].demands["brams"] = 2
    result = pack(_census(slice_cols=48, bram_cols=2, rows=240), units, order="dataflow")
    assert all(p.status == "placed" for p in result.values())

    def gap(a: str, b: str) -> int:
        pa, pb = result[a], result[b]
        dx = max(0, pb.c0 - pa.c1 - 1, pa.c0 - pb.c1 - 1)
        dy = max(0, pb.y0 - pa.y1 - 1, pa.y0 - pb.y1 - 1)
        return dx + dy

    # Every unit ends next to both the unit it receives from and the one it feeds,
    # the carrier included (greedy packing alone left u5 15 tiles from it; the
    # refinement closes that, 16 Oct 2026).
    for a, b in pairwise(["u0", "u1", "u2", "u3", "u4", "u5"]):
        assert gap(a, b) <= 3, f"{b} is {gap(a, b)} tiles from {a}"


@pytest.mark.fpgadataflow
def test_an_slr_crossing_unit_settles_at_the_boundary_near_its_partner() -> None:
    """The first unit of the next SLR anchors to the last unit of the previous one.

    The SLRs are packed one after the other; without the anchor the two sides of
    a crossing landed 240 rows apart (15 Oct 2026). The SLR1 unit must come to the
    rows next to the boundary and to the columns of its SLR0 partner.
    """
    census = Census.from_dict(json.loads((FIXTURES / "census_u55c_ulp.json").read_text()))
    a, b = _unit("a", luts=400, order=0), _unit("b", luts=400, order=1)
    a.demands["ffs"] = b.demands["ffs"] = 400
    b.slr = 1
    result = pack(census, [a, b], order="dataflow")
    pa, pb = result["a"], result["b"]
    assert pa.status == pb.status == "placed" and pa.slr == 0 and pb.slr == 1
    dx = max(0, pb.c0 - pa.c1 - 1, pa.c0 - pb.c1 - 1)
    assert dx <= 8, f"crossing units {dx} tile columns apart"
    slr1_lo = census.slice_bounds(1)[0]
    assert pb.y0 - slr1_lo <= 15, f"SLR1 unit starts {pb.y0 - slr1_lo} rows above the boundary"


@pytest.mark.fpgadataflow
def test_release_gives_back_what_claim_took() -> None:
    """Releasing a claimed span restores the free intervals exactly, merged."""
    packer = SlrPacker(_census(rows=120), 0)
    before = {t: list(iv) for t, iv in packer.free.items()}
    spans = {packer.tile_cols[0]: (10, 19), packer.tile_cols[3]: (0, 59)}
    packer._claim(spans)  # noqa: SLF001
    assert packer.free != before
    packer._release(spans)  # noqa: SLF001
    assert packer.free == before


@pytest.mark.fpgadataflow
def test_a_unit_starts_level_with_its_neighbour() -> None:
    """With a neighbour high up a free column, the unit starts beside it, not at row 0.

    Bottoms of free intervals alone put mobilenetv1's unit 82 at row 0, 185 rows
    below the predecessor it receives from (16 Oct 2026).
    """
    packer = SlrPacker(_census(slice_cols=24, rows=240), 0)
    anchor = Placement(name="a", slr=0, status="placed", c0=0, c1=1, y0=150, y1=159)
    packer._claim({0: (150, 159), 1: (150, 159)})  # noqa: SLF001
    unit = _unit("u", luts=200)
    p = packer.place(unit, anchors=[anchor])
    assert p.status == "placed"
    gap = SlrPacker._gap(p.c0, p.c1, p.y0, p.y1, anchor)  # noqa: SLF001
    assert gap <= 1, f"unit at rows {p.y0}-{p.y1}, {gap} tiles from its neighbour"


@pytest.mark.fpgadataflow
@pytest.mark.slow
def test_refinement_removes_the_long_jumps_on_a_real_fixture() -> None:
    """On mobilenetv1 the refinement leaves no consecutive-unit gap of 100+ tiles.

    The greedy packing alone leaves jumps of up to 236 tiles on this fixture (12 of
    100 or more), where a unit found no room near its predecessor (16 Oct 2026).
    Every unit must stay placed.
    """
    census = Census.from_dict(json.loads((FIXTURES / "census_u55c_ulp.json").read_text()))
    units = [
        Unit.from_dict(d)
        for d in json.loads((FIXTURES / "units_mobilenetv1_attempt1.json").read_text())
    ]
    result = pack(census, units)
    assert all(p.status == "placed" for p in result.values())
    by_order = sorted(units, key=lambda u: u.order)
    gaps = [
        SlrPacker._gap(pb.c0, pb.c1, pb.y0, pb.y1, pa)  # noqa: SLF001
        for pa, pb in ((result[a.name], result[b.name]) for a, b in pairwise(by_order))
        if pa.slr == pb.slr
    ]
    assert max(gaps) < 100, sorted(gaps)[-5:]

"""Tests for the shell closure map: probe planning, log parsing, platform lookup, cache.

Everything here is `quick`-tier. The Vivado side (the probe itself) is exercised by
`finn floorplan map-shell`; the log excerpts below are copied from real probe runs
on the U55C shell (3 Oct 2026).
"""

import pytest

import json
import zipfile
from collections.abc import Callable
from pathlib import Path

from finn.transformation.fpgadataflow.floorplanning.shell_closure import (
    ShellClosureMap,
    Site,
    batch_requests,
    carry_requests,
    carry_segment_request,
    close_partial_columns,
    column_requests,
    count_chain_failures,
    extract_locked_checkpoint,
    find_platform_xsa,
    load_map,
    map_file_name,
    merge_maps,
    parse_enumeration,
    parse_rejections,
    parse_shape_failures,
    save_map,
    site_requests,
    vivado_release,
)
from finn.transformation.fpgadataflow.floorplanning.tcl import (
    TCL_DEMAND_FIELDS,
    FloorplanUnitSpec,
    render_floorplan_tcl,
)

#: A block RAM probe pblock the closure emptied, as the placer reports it.
_BRAM_CLOSED = """ERROR: [Place 30-859] The number of BRAM cells that have to be placed in a \
constraint region exceeds the number of corresponding sites available.

Area constraint: PBlock
PBlock name is: cs_RAMB36_X13Y8
Number of BRAM required by this constraint: 1
Number of BRAM Fixed inside this constraint: 0
Number of BRAM available in this constraint region: 0
Total number of BRAM blocked sites in this constraint region: 3
Number of BRAM sites in this constraint region blocked with place.closeUnroutableSites param: 3

Resolution: Check any pblock constraints that could prevent placement of cells in the \
constraint area. Also consider reducing number of cascaded macros to reduce the overall \
required sites.
"""

#: A DSP probe pblock rejected only because the placer counts a DSP48E2 as eight
#: sub-cells against one site; no closure line, so no site was closed.
_DSP_NOT_CLOSED = """ERROR: [Place 30-859] The number of DSP cells that have to be placed in a \
constraint region exceeds the number of corresponding sites available.

Area constraint: PBlock
PBlock name is: cs_DSP48E2_S2_X31
Number of DSP required by this constraint: 192
Number of DSP Fixed inside this constraint: 0
Number of DSP available in this constraint region: 24

Resolution: Check any pblock constraints that could prevent placement of cells in the \
constraint area.
"""

_DSP_CLOSED = """ERROR: [Place 30-859] The number of DSP cells that have to be placed in a \
constraint region exceeds the number of corresponding sites available.

Area constraint: PBlock
PBlock name is: cs_DSP48E2_S0_X31
Number of DSP required by this constraint: 144
Number of DSP Fixed inside this constraint: 0
Number of DSP available in this constraint region: 2
Total number of DSP blocked sites in this constraint region: 16
Number of DSP sites in this constraint region blocked with place.closeUnroutableSites param: 15

Resolution: Check any pblock constraints that could prevent placement of cells in the \
constraint area.
"""


@pytest.mark.fpgadataflow
def test_rejections_count_only_sites_the_closure_took() -> None:
    """A DSP pblock rejected for sub-cell accounting alone must not read as closed."""
    log = "Phase 2.5 Global Place Phase1\n" + _DSP_NOT_CLOSED + _BRAM_CLOSED + _DSP_CLOSED
    assert parse_rejections(log) == {"cs_RAMB36_X13Y8": 3, "cs_DSP48E2_S0_X31": 15}


@pytest.mark.fpgadataflow
def test_a_rejection_without_a_closure_line_does_not_borrow_the_next_one() -> None:
    """The closure count belongs to the pblock named in the same message."""
    assert parse_rejections(_DSP_NOT_CLOSED + _BRAM_CLOSED) == {"cs_RAMB36_X13Y8": 3}
    assert parse_rejections(_DSP_NOT_CLOSED) == {}


@pytest.mark.fpgadataflow
def test_enumeration_output_is_parsed() -> None:
    """INFO lines become metadata, SITE lines become sites."""
    text = (
        "INFO part xcu55c-fsvh2892-2L-e\nINFO vivado 2024.2.2\n"
        "SITE RAMB36 RAMB36_X12Y0 0 12 0\nSITE DSP48E2 DSP48E2_X29Y3 0 29 3\ngarbage\n"
    )
    info, sites = parse_enumeration(text)
    assert info == {"part": "xcu55c-fsvh2892-2L-e", "vivado": "2024.2.2"}
    assert sites == [
        Site("RAMB36", "RAMB36_X12Y0", 0, 12, 0),
        Site("DSP48E2", "DSP48E2_X29Y3", 0, 29, 3),
    ]


@pytest.mark.fpgadataflow
def test_columns_are_split_by_slr_and_batched_by_cell_count() -> None:
    """A column that spans two SLRs is two pblocks; batches respect the cell budget."""
    sites = [Site("RAMB36", f"RAMB36_X12Y{y}", 0 if y < 12 else 1, 12, y) for y in range(24)]
    sites += [Site("RAMB36", f"RAMB36_X13Y{y}", 0, 13, y) for y in range(12)]
    reqs = column_requests(sites)
    assert [(k, n, len(s)) for k, n, s in reqs] == [
        ("RAMB36", "cs_RAMB36_S0_X12", 12),
        ("RAMB36", "cs_RAMB36_S0_X13", 12),
        ("RAMB36", "cs_RAMB36_S1_X12", 12),
    ]
    assert reqs[0][2][:3] == ["RAMB36_X12Y0", "RAMB36_X12Y1", "RAMB36_X12Y2"]
    assert [len(b) for b in batch_requests(reqs, 24)] == [2, 1]
    slices = column_requests([Site("SLICE", f"SLICE_X5Y{y}", 0, 5, y) for y in range(4)])
    assert [len(b) for b in batch_requests(slices * 3, 64, "columns")] == [2, 1], "8 cells/site"
    assert [len(b) for b in batch_requests(reqs, 5)] == [1, 1, 1], "a column is never split"
    singles = site_requests(sites[:2])
    assert singles == [
        ("RAMB36", "cs_RAMB36_X12Y0", ["RAMB36_X12Y0"]),
        ("RAMB36", "cs_RAMB36_X12Y1", ["RAMB36_X12Y1"]),
    ]


def _fake_platform(root: Path, name: str) -> Path:
    """Lay out a platform directory like an installed Vitis platform."""
    plat = root / name
    (plat / "hw").mkdir(parents=True)
    (plat / f"{name}.xpfm").write_text("<platform/>")
    with zipfile.ZipFile(plat / "hw" / "hw.xsa", "w") as z:
        z.writestr("hw_bb_locked.dcp", b"checkpoint-bytes")
        z.writestr("tcl_hooks/impl.xdc", "# region")
    return plat


@pytest.mark.fpgadataflow
def test_the_locked_checkpoint_is_found_and_extracted_from_the_platform(tmp_path: Path) -> None:
    """PLATFORM_REPO_PATHS may name the platform directory or its parent."""
    name = "xilinx_u55c_gen3x16_xdma_3_202210_1"
    plat = _fake_platform(tmp_path / "platforms", name)
    assert find_platform_xsa(name, [tmp_path / "platforms"]) == plat / "hw" / "hw.xsa"
    assert find_platform_xsa(name, [plat]) == plat / "hw" / "hw.xsa"
    with pytest.raises(FileNotFoundError):
        find_platform_xsa("xilinx_u50_gen3x16_xdma_5_202210_1", [tmp_path / "platforms"])
    dcp = extract_locked_checkpoint(plat / "hw" / "hw.xsa", tmp_path / "work")
    assert dcp.name == "hw_bb_locked.dcp" and dcp.read_bytes() == b"checkpoint-bytes"


@pytest.mark.fpgadataflow
def test_a_platform_without_a_shell_has_no_checkpoint(tmp_path: Path) -> None:
    """A Zynq-style archive with no locked checkpoint is reported, not guessed around."""
    xsa = tmp_path / "flat.xsa"
    with zipfile.ZipFile(xsa, "w") as z:
        z.writestr("design_1_wrapper.bit", b"")
    with pytest.raises(FileNotFoundError, match="not a shell-based platform"):
        extract_locked_checkpoint(xsa, tmp_path / "work")


def _map(vivado: str, closed: list[str]) -> ShellClosureMap:
    return ShellClosureMap(
        platform="xilinx_u55c_gen3x16_xdma_3_202210_1",
        part="xcu55c-fsvh2892-2L-e",
        vivado=vivado,
        kinds=["RAMB36", "DSP48E2"],
        closed={"RAMB36": closed, "DSP48E2": ["DSP48E2_X29Y0"]},
        probed_sites={"RAMB36": 1776, "DSP48E2": 8376},
        generated="2026-10-03",
    )


@pytest.mark.fpgadataflow
def test_maps_round_trip_and_are_chosen_by_vivado_release(tmp_path: Path) -> None:
    """A map is keyed by release; the patch level does not split the cache."""
    assert vivado_release("2024.2.2") == "2024.2"
    old = save_map(_map("2023.2", ["RAMB36_X12Y0"]), tmp_path)
    new = save_map(_map("2024.2.2", ["RAMB36_X12Y0", "RAMB36_X13Y0"]), tmp_path)
    assert new.name == map_file_name("xilinx_u55c_gen3x16_xdma_3_202210_1", "2024.2")
    assert json.loads(old.read_text())["vivado"] == "2023.2"

    m, note = load_map("xilinx_u55c_gen3x16_xdma_3_202210_1", "2024.2", [tmp_path])
    assert m is not None and m.vivado == "2024.2.2" and str(new) == note
    assert m.all_closed == ["RAMB36_X12Y0", "RAMB36_X13Y0", "DSP48E2_X29Y0"]

    m, note = load_map("xilinx_u55c_gen3x16_xdma_3_202210_1", "2025.1", [tmp_path])
    assert m is not None and m.vivado == "2024.2.2", "the newest release stands in"
    assert "measured under Vivado 2024.2.2" in note and "running 2025.1" in note

    m, note = load_map("xilinx_u280_gen3x16_xdma_1_202211_1", "2024.2", [tmp_path])
    assert m is None and "no shell closure map" in note


@pytest.mark.fpgadataflow
def test_the_shipped_u55c_map_is_found() -> None:
    """FINN ships the map for the U55C shell it is validated on."""
    m, _ = load_map("xilinx_u55c_gen3x16_xdma_3_202210_1", "2024.2")
    assert m is not None
    assert m.kinds == ["RAMB36", "DSP48E2", "URAM288", "SLICE", "CARRY8"]
    # Measured 3 Oct 2026: the block RAM closure is confined to SLR0's two edge
    # columns, 13 of their 24 sites; DSP closes 233 sites in 7 columns, listed at
    # tile resolution (282); URAM and SLICE none (all 143,280 slices probed).
    # The column rule (4 Oct 2026) closes the rest of every partly closed block RAM
    # or DSP column: +11 RAMB36 (SLR0 X12/X13 whole), +150 DSP48E2.
    assert len(m.closed["RAMB36"]) == 13 + 11
    assert len(m.closed["DSP48E2"]) == 282 + 150
    assert m.column_rule == {"RAMB36": 11, "DSP48E2": 150}
    assert m.closed["URAM288"] == []
    assert m.closed["SLICE"] == [] and m.probed_sites["SLICE"] == 143280
    edge = sorted(s for s in m.closed["RAMB36"] if s.startswith(("RAMB36_X12", "RAMB36_X13")))
    assert edge == sorted(f"RAMB36_X{x}Y{y}" for x in (12, 13) for y in range(12))
    # Carry chains (4 Oct 2026): 3,580 slices next to the shell take a LUT but no
    # carry chain, among them SLICE_X195 rows 360-404, where the carry macros of two
    # mobilenetv1 input generators failed to commit.
    carry = set(m.closed["CARRY8"])
    assert len(carry) == 3580
    assert {f"SLICE_X195Y{y}" for y in range(360, 405)} <= carry
    assert "SLICE_X100Y360" not in carry


@pytest.mark.fpgadataflow
def test_the_hook_receives_the_closed_sites() -> None:
    """The rendered hook carries the list, its Vivado release and its source."""
    unit = FloorplanUnitSpec(
        name="pblock_finn_unit_0",
        cells=["MVAU_0"],
        slr=0,
        demands=dict.fromkeys(TCL_DEMAND_FIELDS, 0) | {"luts": 10, "ffs": 10},
    )
    lines = render_floorplan_tcl(
        [unit],
        closed_sites=["RAMB36_X12Y0", "DSP48E2_X29Y0"],
        closed_sites_vivado="2024.2.2",
        closed_sites_source="/maps/u55c.json",
    )
    text = "\n".join(lines)
    assert "set FINN_FP_CLOSED_SITES [list RAMB36_X12Y0 DSP48E2_X29Y0]" in text
    assert "set FINN_FP_CLOSED_SITES_VIVADO {2024.2.2}" in text
    assert "set FINN_FP_CLOSED_SITES_SOURCE {/maps/u55c.json}" in text
    # The prohibition must precede the census inside the engine; the variables
    # must be set before the engine is sourced.
    assert text.index("FINN_FP_CLOSED_SITES ") < text.index("source {")
    empty = "\n".join(render_floorplan_tcl([unit]))
    assert "set FINN_FP_CLOSED_SITES [list ]" in empty


@pytest.mark.fpgadataflow
def test_a_run_over_some_kinds_merges_into_the_existing_map(tmp_path: Path) -> None:
    """Mapping SLICE later must not drop the block RAM and DSP results."""
    first = _map("2024.2.2", ["RAMB36_X12Y0"])
    save_map(first, tmp_path)
    later = ShellClosureMap(
        platform=first.platform,
        part=first.part,
        vivado="2024.2.2",
        kinds=["SLICE", "DSP48E2"],
        closed={"SLICE": ["SLICE_X220Y0"], "DSP48E2": []},
        probed_sites={"SLICE": 143280, "DSP48E2": 8376},
    )
    path = save_map(later, tmp_path)
    m, _ = load_map(first.platform, "2024.2", [tmp_path])
    assert m is not None and m.kinds == ["RAMB36", "DSP48E2", "SLICE"]
    assert m.closed == {"RAMB36": ["RAMB36_X12Y0"], "DSP48E2": [], "SLICE": ["SLICE_X220Y0"]}
    assert m.probed_sites == {"RAMB36": 1776, "DSP48E2": 8376, "SLICE": 143280}
    save_map(later, tmp_path, merge=False)
    assert ShellClosureMap.from_dict(json.loads(path.read_text())).kinds == ["SLICE", "DSP48E2"]
    with pytest.raises(ValueError):
        merge_maps(first, _map("2025.1", []))


def _fake_vivado(
    closed: set[str], unusable: set[str], calls: list[int]
) -> Callable[[object, str, list[str]], str]:
    """Stand-in for `_Runner._run`: enumerates a tiny region and answers probes.

    A pblock holding a site in ``closed`` is rejected with a closure line; one
    holding a site in ``unusable`` makes place_design fail without any.
    """

    def run(_self: object, _tag: str, args: list[str]) -> str:
        if args[0] == "enumerate":
            out = Path(args[2])
            out.parent.mkdir(parents=True, exist_ok=True)
            lines = ["INFO part xcu55c-fsvh2892-2L-e", "INFO vivado 2024.2.2"]
            lines += [f"SITE DSP48E2 DSP48E2_X{x}Y{y} 0 {x} {y}" for x in (1, 2) for y in range(4)]
            lines += [f"SITE SLICE SLICE_X{x}Y{y} 0 {x} {y}" for x in (5, 6) for y in range(8)]
            out.write_text("\n".join(lines) + "\n")
            return ""
        calls.append(1)
        text, failed = [], False
        for line in Path(args[2]).read_text().splitlines():
            _kind, name, _cells, *sites = line.split()
            if closed & set(sites):
                text.append(
                    "ERROR: [Place 30-859] The number of DSP cells ...\n"
                    f"PBlock name is: {name}\nNumber of DSP required by this constraint: 8\n"
                    "Number of DSP sites in this constraint region blocked with "
                    f"place.closeUnroutableSites param: {len(closed & set(sites))}\n"
                )
                failed = True
            if unusable & set(sites):
                failed = True
        text.append(f"FINN_CS place_rc {int(failed)}")
        return "\n".join(text)

    return run


@pytest.mark.fpgadataflow
def test_mapping_flags_columns_then_bisects_unexplained_slice_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pass 1 flags the columns, pass 2 names the sites.

    The DSP column is flagged by its closure line. The SLICE column holding a
    site that fails placement without one is isolated by bisection, sent to the
    site pass, and its site isolated again: recorded, never silently dropped.
    """
    from finn.transformation.fpgadataflow.floorplanning import shell_closure as sc

    calls: list[int] = []
    fake = _fake_vivado({"DSP48E2_X2Y3"}, {"SLICE_X6Y5"}, calls)
    monkeypatch.setattr(sc._Runner, "_run", fake)  # noqa: SLF001 - only Vivado is replaced
    dcp = tmp_path / "hw_bb_locked.dcp"
    dcp.write_bytes(b"")
    m = sc.map_shell_closure(
        "xilinx_u55c_gen3x16_xdma_3_202210_1",
        checkpoint=dcp,
        kinds=("DSP48E2", "SLICE"),
        work_dir=tmp_path / "work",
        max_cells={"SLICE": 16},
        log=lambda _msg: None,
    )
    # The DSP column lost one site to the closure, so the column rule closes the
    # rest of it; SLICE is not subject to the rule.
    assert sorted(m.closed["DSP48E2"]) == [f"DSP48E2_X2Y{y}" for y in range(4)]
    assert m.closed["SLICE"] == ["SLICE_X6Y5"]
    assert m.column_rule == {"DSP48E2": 3}
    assert m.kinds == ["DSP48E2", "SLICE"]
    assert m.probed_sites == {"DSP48E2": 8, "SLICE": 16}
    # Columns: 1 DSP run; the two SLICE columns (64 cells each) one run apiece.
    # Sites: 1 DSP run; the failing SLICE column's 8 sites, halved down to one.
    assert len(calls) == 1 + 2 + 1 + (1 + 2 * 3)


@pytest.mark.fpgadataflow
def test_carry_windows_tile_each_column_run_without_crossing_a_gap() -> None:
    """Two-slice windows per column run; a leftover slice joins the last window."""
    sites = [Site("CARRY8", f"SLICE_X195Y{y}", 1, 195, y) for y in (300, 301, 302, 303, 304)]
    sites += [Site("CARRY8", f"SLICE_X195Y{y}", 1, 195, y) for y in (360, 361)]
    sites += [Site("CARRY8", "SLICE_X194Y400", 1, 194, 400)]
    got = [(n, s) for _, n, s in carry_requests(sites)]
    assert got == [
        ("cs_CARRY8_SLICE_X194Y400", ["SLICE_X194Y400"]),
        ("cs_CARRY8_SLICE_X195Y300", ["SLICE_X195Y300", "SLICE_X195Y301"]),
        ("cs_CARRY8_SLICE_X195Y302", ["SLICE_X195Y302", "SLICE_X195Y303", "SLICE_X195Y304"]),
        ("cs_CARRY8_SLICE_X195Y360", ["SLICE_X195Y360", "SLICE_X195Y361"]),
    ]


#: Excerpts of carry probe runs in the U55C shell (3 Oct 2026): an abort while the
#: placer initialises, and a commit failure in detail placement.
_CARRY_ABORT = """Phase 1.2 IO Placement/ Clock Placement/ Build Placer Device
ERROR: [Place 30-642] Placement Validity Check : Failed to find legal placement.
Reason: Could not place shape in pblock cs_CARRY8_SLICE_X194Y300.
The unplaced cells are:
  Cell level0_i/ulp/finn_cs_cs_CARRY8_SLICE_X194Y300_0 of type CARRY8
"""
_CARRY_COMMIT = """ERROR: [Place 30-1153] Failed to commit 2 macros:
level0_i/ulp/finn_cs_cs_CARRY8_SLICE_X195Y360_0 with block Id: 253343
This block contains 2 instances.
Instance no. 0: finn_cs_cs_CARRY8_SLICE_X195Y360_0
Instance no. 1: finn_cs_cs_CARRY8_SLICE_X195Y360_1
This block has a USER PBLOCK associated with it.

level0_i/ulp/finn_cs_cs_CARRY8_SLICE_X195Y364_0 with block Id: 253342
This block contains 2 instances.

ERROR: [Place 30-99] Placer failed with error: 'Could not place all shapes'
"""


@pytest.mark.fpgadataflow
def test_carry_failures_are_read_from_aborts_and_commit_failures() -> None:
    """An abort names one window and leaves the rest unknown; a commit failure names all."""
    assert parse_shape_failures(_CARRY_ABORT) == ({"cs_CARRY8_SLICE_X194Y300"}, set())
    assert parse_shape_failures(_CARRY_COMMIT) == (
        set(),
        {"cs_CARRY8_SLICE_X195Y360", "cs_CARRY8_SLICE_X195Y364"},
    )


@pytest.mark.fpgadataflow
def test_failed_carry_chains_are_counted_per_probe_pblock() -> None:
    """Cells are numbered per pblock; two cells of one chain count once."""
    log = """ERROR: [Place 30-1153] Failed to commit 2 macros:
level0_i/ulp/finn_cs_cs_CARRY8_SLICE_X195Y300_60_0 with block Id: 1
level0_i/ulp/finn_cs_cs_CARRY8_SLICE_X195Y300_60_6 with block Id: 2
ERROR: [Place 30-488] Failed to commit 2 instances:
level0_i/ulp/finn_cs_cs_CARRY8_SLICE_X194Y300_4_2 with block Id: 3 (CARRY) . No site found.
level0_i/ulp/finn_cs_cs_CARRY8_SLICE_X194Y300_4_3 with block Id: 4 (CARRY) . No site found.
"""
    assert count_chain_failures(log) == {
        "cs_CARRY8_SLICE_X195Y300_60": 2,
        "cs_CARRY8_SLICE_X194Y300_4": 1,
    }
    windows = [["SLICE_X1Y0", "SLICE_X1Y1"], ["SLICE_X1Y2", "SLICE_X1Y3"]]
    assert carry_segment_request(windows) == (
        "CARRY8",
        "cs_CARRY8_SLICE_X1Y0_2",
        ["SLICE_X1Y0", "SLICE_X1Y1", "SLICE_X1Y2", "SLICE_X1Y3"],
    )


@pytest.mark.fpgadataflow
def test_a_partly_closed_column_is_closed_as_a_whole_within_its_slr() -> None:
    """The open sites of a column the placer closes in part did not route (Route 35-7)."""
    sites = [Site("RAMB36", f"RAMB36_X12Y{y}", 0, 12, y) for y in range(12)]
    sites += [Site("RAMB36", f"RAMB36_X12Y{y}", 1, 12, y) for y in range(12, 24)]
    sites += [Site("RAMB36", f"RAMB36_X11Y{y}", 0, 11, y) for y in range(12)]
    sites += [Site("DSP48E2", "DSP48E2_X1Y0", 0, 1, 0)]
    closed = {"RAMB36": ["RAMB36_X12Y0", "RAMB36_X12Y2"], "DSP48E2": []}
    added = close_partial_columns(closed, sites)
    assert added == {"RAMB36": 10}
    assert sorted(closed["RAMB36"], key=lambda n: int(n.split("Y")[1])) == [
        f"RAMB36_X12Y{y}" for y in range(12)
    ], "SLR1's part of the column and the clean X11 stay open"
    assert closed["DSP48E2"] == []

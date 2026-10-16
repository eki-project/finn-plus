"""Intra/inter-unit timing split of a routed floorplanned design.

A hard pblock per layer changes two things about where a design's cells end up:
the cells of one unit are confined to a rectangle whose shape the packer chose
(intra-unit locality), and consecutive units are wherever the packer put them,
not where the placer would have wanted them (inter-unit locality). Both show up
as wire delay. This module tells the two apart on a routed checkpoint, so the
next floorplanner change can address whichever carries the loss.

The Vivado side (`finn_floorplan_timing.tcl`) exports the worst setup paths of the
kernel clock domain with their endpoint placement and the placed bounding box
of every unit's cells, in tile columns and rows (uniform across the device).
This module maps each path's endpoints to their units through the floorplan
artefacts (`finn_floorplan_units.json` for the unit cells, `finn_floorplan.json`
for the pblock rectangles), classifies the paths as intra-unit, inter-unit or
other, and summarises slack and route delay per class, per bounding-box aspect
ratio (intra) and per pblock adjacency and distance (inter). Run on the
unfloorplanned baseline with the same unit definitions, it gives the comparison:
the same netlist, the placer's own shapes and distances.

Command line (also `finn floorplan timing-split`)::

    timing_split.py --checkpoint routed.dcp --floorplan-dir <link dir> --out <dir>
                    [--baseline baseline_routed.dcp] [--paths 2000]
                    [--clock 'clk_kernel_00*']
"""

import argparse
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

#: Matches one SLICE range of a Vivado GRID_RANGES string.
SLICE_RANGE_PATTERN: re.Pattern[str] = re.compile(r"SLICE_X(\d+)Y(\d+):SLICE_X(\d+)Y(\d+)")

#: Upper bounds of the aspect-ratio buckets for the intra-unit summary.
ASPECT_BUCKETS: tuple[float, ...] = (2.0, 4.0, 8.0, math.inf)

#: Upper bounds (tile grid units) of the endpoint-distance buckets.
DISTANCE_BUCKETS: tuple[float, ...] = (20.0, 50.0, 100.0, 200.0, math.inf)

#: Default clock pattern: the Vitis kernel clock of an Alveo platform.
DEFAULT_CLOCK: str = "clk_kernel_00*"

#: How many of the worst unit pairs per class the summary lists by name.
TOP_N: int = 10

#: Placed boxes this many tiles apart count as adjacent: a routing (INT) column
#: or a BRAM/DSP column sits between two neighbouring CLB columns.
ADJACENT_GAP: int = 3


def timing_tcl_path() -> Path:
    """Path of the Vivado-side export script."""
    return (
        Path(__file__).resolve().parents[3]
        / "util"
        / "vivado_scripts"
        / "finn_floorplan_timing.tcl"
    )


# -- geometry ----------------------------------------------------------------------


@dataclass
class Box:
    """An inclusive rectangle in some grid."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        """Columns covered."""
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        """Rows covered."""
        return self.y1 - self.y0 + 1

    @property
    def aspect(self) -> float:
        """Longer side over shorter side (>= 1)."""
        a, b = max(self.width, self.height), min(self.width, self.height)
        return a / b if b else math.inf

    @property
    def half_perimeter(self) -> int:
        """Width plus height: the longest Manhattan span inside the box."""
        return self.width + self.height

    def center(self) -> tuple[float, float]:
        """Centre point."""
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def touches(self, other: "Box", tolerance: int = ADJACENT_GAP) -> bool:
        """Return True when the boxes overlap or lie within ``tolerance`` tiles of each other."""
        return self.gap(other) <= tolerance

    def gap(self, other: "Box") -> int:
        """Manhattan gap between the boxes' edges (0 when they touch or overlap)."""
        dx = max(0, other.x0 - self.x1 - 1, self.x0 - other.x1 - 1)
        dy = max(0, other.y0 - self.y1 - 1, self.y0 - other.y1 - 1)
        return dx + dy


def slice_box(grid_ranges: str) -> Box | None:
    """Bounding box of the SLICE ranges in a GRID_RANGES string, or None."""
    boxes = [tuple(map(int, m)) for m in SLICE_RANGE_PATTERN.findall(grid_ranges)]
    if not boxes:
        return None
    return Box(
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


# -- units -------------------------------------------------------------------------


@dataclass
class Unit:
    """One floorplan unit: its cells, its pblock and where its cells were placed."""

    name: str
    slr: int
    cells: list[str]
    roots: list[str] = field(default_factory=list)
    pblock: Box | None = None
    placed: Box | None = None
    n_cells: int = 0

    @property
    def op(self) -> str:
        """The FINN op type of the unit's first cell, e.g. ``MVAU_rtl``."""
        return op_of(self.cells[0]) if self.cells else ""


def op_of(cell: str) -> str:
    """Op type of a FINN node or cell name: ``..._MVAU_rtl_3`` -> ``MVAU_rtl``."""
    leaf = cell.split("/")[-1] if "/" in cell else cell
    m = re.search(r"StreamingDataflowPartition_\d+_(.*?)_\d+$", leaf)
    if m:
        return m.group(1)
    m = re.match(r"(.*?)_\d+$", leaf)
    return m.group(1) if m else leaf


def load_units(floorplan_dir: Path) -> dict[str, Unit]:
    """Read the units (cells, SLR) and their pblocks from a link or artefact directory."""
    units_file = floorplan_dir / "finn_floorplan_units.json"
    report_file = floorplan_dir / "finn_floorplan.json"
    if not units_file.exists():
        raise FileNotFoundError(
            f"{units_file} is missing; it is written by the hook's first attempt"
        )
    raw = json.loads(units_file.read_text())
    entries = raw["units"] if isinstance(raw, dict) else raw
    units = {e["name"]: Unit(e["name"], int(e.get("slr", -1)), list(e["cells"])) for e in entries}
    if report_file.exists():
        report = json.loads(report_file.read_text())
        for name, info in report.get("units", {}).items():
            if name in units and info.get("grid_ranges"):
                units[name].pblock = slice_box(info["grid_ranges"])
    return units


def partition_cells(parent_model: Path) -> list[tuple[str, str]]:
    """List every dataflow node of a stitched build as ``(cell name, op type)`` in graph order.

    The parent model (any build checkpoint from ``step_create_stitched_ip`` on, e.g.
    ``step_deployment_package.onnx``) holds one StreamingDataflowPartition per kernel
    whose ``model`` attribute points at the child model. Partitions and their nodes
    are in topological order, so the list is the dataflow order of the network.
    Node names and tensor names are both renumbered per partition, so neither
    identifies a node across two builds that partitioned the same network
    differently; the position in this order does (see `cell_map`).
    """
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.custom_op.registry import getCustomOp

    parent = ModelWrapper(str(parent_model))
    out: list[tuple[str, str]] = []
    for sdp in parent.graph.node:
        if sdp.op_type != "StreamingDataflowPartition":
            continue
        child_path = getCustomOp(sdp).get_nodeattr("model")
        if not Path(child_path).exists():
            raise FileNotFoundError(f"{sdp.name}: child model {child_path} is missing")
        out += [
            (instance_name(sdp.name, node.name), node.op_type)
            for node in ModelWrapper(child_path).graph.node
        ]
    return out


def instance_name(partition: str, node: str) -> str:
    """Cell name of a partition's node in the stitched kernel: ``<partition>_<node>``.

    The child models of one build carry the prefix already while those of another
    (a build restarted from a checkpoint, 14 Oct 2026) do not; the kernel's
    instances always have it, and so do the floorplan units.
    """
    return node if node.startswith(partition + "_") else f"{partition}_{node}"


def unit_cells(units: dict[str, Unit]) -> list[tuple[str, str]]:
    """List the floorplanned design's node cells as ``(cell name, op type)`` in dataflow order.

    The hook writes the units in graph order with the cell names of the stitched
    kernels, so they are the design's own dataflow order (`partition_cells` of the
    baseline is the other side). The parent model of a floorplanned build is not
    used for this: a build restarted from a checkpoint kept the global node
    numbering in its child models while the kernels number per partition (14 Oct
    2026: 158 of 287 units matched).
    """
    return [(cell, op_of(cell)) for u in units.values() for cell in u.cells]


def cell_map(units: dict[str, Unit], other_model: Path) -> dict[str, str]:
    """Map the cell names of the floorplanned units to the same nodes in another build.

    The two dataflow orders (`unit_cells` and `partition_cells`) are aligned on
    their op-type sequences; a node without a counterpart (e.g. a FIFO the
    floorplanner inserted at an SLR crossing) is left out, so its unit has no cells
    in the other build.
    """
    return align_cells(unit_cells(units), partition_cells(other_model))


def align_cells(here: list[tuple[str, str]], there: list[tuple[str, str]]) -> dict[str, str]:
    """Align two ``(cell, op type)`` sequences on their op types; unmatched cells are left out."""
    matcher = SequenceMatcher(None, [op for _, op in here], [op for _, op in there], autojunk=False)
    mapping: dict[str, str] = {}
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            mapping[here[block.a + k][0]] = there[block.b + k][0]
    return mapping


def remap_units(units: dict[str, Unit], mapping: dict[str, str]) -> dict[str, Unit]:
    """Return the units with their cells renamed through ``mapping``, unmapped cells dropped."""
    out = {}
    for name, u in units.items():
        cells = [mapping[c] for c in u.cells if c in mapping]
        out[name] = Unit(u.name, u.slr, cells, pblock=u.pblock)
    return out


def write_roots_file(units: dict[str, Unit], path: Path) -> None:
    """Write the ``<unit> <root cell>`` list the Tcl export resolves and measures."""
    lines = [f"{u.name} {cell}" for u in units.values() for cell in u.cells]
    path.write_text("\n".join(lines) + "\n")


# -- the export ---------------------------------------------------------------------


def run_export(
    checkpoint: Path,
    roots_file: Path,
    out: Path,
    max_paths: int,
    clock: str,
    vivado: str = "vivado",
    exception_sources: str = "",
) -> Path:
    """Run the Vivado-side export on one checkpoint; returns the JSON it wrote.

    ``exception_sources`` is a ``get_cells -regexp`` pattern for the registers whose
    paths carry a tighter requirement than the clock (the reset tree); they are
    exported as their own group so they cannot crowd the clock-period paths out.
    """
    run_dir = out.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        vivado, "-mode", "batch", "-nojournal", "-log", f"{out.stem}.vivado.log",
        "-source", str(timing_tcl_path()),
        "-tclargs", str(checkpoint), str(out), str(max_paths), clock, str(roots_file),
        exception_sources,
    ]  # fmt: skip
    with (run_dir / f"{out.stem}.stdout.txt").open("w") as fh:
        proc = subprocess.run(cmd, cwd=run_dir, stdout=fh, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"timing export of {checkpoint} failed (exit {proc.returncode}); "
            f"see {run_dir / f'{out.stem}.vivado.log'}"
        )
    return out


# -- classification -----------------------------------------------------------------


@dataclass
class PathRecord:
    """One exported path with its classification."""

    slack: float
    datapath: float
    logic: float | None
    route: float | None
    levels: int
    start_cell: str
    end_cell: str
    start_unit: str | None
    end_unit: str | None
    distance: float | None
    kind: str
    #: Units the path passes through besides its endpoints' (a ready path that leaves
    #: a unit and returns through its consumer is inter-unit although both ends sit in
    #: the one unit; `end_unit` then names the unit it went through).
    via: tuple[str, ...] = ()

    @property
    def route_share(self) -> float | None:
        """Route delay as a share of the datapath delay."""
        if self.route is None or not self.datapath:
            return None
        return self.route / self.datapath


def unit_lookup(units: dict[str, Unit]) -> list[tuple[str, str]]:
    """``(root prefix, unit name)`` pairs, longest prefix first, for cell lookup."""
    pairs = [(root, u.name) for u in units.values() for root in u.roots]
    return sorted(pairs, key=lambda p: -len(p[0]))


def unit_of(cell: str, lookup: list[tuple[str, str]], units: dict[str, Unit]) -> str | None:
    """Return the unit a hierarchical cell name belongs to, or None.

    Exact root prefixes (from the export) are tried first; a cell whose roots were
    not resolved falls back to the hook's rule, the unit's node name as a path
    component.
    """
    for root, name in lookup:
        if cell == root or cell.startswith(root + "/"):
            return name
    parts = set(cell.split("/"))
    for u in units.values():
        if any(c in parts for c in u.cells):
            return u.name
    return None


def outside_of(cell: str) -> str:
    """Name for an endpoint outside every unit: the block under the user region.

    ``level0_i/ulp/proc_sys_reset_kernel_slr0/U0/...`` -> ``proc_sys_reset_kernel_slr0``,
    so the 'other' class can be read by what it connects to.
    """
    parts = cell.split("/")
    return parts[2] if len(parts) > 3 and parts[0] == "level0_i" else "/".join(parts[:2])


def classify(export: dict[str, Any], units: dict[str, Unit]) -> list[PathRecord]:
    """Attach the export's placement to the units and classify every path."""
    for name, info in export.get("units", {}).items():
        if name not in units:
            continue
        u = units[name]
        u.roots = list(info.get("roots", []))
        u.n_cells = int(info.get("cells", 0))
        if info.get("x0") is not None:
            u.placed = Box(info["x0"], info["y0"], info["x1"], info["y1"])
    lookup = unit_lookup(units)
    # Vivado reports slack against a path's tightest requirement; a hop under a
    # set_max_delay (the reset tree in builds of 14-16 Oct 2026) would otherwise look
    # worse than a path of the same length under the clock alone. Every slack is
    # restated against the kernel period, the largest requirement seen, so the classes
    # compare.
    period = max((float(p["requirement"]) for p in export["paths"]), default=0.0)
    records = []
    for p in export["paths"]:
        su = unit_of(p["start_cell"], lookup, units) or outside_of(p["start_cell"])
        eu = unit_of(p["end_cell"], lookup, units) or outside_of(p["end_cell"])
        via: list[str] = []
        for cell in p.get("cells", []):
            vu = unit_of(cell, lookup, units) or outside_of(cell)
            if vu not in (su, eu) and vu not in via:
                via.append(vu)
        if su not in units or eu not in units:
            kind = "other"
        elif su == eu and not via:
            kind = "intra"
        else:
            kind = "inter"
            if su == eu:
                # A round trip: the combinational ready of the consumer answering
                # the valid this unit sent. Report it as the pair it really is.
                eu = via[0]
        dist = None
        if None not in (p.get("sx"), p.get("sy"), p.get("ex"), p.get("ey")):
            dist = abs(p["sx"] - p["ex"]) + abs(p["sy"] - p["ey"])
        records.append(
            PathRecord(
                slack=float(p["slack"]) + period - float(p["requirement"]),
                datapath=float(p["datapath"]),
                logic=None if p.get("logic") is None else float(p["logic"]),
                route=None if p.get("route") is None else float(p["route"]),
                levels=int(p["levels"]),
                start_cell=p["start_cell"],
                end_cell=p["end_cell"],
                start_unit=su,
                end_unit=eu,
                distance=dist,
                kind=kind,
                via=tuple(via),
            )
        )
    return records


# -- summaries ----------------------------------------------------------------------


def _stats(values: list[float]) -> dict[str, float | None]:
    """Min / mean / max of a list, None-safe."""
    if not values:
        return {"n": 0, "min": None, "mean": None, "max": None}
    return {
        "n": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _bucket(value: float, bounds: tuple[float, ...]) -> str:
    """Label of the first bucket whose upper bound the value does not exceed."""
    lo = 0.0
    for hi in bounds:
        if value <= hi:
            return f"{lo:g}-{hi:g}" if hi != math.inf else f">{lo:g}"
        lo = hi
    return f">{lo:g}"


def _class_summary(records: list[PathRecord]) -> dict[str, Any]:
    """Slack, route share and endpoint distance statistics of one class."""
    return {
        "paths": len(records),
        "slack": _stats([r.slack for r in records]),
        "route_share": _stats([s for r in records if (s := r.route_share) is not None]),
        "route_ns": _stats([r.route for r in records if r.route is not None]),
        "distance": _stats([r.distance for r in records if r.distance is not None]),
        "levels": _stats([float(r.levels) for r in records]),
        "round_trips": sum(1 for r in records if r.via),
    }


def _by_bucket(
    records: list[PathRecord], key: Callable[[PathRecord], float | None], bounds: tuple[float, ...]
) -> dict[str, dict[str, Any]]:
    """Class summary per bucket of ``key(record)`` (records without a key are skipped)."""
    groups: dict[str, list[PathRecord]] = {}
    for r in records:
        v = key(r)
        if v is None:
            continue
        groups.setdefault(_bucket(v, bounds), []).append(r)
    order = sorted(groups, key=lambda b: float(b.lstrip(">").split("-")[0]))
    return {b: _class_summary(groups[b]) for b in order}


def _top(records: list[PathRecord], units: dict[str, Unit]) -> list[dict[str, Any]]:
    """Return the worst unit pairs of a class: worst path, path count, mean route and distance.

    A wide stream between two units produces one path per bit with the same
    endpoints, so the list is grouped by unit pair rather than by path.
    """
    groups: dict[tuple[str | None, str | None], list[PathRecord]] = {}
    for r in records:
        groups.setdefault((r.start_unit, r.end_unit), []).append(r)
    ranked = sorted(groups.items(), key=lambda kv: min(r.slack for r in kv[1]))[:TOP_N]
    out = []
    for (su, eu), rs in ranked:
        worst = min(rs, key=lambda r: r.slack)
        dists = [r.distance for r in rs if r.distance is not None]
        out.append(
            {
                "slack": worst.slack,
                "paths": len(rs),
                "route": worst.route,
                "levels": worst.levels,
                "distance": statistics.fmean(dists) if dists else None,
                "from": su,
                "from_op": units[su].op if su in units else op_of(worst.start_cell),
                "to": eu,
                "to_op": units[eu].op if eu in units else op_of(worst.end_cell),
            }
        )
    return out


def summarize(records: list[PathRecord], units: dict[str, Unit], label: str) -> dict[str, Any]:
    """Build the full summary of one design: per class, per geometry bucket, worst paths."""
    intra = [r for r in records if r.kind == "intra"]
    inter = [r for r in records if r.kind == "inter"]
    other = [r for r in records if r.kind == "other"]

    def placed_aspect(r: PathRecord) -> float | None:
        """Aspect ratio of the placed box of the path's start unit."""
        u = units.get(r.start_unit or "")
        return u.placed.aspect if u and u.placed else None

    def placed_span(r: PathRecord) -> float | None:
        """Half-perimeter of the placed box of the path's start unit."""
        u = units.get(r.start_unit or "")
        return float(u.placed.half_perimeter) if u and u.placed else None

    def pblock_aspect(r: PathRecord) -> float | None:
        """Aspect ratio of the pblock of the path's start unit."""
        u = units.get(r.start_unit or "")
        return u.pblock.aspect if u and u.pblock else None

    def pair_gap(r: PathRecord) -> float | None:
        """Gap between the placed boxes of the two units of an inter-unit path."""
        a = units.get(r.start_unit or "")
        b = units.get(r.end_unit or "")
        if a and b and a.placed and b.placed:
            return float(a.placed.gap(b.placed))
        return None

    adjacent = [
        r
        for r in inter
        if (a := units.get(r.start_unit or "")) and (b := units.get(r.end_unit or ""))
        and a.placed and b.placed and a.placed.touches(b.placed)
    ]  # fmt: skip
    apart = [r for r in inter if r not in adjacent]
    with_box = [u for u in units.values() if u.placed]
    return {
        "label": label,
        "paths": len(records),
        "classes": {
            "intra": _class_summary(intra),
            "inter": _class_summary(inter),
            "other": _class_summary(other),
        },
        "intra_by_placed_aspect": _by_bucket(intra, placed_aspect, ASPECT_BUCKETS),
        "intra_by_placed_span": _by_bucket(intra, placed_span, (100.0, 200.0, 400.0, math.inf)),
        "intra_by_pblock_aspect": _by_bucket(intra, pblock_aspect, ASPECT_BUCKETS),
        "inter_adjacent": _class_summary(adjacent),
        "inter_apart": _class_summary(apart),
        "inter_by_gap": _by_bucket(inter, pair_gap, DISTANCE_BUCKETS),
        "inter_by_distance": _by_bucket(inter, lambda r: r.distance, DISTANCE_BUCKETS),
        "units": {
            "count": len(units),
            "placed_aspect": _stats([u.placed.aspect for u in with_box]),
            "placed_span": _stats([float(u.placed.half_perimeter) for u in with_box]),
        },
        "worst_intra": _top(intra, units),
        "worst_inter": _top(inter, units),
        "worst_other": _top(other, units),
    }


# -- rendering ----------------------------------------------------------------------


def _fmt(v: float | None, digits: int = 3) -> str:
    """Format a number for a table cell."""
    return "-" if v is None else f"{v:.{digits}f}"


def _cell(c: dict[str, Any], with_share: bool = True) -> str:
    """Describe one class or bucket summary in a table cell."""
    if not c or not c["paths"]:
        return "-"
    share = c["route_share"]["mean"]
    route = f"route {_fmt(c['route_ns']['mean'], 2)} ns"
    if with_share and share is not None:
        route += f" ({_fmt(share * 100, 0)} %)"
    trips = f", {c['round_trips']} round trips" if c.get("round_trips") else ""
    return (
        f"{c['paths']} paths, worst {_fmt(c['slack']['min'])}, mean {_fmt(c['slack']['mean'])}, "
        f"{route}, dist {_fmt(c['distance']['mean'], 0)}{trips}"
    )


def _class_rows(summaries: list[dict[str, Any]], section: str) -> list[str]:
    """Markdown rows comparing one per-class or per-bucket section across designs."""
    keys: list[str] = []
    for s in summaries:
        keys += [k for k in s[section] if k not in keys]
    return [
        "| " + " | ".join([k, *(_cell(s[section].get(k)) for s in summaries)]) + " |" for k in keys
    ]


def _section(title: str, head: str, sep: str, rows: list[str]) -> list[str]:
    """Build a titled Markdown table."""
    return [f"## {title}", "", head, sep, *rows, ""]


def render_markdown(summaries: list[dict[str, Any]]) -> str:
    """Render a Markdown comparison of one or more design summaries."""
    head = "| | " + " | ".join(s["label"] for s in summaries) + " |"
    sep = "|---|" + "---|" * len(summaries)
    out = [
        "# Intra/inter-unit timing split",
        "",
        (
            "Slack in ns against the kernel period (setup; a path under a tighter max_delay is "
            "restated against the period), route = mean net delay on the path and its share "
            "of the datapath, dist = mean Manhattan distance of the endpoints in tile grid units."
        ),
        "",
    ]
    tables = [
        ("Per class", "classes"),
        (
            "Intra-unit paths by the unit's placed bounding-box aspect ratio",
            "intra_by_placed_aspect",
        ),
        (
            "Intra-unit paths by the unit's placed half-perimeter (tiles)",
            "intra_by_placed_span",
        ),
        (
            "Intra-unit paths by pblock aspect ratio (floorplanned designs)",
            "intra_by_pblock_aspect",
        ),
        ("Inter-unit paths by gap between the two units' placed boxes", "inter_by_gap"),
        ("Inter-unit paths by endpoint distance", "inter_by_distance"),
    ]
    for title, section in tables:
        if any(s[section] for s in summaries):
            out += _section(title, head, sep, _class_rows(summaries, section))
    rows = [
        "| "
        + " | ".join([key.replace("inter_", ""), *(_cell(s[key], False) for s in summaries)])
        + " |"
        for key in ("inter_adjacent", "inter_apart")
    ]
    out += _section("Inter-unit paths: units placed adjacent vs apart", head, sep, rows)
    rows = []
    for key, label in (("placed_aspect", "aspect ratio"), ("placed_span", "half-perimeter")):
        cells = [label]
        for s in summaries:
            st = s["units"][key]
            cells.append(
                f"min {_fmt(st['min'], 1)}, mean {_fmt(st['mean'], 1)}, max {_fmt(st['max'], 1)}"
            )
        rows.append("| " + " | ".join(cells) + " |")
    out += _section("Unit shapes (placed bounding boxes)", head, sep, rows)
    for s in summaries:
        for key, title in (
            ("worst_intra", "intra-unit"),
            ("worst_inter", "inter-unit"),
            ("worst_other", "other"),
        ):
            rows = [
                f"| {_fmt(w['slack'])} | {w['paths']} | {_fmt(w['route'], 2)} | {w['levels']} | "
                f"{_fmt(w['distance'], 0)} | {w['from']} ({w['from_op']}) | "
                f"{w['to']} ({w['to_op']}) |"
                for w in s[key]
            ]
            out += _section(
                f"{s['label']}: worst {title} unit pairs",
                "| slack | paths | route | levels | dist | from | to |",
                "|---|---|---|---|---|---|---|",
                rows,
            )
    return "\n".join(out)


# -- driver -------------------------------------------------------------------------


def analyse(
    export_file: Path, floorplan_dir: Path, label: str
) -> tuple[dict[str, Any], list[PathRecord]]:
    """Classify one export against the floorplan artefacts and summarise it."""
    units = load_units(floorplan_dir)
    export = json.loads(export_file.read_text())
    records = classify(export, units)
    return summarize(records, units, label), records


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; see the module docstring."""
    ap = argparse.ArgumentParser(prog="timing_split", description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="routed checkpoint of the floorplanned design",
    )
    ap.add_argument(
        "--floorplan-dir",
        required=True,
        type=Path,
        help="directory with finn_floorplan.json and finn_floorplan_units.json",
    )
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="routed checkpoint of the unfloorplanned baseline (same netlist)",
    )
    ap.add_argument(
        "--baseline-model",
        type=Path,
        default=None,
        help="parent model of the baseline build (step_deployment_package.onnx); "
        "required with --baseline",
    )
    ap.add_argument("--paths", type=int, default=2000, help="worst paths to export per design")
    ap.add_argument("--clock", default=DEFAULT_CLOCK, help="clock pattern for get_clocks")
    ap.add_argument("--vivado", default="vivado", help="Vivado executable")
    ap.add_argument(
        "--reuse-exports",
        action="store_true",
        help="skip Vivado when the export JSON files already exist",
    )
    args = ap.parse_args(argv)  # fmt: skip

    if not args.reuse_exports and shutil.which(args.vivado) is None:
        print(f"{args.vivado} is not on PATH (load the Xilinx modules first)", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    units = load_units(args.floorplan_dir)
    designs: dict[str, dict[str, Unit]] = {"floorplanned": units}
    if args.baseline is not None:
        if args.baseline_model is None:
            print("--baseline needs --baseline-model to match the nodes", file=sys.stderr)
            return 2
        mapping = cell_map(units, args.baseline_model)
        designs["baseline"] = remap_units(units, mapping)
        cells = [c for u in units.values() for c in u.cells]
        mapped = sum(1 for c in cells if c in mapping)
        print(f"node map: {mapped} of {len(cells)} unit cells have a baseline counterpart")

    jobs = []
    for label, design_units in designs.items():
        roots = args.out / f"unit_roots_{label}.txt"
        write_roots_file(design_units, roots)
        checkpoint = args.checkpoint if label == "floorplanned" else args.baseline
        jobs.append((label, checkpoint, roots, args.out / f"export_{label}.json"))
    # The reset tree's registers are exported as their own group (builds of 14-16 Oct
    # 2026 also put them under a set_max_delay).
    from finn.transformation.fpgadataflow.create_stitched_ip import RESET_REGISTER_PATTERN

    reset_pattern = RESET_REGISTER_PATTERN
    if not args.reuse_exports or not all(j[3].exists() for j in jobs):
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            list(
                pool.map(
                    lambda j: run_export(
                        j[1], j[2], j[3], args.paths, args.clock, args.vivado, reset_pattern
                    ),
                    [j for j in jobs if not (args.reuse_exports and j[3].exists())],
                )
            )
    summaries = []
    for label, _, _, export in jobs:
        export_data = json.loads(export.read_text())
        records = classify(export_data, designs[label])
        summary = summarize(records, designs[label], label)
        summaries.append(summary)
        (args.out / f"paths_{label}.json").write_text(
            json.dumps([asdict(r) for r in records], indent=1)
        )
    (args.out / "timing_split.json").write_text(json.dumps(summaries, indent=1))
    md = render_markdown(summaries)
    (args.out / "timing_split.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

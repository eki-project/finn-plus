"""Two-dimensional pblock packer for the FINN+ floorplanner.

This is the arithmetic half of the Level-2 floorplanner (FLOORPLANNING_PLAN.md,
Milestone 6). Vivado stays the measurement and application layer: the Tcl hook
exports the device census and the measured per-unit demands as JSON, runs this
module as a separate process, and applies the rectangles it returns. Nothing
here imports Vivado, FINN or anything outside the standard library, so the
module runs under whatever interpreter the hook was given and is unit-tested
against JSON fixtures captured from real builds.

Model
-----
A unit is not a fixed rectangle. At a chosen height (in SLICE rows) and a chosen
start column, the width it needs follows from scanning the device's columns
rightwards -- SLICE, block RAM, DSP and URAM columns interleaved in *tile column*
order -- until every resource demand is met from the sites that are actually
available in those rows. Tall-and-narrow and short-and-wide are both legal
shapes of the same unit. Units are placed bottom-left on a skyline per SLR, in
dataflow order, choosing the shape and position with the smallest half-perimeter
(tile columns plus SLICE rows: the longest wire a path inside the unit can need),
SLICE waste deciding between equals.

Cascade chains are first-class: every chain needs its own run of contiguous
rows in one column, block RAM chains additionally inside one clock region. The
census carries each column's available rows as runs and the clock-region bands
per site kind, so both are exact.

Contract
--------
``census.json``::

    {"version": 1, "part": "...", "slrs": [0, 1, 2],
     "bands": {"SLICE": [[0, 59], ...], "RAMB36": [[0, 11], ...], ...},
     "columns": {"0": {"SLICE": [{"x": 0, "tile_col": 12, "slicem": false,
                                  "runs": [[0, 239]]}, ...],
                       "RAMB36": [{"x": 0, "tile_col": 15, "runs": [[0, 47]]}],
                       ...}, ...}}

``units.json``: a list of ``{"name", "slr", "order", "demands": {...},
"chains": {"RAMB36": [8, 8, ...], ...}}``. ``feedback.json`` (later rounds):
``{"unit": {"luts": shortfall, ..., "chains": 1}}``. The result is written both
as JSON (for reports and tests) and as a Tcl ``dict`` the hook can ``source``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: Site kinds the packer knows, in the order they are exported.
KINDS: tuple[str, ...] = ("SLICE", "RAMB36", "RAMB18", "DSP48E2", "URAM288")

#: Kinds that get their own columns (RAMB18 rides on the RAMB36 columns).
SPARSE_KINDS: tuple[str, ...] = ("RAMB36", "DSP48E2", "URAM288")

#: Demand keys, as the Tcl measurement emits them.
DEMAND_KEYS: tuple[str, ...] = (
    "luts",
    "ffs",
    "carries",
    "brams",
    "dsps",
    "urams",
    "lutrams",
    "srls",
    # Distributed-RAM macros (RAM64M8, RAM32M16, RAM64X1D, ...), counted from the
    # netlist by the hook. Each macro occupies a SLICEM slice of its own: an input
    # generator with 32 RAM64M8 and 32 RAM64X1D (320 memory LUTs) placed in 64
    # slices, never fewer, in hard regions on the U55C (2 Oct 2026). Zero when the
    # unit was sized from estimates, in which case the LUT count stands in.
    "ram_macros",
)

#: Which sparse kind satisfies which demand key.
KIND_FOR_DEMAND: dict[str, str] = {"brams": "RAMB36", "dsps": "DSP48E2", "urams": "URAM288"}

#: Kinds whose cascades Vivado refuses to place across a clock-region boundary.
CR_BOUND_KINDS: tuple[str, ...] = ("RAMB36", "RAMB18")

#: UltraScale+ SLICE contents.
SLICE_LUTS = 8
SLICE_FFS = 16
SLICE_CARRY = 1

#: The coarsest tile pitch in SLICE rows: a URAM288 tile spans 15 rows (4 sites),
#: a block RAM tile 5 (1 RAMB36 / 2 RAMB18), a DSP tile 5 (2 DSP48E2). A rectangle
#: that uses sparse sites is placed and sized on this pitch so its sparse ranges
#: partition the columns exactly and no tile is claimed by two units. Vivado
#: would otherwise widen a misaligned range to whole tiles (Vivado 12-25128) and
#: hand the straddling tile to both neighbours.
TILE_PITCH = 15

#: Candidate rectangle heights in SLICE rows for units that use only SLICEs:
#: anything from one block RAM tile up to one clock region and beyond.
CANDIDATE_HEIGHTS: tuple[int, ...] = (5, 10, 15, 20, 30, 40, 60, 80, 120, 180, 240)

#: Whether to shorten a unit's block RAM / DSP / URAM columns to the rows it needs
#: instead of its full SLICE height. Off: measured on the resnet18 post-opt units
#: (2 Oct 2026) it cut block RAM over-claim from 1.5x to 1.07x but dropped coverage
#: from 288/289 to 249/289 -- the lowered column tops change where later units
#: start and what they claim, and the FIFOs packed last lose their SLICEM. Kept as
#: a switch so the interaction can be studied with the pinned-round protocol.
TRIM_SPARSE_COLUMNS = False

#: Candidate heights for units that use block RAM, DSP or URAM: tile-pitch multiples.
ALIGNED_HEIGHTS: tuple[int, ...] = (15, 30, 45, 60, 75, 90, 120, 150, 180, 240)

#: How many skyline start positions (lowest first) to try per unit and height.
MAX_START_POSITIONS = 48

#: Tile columns on either side of a unit's dataflow predecessor that are tried as
#: start columns, so sitting next to it is always among the candidates.
ANCHOR_REACH = 24

#: Weight of the gap to the predecessor's rectangle against the unit's own
#: half-perimeter. A tile of gap is wire one stream must cross; a tile of shape is
#: wire every net inside the unit may have to cross, and the high-fan-out ones
#: (a weight stream's pointer driving 1,027 registers) span the whole unit. At 1.0
#: the 71-DSP MVAUs of mobilenetv1 took 180-row strips (half-perimeter 212, gap 50)
#: over 90-row boxes (137, gap 140), and such a strip held the design's worst path
#: (14 Oct 2026, rebuild5: 2.98 ns against 4.2 ns for the worst inter-unit path at
#: any gap). At 0.5, measured on the four fixtures and that run's units, the units
#: taller than 90 rows drop from 8-11 to 3-9 per design and the mean half-perimeter
#: by 12-16 % for 3-8 tiles more mean gap (median still 0), with every fixture
#: placing the same number of units; 0.25 gains little shape and loses locality.
LOCALITY_WEIGHT = 0.5

#: Pack / apply / verify rounds the hook may run; the packer itself is stateless.
MAX_ROUNDS = 3

#: Shift registers a SLICEM slice holds in practice. Measured on FINN's SRL FIFO
#: (`Q_srl`, 224 deep x 160 wide = 1,120 SRLC32E in chains of seven) placed alone in
#: hard regions on the U55C, 2 Oct 2026: regions of 280, 300 and 320 SLICEM slices
#: fail, 340 places, and the placed shift registers occupy exactly 320 slices --
#: 3.50 per slice, not the eight LUT positions a slice has. A chained lane with its
#: read mux does not share a slice the way the LUT count suggests. `--srl-per-slice`
#: overrides it for another FIFO style.
DEFAULT_SRL_PER_SLICE = 3.5

#: Default fraction of a rectangle's SLICEM LUTs that LUTRAM and SRLs may fill. The
#: legaliser cannot pack memory LUTs to the site count inside a hard pblock
#: ([Place 30-1112] on resnet18, 2 Oct 2026, with 1320 SLICEM LUTs for a demand the
#: count said fit). Overridden per build from the hook (`--slicem-util`).
DEFAULT_SLICEM_UTILIZATION = 0.8

#: Default fraction of a rectangle's slices that plain logic -- LUTs, flip-flops,
#: carries -- may fill. With logic sized at exactly eight LUTs per slice, small
#: units failed detail placement at random (`[Place 30-488]`/`30-1153`): of 285
#: mobilenetv1 units in four placements on 4 Oct 2026 every run lost a different
#: handful, nearly all at a fill of 0.95 or more in pblocks of 5-45 slices, and
#: carry-heavy input generators needed 28-34 slices where 30 were given. Decided
#: 4 Oct 2026: 0.75. Overridden per build from the hook (`--logic-util`).
DEFAULT_LOGIC_UTILIZATION = 0.75


@dataclass
class Column:
    """One column of sites of one kind, with the rows that are actually available.

    Attributes:
        kind: Site kind, one of `KINDS`.
        x: The kind's own X coordinate (``RAMB36_X<x>Y...``).
        tile_col: Column in the device's tile grid, shared across kinds, which is
            what makes rectangles across interleaved kinds well defined.
        runs: Available rows as inclusive ``(lo, hi)`` runs, ascending.
        slicem: For SLICE columns, whether the sites are SLICEM (LUTRAM/SRL capable).
    """

    kind: str
    x: int
    tile_col: int
    runs: list[tuple[int, int]]
    slicem: bool = False

    def available(self, lo: int, hi: int) -> int:
        """Count available rows inside the inclusive row window ``lo..hi``."""
        total = 0
        for a, b in self.runs:
            s, e = max(a, lo), min(b, hi)
            if e >= s:
                total += e - s + 1
        return total

    def segments(self, lo: int, hi: int, bands: list[tuple[int, int]] | None) -> list[int]:
        """Lengths of the contiguous available runs inside ``lo..hi``.

        With ``bands`` given, a run is cut at every band boundary, which is what
        a clock-region-bound cascade sees.
        """
        out: list[int] = []
        for a, b in self.runs:
            s, e = max(a, lo), min(b, hi)
            if e < s:
                continue
            if not bands:
                out.append(e - s + 1)
                continue
            for b0, b1 in bands:
                t0, t1 = max(s, b0), min(e, b1)
                if t1 >= t0:
                    out.append(t1 - t0 + 1)
        return out

    @property
    def lo(self) -> int:
        """Lowest available row."""
        return self.runs[0][0]

    @property
    def hi(self) -> int:
        """Highest available row."""
        return self.runs[-1][1]


@dataclass
class Census:
    """The device as the hook measured it: columns per SLR and kind, plus bands."""

    slrs: list[int]
    bands: dict[str, list[tuple[int, int]]]
    columns: dict[int, dict[str, list[Column]]]
    part: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> Census:
        """Build a census from the JSON the Tcl hook writes."""
        bands = {k: [(int(a), int(b)) for a, b in v] for k, v in data.get("bands", {}).items()}
        columns: dict[int, dict[str, list[Column]]] = {}
        for slr_key, kinds in data["columns"].items():
            slr = int(slr_key)
            columns[slr] = {}
            for kind, cols in kinds.items():
                columns[slr][kind] = [
                    Column(
                        kind=kind,
                        x=int(c["x"]),
                        tile_col=int(c["tile_col"]),
                        runs=[(int(a), int(b)) for a, b in c["runs"]],
                        slicem=bool(c.get("slicem", False)),
                    )
                    for c in cols
                    if c["runs"]
                ]
        slrs = [int(s) for s in data.get("slrs", sorted(columns))]
        return cls(slrs=slrs, bands=bands, columns=columns, part=str(data.get("part", "")))

    def slice_bounds(self, slr: int) -> tuple[int, int]:
        """Lowest and highest available SLICE row in an SLR."""
        cols = self.columns[slr].get("SLICE", [])
        if not cols:
            raise ValueError(f"SLR{slr} has no SLICE columns in the census")
        return min(c.lo for c in cols), max(c.hi for c in cols)

    def map_rows(self, kind: str, y0: int, y1: int) -> tuple[int, int] | None:
        """Map an inclusive SLICE row window onto the rows of another kind.

        Within a clock region a kind's rows follow a fixed pitch (one RAMB36 or
        two DSP48E2 per five SLICE rows, four URAM288 per fifteen), anchored at
        the *top* of the region: the bottom region of the U55C has only 18 DSP
        rows under its 60 SLICE rows because its lowest tiles carry no DSPs, and
        a proportional map would smear the remaining rows over the whole region.
        Boundaries are exclusive-consistent, so adjacent SLICE windows map onto
        disjoint, gap-free kind windows. Falls back to a proportional map over the
        whole device when the band lists do not line up. Returns ``None`` when the
        window holds no row of the kind.
        """
        if kind == "SLICE":
            return (y0, y1)
        sb = self.bands.get("SLICE") or []
        kb = self.bands.get(kind) or []
        if not kb:
            return None
        if len(sb) == len(kb) and sb:
            lo = self._top_anchored(kind, y0, sb, kb)
            hi = self._top_anchored(kind, y1 + 1, sb, kb) - 1
        else:
            s_lo, s_hi = sb[0][0], sb[-1][1]
            k_lo, k_hi = kb[0][0], kb[-1][1]
            scale = (k_hi - k_lo + 1) / (s_hi - s_lo + 1)
            lo = k_lo + int((y0 - s_lo) * scale)
            hi = k_lo + int((y1 + 1 - s_lo) * scale) - 1
        # An empty window is empty: the bottom tiles of the U55C's lowest clock
        # region carry no DSPs, and rounding them up to one row handed DSP row 0
        # to two stacked units.
        return (lo, hi) if hi >= lo else None

    def pitch(self, kind: str) -> float:
        """Kind rows per SLICE row in a regular clock region (the tallest band)."""
        sb = self.bands["SLICE"]
        kb = self.bands[kind]
        s_h = max(b - a + 1 for a, b in sb)
        k_h = max(b - a + 1 for a, b in kb)
        return k_h / s_h

    def _top_anchored(
        self, kind: str, y: int, sb: list[tuple[int, int]], kb: list[tuple[int, int]]
    ) -> int:
        """Kind row index of the boundary just below SLICE row ``y`` (exclusive-consistent)."""
        pitch = self.pitch(kind)
        for (s0, s1), (k0, k1) in zip(sb, kb):  # noqa: B905 -- equal length checked by caller
            if s0 <= y <= s1 + 1:
                k = k1 + 1 - math.ceil((s1 + 1 - y) * pitch)
                return max(k0, min(k1 + 1, k))
        if y > sb[-1][1] + 1:
            return kb[-1][1] + 1
        return kb[0][0]


@dataclass
class Unit:
    """One placement unit with its measured demands and cascade chains."""

    name: str
    slr: int
    order: int
    demands: dict[str, int]
    chains: dict[str, list[int]] = field(default_factory=dict)
    min_height: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> Unit:
        """Build a unit from the JSON the Tcl hook writes."""
        demands = {k: int(data.get("demands", {}).get(k, 0)) for k in DEMAND_KEYS}
        # Every unit owns cells and therefore needs a pblock, however small; a unit
        # measured (or estimated) at zero LUTs still claims one SLICE column.
        demands["luts"] = max(demands["luts"], 1)
        chains = {k: [int(n) for n in v] for k, v in data.get("chains", {}).items() if v}
        return cls(
            name=str(data["name"]),
            slr=int(data["slr"]),
            order=int(data.get("order", 0)),
            demands=demands,
            chains=chains,
            min_height=int(data.get("min_height", 0)),
        )

    def uses_sparse(self) -> bool:
        """Whether the unit needs block RAM, DSP or URAM sites (or carries chains)."""
        d = self.demands
        return bool(d["brams"] or d["dsps"] or d["urams"] or self.chains)

    def chain_lengths(self, kind: str) -> list[int]:
        """Chains a sparse kind's columns must host; RAMB18 folds onto RAMB36."""
        lens = list(self.chains.get(kind, []))
        if kind == "RAMB36":
            lens += [(n + 1) // 2 for n in self.chains.get("RAMB18", [])]
        return lens

    def logic_luts(self) -> int:
        """LUTs used as logic: the measured total minus distributed RAM and shift registers."""
        d = self.demands
        return max(0, d["luts"] - d.get("lutrams", 0) - d.get("srls", 0))

    def slices_needed(
        self,
        slicem_util: float = DEFAULT_SLICEM_UTILIZATION,
        srl_per_slice: float = DEFAULT_SRL_PER_SLICE,
        logic_util: float = DEFAULT_LOGIC_UTILIZATION,
    ) -> int:
        """SLICEs (of either type) the unit's LUTs need: memory slices plus logic.

        A slice that holds a distributed-RAM macro or shift registers holds no
        logic LUT next to them, so the two add up; counting all LUTs at eight per
        slice gave an input generator with 64 macros and 366 logic LUTs 90 slices
        where it needs 126 (job 3551390, 3 Oct 2026). Logic fills slices up to
        ``logic_util`` (see `DEFAULT_LOGIC_UTILIZATION`).
        """
        return self.slicem_slices(slicem_util, srl_per_slice) + math.ceil(
            self.logic_luts() / (SLICE_LUTS * logic_util)
        )

    def needed_slices(
        self,
        slicem_util: float = DEFAULT_SLICEM_UTILIZATION,
        srl_per_slice: float = DEFAULT_SRL_PER_SLICE,
        logic_util: float = DEFAULT_LOGIC_UTILIZATION,
    ) -> int:
        """SLICEs this unit occupies: the binding one of LUTs, flip-flops and carries."""
        d = self.demands
        return max(
            self.slices_needed(slicem_util, srl_per_slice, logic_util),
            math.ceil(d["ffs"] / (SLICE_FFS * logic_util)),
            math.ceil(d["carries"] / (SLICE_CARRY * logic_util)),
            1,
        )

    def slicem_slices(
        self,
        slicem_util: float = DEFAULT_SLICEM_UTILIZATION,
        srl_per_slice: float = DEFAULT_SRL_PER_SLICE,
    ) -> int:
        """SLICEM slices the unit's distributed RAM and shift registers need.

        Every distributed-RAM macro occupies one SLICEM slice once placed (64
        macros, 64 slices, whatever their LUT count), but a region with exactly
        that many does not place reliably: out of context, three FINN input
        generators placed or failed at random with SLICEM = macros, and all three
        placed with SLICEM = macros / 0.8 (3 Oct 2026). So macros get the same
        utilisation ceiling as LUT-counted distributed RAM; without a macro count
        the LUT count under the ceiling stands in. Shift registers pack at the
        measured density `srl_per_slice`, which already is a placing threshold,
        so no further ceiling applies to them.
        """
        d = self.demands
        if d.get("ram_macros", 0) > 0:
            rams = math.ceil(d["ram_macros"] / slicem_util)
        else:
            rams = math.ceil(d["lutrams"] / (SLICE_LUTS * slicem_util)) if d["lutrams"] else 0
        srls = math.ceil(d["srls"] / srl_per_slice) if d["srls"] else 0
        return rams + srls


@dataclass
class Placement:
    """Where one unit went, or why it did not."""

    name: str
    slr: int
    status: str
    c0: int = -1
    c1: int = -1
    y0: int = -1
    y1: int = -1
    ranges: list[str] = field(default_factory=list)
    capacity: dict[str, int] = field(default_factory=dict)
    waste: int = 0
    reason: str = ""
    round: int = 1
    columns: list[int] = field(default_factory=list)
    column_rows: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON form."""
        return {
            "slr": self.slr,
            "status": self.status,
            "tile_col0": self.c0,
            "tile_col1": self.c1,
            "slice_y0": self.y0,
            "slice_y1": self.y1,
            "columns": self.columns,
            "column_rows": {str(k): v for k, v in self.column_rows.items()},
            "ranges": self.ranges,
            "capacity": self.capacity,
            "waste_slices": self.waste,
            "reason": self.reason,
            "round": self.round,
        }

    def to_tcl(self) -> str:
        """Tcl dict literal the hook sources; names and ranges are Vivado-safe."""
        cap = " ".join(f"{k} {v}" for k, v in self.capacity.items())
        return (
            f"{{slr {self.slr} status {self.status} round {self.round} "
            f"x0 {self.c0} x1 {self.c1} y0 {self.y0} y1 {self.y1} "
            f"columns {{{' '.join(str(c) for c in self.columns)}}} "
            f"column_rows {{{' '.join(f'{k} {v}' for k, v in self.column_rows.items())}}} "
            f"ranges {{{' '.join(self.ranges)}}} capacity {{{cap}}} "
            f"reason {{{_tcl_brace_safe(self.reason)}}}}}"
        )


def _tcl_brace_safe(text: str) -> str:
    """Strip characters that would unbalance a Tcl brace literal."""
    return "".join(ch for ch in text if ch not in '{}[]$\\"')


def pack_chains(segments: list[int], lengths: list[int]) -> bool:
    """Can every chain get its own run of rows? Best-fit decreasing.

    Longest chain first, into the smallest segment that still holds it; several
    short chains may stack in one segment.
    """
    segs = list(segments)
    for length in sorted(lengths, reverse=True):
        pick = -1
        for i, size in enumerate(segs):
            if size >= length and (pick < 0 or size < segs[pick]):
                pick = i
        if pick < 0:
            return False
        segs[pick] -= length
    return True


class SlrPacker:
    """Skyline packer for the units of one SLR."""

    def __init__(
        self,
        census: Census,
        slr: int,
        pinned: list[Placement] | None = None,
        slicem_util: float = DEFAULT_SLICEM_UTILIZATION,
        srl_per_slice: float = DEFAULT_SRL_PER_SLICE,
        logic_util: float = DEFAULT_LOGIC_UTILIZATION,
    ) -> None:
        """Set up the column order and the skyline, raised over any pinned rectangles."""
        self.census = census
        self.slr = slr
        self.slicem_util = slicem_util
        self.srl_per_slice = srl_per_slice
        self.logic_util = logic_util
        cols: list[Column] = []
        for kind in KINDS:
            cols.extend(census.columns[slr].get(kind, []))
        self.cols = sorted(cols, key=lambda c: (c.tile_col, KINDS.index(c.kind)))
        self.tile_cols = sorted({c.tile_col for c in self.cols})
        self.by_tile: dict[int, list[Column]] = {}
        for c in self.cols:
            self.by_tile.setdefault(c.tile_col, []).append(c)
        self.y_lo, self.y_hi = census.slice_bounds(slr)
        # Free SLICE-row intervals per tile column (inclusive). A plain skyline top
        # per column lost every row under a unit that had to start higher because
        # of another of its columns: resnet18 SLR2 had its block RAM columns "full"
        # with 52 % of their rows claimed. Intervals keep those rows available.
        self.free: dict[int, list[tuple[int, int]]] = {
            t: [(self.y_lo, self.y_hi)] for t in self.tile_cols
        }
        for p in pinned or []:
            if p.status == "placed":
                cols = p.columns or list(range(p.c0, p.c1 + 1))
                h = p.y1 - p.y0 + 1
                self._claim({t: (p.y0, p.y0 + p.column_rows.get(t, h) - 1) for t in cols})

    # -- skyline -----------------------------------------------------------------

    def _claim(self, spans: dict[int, tuple[int, int]]) -> None:
        """Remove an inclusive row span from each tile column's free intervals."""
        for t, (a, b) in spans.items():
            if t not in self.free:
                continue
            out: list[tuple[int, int]] = []
            for lo, hi in self.free[t]:
                if b < lo or a > hi:
                    out.append((lo, hi))
                    continue
                if lo < a:
                    out.append((lo, a - 1))
                if b < hi:
                    out.append((b + 1, hi))
            self.free[t] = out

    def _free_over(self, t: int, y0: int, y1: int) -> bool:
        """Whether tile column ``t`` is free over the inclusive rows y0..y1."""
        return any(lo <= y0 and y1 <= hi for lo, hi in self.free.get(t, []))

    def _bottoms(self, t: int, h: int, aligned: bool) -> list[int]:
        """Rows at which a rectangle of height ``h`` can start in tile column ``t``."""
        out = []
        for lo, hi in self.free.get(t, []):
            y = self._bottom(lo, aligned)
            if y + h - 1 <= hi:
                out.append(y)
        return out

    @property
    def top(self) -> dict[int, int]:
        """Lowest free row per tile column (the classic skyline), for diagnostics."""
        return {t: (iv[0][0] if iv else self.y_hi + 1) for t, iv in self.free.items()}

    # -- capacity of a rectangle ---------------------------------------------------

    def rectangle_capacity(self, c0: int, c1: int, y0: int, y1: int) -> dict[str, int]:
        """Sites of every kind inside a tile-column span and SLICE row window."""
        cap = {
            "slices": 0,
            "slicem": 0,
            "RAMB36": 0,
            "RAMB18": 0,
            "DSP48E2": 0,
            "URAM288": 0,
        }
        rows: dict[str, tuple[int, int] | None] = {}
        for t in self.tile_cols:
            if t < c0 or t > c1:
                continue
            for col in self.by_tile[t]:
                if col.kind == "SLICE":
                    n = col.available(y0, y1)
                    cap["slices"] += n
                    if col.slicem:
                        cap["slicem"] += n
                    continue
                if col.kind not in rows:
                    rows[col.kind] = self._kind_window(col.kind, y0, y1)
                win = rows[col.kind]
                if win is not None:
                    cap[col.kind] += col.available(*win)
        return cap

    def chain_segments(self, kind: str, c0: int, c1: int, y0: int, y1: int) -> list[int]:
        """Row segments the columns of ``kind`` inside the rectangle offer to chains."""
        win = self._kind_window(kind, y0, y1)
        if win is None:
            return []
        bands = self.census.bands.get(kind) if kind in CR_BOUND_KINDS else None
        segs: list[int] = []
        for t in self.tile_cols:
            if t < c0 or t > c1:
                continue
            for col in self.by_tile[t]:
                if col.kind == kind:
                    segs.extend(col.segments(win[0], win[1], bands))
        return segs

    def shortfall(self, unit: Unit, c0: int, c1: int, y0: int, y1: int) -> dict[str, int]:
        """Return what the rectangle lacks for the unit; empty when it satisfies everything."""
        cap = self._blank()
        segs: dict[str, list[int]] = {}
        rows: dict[str, tuple[int, int] | None] = {}
        for t in self.tile_cols:
            if c0 <= t <= c1:
                for col in self.by_tile[t]:
                    self._add_column(col, y0, y1, cap, segs, rows)
        return self._missing(unit, cap, segs)

    # -- search ------------------------------------------------------------------

    def heights(self, unit: Unit) -> list[int]:
        """Candidate heights for a unit, ascending, within the SLR.

        Units that use sparse sites get tile-pitch multiples (see `TILE_PITCH`);
        SLICE-only units may be as short as one block RAM tile.
        """
        max_h = self.y_hi - self.y_lo + 1
        table = ALIGNED_HEIGHTS if unit.uses_sparse() else CANDIDATE_HEIGHTS
        hs = [h for h in table if unit.min_height <= h <= max_h]
        if max_h not in hs and max_h >= unit.min_height and max_h % TILE_PITCH == 0:
            hs.append(max_h)
        return sorted(set(hs))

    @staticmethod
    def _bottom(y: int, aligned: bool) -> int:
        """Lowest row a rectangle may start at, snapped up to the tile pitch if aligned.

        Only units that use block RAM, DSP or URAM snap: their sparse ranges come
        from whole tiles inside the SLICE window (`_kind_window`), and a window that
        starts mid-tile would need an extra tile of height to contain one -- on
        mobilenetv1 that doubled every small block RAM user's claim and exhausted
        the block RAM columns (274 of 285 placed instead of 285). SLICE-only units
        start on any row, so the stacks of small units leave no gaps.
        """
        if not aligned:
            return y
        return -(-y // TILE_PITCH) * TILE_PITCH

    def _kind_window(self, kind: str, y0: int, y1: int) -> tuple[int, int] | None:
        """Rows of ``kind`` strictly inside the tile-aligned part of a SLICE window.

        A kind range that straddles a tile boundary is widened by Vivado to whole
        tiles (Vivado 12-25128) and would then share a tile with a neighbour, so
        sparse ranges are taken only from the interior whole tiles of the SLICE
        window. SLICE itself is one row per tile and needs no alignment.
        """
        if kind == "SLICE":
            return (y0, y1)
        lo = -(-y0 // TILE_PITCH) * TILE_PITCH
        hi = ((y1 + 1) // TILE_PITCH) * TILE_PITCH - 1
        if hi < lo:
            return None
        return self.census.map_rows(kind, lo, hi)

    def memory_luts(self, slicem_slices: int) -> int:
        """LUTRAM/SRL capacity of that many SLICEM sites under the utilisation ceiling."""
        return int(slicem_slices * SLICE_LUTS * self.slicem_util)

    def _blank(self) -> dict[str, int]:
        """Return an empty running capacity total."""
        return {"slices": 0, "slicem": 0, "RAMB36": 0, "RAMB18": 0, "DSP48E2": 0, "URAM288": 0}

    def _add_column(
        self,
        col: Column,
        y0: int,
        y1: int,
        cap: dict[str, int],
        segs: dict[str, list[int]],
        rows: dict[str, tuple[int, int] | None],
    ) -> None:
        """Accumulate one column's sites (and chain segments) into a running total."""
        if col.kind == "SLICE":
            n = col.available(y0, y1)
            cap["slices"] += n
            if col.slicem:
                cap["slicem"] += n
            return
        if col.kind not in rows:
            rows[col.kind] = self._kind_window(col.kind, y0, y1)
        win = rows[col.kind]
        if win is None:
            return
        cap[col.kind] += col.available(*win)
        if col.kind in SPARSE_KINDS:
            bands = self.census.bands.get(col.kind) if col.kind in CR_BOUND_KINDS else None
            segs.setdefault(col.kind, []).extend(col.segments(win[0], win[1], bands))

    def _missing(
        self, unit: Unit, cap: dict[str, int], segs: dict[str, list[int]]
    ) -> dict[str, int]:
        """Return the shortfall of a running total against the unit's demands and chains."""
        d = unit.demands
        missing: dict[str, int] = {}
        checks = (
            (
                "luts",
                unit.slices_needed(self.slicem_util, self.srl_per_slice, self.logic_util)
                * SLICE_LUTS,
                cap["slices"] * SLICE_LUTS,
            ),
            ("ffs", d["ffs"], int(cap["slices"] * SLICE_FFS * self.logic_util)),
            ("carries", d["carries"], int(cap["slices"] * SLICE_CARRY * self.logic_util)),
            ("lutrams", unit.slicem_slices(self.slicem_util, self.srl_per_slice), cap["slicem"]),
            ("brams", d["brams"], cap["RAMB36"]),
            ("dsps", d["dsps"], cap["DSP48E2"]),
            ("urams", d["urams"], cap["URAM288"]),
        )
        for key, need, have in checks:
            if need > have:
                missing[key] = need - have
        for kind in SPARSE_KINDS:
            lens = unit.chain_lengths(kind)
            if lens and not pack_chains(segs.get(kind, []), lens):
                missing[f"chains:{kind}"] = 1
        return missing

    def _wants(
        self,
        unit: Unit,
        col: Column,
        cap: dict[str, int],
        segs: dict[str, list[int]],
        spare_slicem: bool = True,
    ) -> bool:
        """Whether a unit still needs a column of this kind, given what it has claimed.

        This is what makes a unit's region a *selection* of columns inside its span
        rather than the whole span: a weight memory in distributed RAM takes the
        SLICEM columns and leaves the SLICEL, DSP and block RAM columns between
        them to its neighbours; a DSP-heavy layer takes the DSP columns it needs
        and a few SLICE columns, not every column in between. Nothing is shared --
        every site belongs to one unit -- so all capacity arithmetic stays exact.
        """
        d = unit.demands
        if col.kind == "SLICE":
            memory_short = unit.slicem_slices(self.slicem_util, self.srl_per_slice) > cap["slicem"]
            logic_short = (
                unit.slices_needed(self.slicem_util, self.srl_per_slice, self.logic_util)
                > cap["slices"]
                or d["ffs"] > cap["slices"] * SLICE_FFS * self.logic_util
                or d["carries"] > cap["slices"] * SLICE_CARRY * self.logic_util
            )
            if col.slicem:
                # SLICEM is the scarce half of the fabric for a FINN design: resnet18
                # measured 83-96 % of each SLR's memory-LUT ceiling. Plain logic
                # takes SLICEM only when `spare_slicem` is off, i.e. after a search
                # over SLICEL columns alone found no room.
                return memory_short or (logic_short and not spare_slicem)
            return logic_short
        if col.kind == "RAMB18":
            return False  # rides on the RAMB36 column at the same tile column
        key = next((k for k, kind in KIND_FOR_DEMAND.items() if kind == col.kind), None)
        if key is None:
            return False
        if d[key] > cap[col.kind]:
            return True
        lens = unit.chain_lengths(col.kind)
        return bool(lens) and not pack_chains(segs.get(col.kind, []), lens)

    def place(self, unit: Unit, anchors: list[Placement] | None = None) -> Placement:
        """Find the most compact feasible column selection for a unit and claim it.

        Two searches: the first lets plain logic claim SLICEL columns only, so
        SLICEM stays available for units that need LUTRAM or shift registers; if
        that finds nothing, the second may take SLICEM for logic as well.
        ``anchors`` are the placed rectangles of the unit's dataflow neighbours in
        this SLR (`_neighbours`): candidates next to them rank ahead (`_search`).
        """
        anchors = [a for a in anchors or [] if a.status == "placed"]
        best, last_missing = self._search(unit, spare_slicem=True, anchors=anchors)
        if best is None:
            best, missing2 = self._search(unit, spare_slicem=False, anchors=anchors)
            last_missing = last_missing or missing2
        if best is None:
            reason = "no room" if not last_missing else "starved: " + ",".join(sorted(last_missing))
            return Placement(name=unit.name, slr=self.slr, status="unplaced", reason=reason)
        waste, y, c0, c1, h, cols = best
        y1 = y + h - 1
        column_rows = self._trim(unit, cols, y, y1)
        self._claim({t: (y, y + rows - 1) for t, rows in column_rows.items()})
        cap = self.columns_capacity(cols, y, y1, column_rows)
        return Placement(
            name=unit.name,
            slr=self.slr,
            status="placed",
            c0=c0,
            c1=c1,
            y0=y,
            y1=y1,
            ranges=self.ranges(cols, y, y1, column_rows),
            capacity={
                "luts": cap["slices"] * SLICE_LUTS,
                "ffs": cap["slices"] * SLICE_FFS,
                "carries": cap["slices"] * SLICE_CARRY,
                "lutrams": self.memory_luts(cap["slicem"]),
                "brams": cap["RAMB36"],
                "dsps": cap["DSP48E2"],
                "urams": cap["URAM288"],
            },
            waste=waste,
            columns=cols,
            column_rows=column_rows,
        )

    def _trim(self, unit: Unit, cols: list[int], y0: int, y1: int) -> dict[int, int]:
        """SLICE rows each claimed column really needs: full height, or less for sparse kinds.

        A unit's block RAM, DSP or URAM range need not span its SLICE height: a
        DSP layer that needs 317 of the 384 DSPs its four columns offer over 240
        rows releases the top of the last column. The trimmed height is a
        tile-pitch multiple so the kind rows stay tile-aligned. Columns of a kind
        the unit chains on are not trimmed, because chain segments were sized on
        the full window.
        """
        h = y1 - y0 + 1
        rows = dict.fromkeys(cols, h)
        if not TRIM_SPARSE_COLUMNS:
            return rows
        for kind in SPARSE_KINDS:
            key = next(k for k, kd in KIND_FOR_DEMAND.items() if kd == kind)
            need = unit.demands[key]
            if need <= 0 or unit.chain_lengths(kind):
                continue
            win = self._kind_window(kind, y0, y1)
            if win is None:
                continue
            kind_rows = win[1] - win[0] + 1
            for t in cols:
                for col in self.by_tile[t]:
                    if col.kind != kind:
                        continue
                    avail = col.available(*win)
                    if need >= avail:
                        need -= avail
                        continue
                    if need <= 0:
                        continue
                    # Rows the remaining demand needs, as whole tiles above y0.
                    first_tile = -(-y0 // TILE_PITCH) * TILE_PITCH
                    inner = h - (first_tile - y0)
                    tiles = -(-math.ceil(need / kind_rows * inner) // TILE_PITCH) * TILE_PITCH
                    rows[t] = min(rows[t], max(1, (first_tile - y0) + tiles))
                    need = 0
        return rows

    @staticmethod
    def _gap(c0: int, c1: int, y0: int, y1: int, anchor: Placement) -> int:
        """Manhattan gap in tiles between a rectangle and the anchor's (0 if they touch)."""
        dx = max(0, anchor.c0 - c1 - 1, c0 - anchor.c1 - 1)
        dy = max(0, anchor.y0 - y1 - 1, y0 - anchor.y1 - 1)
        return dx + dy

    def _search(
        self, unit: Unit, spare_slicem: bool, anchors: list[Placement] | None = None
    ) -> tuple[tuple[int, int, int, int, int, list[int]] | None, dict[str, int]]:
        """Best (waste, y, c0, c1, h, columns) over all heights and starts, or None.

        For every candidate height, every start column (lowest free row first) and
        every free interval of that column tall enough, the scan walks rightwards
        one tile column at a time and claims a column only if it is free over the
        rows and the unit still needs its kind (`_wants`), accumulating sites
        incrementally; a column another unit holds over these rows is left to it.
        The first feasible selection at a start is the narrowest one there. Also
        returns the last shortfall seen, for the "starved" diagnosis.

        Candidates rank by half-perimeter -- tile columns spanned plus rows, the
        longest Manhattan wire a path inside the unit can need -- then by SLICE
        waste, then bottom-left. Ranking by waste alone took the first zero-waste
        shape, which at the lowest candidate height is a strip: on mobilenetv1
        (13 Oct 2026) the placed units averaged an aspect ratio of 13:1 against
        2.7:1 when the placer chose, and 575 of the 617 worst intra-unit paths
        ran through strips with aspect > 8. A scan stops as soon as the width it
        has reached can no longer beat the best half-perimeter, and heights no
        shorter than the best half-perimeter are skipped.

        With ``anchors`` (the rectangles of the placed dataflow neighbours) the
        gaps to them are added to the half-perimeter (`LOCALITY_WEIGHT`), and the
        columns around each anchor join the start columns, so a unit settles
        beside or on top of the units it exchanges streams with unless that costs
        more shape than it saves wire. Without them (area order, or the first
        unit of an SLR) units went wherever the skyline was lowest: on mobilenetv1
        no two consecutive units touched and their rectangles were 185 tiles apart
        on average. Both neighbours count because chain carriers are packed
        first (`order_units`): with the predecessor alone, the units between two
        carriers strung along from the earlier one, and mobilenetv1's unit 220 (a
        FIFO) ended 591 tiles from the window buffer it feeds, whose combinational
        ready path came back through it with 1.0 ns of slack (rebuild6, 14 Oct
        2026).
        """
        needed = unit.needed_slices(self.slicem_util, self.srl_per_slice, self.logic_util)
        aligned = unit.uses_sparse()
        # Start at columns the unit would claim first: the lowest few of every
        # kind it needs, so a chain carrier can start at a block RAM column that
        # lies left of the lowest SLICE column, plus the lowest usable columns
        # overall.
        blank = self._blank()
        usable = [
            t
            for t in self.tile_cols
            if any(self._wants(unit, c, blank, {}, spare_slicem) for c in self.by_tile[t])
        ]
        per_kind: dict[str, list[int]] = {}
        for t in usable:
            for col in self.by_tile[t]:
                if self._wants(unit, col, blank, {}, spare_slicem):
                    per_kind.setdefault(col.kind, []).append(t)
        tops = self.top
        starts = set(sorted(usable, key=lambda t: (tops[t], t))[:MAX_START_POSITIONS])
        for cols_of_kind in per_kind.values():
            starts.update(sorted(cols_of_kind, key=lambda t: (tops[t], t))[:8])
        anchors = [a for a in anchors or [] if a.status == "placed"]
        for anchor in anchors:
            lo, hi = anchor.c0 - ANCHOR_REACH, anchor.c1 + ANCHOR_REACH
            starts.update(t for t in self.tile_cols if lo <= t <= hi)
        ordered = sorted(starts or set(self.tile_cols), key=lambda t: (tops[t], t))
        best, last_missing = self._scan(unit, spare_slicem, anchors, needed, aligned, ordered)
        if best is None and len(ordered) < len(self.tile_cols):
            # The start columns above are a shortlist for speed. A unit with many
            # chains can find its contiguous block RAM rows only from a column the
            # shortlist skipped once the columns around the skyline's low points
            # are fragmented (mobilenetv1's unit 266, 28 chains of 4, 13 Oct 2026),
            # so before giving up every remaining column is tried.
            rest = [
                t for t in sorted(self.tile_cols, key=lambda t: (tops[t], t)) if t not in starts
            ]
            best, missing = self._scan(unit, spare_slicem, anchors, needed, aligned, rest)
            last_missing = last_missing or missing
        return best, last_missing

    def _scan(
        self,
        unit: Unit,
        spare_slicem: bool,
        anchors: list[Placement],
        needed: int,
        aligned: bool,
        ordered: list[int],
    ) -> tuple[tuple[int, int, int, int, int, list[int]] | None, dict[str, int]]:
        """Run the height/start/row search of `_search` over the given start columns."""
        best: tuple[int, int, int, int, int, list[int]] | None = None
        best_key: tuple[int, int, int, int] | None = None
        last_missing: dict[str, int] = {}
        n_tiles = len(self.tile_cols)
        for h in self.heights(unit):
            if best_key is not None and h >= best_key[0]:
                break
            for start in ordered:
                for y in self._bottoms(start, h, aligned):
                    y1 = y + h - 1
                    cap = self._blank()
                    segs: dict[str, list[int]] = {}
                    rows: dict[str, tuple[int, int] | None] = {}
                    claimed: list[int] = []
                    j = self.tile_cols.index(start)
                    while j < n_tiles:
                        t = self.tile_cols[j]
                        j += 1
                        # Wider than the best half-perimeter allows: nothing from
                        # this start at this height can win any more.
                        if best_key is not None and (t - start + 1) + h >= best_key[0]:
                            break
                        # A column already holding another unit over these rows is
                        # left to its owner; the scan simply continues to the right.
                        if not self._free_over(t, y, y1):
                            continue
                        cols_here = self.by_tile[t]
                        if not any(
                            self._wants(unit, c, cap, segs, spare_slicem) for c in cols_here
                        ):
                            continue
                        claimed.append(t)
                        for col in self.by_tile[t]:
                            self._add_column(col, y, y1, cap, segs, rows)
                        missing = self._missing(unit, cap, segs)
                        if not missing:
                            waste = cap["slices"] - needed
                            c0, c1 = min(claimed), max(claimed)
                            cost = (c1 - c0 + 1) + h
                            if anchors:
                                # The longest of the streams to the neighbours is what
                                # bounds timing, so the widest gap is charged, not the
                                # sum: the sum is the same anywhere on the line between
                                # two anchors and so never pulls a unit between them.
                                cost += int(
                                    LOCALITY_WEIGHT
                                    * max(self._gap(c0, c1, y, y1, a) for a in anchors)
                                )
                            key = (cost, waste, y, c0)
                            if best_key is None or key < best_key:
                                best = (waste, y, c0, c1, h, claimed)
                                best_key = key
                            break
                        last_missing = missing
        return best, last_missing

    def _nearest_row(self, kind: str, y0: int, y1: int) -> tuple[int, int] | None:
        """One row of ``kind`` next to a SLICE window that holds none, for a borrowed range."""
        kb = self.census.bands.get(kind) or []
        if not kb:
            return None
        above = self.census.map_rows(kind, y1 + 1, y1 + TILE_PITCH)
        if above:
            return (above[0], above[0])
        below = self.census.map_rows(kind, max(self.y_lo, y0 - TILE_PITCH), y0 - 1)
        if below:
            return (below[1], below[1])
        return (kb[0][0], kb[0][0])

    def columns_capacity(
        self, cols: list[int], y0: int, y1: int, column_rows: dict[int, int] | None = None
    ) -> dict[str, int]:
        """Sites of every kind inside the given tile columns, each over its own rows."""
        cap = self._blank()
        for t in cols:
            top = y0 + (column_rows or {}).get(t, y1 - y0 + 1) - 1
            segs: dict[str, list[int]] = {}
            rows: dict[str, tuple[int, int] | None] = {}
            for col in self.by_tile[t]:
                self._add_column(col, y0, top, cap, segs, rows)
        return cap

    # -- ranges ------------------------------------------------------------------

    def ranges(
        self, cols: list[int], y0: int, y1: int, column_rows: dict[int, int] | None = None
    ) -> list[str]:
        """Vivado range strings for a column selection, per site kind.

        Claimed columns of one kind that are consecutive in the kind's own X and
        share a row window become one range; gaps (columns left to neighbours) or
        a trimmed column end a range. Every kind is ranged, because inside a
        reconfigurable partition Vivado requires it (HDPR-18) even for kinds the
        unit does not use: a kind with no claimed column borrows the nearest
        column of that kind, which may overlap a neighbour -- legal for a kind the
        unit has no demand for.
        """
        out: list[str] = []
        claimed = set(cols)
        column_rows = column_rows or {}
        h = y1 - y0 + 1
        for kind in KINDS:
            inside = sorted(
                (c for c in self.cols if c.kind == kind and c.tile_col in claimed),
                key=lambda c: c.x,
            )
            borrowed = False
            if not inside:
                all_kind = [c for c in self.cols if c.kind == kind]
                if not all_kind:
                    continue
                centre = (min(cols) + max(cols)) / 2
                inside = [min(all_kind, key=lambda c: abs(c.tile_col - centre))]
                borrowed = True
            # Per row window, the kind-X columns it covers; consecutive columns
            # with the same window merge into one range.
            by_win: dict[tuple[int, int], list[int]] = {}
            for c in inside:
                top = y0 + (h if borrowed else column_rows.get(c.tile_col, h)) - 1
                # A borrowed range only has to exist (HDPR-18); it may overlap a
                # neighbour, so it needs no tile alignment and must never be empty.
                if borrowed:
                    win = self.census.map_rows(kind, y0, top) or self._nearest_row(kind, y0, top)
                    wins = [win] if win else []
                else:
                    win = self._kind_window(kind, y0, top)
                    # A claimed range covers only rows the census has as available:
                    # a shell-clipped column (SLR0's rightmost block RAM columns on
                    # the U55C keep 12 of 48 rows) must not be ranged into the
                    # static region, where Vivado then counts the sites against
                    # the pblock and the diagnosis of a rejection gets muddied.
                    wins = [] if win is None else _clip_to_runs(win, c.runs)
                for w in wins:
                    by_win.setdefault(w, []).append(c.x)
            for win in sorted(by_win):
                xs = sorted(by_win[win])
                x0 = x1 = xs[0]
                for x in [*xs[1:], None]:
                    if x is not None and x == x1 + 1:
                        x1 = x
                        continue
                    out.append(f"{kind}_X{x0}Y{win[0]}:{kind}_X{x1}Y{win[1]}")
                    if x is not None:
                        x0 = x1 = x
        return out


def _clip_to_runs(win: tuple[int, int], runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Intersect an inclusive row window with a column's available runs."""
    out: list[tuple[int, int]] = []
    for a, b in runs:
        lo, hi = max(a, win[0]), min(b, win[1])
        if hi >= lo:
            out.append((lo, hi))
    return out


def pack(
    census: Census,
    units: list[Unit],
    previous: dict[str, Placement] | None = None,
    feedback: dict[str, dict[str, int]] | None = None,
    round_no: int = 1,
    order: str = "dataflow",
    slicem_util: float = DEFAULT_SLICEM_UTILIZATION,
    srl_per_slice: float = DEFAULT_SRL_PER_SLICE,
    logic_util: float = DEFAULT_LOGIC_UTILIZATION,
) -> dict[str, Placement]:
    """Pack every unit; in later rounds re-pack only the units the hook rejected.

    Args:
        census: The device as measured by the hook.
        units: Units in dataflow order.
        previous: Last round's placements. Units without feedback keep theirs and
            pin the skyline; units with feedback are packed again around them.
        feedback: Per unit, the measured shortfall per demand key (``luts`` ...)
            or ``chains:<KIND>`` -- added to the demand, or raising the minimum
            height by one clock region for a chain failure.
        round_no: Recorded on every placement made in this call.
        order: ``"dataflow"`` (the default) keeps graph order, so every unit is
            packed after the unit it receives its stream from and can settle
            next to it (`_neighbours`); measured 13 Oct 2026 on the fixtures,
            consecutive units end up 3-6 tiles apart with 82 % of them touching,
            against 170 tiles and 11 % in area order. ``"area"`` places the units
            with the largest footprint first, which was the default before the
            half-perimeter objective made shapes compact enough for dataflow
            order to place every unit as well.
        slicem_util: Fraction of SLICEM LUTs that distributed RAM may fill.
        srl_per_slice: Measured shift registers per SLICEM slice (see
            `DEFAULT_SRL_PER_SLICE`).
        logic_util: Fraction of a rectangle's slices plain logic may fill (see
            `DEFAULT_LOGIC_UTILIZATION`).
    """
    previous = previous or {}
    feedback = feedback or {}
    result: dict[str, Placement] = {}
    redo: list[Unit] = []
    for unit in units:
        if (
            unit.name in feedback
            or unit.name not in previous
            or previous[unit.name].status != "placed"
        ):
            fb = feedback.get(unit.name, {})
            for key, extra in fb.items():
                if key.startswith("chains:"):
                    band = census.bands.get("SLICE") or [(0, 59)]
                    unit.min_height = max(
                        unit.min_height,
                        _band_height(band)
                        + (
                            previous[unit.name].y1 - previous[unit.name].y0 + 1
                            if unit.name in previous and previous[unit.name].status == "placed"
                            else 0
                        ),
                    )
                elif key in unit.demands:
                    unit.demands[key] += int(extra)
            redo.append(unit)
        else:
            result[unit.name] = previous[unit.name]
    by_slr: dict[int, list[Unit]] = {}
    for unit in order_units(redo, order):
        by_slr.setdefault(unit.slr, []).append(unit)
    for slr, slr_units in sorted(by_slr.items()):
        if slr not in census.columns:
            for unit in slr_units:
                result[unit.name] = Placement(
                    unit.name, slr, "unplaced", reason="SLR not in census"
                )
            continue
        pinned = [p for p in result.values() if p.slr == slr]
        packer = SlrPacker(
            census,
            slr,
            pinned,
            slicem_util=slicem_util,
            srl_per_slice=srl_per_slice,
            logic_util=logic_util,
        )
        by_name = {u.name: u for u in units}
        for unit in slr_units:
            placement = packer.place(unit, anchors=_neighbours(unit, result, by_name))
            placement.round = round_no
            result[unit.name] = placement
    return result


def _neighbours(
    unit: Unit, result: dict[str, Placement], by_name: dict[str, Unit]
) -> list[Placement]:
    """Placed rectangles of the unit's dataflow neighbours, in any SLR.

    The nearest earlier placed unit (in dataflow order the one whose stream this
    unit receives; in area order whichever earlier unit is placed, which still
    pulls the unit towards its own part of the graph), and the unit directly
    after it when that is already placed: chain carriers go first
    (`order_units`), so the unit feeding a carrier is packed after it. Only the
    direct successor counts -- anchoring every unit to the nearest later carrier
    pulled whole runs of units towards it and, measured on the fixtures, raised
    the mean half-perimeter by a fifth and the tall units by half.

    Neighbours in another SLR count too. The SLRs are packed one after the other,
    so the two FIFOs of an SLR crossing were placed by two packers that knew
    nothing of each other: on mobilenetv1 one sat 180 rows above the boundary and
    the other 65 below it, 32 columns apart (dy 240; the registered stream between
    them was 1.33 ns short at 200 MHz, 15 Oct 2026). With the partner across the
    boundary as an anchor, the gap pulls the crossing unit to the boundary rows and
    to the partner's columns; `_gap` is plain Manhattan distance, so nothing else
    changes.
    """
    before: tuple[int, Placement] | None = None
    after: Placement | None = None
    for name, p in result.items():
        u = by_name.get(name)
        if u is None or p.status != "placed":
            continue
        if u.order < unit.order and (before is None or u.order > before[0]):
            before = (u.order, p)
        if u.order == unit.order + 1:
            after = p
    return [x for x in (before[1] if before else None, after) if x is not None]


def footprint(unit: Unit) -> int:
    """Rough tile area a unit needs: SLICEs plus the die-ratio cost of its sparse kinds.

    A block RAM column holds one RAMB36 per five SLICE rows and a DSP column
    about one DSP48E2 per two and a half, so a unit's sparse demand translates
    into SLICE-row-equivalents that are far larger than its LUT count suggests.
    Chains add their rows. Used only for ordering, never for sizing.
    """
    d = unit.demands
    area = unit.needed_slices()
    area += d["brams"] * 5 + d["dsps"] * 3 + d["urams"] * 4
    area += sum(sum(v) for v in unit.chains.values()) * 5
    return area


def order_units(units: list[Unit], order: str) -> list[Unit]:
    """Order units for packing; see `pack` for the two policies.

    In dataflow order the units that carry cascade chains go first, the most
    chain rows first: a chain needs contiguous rows of one column inside one
    clock region, which everything packed before it fragments. mobilenetv1's
    unit 266 (28 block RAM chains of 4) found no such rows in its SLR when its
    turn came in plain dataflow order, and unit 278 (eleven chains of 8, 125
    block RAMs), ranked by footprint behind a few LUT-heavy carriers, was left
    the three untouched block RAM columns of SLR0 -- 500 tile columns apart
    (13 Oct 2026). Ranked by chain rows it takes adjacent columns, and the
    units after it still settle next to their predecessors.
    """
    if order == "dataflow":
        chained = sorted(
            (u for u in units if u.chains),
            key=lambda u: (-sum(sum(v) for v in u.chains.values()), -footprint(u), u.order),
        )
        rest = sorted((u for u in units if not u.chains), key=lambda u: u.order)
        return chained + rest
    if order == "area":
        return sorted(units, key=lambda u: (-footprint(u), u.order))
    raise ValueError(f"unknown packing order {order!r}")


def _band_height(bands: list[tuple[int, int]]) -> int:
    """Return the row height of the first band (all bands of a kind are equal)."""
    return bands[0][1] - bands[0][0] + 1


def slr_budget(
    census: Census,
    units: list[Unit],
    slicem_util: float = DEFAULT_SLICEM_UTILIZATION,
    srl_per_slice: float = DEFAULT_SRL_PER_SLICE,
    logic_util: float = DEFAULT_LOGIC_UTILIZATION,
) -> dict[int, dict[str, float]]:
    """Demand over available capacity per SLR and resource, before any packing.

    This is the check the SLR assignment cannot make for itself today, because it
    works from estimates that carry no memory LUTs: on resnet18 the measured
    LUTRAM and shift registers came to 96 %, 103 % and 95 % of the three SLRs'
    SLICEM ceilings once the static shell's sites were excluded. No packer places
    103 %; the number has to go back to the assignment, not into a retry loop.
    """
    out: dict[int, dict[str, float]] = {}
    for slr in census.slrs:
        cols = census.columns.get(slr, {})

        def rows(kind: str, cols: dict[str, list[Column]] = cols) -> int:
            """Available rows of a kind in this SLR."""
            return sum(b - a + 1 for c in cols.get(kind, []) for a, b in c.runs)

        slicem_rows = sum(b - a + 1 for c in cols.get("SLICE", []) if c.slicem for a, b in c.runs)
        cap = {
            "luts": rows("SLICE") * SLICE_LUTS,
            "ffs": rows("SLICE") * SLICE_FFS,
            "slicem": slicem_rows,
            "brams": rows("RAMB36"),
            "dsps": rows("DSP48E2"),
            "urams": rows("URAM288"),
        }
        mine = [u for u in units if u.slr == slr]
        dem = {
            "luts": sum(
                u.slices_needed(slicem_util, srl_per_slice, logic_util) * SLICE_LUTS for u in mine
            ),
            "ffs": sum(u.demands["ffs"] for u in mine) / logic_util,
            "slicem": sum(u.slicem_slices(slicem_util, srl_per_slice) for u in mine),
            "brams": sum(u.demands["brams"] for u in mine),
            "dsps": sum(u.demands["dsps"] for u in mine),
            "urams": sum(u.demands["urams"] for u in mine),
        }
        out[slr] = {
            k: (dem[k] / cap[k] if cap[k] else (float("inf") if dem[k] else 0.0)) for k in cap
        }
    return out


def summarize(
    result: dict[str, Placement],
    units: list[Unit],
    census: Census | None = None,
    slicem_util: float = DEFAULT_SLICEM_UTILIZATION,
    srl_per_slice: float = DEFAULT_SRL_PER_SLICE,
    logic_util: float = DEFAULT_LOGIC_UTILIZATION,
) -> dict:
    """Totals worth printing into the Vivado log, plus the per-SLR budget check."""
    placed = [p for p in result.values() if p.status == "placed"]
    by_name = {u.name: u for u in units}
    claimed = sum(p.capacity.get("luts", 0) for p in placed)
    demanded = sum(by_name[p.name].demands["luts"] for p in placed if p.name in by_name)
    budget = (
        slr_budget(census, units, slicem_util, srl_per_slice, logic_util)
        if census is not None
        else {}
    )
    oversubscribed = {
        f"SLR{slr}": {
            k: (v if v == float("inf") else round(v, 2)) for k, v in ratios.items() if v > 0.9
        }
        for slr, ratios in budget.items()
        if any(v > 0.9 for v in ratios.values())
    }
    spans = [((p.c1 - p.c0 + 1), (p.y1 - p.y0 + 1)) for p in placed]
    aspects = [max(w, h) / min(w, h) for w, h in spans]
    return {
        "units": len(units),
        "placed": len(placed),
        "unplaced": sorted(p.name for p in result.values() if p.status != "placed"),
        "luts_claimed": claimed,
        "luts_demanded": demanded,
        # Shape of the placed rectangles: tile columns plus rows (the longest wire a
        # path inside the unit can need) and the longer side over the shorter.
        "half_perimeter_mean": round(sum(w + h for w, h in spans) / len(spans), 1) if spans else 0,
        "aspect_mean": round(sum(aspects) / len(aspects), 2) if aspects else 0,
        "aspect_max": round(max(aspects), 1) if aspects else 0,
        "reasons": {p.name: p.reason for p in result.values() if p.status != "placed"},
        "slr_budget": {
            f"SLR{k}": {r: round(v, 3) for r, v in d.items()} for k, d in budget.items()
        },
        "oversubscribed": oversubscribed,
    }


def write_result(
    path: Path, tcl_path: Path | None, result: dict[str, Placement], summary: dict
) -> None:
    """Write the JSON result and, if asked, the Tcl dict the hook sources."""
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "units": {k: v.to_dict() for k, v in result.items()},
                "summary": summary,
            },
            indent=1,
        )
    )
    if tcl_path is not None:
        # Fully qualified: the shell sources this file from inside a proc, and a
        # plain `set` there would land in that proc's local scope.
        lines = ["set ::finn_fp_result [dict create]"]
        for name, p in result.items():
            lines.append(f"dict set ::finn_fp_result {name} {p.to_tcl()}")
        tcl_path.write_text("\n".join(lines) + "\n")


def load_previous(path: Path | None) -> dict[str, Placement]:
    """Read a previous round's result JSON back into placements."""
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text())
    out: dict[str, Placement] = {}
    for name, d in data.get("units", {}).items():
        out[name] = Placement(
            name=name,
            slr=int(d["slr"]),
            status=str(d["status"]),
            c0=int(d.get("tile_col0", -1)),
            c1=int(d.get("tile_col1", -1)),
            y0=int(d.get("slice_y0", -1)),
            y1=int(d.get("slice_y1", -1)),
            ranges=list(d.get("ranges", [])),
            capacity={k: int(v) for k, v in d.get("capacity", {}).items()},
            waste=int(d.get("waste_slices", 0)),
            reason=str(d.get("reason", "")),
            round=int(d.get("round", 1)),
            columns=[int(c) for c in d.get("columns", [])],
            column_rows={int(k): int(v) for k, v in d.get("column_rows", {}).items()},
        )
    return out


def main(argv: list[str] | None = None) -> int:
    """Command-line entry: pack units.json against census.json."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--census", required=True, type=Path)
    ap.add_argument("--units", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--out-tcl", type=Path, default=None)
    ap.add_argument("--previous", type=Path, default=None)
    ap.add_argument("--feedback", type=Path, default=None)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--order", choices=("area", "dataflow"), default="dataflow")
    ap.add_argument("--slicem-util", type=float, default=DEFAULT_SLICEM_UTILIZATION)
    ap.add_argument("--srl-per-slice", type=float, default=DEFAULT_SRL_PER_SLICE)
    ap.add_argument("--logic-util", type=float, default=DEFAULT_LOGIC_UTILIZATION)
    args = ap.parse_args(argv)

    census = Census.from_dict(json.loads(args.census.read_text()))
    units = [Unit.from_dict(d) for d in json.loads(args.units.read_text())]
    feedback = (
        json.loads(args.feedback.read_text()) if args.feedback and args.feedback.is_file() else {}
    )
    previous = load_previous(args.previous)
    result = pack(
        census,
        units,
        previous=previous,
        feedback=feedback,
        round_no=args.round,
        order=args.order,
        slicem_util=args.slicem_util,
        srl_per_slice=args.srl_per_slice,
        logic_util=args.logic_util,
    )
    summary = summarize(
        result, units, census, args.slicem_util, args.srl_per_slice, args.logic_util
    )
    write_result(args.out, args.out_tcl, result, summary)
    for slr, ratios in summary["oversubscribed"].items():
        # A kind the SLR has no sites of at all shows as infinity (a part without URAM).
        over = ", ".join(
            f"{k} {'no sites' if v == float('inf') else f'{int(v * 100)}%'}"
            for k, v in ratios.items()
        )
        print(
            f"WARNING: {slr} demand is above 90% of its capacity ({over}); this is the SLR "
            "assignment's input, not a packing problem -- expect unplaced units there"
        )
    print(
        f"packer round {args.round}: {summary['placed']} of {summary['units']} unit(s) placed; "
        f"LUTs claimed {summary['luts_claimed']} for {summary['luts_demanded']} demanded"
    )
    for name in summary["unplaced"]:
        print(f"  unplaced: {name} ({summary['reasons'][name]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

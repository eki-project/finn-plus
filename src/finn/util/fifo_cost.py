# Copyright (C) 2026, Paderborn University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""FPGA primitive cost of the consolidated RTL FIFO, for every FIFO sizing method.

Every FIFO sizing method of FINN+ (the distributed RTL simulation, the TEG based abstract
simulation and MILP, the live sizing on the board) trades FIFO depth against FPGA resources,
so they all need one answer to "what does a FIFO of depth *d* and width *W* cost once
``finn-rtllib/fifo/hdl/fifo.sv`` has built it?". This module gives that answer as a vector of
primitives (:class:`FIFOCost`), the depths at which that vector changes
(:func:`cost_breakpoints`: the candidate depths of any search, since no depth in between can be
cheaper) and one documented way of collapsing the vector into a scalar (:func:`scalarize`).

Sources
-------
The storage style and the implemented capacity are read off ``fifo.sv`` (``RAM_STYLE_EFF``,
``INIT_ABITS``, ``MEM_IMPL``, ``PIPE_DEPTH``, ``QDEPTH`` and ``DEPTH_ACTUAL``); the Python mirror
of the style ladder lives in :mod:`finn.util.resource_models` (``_resolve``, guarded against the
RTL by ``tests/fpgadataflow/test_fifo_ram_style_mirror.py``) and is reused here, not restated.
The LUT, RAMB18 and URAM288 counts are the fitted model of ``resource_models._fifo_cost``
(Vivado 2026.1 runs over all styles, widths 1..256, depths 2..262145; tiles exact, LUTs within a
few percent), which ``StreamingFIFO.get_fifo_cost()`` also uses, so the sizing searches and the
resource estimation report of a build agree by construction. The flip-flop count is not fitted:
it is the register inventory of ``fifo.sv`` (see :func:`ff_estimate`), with the read pipeline
stages that the memory primitives absorb discounted.

Depth convention
----------------
``depth`` is always the ``DEPTH`` parameter of ``fifo.sv``, i.e. the ``depth`` attribute of the
``StreamingFIFO`` node and the value in ``fifo_config.json``. The live sizing driver's capacity
register is one less (``fifo_depth_offset``); its cost tables are evaluated at register plus
offset (see ``make_driver.MakePYNQDriver._write_fifo_cost_tables``).

Integration with ``fifo_depth_search`` (branch ``feature/fifo_sizing_model``)
----------------------------------------------------------------------------
The TEG branch factors the cost model and the candidate ladder of the distributed simulation
into ``finn/transformation/fpgadataflow/fifo_depth_search.py``. It can switch to this module
with the following mapping (``w`` the stream width, ``upper`` the known-good starting depth):

- ``calculate_srl16e_luts(d, w)``, ``calculate_bram_blocks(d, w)``, ``calculate_uram_blocks(d,
  w)`` and ``max_qsrl_depth`` → ``fifo_cost(d, w, style, versal)`` (vector) and
  ``scalarize(...)`` (scalar); the style comes from the node's ``ram_style`` and the ladder of
  ``fifo.sv`` decides, there is no fixed LUTRAM/BRAM boundary any more.
- ``calculate_*_depth_range`` / ``get_valid_block_counts`` / ``candidate_depths(w, upper,
  max_qsrl_depth)`` → ``cost_breakpoints(w, upper, style, versal)`` (every depth that is the
  largest of a constant-cost step, ``upper`` included); ``monotone_steps`` drops the steps a
  deeper one dominates under the chosen weights, which is what a search that may only move
  up wants.
- ``round_up_to_full_bram_block(d, w, max_qsrl_depth)`` → ``realize_depth(d, w, style,
  versal, weights)``: the largest depth with the cheapest cost among all depths ``>= d``.
- ``minimize_fifo_depth``'s fixed sequence "32, then 256, then BRAM block counts" becomes a
  binary search over ``cost_breakpoints`` (equal-cost depths need not be tested) or, with
  ``monotone_steps``, a search whose cost is non-decreasing in the candidate index.
- ``needs_minimization`` and ``effective_capacity`` are unchanged (capacity equals depth).

What changes in the results: (1) a FIFO narrower than 5 bits is LUTRAM at every depth in
``fifo.sv`` (``DATA_WIDTH < 5``), so the old "above 256 it is BRAM" rounding to 4096-entry
blocks is replaced by LUTRAM steps of powers of two; (2) depths above 2028 with ``auto`` are
URAM, which the old model never counted (its BRAM tiles above 2028 were fictitious; on a part
without URAM Vivado falls back to BRAM, see ``resource_models._resolve``); (3) the LUTRAM
tier has steps at every power of two of ``depth - 1`` rather than every 32 entries, and the
shift tier is only used up to depth 33 (64 for words narrower than 12 bits); (4) the BRAM tier
has sub-steps where ``fifo.sv`` splits a non-power-of-two depth into a lo and a hi memory
space, so a depth just above a power of two costs more LUTs (hi space in LUTRAM) or more tiles
than the power of two itself; the dominated steps are removed by ``monotone_steps``.
"""

import bisect
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from finn.util.resource_models import _clog2, _fifo_cost, _geometry, _resolve

#: Order of the primitives in the cost vector (and in the exported driver tables).
RESOURCES = ("lut", "ff", "bram18", "uram")

#: Storage styles a ``StreamingFIFO`` node may request; "auto" lets ``fifo.sv`` decide.
STYLES = ("auto", "srl", "distributed", "block", "ultra")

#: Smallest FIFO ``fifo.sv`` builds (``DEPTH < 2`` is an elaboration error).
MIN_DEPTH = 2

#: Bits of one RAMB18 and one URAM288 (the URAM is 16 RAMB18 worth of bits).
BRAM18_BITS = 18 * 1024
URAM_BITS = 72 * 4096

#: Fallback LUT equivalents of one RAMB18 / URAM288 for :func:`capacity_weights` when the
#: device has none of that primitive; see there.
URAM_PER_BRAM18 = URAM_BITS // BRAM18_BITS


@dataclass(frozen=True)
class FIFOCost:
    """Primitives one FIFO instance occupies: LUTs, flip-flops, RAMB18 tiles and URAM288s.

    A RAMB36 is two RAMB18 tiles (``bram36``). The vector is kept as is everywhere so that
    the way it is collapsed into one number (:func:`scalarize`) can be changed later.
    """

    lut: int = 0
    ff: int = 0
    bram18: int = 0
    uram: int = 0

    @property
    def bram36(self) -> float:
        """RAMB36 equivalents (two RAMB18 tiles each)."""
        return self.bram18 / 2

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """The vector in :data:`RESOURCES` order."""
        return (self.lut, self.ff, self.bram18, self.uram)

    def as_dict(self) -> Dict[str, int]:
        """The vector keyed by :data:`RESOURCES`."""
        return dict(zip(RESOURCES, self.as_tuple()))

    def __add__(self, other: "FIFOCost") -> "FIFOCost":
        """Component-wise sum (the cost of several FIFOs)."""
        return FIFOCost(*(a + b for a, b in zip(self.as_tuple(), other.as_tuple())))

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "FIFOCost":
        """Build from a sequence in :data:`RESOURCES` order."""
        return cls(*values)


@dataclass(frozen=True)
class FIFOGeometry:
    """How ``fifo.sv`` lays out a FIFO of a given depth, width and resolved style.

    All fields are the module's localparams of the same name:

    - ``style``: ``RAM_STYLE_EFF`` in FINN vocabulary (srl, distributed, block, ultra).
    - ``capacity``: ``DEPTH_ACTUAL``, the implemented capacity, which exceeds ``depth`` for
      shift registers below four stages and for memory backings rounded up to primitives.
    - ``count_width``: ``COUNT_WIDTH`` of the occupancy counter.
    - ``srl_depth`` / ``srl_stages``: ``DEPTH_IMPL`` of the shift path and the SRLC32E stages
      it needs (0 for the other styles).
    - ``lutram_rows``: ``MEM_SIZE`` of the distributed path (0 for the other styles).
    - ``abits_lo`` / ``abits_hi`` / ``mem_impl`` / ``pipe_depth`` / ``qdepth``: ``ABITS``,
      ``MEM_IMPL``, ``PIPE_DEPTH`` and ``QDEPTH`` of the memory path (0 for the other styles).
    """

    style: str
    depth: int
    width: int
    capacity: int
    count_width: int
    srl_depth: int = 0
    srl_stages: int = 0
    lutram_rows: int = 0
    abits_lo: int = 0
    abits_hi: int = 0
    mem_impl: int = 0
    pipe_depth: int = 0
    qdepth: int = 0

    @property
    def is_memory(self) -> bool:
        """Whether the FIFO is backed by BRAM or URAM."""
        return self.style in ("block", "ultra")


def _check(depth: int, width: int, style: str):
    """Validate the arguments common to every function here."""
    if style not in STYLES:
        raise ValueError(f"unknown ram_style {style!r}, expected one of {STYLES}")
    if width < 1:
        raise ValueError(f"width must be at least 1 bit, got {width}")
    if depth < 0:
        raise ValueError(f"depth must not be negative, got {depth}")


def resolve_style(depth: int, width: int, requested_style: str = "auto", versal: bool = False):
    """The storage style ``fifo.sv`` elaborates for a request (``RAM_STYLE_EFF``).

    Mirrors ``fifo.sv``'s ladder exactly: a depth of at most 33 is always a shift register
    (``srl``), then an explicit request is honoured, ``auto`` takes a shift register up to 64
    for words narrower than 12 bits, LUTRAM (``distributed``) up to 257 or for words narrower
    than 5 bits at any depth, BRAM (``block``) up to 2028 and URAM (``ultra``) beyond. The
    decision does not depend on the part (``fifo.sv`` cannot see it), ``versal`` is accepted
    for API symmetry with :func:`fifo_cost` only. Delegates to ``resource_models._resolve``,
    the single mirror of the RTL that is tested against ``fifo.sv``.
    """
    _check(depth, width, requested_style)
    return _resolve(depth, width, requested_style)


def geometry(depth: int, width: int, style: str = "auto", versal: bool = False) -> FIFOGeometry:
    """The ``fifo.sv`` layout of a FIFO: resolved style, capacity and memory decomposition.

    Reads off ``fifo.sv``: ``DEPTH_IMPL = DEPTH > 4 ? DEPTH-1 : 4`` for the shift path (one
    slot lives in the output register, minimum four stages), ``MEM_SIZE = 2**clog2(DEPTH-1)``
    for LUTRAM, and for the memory path ``DEPTH_REQ = DEPTH - QDEPTH - 1`` (``QDEPTH`` 16 for
    URAM, whose output queue holds that many, else 0), ``INIT_ABITS`` (a lo space of
    ``2**lo`` rows, optionally a hi space of ``2**hi`` rows when that saves a primitive),
    ``PIPE_DEPTH = 2`` for BRAM and ``3 + (2**lo - 1) // 8192`` for URAM, and
    ``DEPTH_ACTUAL = MEM_IMPL + PIPE_DEPTH + QDEPTH + 1``.
    """
    _check(depth, width, style)
    eff = resolve_style(depth, width, style, versal)
    count_width = _clog2(depth + 1) + 1
    if eff == "srl":
        srl_depth = depth - 1 if depth > 4 else 4
        return FIFOGeometry(
            style=eff,
            depth=depth,
            width=width,
            capacity=depth if depth > 4 else 5,
            count_width=count_width,
            srl_depth=srl_depth,
            srl_stages=math.ceil(srl_depth / 32),
        )
    if eff == "distributed":
        return FIFOGeometry(
            style=eff,
            depth=depth,
            width=width,
            capacity=depth,
            count_width=count_width,
            lutram_rows=2 ** _clog2(depth - 1),
        )
    is_uram = eff == "ultra"
    qdepth = 16 if is_uram else 0
    lo, hi = _geometry(depth, is_uram)
    mem_impl = 2**lo + (2**hi if hi else 0)
    pipe_depth = 3 + (2**lo - 1) // 8192 if is_uram else 2
    return FIFOGeometry(
        style=eff,
        depth=depth,
        width=width,
        capacity=mem_impl + pipe_depth + qdepth + 1,
        count_width=count_width,
        abits_lo=lo,
        abits_hi=hi,
        mem_impl=mem_impl,
        pipe_depth=pipe_depth,
        qdepth=qdepth,
    )


def ff_estimate(geom: FIFOGeometry) -> int:
    """Flip-flops of a FIFO: the register inventory of ``fifo.sv``, not fitted.

    Counted per style from the RTL (``W`` the width, ``cw`` the occupancy counter width):

    - every style: ``Count`` and ``MaxCount`` of the occupancy monitor, ``2 * cw``;
    - shift: ``Ptr`` (``clog2(DEPTH_IMPL) + 1`` bits), ``Rdy``, ``Vld`` and the output
      register ``B`` (``W``); the shift register itself is in SRLC32E LUTs;
    - distributed: ``WrPtr``/``RdPtr`` (``ABITS`` each), ``Credit``/``Fill``
      (``clog2(MEM_DEPTH) + 1`` each), ``Vld``, ``B`` (``W``); the array is LUTRAM;
    - block/ultra: ``WrPtr``/``RdPtr`` (``ABITS.lo + (ABITS.hi != 0)`` each), ``Credit``/
      ``Fill`` (``clog2(MEM_IMPL) + 1`` each), the ``OVld`` pipeline (``PIPE_DEPTH``) and,
      with a hi space, the ``OSelLo`` pipeline (``PIPE_DEPTH``) and its data pipeline. The
      ``ODatLo`` data pipeline is ``PIPE_DEPTH * W`` registers in the RTL, of which the
      primitive's own output register absorbs one stage for BRAM and the cascade and output
      registers two stages for URAM (``resource_models`` fits the further URAM stages as SRL
      LUTs, so they are not counted here); a hi space is charged one stage of ``W``;
    - ultra additionally: ``QCredit`` (``clog2(QDEPTH + 1) + 1``) and the output queue, a
      recursive depth-17 shift FIFO whose own occupancy monitor is unconnected and trimmed:
      ``Ptr`` (5), ``Rdy``, ``Vld``, ``B`` (``W``).
    """
    W = geom.width
    ff = 2 * geom.count_width
    if geom.style == "srl":
        return ff + (_clog2(geom.srl_depth) + 1) + 2 + W
    if geom.style == "distributed":
        abits = _clog2(geom.depth - 1)
        return ff + 2 * abits + 2 * (abits + 1) + 1 + W
    ptr_width = geom.abits_lo + (1 if geom.abits_hi else 0)
    ff += 2 * ptr_width + 2 * (_clog2(geom.mem_impl) + 1) + geom.pipe_depth
    absorbed = 2 if geom.style == "ultra" else 1
    ff += max(geom.pipe_depth - absorbed, 0) * W
    if geom.abits_hi:
        ff += geom.pipe_depth + W
    if geom.style == "ultra":
        ff += (_clog2(geom.qdepth + 1) + 1) + 5 + 2 + W
    return ff


def fifo_cost(depth: int, width: int, style: str = "auto", versal: bool = False) -> FIFOCost:
    """The primitives a ``fifo.sv`` instance of this depth, width and requested style costs.

    ``width`` is the padded stream width (a multiple of 8 for AXI streams,
    ``get_instream_width_padded()``), the width the RTL is instantiated with. LUT, RAMB18 and
    URAM288 counts are ``resource_models._fifo_cost`` evaluated at the style
    :func:`resolve_style` predicts, the flip-flops :func:`ff_estimate`. ``versal`` selects the
    Versal aspect tables and the Versal-specific LUT terms of the fitted model. A depth below
    :data:`MIN_DEPTH` costs nothing (``fifo.sv`` does not build it; ``set_fifo_depths`` leaves
    depth-0 placeholders on parameter inputs).
    """
    _check(depth, width, style)
    if depth < MIN_DEPTH:
        return FIFOCost()
    geom = geometry(depth, width, style, versal)
    fitted = _fifo_cost(depth, width, geom.style, versal)
    return FIFOCost(
        lut=int(fitted.lut), ff=ff_estimate(geom), bram18=int(fitted.bram), uram=int(fitted.uram)
    )


# --------------------------------------------------------------------------------------
# Breakpoints: the depths at which the cost vector changes
# --------------------------------------------------------------------------------------


def _change_candidates(width: int, max_depth: int, style: str, versal: bool) -> List[int]:
    """Every depth ``d`` at which ``fifo_cost(d) != fifo_cost(d - 1)`` can possibly hold.

    The fitted model and the RTL geometry depend on the depth only through a few integer
    functions, each of which changes value at depths of a known shape:

    - the style ladder (33/64/257/2028) and the shift-register floor at depth 5;
    - the occupancy counter width ``clog2(depth + 1) + 1``: at ``depth = 2**k``;
    - SRLC32E stages ``ceil((depth - 1) / 32)``: at ``depth = 32 * m + 2`` (requested
      ``srl`` only, since ``auto`` never shifts beyond 64);
    - LUTRAM rows ``2**clog2(depth - 1)``: at ``depth = 2**k + 2``;
    - the memory geometry ``INIT_ABITS`` of ``depth_req = depth - qdepth - 1`` (``qdepth``
      0 or 16): ``lo`` and ``hi`` change where ``depth_req`` passes ``2**k`` or
      ``2**k + 2**j`` (``j < k``) by one, and with them the tile plans, the hi-space LUTRAM
      fallback and the pipeline depth.

    Each shape is emitted with a margin of a few depths on either side; the superset is
    cheap (a few thousand depths) and the test suite checks it against a full scan.
    """
    cands = set(range(MIN_DEPTH, min(max_depth, 70) + 1))
    for threshold in (33, 64, 257, 2028):
        cands.update(range(threshold - 2, threshold + 4))
    top = max_depth.bit_length() + 1
    for k in range(top):
        bases = [2**k] + [2**k + 2**j for j in range(k)]
        for base in bases:
            for shift in (0, 1, 17):
                for off in range(-3, 4):
                    cands.add(base + shift + off)
    if style == "srl":
        cands.update(range(2, max_depth + 1, 32))
        cands.update(range(3, max_depth + 1, 32))
    return sorted(c for c in cands if MIN_DEPTH < c <= max_depth)


def cost_steps(
    width: int, max_depth: int, style: str = "auto", versal: bool = False
) -> List[Tuple[int, FIFOCost]]:
    """The cost ladder up to ``max_depth``: ``(largest depth of the step, cost)`` per step.

    Consecutive depths with the same cost vector form one step; the list is ascending in
    depth, starts with the step containing :data:`MIN_DEPTH` and ends with the step containing
    ``max_depth`` (whose recorded depth is ``max_depth`` itself, even if the step would extend
    beyond). Any search that may pick a depth freely within a step should pick the largest
    one: same cost, most slack.
    """
    _check(max_depth, width, style)
    if max_depth < MIN_DEPTH:
        return []
    steps: List[Tuple[int, FIFOCost]] = []
    current = fifo_cost(MIN_DEPTH, width, style, versal)
    for cand in _change_candidates(width, max_depth, style, versal):
        cost = fifo_cost(cand, width, style, versal)
        if cost != current:
            steps.append((cand - 1, current))
            current = cost
    steps.append((max_depth, current))
    return steps


def cost_breakpoints(
    width: int, max_depth: int, style: str = "auto", versal: bool = False
) -> List[int]:
    """The depths at which the cost vector changes: the largest depth of every step.

    This is the candidate set of a depth search: a depth strictly inside a step costs the
    same as the step's largest depth and is never preferable. ``max_depth`` is always the last
    entry (the search's known-good starting depth stays a candidate).
    """
    return [d for d, _ in cost_steps(width, max_depth, style, versal)]


# --------------------------------------------------------------------------------------
# Scalarization
# --------------------------------------------------------------------------------------


def capacity_weights(lut: int, ff: int, bram18: int, uram: int) -> Dict[str, float]:
    """Weights that express every primitive in LUT equivalents of a device's capacity.

    ``weight[r] = lut / capacity[r]``: a FIFO's scalar cost is then the sum over the
    primitives of the fraction of the device it occupies, in units of LUTs (so that a LUT has
    weight 1 and the numbers remain readable). A primitive the device does not have (``uram``
    of a Zynq-7000, where Vivado backs an ``ultra`` array with BRAM) is charged the BRAM bits
    it displaces: :data:`URAM_PER_BRAM18` RAMB18 tiles.
    """
    if lut <= 0 or ff <= 0 or bram18 <= 0:
        raise ValueError("a device needs LUTs, flip-flops and BRAM")
    weights = {"lut": 1.0, "ff": lut / ff, "bram18": lut / bram18}
    weights["uram"] = lut / uram if uram > 0 else URAM_PER_BRAM18 * weights["bram18"]
    return weights


#: Default weights of :func:`scalarize`: the capacity of a ZU7EV (ZCU104, the usual live
#: sizing target) from ``finn.util.platforms.ZU7EV_Platform``: 230000 LUTs, 460000 FFs,
#: 610 RAMB18, 92 URAM288, i.e. 1 RAMB18 = 377 LUTs and 1 URAM288 = 2500 LUTs.
#: :func:`device_weights` derives the same for another part.
DEFAULT_WEIGHTS: Dict[str, float] = capacity_weights(230000, 460000, 610, 92)


def _device_of_part(fpgapart: str) -> str:
    """The device of a part name (``xczu28dr-ffvg1517-2-e`` -> ``xczu28dr``)."""
    return fpgapart.split("-")[0].lower()


def device_weights(fpgapart: Optional[str] = None, board: Optional[str] = None):
    """Capacity weights of a target, from ``finn.util.platforms``; the defaults otherwise.

    Looks the board up in the platform table (``platforms.platforms``), or, failing that, any
    board there built on the same device as ``fpgapart`` (an RFSoC2x2 shares the ZU28DR of
    the ZCU111), and uses the compute resources of one SLR. A target without a table entry
    gets :data:`DEFAULT_WEIGHTS`. The result also carries the source of the numbers under
    ``"source"`` so a report can say which device the scalar cost refers to.
    """
    from finn.util.basic import part_map
    from finn.util.platforms import platforms

    platform_cls = platforms.get(board) if board is not None else None
    if platform_cls is None and fpgapart is not None:
        for other, other_part in part_map.items():
            if other in platforms and _device_of_part(other_part) == _device_of_part(fpgapart):
                board, platform_cls = other, platforms[other]
                break
    if platform_cls is None:
        weights = dict(DEFAULT_WEIGHTS)
        weights["source"] = "default (ZU7EV capacity)"
        return weights
    lut, ff, bram18, uram, _dsp = platform_cls().compute_resources[0]
    weights = capacity_weights(lut, ff, bram18, uram)
    weights["source"] = f"{board} capacity ({platform_cls.__name__})"
    return weights


def scalarize(cost: FIFOCost, weights: Optional[Dict[str, float]] = None) -> float:
    """Collapse a cost vector into one number: the weighted sum over the primitives.

    The default weighting (:data:`DEFAULT_WEIGHTS`) charges each primitive the fraction of
    the device it takes, scaled to LUT equivalents, which is the only consolidation that
    treats "one more URAM" and "300 more LUTs" as the comparable things they are on the
    device the design must fit on; a plain bit count (the live driver's previous metric)
    cannot see that a depth-2029 FIFO in one URAM is cheaper than a depth-2028 FIFO in eight
    RAMB18. The weights are explicit so that a search can use the capacity of its actual
    target (:func:`device_weights`) or, say, zero out flip-flops. The sum is additive over
    FIFOs, which the exact searches need for their incremental cost updates; the vector
    itself stays available for a different consolidation.
    """
    w = DEFAULT_WEIGHTS if weights is None else weights
    return sum(w.get(r, 0.0) * v for r, v in zip(RESOURCES, cost.as_tuple()))


def monotone_steps(
    steps: Sequence[Tuple[int, FIFOCost]], weights: Optional[Dict[str, float]] = None
) -> List[Tuple[int, FIFOCost]]:
    """The steps of a ladder that no deeper step undercuts under the given weights.

    The scalar cost of ``fifo.sv`` is not monotone in the depth (a URAM FIFO just above 2028
    can be cheaper than the BRAM FIFO just below, a power-of-two BRAM depth cheaper than the
    one with a small hi space above it). A sizing search whose feasibility is monotone in the
    depth can always replace a depth by any deeper one, so the cost it should minimise is the
    *envelope* ``min(cost(d') for d' >= d)``; its steps are exactly the undominated ones
    returned here (ascending depth, strictly ascending scalar cost). The search's candidate
    depths are these; :func:`realize_depth` maps any other depth onto them.
    """
    scalars = [scalarize(c, weights) for _, c in steps]
    kept: List[Tuple[int, FIFOCost]] = []
    best = math.inf
    for (depth, cost), scalar in zip(reversed(steps), reversed(scalars)):
        if scalar < best:
            kept.append((depth, cost))
            best = scalar
    kept.reverse()
    return kept


def realize_depth(
    depth: int,
    width: int,
    style: str = "auto",
    versal: bool = False,
    weights: Optional[Dict[str, float]] = None,
    max_depth: Optional[int] = None,
) -> int:
    """The depth to build for a FIFO that needs at least ``depth``: the cheapest at or above.

    Searches ``[depth, max_depth]`` (default: up to the next larger power of two times four,
    enough to reach the next URAM step) for the smallest scalar cost and returns the largest
    depth of that step, i.e. the depth with the most slack at that cost. Replaces the
    distributed simulation's rounding to full BRAM blocks.
    """
    if depth < MIN_DEPTH:
        return depth
    if max_depth is None:
        max_depth = 4 * 2 ** _clog2(depth + 1)
    steps = cost_steps(width, max_depth, style, versal)
    idx = bisect.bisect_left([d for d, _ in steps], depth)
    envelope = monotone_steps(steps[idx:], weights)
    return envelope[0][0] if envelope else depth


# --------------------------------------------------------------------------------------
# Export for consumers without finn (the PYNQ driver)
# --------------------------------------------------------------------------------------


def cost_table(width: int, max_depth: int, style: str = "auto", versal: bool = False) -> Dict:
    """The ladder of one FIFO as plain JSON data: ``{"width", "ram_style", "steps"}``.

    ``steps`` is ``[[largest depth of the step, [lut, ff, bram18, uram]], ...]`` in
    :data:`RESOURCES` order, as :func:`cost_steps` returns it. ``MakePYNQDriver`` writes one
    such table per FIFO into the driver's ``settings.json`` (``fifo_cost_tables``), where the
    live sizing driver, which runs without finn, looks its primitive cost up.
    """
    return {
        "width": width,
        "ram_style": style,
        "steps": [[d, list(c.as_tuple())] for d, c in cost_steps(width, max_depth, style, versal)],
    }

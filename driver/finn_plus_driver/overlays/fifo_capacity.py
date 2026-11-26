"""The capacity ``fifo.sv`` really builds for a requested FIFO depth.

The synthesizable FIFO of FINN (``finn-rtllib/fifo/hdl/fifo.sv``) selects its storage by depth
and width and rounds the capacity up to what that storage provides (its ``DEPTH_ACTUAL``):
a shift register never has fewer than five slots, a block RAM FIFO holds whole primitives plus
its read pipeline, a URAM FIFO at least 4096 rows plus its output queue. The live FIFO sizing
driver configures every virtual FIFO with this capacity, so that a candidate configuration is
evaluated as it will be built, and reports the storage of a configuration as the capacity
times the data width ("realized bits").

This module is a dependency-free mirror of the RTL geometry for the PYNQ driver, which runs
without finn. ``finn.util.fifo_cost.geometry`` is the other mirror; the finn test suite checks
that both agree, so keep the thresholds here in sync with ``fifo.sv`` and that module.
"""

from typing import List, Optional, Sequence

#: ``RAM_STYLE`` values of ``fifo.sv`` in FINN vocabulary (``srl`` is the RTL's ``shift``).
STYLES = ("auto", "srl", "distributed", "block", "ultra")
#: Smallest depth FINN requests.
MIN_DEPTH = 2
#: A depth up to this is a shift register whatever the requested style (``fifo.sv:85``).
SHIFT_MAX_DEPTH = 33
#: ``auto`` keeps a shift register up to this depth for words narrower than ``NARROW_WIDTH``.
NARROW_SHIFT_MAX_DEPTH = 64
NARROW_WIDTH = 12
#: ``auto`` takes LUTRAM up to this depth, and at any depth for words narrower than
#: ``LUTRAM_WIDTH`` bits.
LUTRAM_MAX_DEPTH = 257
LUTRAM_WIDTH = 5
#: ``auto`` takes block RAM up to this depth and URAM beyond.
BRAM_MAX_DEPTH = 2028
#: Smallest address space of a memory primitive: RAMB18 512 x 36, URAM288 4096 x 72.
BRAM_MIN_ABITS = 9
URAM_MIN_ABITS = 12
#: Slots of the output queue of a URAM FIFO.
URAM_QDEPTH = 16


def clog2(x: int) -> int:
    """``$clog2()`` as SystemVerilog defines it: 0 for ``x <= 1``."""
    return max(x - 1, 0).bit_length()


def resolve_style(depth: int, width: int, requested: str = "auto") -> str:
    """The storage ``fifo.sv`` elaborates (``RAM_STYLE_EFF``) for a requested style."""
    if requested not in STYLES:
        raise ValueError(f"unknown ram_style {requested!r}, expected one of {STYLES}")
    if depth <= SHIFT_MAX_DEPTH:
        return "srl"
    if requested != "auto":
        return requested
    if depth <= NARROW_SHIFT_MAX_DEPTH and width < NARROW_WIDTH:
        return "srl"
    if depth <= LUTRAM_MAX_DEPTH or width < LUTRAM_WIDTH:
        return "distributed"
    return "block" if depth <= BRAM_MAX_DEPTH else "ultra"


def _memory_geometry(depth: int, is_uram: bool):
    """``(lo, hi, depth_req)`` of the memory path: address bits of the lo/hi spaces."""
    min_abits, qdepth = (URAM_MIN_ABITS, URAM_QDEPTH) if is_uram else (BRAM_MIN_ABITS, 0)
    # one slot lives in the output register, QDEPTH more in the URAM output queue
    depth_req = max(depth - qdepth - 1, 1)
    lo = clog2(depth_req)
    hi = 0
    if lo > min_abits:
        hi_abits = clog2(depth_req - 2 ** (lo - 1))
        if hi_abits < lo - 1:
            lo -= 1
            hi = max(hi_abits, 1)
    return lo, hi, depth_req


def capacity(depth: int, width: int, requested: str = "auto") -> int:
    """The number of items a ``fifo.sv`` instance of the requested depth really holds.

    ``DEPTH_ACTUAL`` of the RTL: ``max(depth, 5)`` for a shift register (four SRL stages at
    least plus the output register), ``depth`` for LUTRAM, and for block RAM and URAM the
    implemented address space (``2**lo + 2**hi``) plus the read pipeline (2 for BRAM, 3 or more
    for URAM), the URAM output queue and the output register.
    """
    if depth < MIN_DEPTH:
        raise ValueError(f"depth must be at least {MIN_DEPTH}, got {depth}")
    style = resolve_style(depth, width, requested)
    if style == "srl":
        return depth if depth > 4 else 5
    if style == "distributed":
        return depth
    is_uram = style == "ultra"
    lo, hi, _ = _memory_geometry(depth, is_uram)
    mem_impl = 2**lo + (2**hi if hi else 0)
    pipe_depth = 3 + (2**lo - 1) // 8192 if is_uram else 2
    return mem_impl + pipe_depth + (URAM_QDEPTH if is_uram else 0) + 1


def plateau_top(depth: int, width: int, requested: str = "auto") -> int:
    """The largest depth that builds the same capacity as ``depth``.

    Every depth of a plateau produces the same FIFO, so a search need only consider the
    plateau tops, and the top is the depth to request for a found configuration (it says
    most clearly what gets built). Shift registers below five stages share the plateau of
    depth 5, LUTRAM depths are their own plateau, and a memory plateau spans the depths that
    share the address-space decomposition.
    """
    cap = capacity(depth, width, requested)
    style = resolve_style(depth, width, requested)
    if style == "srl":
        top = max(depth, 5)
    elif style == "distributed":
        top = depth
    else:
        is_uram = style == "ultra"
        lo, hi, _ = _memory_geometry(depth, is_uram)
        # the decomposition (lo, hi) is constant up to a request of 2**lo + 2**hi rows
        top = 2**lo + (2**hi if hi else 0) + (URAM_QDEPTH if is_uram else 0) + 1
        if not is_uram and requested == "auto":
            top = min(top, BRAM_MAX_DEPTH)
    top = max(top, depth)
    # guard the analytic top against the RTL ladder itself
    while (
        capacity(top + 1, width, requested) == cap
        and resolve_style(top + 1, width, requested) == style
    ):
        top += 1
    while capacity(top, width, requested) != cap:
        top -= 1
    return top


def plateau_tops(width: int, max_depth: int, requested: str = "auto") -> List[int]:
    """Ascending tops of the capacity plateaus up to ``max_depth`` (which is the last entry)."""
    tops: List[int] = []
    depth = MIN_DEPTH
    while depth <= max_depth:
        top = min(plateau_top(depth, width, requested), max_depth)
        tops.append(top)
        depth = top + 1
    return tops


def identity_above_shift(width: int, requested: str = "auto") -> bool:
    """Whether every depth above the shift-register floor is its own plateau for this FIFO.

    True for shift registers of any depth (``srl`` requested) and LUTRAM (``distributed``
    requested, or ``auto`` for words narrower than ``LUTRAM_WIDTH`` bits), whose capacity is
    the depth; the integer lattice then needs no candidate list.
    """
    return requested in ("srl", "distributed") or (requested == "auto" and width < LUTRAM_WIDTH)


class CapacityModel:
    """Per-FIFO capacities of a design: padded stream widths and requested ``ram_style``.

    ``padded_widths`` are the ``DATA_WIDTH`` of the FIFO instances (the padded stream widths
    the style ladder sees), ``styles`` the ``ram_style`` attributes the build applies; depths
    are real FIFO depths (register value plus the driver's offset).
    """

    def __init__(self, padded_widths: Sequence[int], styles: Optional[Sequence[str]] = None):
        """Store the widths and styles (``auto`` for every FIFO when not given)."""
        self.padded_widths = [int(w) for w in padded_widths]
        self.styles = (
            ["auto"] * len(self.padded_widths) if styles is None else [str(s) for s in styles]
        )
        if len(self.styles) != len(self.padded_widths):
            raise ValueError("one ram_style per FIFO is required")

    @property
    def num_fifos(self) -> int:
        """Number of FIFOs."""
        return len(self.padded_widths)

    def capacity(self, i: int, depth: int) -> int:
        """Capacity of FIFO ``i`` at a requested depth."""
        return capacity(depth, self.padded_widths[i], self.styles[i])

    def plateau_top(self, i: int, depth: int) -> int:
        """Largest depth of FIFO ``i`` with the same capacity as ``depth``."""
        return plateau_top(depth, self.padded_widths[i], self.styles[i])

    def plateau_tops(self, i: int, max_depth: int) -> List[int]:
        """Plateau tops of FIFO ``i`` up to ``max_depth``."""
        return plateau_tops(self.padded_widths[i], max_depth, self.styles[i])

    def identity_above_shift(self, i: int) -> bool:
        """Whether FIFO ``i`` has no plateaus above the shift-register floor."""
        return identity_above_shift(self.padded_widths[i], self.styles[i])

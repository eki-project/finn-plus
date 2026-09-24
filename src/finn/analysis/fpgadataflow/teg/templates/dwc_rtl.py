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

"""Class B1: RTL stream data width converter (``finn-rtllib/dwc/hdl/dwc.sv``).

**Up-conversion** (``IBITS < OBITS``, ``K = OBITS/IBITS``, ``dwc.sv:genUp``): an assembly
register ``ADat`` of ``K`` input words (valid, i.e. ``ovld``, when ``ACnt`` turns negative)
and a one-word skid register ``BDat``/``BRdy`` in front of it (``irdy = BRdy``). Chains:

* ``R``: input handshakes; ``S``: shifts into ``ADat``; ``W``: output handshakes.
* ``R -> S`` skid edge (capacity 1, ``lf = 0``, ``lb = 1``): an accepted word shifts in the
  same cycle when ``rdy = !ovld || ordy`` (``:78``), otherwise it waits in ``BDat`` and the
  next input is accepted the cycle after the shift (``BRdy <= rdy || (BRdy && !ivld)``,
  ``:86``).
* ``S -> W`` assembly edge (``lf = 1``): the ``K``-th shift of a group makes ``ovld`` rise in
  the next cycle (``ACnt`` decrement, ``:79-82``, ``ovld = ACnt[$left]``, ``:90``).
* ``W -> S`` "assembly register free" edge (one initial token, ``lf = 0``): the first shift of
  the next group happens no earlier than the output handshake of the previous one, same cycle
  allowed (``rdy`` is combinational in ``ordy``, ``:78``).

**Down-conversion** (``IBITS > OBITS``, ``K = IBITS/OBITS``, ``dwc.sv:genDown``): the input
word is loaded into ``ADat`` and its ``K`` sub-words move one per cycle through ``BDat`` into
the output register ``CDat``/``CVld``. Chains:

* ``R``: input handshakes; ``S``: sub-word moves; ``W``: output handshakes.
* ``R -> S`` split edge (``lf = 1``): sub-word ``i`` of a word accepted in ``t`` moves at
  ``t + 1 + i`` at the earliest (``ADat`` loaded at the edge of ``t``, first move at ``t+1``,
  ``:118-126``); only every ``K``-th ``S`` event reads the edge.
* ``S -> W`` output edge (capacity 2 = ``BDat`` + ``CDat``, ``lf = 1``): a move in ``u``
  raises ``CVld`` in ``u + 1`` (``:129-131``); moves stall when both registers hold unaccepted
  sub-words (``BRdy``, ``:128``).
* ``S -> R`` "input register free" edge (one initial token, ``lf = 0``): the next input is
  accepted in the cycle of the last sub-word's move, which is the cycle in which ``ADat`` is
  reloaded (``irdy = BRdy && !ACnt[$left]``, ``:133``; ``ACnt`` reaches 0 after ``K-1``
  increments, ``:113-127``, and ``BRdy`` requires the output register to be free or
  handshaking in the previous cycle, ``:128``, which the move's own space constraint on the
  output edge expresses).

Equal widths reduce to a wire (``genNoop``, see ``passthrough``). All latencies are
transcriptions of the RTL and are confirmed by the differential test of the operator, with one
known deviation of the down-converter: when the input runs empty while the output register is
blocked, ``BRdy`` (``:128``) drops in the cycle after the last sub-word moved and only rises
again with the next output handshake, so the RTL accepts the next word one handshake later than
the model (which lets the empty converter accept one word into ``ADat``). The deviation is
bounded by one word per input bubble under back-pressure and is optimistic for sizing.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from finn.analysis.fpgadataflow.teg.model import Chain, FIFOEdge
from finn.analysis.fpgadataflow.teg.templates import OpModel, register
from finn.analysis.fpgadataflow.teg.templates.passthrough import passthrough
from finn.util.exception import FINNUserError

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


def dwc_up(prefix: str, n_in: int, k: int, in_edge: str, out_edge: str) -> OpModel:
    """Up-converter: ``k`` input words per output word, ``n_in`` input words per frame."""
    if n_in % k:
        raise FINNUserError(f"DWC up: {n_in} input words per frame not a multiple of K={k}")
    n_out = n_in // k
    skid, asm, free = f"{prefix}.skid", f"{prefix}.asm", f"{prefix}.free"
    r = Chain(f"{prefix}.R")
    r.events(n_in, 1, reads=[in_edge], writes=[skid])
    s = Chain(f"{prefix}.S")
    for _ in range(n_out):
        for i in range(k):
            reads = [skid, free] if i == 0 else [skid]
            writes = [asm] if i == k - 1 else []
            s.event(1, reads=reads, writes=writes)
    w = Chain(f"{prefix}.W")
    w.events(n_out, 1, reads=[asm], writes=[out_edge, free])
    edges = [
        FIFOEdge(skid, r.name, s.name, depth=1, lf=0, lb=1),  # dwc.sv:78-86
        FIFOEdge(asm, s.name, w.name, depth=None, lf=1, lb=1),  # dwc.sv:79-82, :90
        FIFOEdge(free, w.name, s.name, depth=None, lf=0, lb=1, initial_tokens=1),  # dwc.sv:78
    ]
    return OpModel([r, s, w], edges, [r.name], [w.name], {"K": k, "direction": "up"})


def dwc_down(prefix: str, n_in: int, k: int, in_edge: str, out_edge: str) -> OpModel:
    """Down-converter: ``k`` output sub-words per input word, ``n_in`` input words per frame."""
    split, outq, free = f"{prefix}.split", f"{prefix}.outq", f"{prefix}.free"
    r = Chain(f"{prefix}.R")
    r.events(n_in, 1, reads=[in_edge, free], writes=[split])
    s = Chain(f"{prefix}.S")
    for _ in range(n_in):
        for i in range(k):
            reads = [split] if i == 0 else []
            writes = [outq, free] if i == k - 1 else [outq]
            s.event(1, reads=reads, writes=writes)
    w = Chain(f"{prefix}.W")
    w.events(n_in * k, 1, reads=[outq], writes=[out_edge])
    edges = [
        FIFOEdge(split, r.name, s.name, depth=None, lf=1, lb=1),  # dwc.sv:118-126
        FIFOEdge(outq, s.name, w.name, depth=2, lf=1, lb=1),  # dwc.sv:128-131
        FIFOEdge(free, s.name, r.name, depth=None, lf=0, lb=1, initial_tokens=1),  # dwc.sv:133
    ]
    return OpModel([r, s, w], edges, [r.name], [w.name], {"K": k, "direction": "down"})


@register("StreamingDataWidthConverter", "rtl")
def dwc_rtl(node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]) -> OpModel:
    """Build the model of a ``StreamingDataWidthConverter_rtl`` node."""
    iw, ow = int(node.get_nodeattr("inWidth")), int(node.get_nodeattr("outWidth"))
    n_in = int(np.prod(node.get_folded_input_shape()[:-1]))
    if iw == ow:
        return passthrough(prefix, n_in, in_edges[0], out_edges[0])
    if iw < ow:
        if ow % iw:
            raise FINNUserError("RTL DWC up-conversion needs an integer width ratio")
        return dwc_up(prefix, n_in, ow // iw, in_edges[0], out_edges[0])
    if iw % ow:
        raise FINNUserError("RTL DWC down-conversion needs an integer width ratio")
    return dwc_down(prefix, n_in, iw // ow, in_edges[0], out_edges[0])

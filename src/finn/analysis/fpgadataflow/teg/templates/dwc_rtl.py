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

"""Class B1: RTL stream data width converter (``finn-rtllib/dwc/hdl/vpc.sv`` behind ``dwc_axi.sv``).

**Up-conversion** (``IBITS < OBITS``, ``K = OBITS/IBITS``, ``vpc.sv:genDes``, timing identical to
the former ``dwc.sv:genUp`` whose line numbers the edges still cite): an assembly
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

**Down-conversion** (``IBITS > OBITS``, ``K = IBITS/OBITS``, ``vpc.sv:genSer/genFull``, the
full-rate serializer of the vector pack converter that replaced ``dwc.sv``): the input word is
loaded in parallel into ``Buf`` and its ``K`` sub-words move one per cycle into the output
side register ``Side``/``SVld``; sub-word 0 bypasses ``Buf`` and lands in ``Side`` in the
cycle of the input handshake when ``Side`` is free or being consumed (``bypass``). Chains:

* ``R``: input handshakes; ``S``: sub-word moves into ``Side``; ``W``: output handshakes.
* ``R -> S`` split edge (``lf = 0``): sub-word 0 of a word accepted in ``t`` may move in
  ``t`` itself (bypass load), later sub-words shift one per cycle (``shift``).
* ``S -> W`` output edge (capacity 1 = ``Side``, ``lf = 1``, ``lb = 0``): a move in ``u``
  raises ``SVld`` in ``u + 1``; the next move may happen in the cycle of the output
  handshake (``bypass = !SVld || otrn``).
* ``S -> R`` "buffer empty" edge (one initial token, ``lf = 1``): ``irdy`` is ``Cnt == 0``,
  i.e. the next word is accepted in the cycle after the last sub-word of the previous one
  left ``Buf``.

**Non-integer ratios** (``vpc.sv:genGeneric``): after normalising both widths by their gcd,
the converter is an element FIFO of ``PI0 + PO0`` elements whose registered capacity counter
gates both handshakes (``ovld``/``irdy`` one cycle after the beat that completes / frees the
elements); see ``dwc_generic``. Equal widths reduce to a wire (``genNoop``, see
``passthrough``). All latencies are
transcriptions of the RTL and are confirmed by the differential test of the operator.
"""

from __future__ import annotations

import math

import numpy as np
from typing import TYPE_CHECKING

from finn.analysis.fpgadataflow.teg.model import Arc, Chain, FIFOEdge
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
        FIFOEdge(split, r.name, s.name, depth=None, lf=0, lb=1),  # vpc.sv genFull: bypass load
        FIFOEdge(outq, s.name, w.name, depth=1, lf=1, lb=0),  # vpc.sv: Side/SVld, bypass = otrn
        FIFOEdge(free, s.name, r.name, depth=None, lf=1, lb=1, initial_tokens=1),  # irdy = Cnt == 0
    ]
    return OpModel([r, s, w], edges, [r.name], [w.name], {"K": k, "direction": "down"})


def dwc_generic(
    prefix: str, n_in: int, n_out: int, pi0: int, po0: int, in_edge: str, out_edge: str
) -> OpModel:
    """Generic converter (``vpc.sv:genGeneric``): element FIFO of ``CAP`` elements.

    Widths are normalised by their gcd: an input beat deposits ``pi0`` elements, an output beat
    takes ``po0``. ``ovld`` (``F >= po0``) and ``irdy`` (``F <= CAP - pi0``) are derived from
    the registered capacity counter ``ICap``, so an output beat may follow the input beat that
    completed its elements one cycle later and an input beat may follow the output beat that
    freed the slots of its elements one cycle later (simultaneous handshakes are allowed).
    ``CAP = pi0 + po0 + SLACK`` with ``SLACK = min(pi0, po0) - 1`` (eki-project#290: the slack
    lets the narrower interface sustain full rate for non-integer ratios). The counter is
    never reset (``genSimple``: FINN instantiates the converter with ``N = IBITS * OBITS`` so
    no padding beats exist), i.e. element indices run on across frames.
    """
    cap = pi0 + po0 + min(pi0, po0) - 1
    r = Chain(f"{prefix}.R")
    r.events(n_in, 1, reads=[in_edge])
    w = Chain(f"{prefix}.W")
    w.events(n_out, 1, writes=[out_edge])
    # output beat j needs its last element (index (j+1)*po0-1), deposited by input beat i(j)
    for j in range(n_out):
        w.add_arc(j, Arc(r.name, ((j + 1) * po0 - 1) // pi0, 0, 1))
    # input beat i needs the slot of element (i+1)*pi0-1-cap, taken by output beat j(i)
    # (of the previous frame for the first beats of a frame)
    for i in range(n_in):
        e = (i + 1) * pi0 - 1 - cap
        if e < 0:
            e += n_out * po0
            if e < 0:
                continue
            r.add_arc(i, Arc(w.name, e // po0, 1, 1))
        else:
            r.add_arc(i, Arc(w.name, e // po0, 0, 1))
    r.history_window = -(-cap // pi0) + 4
    w.history_window = -(-cap // po0) + 4
    return OpModel([r, w], [], [r.name], [w.name], {"PI0": pi0, "PO0": po0, "direction": "generic"})


@register("StreamingDataWidthConverter", "rtl")
def dwc_rtl(node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]) -> OpModel:
    """Build the model of a ``StreamingDataWidthConverter_rtl`` node."""
    iw, ow = int(node.get_nodeattr("inWidth")), int(node.get_nodeattr("outWidth"))
    n_in = int(np.prod(node.get_folded_input_shape()[:-1]))
    if iw == ow:
        return passthrough(prefix, n_in, in_edges[0], out_edges[0])
    if iw < ow and ow % iw == 0:
        return dwc_up(prefix, n_in, ow // iw, in_edges[0], out_edges[0])
    if iw > ow and iw % ow == 0:
        return dwc_down(prefix, n_in, iw // ow, in_edges[0], out_edges[0])
    # non-integer ratio: vpc.sv normalises the widths by their gcd (dwc_axi.sv instantiates
    # it with W = 1, PI = IBITS, PO = OBITS, N = IBITS * OBITS)
    g = math.gcd(iw, ow)
    n_out = int(np.prod(node.get_folded_output_shape()[:-1]))
    return dwc_generic(prefix, n_in, n_out, iw // g, ow // g, in_edges[0], out_edges[0])

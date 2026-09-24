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

"""Class B1: RTL feature map padding (``finn-rtllib/fmpadding/hdl/fmpadding.sv``).

The core walks the padded output positions with counters ``SCount`` (channel fold, innermost),
``XCount`` and ``YCount`` (``:118-146``); a position is *forwarded* (``fwd = xfwd && yfwd``,
``:147-159``) when it lies inside ``[XOn, XOff) x [YOn, YOff)`` and consumes one input word,
otherwise it emits a zero word. Two registers:

* ``B``: the output register (``m_axis_tvalid = B.vld``, ``:163``); it is loaded when free or
  being accepted (``m_axis_tready || !B.vld``, ``:165-168``);
* ``A``: a one-word skid register in front of it (``s_axis_tready = !A.vld``, ``:161``); an
  input arriving while the position cannot advance is parked there (``:171-177``).

The position advances (``sen``, ``:160``) when ``B`` can take the word and the position's input
is present (or it is a pad). Model:

* chain ``R``: input handshakes, into a skid edge of capacity 1 (``lf = 0``: the word can be
  forwarded in the cycle it arrives, ``lb = 1``: the next word is accepted the cycle after
  ``A`` is drained);
* chain ``S``: one event per output position (reads the skid edge at forwarded positions),
  writing into the output register edge of capacity 1 (``lf = 1``, ``lb = 0``: the next
  position may be loaded in the cycle of the handshake);
* chain ``W``: output handshakes.

Parameters from ``fmpadding_rtl.py::get_template_values``: ``XOn = padL``, ``XOff = padL +
dimX``, ``XEnd = padL + dimX + padR - 1`` (``Y`` alike with ``padT``/``padB``), ``SF =
NumChannels / SIMD``. The counters only wrap at the end of a frame, so frames follow each
other without a gap.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from finn.analysis.fpgadataflow.teg.model import Chain, FIFOEdge
from finn.analysis.fpgadataflow.teg.templates import OpModel, register

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


def fmpadding(
    prefix: str,
    in_edge: str,
    out_edge: str,
    dim_y: int,
    dim_x: int,
    pads: tuple[int, int, int, int],
    sf: int,
) -> OpModel:
    """Build the padding model; ``pads`` is ``(top, left, bottom, right)``."""
    pad_t, pad_l, pad_b, pad_r = pads
    x_on, x_off, x_end = pad_l, pad_l + dim_x, pad_l + dim_x + pad_r - 1
    y_on, y_off, y_end = pad_t, pad_t + dim_y, pad_t + dim_y + pad_b - 1
    skid, breg = f"{prefix}.skid", f"{prefix}.breg"
    n_in = dim_y * dim_x * sf
    r = Chain(f"{prefix}.R")
    r.events(n_in, 1, reads=[in_edge], writes=[skid])
    # the position counters run from the reset release on: the first pad word is loaded
    # into B in the cycle before the environment's cycle 0 (measured, see the test)
    s = Chain(f"{prefix}.S", start=-1)
    n_out = 0
    for y in range(y_end + 1):
        for x in range(x_end + 1):
            fwd = x_on <= x < x_off and y_on <= y < y_off
            for _ in range(sf):
                s.event(1, reads=[skid] if fwd else [], writes=[breg])
                n_out += 1
    w = Chain(f"{prefix}.W")
    w.events(n_out, 1, reads=[breg], writes=[out_edge])
    edges = [
        FIFOEdge(skid, r.name, s.name, depth=1, lf=0, lb=1),  # fmpadding.sv:161, :171-177
        FIFOEdge(breg, s.name, w.name, depth=1, lf=1, lb=0),  # fmpadding.sv:160, :165-168
    ]
    return OpModel([r, s, w], edges, [r.name], [w.name], {"n_in": n_in, "n_out": n_out})


@register("FMPadding", "rtl")
def fmpadding_rtl(
    node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]
) -> OpModel:
    """Build the model of an ``FMPadding_rtl`` node."""
    dim_y, dim_x = node.get_nodeattr("ImgDim")
    pads = tuple(int(p) for p in node.get_nodeattr("Padding"))
    sf = node.get_nodeattr("NumChannels") // node.get_nodeattr("SIMD")
    vecs = node.get_nodeattr("numInputVectors")
    reps = int(np.prod(vecs)) if isinstance(vecs, list) else int(vecs)
    m = fmpadding(prefix, in_edges[0], out_edges[0], dim_y, dim_x, pads, sf)  # type: ignore
    if reps != 1:
        raise NotImplementedError("FMPadding template: numInputVectors != 1 not supported")
    return m

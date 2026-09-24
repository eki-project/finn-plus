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

"""Class B1: RTL thresholding (``finn-rtllib/thresholding/hdl/thresholding.sv``).

A free-running binary-search pipeline of ``M = $clog2(N+1)`` stages (``:128``, one extra
register per stage with ``DEEP_PIPELINE``, ``:329-342``) behind a credit semaphore:

* ``irdy = !cfg_en && !th_full`` (``:210``); ``th_full`` is the sign of ``GuardSem``, which
  starts at ``MAX_PENDING-1``, decrements on every accepted input and increments on every
  output handshake (``:135-142``). An input is therefore accepted in cycle ``t`` iff fewer
  than ``MAX_PENDING`` results are pending after cycle ``t-1``: a FIFO of capacity
  ``MAX_PENDING = (DEEP_PIPELINE+1)*M + 3`` (``:129``) with ``lb = 1`` (registered semaphore).
* The result of an input accepted in ``t`` reaches ``pipe[M]`` after ``M*(1+DEEP_PIPELINE)``
  cycles, is loaded into the output shift register ``ADat`` in that cycle (``aload``,
  ``:372-380``) and into the output register ``BDat``/``BVld`` in the next (``:387-391``), so
  ``ovld`` rises ``M*(1+DEEP_PIPELINE) + 2`` cycles after the input handshake: ``lf`` of the
  internal edge.
* Under back-pressure the results queue up in ``ADat`` (``A_DEPTH = MAX_PENDING - 1``,
  ``:363``) plus ``BDat``; the semaphore guarantees the queue is never overrun.

Parameters from ``thresholding_rtl.py::prepare_codegen_rtl_values``: ``N = numSteps``,
``DEEP_PIPELINE = deep_pipeline``. Channel folding (``CF``) and ``PE`` do not change the
control path.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from finn.analysis.fpgadataflow.teg.model import Chain, FIFOEdge
from finn.analysis.fpgadataflow.teg.templates import OpModel, register
from finn.analysis.fpgadataflow.teg.templates.hls_loop import clog2

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


def credit_pipeline(
    prefix: str, tokens: int, in_edge: str, out_edge: str, latency: int, credit: int
) -> OpModel:
    """1:1 free-running pipeline with ``latency`` behind a credit semaphore of ``credit``."""
    pipe = f"{prefix}.pipe"
    r = Chain(f"{prefix}.R")
    r.events(tokens, 1, reads=[in_edge], writes=[pipe])
    w = Chain(f"{prefix}.W")
    w.events(tokens, 1, reads=[pipe], writes=[out_edge])
    edge = FIFOEdge(pipe, r.name, w.name, depth=credit, lf=latency, lb=1)
    return OpModel(
        chains=[r, w],
        internal_edges=[edge],
        inputs=[r.name],
        outputs=[w.name],
        notes={"latency": latency, "credit": credit},
    )


def thresholding_rtl_params(num_steps: int, deep_pipeline: int) -> tuple[int, int]:
    """``(latency, credit)`` of the RTL thresholding core (see module docstring)."""
    m = clog2(num_steps + 1)  # thresholding.sv:128
    stages = m * (1 + deep_pipeline)
    max_pending = (deep_pipeline + 1) * m + 3  # thresholding.sv:129
    return stages + 2, max_pending


@register("Thresholding", "rtl")
def thresholding_rtl(
    node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]
) -> OpModel:
    """Build the credit-pipeline model of a ``Thresholding_rtl`` node."""
    tokens = int(np.prod(node.get_folded_input_shape()[:-1]))
    lat, credit = thresholding_rtl_params(
        int(node.get_nodeattr("numSteps")), int(node.get_nodeattr("deep_pipeline"))
    )
    return credit_pipeline(prefix, tokens, in_edges[0], out_edges[0], lat, credit)

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

"""Class C: combinational AXI-Stream pass-through.

``Reshape_rtl`` instantiates the RTL data width converter with equal widths
(``reshape_rtl.py::generate_hdl``), which reduces to ``dwc.sv:genNoop``: ``irdy = ordy``,
``ovld = ivld`` - no register, no buffering. The input handshake and the output handshake are
the same cycle: one chain with one event per token that reads the input and writes the output.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from finn.analysis.fpgadataflow.teg.model import Chain
from finn.analysis.fpgadataflow.teg.templates import OpModel, register

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


def passthrough(prefix: str, tokens: int, in_edge: str, out_edge: str) -> OpModel:
    """One event per token reading ``in_edge`` and writing ``out_edge`` in the same cycle."""
    c = Chain(f"{prefix}.P")
    c.events(tokens, 1, reads=[in_edge], writes=[out_edge])
    return OpModel(chains=[c], internal_edges=[], inputs=[c.name], outputs=[c.name])


@register("Reshape", "rtl")
def reshape_rtl(
    node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]
) -> OpModel:
    """``Reshape_rtl``: equal-width DWC, i.e. a wire (``dwc.sv:genNoop``)."""
    tokens = int(np.prod(node.get_folded_input_shape()[:-1]))
    return passthrough(prefix, tokens, in_edges[0], out_edges[0])

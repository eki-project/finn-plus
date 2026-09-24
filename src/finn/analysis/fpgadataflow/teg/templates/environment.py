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

"""Environment chains: pattern-driven sources and sinks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from finn.analysis.fpgadataflow.teg.model import Chain, TEGModel

if TYPE_CHECKING:
    from finn.analysis.fpgadataflow.teg.patterns import StallPattern


def pattern_source(
    name: str,
    edge: str,
    tokens_per_frame: int,
    pattern: StallPattern | None = None,
    period: int = 0,
) -> Chain:
    """Build a source writing ``tokens_per_frame`` tokens per frame to ``edge``.

    The source presents a token whenever ``pattern`` is true and holds it until it is accepted
    (AXI-Stream). ``period`` paces the frames.
    """
    c = Chain(name, kind="source", pattern=pattern, period=period)
    c.events(tokens_per_frame, 1, writes=[edge])
    return c


def pattern_sink(
    name: str, edge: str, tokens_per_frame: int, pattern: StallPattern | None = None
) -> Chain:
    """Build a sink reading ``tokens_per_frame`` tokens per frame from ``edge``.

    TREADY follows ``pattern``.
    """
    c = Chain(name, kind="sink", pattern=pattern)
    c.events(tokens_per_frame, 1, reads=[edge])
    return c


def attach_environment(
    model: TEGModel,
    in_edges: dict[str, int],
    out_edges: dict[str, int],
    in_patterns: dict[str, StallPattern] | None = None,
    out_patterns: dict[str, StallPattern] | None = None,
    period: int = 0,
) -> None:
    """Add one source per entry of ``in_edges`` and one sink per entry of ``out_edges``.

    The dict values are the tokens per frame of the edge; the edge's src/dst chain names must
    be ``"src_<edge>"`` / ``"sink_<edge>"`` respectively (as created by ``from_onnx``).
    """
    for edge, n in in_edges.items():
        pat = (in_patterns or {}).get(edge)
        model.add_chain(pattern_source(f"src_{edge}", edge, n, pat, period))
    for edge, n in out_edges.items():
        pat = (out_patterns or {}).get(edge)
        model.add_chain(pattern_sink(f"sink_{edge}", edge, n, pat))

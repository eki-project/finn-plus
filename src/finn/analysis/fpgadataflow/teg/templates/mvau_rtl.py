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

"""Class B1/B3: RTL matrix-vector unit (``finn-rtllib/mvu/mvu_vvu_axi.sv``).

* **Replay buffer** (``replay_buffer.sv``, ``LEN = SF``, ``REP = NF``): accepts the ``SF`` words
  of an input vector into a memory of ``2^clog2(SF)`` entries (``irdy``, ``:99``) and replays
  them ``NF`` times through a one-word output register (``ODat``/``OVld``, ``:106-110``); a
  word is written in cycle ``t``, read into the register in ``t+1`` and consumed by the core
  from ``t+2`` on; its slot is freed (``FP``, ``:125``) when its last repetition is read into
  the register, one cycle before the core consumes it: an edge of capacity ``2^clog2(SF)``,
  ``lf = 2``, ``lb = 0``, read by the core event of the last repetition; the earlier
  repetitions carry an arc to the word's input event (delay 2). ``REP = 1`` (``NF = 1``)
  bypasses the buffer (``:64-71``): ``lf = 0``, capacity 1.
* **Core** (``mvu.sv`` / ``mvu_vvu_8sx9_dsp58.sv``): free-running (``en = 1``), one folded
  input per cycle when the weight stream is valid (``ardy``, ``:141-142``; the weight
  memstream of decoupled nodes is always ready and not modelled), pipeline depth
  ``CORE_PIPELINE_DEPTH`` (``:345-347``): ``3 + (SEGMENTLEN == 0 ? 0 : ((SIMD+2)/3 - 1) /
  SEGMENTLEN)`` on DSP58 (``VERSION 3``), ``3 + clog2(SIMD+1) + (SIMD == 1)`` otherwise; one
  result per ``SF`` iterations.
* **Output queue** (``:348-389``): ``OBuf`` of ``MAX_IN_FLIGHT + 1 = L + 1`` entries plus the
  output register ``OReg``; a result is presented on the output ``L + 2`` cycles after the core
  consumed the last word of the fold (queue entry and output register add one cycle each;
  measured on DSP48E1, SIMD 4: first handshake 8 cycles after the last input, see the test).
  ``OLock`` (``:380-389``) stops the input (``idle``) when the output is valid but
  not accepted and releases it only once ``OBuf`` is empty: the output chain is a freezer that
  keeps the core frozen until at most one result (the one in ``OReg``) is left.

Parameters are read from the generated wrapper ``{gen_top_module}_wrapper.v`` (``VERSION``,
``SEGMENTLEN``, ``PUMPED_COMPUTE``). Pumped compute (2x clock) is not modelled. The constants
of this template are still being calibrated by ``test_teg_op_mvau_rtl.py`` (the SIMD = 6
configuration shows one extra cycle of core latency); its cases are marked as expected
deviations until the calibration is complete.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

from finn.analysis.fpgadataflow.teg.model import Arc, Chain, FIFOEdge
from finn.analysis.fpgadataflow.teg.templates import OpModel, register
from finn.analysis.fpgadataflow.teg.templates.hls_loop import clog2
from finn.util.exception import FINNUserError

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


def wrapper_params(node: HWCustomOp) -> dict[str, int]:
    """Integer parameters of the generated ``*_wrapper.v`` of an RTL MVAU/VVAU node."""
    code_gen_dir = cast("str", node.get_nodeattr("code_gen_dir_ipgen"))
    top = cast("str", node.get_nodeattr("gen_top_module"))
    if not code_gen_dir or not top:
        return {}
    path = Path(code_gen_dir) / f"{top}_wrapper.v"
    if not path.is_file():
        return {}
    text = path.read_text(errors="replace")
    params: dict[str, int] = {}
    for m in re.finditer(r"parameter\s+(\w+)\s*=\s*(-?\d+)", text):
        params[m.group(1)] = int(m.group(2))
    return params


def core_pipeline_depth(version: int, simd: int, segmentlen: int) -> int:
    """``CORE_PIPELINE_DEPTH`` of ``mvu_vvu_axi.sv:345-347``."""
    if version == 3:
        return 3 + (0 if segmentlen == 0 else ((simd + 2) // 3 - 1) // segmentlen)
    return 3 + clog2(simd + 1) + (1 if simd == 1 else 0)


def mvu_rtl(
    prefix: str, in_edge: str, out_edge: str, n_vectors: int, sf: int, nf: int, depth: int
) -> OpModel:
    """Build the replay-buffer / core / output-queue model of one RTL MVU."""
    rb, core = f"{prefix}.replay", f"{prefix}.core"
    rin = Chain(f"{prefix}.in")
    rin.events(n_vectors * sf, 1, reads=[in_edge], writes=[rb])
    c = Chain(f"{prefix}.core", freeze_group=f"{prefix}.olock")
    wout = Chain(
        f"{prefix}.out", freeze_group=f"{prefix}.olock", freezer=True, freeze_until_empty=True
    )
    replay_lf = 2 if nf > 1 else 0
    for v in range(n_vectors):
        for nf_i in range(nf):
            for sf_i in range(sf):
                last_rep = nf_i == nf - 1
                arcs = []
                if not last_rep:
                    # the word arrived at input event v*sf + sf_i (replay_buffer.sv:106-110)
                    arcs.append(Arc(rin.name, v * sf + sf_i, 0, replay_lf))
                c.event(
                    1,
                    reads=[rb] if last_rep else [],
                    writes=[core] if sf_i == sf - 1 else [],
                    arcs=arcs,
                )
    wout.events(n_vectors * nf, 1, reads=[core], writes=[out_edge])
    cap_rb = 1 if nf == 1 else 1 << max(1, clog2(sf))
    edges = [
        FIFOEdge(rb, rin.name, c.name, depth=cap_rb, lf=replay_lf, lb=0),  # replay_buffer.sv:99-125
        FIFOEdge(
            core, c.name, wout.name, depth=depth + 2, lf=depth + 2, lb=1
        ),  # mvu_vvu_axi.sv:348-389
    ]
    return OpModel(
        [rin, c, wout],
        edges,
        [rin.name],
        [wout.name],
        {"core_depth": depth, "replay_capacity": cap_rb, "SF": sf, "NF": nf},
    )


@register("MVAU", "rtl")
def mvau_rtl(node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]) -> OpModel:
    """Build the model of an ``MVAU_rtl`` node."""
    import numpy as np

    params = wrapper_params(node)
    if params.get("PUMPED_COMPUTE", 0):
        raise FINNUserError("MVAU_rtl template: pumped compute (2x clock) is not modelled")
    mw, mh = int(node.get_nodeattr("MW")), int(node.get_nodeattr("MH"))
    simd, pe = int(node.get_nodeattr("SIMD")), int(node.get_nodeattr("PE"))
    sf, nf = mw // simd, mh // pe
    reps = int(np.prod(node.get_folded_input_shape()[:-2]))
    version = params.get("VERSION", 1)
    segmentlen = params.get("SEGMENTLEN", 0)
    depth = core_pipeline_depth(version, simd, segmentlen)
    m = mvu_rtl(prefix, in_edges[0], out_edges[0], reps, sf, nf, depth)
    m.notes.update(
        {"VERSION": version, "SEGMENTLEN": segmentlen, "source": "wrapper" if params else "default"}
    )
    return m

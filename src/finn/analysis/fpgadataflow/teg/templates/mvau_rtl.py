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
  the register, one cycle before the core consumes it. The register loads a word in the cycle
  in which the core consumes the previous one, and it keeps doing so while the output lock
  idles the core: the word is then consumed only after the lock falls. The register is
  therefore its own chain (``.reg``, one event per word of the last repetition, pinned to the
  core's preceding event, outside the lock's freeze group) that reads the memory edge
  (capacity ``2^clog2(SF)``, ``lf = 1``, ``lb = 1``); the core's consumption follows the load
  by one cycle (arc, delay 1) and the earlier repetitions carry an arc to the word's input
  event (delay 2). ``REP = 1`` (``NF = 1``) bypasses the buffer (``:64-71``): ``lf = 0``,
  capacity 1, the core consumes the input directly.
* **Core** (``mvu.sv`` / ``mvu_vvu_8sx9_dsp58.sv``): free-running (``en = 1``), one folded
  input per cycle when the weight stream is valid (``ardy``, ``:141-142``; the weight
  memstream of decoupled nodes is always ready and not modelled); one result per ``SF``
  iterations after the core's own latency ``L``: ``mvu.sv:242`` ``PIPELINE_DEPTH = 3 +
  clog2(SIMD) + (SIMD == 1) + 1`` (DSP lanes plus the SIMD reduction tree; SIMD 4 measured 6,
  SIMD 6 measured 7), and ``2 + ceil(((SIMD+2)/3) / SEGLEN)`` for the DSP58 INT8 core
  (``mvu_vvu_8sx9_dsp58.sv:100-110``, chosen by ``mvu_vvu_axi.sv:311`` for VVUs and for
  narrow MVUs on ``VERSION 3``). The wrapper's ``CORE_PIPELINE_DEPTH`` (``:345-347``: ``3 +
  clog2(SIMD+1) + (SIMD == 1)``, or ``3 + ((SIMD+2)/3 - 1) / SEGMENTLEN`` on DSP58) is a
  conservative bound that only sizes the output queue, ``MAX_IN_FLIGHT + 1`` entries.
* **Output queue** (``:348-389``): ``OBuf`` of ``MAX_IN_FLIGHT + 1 = L + 1`` entries plus the
  output register ``OReg``; a result is presented on the output ``L + 2`` cycles after the core
  consumed the last word of the fold (queue entry and output register add one cycle each;
  measured on DSP48E1, SIMD 4: first handshake 8 cycles after the last input, see the test).
  The pipeline is free-running, so results in flight do not throttle the core: between the
  core consuming a fold and its result leaving ``OReg`` there is room for ``L`` results in the
  pipeline, ``MAX_IN_FLIGHT + 1`` in ``OBuf`` and one in ``OReg`` (the edge needs at least
  ``lf + lb`` slots to pass one result per cycle: an SF = 1 node measured 277 instead of 256
  cycles per vector when the capacity equalled the latency). ``OLock`` (``:380-389``) stops the
  input (``idle``) when the output is valid but not accepted and releases it only once ``OBuf`` is
  empty: the output chain is a freezer that keeps the core frozen until at most one result
  (the one in ``OReg``) is left.
  The lock is registered and engages only on the second entry into the queue (``:382-383``:
  the empty-queue case has priority): a result enters ``OBuf`` one cycle before ``OReg``
  offers it (``freeze_lead = 1``), and the core runs on until one cycle after a second result
  has entered the queue behind the stalled one (``freeze_threshold = 2``,
  ``freeze_delay = 1``); the queue is empty one cycle after the second-to-last result leaves
  ``OReg`` and the lock falls one cycle later (``thaw_delay = 2``). XSI traces of
  ``test_teg_op_mvau_rtl.py`` under output stalls fix these numbers.

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
    """``CORE_PIPELINE_DEPTH`` of ``mvu_vvu_axi.sv:345-347`` (sizes the output queue)."""
    if version == 3:
        return 3 + (0 if segmentlen == 0 else ((simd + 2) // 3 - 1) // segmentlen)
    return 3 + clog2(simd + 1) + (1 if simd == 1 else 0)


def core_latency(version: int, simd: int, segmentlen: int, int8_core: bool) -> int:
    """Cycles from the core consuming the last word of a fold to its result (``vld``).

    ``mvu.sv:242`` for the generic core, ``mvu_vvu_8sx9_dsp58.sv:100-110`` for the DSP58 INT8
    core (``last`` enters a shift register of ``2 + MAX_PIPELINE_STAGES`` stages).
    """
    if int8_core:
        chainlen = (simd + 2) // 3
        seglen = chainlen if segmentlen == 0 else segmentlen
        return 2 + (chainlen + seglen - 1) // seglen
    return 4 + clog2(simd) + (1 if simd == 1 else 0)


def uses_int8_core(params: dict[str, int], is_mvu: bool = True) -> bool:
    """``mvu_vvu_axi.sv:305-311``: VVUs and narrow MVUs on DSP58 use the INT8 core."""
    if not is_mvu:
        return True
    version = params.get("VERSION", 1)
    if version <= 2:
        return False
    ww, aw = params.get("WEIGHT_WIDTH", 8), params.get("ACTIVATION_WIDTH", 8)
    narrow = params.get("NARROW_WEIGHTS", 0)
    a_width = 25 + 2 * (version > 1)
    min_lane = ww + aw - 1
    lanes = 1 if a_width == ww else 1 + (a_width - (0 if narrow else 1) - ww) // min_lane
    return lanes <= 3 and ww <= 8 and aw <= 9


def mvu_rtl(
    prefix: str,
    in_edge: str,
    out_edge: str,
    n_vectors: int,
    sf: int,
    nf: int,
    depth: int,
    queue: int | None = None,
) -> OpModel:
    """Build the replay-buffer / core / output-queue model of one RTL MVU.

    ``depth`` is the core latency, ``queue`` the number of ``OBuf`` entries (``MAX_IN_FLIGHT +
    1``, defaults to ``depth + 1``).
    """
    if queue is None:
        queue = depth + 1
    rb, core = f"{prefix}.replay", f"{prefix}.core"
    c = Chain(f"{prefix}.core", freeze_group=f"{prefix}.olock")
    # the registered lock stops the core one cycle after a second result entered the output
    # queue (one cycle before OReg offers it) behind a stalled one
    wout = Chain(
        f"{prefix}.out",
        freeze_group=f"{prefix}.olock",
        freezer=True,
        freeze_until_empty=True,
        freeze_threshold=2,
        freeze_lead=1,
        freeze_delay=1,
        thaw_delay=2,
    )
    chains: list[Chain] = []
    edges: list[FIFOEdge] = []
    if nf == 1:
        # REP == 1: the replay buffer is a wire (replay_buffer.sv:84-90, ``irdy = ordy``,
        # ``ovld = ivld``); the core reads the input stream directly
        for _v in range(n_vectors):
            for sf_i in range(sf):
                c.event(1, reads=[in_edge], writes=[core] if sf_i == sf - 1 else [])
        first = c
    else:
        rin = Chain(f"{prefix}.in")
        rin.events(n_vectors * sf, 1, reads=[in_edge], writes=[rb])
        chains.append(rin)
        # the output register of the replay buffer: loads the words of the last repetition
        # (freeing their slots) in the cycle of the core's preceding consumption; only the
        # core's own preceding event and the latest register load are referenced
        reg = Chain(f"{prefix}.reg", history_window=2 * sf + 8)
        c.history_window = 2 * sf + 8
        chains.append(reg)
        replay_lf = 2
        for v in range(n_vectors):
            for nf_i in range(nf):
                for sf_i in range(sf):
                    last_rep = nf_i == nf - 1
                    arcs = []
                    if not last_rep:
                        # the word arrived at input event v*sf + sf_i (replay_buffer.sv:106-110)
                        arcs.append(Arc(rin.name, v * sf + sf_i, 0, replay_lf))
                    else:
                        k = v * sf + sf_i
                        prev = (v * nf + nf_i) * sf + sf_i - 1
                        reg.event(1, reads=[rb], arcs=[Arc(c.name, prev, 0, 0)])
                        arcs.append(Arc(reg.name, k, 0, 1))
                    c.event(1, writes=[core] if sf_i == sf - 1 else [], arcs=arcs)
        cap_rb = 1 << max(1, clog2(sf))
        # replay_buffer.sv:99-125
        edges.append(FIFOEdge(rb, rin.name, reg.name, depth=cap_rb, lf=1, lb=1))
        first = rin
    wout.events(n_vectors * nf, 1, reads=[core], writes=[out_edge])
    edges.append(
        FIFOEdge(core, c.name, wout.name, depth=depth + queue + 1, lf=depth + 2, lb=1)
    )  # mvu_vvu_axi.sv:348-389: pipeline (depth) + OBuf (queue) + OReg
    return OpModel(
        [*chains, c, wout],
        edges,
        [first.name],
        [wout.name],
        {
            "core_depth": depth,
            "queue": queue,
            "replay_capacity": 0 if nf == 1 else cap_rb,
            "SF": sf,
            "NF": nf,
        },
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
    int8 = uses_int8_core(params)
    depth = core_latency(version, simd, segmentlen, int8)
    queue = core_pipeline_depth(version, simd, segmentlen) + 1
    m = mvu_rtl(prefix, in_edges[0], out_edges[0], reps, sf, nf, depth, queue)
    m.notes.update(
        {
            "VERSION": version,
            "SEGMENTLEN": segmentlen,
            "int8_core": int8,
            "source": "wrapper" if params else "default",
        }
    )
    return m

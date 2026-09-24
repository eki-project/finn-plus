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

"""Class A3: ``ScaledDotProductAttention_hls`` (attention-hlslib), a dataflow region.

``attention.hpp:234-270`` builds, per invocation (one query sequence, ``QLen`` rows), a
``#pragma HLS dataflow`` region of nine processes connected by ``hls::stream`` FIFOs and two
ping-pong buffers; Vitis reports every process as its own function (identified here by the
streams it reads and writes, not by the generated names):

* ``KT_R`` / ``VT_R``: ``StreamTiler::read2buffer`` (``stream_tiler.hpp:196-215``): reads the
  whole key / value matrix (``KVLen * EmbFold`` chunks of ``in1_V`` / ``in2_V``) into a
  buffer; ``KT_W`` / ``VT_W`` (``:165-171``): emit the tiles ``QLen * EmbFold * SeqFold`` times
  into ``k_tiles`` / ``v_tiles`` (FIFO depth ``EmbFold * SeqFold``). The buffer is a dataflow
  array channel, i.e. a ping-pong buffer: the writer of frame ``n`` starts after the reader
  completed frame ``n``, the reader of frame ``n + 2`` after the writer completed frame ``n``.
* ``QK``: ``MatMul`` (``matmul.hpp:110-227``) over ``QLen * EmbFold * SeqFold`` iterations
  (``tr`` inner over ``EmbFold``, ``tc`` outer over ``SeqFold``): reads a query chunk from
  ``in0_V`` iff ``tc == 0``, a key tile every iteration, writes ``qk_out`` (depth ``SeqFold``)
  iff ``tr == EmbFold - 1``.
* ``SM0``/``SM1``/``SM2``: the three softmax stages (``softmax.hpp:150-350``) over
  ``QLen * SeqFold`` iterations; ``tmp``/``p_mask``/``value_buffer`` have depth ``SeqFold``,
  ``max_buffer``/``state_buffer`` depth 2; one max/state token per row.
* ``AV``: ``MatMul`` over ``QLen * SeqFold * EmbFold`` iterations (``tr`` inner over
  ``SeqFold``): reads ``softmax_out`` iff ``tc == 0``, a value tile every iteration, writes
  ``out0_V`` iff ``tr == SeqFold - 1``.

Every process is an flp loop that rewinds by itself (gap ``1 + rewind delay``; the softmax
stages have delays of tens of cycles because of their loop-carried accumulations); the
top-level ports carry register slices, the internal FIFOs do not (``lf = lb = 1``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from finn.analysis.fpgadataflow.teg.hls_report import HLSLoopParams, hls_function_params
from finn.analysis.fpgadataflow.teg.model import FIFOEdge
from finn.analysis.fpgadataflow.teg.templates import OpModel, register
from finn.analysis.fpgadataflow.teg.templates.hls_loop import Iteration, flp_loop
from finn.util.exception import FINNUserError

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

#: latencies of the internal ``hls::stream`` FIFOs: a token written in cycle t is readable in
#: t, the slot is free again one cycle after the read (calibrated against XSI, see the
#: sweep in the test notes; the constants below are the best fit of two configurations)
HLS_FIFO_LF = 0
HLS_FIFO_LB = 1
#: buffers of the stream tiler's dataflow array channel (1: the read loop of the next frame
#: waits for the tile loop of this frame; 2: ping-pong)
PIPO_BUFFERS = 1
#: cycles from the last read of the tiler's read loop to the first iteration of its tile loop
#: (loop exit, pipeline drain and the start of the second loop)
TILER_LOOP_GAP = 2
#: cycles from the last tile write of a frame until the read loop may start the next frame
TILER_RESTART_GAP = 1
#: the region is invoked once per frame by the top function and restarts only after all of
#: its processes completed: cycles from the last output handshake of frame n to the first
#: event of any process in frame n + 1 (measured)
REGION_RESTART_GAP = 11


def _find(funcs: dict[str, HLSLoopParams], reads: set[str], writes: set[str]) -> HLSLoopParams:
    """Process whose stream reads and writes contain the given ports."""
    for p in funcs.values():
        if reads <= set(p.read_stage) and writes <= set(p.write_stage):
            return p
    return HLSLoopParams()


def _process(
    prefix: str,
    its: list[Iteration],
    params: HLSLoopParams,
    external_in: set[str],
    external_out: set[str],
    internal_out: set[str],
) -> OpModel:
    params.top_level = False
    params.dataflow = True
    return flp_loop(
        prefix,
        its,
        params,
        in_regslices=external_in,
        out_regslices=external_out,
        internal_outputs=internal_out,
        drain=False,
    )


@register("ScaledDotProductAttention", "hls")
def attention_hls(
    node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]
) -> OpModel:
    """Build the nine-process model of one attention head (see the module docstring)."""
    if node.get_nodeattr("mask_mode") == "input":
        raise FINNUserError("Attention template: mask_mode 'input' (a fourth stream) not modelled")
    if len(in_edges) != 3:
        raise FINNUserError(f"Attention template expects 3 stream inputs, got {len(in_edges)}")
    q_in, k_in, v_in = in_edges
    out = out_edges[0]
    qlen, kvlen = node.get_nodeattr("QLen"), node.get_nodeattr("KVLen")
    embfold, seqfold = node.get_nodeattr("EmbFold"), node.get_nodeattr("SeqFold")
    funcs = hls_function_params(node)

    def e(name: str) -> str:
        return f"{prefix}.{name}"

    models: list[OpModel] = []
    fifos: list[tuple[str, int | None, int, int, int]] = []  # (edge, depth, initial, lf, lb)

    # ---- tilers: read loop -> ping-pong buffer -> write loop
    for tag, src in (("k", k_in), ("v", v_in)):
        tiles = e(f"{tag}_tiles")
        full, free = e(f"{tag}_buf_full"), e(f"{tag}_buf_free")
        n_read, n_write = kvlen * embfold, qlen * embfold * seqfold
        read_its: list[Iteration] = [((src, free), ())] + [((src,), ())] * (n_read - 2)
        read_its.append(((src,), (full,)))
        p_read = _find(funcs, {_port(src, in_edges)}, set())
        models.append(_process(e(f"{tag}t_r"), read_its, p_read, {src}, set(), {full}))
        write_its: list[Iteration] = [((full,), (tiles,))] + [((), (tiles,))] * (n_write - 2)
        write_its.append(((), (tiles, free)))
        p_write = _find(funcs, set(), {"k_tiles1" if tag == "k" else "v_tiles2"})
        models.append(_process(e(f"{tag}t_w"), write_its, p_write, set(), set(), {tiles, free}))
        # buffer hand-over (full: filled buffers, free: released ones)
        fifos += [
            (full, None, 0, TILER_LOOP_GAP, 0),
            (free, None, PIPO_BUFFERS, TILER_RESTART_GAP, 0),
            (tiles, embfold * seqfold, 0, HLS_FIFO_LF, HLS_FIFO_LB),
        ]

    # ---- QK matmul
    qk_out = e("qk_out")
    its: list[Iteration] = []
    for _ in range(qlen):
        for tc in range(seqfold):
            for tr in range(embfold):
                rd = (q_in, e("k_tiles")) if tc == 0 else (e("k_tiles"),)
                its.append((rd, (qk_out,) if tr == embfold - 1 else ()))
    p_qk = _find(funcs, {_port(q_in, in_edges)}, {"qk_out"})
    models.append(_process(e("qk"), its, p_qk, {q_in}, set(), {qk_out}))
    fifos.append((qk_out, seqfold, 0, HLS_FIFO_LF, HLS_FIFO_LB))

    # ---- softmax stages
    tmp, maxb, valb = e("tmp"), e("max_buffer"), e("value_buffer")
    stateb, sm_out = e("state_buffer"), e("softmax_out")
    n_sm = qlen * seqfold
    its0: list[Iteration] = [
        ((qk_out,), (tmp, maxb) if (i + 1) % seqfold == 0 else (tmp,)) for i in range(n_sm)
    ]
    its1: list[Iteration] = [
        (
            (maxb, tmp) if i % seqfold == 0 else (tmp,),
            (valb, stateb) if (i + 1) % seqfold == 0 else (valb,),
        )
        for i in range(n_sm)
    ]
    its2: list[Iteration] = [
        ((stateb, valb) if i % seqfold == 0 else (valb,), (sm_out,)) for i in range(n_sm)
    ]
    models.append(
        _process(e("sm0"), its0, _find(funcs, {"qk_out"}, {"tmp"}), set(), set(), {tmp, maxb})
    )
    models.append(
        _process(
            e("sm1"), its1, _find(funcs, {"tmp"}, {"value_buffer"}), set(), set(), {valb, stateb}
        )
    )
    models.append(
        _process(
            e("sm2"), its2, _find(funcs, {"value_buffer"}, {"softmax_out"}), set(), set(), {sm_out}
        )
    )
    for name, depth in ((tmp, seqfold), (maxb, 2), (valb, seqfold), (stateb, 2), (sm_out, seqfold)):
        fifos.append((name, depth, 0, HLS_FIFO_LF, HLS_FIFO_LB))

    # ---- AV matmul
    its_av: list[Iteration] = []
    for _ in range(qlen):
        for tc in range(embfold):
            for tr in range(seqfold):
                rd = (sm_out, e("v_tiles")) if tc == 0 else (e("v_tiles"),)
                its_av.append((rd, (out,) if tr == seqfold - 1 else ()))
    p_av = _find(funcs, {"softmax_out"}, {"out0_V"})
    models.append(_process(e("av"), its_av, p_av, set(), {out}, set()))

    # ---- assemble: the internal FIFOs connect the port chains of the processes
    chains = [c for m in models for c in m.chains]
    edges = [ed for m in models for ed in m.internal_edges]
    writer: dict[str, str] = {}
    reader: dict[str, str] = {}
    for c in chains:
        for ws in c.writes:
            for w in ws:
                writer.setdefault(w, c.name)
        for rs in c.reads:
            for r in rs:
                reader.setdefault(r, c.name)
    for name, depth, init, lf, lb in fifos:
        edges.append(
            FIFOEdge(
                name, writer[name], reader[name], depth=depth, lf=lf, lb=lb, initial_tokens=init
            )
        )
    # ---- per-frame invocation: every process restarts after the region completed, i.e.
    # after the last output handshake (the top function calls the region once per frame)
    av_port = writer[out]
    port_chain = next(c for c in chains if c.name == av_port)
    restarts = []
    for m in models:
        r_chain = next(c for c in m.chains if c.name.endswith((".R", ".RW")))
        name = f"{r_chain.name}.restart"
        restarts.append(name)
        r_chain.reads[0] = (*r_chain.reads[0], name)
        edges.append(
            FIFOEdge(
                name,
                av_port,
                r_chain.name,
                depth=None,
                lf=REGION_RESTART_GAP,
                lb=0,
                initial_tokens=1,
            )
        )
    port_chain.writes[-1] = (*port_chain.writes[-1], *restarts)
    by_input = {q_in: reader[q_in], k_in: reader[k_in], v_in: reader[v_in]}
    return OpModel(
        chains=chains,
        internal_edges=edges,
        inputs=[by_input[q_in], by_input[k_in], by_input[v_in]],
        outputs=[writer[out]],
        notes={"processes": [m.notes for m in models]},
    )


def _port(edge: str, in_edges: list[str]) -> str:
    """HLS port name (``in0_V``...) of the stream input fed by ``edge``."""
    return f"in{in_edges.index(edge)}_V"

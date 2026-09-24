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

"""Class A1: Vitis HLS pipelined loop, ``style=flp``, II=1, blocking stream I/O.

FINN+ synthesises every HLS operator with ``config_compile -pipeline_style flp``
(``hlsbackend.py::ipgen_default_directives``). The generated RTL (e.g.
``StreamingDataWidthConverter_hls_0.v``) shows the stall rule precisely:

* every AXI-Stream port goes through a Vitis register slice (``*_regslice_both.v``, two
  entries, registered valid and ready);
* a pipeline stage is blocked when its own stream read finds the input slice empty or its
  stream write finds the output slice not ready, and the last stage is additionally blocked
  while the output slice holds a token that is not being accepted (``apdone_blk``);
* a stage advances only when no stage at or after it is blocked: a blocked write freezes the
  whole pipeline, a starved read stage lets the later stages drain (the ``flp`` bubbles).

Model:

* chain ``PI.<edge>`` per input: the port handshake, feeding an internal edge of capacity 2
  (``lf = lb = 1``) that the loop reads from;
* chain ``R``: one event per loop iteration in the stage of the stream read, performing the
  iteration's reads; chain ``W``: one event per iteration in the stage of the stream write;
  an edge ``R -> W`` with forward latency ``lat`` = write stage - read stage and capacity
  ``lat`` (``lb = 0``). When both stages coincide the two chains are merged into one (the
  stage completes reads and writes together);
* the output slice is an edge of capacity 1 (``lf = 1``, ``lb = 0``) to the port chain
  ``PO.<edge>``, which is the *freezer* of the pipeline's freeze group: while it holds a token
  that the successor does not accept, ``R`` and ``W`` stop (``apdone_blk``). This holds for
  loops inlined into the top function. A loop in a non-inlined hlslib sub-function (e.g.
  ``Matrix_Vector_Activate_Batch`` with embedded weights) only stalls when the two-entry slice
  is full (plain FIFO, ``lf = lb = 1``); ``apdone_blk`` then delays the end of the invocation,
  so the next frame starts only after the slice has drained (edge ``PO -> R`` with one token
  per frame).

The stages are transcribed from the HLS schedule (``hls_report.hls_loop_params``); the loop
continues across frames when the report says ``auto-rewind`` with ``rewind_delay`` idle cycles
between the last and the first iteration, and with ``interval - trip count`` idle cycles when
the hlslib function is not inlined (one invocation per frame).

Iteration traces are transcribed from finn-hlslib; see the per-operator builders.
"""

from __future__ import annotations

import math
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING

from finn.analysis.fpgadataflow.teg.hls_report import HLSLoopParams
from finn.analysis.fpgadataflow.teg.model import Chain, FIFOEdge
from finn.analysis.fpgadataflow.teg.templates import OpModel, register
from finn.util.exception import FINNUserError

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

#: read-to-write latency assumed when no schedule is available (measured on MVAU_hls)
DEFAULT_LATENCY = 3
#: entries of the Vitis AXI-Stream register slice on every port (``*_regslice_both.v``)
REGSLICE_DEPTH = 2

#: one loop iteration: (edges read, edges written)
Iteration = tuple[Sequence[str], Sequence[str]]


def loop_params(node: HWCustomOp) -> HLSLoopParams:
    """Return the timing parameters of the node's loop from the synthesis outputs."""
    from finn.analysis.fpgadataflow.teg.hls_report import hls_loop_params

    return hls_loop_params(node)


def flp_loop(
    prefix: str,
    iterations: Sequence[Iteration],
    params: HLSLoopParams | None = None,
    latency: int | None = None,
    frame_gap: int | None = None,
    regslices: bool = True,
    in_regslices: bool | None = None,
    out_regslices: bool | None = None,
    capacity: int | None = None,
    back_latency: int | None = None,
) -> OpModel:
    """Build the model of one flp loop with its port register slices.

    Args:
        prefix: chain name prefix (usually the node name).
        iterations: per loop iteration of one frame, the edges read and written; the edges
            are the *external* edges of the node, the register slices are inserted here.
        params: loop parameters from the HLS reports (stages, rewind delay).
        latency: read-to-write latency override; defaults to the stage difference of the
            first output write and the first input read, else :data:`DEFAULT_LATENCY`.
        frame_gap: gap between the last iteration of a frame and the first of the next;
            defaults to ``1 + rewind_delay`` (auto-rewind) or ``1 + interval - trip count``.
        regslices: insert the port register slices (False for bare loops in unit tests).
        in_regslices: override ``regslices`` for the input ports.
        out_regslices: override ``regslices`` for the output ports.
        capacity: override the capacity of the read-to-write edge (experiments).
        back_latency: override its backward latency (experiments).
    """
    params = params or HLSLoopParams()
    in_edges = sorted({e for reads, _ in iterations for e in reads})
    out_edges = sorted({e for _, writes in iterations for e in writes})
    if latency is None:
        rs = [params.read_stage[p] for p in params.read_stage]
        ws = [params.write_stage[p] for p in params.write_stage]
        latency = (max(ws) - min(rs)) if rs and ws else DEFAULT_LATENCY
    if frame_gap is None:
        if params.rewind or params.interval is None or params.trip_count is None:
            frame_gap = 1 + params.rewind_delay
        else:
            # non-inlined loop: one invocation per frame; the top-level interval is the
            # frame period (measured: first iteration of the next frame follows the last one
            # after interval - trip count cycles)
            frame_gap = max(1, params.interval - params.trip_count)
    lat = max(0, latency)
    cap, lb = (lat, 0) if lat > 0 else (1, 1)
    if capacity is not None:
        cap = capacity
    if back_latency is not None:
        lb = back_latency
    in_rs = regslices if in_regslices is None else in_regslices
    out_rs = regslices if out_regslices is None else out_regslices
    group = f"{prefix}.pipeline"
    merged = lat == 0 and capacity is None and back_latency is None
    # iteration 0 enters stage 0 in the cycle after reset (-1 on the harness time base) for a
    # top-level loop and one cycle later when the loop is called as a sub-function
    entry = -1 if params.top_level else 0
    rs_list = list(params.read_stage.values())
    ws_list = list(params.write_stage.values())
    r_start = entry + (min(rs_list) if rs_list else 0)
    w_start = entry + (max(ws_list) if ws_list else lat)
    r = Chain(f"{prefix}.RW" if merged else f"{prefix}.R", freeze_group=group, start=r_start)
    w = r if merged else Chain(f"{prefix}.W", freeze_group=group, start=w_start)
    chains = [r] if merged else [r, w]
    edges: list[FIFOEdge] = []
    pipe = f"{prefix}.pipe"
    if not merged:
        edges.append(FIFOEdge(pipe, r.name, w.name, depth=cap, lf=lat, lb=lb))
    inputs: dict[str, str] = {}
    outputs: dict[str, str] = {}
    rename_in: dict[str, str] = {}
    rename_out: dict[str, str] = {}
    if in_rs:
        for e in in_edges:
            slice_edge = f"{prefix}.inslice.{e}"
            port = Chain(f"{prefix}.PI.{e}")
            n = sum(1 for reads, _ in iterations if e in reads)
            port.events(n, 1, reads=[e], writes=[slice_edge])
            chains.append(port)
            edges.append(FIFOEdge(slice_edge, port.name, r.name, depth=REGSLICE_DEPTH, lf=1, lb=1))
            rename_in[e] = slice_edge
            inputs[e] = port.name
    drain_edges: list[str] = []
    if out_rs:
        for e in out_edges:
            slice_edge = f"{prefix}.outslice.{e}"
            n = sum(1 for _, writes in iterations if e in writes)
            if params.top_level:
                port = Chain(f"{prefix}.PO.{e}", freeze_group=group, freezer=True)
                port.events(n, 1, reads=[slice_edge], writes=[e])
                # apdone_blk: one entry, the next write may follow in the cycle of the handshake
                edges.append(FIFOEdge(slice_edge, w.name, port.name, depth=1, lf=1, lb=0))
            else:
                port = Chain(f"{prefix}.PO.{e}")
                drain = f"{prefix}.drain.{e}"
                port.events(n - 1, 1, reads=[slice_edge], writes=[e])
                port.event(1, reads=[slice_edge], writes=[e, drain])
                edges.append(
                    FIFOEdge(slice_edge, w.name, port.name, depth=REGSLICE_DEPTH, lf=1, lb=1)
                )
                # the next invocation starts after the slice drained (apdone_blk on ap_done)
                edges.append(
                    FIFOEdge(drain, port.name, r.name, depth=None, lf=1, lb=1, initial_tokens=1)
                )
                drain_edges.append(drain)
            chains.append(port)
            rename_out[e] = slice_edge
            outputs[e] = port.name
    first = True
    for reads, writes in iterations:
        gap = frame_gap if first else 1
        rd = [rename_in.get(e, e) for e in reads] + (drain_edges if first else [])
        wrt = [rename_out.get(e, e) for e in writes]
        if merged:
            r.event(gap, reads=rd, writes=wrt)
        else:
            r.event(gap, reads=rd, writes=[pipe])
            w.event(gap, reads=[pipe], writes=wrt)
        first = False
    return OpModel(
        chains=chains,
        internal_edges=edges,
        inputs=[inputs.get(e, r.name) for e in in_edges] or [r.name],
        outputs=[outputs.get(e, w.name) for e in out_edges] or [w.name],
        notes={
            "latency": lat,
            "capacity": cap,
            "merged": merged,
            "frame_gap": frame_gap,
            "pipeline_depth": params.depth,
            "read_stage": dict(params.read_stage),
            "write_stage": dict(params.write_stage),
            "rewind": params.rewind,
            "source": params.source,
        },
    )


def _stage_latency(params: HLSLoopParams, in_port: str | None, out_port: str | None) -> int | None:
    """Write stage minus read stage of one function; ports default to its only streams."""
    rs = params.read_stage.get(in_port) if in_port else next(iter(params.read_stage.values()), None)
    ws = (
        params.write_stage.get(out_port)
        if out_port
        else next(iter(params.write_stage.values()), None)
    )
    if rs is None:
        rs = next(iter(params.read_stage.values()), None)
    if ws is None:
        ws = next(iter(params.write_stage.values()), None)
    if rs is None or ws is None:
        return None
    return ws - rs


def _reps(node: HWCustomOp) -> int:
    """Return the number of input vectors per frame (product of the leading folded dims)."""
    return int(np.prod(node.get_folded_input_shape()[:-2]))


# ---------------------------------------------------------------------------- MVAU / VVAU
def mvau_iterations(reps: int, sf: int, nf: int, in_edge: str, out_edge: str) -> list[Iteration]:
    """finn-hlslib ``mvau.hpp:121-172``: flat loop over ``reps*NF*SF``; the input vector is
    read during the first output fold (``nf == 0``) and buffered; one write per ``SF``
    iterations (``++sf == SF``)."""
    its: list[Iteration] = []
    for _ in range(reps):
        for nf_i in range(nf):
            for sf_i in range(sf):
                its.append(((in_edge,) if nf_i == 0 else (), (out_edge,) if sf_i == sf - 1 else ()))
    return its


@register("MVAU", "hls")
def mvau_hls(node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]) -> OpModel:
    """``Matrix_Vector_Activate_Batch`` (``mvau.hpp``): read iff ``nf == 0``, write iff
    ``sf == SF - 1``. Weights of ``internal_decoupled``/``external`` nodes are streamed by a
    memstream that is always ready and are not modelled."""
    mw, mh = node.get_nodeattr("MW"), node.get_nodeattr("MH")
    simd, pe = node.get_nodeattr("SIMD"), node.get_nodeattr("PE")
    sf, nf = mw // simd, mh // pe
    its = mvau_iterations(_reps(node), sf, nf, in_edges[0], out_edges[0])
    return flp_loop(prefix, its, loop_params(node))


@register("VVAU", "hls")
def vvau_hls(node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]) -> OpModel:
    """``Vector_Vector_Activate_Batch`` (``vvau.hpp:110-157``): unconditional read every
    iteration, one write per ``SF`` iterations."""
    k = node.get_nodeattr("Kernel")
    ch, pe, simd = node.get_nodeattr("Channels"), node.get_nodeattr("PE"), node.get_nodeattr("SIMD")
    sf = (int(np.prod(k)) * 1) // simd
    nf = ch // pe
    dim = node.get_nodeattr("Dim")
    reps = int(np.prod(dim))
    its: list[Iteration] = []
    for _ in range(reps * nf):
        for sf_i in range(sf):
            its.append(((in_edges[0],), (out_edges[0],) if sf_i == sf - 1 else ()))
    return flp_loop(prefix, its, loop_params(node))


# ---------------------------------------------------------------------------- Thresholding
@register("Thresholding", "hls")
def thresholding_hls(
    node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]
) -> OpModel:
    """``Thresholding_Batch`` (``activations.hpp:294-306``): one read and one write per
    iteration over ``reps * ImgDim * NF``. Synthesised with ``frp`` (``thresholding_hls.py::
    ipgen_extra_directives``); the flp two-chain model is used until the differential test
    says otherwise."""
    n = int(np.prod(node.get_folded_input_shape()[:-1]))
    its: list[Iteration] = [((in_edges[0],), (out_edges[0],))] * n
    m = flp_loop(prefix, its, loop_params(node))
    m.notes["pipeline_style"] = "frp"
    return m


# ---------------------------------------------------------------------------- DWC
def dwc_iterations(
    in_width: int, out_width: int, num_in_words: int, in_edge: str, out_edge: str
) -> list[Iteration]:
    """``StreamingDataWidthConverter_Batch`` (``streamtools.h:526-575``) for one frame.

    ``InWidth > OutWidth``: ``NumInWords * outPerIn`` iterations, read iff ``o == 0``, write
    every iteration. ``InWidth < OutWidth``: ``NumInWords`` iterations, read every iteration,
    write iff ``i == inPerOut - 1``. Equal widths: 1:1.
    """
    its: list[Iteration] = []
    if in_width > out_width:
        if in_width % out_width:
            raise FINNUserError("DWC down-conversion needs an integer width ratio")
        out_per_in = in_width // out_width
        for _ in range(num_in_words):
            for o in range(out_per_in):
                its.append(((in_edge,) if o == 0 else (), (out_edge,)))
    elif in_width < out_width:
        if out_width % in_width:
            raise FINNUserError("DWC up-conversion needs an integer width ratio")
        in_per_out = out_width // in_width
        for _ in range(num_in_words // in_per_out):
            for i in range(in_per_out):
                its.append(((in_edge,), (out_edge,) if i == in_per_out - 1 else ()))
    else:
        its = [((in_edge,), (out_edge,))] * num_in_words
    return its


@register("StreamingDataWidthConverter", "hls")
def dwc_hls(node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]) -> OpModel:
    """HLS data width converter; a non-integer ratio uses two converters through an
    intermediate ``hls::stream`` of depth 2 (``streamingdatawidthconverter_hls.py::docompute``,
    ``#pragma HLS DATAFLOW``)."""
    iw, ow = node.get_nodeattr("inWidth"), node.get_nodeattr("outWidth")
    n_in = int(np.prod(node.get_folded_input_shape()[:-1]))
    params = loop_params(node)
    if not node.needs_lcm():
        return flp_loop(prefix, dwc_iterations(iw, ow, n_in, in_edges[0], out_edges[0]), params)
    # Two converters under #pragma HLS DATAFLOW: each sub-function has its own loop and
    # stages; the intermediate hls::stream has the default depth 2. Only the outer ports
    # carry register slices.
    lcm = node.get_iowidth_lcm()
    mid = f"{prefix}.lcm"
    # each converter is a sub-function with its own schedule and rewind delay
    from finn.analysis.fpgadataflow.teg.hls_report import hls_function_params

    funcs = hls_function_params(node)
    params_a = next((p for p in funcs.values() if "in0_V" in p.read_stage), params)
    params_b = next((p for p in funcs.values() if "out0_V" in p.write_stage), params)
    its_a = dwc_iterations(iw, lcm, n_in, in_edges[0], mid)
    lat_a = _stage_latency(params_a, "in0_V", None)
    a = flp_loop(f"{prefix}.a", its_a, params_a, lat_a, in_regslices=True, out_regslices=False)
    n_mid = n_in * iw // lcm
    its_b = dwc_iterations(lcm, ow, n_mid, mid, out_edges[0])
    lat_b = _stage_latency(params_b, None, "out0_V")
    b = flp_loop(f"{prefix}.b", its_b, params_b, lat_b, in_regslices=False, out_regslices=True)
    mid_edge = FIFOEdge(mid, a.outputs[0], b.inputs[0], depth=2, lf=1, lb=1)
    return OpModel(
        chains=a.chains + b.chains,
        internal_edges=a.internal_edges + b.internal_edges + [mid_edge],
        inputs=a.inputs,
        outputs=b.outputs,
        notes={"lcm": lcm, **a.notes},
    )


# ---------------------------------------------------------------------------- LabelSelect
@register("LabelSelect", "hls")
def labelselect_hls(
    node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]
) -> OpModel:
    """``LabelSelect_Batch`` (``maxpool.h:199-240``): per image ``NumClasses/PE`` pipelined
    reads, then ``NumTop`` writes in a plain loop; the outer image loop is not pipelined."""
    labels, pe, k = node.get_nodeattr("Labels"), node.get_nodeattr("PE"), node.get_nodeattr("K")
    reps = int(np.prod(node.get_nodeattr("numInputVectors")))
    its: list[Iteration] = []
    for _ in range(reps):
        its += [((in_edges[0],), ())] * (labels // pe)
        its += [((), (out_edges[0],))] * k
    return flp_loop(prefix, its, loop_params(node))


# ---------------------------------------------------------------------------- Pool
@register("Pool", "hls")
def pool_hls(node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]) -> OpModel:
    """``Pool_batch`` (``pool.hpp:245-262``): one read per iteration, one write every ``K``
    iterations (``ModCounter<K>``)."""
    k = int(np.prod(node.get_nodeattr("KernelSize")))
    n_in = int(np.prod(node.get_folded_input_shape()[:-1]))
    its: list[Iteration] = [
        ((in_edges[0],), (out_edges[0],) if i % k == k - 1 else ()) for i in range(n_in)
    ]
    return flp_loop(prefix, its, loop_params(node))


# ---------------------------------------------------------------------------- GlobalAccPool
@register("GlobalAccPool", "hls")
def globalaccpool_hls(
    node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]
) -> OpModel:
    """``AccPool_Batch`` (``maxpool.h:130-149``): all ``ImgDim^2 * NF`` reads of an image,
    then ``NF`` writes from a drain loop."""
    n_in = int(np.prod(node.get_folded_input_shape()[:-1]))
    n_out = int(np.prod(node.get_folded_output_shape()[:-1]))
    reps = int(np.prod(node.get_nodeattr("numInputVectors")))
    its: list[Iteration] = []
    for _ in range(reps):
        its += [((in_edges[0],), ())] * (n_in // reps)
        its += [((), (out_edges[0],))] * (n_out // reps)
    return flp_loop(prefix, its, loop_params(node))


def clog2(n: int) -> int:
    """SystemVerilog ``$clog2``."""
    return 0 if n <= 1 else math.ceil(math.log2(n))

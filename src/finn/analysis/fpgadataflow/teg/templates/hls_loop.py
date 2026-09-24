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
#: per-token invocations (StreamingSplit/Concat): cycles from the port handshake of the write
#: to the read of the next invocation (measured)
INVOCATION_CREDIT = 2


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
    in_regslices: bool | set[str] | None = None,
    out_regslices: bool | set[str] | None = None,
    capacity: int | None = None,
    back_latency: int | None = None,
    drain: bool | None = None,
    iteration_gap: int = 1,
    internal_outputs: set[str] | None = None,
    entry: int | None = None,
    invocation_credit: int | None = None,
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
            defaults to 1 for inlined top-level loops, ``1 + rewind_delay`` for loops of
            dataflow processes and ``interval - trip count`` for non-inlined functions.
        regslices: insert the port register slices (False for bare loops in unit tests).
        in_regslices: override ``regslices`` for the input ports (a bool, or the set of
            input edges that are top-level AXI-Stream ports; the others are internal
            ``hls::stream`` FIFOs read directly).
        out_regslices: override ``regslices`` for the output ports (bool or set, as above).
        iteration_gap: cycles between consecutive iterations (1 for a pipelined loop; the
            top-level interval for a function that is invoked once per token).
        internal_outputs: output edges that are internal ``hls::stream`` FIFOs of a dataflow
            region: written without a register slice, but a full FIFO stalls the stage in
            the same cycle (a zero-latency freezer port).
        entry: cycle in which iteration 0 enters stage 0 (override of the -1 / 0 default).
        invocation_credit: for a top function invoked once per token: the next invocation's
            read may start this many cycles after the port handshake of the previous write
            (one invocation in flight).
        capacity: override the capacity of the read-to-write edge (experiments).
        back_latency: override its backward latency (experiments).
        drain: a non-inlined function is invoked once per frame and the next invocation
            waits until its output slices drained (``ap_done`` gating); False for a
            process inside a dataflow region, which runs continuously and whose output
            ports stall the pipeline like the ports of an inlined loop (``params.dataflow``).
            Defaults to ``not params.top_level and not params.dataflow``.
    """
    params = params or HLSLoopParams()
    # port order = order of first appearance in the iterations (the builders list the reads
    # and writes of an iteration in the node's stream order); ``OpModel.inputs/outputs``
    # must follow the node's stream order
    in_edges = list(dict.fromkeys(e for reads, _ in iterations for e in reads))
    out_edges = list(dict.fromkeys(e for _, writes in iterations for e in writes))
    if latency is None:
        rs = [params.read_stage[p] for p in params.read_stage]
        ws = [params.write_stage[p] for p in params.write_stage]
        latency = (max(ws) - min(rs)) if rs and ws else DEFAULT_LATENCY
    if frame_gap is None:
        if params.top_level:
            # inlined top-level loop under ap_ctrl_none: the loop rewinds without a bubble
            # even when the report lists a rewind delay (measured on a 9-iteration
            # ElementwiseAdd nest reported as "auto-rewind flp (delay=1)")
            frame_gap = 1
        elif params.rewind or params.interval is None or params.trip_count is None:
            # dataflow process: the loop rewinds by itself after the reported delay
            # (measured on Pool_batch, delay=1: one idle cycle between frames)
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
    in_rs_set = _slice_set(in_edges, regslices if in_regslices is None else in_regslices)
    out_rs_set = _slice_set(out_edges, regslices if out_regslices is None else out_regslices)
    internal = set(internal_outputs or ())
    out_rs_set -= internal
    group = f"{prefix}.pipeline"
    merged = lat == 0 and capacity is None and back_latency is None
    # iteration 0 enters stage 0 in the cycle after reset (-1 on the harness time base) for a
    # top-level loop and one cycle later when the loop is called as a sub-function
    if entry is None:
        entry = -1 if params.top_level else 0
    if drain is None:
        drain = not params.top_level and not params.dataflow
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
    for e in in_edges:
        if e in in_rs_set:
            slice_edge = f"{prefix}.inslice.{e}"
            port = Chain(f"{prefix}.PI.{e}")
            n = sum(1 for reads, _ in iterations if e in reads)
            port.events(n, 1, reads=[e], writes=[slice_edge])
            chains.append(port)
            edges.append(FIFOEdge(slice_edge, port.name, r.name, depth=REGSLICE_DEPTH, lf=1, lb=1))
            rename_in[e] = slice_edge
            inputs[e] = port.name
    drain_edges: list[str] = []
    # per-token invocations: iteration k reads the credit of the output written by iteration
    # k - 1 (cyclic over the frame), granted invocation_credit cycles after its port handshake
    credit_edges: dict[str, str] = (
        {e: f"{prefix}.credit.{e}" for e in out_edges} if invocation_credit is not None else {}
    )
    for e in out_edges:
        if e in internal:
            # internal FIFO of a dataflow region: the stage's write completes in the cycle
            # of the FIFO push; a full FIFO freezes the pipeline immediately
            slice_edge = f"{prefix}.outslice.{e}"
            n = sum(1 for _, writes in iterations if e in writes)
            port = Chain(f"{prefix}.PO.{e}", freeze_group=group, freezer=True)
            port.events(n, 1, reads=[slice_edge], writes=[e])
            edges.append(FIFOEdge(slice_edge, w.name, port.name, depth=1, lf=0, lb=1))
            chains.append(port)
            rename_out[e] = slice_edge
            outputs[e] = port.name
            continue
        if e in out_rs_set:
            slice_edge = f"{prefix}.outslice.{e}"
            n = sum(1 for _, writes in iterations if e in writes)
            if params.top_level or not drain:
                # inlined loop (apdone_blk) or dataflow process: one output entry, a blocked
                # handshake stalls the whole stage in the same cycle (all writes of the
                # stage wait: dup.hpp's "blocking write to all outputs"), the next write may
                # follow in the cycle of the handshake
                port = Chain(f"{prefix}.PO.{e}", freeze_group=group, freezer=True)
                port.events(
                    n,
                    1,
                    reads=[slice_edge],
                    writes=[e, credit_edges[e]] if e in credit_edges else [e],
                )
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
    if credit_edges:
        last_out = iterations[-1][1]
        if any(len(writes) != 1 for _, writes in iterations):
            raise FINNUserError("invocation_credit needs exactly one write per iteration")
        for e, name in credit_edges.items():
            edges.append(
                FIFOEdge(
                    name,
                    outputs[e],
                    r.name,
                    depth=None,
                    lf=invocation_credit,
                    lb=0,
                    initial_tokens=1 if e == last_out[0] else 0,
                )
            )
    first = True
    prev_out: str | None = iterations[-1][1][0] if credit_edges else None
    for reads, writes in iterations:
        gap = frame_gap if first else iteration_gap
        rd = [rename_in.get(e, e) for e in reads] + (drain_edges if first else [])
        if credit_edges:
            rd.append(credit_edges[prev_out])
            prev_out = writes[0]
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


def _slice_set(edges: list[str], spec: bool | set[str]) -> set[str]:
    """Edges that get a register slice: all/none for a bool, else the given subset."""
    if isinstance(spec, bool):
        return set(edges) if spec else set()
    return set(spec) & set(edges)


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


# ---------------------------------------------------------------------------- ReplicateStream
@register("ReplicateStream", "hls")
def replicate_stream_hls(
    node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]
) -> OpModel:
    """``replicate_stream_hls.py::docompute``: one flp loop over all folded elements, one read
    and one blocking write to every replica per iteration (zero skew tolerance between the
    outputs, see the inventory: class A2)."""
    n = int(np.prod(node.get_folded_output_shape()[:-1]))
    its: list[Iteration] = [((in_edges[0],), tuple(out_edges))] * n
    return flp_loop(prefix, its, loop_params(node))


# ---------------------------------------------------------------------------- 1:1 loops
def _one_to_one(
    node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]
) -> OpModel:
    """One read and one write per iteration over all folded output elements."""
    n = int(np.prod(node.get_folded_output_shape()[:-1]))
    its: list[Iteration] = [((in_edges[0],), (out_edges[0],))] * n
    return flp_loop(prefix, its, loop_params(node))


@register("Squeeze", "hls")
def squeeze_hls(
    node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]
) -> OpModel:
    """``squeeze_hls.py::docompute``: flp loop, ``out0_V.write(in0_V.read())``."""
    return _one_to_one(node, prefix, in_edges, out_edges)


@register("Unsqueeze", "hls")
def unsqueeze_hls(
    node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]
) -> OpModel:
    """``unsqueeze_hls.py::docompute``: flp loop, ``out0_V.write(in0_V.read())``."""
    return _one_to_one(node, prefix, in_edges, out_edges)


@register("Requant", "hls")
def requant_hls(
    node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]
) -> OpModel:
    """``requant_hls.py::docompute``: one pipelined loop over ``TOTAL_FOLD``, one read and one
    write per fold."""
    return _one_to_one(node, prefix, in_edges, out_edges)


@register("Lookup", "hls")
def lookup_hls(node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]) -> OpModel:
    """``StreamingLookup`` (``custom_hls/lookup.hpp:47-58``, ``internal_embedded``): one
    pipelined loop over ``NumInputs`` with one read and one write per iteration (``II=1``,
    default pipeline style). The ``external`` mode (AXI-MM burst per input) is not modelled."""
    if node.get_nodeattr("mem_mode") != "internal_embedded":
        raise FINNUserError("Lookup template: only mem_mode internal_embedded is modelled")
    return _one_to_one(node, prefix, in_edges, out_edges)


# ---------------------------------------------------------------------------- Split / Concat
@register("StreamingSplit", "hls")
def split_hls(node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]) -> OpModel:
    """``StreamingSplit`` (``split.hpp:83-110``): a pipelined function (``II=1``, flp) invoked
    once per input token; the token is written to output ``sel``, which advances after the
    ``C_k = ChannelsPerStream[k] / SIMD`` tokens of that output; the outputs are served in
    order for every input vector."""
    simd = node.get_nodeattr("SIMD")
    folds = [int(ch) // simd for ch in node.get_nodeattr("ChannelsPerStream")]
    vecs = int(np.prod(node.get_nodeattr("numInputVectors")))
    its: list[Iteration] = []
    for _ in range(vecs):
        for k, fold in enumerate(folds):
            its += [((in_edges[0],), (out_edges[k],))] * fold
    params, gap = _per_token_invocation(node)
    # the next invocation starts 2 cycles after the port handshake of the previous write
    return flp_loop(
        prefix,
        its,
        params,
        frame_gap=gap,
        iteration_gap=gap,
        entry=gap - 1,
        invocation_credit=INVOCATION_CREDIT,
    )


@register("StreamingConcat", "hls")
def concat_hls(node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]) -> OpModel:
    """``StreamingConcat`` (``concat.hpp:142-170``): a pipelined function invoked once per
    output token; a non-blocking read from input ``sel`` (a bubble when it is empty) and one
    write; ``sel`` advances after ``C_k = ChannelsPerStream[k] / SIMD`` tokens."""
    simd = node.get_nodeattr("SIMD")
    folds = [int(ch) // simd for ch in node.get_nodeattr("ChannelsPerStream")]
    vecs = int(np.prod(node.get_nodeattr("numInputVectors")))
    its: list[Iteration] = []
    for _ in range(vecs):
        for k, fold in enumerate(folds):
            its += [((in_edges[k],), (out_edges[0],))] * fold
    params, gap = _per_token_invocation(node)
    # the next invocation starts 2 cycles after the port handshake of the previous write
    return flp_loop(
        prefix,
        its,
        params,
        frame_gap=gap,
        iteration_gap=gap,
        entry=gap - 1,
        invocation_credit=INVOCATION_CREDIT,
    )


def _per_token_invocation(node: HWCustomOp) -> tuple[HLSLoopParams, int]:
    """Parameters of a top function without a loop that calls a pipelined hlslib function
    once per token: the top function is not pipelined, so one token is transferred per
    top-level interval (``Interval-min`` of the top report, e.g. 5 cycles for
    ``StreamingSplit``/``StreamingConcat``); the call sits in the last state of the top FSM,
    so the first read happens ``interval - 1`` cycles after reset (measured)."""
    params = loop_params(node)
    params.top_level = True
    params.rewind = True
    params.rewind_delay = 0
    gap = params.interval if params.interval else 1
    return params, gap


# ---------------------------------------------------------------------------- DuplicateStreams
@register("DuplicateStreams", "hls")
def duplicatestreams_hls(
    node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]
) -> OpModel:
    """``StreamingDup`` (``dup.hpp:28-33``): a free-running flp function inside a dataflow
    region (``duplicatestreams_hls.py::pragmas``): non-blocking read when the input is not
    empty, blocking writes to every output. One iteration per input token, no frame gap."""
    n = int(np.prod(node.get_folded_input_shape()[:-1]))
    its: list[Iteration] = [((in_edges[0],), tuple(out_edges))] * n
    params = loop_params(node)
    params.top_level = False
    params.dataflow = True
    return flp_loop(prefix, its, params, frame_gap=1, drain=False)


# ---------------------------------------------------------------------------- ElementwiseBinary
def broadcast_read_schedule(in_shape: Sequence[int], out_shape: Sequence[int]) -> list[bool]:
    """Per output element (row-major over ``out_shape``) whether the operand is read.

    Transcribes ``elementwise_binary_hls.py::read_stream_condition``: a dimension of size 1
    in the operand but larger in the output is broadcast, and the operand is read only when
    that index wraps to 0 (all broadcast indices at once).
    """
    padded = (len(out_shape) - len(in_shape)) * (1,) + tuple(in_shape)
    bdims = [d for d, (si, so) in enumerate(zip(padded, out_shape, strict=True)) if si == 1 != so]
    return [all(idx[d] == 0 for d in bdims) for idx in np.ndindex(*out_shape)]


def elementwise_binary_hls(
    node: HWCustomOp, prefix: str, in_edges: list[str], out_edges: list[str]
) -> OpModel:
    """``ElementwiseBinaryOperation_hls`` (``elementwise_binary_hls.py::docompute``): a perfect
    flp loop nest over the folded output shape; every streamed operand is read on its
    broadcast schedule, one write per iteration. A constant operand is either embedded
    (no stream) or fed by an always-ready memstream (``internal_decoupled``), which is not
    modelled, like the weight stream of a decoupled MVAU."""
    out_shape = tuple(int(d) for d in node.get_folded_output_shape()[:-1])
    styles = (node.get_nodeattr("lhs_style"), node.get_nodeattr("rhs_style"))
    edges = iter(in_edges)
    schedules: list[tuple[str, list[bool]]] = []
    for ind, style in enumerate(styles):
        if style != "input":
            continue
        edge = next(edges)
        in_shape = tuple(int(d) for d in node.get_folded_input_shape(ind)[:-1])
        schedules.append((edge, broadcast_read_schedule(in_shape, out_shape)))
    its: list[Iteration] = []
    for i in range(int(np.prod(out_shape))):
        its.append((tuple(e for e, sched in schedules if sched[i]), (out_edges[0],)))
    return flp_loop(prefix, its, loop_params(node))


#: op types derived from ``ElementwiseBinaryOperation`` (``elementwise_binary.py``)
ELEMENTWISE_BINARY_OPS = (
    "Add",
    "Sub",
    "AbsDiff",
    "Mul",
    "Div",
    "And",
    "Or",
    "Xor",
    "Equal",
    "Less",
    "LessOrEqual",
    "Greater",
    "GreaterOrEqual",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "BitShift",
    "Max",
)
for _op in ELEMENTWISE_BINARY_OPS:
    register(f"Elementwise{_op}", "hls")(elementwise_binary_hls)


def clog2(n: int) -> int:
    """SystemVerilog ``$clog2``."""
    return 0 if n <= 1 else math.ceil(math.log2(n))

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

"""Build a :class:`TEGModel` from a FINN dataflow graph.

Input: the ``ModelWrapper`` at the point of ``step_set_fifo_depths`` after DWC insertion and
``SpecializeLayers`` (all nodes have an ``_hls``/``_rtl`` backend suffix) and before FIFO
insertion. Every node is instantiated through its operator template; every node output
becomes one external edge (the FIFO that gets sized) to the consumer's input chain, graph
inputs become pattern sources behind the shallow (depth 2) input FIFO that ``InsertFIFO``
creates, graph outputs become pattern sinks. Tokens per frame are taken from the producer's
folded output shape and checked against the consumer's folded input shape.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, cast

from finn.analysis.fpgadataflow.teg.model import FIFOEdge, TEGModel
from finn.analysis.fpgadataflow.teg.templates import get_builder
from finn.analysis.fpgadataflow.teg.templates.environment import pattern_sink, pattern_source
from finn.util.basic import getHWCustomOp
from finn.util.exception import FINNUserError
from finn.util.logging import log

if TYPE_CHECKING:
    from onnx import NodeProto
    from qonnx.core.modelwrapper import ModelWrapper

    from finn.analysis.fpgadataflow.teg.patterns import StallPattern

#: FIFO latencies of FINN's Q_srl FIFO (Q_srl.v: registered o_v / i_b)
QSRL_LF = 1
QSRL_LB = 1
#: depth of the input FIFO that InsertFIFO(create_shallow_fifos=True) puts in front of node 0
INPUT_FIFO_DEPTH = 2


def edge_name(node: NodeProto, output: int) -> str:
    """Name of the external edge attached to output ``output`` of ``node``."""
    return f"{node.name}.out{output}"


def _stream_inputs(model: ModelWrapper, node: NodeProto) -> list[tuple[int, str]]:
    """``(input index, tensor)`` of the node inputs that are AXI streams (not initializers)."""
    inst = getHWCustomOp(node)
    ret = []
    for i, tensor in enumerate(node.input):
        if model.get_initializer(tensor) is not None:
            continue
        if inst.get_instream_width_padded(i) == 0:
            continue
        ret.append((i, tensor))
    return ret


def tokens_per_frame_out(node: NodeProto, output: int) -> int:
    """Tokens per frame on output ``output`` (product of the leading folded dims)."""
    return int(np.prod(getHWCustomOp(node).get_folded_output_shape(output)[:-1]))


def tokens_per_frame_in(node: NodeProto, inp: int) -> int:
    """Tokens per frame on input ``inp``."""
    return int(np.prod(getHWCustomOp(node).get_folded_input_shape(inp)[:-1]))


def build_model(
    model: ModelWrapper,
    in_patterns: dict[str, StallPattern] | None = None,
    out_patterns: dict[str, StallPattern] | None = None,
    input_fifo_depth: int = INPUT_FIFO_DEPTH,
    direct_environment: bool = False,
) -> TEGModel:
    """Instantiate the TEG model of ``model``.

    Args:
        model: specialised dataflow graph without FIFOs.
        in_patterns: stall pattern per graph input tensor name.
        out_patterns: stall pattern per graph output tensor name.
        input_fifo_depth: depth of the fixed input FIFO in front of every graph input.
        direct_environment: connect sources and sinks directly (no FIFOs), as a single-node
            RTL test bench does; the edges to the sinks are then not sized.

    Raises:
        FINNUserError: for nodes without a template or inconsistent token counts.
    """
    teg = TEGModel()
    nodes = list(model.graph.node)
    graph_inputs = [i.name for i in model.graph.input]
    graph_outputs = [o.name for o in model.graph.output]
    tensor_edge: dict[str, str] = {}  # tensor -> external edge name
    tensor_producer: dict[str, tuple[NodeProto, int]] = {}
    for node in nodes:
        for j, tensor in enumerate(node.output):
            tensor_edge[tensor] = edge_name(node, j)
            tensor_producer[tensor] = (node, j)
    for tensor in graph_inputs:
        if tensor not in tensor_edge:
            tensor_edge[tensor] = f"in.{tensor}"

    # ---- operator templates
    in_chain: dict[str, str] = {}  # edge -> reading chain
    out_chain: dict[str, str] = {}  # edge -> writing chain
    node_outputs: list[tuple[str, list[str]]] = []
    for node in nodes:
        inst = getHWCustomOp(node)
        ins = _stream_inputs(model, node)
        in_edges = [tensor_edge[t] for _, t in ins]
        out_edges = [edge_name(node, j) for j in range(len(node.output))]
        builder = get_builder(node.op_type)
        op = builder(inst, node.name, in_edges, out_edges)
        for c in op.chains:
            teg.add_chain(c)
        for e in op.internal_edges:
            teg.add_edge(e)
        for (i, _t), e in zip(ins, in_edges, strict=True):
            if i < len(op.inputs) and op.inputs[i] is not None:
                in_chain[e] = op.inputs[i]
        for j, e in enumerate(out_edges):
            out_chain[e] = op.outputs[j]
        node_outputs.append((node.name, out_edges))
    teg.meta["nodes"] = node_outputs

    # ---- external edges between nodes and to the sinks
    for node in nodes:
        inst = getHWCustomOp(node)
        for j, tensor in enumerate(node.output):
            e = edge_name(node, j)
            n_out = tokens_per_frame_out(node, j)
            consumers = model.find_consumers(tensor)
            if tensor in graph_outputs or not consumers:
                sink = pattern_sink(f"sink.{tensor}", e, n_out, (out_patterns or {}).get(tensor))
                teg.add_chain(sink)
                dst = sink.name
            else:
                if len(consumers) > 1:
                    raise FINNUserError(
                        f"Tensor {tensor} has {len(consumers)} consumers; insert DuplicateStreams"
                    )
                consumer = consumers[0]
                cin = [i for i, t in _stream_inputs(model, consumer) if t == tensor]
                if not cin:
                    raise FINNUserError(f"{consumer.name} does not read {tensor} as a stream")
                n_in = tokens_per_frame_in(consumer, cin[0])
                if n_in != n_out:
                    raise FINNUserError(
                        f"Token count mismatch on {tensor}: {node.name} writes {n_out} per "
                        f"frame, {consumer.name} reads {n_in}"
                    )
                if e not in in_chain:
                    raise FINNUserError(f"Template of {consumer.name} does not read {tensor}")
                dst = in_chain[e]
            direct = direct_environment and dst.startswith("sink.")
            teg.add_edge(
                FIFOEdge(
                    e,
                    out_chain[e],
                    dst,
                    depth=None,
                    lf=QSRL_LF,
                    lb=QSRL_LB,
                    width=inst.get_outstream_width(j),
                    external=not direct,
                    direct=direct,
                )
            )

    # ---- graph inputs: source -> shallow input FIFO -> first consumer
    for tensor in graph_inputs:
        e = tensor_edge[tensor]
        consumers = model.find_consumers(tensor)
        if not consumers:
            continue
        if e not in in_chain:
            log.info(f"Graph input {tensor} is not read by any operator template; skipped")
            continue
        consumer = consumers[0]
        cin = [i for i, t in _stream_inputs(model, consumer) if t == tensor]
        n = tokens_per_frame_in(consumer, cin[0])
        src = pattern_source(f"src.{tensor}", e, n, (in_patterns or {}).get(tensor))
        teg.add_chain(src)
        teg.add_edge(
            FIFOEdge(
                e,
                src.name,
                in_chain[e],
                depth=input_fifo_depth,
                lf=QSRL_LF,
                lb=QSRL_LB,
                width=getHWCustomOp(consumer).get_instream_width(cin[0]),
                external=False,
                direct=direct_environment,
            )
        )
    teg.validate()
    return teg


def depths_to_fifo_config(teg: TEGModel, depths: dict[str, int]) -> list[dict[str, object]]:
    """Translate external edge depths into the ``FIFODepthConfig`` list of
    ``ApplySimulatedFIFOSizes`` (one entry per node in graph order, one depth per output)."""
    nodes = cast("list[tuple[str, list[str]]]", teg.meta["nodes"])
    return [{"node": name, "depths": [int(depths[e]) for e in outs]} for name, outs in nodes]


def edge_widths(teg: TEGModel) -> dict[str, int]:
    """Token width in bits per external edge."""
    return {e: teg.edges[e].width for e in teg.external_edges}

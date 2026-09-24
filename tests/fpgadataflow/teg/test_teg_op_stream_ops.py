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

"""Differential tests of the simple HLS stream operators against XSI: Squeeze, Unsqueeze,
Requant and Lookup (1:1 loops), StreamingSplit (1:N) and StreamingConcat (N:1)."""

import pytest

import numpy as np
from onnx import NodeProto, TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.analysis.fpgadataflow.teg.patterns import StallPattern
from finn.util.basic import getHWCustomOp
from tests.fpgadataflow.teg.stall_injection import (
    is_prepared,
    prepare_node_rtlsim,
    run_abstract,
    run_xsi,
    stall_kinds,
    tokens_in,
    tokens_out,
)
from tests.fpgadataflow.teg.trace_compare import assert_traces_equal, compare_traces

TEST_FPGA_PART = "xc7z020clg400-1"
TARGET_CLK_NS = 5.0

pytestmark = [
    pytest.mark.fpgadataflow,
    pytest.mark.fifo_model,
    pytest.mark.vivado,
    pytest.mark.slow,
]


def _wrap(
    node: NodeProto, inputs: dict[str, tuple], outputs: dict[str, tuple], name: str
) -> ModelWrapper:
    """Graph with one node; ``inputs``/``outputs`` map tensor names to (shape, datatype)."""
    vi_in = [helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, (s, _) in inputs.items()]
    vi_out = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, (s, _) in outputs.items()
    ]
    graph = helper.make_graph(nodes=[node], name=name, inputs=vi_in, outputs=vi_out)
    model = ModelWrapper(qonnx_make_model(graph, producer_name=f"teg-{name}"))
    for n, (_, dt) in {**inputs, **outputs}.items():
        model.set_tensor_datatype(n, dt)
    return model


def make_squeeze_model(unsqueeze: bool, shape: list[int], pe: int) -> ModelWrapper:
    """``Squeeze``/``Unsqueeze`` node; the singleton axes are removed or added at the front."""
    dt = DataType["INT4"]
    if unsqueeze:
        inp_shape, out_shape, axes = shape, [1, *shape], [0]
    else:
        inp_shape, out_shape, axes = [1, *shape], shape, [0]
    node = helper.make_node(
        "Unsqueeze" if unsqueeze else "Squeeze",
        ["inp"],
        ["out"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        axes=axes,
        inp_dtype=dt.name,
        out_dtype=dt.name,
        inp_shape=inp_shape,
        out_shape=out_shape,
        PE=pe,
        preferred_impl_style="hls",
    )
    return _wrap(node, {"inp": (inp_shape, dt)}, {"out": (out_shape, dt)}, "squeeze")


def make_requant_model(channels: int, pe: int, vecs: list[int]) -> ModelWrapper:
    """``Requant`` node with per-channel scale and bias parameters."""
    idt, odt = DataType["INT8"], DataType["UINT4"]
    shape = [*vecs, channels]
    node = helper.make_node(
        "Requant",
        ["inp", "scale", "bias"],
        ["out"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        PE=pe,
        NumChannels=channels,
        inputDataType=idt.name,
        outputDataType=odt.name,
        numInputVectors=vecs,
        narrow=0,
        preferred_impl_style="hls",
    )
    model = _wrap(node, {"inp": (shape, idt)}, {"out": (shape, odt)}, "requant")
    model.set_initializer("scale", np.full((channels,), 0.25, dtype=np.float32))
    model.set_initializer("bias", np.zeros((channels,), dtype=np.float32))
    return model


def make_lookup_model(num_embeddings: int, dim: int, inputs: int) -> ModelWrapper:
    """``Lookup`` node with embedded embedding table."""
    idt, edt = DataType["UINT8"], DataType["INT4"]
    node = helper.make_node(
        "Lookup",
        ["inp", "embeddings"],
        ["out"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumEmbeddings=num_embeddings,
        EmbeddingDim=dim,
        EmbeddingType=edt.name,
        InputType=idt.name,
        InputShape=[1, inputs],
        mem_mode="internal_embedded",
        preferred_impl_style="hls",
    )
    model = _wrap(node, {"inp": ([1, inputs], idt)}, {"out": ([1, inputs, dim], edt)}, "lookup")
    model.set_initializer(
        "embeddings", gen_finn_dt_tensor(edt, (num_embeddings, dim)).astype(np.float32)
    )
    return model


def make_split_model(channels: list[int], simd: int, vecs: list[int]) -> ModelWrapper:
    """``StreamingSplit`` node: one input split into ``len(channels)`` outputs."""
    dt = DataType["UINT4"]
    outs = [f"out{i}" for i in range(len(channels))]
    node = helper.make_node(
        "StreamingSplit",
        ["inp"],
        outs,
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        SIMD=simd,
        ChannelsPerStream=channels,
        inputDataType=dt.name,
        numInputVectors=vecs,
        preferred_impl_style="hls",
    )
    return _wrap(
        node,
        {"inp": ([*vecs, sum(channels)], dt)},
        {o: ([*vecs, c], dt) for o, c in zip(outs, channels, strict=True)},
        "split",
    )


def make_concat_model(channels: list[int], simd: int, vecs: list[int]) -> ModelWrapper:
    """``StreamingConcat`` node: ``len(channels)`` inputs concatenated into one output."""
    dt = DataType["UINT4"]
    ins = [f"inp{i}" for i in range(len(channels))]
    node = helper.make_node(
        "StreamingConcat",
        ins,
        ["out"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        SIMD=simd,
        ChannelsPerStream=channels,
        inputDataTypes=[dt.name] * len(channels),
        numInputVectors=vecs,
        preferred_impl_style="hls",
    )
    return _wrap(
        node,
        {i: ([*vecs, c], dt) for i, c in zip(ins, channels, strict=True)},
        {"out": ([*vecs, sum(channels)], dt)},
        "concat",
    )


MAKERS = {
    "Squeeze": lambda *a: make_squeeze_model(False, *a),
    "Unsqueeze": lambda *a: make_squeeze_model(True, *a),
    "Requant": make_requant_model,
    "Lookup": make_lookup_model,
    "Split": make_split_model,
    "Concat": make_concat_model,
}

CONFIGS = [  # hashable: shapes as tuples, converted to lists by the makers
    ("Squeeze", (3, 8), 2),
    ("Unsqueeze", (3, 8), 4),
    ("Requant", 8, 2, (1, 3)),
    ("Requant", 6, 6, (1,)),
    ("Lookup", 16, 4, 6),
    ("Split", (4, 8, 4), 2, (1, 3)),
    ("Split", (2, 2), 2, (1, 2, 2)),
    ("Concat", (4, 8, 4), 2, (1, 3)),
    ("Concat", (2, 2), 2, (1, 2, 2)),
]


def _cfg_id(c: tuple) -> str:
    return "_".join(("x".join(map(str, x)) if isinstance(x, tuple) else str(x)) for x in c)


#: ``StreamingLookup`` is pipelined with the default (non-flushable) style: input starvation
#: stalls the whole pipeline, which the flushable-loop model does not reproduce under
#: simultaneous back-pressure
KNOWN_DEVIATIONS = {
    ("Lookup", "both_bernoulli"),
    ("Lookup", "both_bursty"),
    # Requant: 24-stage float pipeline, flp bubbles under back-pressure (up to 7 cycles)
    ("Requant", "both_bernoulli"),
    ("Requant", "both_bursty"),
}


_PREPARED: dict[tuple, ModelWrapper] = {}


def prepared_model(cfg: tuple) -> ModelWrapper:
    """Prepare (and cache per configuration) the XSI library of the node."""
    if not is_prepared(_PREPARED, cfg):
        args = [list(a) if isinstance(a, tuple) else a for a in cfg[1:]]
        model = MAKERS[cfg[0]](*args)
        _PREPARED[cfg] = prepare_node_rtlsim(model, TEST_FPGA_PART, TARGET_CLK_NS)
    return _PREPARED[cfg]


def stall_kinds_multi(seed: int, n_in: int, n_out: int) -> dict[str, tuple[list, list]]:
    """Patterns per input and per output stream."""
    none = StallPattern.none()
    return {
        "free": ([none] * n_in, [none] * n_out),
        "in_bernoulli": (
            [StallPattern.bernoulli(0.3, seed + i) for i in range(n_in)],
            [none] * n_out,
        ),
        "out_bernoulli": (
            [none] * n_in,
            [StallPattern.bernoulli(0.3, seed + 10 + j) for j in range(n_out)],
        ),
        "both_bernoulli": (
            [StallPattern.bernoulli(0.3, seed + 20 + i) for i in range(n_in)],
            [StallPattern.bernoulli(0.4, seed + 30 + j) for j in range(n_out)],
        ),
        "both_bursty": (
            [StallPattern.bursty(5, 3, phase=2 * i) for i in range(n_in)],
            [StallPattern.bursty(3, 6, phase=j) for j in range(n_out)],
        ),
        "out_single_stall": (
            [none] * n_in,
            [StallPattern.single_stall(12 + 3 * j, 25) for j in range(n_out)],
        ),
    }


@pytest.mark.parametrize("cfg", CONFIGS, ids=_cfg_id)
@pytest.mark.parametrize("kind", list(stall_kinds(0)))
def test_teg_op_stream_ops(
    cfg: tuple, kind: str, finn_test_seed: int, request: pytest.FixtureRequest
) -> None:
    """XSI and the abstract model must produce identical handshake traces on all streams."""
    if (cfg[0], kind) in KNOWN_DEVIATIONS:
        request.applymarker(
            pytest.mark.xfail(
                strict=False, reason="pipeline bubbles under back-pressure (not a fixed TEG)"
            )
        )
    model = prepared_model(cfg)
    node = model.graph.node[0]
    inst = getHWCustomOp(node)
    n_in = len([t for t in node.input if model.get_initializer(t) is None])
    n_out = len(node.output)
    in_pats, out_pats = stall_kinds_multi(finn_test_seed, n_in, n_out)[kind]
    frames = 2
    xsi = run_xsi(model, frames, in_pats, out_pats)
    model_in, model_out = run_abstract(model, frames, in_pats, out_pats)
    xsi_in, xsi_out = xsi.relative()
    for key in xsi_in:
        assert len(xsi_in[key]) == tokens_in(inst, int(key[2:])) * frames, key
    for key in xsi_out:
        assert len(xsi_out[key]) == tokens_out(inst, int(key[3:])) * frames, key
    print(
        f"\n{_cfg_id(cfg)} {kind}: XSI {xsi.cycles} cycles, first "
        f"{[v[0] for v in xsi_in.values()]}/{[v[0] for v in xsi_out.values()]} (model "
        f"{[v[0] for v in model_in.values()]}/{[v[0] for v in model_out.values()]}), last "
        f"{[v[-1] for v in xsi_in.values()]}/{[v[-1] for v in xsi_out.values()]} (model "
        f"{[v[-1] for v in model_in.values()]}/{[v[-1] for v in model_out.values()]})"
    )
    diffs = compare_traces(xsi_in, xsi_out, model_in, model_out)
    assert_traces_equal(diffs, f"{_cfg_id(cfg)} stall={kind}")

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

"""Differential test of the HLS ScaledDotProductAttention template (dataflow region with
tilers, two matmuls and the three-stage softmax) against XSI."""

import pytest

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.analysis.fpgadataflow.teg.patterns import StallPattern
from finn.util.basic import getHWCustomOp
from tests.fpgadataflow.teg.stall_injection import (
    is_prepared,
    prepare_node_rtlsim,
    run_abstract,
    run_xsi,
    tokens_in,
    tokens_out,
)
from tests.fpgadataflow.teg.trace_compare import assert_traces_equal, compare_traces
from tests.fpgadataflow.test_fpgadataflow_attention import MockScaledDotProductAttention

TEST_FPGA_PART = "xc7z020clg400-1"
TARGET_CLK_NS = 5.0

pytestmark = [
    pytest.mark.fpgadataflow,
    pytest.mark.fifo_model,
    pytest.mark.vivado,
    pytest.mark.slow,
]


def make_attention_model(dims: tuple[int, ...]) -> ModelWrapper:
    """Single ``ScaledDotProductAttention`` node with identity threshold activations."""
    qkdim, qlen, vdim, kvlen, embfold, seqfold = dims
    dt = DataType["UINT4"]
    mock = MockScaledDotProductAttention(
        QKDim=qkdim,
        QLen=qlen,
        VDim=vdim,
        KVLen=kvlen,
        EmbFold=embfold,
        SeqFold=seqfold,
        QType=dt,
        KType=dt,
        VType=dt,
        MType=dt,
        AType=dt,
        OType=dt,
    )
    model = mock.make_modelwrapper()
    getHWCustomOp(model.graph.node[0]).set_nodeattr("preferred_impl_style", "hls")
    return model


_PREPARED: dict[tuple, ModelWrapper] = {}


def prepared_model(cfg: tuple) -> ModelWrapper:
    """Prepare (and cache per configuration) the XSI library of the node."""
    if not is_prepared(_PREPARED, cfg):
        _PREPARED[cfg] = prepare_node_rtlsim(
            make_attention_model(cfg), TEST_FPGA_PART, TARGET_CLK_NS
        )
    return _PREPARED[cfg]


CONFIGS = [
    # (QKDim, QLen, VDim, KVLen, EmbFold, SeqFold)
    (4, 4, 4, 4, 2, 2),
    (8, 6, 4, 6, 2, 3),
]


def _cfg_id(c: tuple) -> str:
    return "QK{}_QL{}_V{}_KV{}_EF{}_SF{}".format(*c)


def stall_kinds_attention(seed: int) -> dict[str, tuple[list[StallPattern], StallPattern]]:
    """Patterns for the query, key and value streams and the output."""
    none = StallPattern.none()
    return {
        "free": ([none] * 3, none),
        "q_bernoulli": ([StallPattern.bernoulli(0.3, seed), none, none], none),
        "k_bernoulli": ([none, StallPattern.bernoulli(0.3, seed + 1), none], none),
        "v_bernoulli": ([none, none, StallPattern.bernoulli(0.3, seed + 2)], none),
        "out_bernoulli": ([none] * 3, StallPattern.bernoulli(0.3, seed + 3)),
        "out_single_stall": ([none] * 3, StallPattern.single_stall(40, 60)),
        "all_bernoulli": (
            [StallPattern.bernoulli(0.3, seed + 4 + i) for i in range(3)],
            StallPattern.bernoulli(0.4, seed + 7),
        ),
    }


@pytest.mark.parametrize("cfg", CONFIGS, ids=_cfg_id)
@pytest.mark.parametrize("kind", list(stall_kinds_attention(0)))
@pytest.mark.xfail(
    strict=False,
    reason="attention template under calibration: the per-frame region restart and the "
    "process structure are modelled, but the query reads and the output start are still "
    "2-5 cycles off and the real design shows one idle cycle per query row on the output "
    "(softmax row hand-over) that the model does not reproduce",
)
def test_teg_op_attention_hls(cfg: tuple, kind: str, finn_test_seed: int) -> None:
    """XSI and the abstract model must produce identical handshake traces on all streams."""
    model = prepared_model(cfg)
    inst = getHWCustomOp(model.graph.node[0])
    in_pats, out_pat = stall_kinds_attention(finn_test_seed)[kind]
    frames = 3
    xsi = run_xsi(model, frames, in_pats, out_pat)
    model_in, model_out = run_abstract(model, frames, in_pats, out_pat)
    xsi_in, xsi_out = xsi.relative()
    for key in xsi_in:
        assert len(xsi_in[key]) == tokens_in(inst, int(key[2:])) * frames, key
    assert len(xsi_out["out0"]) == tokens_out(inst) * frames
    print(
        f"\n{_cfg_id(cfg)} {kind}: XSI {xsi.cycles} cycles, first "
        f"{[v[0] for v in xsi_in.values()]}/{xsi_out['out0'][0]} (model "
        f"{[v[0] for v in model_in.values()]}/{model_out['out0'][0]}), last "
        f"{[v[-1] for v in xsi_in.values()]}/{xsi_out['out0'][-1]} (model "
        f"{[v[-1] for v in model_in.values()]}/{model_out['out0'][-1]})"
    )
    diffs = compare_traces(xsi_in, xsi_out, model_in, model_out)
    assert_traces_equal(diffs, f"Attention {_cfg_id(cfg)} stall={kind}")

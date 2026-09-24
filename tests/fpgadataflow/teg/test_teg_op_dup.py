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

"""Differential test of the HLS DuplicateStreams template (fork node) against XSI."""

import pytest

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

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

TEST_FPGA_PART = "xc7z020clg400-1"
TARGET_CLK_NS = 5.0

pytestmark = [
    pytest.mark.fpgadataflow,
    pytest.mark.fifo_model,
    pytest.mark.vivado,
    pytest.mark.slow,
]


def make_dup_model(ch: int, pe: int, idim: int, n_dupl: int) -> ModelWrapper:
    """Single ``DuplicateStreams`` node (HLS ``StreamingDup``)."""
    dt = DataType["INT4"]
    shape = [1, idim, idim, ch]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outs = [f"outp{i}" for i in range(n_dupl)]
    out_vi = [helper.make_tensor_value_info(o, TensorProto.FLOAT, shape) for o in outs]
    node = helper.make_node(
        "DuplicateStreams",
        ["inp"],
        outs,
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=ch,
        NumOutputStreams=n_dupl,
        PE=pe,
        inputDataType=dt.name,
        numInputVectors=[1, idim, idim],
        preferred_impl_style="hls",
    )
    graph = helper.make_graph(nodes=[node], name="dup", inputs=[inp], outputs=out_vi)
    model = ModelWrapper(qonnx_make_model(graph, producer_name="teg-dup"))
    model.set_tensor_datatype("inp", dt)
    for o in outs:
        model.set_tensor_datatype(o, dt)
    return model


_PREPARED: dict[tuple, ModelWrapper] = {}


def prepared_model(cfg: tuple) -> ModelWrapper:
    """Prepare (and cache per configuration) the XSI library of the node."""
    if not is_prepared(_PREPARED, cfg):
        _PREPARED[cfg] = prepare_node_rtlsim(make_dup_model(*cfg), TEST_FPGA_PART, TARGET_CLK_NS)
    return _PREPARED[cfg]


CONFIGS = [
    # (channels, PE, image dim, replicas)
    (8, 2, 3, 2),
    (8, 8, 2, 3),
]


def _cfg_id(c: tuple) -> str:
    return f"C{c[0]}_PE{c[1]}_D{c[2]}_x{c[3]}"


def stall_kinds_fork(seed: int, n: int) -> dict[str, tuple[StallPattern, list[StallPattern]]]:
    """Patterns for the input and the ``n`` replica outputs."""
    none = StallPattern.none()
    return {
        "free": (none, [none] * n),
        "in_bernoulli": (StallPattern.bernoulli(0.3, seed), [none] * n),
        "out0_bernoulli": (none, [StallPattern.bernoulli(0.3, seed + 1)] + [none] * (n - 1)),
        "out_last_single_stall": (none, [none] * (n - 1) + [StallPattern.single_stall(12, 25)]),
        "all_bernoulli": (
            StallPattern.bernoulli(0.3, seed + 2),
            [StallPattern.bernoulli(0.4, seed + 3 + i) for i in range(n)],
        ),
        "outs_bursty": (none, [StallPattern.bursty(5, 3, phase=2 * i) for i in range(n)]),
    }


@pytest.mark.parametrize("cfg", CONFIGS, ids=_cfg_id)
@pytest.mark.parametrize("kind", list(stall_kinds_fork(0, 2)))
def test_teg_op_dup_hls(cfg: tuple, kind: str, finn_test_seed: int) -> None:
    """XSI and the abstract model must produce identical handshake traces on all streams."""
    model = prepared_model(cfg)
    in_pat, out_pats = stall_kinds_fork(finn_test_seed, cfg[3])[kind]
    frames = 2
    xsi = run_xsi(model, frames, in_pat, out_pats)
    model_in, model_out = run_abstract(model, frames, in_pat, out_pats)
    xsi_in, xsi_out = xsi.relative()
    inst = getHWCustomOp(model.graph.node[0])
    assert len(xsi_in["in0"]) == tokens_in(inst) * frames
    for key in xsi_out:
        assert len(xsi_out[key]) == tokens_out(inst, int(key[3:])) * frames, key
    print(
        f"\n{_cfg_id(cfg)} {kind}: XSI {xsi.cycles} cycles, first {xsi_in['in0'][0]}/"
        f"{[v[0] for v in xsi_out.values()]} (model {model_in['in0'][0]}/"
        f"{[v[0] for v in model_out.values()]}), last {xsi_in['in0'][-1]}/"
        f"{[v[-1] for v in xsi_out.values()]} (model {model_in['in0'][-1]}/"
        f"{[v[-1] for v in model_out.values()]})"
    )
    diffs = compare_traces(xsi_in, xsi_out, model_in, model_out)
    assert_traces_equal(diffs, f"DuplicateStreams {_cfg_id(cfg)} stall={kind}")

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

"""Differential test of the RTL FMPadding template against XSI."""

import pytest

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

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


def make_fmpadding_model(
    idim: int, pads: tuple[int, int, int, int], ch: int, simd: int
) -> ModelWrapper:
    """Single ``FMPadding`` node (RTL)."""
    dt = DataType["UINT4"]
    odim_h = idim + pads[0] + pads[2]
    odim_w = idim + pads[1] + pads[3]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, idim, idim, ch])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, odim_h, odim_w, ch])
    node = helper.make_node(
        "FMPadding",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        ImgDim=[idim, idim],
        Padding=list(pads),
        NumChannels=ch,
        inputDataType=dt.name,
        numInputVectors=1,
        SIMD=simd,
        preferred_impl_style="rtl",
    )
    graph = helper.make_graph(nodes=[node], name="fmpadding_graph", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="teg-fmpadding"))
    model.set_tensor_datatype("inp", dt)
    model.set_tensor_datatype("outp", dt)
    return model


_PREPARED: dict[tuple, ModelWrapper] = {}


def prepared_model(cfg: tuple) -> ModelWrapper:
    """Prepare (and cache per configuration) the XSI library of the node."""
    if not is_prepared(_PREPARED, cfg):
        _PREPARED[cfg] = prepare_node_rtlsim(
            make_fmpadding_model(*cfg), TEST_FPGA_PART, TARGET_CLK_NS
        )
    return _PREPARED[cfg]


CONFIGS = [
    # (IFMDim, (top, left, bottom, right), NumChannels, SIMD)
    (4, (1, 1, 1, 1), 2, 2),
    (5, (1, 1, 1, 1), 4, 2),
    (4, (0, 1, 1, 0), 3, 1),
]


def _cfg_id(c: tuple) -> str:
    return f"D{c[0]}_p{''.join(map(str, c[1]))}_C{c[2]}_SIMD{c[3]}"


@pytest.mark.parametrize("cfg", CONFIGS, ids=_cfg_id)
@pytest.mark.parametrize("kind", list(stall_kinds(0)))
def test_teg_op_fmpadding(cfg: tuple, kind: str, finn_test_seed: int) -> None:
    """XSI and the abstract model must produce identical handshake traces."""
    model = prepared_model(cfg)
    in_pat, out_pat = stall_kinds(finn_test_seed)[kind]
    frames = 2
    xsi = run_xsi(model, frames, in_pat, out_pat)
    model_in, model_out = run_abstract(model, frames, in_pat, out_pat)
    xsi_in, xsi_out = xsi.relative()
    inst = getHWCustomOp(model.graph.node[0])
    n_in, n_out = tokens_in(inst) * frames, tokens_out(inst) * frames
    assert len(xsi_in["in0"]) == n_in and len(xsi_out["out0"]) == n_out
    print(
        f"\n{_cfg_id(cfg)} {kind}: {n_in} in / {n_out} out, XSI {xsi.cycles} cycles, first "
        f"in/out {xsi_in['in0'][0]}/{xsi_out['out0'][0]} (model "
        f"{model_in['in0'][0]}/{model_out['out0'][0]}), last {xsi_in['in0'][-1]}/"
        f"{xsi_out['out0'][-1]} (model {model_in['in0'][-1]}/{model_out['out0'][-1]})"
    )
    diffs = compare_traces(xsi_in, xsi_out, model_in, model_out)
    assert_traces_equal(
        diffs, f"FMPadding_rtl {_cfg_id(cfg)} stall={kind} in={in_pat!r} out={out_pat!r}"
    )

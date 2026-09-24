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

"""Differential test of the RTL ConvolutionInputGenerator (default template) against XSI."""

import pytest

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.im2col import compute_conv_output_dim
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


def make_swg_model(
    ifm_dim: int, k: int, stride: int, ifm_ch: int, simd: int, depthwise: int = 0
) -> ModelWrapper:
    """Single ``ConvolutionInputGenerator`` node (RTL, default template)."""
    dt = DataType["UINT4"]
    ofm_dim = compute_conv_output_dim(ifm_dim, k, stride, 0, 1)
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_dim, ifm_dim, ifm_ch])
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_dim, ofm_dim, k * k * ifm_ch]
    )
    node = helper.make_node(
        "ConvolutionInputGenerator",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        ConvKernelDim=[k, k],
        IFMChannels=ifm_ch,
        IFMDim=[ifm_dim, ifm_dim],
        OFMDim=[ofm_dim, ofm_dim],
        SIMD=simd,
        Stride=[stride, stride],
        Dilation=[1, 1],
        inputDataType=dt.name,
        outputDataType=dt.name,
        depthwise=depthwise,
        parallel_window=0,
        preferred_impl_style="rtl",
    )
    graph = helper.make_graph(nodes=[node], name="swg_graph", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="teg-swg"))
    model.set_tensor_datatype("inp", dt)
    model.set_tensor_datatype("outp", dt)
    return model


_PREPARED: dict[tuple, ModelWrapper] = {}


def prepared_model(cfg: tuple) -> ModelWrapper:
    """Prepare (and cache per configuration) the XSI library of the node."""
    if not is_prepared(_PREPARED, cfg):
        _PREPARED[cfg] = prepare_node_rtlsim(make_swg_model(*cfg), TEST_FPGA_PART, TARGET_CLK_NS)
    return _PREPARED[cfg]


CONFIGS = [
    # (IFMDim, k, stride, IFMChannels, SIMD, depthwise)
    (6, 3, 1, 1, 1, 0),
    (6, 3, 1, 4, 2, 0),
    (7, 3, 2, 2, 2, 0),
    (6, 3, 1, 4, 2, 1),  # depthwise, cf = 2 (rearranged loop nest)
    (6, 2, 2, 6, 2, 1),  # depthwise pooling window, cf = 3
    (8, 3, 2, 2, 2, 0),  # imperfect stride: one skipped row and column
    (8, 3, 2, 4, 2, 1),  # imperfect stride, depthwise, cf = 2
    (7, 7, 7, 4, 2, 1),  # global pooling window (MobileNet: one window per channel fold)
    # parallel template (1x1 kernels, ResNet downsampling)
    (6, 1, 1, 4, 2, 0),
    (7, 1, 2, 2, 1, 0),
    (8, 1, 2, 4, 2, 0),  # imperfect stride
]


def _cfg_id(c: tuple) -> str:
    style = "_par" if c[1] == 1 else ""
    return f"D{c[0]}_k{c[1]}_s{c[2]}_C{c[3]}_SIMD{c[4]}{'_dw' if c[5] else ''}{style}"


@pytest.mark.parametrize("cfg", CONFIGS, ids=_cfg_id)
@pytest.mark.parametrize("kind", list(stall_kinds(0)))
def test_teg_op_swg(cfg: tuple, kind: str, finn_test_seed: int) -> None:
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
    assert_traces_equal(diffs, f"SWG {_cfg_id(cfg)} stall={kind} in={in_pat!r} out={out_pat!r}")

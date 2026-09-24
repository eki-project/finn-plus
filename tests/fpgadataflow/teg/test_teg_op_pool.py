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

"""Differential tests of the HLS Pool and LabelSelect templates against XSI."""

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


def make_pool_model(ch: int, pe: int, k: int, odim: int, function: str) -> ModelWrapper:
    """Single ``Pool`` node fed with pre-generated windows (as after the SWG)."""
    dt = DataType["UINT4"]
    in_shape = [1, odim, odim, k * k * ch]
    out_shape = [1, odim, odim, ch]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape)
    node = helper.make_node(
        "Pool",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        Channels=ch,
        PE=pe,
        KernelSize=[k, k],
        Function=function,
        OutImgDims=[odim, odim],
        InputDataType=dt.name,
        OutputDataType=dt.name,
        # QuantAvgPool: accumulator of k*k UINT4 values, shifted right by Size bits
        AccumBits=(dt.bitwidth() + (k * k - 1).bit_length()) if function == "QuantAvgPool" else 0,
        Size=(k * k).bit_length() - 1 if function == "QuantAvgPool" else 0,
        BatchSize=1,
        preferred_impl_style="hls",
    )
    graph = helper.make_graph(nodes=[node], name="pool", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="teg-pool"))
    model.set_tensor_datatype("inp", dt)
    model.set_tensor_datatype("outp", dt)
    return model


def make_labelselect_model(labels: int, pe: int, k: int) -> ModelWrapper:
    """Single ``LabelSelect`` node (top-``k`` of one vector)."""
    dt = DataType["INT8"]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, labels])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, k])
    node = helper.make_node(
        "LabelSelect",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        Labels=labels,
        PE=pe,
        K=k,
        inputDataType=dt.name,
        numInputVectors=[1],
        preferred_impl_style="hls",
    )
    graph = helper.make_graph(nodes=[node], name="labelselect", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="teg-labelselect"))
    model.set_tensor_datatype("inp", dt)
    return model


_PREPARED: dict[tuple, ModelWrapper] = {}


def prepared_model(cfg: tuple) -> ModelWrapper:
    """Prepare (and cache per configuration) the XSI library of the node."""
    if not is_prepared(_PREPARED, cfg):
        maker = make_labelselect_model if cfg[0] == "LabelSelect" else make_pool_model
        _PREPARED[cfg] = prepare_node_rtlsim(maker(*cfg[1:]), TEST_FPGA_PART, TARGET_CLK_NS)
    return _PREPARED[cfg]


CONFIGS = [
    # ("Pool", channels, PE, kernel, output dim, function)
    ("Pool", 4, 2, 2, 3, "MaxPool"),
    ("Pool", 8, 8, 3, 2, "MaxPool"),  # MobileNet/ResNet global pooling: PE = channels
    ("Pool", 6, 2, 2, 2, "QuantAvgPool"),
    # ("LabelSelect", labels, PE, K)
    ("LabelSelect", 10, 1, 1),
    ("LabelSelect", 16, 4, 3),
]


def _cfg_id(c: tuple) -> str:
    return "_".join(str(x) for x in c)


@pytest.mark.parametrize("cfg", CONFIGS, ids=_cfg_id)
@pytest.mark.parametrize("kind", list(stall_kinds(0)))
def test_teg_op_pool_hls(
    cfg: tuple, kind: str, finn_test_seed: int, request: pytest.FixtureRequest
) -> None:
    """XSI and the abstract model must produce identical handshake traces."""
    if cfg[0] == "LabelSelect":
        request.node.add_marker(
            pytest.mark.xfail(
                reason="LabelSelect_Batch: the top-level FSM calls an outlined pipelined read "
                "loop and runs the NumTop writes unpipelined; the per-frame offsets (first "
                "read 1-2 cycles after entry, writes 2-5 cycles after the last read, period = "
                "top latency - 1) are not derived from the reports yet",
                strict=False,
            )
        )
    model = prepared_model(cfg)
    in_pat, out_pat = stall_kinds(finn_test_seed)[kind]
    frames = 3
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
    assert_traces_equal(diffs, f"{_cfg_id(cfg)} stall={kind}")

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

"""Differential test of the Thresholding operator templates (RTL and HLS) against XSI."""

import pytest

import math
import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

from finn.util.basic import getHWCustomOp
from tests.fpgadataflow.teg.stall_injection import (
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


def make_thresholding_model(
    num_channels: int, pe: int, num_steps: int, num_input_vecs: list[int], impl_style: str
) -> ModelWrapper:
    """Single ``Thresholding`` HW node with sorted integer thresholds."""
    idt = DataType["INT8"]
    tdt = DataType["INT8"]
    odt = DataType[f"UINT{max(1, math.ceil(math.log2(num_steps + 1)))}"]
    shape = [*num_input_vecs, num_channels]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)
    thresh = helper.make_tensor_value_info("thresh", TensorProto.FLOAT, [num_channels, num_steps])
    node = helper.make_node(
        "Thresholding",
        ["inp", "thresh"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=num_channels,
        PE=pe,
        numSteps=num_steps,
        inputDataType=idt.name,
        weightDataType=tdt.name,
        outputDataType=odt.name,
        numInputVectors=num_input_vecs,
        ActVal=0,
        preferred_impl_style=impl_style,
    )
    graph = helper.make_graph(
        nodes=[node], name="thresholding_graph", inputs=[inp], outputs=[outp], value_info=[thresh]
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="teg-thresholding"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("thresh", tdt)
    steps = np.round(np.linspace(-100, 100, num_steps)).astype(np.float32)
    model.set_initializer("thresh", np.tile(steps, (num_channels, 1)).astype(np.float32))
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


_PREPARED: dict[tuple, ModelWrapper] = {}


def prepared_model(cfg: tuple, impl_style: str) -> ModelWrapper:
    """Prepare (and cache per configuration) the XSI library of the node."""
    key = (*cfg, impl_style)
    if key not in _PREPARED:
        num_channels, pe, num_steps, vecs = cfg
        model = make_thresholding_model(num_channels, pe, num_steps, list(vecs), impl_style)
        _PREPARED[key] = prepare_node_rtlsim(model, TEST_FPGA_PART, TARGET_CLK_NS)
    return _PREPARED[key]


CONFIGS = [
    # (NumChannels, PE, numSteps, numInputVectors)
    (4, 1, 3, (1,)),
    (8, 2, 7, (1, 2, 2)),
    (6, 3, 15, (1,)),
]


def _cfg_id(c: tuple) -> str:
    return f"C{c[0]}_PE{c[1]}_N{c[2]}_V{'x'.join(map(str, c[3]))}"


@pytest.mark.parametrize("cfg", CONFIGS, ids=_cfg_id)
@pytest.mark.parametrize("kind", list(stall_kinds(0)))
@pytest.mark.parametrize("impl_style", ["rtl"])
def test_teg_op_thresholding(cfg: tuple, kind: str, impl_style: str, finn_test_seed: int) -> None:
    """XSI and the abstract model must produce identical handshake traces."""
    model = prepared_model(cfg, impl_style)
    in_pat, out_pat = stall_kinds(finn_test_seed)[kind]
    frames = 2
    xsi = run_xsi(model, frames, in_pat, out_pat)
    model_in, model_out = run_abstract(model, frames, in_pat, out_pat)
    xsi_in, xsi_out = xsi.relative()
    n_in = tokens_in(getHWCustomOp(model.graph.node[0])) * frames
    n_out = tokens_out(getHWCustomOp(model.graph.node[0])) * frames
    assert len(xsi_in["in0"]) == n_in and len(xsi_out["out0"]) == n_out
    print(
        f"\n{_cfg_id(cfg)} {kind}: {n_in} in / {n_out} out tokens, XSI {xsi.cycles} cycles, "
        f"first in/out handshake {xsi_in['in0'][0]}/{xsi_out['out0'][0]} (model "
        f"{model_in['in0'][0]}/{model_out['out0'][0]}), last {xsi_in['in0'][-1]}/"
        f"{xsi_out['out0'][-1]} (model {model_in['in0'][-1]}/{model_out['out0'][-1]})"
    )
    diffs = compare_traces(xsi_in, xsi_out, model_in, model_out)
    assert_traces_equal(
        diffs, f"Thresholding_{impl_style} {cfg} stall={kind} in={in_pat!r} out={out_pat!r}"
    )

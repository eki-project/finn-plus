# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2023, Advanced Micro Devices, Inc.
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

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import pynq_part_map

test_pynq_board = "Pynq-Z1"
test_fpga_part = pynq_part_map[test_pynq_board]
target_clk_ns = 10


def make_single_fmpadding_modelwrapper(impl_style, idim, padding, num_ch, simd, idt):
    pad_h = padding[0] + padding[2]
    pad_w = padding[1] + padding[3]
    idim_h, idim_w = idim

    assert pad_h > 0 or pad_w > 0, "Output dim should be greater than input dim"
    odim_h = idim_h + pad_h
    odim_w = idim_w + pad_w

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, idim_h, idim_w, num_ch])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, odim_h, odim_w, num_ch])

    FMPadding = helper.make_node(
        "FMPadding",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        ImgDim=idim,
        Padding=padding,
        NumChannels=num_ch,
        inputDataType=str(idt.name),
        numInputVectors=1,
        SIMD=simd,
        preferred_impl_style=impl_style,
    )

    graph = helper.make_graph(
        nodes=[FMPadding], name="fmpadding_graph", inputs=[inp], outputs=[outp]
    )

    model = qonnx_make_model(graph, producer_name="fmpadding-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", idt)

    return model


# input image dimension
@pytest.mark.parametrize("idim", [[8, 8], [10, 8]])
# number of rows and number of cols to add
@pytest.mark.parametrize("pad", [[1, 1, 1, 1], [1, 1, 2, 2], [1, 3, 2, 3], [7, 0, 8, 0]])
# number of channels
@pytest.mark.parametrize("num_ch", [2, 4])
# Input parallelism
@pytest.mark.parametrize("simd", [1, 2])
# FINN input datatype
@pytest.mark.parametrize("idt", [DataType["INT2"], DataType["INT4"]])
# execution mode
@pytest.mark.parametrize("mode", ["cppsim", "rtlsim"])
# implementation style
@pytest.mark.parametrize("impl_style", ["rtl", "hls"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_fmpadding(idim, pad, num_ch, simd, idt, mode, impl_style):
    if num_ch % simd != 0:
        pytest.skip(" num_ch % simd != 0, skipping")

    idim_h, idim_w = idim
    pad_h = pad[0] + pad[2]
    pad_w = pad[1] + pad[3]

    # generate input data
    x = gen_finn_dt_tensor(idt, [1, idim_h, idim_w, num_ch])
    input_dict = {"inp": x}
    odim_h = idim_h + pad_h
    odim_w = idim_w + pad_w

    y_expected = np.pad(x, ((0, 0), (pad[0], pad[2]), (pad[1], pad[3]), (0, 0)), "constant")
    expected_oshape = (1, odim_h, odim_w, num_ch)

    model = make_single_fmpadding_modelwrapper(impl_style, idim, pad, num_ch, simd, idt)

    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert y_produced.shape == expected_oshape
    assert (y_produced == y_expected).all(), "HW layer execution failed"

    model = model.transform(SpecializeLayers(test_fpga_part))

    model = model.transform(InferShapes())
    model = model.transform(SetExecMode(mode))
    model = model.transform(GiveUniqueNodeNames())
    if mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    elif mode == "rtlsim":
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())

    y_produced = oxe.execute_onnx(model, input_dict)["outp"]

    assert y_produced.shape == expected_oshape
    assert (y_produced == y_expected).all()

    if mode == "rtlsim":
        op_type = "FMPadding_" + impl_style
        node = model.get_nodes_by_op_type(op_type)[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
        assert exp_cycles != 0

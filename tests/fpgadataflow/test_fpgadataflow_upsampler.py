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
import os
import shutil
import torch
from brevitas.export import export_qonnx
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.make_input_chanlast import MakeInputChannelsLast
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from torch import nn

import finn.core.onnx_exec as oxe
import finn.transformation.streamline.absorb as absorb
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferUpsample
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.basic import make_build_dir

tmpdir = os.environ["FINN_BUILD_DIR"]


class ForceDataTypeForTensors(Transformation):
    """
    Forces a certain datatype for all tensors in a model.
    """

    def __init__(self, dType=DataType["INT8"]):
        super().__init__()
        self._dType = dType

    def apply(self, model):
        graph = model.graph
        for n in graph.node:
            for inp in n.input:
                model.set_tensor_datatype(inp, self._dType)
            for inp in n.output:
                model.set_tensor_datatype(inp, self._dType)

        return model, False


_to_chan_last_args = (0, 2, 3, 1)
_to_chan_first_args = (0, 3, 1, 2)


class PyTorchTestModel(nn.Module):
    def __init__(self, upscale_factor=2, size_factor=None):
        super(PyTorchTestModel, self).__init__()
        self.m = nn.Upsample(
            scale_factor=upscale_factor,
            size=size_factor,
            mode="nearest",
        )

    def forward(self, x):
        x = self.m(x)
        return x


# param datatype
@pytest.mark.parametrize("dt", [DataType["INT8"]])
# spatial dim input feature map
@pytest.mark.parametrize("IFMDim", [3, 5])
# upscaling factor
@pytest.mark.parametrize("scaling", [("size", 15), ("size", 30), ("scale", 2), ("scale", 3)])
# Number of input/output channels
@pytest.mark.parametrize("NumChannels", [4])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
# whether to use 1D or 2D square testcases
@pytest.mark.parametrize("is_1d", [False, True])
# Onnx opset version
@pytest.mark.parametrize("onnx_opset", [11, 13])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_upsampler(dt, IFMDim, scaling, NumChannels, exec_mode, is_1d, onnx_opset):
    tmpdir = make_build_dir("upsample_export_")
    atol = 1e-3
    scaling_method, value = scaling
    if is_1d:
        input_shape = (1, NumChannels, IFMDim, 1)
        factor = (value, 1)
    else:
        input_shape = (1, NumChannels, IFMDim, IFMDim)
        factor = (value, value)

    if scaling_method == "scale":
        size_factor = None
        upscale_factor = factor
    elif scaling_method == "size":
        upscale_factor = None
        size_factor = factor

    # Create the test model and inputs for it
    torch_model = PyTorchTestModel(upscale_factor=upscale_factor, size_factor=size_factor)
    test_in = torch.arange(0, np.prod(np.asarray(input_shape)))
    # Limit the input to values valid for the given datatype
    test_in %= dt.max() - dt.min() + 1
    test_in += dt.min()
    # Additionally make sure we always start with 0, for convenience purposes.
    test_in = torch.roll(test_in, dt.min())
    test_in = test_in.view(*input_shape).type(torch.float32)

    # Get golden PyTorch and ONNX inputs
    golden_torch_float = torch_model(test_in)
    export_path = f"{tmpdir}/Upsample_exported.onnx"
    export_qonnx(torch_model, torch.randn(input_shape), export_path, opset_version=onnx_opset)
    qonnx_cleanup(export_path, out_file=export_path)
    model = ModelWrapper(export_path)
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    input_dict = {model.graph.input[0].name: test_in.numpy().astype(np.int32)}
    input_dict = {model.graph.input[0].name: test_in.numpy()}
    golden_output_dict = oxe.execute_onnx(model, input_dict, True)
    golden_result = golden_output_dict[model.graph.output[0].name]

    # Make sure PyTorch and ONNX match
    pyTorch_onnx_match = np.isclose(golden_result, golden_torch_float).all()
    assert pyTorch_onnx_match, "ONNX and PyTorch upsampling output don't match."

    # Prep model for execution
    model = ModelWrapper(export_path)
    model = model.transform(MakeInputChannelsLast())
    model = model.transform(InferDataLayouts())
    model = model.transform(absorb.AbsorbTransposeIntoResize())
    model = model.transform(InferShapes())
    model = model.transform(ForceDataTypeForTensors(dType=dt))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferUpsample())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Check that all nodes are UpsampleNearestNeighbour_Batch nodes
    for n in model.get_finn_nodes():
        node_check = n.op_type == "UpsampleNearestNeighbour"
        assert node_check, "All nodes should be UpsampleNearestNeighbour nodes."

    test_in_transposed = test_in.numpy().transpose(_to_chan_last_args)
    input_dict = {model.graph.input[0].name: test_in_transposed}

    # Run sim
    output_dict = oxe.execute_onnx(model, input_dict, True)
    test_result = output_dict[model.graph.output[0].name]
    output_matches = np.isclose(golden_result, test_result, atol=atol).all()

    model = model.transform(SpecializeLayers("xc7z020clg400-1"))

    # Prep sim
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP("xc7z020clg400-1", 10))
        model = model.transform(HLSSynthIP())
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")

    # Run sim
    output_dict = oxe.execute_onnx(model, input_dict, True)
    test_result = output_dict[model.graph.output[0].name]
    output_matches = np.isclose(golden_result, test_result, atol=atol).all()

    if exec_mode == "cppsim":
        assert output_matches, "Cppsim output doesn't match ONNX/PyTorch."
    elif exec_mode == "rtlsim":
        assert output_matches, "Rtlsim output doesn't match ONNX/PyTorch."
    shutil.rmtree(tmpdir, ignore_errors=True)

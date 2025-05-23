# Copyright (c) 2020, Xilinx
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

import importlib_resources as importlib
import numpy as np
import os
import torch
from brevitas.export import export_qonnx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.basic import make_build_dir
from finn.util.test import get_test_model_trained


@pytest.mark.brevitas_export
@pytest.mark.parametrize("abits", [1, 2])
@pytest.mark.parametrize("wbits", [1, 2])
def test_brevitas_cnv_export_exec(wbits, abits):
    if wbits > abits:
        pytest.skip("No wbits > abits cases at the moment")
    build_dir = make_build_dir("test_brevitas_cnv_export_exec")
    export_onnx_path = os.path.join(build_dir, "test_brevitas_cnv.onnx")
    cnv = get_test_model_trained("CNV", wbits, abits)
    ishape = (1, 3, 32, 32)
    export_qonnx(cnv, torch.randn(ishape), export_onnx_path)
    qonnx_cleanup(export_onnx_path, out_file=export_onnx_path)
    model = ModelWrapper(export_onnx_path)
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(RemoveStaticGraphInputs())
    assert len(model.graph.input) == 1
    assert len(model.graph.output) == 1
    ref = importlib.files("finn.qnn-data") / "cifar10/cifar10-test-data-class3.npz"
    with importlib.as_file(ref) as fn:
        input_tensor = np.load(fn)["arr_0"].astype(np.float32)
    input_tensor = input_tensor / 255
    assert input_tensor.shape == (1, 3, 32, 32)
    # run using FINN-based execution
    input_dict = {model.graph.input[0].name: input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict, True)
    produced = output_dict[model.graph.output[0].name]
    # do forward pass in PyTorch/Brevitas
    input_tensor = torch.from_numpy(input_tensor).float()
    expected = cnv.forward(input_tensor).detach().numpy()
    assert np.isclose(produced, expected, atol=1e-3).all()
    assert np.argmax(produced) == 3

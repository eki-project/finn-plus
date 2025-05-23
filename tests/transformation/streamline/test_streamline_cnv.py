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
import torch
from brevitas.export import export_qonnx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
from finn.util.basic import make_build_dir
from finn.util.test import get_test_model_trained


@pytest.mark.streamline
# act bits
@pytest.mark.parametrize("abits", [1, 2])
# weight bits
@pytest.mark.parametrize("wbits", [1, 2])
# network topology / size
@pytest.mark.parametrize("size", ["CNV"])
def test_streamline_cnv(size, wbits, abits):
    if wbits > abits:
        pytest.skip("No wbits > abits cases at the moment")
    nname = "%s_%dW%dA" % (size, wbits, abits)
    export_onnx_path = make_build_dir("test_streamline_cnv_")
    finn_onnx = export_onnx_path + "/%s.onnx" % nname
    fc = get_test_model_trained(size, wbits, abits)
    export_qonnx(fc, torch.randn(1, 3, 32, 32), finn_onnx)
    qonnx_cleanup(finn_onnx, out_file=finn_onnx)
    model = ModelWrapper(finn_onnx)
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())
    # load one of the test vectors
    ref = importlib.files("finn.qnn-data") / "cifar10/cifar10-test-data-class3.npz"
    with importlib.as_file(ref) as fn:
        input_tensor = np.load(fn)["arr_0"].astype(np.float32)
    input_tensor = input_tensor / 255
    assert input_tensor.shape == (1, 3, 32, 32)
    # run using FINN-based execution
    input_dict = {"global_in": input_tensor}
    expected_ctx = oxe.execute_onnx(model, input_dict, True)
    expected = expected_ctx[model.graph.output[0].name]
    # model.save("orig_cnv.onnx")
    model = model.transform(Streamline())
    model = model.transform(RemoveUnusedTensors())
    assert len(model.graph.initializer) == 21
    assert len(model.graph.value_info) == 43
    # model.save("streamlined_cnv.onnx")
    assert len(model.graph.node) == 23
    produced_ctx = oxe.execute_onnx(model, input_dict, True)
    produced = produced_ctx[model.graph.output[0].name]
    assert np.isclose(expected, produced, atol=1e-3).all()
    assert model.graph.node[0].op_type == "MultiThreshold"
    assert np.argmax(produced) == 3

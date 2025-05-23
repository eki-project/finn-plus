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

import numpy as np
import os
import torch
from brevitas.export import export_qonnx
from PIL import Image
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
import finn.transformation.streamline.absorb as absorb
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.basic import make_build_dir
from finn.util.pytorch import NormalizePreProc
from finn.util.test import crop_center, get_test_model_trained, resize_smaller_side


@pytest.mark.brevitas_export
@pytest.mark.xfail
def test_brevitas_mobilenet():
    # get single image as input and prepare image
    img = Image.open(os.path.join(os.environ["FINN_TESTS"], "brevitas/king_charles.jpg"))
    # resize smallest side of the image to 256 pixels and resize larger side
    # with same ratio
    img = resize_smaller_side(256, img)
    # crop central 224*224 window
    img = crop_center(224, img)
    # save image as numpy array and as torch tensor to enable testing in
    # brevitas/pytorch and finn and transpose from (H, W, C) to (C, H, W)
    img_np = np.asarray(img).copy().astype(np.float32).transpose(2, 0, 1)
    img_np = img_np.reshape(1, 3, 224, 224)
    img_torch = torch.from_numpy(img_np).float()

    # export preprocess
    export_onnx_path = make_build_dir("test_brevitas_mobilenet-v1_")
    preproc_onnx = export_onnx_path + "/quant_mobilenet_v1_4b_preproc.onnx"
    mean = [0.485, 0.456, 0.406]
    std = 0.226
    ch = 3
    preproc = NormalizePreProc(mean, std, ch)
    export_qonnx(preproc, torch.randn(1, 3, 224, 224), preproc_onnx)
    qonnx_cleanup(preproc_onnx, out_file=preproc_onnx)
    preproc_model = ModelWrapper(preproc_onnx)
    preproc_model = preproc_model.transform(ConvertQONNXtoFINN())
    # set input finn datatype to UINT8
    preproc_model.set_tensor_datatype(preproc_model.graph.input[0].name, DataType["UINT8"])
    preproc_model = preproc_model.transform(InferShapes())
    preproc_model = preproc_model.transform(GiveUniqueNodeNames())
    preproc_model = preproc_model.transform(GiveUniqueParameterTensors())
    preproc_model = preproc_model.transform(GiveReadableTensorNames())

    finn_onnx = export_onnx_path + "/quant_mobilenet_v1_4b_exported.onnx"
    mobilenet = get_test_model_trained("mobilenet", 4, 4)
    export_qonnx(mobilenet, torch.randn(1, 3, 224, 224), finn_onnx)
    qonnx_cleanup(finn_onnx, out_file=finn_onnx)

    # do forward pass in PyTorch/Brevitas
    input_tensor = preproc.forward(img_torch)
    expected = mobilenet.forward(input_tensor).detach().numpy()
    expected_topk = expected.flatten()
    expected_top5 = np.argsort(expected_topk)[-5:]
    expected_top5 = np.flip(expected_top5)
    expected_top5_prob = []
    for index in expected_top5:
        expected_top5_prob.append(expected_topk[index])

    model = ModelWrapper(finn_onnx)
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(InsertTopK())
    # get initializer from Mul that will be absorbed into topk
    a0 = model.get_initializer(model.graph.node[-2].input[1])
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(GiveReadableTensorNames())
    model.save(export_onnx_path + "/quant_mobilenet_v1_4b_wo_preproc.onnx")
    model = model.transform(MergeONNXModels(preproc_model))
    model.save(export_onnx_path + "/quant_mobilenet_v1_4b.onnx")
    idict = {model.graph.input[0].name: img_np}
    odict = oxe.execute_onnx(model, idict, True)
    produced = odict[model.graph.output[0].name]
    produced_prob = odict["TopK_0_out0"] * a0
    assert (produced.flatten() == expected_top5).all()
    assert np.isclose(produced_prob.flatten(), expected_top5_prob, atol=2.2 * 1e-1).all()

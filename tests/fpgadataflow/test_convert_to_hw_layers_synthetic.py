# Copyright (C) 2024, Advanced Micro Devices, Inc.
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
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, SortGraph
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.streamline.absorb import (
    AbsorbConsecutiveTransposes,
    AbsorbScalarMulAddIntoTopK,
)
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedAdd,
    CollapseRepeatedMul,
)
from finn.transformation.streamline.reorder import MoveAddPastMul, MoveScalarLinearPastInvariants
from finn.util.test import soft_verify_topk

export_onnx_path = "test_output_synthetic.onnx"

# construct a synthetic graph to test:
# topk insertion, topk conversion to hw, add conversion to hw
# graph should just be a sum


def make_model(ch, ifmdim):
    shape = [1, ch, ifmdim, ifmdim]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    inp1_add0_ct = helper.make_tensor_value_info("inp1_add0_ct", TensorProto.FLOAT, [1])
    inp1_add = helper.make_tensor_value_info("inp1_add", TensorProto.FLOAT, shape)
    inp1_add_ct = helper.make_tensor_value_info("inp1_add_ct", TensorProto.FLOAT, [1])
    inp2_add = helper.make_tensor_value_info("inp2_add", TensorProto.FLOAT, shape)
    inp2_add_ct = helper.make_tensor_value_info("inp2_add_ct", TensorProto.FLOAT, [1])
    inp1_mul = helper.make_tensor_value_info("inp1_mul", TensorProto.FLOAT, shape)
    inp1_mul_ct = helper.make_tensor_value_info("inp1_mul_ct", TensorProto.FLOAT, [1])
    inp2_mul = helper.make_tensor_value_info("inp2_mul", TensorProto.FLOAT, shape)
    inp2_mul_ct = helper.make_tensor_value_info("inp2_mul_ct", TensorProto.FLOAT, [1])
    eltwise_add = helper.make_tensor_value_info("eltwise_add", TensorProto.FLOAT, shape)
    pool = helper.make_tensor_value_info("pool", TensorProto.FLOAT, [1, ch, 1, 1])
    reshape_ct = helper.make_tensor_value_info("reshape_ct", TensorProto.INT64, [2])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ch])

    add0_node = helper.make_node("Add", [inp.name, inp1_add0_ct.name], ["out_add0"])
    add1_node = helper.make_node("Add", ["out_add0", inp1_add_ct.name], [inp1_add.name])
    add2_node = helper.make_node("Add", ["out_add0", inp2_add_ct.name], [inp2_add.name])
    mul1_node = helper.make_node("Mul", [inp1_add.name, inp1_mul_ct.name], [inp1_mul.name])
    mul2_node = helper.make_node("Mul", [inp2_add.name, inp2_mul_ct.name], [inp2_mul.name])
    eltwise_add_node = helper.make_node("Add", [inp1_mul.name, inp2_mul.name], [eltwise_add.name])
    globalavgpool_node = helper.make_node("GlobalAveragePool", [eltwise_add.name], [pool.name])
    reshape_node = helper.make_node("Reshape", [pool.name, reshape_ct.name], [outp.name])

    graph = helper.make_graph(
        nodes=[
            add0_node,
            add1_node,
            add2_node,
            mul1_node,
            mul2_node,
            eltwise_add_node,
            globalavgpool_node,
            reshape_node,
        ],
        name="graph",
        inputs=[inp],
        outputs=[outp],
    )

    model = qonnx_make_model(graph, producer_name="add-model")
    model = ModelWrapper(model)

    # set initializers for scalar add/mul nodes
    model.set_initializer(add0_node.input[1], np.array([0.0], dtype=np.float32))
    model.set_initializer(add1_node.input[1], np.array([7.0], dtype=np.float32))
    model.set_initializer(add2_node.input[1], np.array([8.0], dtype=np.float32))
    model.set_initializer(mul1_node.input[1], np.array([2.0], dtype=np.float32))
    model.set_initializer(mul2_node.input[1], np.array([2.0], dtype=np.float32))
    model.set_initializer(reshape_node.input[1], np.array([1, -1], dtype=np.int64))

    return model


# data types
@pytest.mark.parametrize("idt", [DataType["UINT2"]])
# channels
@pytest.mark.parametrize("ch", [16])
# ifmdim
@pytest.mark.parametrize("ifmdim", [5])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_convert_to_hw_layers_synthetic(ch, ifmdim, idt):
    model = make_model(ch, ifmdim)
    model.save(export_onnx_path)
    model = ModelWrapper(export_onnx_path, fix_float64=True)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataLayouts())
    # generate test vectors of correct shape
    if ifmdim == -1:
        input_tensor_shape = (1, ch)
    else:
        input_tensor_shape = (1, ch, ifmdim, ifmdim)

    x = gen_finn_dt_tensor(idt, input_tensor_shape)

    # generate expected value from streamlined net
    input_dict = {model.graph.input[0].name: x}

    output_dict = oxe.execute_onnx(model, input_dict, True)
    produced_sum = output_dict[model.graph.output[0].name]
    chw_mul = model.get_initializer(model.graph.node[-1].input[1])
    chw_mul = 1
    expected_sum = chw_mul * np.sum(2 * (2 * x + 15.0), axis=(2, 3)) / (ifmdim * ifmdim)
    assert (produced_sum.flatten() == expected_sum.flatten()).all()

    model = model.transform(InferDataLayouts())

    # convert to hw
    model.set_tensor_datatype(model.graph.input[0].name, idt)
    # extra streamlining
    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(MoveAddPastMul())
    model = model.transform(CollapseRepeatedMul())
    model = model.transform(CollapseRepeatedAdd())
    # insert top-k node, which should absorb linear ops before it

    model = model.transform(InferShapes())
    model = model.transform(InferDataLayouts())
    model = model.transform(InferDataTypes())

    model = model.transform(to_hw.InferChannelwiseLinearLayer())
    model = model.transform(to_hw.InferAddStreamsLayer())
    model = model.transform(to_hw.InferGlobalAccPoolLayer())
    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(InsertTopK())
    model = model.transform(AbsorbScalarMulAddIntoTopK())
    model = model.transform(InferDataTypes())
    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(AbsorbConsecutiveTransposes())
    model = model.transform(InferDataTypes())
    model = model.transform(to_hw.InferDuplicateStreamsLayer())

    model = model.transform(SortGraph())

    # check topology status

    finn_nodes = model.get_finn_nodes()
    assert len(finn_nodes) == 9
    add_nodes = model.get_nodes_by_op_type("AddStreams")
    assert len(add_nodes) == 1
    pool_nodes = model.get_nodes_by_op_type("GlobalAccPool")
    assert len(pool_nodes) == 1
    label_nodes = model.get_nodes_by_op_type("LabelSelect")
    assert len(label_nodes) == 1
    channelwise_nodes = model.get_nodes_by_op_type("ChannelwiseOp")
    assert len(channelwise_nodes) == 5
    dup_nodes = model.get_nodes_by_op_type("DuplicateStreams")
    assert len(dup_nodes) == 1

    output_hw = oxe.execute_onnx(model, input_dict, True)

    model = model.transform(SpecializeLayers("xc7z020clg400-1"))

    # check topology status

    finn_nodes = model.get_finn_nodes()
    assert len(finn_nodes) == 9
    add_nodes = model.get_nodes_by_op_type("AddStreams_hls")
    assert len(add_nodes) == 1
    pool_nodes = model.get_nodes_by_op_type("GlobalAccPool_hls")
    assert len(pool_nodes) == 1
    label_nodes = model.get_nodes_by_op_type("LabelSelect_hls")
    assert len(label_nodes) == 1
    channelwise_nodes = model.get_nodes_by_op_type("ChannelwiseOp_hls")
    assert len(channelwise_nodes) == 5
    dup_nodes = model.get_nodes_by_op_type("DuplicateStreams_hls")
    assert len(dup_nodes) == 1

    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    model = model.transform(SetExecMode("cppsim"))

    output_dict = oxe.execute_onnx(model, input_dict, True)

    # verify execution
    outp_name = model.graph.output[0].name
    # comparison before and after layer specialization
    assert (output_dict[outp_name] == output_hw[outp_name]).all()
    # comparison with golden output
    produced_topk_hls = output_dict[outp_name]
    topk_input = output_dict[model.graph.node[-1].input[0]]
    assert soft_verify_topk(topk_input, produced_topk_hls, 5)

    os.remove(export_onnx_path)

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
import copy
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.builder.build_dataflow_config import DataflowBuildConfig
from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors
from qonnx.transformation.general import (
    ConvertDivToMul,
    ConvertSubToAdd,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    RemoveStaticGraphInputs,
    SortGraph,
)
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
import finn.transformation.streamline.absorb as absorb


from finn.core.onnx_exec import execute_onnx
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from qonnx.transformation.extract_conv_bias import ExtractBiasFromConv
from finn.transformation.qonnx.quant_act_to_multithreshold import ConvertQuantActToMultiThreshold
from qonnx.transformation.insert_topk import InsertTopK
from onnx.helper import make_attribute
from qonnx.core.datatype import DataType

from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbConsecutiveTransposes,
    AbsorbTransposeIntoFlatten,
    AbsorbTransposeIntoMultiThreshold,
)
from qonnx.transformation.remove import RemoveIdentityOps, RemoveUnusedNodes
from qonnx.transformation.general import RemoveUnusedTensors
from finn.transformation.qonnx.fold_quant_weights import FoldQuantWeights
from finn.transformation.qonnx.extract_bias_quant import ExtractConvQuantBias
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.util.cleanup import cleanup_model

from qonnx.transformation.qcdq_to_qonnx import QCDQToQuant
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_shapes import InferShapes

from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from qonnx.util.basic import get_by_name

from finn.transformation.streamline.collapse_repeated import CollapseRepeatedAdd, CollapseRepeatedMul
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.sign_to_thres import ConvertSignToThres
import json

def auto_pad_to_explicit_padding(autopad_str, idim_h, idim_w, k_h, k_w, stride_h, stride_w, n_dims):
    pad_total_h = (stride_h - 1) * idim_h - stride_h + k_h
    pad_total_w = (stride_w - 1) * idim_w - stride_w + k_w
    pad_half_small_h = int((pad_total_h / 2))
    pad_half_small_w = int((pad_total_w / 2))
    pad_half_large_h = pad_total_h - pad_half_small_h
    pad_half_large_w = pad_total_w - pad_half_small_w
    if autopad_str == "VALID":
        return [0 for i in range(2 * n_dims)]
    elif autopad_str == "SAME_UPPER":
        return [pad_half_small_h, pad_half_small_w, pad_half_large_h, pad_half_large_w]
    elif autopad_str == "SAME_LOWER":
        return [pad_half_large_h, pad_half_large_w, pad_half_small_h, pad_half_small_w]
    else:
        raise Exception("Unsupported auto_pad: " + autopad_str)


def step_pre_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    conv_nodes = model.get_nodes_by_op_type("Conv")

    for conv_n in conv_nodes:
        attrs = conv_n.attribute

        for attr in attrs:
            if attr.name == "auto_pad":

                idim = model.get_tensor_shape(conv_n.input[0])
                [_, ifm_ch, ifm_dim_h, ifm_dim_w] = idim
                kshape = get_by_name(conv_n.attribute, "kernel_shape").ints
                stride_h = get_by_name(conv_n.attribute, "strides").ints[0]
                stride_w = get_by_name(conv_n.attribute, "strides").ints[1]
                pads = auto_pad_to_explicit_padding(
                    attr.s.decode("utf-8"),
                    ifm_dim_h,
                    ifm_dim_w,
                    kshape[0],
                    kshape[1],
                    stride_h,
                    stride_w,
                    len(model.get_tensor_shape(conv_n.input[0])) - 2,
                )
                new_attr = make_attribute("pads", pads)
                attr.s = b"NOTSET"
                attrs.append(new_attr)
        attr_t = type(attrs)

    model = model.transform(Change3DTo4DTensors())
    # model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    return model


def step_convert_final_layers(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(to_hw.InferChannelwiseLinearLayer())
    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(GiveUniqueNodeNames())
    return model


def step_softmax_to_topk(model: ModelWrapper, cfg: DataflowBuildConfig):
    # add topk node (https://github.com/Xilinx/finn/discussions/420#discussioncomment-1625473)
    softmax_node = model.graph.node[-1]
    softmax_in_tensor = model.get_tensor_valueinfo(softmax_node.input[0])
    softmax_out_tensor = model.get_tensor_valueinfo(softmax_node.output[0])
    model.graph.output.remove(softmax_out_tensor)
    model.graph.output.append(softmax_in_tensor)
    model.graph.node.remove(softmax_node)

    # remove redundant value_info for primary input/output
    # othwerwise, newer FINN versions will not accept the model
    if model.graph.input[0] in model.graph.value_info:
        model.graph.value_info.remove(model.graph.input[0])
    if model.graph.output[0] in model.graph.value_info:
        model.graph.value_info.remove(model.graph.output[0])

    # insert topK node in place of the final softmax node
    model = model.transform(InsertTopK(k=1))
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())

    return model


def step_extract_input_quant(model: ModelWrapper, cfg: DataflowBuildConfig):
    # model = model.transform(ConvertQuantActToMultiThreshold(lambda model, q_node: True)) #TODO this was used to bypass bitdiwth
    conv_nodes = model.get_nodes_by_op_type("Conv")

    for conv_n in conv_nodes:
        attrs = conv_n.attribute

        for attr in attrs:
            if attr.name == "auto_pad":
                attr.s = b"NOTSET"
                new_attr = make_attribute("pads", [1, 1])
                attrs.append(new_attr)

    # extract input quantization thresholds for sw-based quantization
    # (in case they were not fixed before training)
    input_mt_node = model.get_nodes_by_op_type("MultiThreshold")[0]
    input_mt_thresholds = model.get_initializer(input_mt_node.input[1])
    print("input quant thresholds")
    print(input_mt_thresholds)

    # preprocessing: remove input reshape/quantization from graph
    new_input_node = model.get_nodes_by_op_type("Mul")[0]
    new_input_tensor = model.get_tensor_valueinfo(new_input_node.input[0])
    old_input_tensor = model.graph.input[0]
    model.graph.input.remove(old_input_tensor)
    model.graph.input.append(new_input_tensor)
    new_input_index = model.get_node_index(new_input_node)
    del model.graph.node[0:new_input_index]

    # postprocessing: remove final softmax node from training
    softmax_node = model.graph.node[-1]
    softmax_in_tensor = model.get_tensor_valueinfo(softmax_node.input[0])
    softmax_out_tensor = model.get_tensor_valueinfo(softmax_node.output[0])
    model.graph.output.remove(softmax_out_tensor)
    model.graph.output.append(softmax_in_tensor)
    model.graph.node.remove(softmax_node)

    # remove redundant value_info for primary input/output
    # othwerwise, newer FINN versions will not accept the model
    if model.graph.input[0] in model.graph.value_info:
        model.graph.value_info.remove(model.graph.input[0])
    if model.graph.output[0] in model.graph.value_info:
        model.graph.value_info.remove(model.graph.output[0])

    # insert topK node in place of the final softmax node
    model = model.transform(InsertTopK(k=1))

    # manually set input datatype (not done by brevitas yet)
    finnonnx_in_tensor_name = model.graph.input[0].name
    finnonnx_model_in_shape = model.get_tensor_shape(finnonnx_in_tensor_name)
    model.set_tensor_datatype(finnonnx_in_tensor_name, DataType["INT8"])
    print("Input tensor name: %s" % finnonnx_in_tensor_name)
    print("Input tensor shape: %s" % str(finnonnx_model_in_shape))
    print("Input tensor datatype: %s" % str(model.get_tensor_datatype(finnonnx_in_tensor_name)))

    # model = model.transform(RemoveUnusedTensors())

    return model


def step_debug(model: ModelWrapper, cfg: DataflowBuildConfig):
    non_finn_nodes = model.get_non_finn_nodes()
    exec_context = model.make_empty_exec_context()
    nodes = model.graph.node

    first = nodes[1]

    check_node = model.find_producer("Quant_1_out0")
    tensor_list = model.get_all_tensor_names()

    in0 = model.get_initializer("Quant_1_param0")
    in1 = model.get_initializer("Quant_1_param1")
    in2 = model.get_initializer("Quant_29_param1")
    in3 = model.get_initializer("Quant_29_param2")

    input_dict = {0: in0, 1: in1, 2: in2, 3: in3}

    output_dict = execute_onnx(model, input_dict, True, check_node, check_node)
    weight_tensor = output_dict["Quant_1_out0"]

    model.set_initializer("Quant_1_out0", weight_tensor)
    # execute_onnx(model, None, False,

    return model


# def step_qonnx_to_finn(model: ModelWrapper, cfg: DataflowBuildConfig):
#     graph_out_name = model.graph.output[0].name

#     model = model.transform(ConvertQONNXtoFINN())

#     graph_out_name = model.graph.output[0].name
#     return model


def step_change_tensors(model: ModelWrapper, cfg: DataflowBuildConfig):
    tensors = []

    for n in model.graph.node:
        input_name = n.input[0]
        input_shape = model.get_tensor_shape(input_name)
        if input_shape is not None:
            if input_shape[0] == 512:
                new_shape = input_shape
                new_shape[0] = 1
                model.set_tensor_shape(input_name, new_shape)

        if n.op_type == "TopK":
            output_name = n.output[0]
            output_shape = model.get_tensor_shape(output_name)

            if output_shape[0] == 512:
                new_shape = output_shape
                new_shape[0] = 1
                model.set_tensor_shape(output_name, output_shape)

            output_name = n.output[1]
            output_shape = model.get_tensor_shape(output_name)

            if output_shape[0] == 512:
                new_shape = output_shape
                new_shape[0] = 1
                model.set_tensor_shape(output_name, output_shape)

        if n.op_type == "Softmax":
            output_name = n.output[0]
            output_shape = model.get_tensor_shape(output_name)

            if output_shape[0] == 512:
                new_shape = output_shape
                new_shape[0] = 1
                model.set_tensor_shape(output_name, output_shape)

    return model


def step_move_weight_into_conv(model: ModelWrapper, cfg: DataflowBuildConfig):
    conv_nodes = model.get_nodes_by_op_type("Conv")

    # get initializers per quant node that are responsible for initializing the weights of conv nodes
    weight_tensors = [conv_node.input[1] for conv_node in conv_nodes]
    weight_nodes = [model.find_producer(weight_tensor) for weight_tensor in weight_tensors]
    weight_node_inputs = [node.input for node in weight_nodes]
    weight_node_initializers = [
        [model.get_initializer(sin_input) for sin_input in node_inputs] for node_inputs in weight_node_inputs
    ]

    # get the output
    for node_idx, node in enumerate(weight_nodes):
        node_name = node.name
        input_dict = {idx: initializers for idx, initializers in enumerate(weight_node_initializers[node_idx])}
        output_dict = execute_onnx(model, input_dict, True, node, node)
        output = output_dict[node.output[0]]

        model.set_initializer(node.output[0], output)
        model.graph.node.remove(node)  # remove now useless node

    bias_tensors = [conv_node.input[2] for conv_node in conv_nodes]
    bias_nodes = [model.find_producer(bias_tensor) for bias_tensor in bias_tensors]
    bias_node_inputs = [node.input for node in bias_nodes]
    bias_node_initializers = [
        [model.get_initializer(sin_input) for sin_input in node_inputs] for node_inputs in bias_node_inputs
    ]

    for node_idx, node in enumerate(bias_nodes):
        node_name = node.name
        input_dict = {idx: initializers for idx, initializers in enumerate(bias_node_initializers[node_idx])}
        output_dict = execute_onnx(model, input_dict, True, node, node)
        output = output_dict[node.output[0]]

        model.set_initializer(node.output[0], output)
        model.graph.node.remove(node)

    model = model.transform(ExtractBiasFromConv())

    return model


def step_remove_input_quant(model: ModelWrapper, cfg: DataflowBuildConfig):
    # preprocessing: remove input reshape/quantization from graph
    new_input_node = model.get_nodes_by_op_type("Mul")[0]
    new_input_tensor = model.get_tensor_valueinfo(new_input_node.input[0])
    old_input_tensor = model.graph.input[0]
    model.graph.input.remove(old_input_tensor)
    model.graph.input.append(new_input_tensor)
    new_input_index = model.get_node_index(new_input_node)
    del model.graph.node[0:new_input_index]

    # remove redundant value_info for primary input/output
    # othwerwise, newer FINN versions will not accept the model
    if model.graph.input[0] in model.graph.value_info:
        model.graph.value_info.remove(model.graph.input[0])
    if model.graph.output[0] in model.graph.value_info:
        model.graph.value_info.remove(model.graph.output[0])

    # manually set input datatype (not done by brevitas yet)
    finnonnx_in_tensor_name = model.graph.input[0].name
    finnonnx_model_in_shape = model.get_tensor_shape(finnonnx_in_tensor_name)
    model.set_tensor_datatype(finnonnx_in_tensor_name, DataType["INT8"])
    print("Input tensor name: %s" % finnonnx_in_tensor_name)
    print("Input tensor shape: %s" % str(finnonnx_model_in_shape))
    print("Input tensor datatype: %s" % str(model.get_tensor_datatype(finnonnx_in_tensor_name)))

    return model


def step_pre_hw(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(RemoveUnusedNodes())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(AbsorbTransposeIntoMultiThreshold())

    return model


def step_pre_qonnx_to_finn(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = cleanup_model(model)
    # Make sure the datatypes exist, these are required for folding the weights
    model = model.transform(InferDataTypes())
    # # # Gemm operations are not supported by FINN, so we convert them to MatMul
    # model = model.transform(GemmToMatMul())
    # model = model.transform(FoldTransposeIntoQuantInit())
    # # Fold weights
    model = model.transform(BatchNormToAffine())
    model = model.transform(FoldQuantWeights())
    return model

    # Extract the bias from Conv node
    model = model.transform(ExtractBiasFromConv())

    return model


def step_qcdq_to_qonnx(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(QCDQToQuant())

    return model


def step_test_slice(model: ModelWrapper, cfg: DataflowBuildConfig):

    # Create the transformation
    slice_transformation = ApplySliceTransformation()

    # Apply the transformation to the model
    model, graph_modified = slice_transformation.apply(model)

    return model


import numpy as np
from qonnx.transformation.base import Transformation
import qonnx.core.onnx_exec as oxe


"""
residual block custom steps
"""
import finn.transformation.streamline.reorder as reorder
import finn.transformation.streamline.absorb as absorb


def step_residual_tidy(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    # model = model.transform(GemmToMatMul()) # for new model
    # model = model.transform(FoldTransposeIntoQuantInit()) # for new model
    model = model.transform(ExtractBiasFromConv())
    model = model.transform(FoldQuantWeights())
    # model = model.transform(FoldConstants())
    model.save("models/metaFi_test3.onnx")
    model = model.transform(ConvertQuantActToMultiThreshold())
    model.save("models/metaFi_test4.onnx")
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    # model = model.transform(InsertTopK())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    return model


def step_extract_absorb_bias(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(ExtractBiasFromConv())
    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())

    return model


def step_residual_topo(model: ModelWrapper, cfg: DataflowBuildConfig):
    tmp_output = copy.deepcopy(list(model.graph.output))  # deep copy protobuf array
    for node in tmp_output:
        if node.name != "global_out":
            print(f"remove node {node.name}")
            model.graph.output.remove(node)

    remove_node_names = ["Log_0", "Div_0", "Ceil_0"]  # hard code first
    remove_nodes = []
    for node in model.graph.node:
        if node.name in remove_node_names:
            remove_nodes.append(node)

    for node in remove_nodes:
        model.graph.node.remove(node)

    pre_slice_node_name = ""
    tmp_last_node = None

    # List to store nodes to be removed
    nodes_to_remove = []
    # Iterate through the nodes
    for index, node in enumerate(model.graph.node):
        if index != 0:
            tmp_last_node = model.graph.node[index - 1]
        if node.op_type == "Slice":
            nodes_to_remove.append(node)
            if pre_slice_node_name == "":
                pre_slice_node_name = tmp_last_node.name

    for node in model.graph.node:
        if node.name == "Add_3":  # Identify the 'Add' node or its index
            # Update the input of the 'Add' node to take input directly from 'Quant'
            node.input[1] = "Mul_6_out0"  # Use the correct name of the Quant output tensor
            # node.input[1] = pre_slice_node_name  # Use the correct name of the Quant output tensor

    # Remove the identified slice nodes
    for node in nodes_to_remove:
        model.graph.node.remove(node)

    return model


def step_residual_streamline_linear(model: ModelWrapper, cfg: DataflowBuildConfig):
    streamline_transformations = [
        # absorb.AbsorbScalarMulAddIntoTopK(),  # before MoveAddPastMul to avoid int->float
        ConvertSubToAdd(),
        ConvertDivToMul(),
        RemoveIdentityOps(),
        CollapseRepeatedMul(),
        # BatchNormToAffine(),
        ConvertSignToThres(),
        reorder.MoveAddPastMul(),
        reorder.MoveScalarAddPastMatMul(),
        reorder.MoveAddPastConv(),
        reorder.MoveScalarMulPastMatMul(),
        reorder.MoveScalarMulPastConv(),
        reorder.MoveScalarLinearPastInvariants(),
        reorder.MoveAddPastMul(),
        CollapseRepeatedAdd(),
        CollapseRepeatedMul(),
        absorb.AbsorbAddIntoMultiThreshold(),
        absorb.FactorOutMulSignMagnitude(),
        # reorder.MoveMaxPoolPastMultiThreshold(),
        reorder.MoveMulPastMaxPool(),
        absorb.AbsorbMulIntoMultiThreshold(),
        absorb.Absorb1BitMulIntoMatMul(),
        absorb.Absorb1BitMulIntoConv(),
        RoundAndClipThresholds(),
    ]
    for trn in streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
    return model


def step_residual_streamline_nonlinear(model: ModelWrapper, cfg: DataflowBuildConfig):
    streamline_transformations = [
        reorder.MoveLinearPastEltwiseAdd(),
        reorder.MoveLinearPastFork(),
    ]
    for trn in streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
    return model


def step_residual_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    for iter_id in range(4):
        model = step_residual_streamline_linear(model, cfg)
        model.save("models/Residual_no_slicing_final4.onnx")
        model = step_residual_streamline_nonlinear(model, cfg)
        model.save("models/Residual_no_slicing_final5.onnx")

        # big loop tidy up
        model = model.transform(RemoveUnusedTensors())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        model = model.transform(SortGraph())
        # model.save("models/Residual_no_slicing_final6.onnx")

    model = model.transform(DoubleToSingleFloat())

    return model



def step_residual_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(InferDataLayouts())
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferDataTypes())
    model = model.transform(SortGraph())
    to_hw_transformations = [
        to_hw.InferAddStreamsLayer,
        LowerConvsToMatMul,
        to_hw.InferChannelwiseLinearLayer,
        to_hw.InferPool,
        reorder.MoveTransposePastFork, # for new model
        AbsorbTransposeIntoMultiThreshold,
        RoundAndClipThresholds,
        to_hw.InferQuantizedMatrixVectorActivation,
        to_hw.InferThresholdingLayer,
        absorb.AbsorbConsecutiveTransposes,
        to_hw.InferConvInpGen,
        to_hw.InferDuplicateStreamsLayer,
        to_hw.InferLabelSelectLayer,
    ]
    for trn in to_hw_transformations:
        model = model.transform(trn())
        # model.save("models/Residual_no_slicing_final8.onnx")
        model = model.transform(InferDataLayouts())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataTypes())
        # model.save("models/Residual_no_slicing_final9.onnx")

    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(SortGraph())

    return model


def step_set_preferred_impl_style(model: ModelWrapper, cfg: DataflowBuildConfig):
  with open(cfg.output_dir + "/template_specialize_layers_config.json", 'r') as json_file:
    specialize_layers_config = json.load(json_file)
  
  # Update the "preferred_impl_style" for each MVAU entry
  for key in specialize_layers_config.keys():
      # if key.startswith("ConvolutionInputGenerator"):
      if cfg.use_conv_rtl:
        specialize_layers_config[key]["preferred_impl_style"] = "rtl"
      else:
        specialize_layers_config[key]["preferred_impl_style"] = "hls"

  # Save the updated JSON data back to the file
  with open(cfg.output_dir + "/template_specialize_layers_config.json", 'w') as file:
      json.dump(specialize_layers_config, file, indent=2)
  
  return model

class ApplySliceTransformation(Transformation):
    """
    Apply a Slice operation to the specified tensors in the model.
    The Slice operation extracts a sub-tensor from the input tensor.
    """

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph_modified = False
        nodes_to_remove = []
        # Infer the shapes of each tensor, remove unused tensors
        # and give each tensor a readable name
        model = model.transform(InferShapes())
        model = model.transform(RemoveUnusedTensors())

        graph = model.graph
        execution_context = model.make_empty_exec_context()

        previous_output_data = None
        slice_outputs = []

        for node in graph.node:

            # Apply the Slice operation only to the relevant nodes
            if node.op_type in ["Slice"]:

                input_name = node.input[0]
                output_name = node.output[0]

                if previous_output_data:
                    execution_context[input_name] = previous_output_data

                oxe.execute_node(node, execution_context, graph)
                print(f"Execution context output shape: {execution_context[output_name].shape}")
                slice_outputs.append(execution_context[output_name])

                # Update the tensor shape after slicing
                # sliced_shape = self._compute_sliced_shape(input_shape, axes, starts, ends, steps)

                for n in model.graph.node:
                    for i, input_name_in_node in enumerate(n.input):
                        if input_name_in_node == output_name:
                            n.input[i] = input_name
                            # model.set_tensor_shape(input_name, sliced_shape)

                nodes_to_remove.append(node)
                graph_modified = True

        # remove slice nodes
        for node in nodes_to_remove:
            model.graph.node.remove(node)

        return (model, graph_modified)

    def _compute_sliced_shape(self, input_shape, axes, starts, ends, steps):
        # def _compute_sliced_shape(self, input_shape):
        """
        Compute the shape of the tensor after slicing.
        """
        sliced_shape = list(input_shape)
        # for axis, start, end, step in zip(axes, starts, ends, steps):
        for axis, start, end, step in zip(axes, starts, ends, steps):
            sliced_shape[axis] = int((end - start) // step)
        return sliced_shape

    def _create_dummy_sum(self, input_list, sum_output):
        import onnx

        Sum_node = onnx.helper.make_node("Sum", inputs=input_list, outputs=[sum_output], name="Sum")




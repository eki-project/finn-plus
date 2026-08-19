# Copyright (C) 2023-2024, Advanced Micro Devices, Inc.
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

"""Convert Im2Col layers to ConvolutionInputGenerator HW layers."""

from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from typing import cast

from finn.util.exception import FINNUserError
from finn.util.logging import log


class InferConvInpGen(Transformation):
    """Convert Im2Col layers to ConvolutionInputGenerator layers."""

    def __init__(self) -> None:
        """Initialize the transformation."""
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transformation to infer ConvolutionInputGenerator layers."""
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Im2Col":
                i2c_input = n.input[0]
                i2c_output = n.output[0]
                if (i2c_in_shape := model.get_tensor_shape(i2c_input)) is None:
                    log.warning(f"{n.name}: Tensor Shape on input not set. Can't infer ConvInpGen.")
                    continue
                if (i2c_out_shape := model.get_tensor_shape(i2c_output)) is None:
                    log.warning(
                        f"{n.name}: Tensor Shape on output not set. Can't infer ConvInpGen."
                    )
                    continue
                dt = model.get_tensor_datatype(i2c_input)
                if not dt.is_integer():
                    log.warning(f"{n.name} : Input is not int. Can't infer ConvInpGen.")
                    continue
                i2c_inst = getCustomOp(n)
                stride_h, stride_w = cast("list[int]", i2c_inst.get_nodeattr("stride"))
                k_h, k_w = cast("list[int]", i2c_inst.get_nodeattr("kernel_size"))
                pad_attr = cast("list[int]", i2c_inst.get_nodeattr("pad_amount"))
                pad_h = pad_attr[0] + pad_attr[2]
                pad_w = pad_attr[1] + pad_attr[3]
                dilation_h, dilation_w = cast("list[int]", i2c_inst.get_nodeattr("dilations"))
                pad_val = i2c_inst.get_nodeattr("pad_value")
                depthwise = i2c_inst.get_nodeattr("depthwise")
                ifm_ch = i2c_in_shape[-1]
                ifm_dim_h = i2c_in_shape[1]
                ifm_dim_w = i2c_in_shape[2]
                ofm_dim_h = i2c_out_shape[1]
                ofm_dim_w = i2c_out_shape[2]

                # default params for ConvolutionInputGenerator
                conv_inp_gen_node_idx = node_ind
                conv_inp_gen_input = i2c_input
                conv_inp_gen_idim_h = ifm_dim_h
                conv_inp_gen_idim_w = ifm_dim_w

                if pad_h > 0 or pad_w > 0:
                    if pad_val != 0:
                        raise FINNUserError(
                            f"{n.name} : FMPadding_Batch doesn't currently support pad_val != 0"
                        )

                    odim_padding_h = ifm_dim_h + pad_h
                    odim_padding_w = ifm_dim_w + pad_w

                    padding_out = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        (1, odim_padding_h, odim_padding_w, ifm_ch),
                    )
                    graph.value_info.append(padding_out)
                    padding_out = padding_out.name
                    model.set_tensor_datatype(padding_out, dt)

                    conv_inp_gen_node_idx += 1
                    conv_inp_gen_input = padding_out
                    conv_inp_gen_idim_h = odim_padding_h
                    conv_inp_gen_idim_w = odim_padding_w

                    padding_node = helper.make_node(
                        "FMPadding",
                        [i2c_input],
                        [padding_out],
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        ImgDim=[ifm_dim_h, ifm_dim_w],
                        Padding=pad_attr,
                        NumChannels=ifm_ch,
                        inputDataType=dt.name,
                        SIMD=ifm_ch,
                        name="FMPadding_Batch_" + n.name,
                    )
                    graph.node.insert(node_ind, padding_node)

                is_1d = (ifm_dim_h == 1) or (ifm_dim_w == 1)
                conv_inp_gen_node = helper.make_node(
                    "ConvolutionInputGenerator",
                    [conv_inp_gen_input],
                    [i2c_output],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    ConvKernelDim=[k_h, k_w],
                    IFMChannels=ifm_ch,
                    IFMDim=[conv_inp_gen_idim_h, conv_inp_gen_idim_w],
                    OFMDim=[ofm_dim_h, ofm_dim_w],
                    SIMD=ifm_ch,
                    Stride=[stride_h, stride_w],
                    Dilation=[dilation_h, dilation_w],
                    inputDataType=dt.name,
                    outputDataType=dt.name,
                    depthwise=depthwise,
                    is1D=is_1d,
                    name="ConvolutionInputGenerator_" + n.name,
                )
                graph.node.insert(conv_inp_gen_node_idx, conv_inp_gen_node)
                # remove old nodes
                graph.node.remove(n)
                graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

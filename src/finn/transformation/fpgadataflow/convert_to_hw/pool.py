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

"""Replace Pool layers with Im2col + Pool HW layer combinations when kernel_shape > strides."""

from onnx import AttributeProto, NodeProto, TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name
from typing import TYPE_CHECKING, cast

from finn.util.basic import getHWCustomOp
from finn.util.exception import FINNInternalError, FINNUserError

if TYPE_CHECKING:
    import numpy as np
    from qonnx.custom_op.general.quantavgpool2d import QuantAvgPool2d


class InferPool(Transformation):
    """If kernel_shape > strides, replace Pool layer with Im2col + pool combination.

    When kernel_shape > strides, replaces Pool layer with Im2col followed by
    pool (with kernel_shape == strides), plus Transpose layers to keep the
    original data layout.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply transformation to convert Pool operations with kernel_shape > strides."""
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type in ["MaxPool", "QuantAvgPool2d", "MaxPoolNHWC"]:
                node_input = node.input[0]
                ishape = model.get_tensor_shape(node_input)
                node_output = node.output[0]
                idt = model.get_tensor_datatype(node_input)
                oshape = model.get_tensor_shape(node_output)
                if ishape is None or oshape is None:
                    raise FINNUserError(
                        f"{node.name}: Expected shape information to be present during InferPool."
                    )
                # only support 4D input tensors (1D convs need extra dummy dim)
                if len(ishape) != 4:
                    continue

                # extract pool parameters
                kh = kw = sh = sw = 0
                dlayout = "None"
                if node.op_type == "MaxPool":
                    kh, kw = list(
                        cast("AttributeProto", get_by_name(node.attribute, "kernel_shape")).ints
                    )
                    sh, sw = list(
                        cast("AttributeProto", get_by_name(node.attribute, "strides")).ints
                    )
                    dlayout = "NCHW"
                elif node.op_type == "QuantAvgPool2d":
                    inst = getCustomOp(node)
                    # QuantAvgPool2d has a single scalar attribute
                    # for kernel size and stride (implicit square)
                    kh = kw = cast("int", inst.get_nodeattr("kernel"))
                    sh = sw = cast("int", inst.get_nodeattr("stride"))
                    dlayout = cast("str", inst.get_nodeattr("data_layout"))
                elif node.op_type == "MaxPoolNHWC":
                    inst = getCustomOp(node)
                    kh, kw = cast("np.ndarray", inst.get_nodeattr("kernel_shape"))
                    sh, sw = cast("np.ndarray", inst.get_nodeattr("strides"))
                    dlayout = "NHWC"
                try:
                    pad = list(cast("AttributeProto", get_by_name(node.attribute, "pads")).ints)
                except AttributeError:
                    pad = [0, 0, 0, 0]

                if not idt.is_integer():
                    continue

                if (kh < sh) or (kw < sw):
                    # TODO check/implement swg support
                    continue

                odt = model.get_tensor_datatype(node_output)

                if dlayout == "NCHW":
                    _, ifm_ch, ifm_h, ifm_w = ishape
                    _, ofm_ch, ofm_h, ofm_w = oshape
                elif dlayout == "NHWC":
                    _, ifm_h, ifm_w, ifm_ch = ishape
                    _, ofm_h, ofm_w, ofm_ch = oshape
                else:
                    raise FINNInternalError("Unknown dlayout: " + str(dlayout))

                # if data layout NCHW, we need transpose nodes surrounding
                # the hw layer
                inp_trans_out = None
                pool_output = ""
                if dlayout == "NCHW":
                    # create new intermediate values
                    inp_trans_out = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        (1, ifm_h, ifm_w, ifm_ch),  # NHWC
                    )
                    graph.value_info.append(inp_trans_out)
                    inp_trans_out = inp_trans_out.name
                    model.set_tensor_datatype(inp_trans_out, idt)

                    pool_output = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        (1, ofm_h, ofm_w, ofm_ch),
                    )
                    graph.value_info.append(pool_output)
                    pool_output = pool_output.name

                im2col_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    (1, ofm_h, ofm_w, ifm_ch * kh * kw),
                )
                graph.value_info.append(im2col_out)
                im2col_out = im2col_out.name
                model.set_tensor_datatype(im2col_out, idt)

                # create new nodes
                inp_trans_node = None
                if dlayout == "NCHW":
                    # NCHW -> NHWC
                    inp_trans_node = helper.make_node(
                        "Transpose", [node_input], [cast("str", inp_trans_out)], perm=[0, 2, 3, 1]
                    )
                    im2col_in = cast("str", inp_trans_out)
                else:
                    im2col_in = node_input
                    pool_output = node_output

                accum_bits = 0
                pool_size_param = 0  # will be overridden if neededs
                pad_value = 0
                if node.op_type in ["MaxPool", "MaxPoolNHWC"]:
                    pool_fxn = "MaxPool"
                    odt = idt
                    pad_value = idt.min()
                elif node.op_type == "QuantAvgPool2d":
                    assert odt.is_integer(), """Output data type for QuantAvgPool2d
                    needs to be integer"""
                    assert all(x == 0 for x in pad), "Padding is not supported for QuantAvgPool2d"
                    inst = cast("QuantAvgPool2d", getHWCustomOp(node))
                    pool_fxn = "QuantAvgPool"
                    pool_size_param = inst.get_shifts()
                    accum_bits = inst.get_accum_size()

                else:
                    raise Exception(f"pad_value and pool_fxn not configured for {node.op_type}")

                # format input tensor
                im2col_node = helper.make_node(
                    "Im2Col",
                    [im2col_in],
                    [im2col_out],
                    domain="qonnx.custom_op.general",
                    stride=[sh, sw],
                    kernel_size=[kh, kw],
                    pad_amount=pad,
                    pad_value=pad_value,
                    depthwise=1,
                    input_shape=f"(1,{ifm_h},{ifm_w},{ifm_ch})",
                    name="Im2Col_" + node.name,
                )

                # Warning PE has to be equal to ifm_ch until Im2Col is replaced by
                # ConvolutionInputGenerator with depthwise=1.
                # For other settings the output will be incorrect due to incorrect input
                # data layout
                pool_node = helper.make_node(
                    "Pool",
                    [im2col_out],
                    [pool_output],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    InputDataType=idt.name,
                    OutputDataType=odt.name,
                    Channels=ifm_ch,
                    PE=ifm_ch,
                    KernelSize=[kh, kw],
                    Function=pool_fxn,
                    OutImgDims=[ofm_h, ofm_w],
                    AccumBits=accum_bits,
                    Size=pool_size_param,
                    BatchSize=1,
                    cpp_interface="hls_vector",
                    name="Pool_" + node.name,
                )

                out_trans_node = None
                if dlayout == "NCHW":
                    # NHWC -> NCHW
                    out_trans_node = helper.make_node(
                        "Transpose", [pool_output], [node_output], perm=[0, 3, 1, 2]
                    )

                # insert nodes where the conv is to preserve topological ordering
                if dlayout == "NCHW":
                    graph.node.insert(node_ind, cast("NodeProto", inp_trans_node))
                    graph.node.insert(node_ind + 1, im2col_node)
                    graph.node.insert(node_ind + 2, pool_node)
                    graph.node.insert(node_ind + 3, cast("NodeProto", out_trans_node))
                else:
                    graph.node.insert(node_ind, im2col_node)
                    graph.node.insert(node_ind + 1, pool_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

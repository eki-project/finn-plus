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

"""Convert MatMul layers belonging to depthwise convolutions to VVAU HW layers."""

import numpy as np
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from typing import cast

from finn.util.exception import FINNUserError


class InferVectorVectorActivation(Transformation):
    """Convert MatMul layers to VectorVectorActivation layers for depthwise convolutions.

    Converts MatMul layers with quantized inputs and weights to VectorVectorActivation
    layers, if the sparsity annotation of the weight matrix indicates that the MatMul
    layer belongs to a depthwise convolution. Any immediately following MultiThreshold
    layers will also be absorbed into the VVAU.
    """

    def __init__(self) -> None:
        """Initialize the transformation."""
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply transformation to convert MatMul to VVAU nodes for depthwise convolutions."""
        graph = model.graph
        graph_modified = False
        for node_ind, n in enumerate(graph.node, start=1):
            if n.op_type != "MatMul":
                continue
            if (sparsity := model.get_tensor_sparsity(n.input[1])) is None:
                continue
            try:
                k_h, k_w = sparsity["dw"]["kernel_shape"]
            except KeyError as e:
                raise FINNUserError(
                    f"{n.name}: sparsity annotation doesn't indicate that MatMul "
                    "belongs to a depthwise convolution."
                ) from e

            mm_input = n.input[0]
            mm_weight = n.input[1]
            mm_output = n.output[0]
            if (mm_in_shape := model.get_tensor_shape(mm_input)) is None or (
                mm_out_shape := model.get_tensor_shape(mm_output)
            ) is None:
                raise FINNUserError(f"{n.name}: Expected shape information to be available.")
            idt = model.get_tensor_datatype(mm_input)
            wdt = model.get_tensor_datatype(mm_weight)
            if idt.is_integer() and wdt.is_integer():
                mm_output = n.output[0]
                if (w := cast("np.ndarray | None", model.get_initializer(mm_weight))) is None:
                    raise FINNUserError(f"{n.name}: Cannot infer VVAU, weights are not static.")
                # infer dense weight tensor from sparse weight matrix
                # kernel size (k_h, k_w) which was extracted above and the value of
                # the channels is used.
                # the weight matrix has a shape of (k_h * k_w * Channels, Channels)
                # we need to reverse the creation of the sparse weight matrix
                # to achieve a weight tensor of shape (Channels, 1, k_h, k_w)
                channels = int(w.shape[1])
                # transpose to achieve a shape of (k_h * k_w * Channels, Channels)
                w = w.T
                # reshape to (Channels, k_h, k_w, Channels) to transpose afterwards
                # to (Channels, Channels, k_h, k_w)
                w = w.reshape(channels, k_h, k_w, channels)
                w = w.transpose(0, 3, 1, 2)
                # now we can extract the values using a for loop over the channels
                # and fill a zero numpy array in the correct shape
                w_tensor = np.zeros((channels, 1, k_h, k_w), dtype=np.float32)
                for ch in range(channels):
                    w_tensor[ch][0] = w[ch][ch]
                model.set_initializer(mm_weight, w_tensor)
                model.set_tensor_shape(mm_weight, (channels, 1, k_h, k_w))
                # create node with pe=channels as default
                pe = channels
                # see if we have any following thresholds
                consumers = model.find_consumers(mm_output)
                # Only a single consumer node can be absorbed. Absorbing one
                # branch of a forking matmul would lead to detached nodes
                # breaking the graph.
                consumer = consumers[0] if len(consumers) == 1 else None
                if consumer is not None and consumer.op_type == "MultiThreshold":
                    # create VVAU (i.e. including activation)
                    mt_output = consumer.output[0]
                    if (mt_out_shape := model.get_tensor_shape(mt_output)) is None:
                        raise FINNUserError(
                            f"{consumer.name}: Expected shape information to be available."
                        )
                    mt_thres = consumer.input[1]
                    if (t := cast("np.ndarray | None", model.get_initializer(mt_thres))) is None:
                        raise FINNUserError(
                            f"{consumer.name}: Cannot infer VVAU, thresholds are not static."
                        )
                    if t.shape[0] != 1 and t.shape[0] != channels:
                        raise FINNUserError(
                            f"{consumer.name}: First dimension of thresholds neither 1 nor "
                            "Channels."
                        )
                    odt = model.get_tensor_datatype(mt_output)
                    scale = getCustomOp(consumer).get_nodeattr("out_scale")
                    if scale != 1.0:
                        raise FINNUserError(
                            f"{consumer.name}: out_scale must be equal to 1.0 for HLS conversion."
                        )
                    actval = cast("float", getCustomOp(consumer).get_nodeattr("out_bias"))
                    if int(actval) != actval:
                        raise FINNUserError(
                            f"{consumer.name}: out_bias must be integer for HLS conversion."
                        )
                    actval = int(actval)
                    if odt.signed() and actval >= 0:
                        raise FINNUserError(f"{consumer.name}: Signed output requires actval < 0")
                    model.set_tensor_shape(mm_input, mm_in_shape)
                    model.set_tensor_shape(mt_output, mt_out_shape)
                    # create and insert new VectorVectorActivation node
                    new_node = helper.make_node(
                        "VVAU",
                        [mm_input, mm_weight, mt_thres],
                        [mt_output],
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        PE=pe,
                        Dim=[mm_in_shape[1], mm_in_shape[2]],
                        Channels=channels,
                        Kernel=[k_h, k_w],
                        inputDataType=idt.name,
                        weightDataType=wdt.name,
                        outputDataType=odt.name,
                        ActVal=actval,
                        noActivation=0,
                        name="VVAU_" + n.name,
                    )
                    graph.node.insert(node_ind, new_node)
                    # remove old nodes
                    graph.node.remove(n)
                    graph.node.remove(consumer)
                    graph_modified = True
                else:
                    # no activation, matmul only
                    odt = model.get_tensor_datatype(mm_output)
                    model.set_tensor_shape(mm_input, mm_in_shape)
                    model.set_tensor_shape(mm_output, mm_out_shape)
                    # create and insert new VVAU node
                    new_node = helper.make_node(
                        "VVAU",
                        [mm_input, mm_weight],
                        [mm_output],
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        PE=pe,
                        Dim=[mm_in_shape[1], mm_in_shape[2]],
                        Channels=channels,
                        Kernel=[k_h, k_w],
                        inputDataType=idt.name,
                        weightDataType=wdt.name,
                        outputDataType=odt.name,
                        ActVal=0,
                        noActivation=1,
                        name="VVAU_" + n.name,
                    )
                    graph.node.insert(node_ind, new_node)
                    # remove old node
                    graph.node.remove(n)
                    graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

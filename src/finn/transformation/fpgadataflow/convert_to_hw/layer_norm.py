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

"""Convert LayerNormalization nodes to LayerNorm HW layers."""

import numpy as np
import qonnx.core.data_layout
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.onnx import nchw_to_nhwc
from typing import Literal, cast

from finn.util.exception import FINNUserError
from finn.util.logging import log


class InferLayerNorm(Transformation):
    """Convert LayerNorm into HW, only norming over channel dim.
    This transform is adapted from Brainsmith InferLayerNorm."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, Literal[False]]:
        """Apply the transformation."""
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "LayerNormalization":
                scale = cast("np.ndarray | None", model.get_initializer(node.input[1]))
                bias = (
                    cast("np.ndarray|None", model.get_initializer(node.input[2]))
                    if len(node.input) > 2
                    else None
                )
                scale_is_one = (scale == 1).all()
                bias_is_zero = not np.any(bias) if bias is not None else None
                if not (scale_is_one and (bias_is_zero or bias is not None)):
                    log.warning(
                        f"""{node.name}: Scale is not one or bias is not zero.
                        Can't be converted to HWCustomOp. Please run ExtractNormScaleBias first."""
                    )
                    continue
                act_in = node.input[0]
                act_out = node.output[0]
                # Get any shape info that needs reuse
                if (shape_in := model.get_tensor_shape(act_in)) is None:
                    raise FINNUserError(
                        f"{node.name}: No shape information avaible. Infer layer norm failed."
                    )

                # Get datatypes
                idt = model.get_tensor_datatype(act_in)
                odt = model.get_tensor_datatype(act_out)

                norm_axis = helper.get_node_attr_value(node, "axis")
                if model.get_tensor_layout(act_in) == qonnx.core.data_layout.NCHW:
                    act_in = nchw_to_nhwc(act_in, model, node_ind)
                    node_ind += 1
                    shape_in = cast("list[int]", model.get_tensor_shape(act_in))
                    # shift axis for norm appropriately
                    norm_axis = (norm_axis + 2) % 4
                ch = shape_in[-1]

                # keep track of where we need to insert the HLS Op
                # it has to be ahead of the output transform
                insert_point = node_ind
                if model.get_tensor_layout(act_out) == qonnx.core.data_layout.NCHW:
                    act_out = nchw_to_nhwc(act_out, model, node_ind, reverse=True)
                    node_ind += 1

                # Check if 1D, norming on channel axis
                if not (norm_axis == -1 or norm_axis == len(shape_in) - 1):
                    continue

                # create node with no parallelization first
                simd = 1
                assert ch % simd == 0, "Requirement IFC divisable by PE is violated."
                # create and insert nodes
                new_node = helper.make_node(
                    "LayerNorm",
                    [act_in],
                    [act_out],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    SIMD=simd,
                    ifm_dim=shape_in,
                    epsilon=helper.get_node_attr_value(node, "epsilon"),
                    inputDataType=idt.name,
                    outputDataType=odt.name,
                    name="LayerNorm_" + node.name,
                )
                graph.node.insert(insert_point, new_node)
                # remove old node
                graph.node.remove(node)

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

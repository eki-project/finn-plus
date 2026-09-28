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

"""Convert scalar Gather (token selection) nodes into SelectToken HW layers."""

import numpy as np
from onnx import helper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name


class InferSelectTokenLayer(Transformation):
    """Convert a scalar Gather on the token axis into SelectToken."""

    def apply(self, model):
        """Replace scalar Gather nodes on the token axis by SelectToken nodes."""
        graph = model.graph
        graph_modified = False
        for node_ind, node in enumerate(list(graph.node)):
            if node.op_type != "Gather" or len(node.input) != 2:
                continue
            axis_attr = get_by_name(node.attribute, "axis")
            axis = axis_attr.i if axis_attr is not None else 0
            seq_shape = model.get_tensor_shape(node.input[0])
            if seq_shape is None or len(seq_shape) != 3:
                continue
            if axis < 0:
                axis += len(seq_shape)
            if axis != 1 or seq_shape[0] != 1:
                continue

            indices = model.get_initializer(node.input[1])
            if indices is None or indices.ndim != 0 or not np.issubdtype(indices.dtype, np.integer):
                continue
            token_index = int(indices.item())
            num_tokens = int(seq_shape[1])
            if not -num_tokens <= token_index < num_tokens:
                continue

            idt = model.get_tensor_datatype(node.input[0])
            odt = model.get_tensor_datatype(node.output[0])
            if idt is None or (odt is not None and odt != idt):
                continue
            new_node = helper.make_node(
                "SelectToken",
                [node.input[0]],
                node.output,
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                name="SelectToken_" + node.name,
                NumTokens=num_tokens,
                NumChannels=int(seq_shape[2]),
                TokenIndex=token_index,
                SIMD=1,
                inputDataType=idt.name,
                outputDataType=idt.name,
            )
            graph.node.insert(node_ind, new_node)
            graph.node.remove(node)
            graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

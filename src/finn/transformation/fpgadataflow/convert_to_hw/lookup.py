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

"""Convert Gather nodes with a constant first operand into Lookup HW layers."""

from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name
from typing import TYPE_CHECKING, cast

from finn.util.exception import FINNUserError

if TYPE_CHECKING:
    import numpy as np


class InferLookupLayer(Transformation):
    """Convert Gather nodes with constant op0 into Lookup HW layers."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply transformation to convert Gather operations to Lookup hardware layers.

        This transformation identifies Gather operations with constant first operand
        and converts them to FINN's custom Lookup nodes for hardware acceleration.
        """
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "Gather":
                emb_name = node.input[0]
                embs = cast("np.ndarray|None", model.get_initializer(emb_name))
                axis = get_by_name(node.attribute, "axis")
                # skip conversion if input0 is not constant
                if embs is None:
                    continue
                # skip conversion if axis != 0
                if axis is not None and axis.i != 0:
                    continue
                ind_name = node.input[1]
                ind_dtype = model.get_tensor_datatype(ind_name)
                emb_dtype = model.get_tensor_datatype(emb_name)
                # skip conversion if inputs are not unsigned integers
                if (not ind_dtype.is_integer()) or ind_dtype.signed():
                    continue
                num_embs, emb_dim = embs.shape
                out_name = node.output[0]
                if (ishape := model.get_tensor_shape(node.input[1])) is None:
                    raise FINNUserError(
                        f"{node.name}: Cannot infer Lookup Layer without shape information."
                    )
                # create and insert new Lookup node
                new_node = helper.make_node(
                    "Lookup",
                    [ind_name, emb_name],
                    [out_name],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    name="Lookup_" + node.name,
                    NumEmbeddings=num_embs,
                    EmbeddingDim=emb_dim,
                    EmbeddingType=emb_dtype.name,
                    InputType=ind_dtype.name,
                    InputShape=list(ishape),
                )
                graph.node.insert(node_ind, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

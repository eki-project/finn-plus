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

"""Convert GlobalAveragePool nodes to GlobalAccPool HW layers plus a scalar Mul."""

import numpy as np
import qonnx.core.data_layout
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.onnx import nchw_to_nhwc
from typing import cast

from finn.util.exception import FINNInternalError


class InferGlobalAccPoolLayer(Transformation):
    """Convert any GlobalAveragePool into a GlobalAccPool HW layer and a scalar Mul."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply transformation to infer GlobalAccPool hardware layers."""
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "GlobalAveragePool":
                in0 = node.input[0]
                result = node.output[0]
                if (in0_shape := model.get_tensor_shape(in0)) is None:
                    raise FINNInternalError(
                        "Expected shape information to exist during convert to hw."
                    )

                idt = model.get_tensor_datatype(in0)

                # skip conversion for layers with float input
                if not idt.is_integer():
                    continue

                # check layout and convert if necessary
                in0_layout = model.get_tensor_layout(in0)
                result_layout = model.get_tensor_layout(result)

                if in0_layout == qonnx.core.data_layout.NCHW:
                    in0 = nchw_to_nhwc(in0, model, node_ind)
                    node_ind += 1
                    if (in0_shape := model.get_tensor_shape(in0)) is None:
                        raise FINNInternalError(
                            "Expected shape information to exist during convert to hw."
                        )

                # keep track of where we need to insert the HW Op
                # it has to be ahead of the output transform
                insert_point = node_ind

                if result_layout == qonnx.core.data_layout.NCHW:
                    result = nchw_to_nhwc(result, model, node_ind, reverse=True)
                    node_ind += 1

                num_ch = int(in0_shape[-1])
                vecs = in0_shape[:-1]
                # create node with no parallelization first
                pe = 1

                # create an additional tensor of the same shape and layout as result
                out_shape = model.get_tensor_shape(result)
                pool_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, out_shape
                )
                model.graph.value_info.append(pool_out)
                pool_out = pool_out.name
                model.set_tensor_layout(
                    pool_out, cast("list[str]", model.get_tensor_layout(result))
                )

                new_pool = helper.make_node(
                    "GlobalAccPool",
                    [in0],
                    [pool_out],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    NumChannels=num_ch,
                    PE=pe,
                    inputDataType=idt.name,
                    numInputVectors=vecs,
                    name="GlobalAccPool_" + node.name,
                )

                mul_value = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, [1]
                )
                model.graph.value_info.append(mul_value)
                model.set_initializer(
                    mul_value.name, np.array(1 / (vecs[1] * vecs[2]), dtype=np.float32)
                )
                new_mul = helper.make_node(
                    "Mul",
                    [pool_out, mul_value.name],
                    [result],
                )
                graph.node.insert(insert_point, new_pool)
                graph.node.insert(insert_point + 1, new_mul)
                node_ind += 1
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

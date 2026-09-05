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

"""Convert Concat nodes operating on the last axis into StreamingConcat HW layers."""

from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name
from typing import cast

from finn.util.logging import log


class InferConcatLayer(Transformation):
    """Convert suitable Concat nodes (operating on last/-1 axis) into StreamingConcat HW layers."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply transformation to convert Concat operations to StreamingConcat hardware layers.

        This transformation identifies Concat operations operating on the last axis
        and converts them to FINN's custom StreamingConcat nodes.
        """
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "Concat":
                ishape = model.get_tensor_shape(node.input[0])
                axis = get_by_name(node.attribute, "axis")
                if (axis is None) or (ishape is None):
                    continue
                axis = axis.i
                last_axis = len(ishape) - 1
                # skip conversion if not using last axis
                if (axis != -1) and (axis != last_axis):
                    continue
                # check datatype coherence
                if any(model.get_tensor_datatype(x) is None for x in node.input):
                    log.warning(
                        "Inputs with undefined datatype detected, skipping InferConcatLayer()"
                    )
                    continue
                if any(model.get_tensor_shape(x) is None for x in node.input):
                    log.warning(
                        "Found input without shape information, skipping InferConcatLayer()"
                    )
                    continue
                # skip conversion if any inputs are static
                any_static = any(model.get_initializer(x) is not None for x in node.input)
                if any_static:
                    continue
                # skip conversion if inputs are not integers
                all_integer = all(model.get_tensor_datatype(x).is_integer() for x in node.input)
                if not all_integer:
                    log.warning(
                        "Inputs with non-integer datatype detected, skipping InferConcatLayer()"
                    )
                    continue
                # ready for conversion
                channels_per_stream = [
                    cast("list[int]", model.get_tensor_shape(x))[-1] for x in node.input
                ]
                inp_vec = list(ishape[:-1])
                new_node = helper.make_node(
                    "StreamingConcat",
                    node.input,
                    node.output,
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    name="StreamingConcat_" + node.name,
                    SIMD=1,
                    ChannelsPerStream=channels_per_stream,
                    inputDataTypes=[model.get_tensor_datatype(x).name for x in node.input],
                    numInputVectors=inp_vec,
                    inFIFODepths=[2] * len(node.input),
                    cpp_interface="hls_vector",
                    hls_style="freerunning",
                )
                graph.node.insert(node_ind, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

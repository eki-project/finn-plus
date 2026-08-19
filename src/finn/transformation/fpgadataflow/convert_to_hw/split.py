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

"""Convert Split nodes operating on the last axis into StreamingSplit HW layers."""

from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name
from typing import cast

from finn.util.logging import log


class InferSplitLayer(Transformation):
    """Convert suitable Split nodes (operating on last/-1 axis) into StreamingSplit HW layers."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply transformation to convert Split operations to StreamingSplit hardware layers.

        This transformation identifies Split operations operating on the last axis
        and converts them to FINN's custom StreamingSplit nodes.
        """
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "Split":
                split_param = node.input[1]
                if model.get_initializer(split_param) is None:
                    log.warning("Split param not constant, skipping InferSplitLayer()")
                    continue
                ishape = model.get_tensor_shape(node.input[0])
                axis = get_by_name(node.attribute, "axis")
                if (axis is None) or (ishape is None):
                    continue
                axis = axis.i
                last_axis = len(ishape) - 1
                # skip conversion if not using last axis
                if (axis != -1) and (axis != last_axis):
                    log.warning(
                        "StreamingSplit supports only last axis, skipping InferSplitLayer()"
                    )
                    continue
                # only one input allowed (two including split_param)
                if len(node.input) != 2:
                    log.warning("Only one input allowed, skipping InferSplitLayer()")
                    continue
                # skip conversion if the input is static
                if model.get_initializer(node.input[0]) is not None:
                    log.warning("Static input detected, skipping InferSplitLayer()")
                    continue
                # skip conversion if inputs are not integers
                if not model.get_tensor_datatype(node.input[0]).is_integer():
                    log.warning("Non-integer input detected, skipping InferSplitLayer()")
                    continue
                # ready for conversion
                out_shapes = [model.get_tensor_shape(x) for x in node.output]
                if any(s is None for s in out_shapes):
                    log.warning("Missing output shape information, skipping InferSplitLayer()")
                    continue
                channels_per_stream = [cast("list[int]", s)[-1] for s in out_shapes]
                inp_vec = list(ishape[:-1])
                # when creating the fpgadataflow node we remove the second parameter input
                new_node = helper.make_node(
                    "StreamingSplit",
                    [node.input[0]],
                    node.output,
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    name="StreamingSplit_" + node.name,
                    SIMD=1,
                    cpp_interface="hls_vector",
                    hls_style="freerunning",
                    ChannelsPerStream=channels_per_stream,
                    inputDataType=model.get_tensor_datatype(node.input[0]).name,
                    numInputVectors=inp_vec,
                    outFIFODepths=[2] * len(node.output),
                )
                graph.node.insert(node_ind, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

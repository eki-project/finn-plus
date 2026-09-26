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

"""Convert Where nodes into HWWhere HW layers."""

import numpy as np
from onnx import AttributeProto, helper
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

from finn.util.logging import log


class InferWhereLayer(Transformation):
    """Convert ONNX Where(condition, X, Y) into a streaming HWWhere layer."""

    @staticmethod
    def _warn_skip(node, reason):
        """Log a warning that the given Where node is skipped for the given reason."""
        node_name = node.name if node.name else "<unnamed Where>"
        log.warning("%s: %s. Can't infer HWWhere layer." % (node_name, reason))

    def apply(self, model):
        """Replace supported Where nodes by HWWhere nodes."""
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type != "Where":
                continue
            if len(node.input) != 3:
                self._warn_skip(node, "Expected exactly three inputs")
                continue

            cond_name, x_name, y_name = node.input
            if any(model.get_initializer(inp) is not None for inp in node.input):
                self._warn_skip(node, "Initializer inputs are not supported by the streaming op")
                continue

            cond_shape = model.get_tensor_shape(cond_name)
            x_shape = model.get_tensor_shape(x_name)
            y_shape = model.get_tensor_shape(y_name)
            out_shape = model.get_tensor_shape(node.output[0])
            if any(s is None for s in [cond_shape, x_shape, y_shape, out_shape]):
                self._warn_skip(node, "Input and output shapes must be known")
                continue
            all_dims = list(cond_shape) + list(x_shape) + list(y_shape) + list(out_shape)
            if any(x is None for x in all_dims):
                self._warn_skip(node, "Dynamic dimensions are not supported")
                continue
            try:
                broadcast_shape = np.broadcast_shapes(
                    tuple(cond_shape), tuple(x_shape), tuple(y_shape)
                )
            except ValueError:
                self._warn_skip(node, "Input shapes are not broadcast-compatible")
                continue
            if list(out_shape) != [int(x) for x in broadcast_shape]:
                self._warn_skip(node, "Output shape does not match the broadcast input shape")
                continue
            x_dt = model.get_tensor_datatype(x_name)
            y_dt = model.get_tensor_datatype(y_name)
            if x_dt is None or y_dt is None or x_dt != y_dt:
                self._warn_skip(node, "X and Y must have the same known datatype")
                continue
            supported_dt = (
                x_dt.is_integer()
                or x_dt.is_fixed_point()
                or x_dt in [DataType["FLOAT32"], DataType["FLOAT16"]]
            )
            if not supported_dt:
                self._warn_skip(node, "X and Y datatype %s is not supported" % str(x_dt))
                continue
            out_dt = model.get_tensor_datatype(node.output[0])
            if out_dt is not None and out_dt != x_dt:
                self._warn_skip(node, "Output datatype must match X and Y")
                continue

            cond_dt = model.get_tensor_datatype(cond_name)
            if cond_dt is None:
                model.set_tensor_datatype(cond_name, DataType["BINARY"])
                cond_dt = DataType["BINARY"]
            if cond_dt != DataType["BINARY"]:
                self._warn_skip(node, "Condition datatype must be BINARY")
                continue

            new_node = helper.make_node(
                "HWWhere",
                node.input,
                node.output,
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                name="HWWhere_" + node.name,
                CondRank=len(cond_shape),
                XRank=len(x_shape),
                YRank=len(y_shape),
                PE=1,
                conditionDataType=cond_dt.name,
                inputDataType=x_dt.name,
                outputDataType=x_dt.name,
                inFIFODepths=[2, 2, 2],
            )
            for attr_name, attr_value in [
                ("Shape", [int(x) for x in broadcast_shape]),
                ("CondShape", [int(x) for x in cond_shape]),
                ("XShape", [int(x) for x in x_shape]),
                ("YShape", [int(x) for x in y_shape]),
            ]:
                new_node.attribute.append(
                    helper.make_attribute(attr_name, attr_value, attr_type=AttributeProto.INTS)
                )
            graph.node.insert(node_ind, new_node)
            graph.node.remove(node)
            graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

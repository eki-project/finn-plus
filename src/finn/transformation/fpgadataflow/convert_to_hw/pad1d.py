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

"""Convert Concat-based 1D padding / CLS token insertion into Pad1D HW layers."""

import numpy as np
from onnx import helper, numpy_helper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name


class InferPad1DLayer(Transformation):
    """Convert 1D Concat padding patterns into Pad1D.

    This covers class-token insertion as ``Concat([cls_token, tokens], axis=1)``
    as well as constant left/right 1D padding around one streamed tensor.
    """

    def _make_pad_initializer(self, model, graph, values, idt):
        """Create a new float32 initializer holding the pad values and return its name."""
        values = np.asarray(values, dtype=np.float32)
        pad_name = model.make_new_valueinfo_name()
        graph.initializer.append(numpy_helper.from_array(values, name=pad_name))
        model.set_tensor_datatype(pad_name, idt)
        return pad_name

    def _make_or_reuse_pad_initializer(self, model, graph, values, tensor_names, idt):
        """Return the single existing pad tensor name or create a new pad initializer."""
        if len(tensor_names) == 1:
            return tensor_names[0]
        return self._make_pad_initializer(model, graph, values, idt)

    def apply(self, model):
        """Replace Concat nodes that pad one streamed tensor with constants by Pad1D nodes."""
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type != "Concat":
                continue

            axis = get_by_name(node.attribute, "axis")
            if axis is None:
                continue

            stream_inds = [
                ind for ind, inp in enumerate(node.input) if model.get_initializer(inp) is None
            ]
            if len(stream_inds) != 1:
                continue
            stream_ind = stream_inds[0]
            stream_name = node.input[stream_ind]

            stream_shape = model.get_tensor_shape(stream_name)
            if stream_shape is None:
                continue
            if any(x is None for x in stream_shape):
                continue

            rank = len(stream_shape)
            concat_axis = axis.i if axis.i >= 0 else axis.i + rank
            if rank != 3 or concat_axis != 1:
                continue
            if stream_shape[0] != 1:
                continue

            const_values = []
            valid_const_inputs = True
            for inp in node.input:
                init = model.get_initializer(inp)
                if init is None:
                    const_values.append(None)
                    continue

                const_shape = list(init.shape)
                if (
                    len(const_shape) != 3
                    or const_shape[0] != 1
                    or const_shape[2] != stream_shape[2]
                ):
                    valid_const_inputs = False
                    break
                const_values.append(np.asarray(init, dtype=np.float32))
            if not valid_const_inputs:
                continue

            left_values = [const_values[ind] for ind in range(stream_ind)]
            right_values = [const_values[ind] for ind in range(stream_ind + 1, len(node.input))]
            left_names = [node.input[ind] for ind in range(stream_ind)]
            right_names = [node.input[ind] for ind in range(stream_ind + 1, len(node.input))]
            pad_left = sum(x.shape[1] for x in left_values)
            pad_right = sum(x.shape[1] for x in right_values)
            if pad_left == 0 and pad_right == 0:
                continue

            out_shape = model.get_tensor_shape(node.output[0])
            exp_oshape = [1, stream_shape[1] + pad_left + pad_right, stream_shape[2]]
            if out_shape is not None and list(out_shape) != exp_oshape:
                continue

            idt = model.get_tensor_datatype(stream_name)
            if idt is None or not idt.is_integer():
                continue

            const_dtypes_valid = True
            missing_const_dtypes = []
            for inp in node.input:
                if inp == stream_name:
                    continue
                pad_dt = model.get_tensor_datatype(inp)
                if pad_dt is None:
                    missing_const_dtypes.append(inp)
                elif pad_dt != idt:
                    const_dtypes_valid = False
                    break
            if not const_dtypes_valid:
                continue

            if pad_left == 0:
                left_pad = np.zeros((1, 1, stream_shape[2]), dtype=np.float32)
            else:
                left_pad = np.concatenate(left_values, axis=1)
            if pad_right == 0:
                right_pad = np.zeros((1, 1, stream_shape[2]), dtype=np.float32)
            else:
                right_pad = np.concatenate(right_values, axis=1)

            left_pad_name = self._make_or_reuse_pad_initializer(
                model, graph, left_pad, left_names, idt
            )
            right_pad_name = self._make_or_reuse_pad_initializer(
                model, graph, right_pad, right_names, idt
            )

            new_node = helper.make_node(
                "Pad1D",
                [stream_name, left_pad_name, right_pad_name],
                node.output,
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                name="Pad1D_" + node.name,
                NumTokens=int(stream_shape[1]),
                NumChannels=int(stream_shape[2]),
                PadLeft=int(pad_left),
                PadRight=int(pad_right),
                SIMD=1,
                inputDataType=idt.name,
                outputDataType=idt.name,
            )
            for inp in missing_const_dtypes:
                model.set_tensor_datatype(inp, idt)
            graph.node.insert(node_ind, new_node)
            graph.node.remove(node)
            graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

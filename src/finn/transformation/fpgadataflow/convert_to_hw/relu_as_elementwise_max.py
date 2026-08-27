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

"""Convert ReLU into ElementwiseMaximum(in, 0) HW layers."""

import numpy as np
from onnx import NodeProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from typing import TYPE_CHECKING

from finn.transformation.fpgadataflow.convert_to_hw.elementwise_binary_operation import (
    lift_to_rank1,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class InferReLUAsElementwiseMax(Transformation):
    """Converts ReLU into ElementwiseMaximum(in, 0)."""

    @staticmethod
    def reject_unsupported_dtypes(model: ModelWrapper, node: NodeProto) -> bool:
        """Filter function to filter out any operation involving any floating-point tensor."""

        def dtype_ok(tname: str) -> bool:
            """Check if a datatype is okay."""
            dt = model.get_tensor_datatype(tname)
            if dt is None:
                return False
            return bool(
                dt.is_integer()
                or dt.is_fixed_point()
                or dt in [DataType["FLOAT32"], DataType["FLOAT16"]]
            )

        return all(dtype_ok(tname) for tname in list(node.input) + list(node.output))

    def __init__(self, _filter: "Callable[..., bool] | None" = reject_unsupported_dtypes) -> None:
        """Initialize the transformation method with an optional filter function."""
        # Initialize the base class Transformation object
        super().__init__()
        # Register the filter function as attribute
        self._filter = _filter if _filter is not None else lambda *_: True

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transformation."""
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Skip transforming nodes rejected by the filter
            if not self._filter(model, node):
                continue
            if node.op_type == "Relu":
                inp = node.input[0]
                # add a second 0-valued input for ReLU
                new_tname = model.make_new_valueinfo_name()
                model.set_initializer(new_tname, np.asarray(0.0, dtype=np.float32))
                # comparison of fp16 and uint2 is not possible in HLS
                new_tdtype = (
                    "FLOAT16"
                    if model.get_tensor_datatype(inp).get_canonical_name() == "FLOAT16"
                    else "UINT2"
                )
                # for the constant 0 input, use a small-width datatype
                # (to avoid unnecessarily promoting output type to something larger)
                model.set_tensor_datatype(new_tname, DataType[new_tdtype])
                result = node.output[0]

                # Need to "lift" potential scalar inputs to rank-1 tensors
                lift_to_rank1(inp, model)
                lift_to_rank1(new_tname, model)

                new_node = helper.make_node(
                    "ElementwiseMax",
                    [inp, new_tname],
                    [result],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    lhs_shape=model.get_tensor_shape(inp),
                    rhs_shape=model.get_tensor_shape(new_tname),
                    out_shape=model.get_tensor_shape(result),
                    lhs_dtype=str(model.get_tensor_datatype(inp)),
                    rhs_dtype=str(model.get_tensor_datatype(new_tname)),
                    out_dtype=str(model.get_tensor_datatype(result)),
                )
                graph.node.insert(index + 1, new_node)
                graph.node.remove(node)

                # Consider the graph to be modified, triggering exhaustive
                # re-application of this transformation
                graph_modified = True
                # Exiting here triggers type and shape inference and cleanup
                # after each transformed node. This helps QONNX to behave
                # better / more consistent in certain cases...
                break
        # Re-do shape and data type annotations after potential changes to the
        # model graph
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified

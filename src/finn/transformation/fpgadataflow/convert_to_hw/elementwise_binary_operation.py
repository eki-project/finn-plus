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

"""Convert supported elementwise binary operations to FINN custom HW ops."""

import numpy as np
from onnx import NodeProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from typing import TYPE_CHECKING, cast

# Module containing specializations of elementwise binary operations
import finn.custom_op.fpgadataflow.elementwise_binary as elementwise_binary
from finn.util.exception import FINNInternalError, FINNUserError

if TYPE_CHECKING:
    from collections.abc import Callable


def lift_to_rank1(name: str, model: ModelWrapper) -> None:
    """Lift scalar to rank-1 tensor.

    Converts scalar tensors (shape []) to rank-1 tensors with a single element (shape [1]).
    """
    if (shape := model.get_tensor_shape(name)) is None:
        raise FINNInternalError(
            "Cannot lift tensor to rank 1, because no shape information is available."
        )
    # Scalars have a shape of lengths zero
    if len(shape) == 0:
        # Lift shape to rank-1 tensor with single element
        model.set_tensor_shape(name, [1])
        # Check whether this tensor has an initializer
        if (tensor := cast("np.ndarray | None", model.get_initializer(name))) is not None:
            # Set new initializer tensor of shape [1]
            model.set_initializer(name, tensor.reshape(1))


class InferElementwiseBinaryOperation(Transformation):
    """Convert supported elementwise binary operations to their FINN custom operation."""

    @staticmethod
    def reject_output_dequant(model: ModelWrapper, node: NodeProto) -> bool:
        """Filter function to filter out the last elementwise Mul operation.

        Typically filters output de-quantization operations which should happen off-chip.
        """
        # The operator must be a Mul and have no successor nodes
        # If the output is a floating-point tensors, reject this
        return not (
            node.op_type == "Mul"
            and not model.find_direct_successors(node)
            and model.get_tensor_datatype(node.output[0]) == "FLOAT32"
        )

    def __init__(self, _filter: "Callable[..., bool] | None" = None) -> None:
        """Initialize the transformation method with an optional filter function."""
        # Initialize the base class Transformation object
        super().__init__()
        # Register the filter function as attribute
        self._filter = _filter if _filter is not None else lambda *_: True

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transform to convert elementwise binary operations to FINN custom ops."""
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Skip transforming nodes rejected by the filter
            if not self._filter(model, node):
                continue
            # If a custom operation with corresponding name is implemented in
            # the module, this operator is supported for conversion
            if f"Elementwise{node.op_type}" in dir(elementwise_binary):
                in0 = node.input[0]
                in1 = node.input[1]
                # if both inputs are constant, throw an error and
                # ask user to run FoldConstants transform first
                if (
                    model.get_initializer(in0) is not None
                    and model.get_initializer(in1) is not None
                ):
                    raise FINNUserError(
                        f"{node.name}: Both inputs are constant, please run FoldConstants "
                        "from qonnx.transformation.fold_constants first."
                    )
                lhs_style = "input" if model.get_initializer(in0) is None else "const"
                rhs_style = "input" if model.get_initializer(in1) is None else "const"
                result = node.output[0]

                # Need to "lift" potential scalar inputs to rank-1 tensors
                lift_to_rank1(in0, model)
                lift_to_rank1(in1, model)

                in0_shape = model.get_tensor_shape(in0)
                in1_shape = model.get_tensor_shape(in1)
                out_shape = model.get_tensor_shape(result)

                idt0 = model.get_tensor_datatype(in0)
                idt1 = model.get_tensor_datatype(in1)
                odt0 = model.get_tensor_datatype(result)

                # For constant inputs with FLOAT32 type, check if values are
                # actually integers and infer the smallest FINN datatype.
                if lhs_style == "const":
                    lhs_init = cast("np.ndarray", model.get_initializer(in0))
                    if (
                        idt0 == DataType["FLOAT32"]
                        and (lhs_init == lhs_init.astype(np.int64)).all()
                    ):
                        # Values are integers, find smallest datatype
                        _min, _max = lhs_init.min(), lhs_init.max()
                        _mag = _max if _min >= 0 else _min if (abs(_min) > _max) else (-_max - 1)
                        idt0 = DataType.get_smallest_possible(_mag)
                        model.set_tensor_datatype(in0, idt0)

                if rhs_style == "const":
                    rhs_init = cast("np.ndarray", model.get_initializer(in1))
                    if (
                        idt1 == DataType["FLOAT32"]
                        and (rhs_init == rhs_init.astype(np.int64)).all()
                    ):
                        # Values are integers, find smallest datatype
                        _min, _max = rhs_init.min(), rhs_init.max()
                        _mag = _max if _min >= 0 else _min if (abs(_min) > _max) else (-_max - 1)
                        idt1 = DataType.get_smallest_possible(_mag)
                        model.set_tensor_datatype(in1, idt1)

                # If both inputs are integers, set output to INT32 as default.
                # MinimizeAccumulatorWidth will optimize this later.
                if idt0.is_integer() and idt1.is_integer():
                    odt0 = DataType["INT32"]
                    model.set_tensor_datatype(result, odt0)

                # Determine the operation type - check for Sub->Abs pattern (AbsDiff)
                op_type = node.op_type
                nodes_to_remove = [node]
                if node.op_type == "Sub":
                    # Look for a downstream Abs node to fuse into AbsDiff
                    res_consumer = model.find_consumer(result)
                    if (res_consumer is not None) and (res_consumer.op_type == "Abs"):
                        op_type = "AbsDiff"
                        result = res_consumer.output[0]
                        out_shape = model.get_tensor_shape(result)
                        # Update output datatype - AbsDiff result is unsigned
                        if idt0.is_integer() and idt1.is_integer():
                            odt0 = DataType["UINT32"]
                            model.set_tensor_datatype(result, odt0)
                        nodes_to_remove.append(res_consumer)

                new_node = helper.make_node(
                    f"Elementwise{op_type}",
                    [in0, in1],
                    [result],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    lhs_shape=in0_shape,
                    rhs_shape=in1_shape,
                    out_shape=out_shape,
                    lhs_dtype=str(idt0),
                    rhs_dtype=str(idt1),
                    out_dtype=str(odt0),
                    lhs_style=lhs_style,
                    rhs_style=rhs_style,
                )
                graph.node.insert(index + 1, new_node)
                for n in nodes_to_remove:
                    graph.node.remove(n)

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

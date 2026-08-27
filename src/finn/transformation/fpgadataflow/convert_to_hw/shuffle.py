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

"""Convert Transpose (optionally with surrounding Reshape) layers into Shuffle HW layers."""

from onnx import NodeProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from typing import TYPE_CHECKING

from finn.util.exception import FINNInternalError, FINNUserError

if TYPE_CHECKING:
    from collections.abc import Callable


def skip_first_node_transpose(model: ModelWrapper, node: NodeProto) -> bool:
    """Default filter for InferShuffle: skip Transpose if it's the first node in the graph.
    This is useful for image classification networks where the first transpose converts
    NCHW to NHWC layout for data preprocessing."""
    return node != model.graph.node[0]


class InferShuffle(Transformation):
    """Find transpose layers with (optionally) reshape layers around them
    and convert them into a shuffle operator.
    """

    def __init__(
        self, _filter: "Callable[[ModelWrapper, NodeProto], bool]" = skip_first_node_transpose
    ) -> None:
        """Initialize instance."""
        super().__init__()
        # Register the filter function as attribute
        self._filter = _filter

    def _is_streaming_ptranspose(self, perm: list[int], shape: list[int]) -> bool:
        """Check if the permutation represents a streaming InnerShuffle case.
        A streaming InnerShuffle works when the last two dimensions are swapped,
        regardless of how many outer dimensions there are.
        """
        if len(perm) < 2 or len(shape) < 2:
            return False

        # Check if last two dimensions are swapped while others stay in order
        expected_perm = [*range(len(perm) - 2), len(perm) - 1, len(perm) - 2]
        return perm == expected_perm

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transformation."""
        graph = model.graph
        graph_modified = False
        for node_ind, n in enumerate(graph.node, start=1):
            if n.op_type == "Transpose":
                # Apply filter function to decide whether to convert this node
                if not self._filter(model, n):
                    continue
                to_remove = [n]

                new_in_tensor = None
                new_out_tensor = None

                perm = n.attribute[0]

                new_in_tensor = n.input[0]
                in_shape = model.get_tensor_shape(n.input[0])
                in_reshaped = in_shape

                # Detect a reshape at the input and capture it
                producer = model.find_producer(n.input[0])
                if producer is not None and producer.op_type == "Reshape":
                    new_in_tensor = producer.input[0]
                    in_shape = model.get_tensor_shape(new_in_tensor)
                    in_reshaped = model.get_tensor_shape(n.input[0])
                    to_remove.append(producer)

                new_out_tensor = n.output[0]
                out_shape = model.get_tensor_shape(new_out_tensor)
                out_reshaped = out_shape

                # Detect a reshape at the output and capture it
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "Reshape":
                    new_out_tensor = consumer.output[0]
                    out_shape = model.get_tensor_shape(n.output[0])
                    out_reshaped = model.get_tensor_shape(new_out_tensor)
                    to_remove.append(consumer)

                # Handle None shapes (shape inference might have failed)
                if in_reshaped is None:
                    raise FINNUserError(
                        f"Could not infer shape for tensor {n.input[0]}. "
                        "Please run InferShapes first."
                    )
                if out_shape is None or out_reshaped is None:
                    raise FINNUserError(
                        f"Could not infer shape for tensor {new_out_tensor}. "
                        "Please run InferShapes first."
                    )

                idt = model.get_tensor_datatype(new_in_tensor)
                odt = model.get_tensor_datatype(new_out_tensor)

                # Some sanity checks for the transformation
                if idt != odt:
                    raise FINNInternalError(
                        f"{n.name}: Input datatype and output datatype of the shuffle must be "
                        "the same, did something go wrong during transformation?"
                    )

                if len(perm.ints) != len(in_reshaped):
                    raise FINNUserError(
                        f"{n.name}: Permutation list {perm.ints=} does not match the reshaped "
                        f"input dimension {in_reshaped=}"
                    )

                if len(perm.ints) != len(out_shape):
                    raise FINNUserError(
                        f"{n.name}: Permutation list {perm.ints=} does not match the reshaped "
                        f"out dimension {out_reshaped=}"
                    )

                simd = 1

                new_node = helper.make_node(
                    "Shuffle",
                    [new_in_tensor],
                    [new_out_tensor],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    in_shape=in_shape,
                    transpose_in_shape=in_reshaped,
                    out_shape=out_reshaped,
                    transpose_out_shape=out_shape,
                    data_type=idt.name,
                    name=f"Shuffle_{n.name}",
                    SIMD=simd,
                    NumChannels=in_reshaped[-1],
                )
                new_node.attribute.extend([perm])
                graph.node.insert(node_ind, new_node)

                for i in to_remove:
                    graph.node.remove(i)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())

        return (model, graph_modified)

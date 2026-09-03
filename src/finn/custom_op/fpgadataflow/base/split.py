# Copyright (C) 2024, Advanced Micro Devices, Inc.
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

"""Streaming channel-split hardware custom operator (splits along the last axis)."""

import numpy as np
from onnx import NodeProto, helper
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import roundup_to_integer_multiple
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow import register_custom_op
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.exception import FINNInternalError
from finn.util.logging import log

if TYPE_CHECKING:
    from onnx import GraphProto

# Type of the dictionary returned by get_nodeattr_types: maps attribute names to
# their (dtype, required, default[, allowed_values]) specification tuples
NodeAttrTypes = dict[
    str,
    tuple[str, bool, int | float | str | bool | np.ndarray | list]
    | tuple[str, bool, int | float | str | bool | np.ndarray | list, set | None],
]


@register_custom_op
class StreamingSplit(HWCustomOp):
    """Abstraction layer for HW implementation of Split.

    Only supports splitting along the last (channel) axis.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {
            "SIMD": ("i", True, 0),
            # number of elements of each output streams
            "ChannelsPerStream": ("ints", True, []),
            # FINN DataTypes for input; output datatypes inferred from input
            "inputDataType": ("s", True, ""),
            # number of input vectors for non-split axes, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    @property
    def simd(self) -> int:
        """Get the SIMD parallelism."""
        return cast("int", self.get_nodeattr("SIMD"))

    @property
    def channels_per_stream(self) -> list[int]:
        """Get the element count of each output stream."""
        return list(cast("list[int]", self.get_nodeattr("ChannelsPerStream")))

    @property
    def num_input_vectors(self) -> list[int]:
        """Get the number of input vectors along the non-split axes."""
        return list(cast("list[int]", self.get_nodeattr("numInputVectors")))

    def get_n_outputs(self) -> int:
        """Return the number of output streams."""
        return len(self.channels_per_stream)

    def get_total_elems(self) -> int:
        """Return the total element count of the (unsplit) input channel axis."""
        return int(np.sum(self.channels_per_stream))

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return normal input shape."""
        return (*self.num_input_vectors, self.get_total_elems())

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded input shape."""
        folds = self.get_total_elems() // self.simd
        return (*self.num_input_vectors, folds, self.simd)

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return normal output shape."""
        return (*self.num_input_vectors, self.channels_per_stream[ind])

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return folded output shape."""
        folds = self.channels_per_stream[ind] // self.simd
        return (*self.num_input_vectors, folds, self.simd)

    def make_shape_compatible_op(self, model: ModelWrapper) -> NodeProto:
        """Create shape compatible op."""
        exp_ishape = self.get_normal_input_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]) or ())
        if ishape != exp_ishape:
            raise FINNInternalError(
                f"{self.onnx_node.name}: unexpected input shape {ishape}, expected {exp_ishape}"
            )
        if len(self.onnx_node.output) != self.get_n_outputs():
            raise FINNInternalError(
                f"{self.onnx_node.name}: unexpected number of outputs "
                f"({len(self.onnx_node.output)}), expected {self.get_n_outputs()}"
            )
        return helper.make_node("Split", self.onnx_node.input, self.onnx_node.output, axis=-1)

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer node datatype."""
        inp = self.onnx_node.input[0]
        idt = model.get_tensor_datatype(inp)
        if idt != self.get_input_datatype():
            log.warning(
                f"inputDataType changing for {self.onnx_node.name}: "
                f"{self.get_input_datatype()!s} -> {idt!s}"
            )
            self.set_nodeattr("inputDataType", idt.name)
        odt = self.get_output_datatype()
        for out in self.onnx_node.output:
            model.set_tensor_datatype(out, odt)

    def verify_node(self) -> list[str]:
        """Verify node."""
        return []

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return input datatype."""
        return DataType[cast("str", self.get_nodeattr("inputDataType"))]

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return output datatype (all outputs share the input datatype)."""
        return self.get_input_datatype()

    def get_instream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return instream width."""
        return self.get_input_datatype().bitwidth() * self.simd

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return outstream width."""
        return self.get_output_datatype().bitwidth() * self.simd

    def get_number_output_values(self) -> dict[str, int]:
        """Return number output values, one entry per output stream."""
        return {
            f"out{i}": int(np.prod(self.get_folded_output_shape(i)[1:-1]))
            for i in range(len(self.onnx_node.output))
        }

    def get_exp_cycles(self) -> int:
        """Return exp cycles."""
        return int(np.prod(self.get_folded_input_shape()[:-1]))

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute node."""
        node = self.onnx_node
        split = self.channels_per_stream
        np_split_param = np.cumsum(split[:-1])
        np_result = np.split(context[node.input[0]], np_split_param, axis=-1)
        for i, out in enumerate(node.output):
            context[out] = np_result[i]

    def get_instream_width_padded(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return instream width padded."""
        return roundup_to_integer_multiple(self.get_instream_width(), 8)

# Copyright (c) 2021, Xilinx
# Copyright (C) 2023, Advanced Micro Devices, Inc.
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

"""Streaming concatenation hardware custom operator."""

import math
import numpy as np
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow import register_custom_op
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
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
class StreamingConcat(HWCustomOp):
    """Abstraction layer for HW implementation of Concat.

    Only supports concatenating along the last (channel) axis.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {
            "SIMD": ("i", True, 0),
            # number of elements from each stream to concat
            "ChannelsPerStream": ("ints", True, []),
            # FINN DataTypes for inputs; output datatype inferred from inputs
            "inputDataTypes": ("strings", True, [""]),
            # number of input vectors for non-concat axes, examples:
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
        """Get the number of channels concatenated from each input stream."""
        return cast("list[int]", self.get_nodeattr("ChannelsPerStream"))

    @property
    def num_input_vectors(self) -> list[int]:
        """Get the number of input vectors along the non-concat axes."""
        return list(cast("list[int]", self.get_nodeattr("numInputVectors")))

    @property
    def input_datatypes(self) -> list[str]:
        """Get the per-input FINN datatype names."""
        return cast("list[str]", self.get_nodeattr("inputDataTypes"))

    def get_n_inputs(self) -> int:
        """Return number of inputs."""
        return len(self.channels_per_stream)

    def get_total_elems(self) -> int:
        """Return total elems."""
        return int(np.sum(self.channels_per_stream))

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return normal input shape."""
        elems = self.channels_per_stream[ind]
        return (*self.num_input_vectors, elems)

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return folded input shape."""
        folds = self.channels_per_stream[ind] // self.simd
        return (*self.num_input_vectors, folds, self.simd)

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return normal output shape."""
        return (*self.num_input_vectors, self.get_total_elems())

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded output shape."""
        folds = self.get_total_elems() // self.simd
        return (*self.num_input_vectors, folds, self.simd)

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer node datatype."""
        # check all input datatypes
        for i, inp in enumerate(self.onnx_node.input):
            idt = model.get_tensor_datatype(inp)
            if idt != self.get_input_datatype(i):
                log.warning(
                    f"inputDataType changing for {self.onnx_node.name}: "
                    f"{self.get_input_datatype(i)} -> {idt} "
                )
                old_datatypes_attr = list(self.input_datatypes)
                old_datatypes_attr[i] = idt.name
                self.set_nodeattr(
                    "inputDataTypes",
                    cast("list[str | int | float]", old_datatypes_attr),
                )
        odt = self.get_output_datatype()
        model.set_tensor_datatype(self.onnx_node.output[0], odt)

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:
        """Return input datatype."""
        # input dt identical for all inputs
        return DataType[self.input_datatypes[ind]]

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return output datatype."""
        # infer output datatype from declared inputDataTypes
        min_input = 0
        max_input = 0
        for i in range(len(self.input_datatypes)):
            idt = self.get_input_datatype(i)
            if idt.min() < min_input:
                min_input = idt.min()
            if idt.max() > max_input:
                max_input = idt.max()
        # if the input range is always greater than 0, then acc_max <= 2^P - 1
        if min_input >= 0:
            out_bit_width = math.ceil(np.log2(max_input + 1))
            odt = DataType[f"UINT{out_bit_width}"]
        # if the input range is signed, then acc_min >= -2^{P-1} and acc_max <=
        # 2^{P - 1} - 1, which means 2^{P - 1} >= max(-acc_min, 1 + acc_max)
        else:
            max_abs_input = max(-min_input, 1 + max_input)
            out_bit_width = math.ceil(np.log2(max_abs_input) + 1)
            odt = DataType[f"INT{out_bit_width}"]
        return odt

    def get_instream_width(self, ind: int = 0) -> int:
        """Return instream width."""
        ibits = self.get_input_datatype(ind).bitwidth()
        return ibits * self.simd

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return outstream width."""
        obits = self.get_output_datatype().bitwidth()
        return obits * self.simd

    def get_exp_cycles(self) -> int:
        """Return exp cycles."""
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute node."""
        node = self.onnx_node
        inp_values = [context[inp] for inp in node.input]
        result = np.concatenate(inp_values, axis=-1)
        context[node.output[0]] = result

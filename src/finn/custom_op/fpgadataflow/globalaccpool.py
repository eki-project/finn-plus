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

"""Global accumulate-pooling hardware custom operator."""

import numpy as np
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING, cast

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


class GlobalAccPool(HWCustomOp):
    """Abstraction layer for HW implementation of GlobalAccPool.

    Sums each channel over the spatial axes (a non-normalized global average
    pool); the output datatype is widened to hold the accumulated sum.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {
            "NumChannels": ("i", True, 0),
            "PE": ("i", True, 0),
            # FINN DataTypes for input
            "inputDataType": ("s", True, ""),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    @property
    def num_channels(self) -> int:
        """Get the number of channels."""
        return cast("int", self.get_nodeattr("NumChannels"))

    @property
    def pe(self) -> int:
        """Get the PE parallelism."""
        return cast("int", self.get_nodeattr("PE"))

    @property
    def num_input_vectors(self) -> list[int]:
        """Get the number of input vectors along the non-channel axes."""
        return list(cast("list[int]", self.get_nodeattr("numInputVectors")))

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return normal input shape."""
        return (*self.num_input_vectors, self.num_channels)

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded input shape."""
        ch = self.num_channels
        pe = self.pe
        if ch % pe != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: PE ({pe}) must divide NumChannels ({ch})"
            )
        folds = ch // pe
        return (*self.num_input_vectors, folds, pe)

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return normal output shape."""
        vecs = self.num_input_vectors
        if len(vecs) == 1:
            return (*vecs, self.num_channels)
        if len(vecs) == 3:
            return (vecs[0], 1, 1, self.num_channels)
        raise FINNInternalError(
            f"{self.onnx_node.name}: numInputVectors must have length 1 or 3, got {vecs}"
        )

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded output shape."""
        ch = self.num_channels
        pe = self.pe
        unfolded_shape = list(self.get_normal_output_shape())
        if ch % pe != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: PE ({pe}) must divide NumChannels ({ch})"
            )
        folds = ch // pe
        return (*unfolded_shape[:-1], folds, pe)

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer node datatype."""
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            log.warning(
                f"inputDataType changing for {node.name}: "
                f"{self.get_input_datatype()!s} -> {idt!s} "
            )
        self.set_nodeattr("inputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], self.get_output_datatype())

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of input."""
        return DataType[cast("str", self.get_nodeattr("inputDataType"))]

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of output.

        Determined from the accumulation extent and the input datatype.
        """
        idt = self.get_input_datatype()
        vecs = self.num_input_vectors
        npixels = vecs[-1] * vecs[-2]
        extreme_value = npixels * (idt.min() if idt.signed() else idt.max())
        return DataType.get_smallest_possible(extreme_value)

    def get_instream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return input stream width."""
        return self.pe * self.get_input_datatype().bitwidth()

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return output stream width."""
        return self.pe * self.get_output_datatype().bitwidth()

    def get_exp_cycles(self) -> int:
        """Return exp cycles.

        Channels/PE * batch size * idim * idim + Channels/PE.
        """
        folds = self.num_channels // self.pe
        return int(np.prod(self.get_folded_input_shape()[:-1]) + folds)

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute node.

        Simulates the behavior with plain Python.
        """
        node = self.onnx_node
        inp_values = context[node.input[0]]
        oshape = context[node.output[0]].shape
        result = np.apply_over_axes(np.sum, inp_values, [1, 2])
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(oshape)

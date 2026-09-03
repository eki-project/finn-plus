############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3 Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

"""Hardware softmax custom operator."""

import numpy as np
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from scipy.special import softmax
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow import register_custom_op
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.exception import FINNInternalError, FINNUserError
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
class HWSoftmax(HWCustomOp):
    """Abstraction layer for HW implementation of SoftMax layers.

    Applies ``softmax`` along the last axis; the output is always ``FLOAT32``.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {
            "ifm_dim": ("ints", True, []),
            "SIMD": ("i", False, 1),
            # FINN DataTypes for inputs, weights, outputs
            "input_data_type": ("s", True, ""),
            "NumChannels": ("i", False, 128),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    @property
    def ifm_dim(self) -> list[int]:
        """Get the input feature map shape."""
        return cast("list[int]", self.get_nodeattr("ifm_dim"))

    @property
    def simd(self) -> int:
        """Get the SIMD parallelism."""
        return cast("int", self.get_nodeattr("SIMD"))

    @property
    def num_channels(self) -> int:
        """Get the number of channels."""
        return cast("int", self.get_nodeattr("NumChannels"))

    def get_normal_input_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return normal input shape."""
        return self.ifm_dim

    def get_normal_output_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return normal output shape."""
        return self.get_normal_input_shape()

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute node."""
        node = self.onnx_node
        input_data = context[node.input[0]]
        context[node.output[0]] = softmax(input_data, axis=-1)

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of input."""
        data_type = DataType[cast("str", self.get_nodeattr("input_data_type"))]
        # the accumulation path needs to represent zero
        if not data_type.allowed(0):
            raise FINNUserError(
                f"{self.onnx_node.name}: input_data_type ({data_type}) must support zero"
            )
        return data_type

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer node datatype."""
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            log.warning(
                f"input_data_type changing for {node.name}: "
                f"{self.get_input_datatype()!s} -> {idt!s} "
            )
        self.set_nodeattr("input_data_type", idt.name)
        # set output datatype from property
        model.set_tensor_datatype(node.output[0], self.get_output_datatype())

    def get_instream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return instream width."""
        return self.get_input_datatype().bitwidth() * self.simd

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return outstream width."""
        return self.get_output_datatype().bitwidth() * self.simd

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of output."""
        return DataType["FLOAT32"]

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded output shape."""
        return self.get_folded_input_shape()

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded input shape."""
        normal_ishape = list(self.get_normal_input_shape())
        if normal_ishape[-1] % self.simd != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: SIMD ({self.simd}) must divide the input "
                f"dimension ({normal_ishape[-1]})"
            )
        fold = normal_ishape[-1] // self.simd
        return (*normal_ishape[:-1], fold, self.simd)

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Uniform-affine requantization hardware custom operator."""

import numpy as np
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.quant import max_int, min_int
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
class Requant(HWCustomOp):
    """Abstraction layer for HW implementation of Requantization.

    Requantization computes: ``clip(round(x * scale + bias), min, max)``

    This is an alternative to Thresholding for cases where the thresholds
    are uniformly spaced. Instead of comparing against N thresholds, we
    compute the output directly using a multiply-add operation.

    Inputs:
        input[0]: Data tensor to requantize
        input[1]: Scale tensor (per-channel or scalar, stored as initializer)
        input[2]: Bias tensor (per-channel or scalar, stored as initializer)
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {
            # parallelization; channels processed per cycle
            "PE": ("i", False, 1),
            # number of channels
            "NumChannels": ("i", True, 0),
            # FINN DataTypes for inputs, outputs
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
            # Whether to use narrow range (1) or full range (0)
            # Note: RTL backend only supports narrow=0 and unsigned output
            "narrow": ("i", False, 0),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    @property
    def pe(self) -> int:
        """Get the PE (channel) parallelism."""
        return cast("int", self.get_nodeattr("PE"))

    @property
    def num_channels(self) -> int:
        """Get the number of channels."""
        return cast("int", self.get_nodeattr("NumChannels"))

    @property
    def narrow(self) -> int:
        """Get whether narrow-range quantization is used (0/1)."""
        return cast("int", self.get_nodeattr("narrow"))

    @property
    def num_input_vectors(self) -> list[int]:
        """Get the number of input vectors along the non-channel axes."""
        return list(cast("list[int]", self.get_nodeattr("numInputVectors")))

    def get_scale(self, model: ModelWrapper) -> np.ndarray:
        """Get scale tensor from model initializer (input[1])."""
        if len(self.onnx_node.input) > 1:
            scale = model.get_initializer(self.onnx_node.input[1])
            if isinstance(scale, np.ndarray):
                return scale.flatten()
        # Default: scale = 1.0
        return np.array([1.0], dtype=np.float32)

    def get_bias(self, model: ModelWrapper) -> np.ndarray:
        """Get bias tensor from model initializer (input[2])."""
        if len(self.onnx_node.input) > 2:
            bias = model.get_initializer(self.onnx_node.input[2])
            if isinstance(bias, np.ndarray):
                return bias.flatten()
        # Default: bias = 0.0
        return np.array([0.0], dtype=np.float32)

    def is_per_channel(self, model: ModelWrapper) -> bool:
        """Check if scale/bias are per-channel (vs per-tensor)."""
        return self.get_scale(model).size > 1 or self.get_bias(model).size > 1

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer node datatype."""
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            log.warning(
                f"inputDataType changing for {node.name}: "
                f"{self.get_input_datatype().name} -> {idt.name}"
            )
        self.set_nodeattr("inputDataType", idt.name)
        # set output datatype from property
        model.set_tensor_datatype(node.output[0], self.get_output_datatype())

    def verify_node(self) -> list[str]:
        """Verify node."""
        return []

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:
        """Return FINN DataType of input."""
        if ind == 0:
            return DataType[cast("str", self.get_nodeattr("inputDataType"))]
        # Scale and bias are float
        return DataType["FLOAT32"]

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of output."""
        return DataType[cast("str", self.get_nodeattr("outputDataType"))]

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return input shape in format [N, H, W, C] or [N, C]."""
        return (*self.num_input_vectors, self.num_channels)

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return output shape."""
        return self.get_normal_input_shape(0)

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return folded input shape."""
        if ind == 0:
            normal_shape = self.get_normal_input_shape(0)
            fold = self.num_channels // self.pe
            return (*normal_shape[:-1], fold, self.pe)
        return self.get_normal_input_shape(ind)

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded output shape."""
        return self.get_folded_input_shape(0)

    def get_exp_cycles(self) -> int:
        """Return expected number of cycles for execution."""
        return self.get_number_output_values_for_stream(0)

    def execute_node(self, context: dict[str, np.ndarray], graph: "GraphProto") -> None:
        """Execute the requant operation."""
        node = self.onnx_node
        x = context[node.input[0]]

        # Get scale and bias (the graph carries a back-reference to its model
        # during node-by-node execution; fall back to the context otherwise)
        model = getattr(graph, "model", None)
        if model is not None:
            scale = self.get_scale(model)
            bias = self.get_bias(model)
        else:
            scale = context.get(node.input[1], np.array([1.0]))
            bias = context.get(node.input[2], np.array([0.0]))

        # Get output range from output datatype
        odt = self.get_output_datatype()
        signed = odt.signed()
        narrow = bool(self.narrow)
        bitwidth = odt.bitwidth()
        min_val = min_int(signed, narrow, bitwidth)
        max_val = max_int(signed, narrow, bitwidth)

        # Apply requantization: clip(round(x * scale + bias), min, max)
        # Use floor(x + 0.5) for round-half-up, not np.round which uses banker's rounding
        x_scaled = x * scale + bias
        x_rounded = np.floor(x_scaled + 0.5)
        x_clipped = np.clip(x_rounded, min_val, max_val)

        context[node.output[0]] = x_clipped.astype(np.float32)

    def get_instream_width(self, ind: int = 0) -> int:
        """Return input stream width."""
        if ind == 0:
            return self.pe * self.get_input_datatype(0).bitwidth()
        # Scale and bias (inputs 1, 2) are embedded, not streamed
        return 0

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return output stream width."""
        return self.pe * self.get_output_datatype().bitwidth()

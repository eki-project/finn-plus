# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Hardware abstraction layer for one-dimensional token stream padding."""

import numpy as np
from collections.abc import Sequence
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
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
class Pad1D(HWCustomOp):
    """One-dimensional padding for token streams.

    The first input is the streamed token sequence with shape
    ``(1, NumTokens, NumChannels)``. Optional second and third inputs provide
    left and right padding data. Each pad input can be a single token
    ``(1, 1, NumChannels)`` to be repeated, or a full pad sequence
    ``(1, PadLeft/PadRight, NumChannels)``.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return node attribute types, adding token, channel,
        padding, SIMD and datatype attributes.
        """
        my_attrs: NodeAttrTypes = super().get_nodeattr_types()
        my_attrs.update(
            {
                "NumTokens": ("i", True, 0),
                "NumChannels": ("i", True, 0),
                "PadLeft": ("i", False, 0),
                "PadRight": ("i", False, 0),
                "SIMD": ("i", False, 1),
                "inputDataType": ("s", True, ""),
                "outputDataType": ("s", False, ""),
            }
        )
        return my_attrs

    @property
    def num_tokens(self) -> int:
        """Get the number of tokens in the streamed input sequence."""
        return cast("int", self.get_nodeattr("NumTokens"))

    @property
    def num_channels(self) -> int:
        """Get the number of channels per token."""
        return cast("int", self.get_nodeattr("NumChannels"))

    @property
    def pad_left(self) -> int:
        """Get the number of pad tokens inserted before the sequence."""
        return cast("int", self.get_nodeattr("PadLeft"))

    @property
    def pad_right(self) -> int:
        """Get the number of pad tokens appended after the sequence."""
        return cast("int", self.get_nodeattr("PadRight"))

    @property
    def simd(self) -> int:
        """Get the SIMD parallelism."""
        return cast("int", self.get_nodeattr("SIMD"))

    def _get_pad_count(self, ind: int) -> int:
        """Return the number of pad tokens for pad input `ind` (1 = left, 2 = right)."""
        if ind == 1:
            return self.pad_left
        if ind == 2:
            return self.pad_right
        raise FINNInternalError("Pad1D pad inputs are indices 1 and 2")

    def _get_pad_side_name(self, ind: int) -> str:
        """Return "left" or "right" for pad input `ind`."""
        if ind == 1:
            return "left"
        if ind == 2:
            return "right"
        raise FINNInternalError("Pad1D pad inputs are indices 1 and 2")

    def _get_pad_input_shape(self, ind: int) -> tuple[int, ...]:
        """Return the canonical shape of pad input `ind`."""
        pad_count = self._get_pad_count(ind)
        return (1, max(1, pad_count), self.num_channels)

    def _validate_pad_shape(self, shape: Sequence[int], ind: int) -> None:
        """Check that `shape` is a valid single-token or full-
        sequence pad shape for input `ind`.
        """
        pad_count = self._get_pad_count(ind)
        num_channels = self.num_channels
        valid_shapes = {(1, max(1, pad_count), num_channels)}
        if pad_count > 1:
            valid_shapes.add((1, 1, num_channels))
        if tuple(shape) not in valid_shapes:
            raise FINNUserError(
                f"Pad1D {self._get_pad_side_name(ind)} pad shape must be one of "
                f"{sorted(valid_shapes)}, got {tuple(shape)}"
            )

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return the unfolded shape of input `ind` (tokens or pad data)."""
        if ind == 0:
            return (1, self.num_tokens, self.num_channels)
        if ind in [1, 2]:
            return self._get_pad_input_shape(ind)
        raise FINNInternalError("Pad1D has at most three inputs")

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return the folded shape of input `ind` (only the token input is folded by SIMD)."""
        normal_shape = self.get_normal_input_shape(ind)
        if ind != 0:
            return normal_shape

        simd = self.simd
        num_channels = normal_shape[-1]
        if num_channels % simd != 0:
            raise FINNUserError("SIMD must divide NumChannels")
        return (*normal_shape[:-1], num_channels // simd, simd)

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the unfolded output shape with padding added to the token dimension."""
        return (1, self.num_tokens + self.pad_left + self.pad_right, self.num_channels)

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return the folded output shape."""
        normal_shape = self.get_normal_output_shape(ind)
        simd = self.simd
        num_channels = normal_shape[-1]
        if num_channels % simd != 0:
            raise FINNUserError("SIMD must divide NumChannels")
        return (*normal_shape[:-1], num_channels // simd, simd)

    def make_shape_compatible_op(self, model: ModelWrapper) -> NodeProto:
        """Check the input and pad shapes and return a constant-shape op for the output."""
        exp_ishape = self.get_normal_input_shape(0)
        ishape = model.get_tensor_shape(self.onnx_node.input[0])
        if ishape is None or tuple(ishape) != exp_ishape:
            raise FINNUserError("Unexpected input shape for Pad1D tokens.")

        for ind in [1, 2]:
            if len(self.onnx_node.input) <= ind:
                if self._get_pad_count(ind) != 0:
                    raise FINNUserError(
                        f"Pad1D {self._get_pad_side_name(ind)} padding requires input index {ind}"
                    )
                continue

            pad_name = self.onnx_node.input[ind]
            pad_shape = model.get_tensor_shape(pad_name)
            if pad_shape is None:
                pad_init = cast("np.ndarray | None", model.get_initializer(pad_name))
                if pad_init is not None:
                    pad_shape = pad_init.shape
            if pad_shape is not None:
                self._validate_pad_shape(pad_shape, ind)

        return super().make_const_shape_op(self.get_normal_output_shape())

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer and set the input, pad and output datatypes from the model tensors."""
        node = self.onnx_node
        attr_idt = None
        if self.get_nodeattr("inputDataType") != "":
            attr_idt = self.get_input_datatype()

        idt = model.get_tensor_datatype(node.input[0])
        if idt is None:
            idt = attr_idt
        if idt is None:
            raise FINNUserError("Pad1D input datatype is not set")

        if attr_idt is not None and attr_idt != idt:
            log.warning(f"inputDataType changing for {node.name}: {attr_idt!s} -> {idt!s}")
        self.set_nodeattr("inputDataType", idt.name)

        for pad_input in node.input[1:]:
            pad_dt = model.get_tensor_datatype(pad_input)
            if pad_dt is None:
                model.set_tensor_datatype(pad_input, idt)
            elif pad_dt != idt:
                raise FINNUserError("Pad1D pad datatype must match input datatype.")

        self.set_nodeattr("outputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def verify_node(self) -> list[str]:
        """Check that the size, padding and SIMD attributes are valid.

        Returns:
            An empty list; invalid attributes raise a ``FINNUserError`` instead.

        """
        if self.num_tokens <= 0:
            raise FINNUserError("NumTokens must be positive")
        if self.num_channels <= 0:
            raise FINNUserError("NumChannels must be positive")
        if self.pad_left < 0:
            raise FINNUserError("PadLeft cannot be negative")
        if self.pad_right < 0:
            raise FINNUserError("PadRight cannot be negative")
        if self.simd <= 0:
            raise FINNUserError("SIMD must be positive")
        return []

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return the FINN DataType of the input."""
        return DataType[cast("str", self.get_nodeattr("inputDataType"))]

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:
        """Return the FINN DataType of the output (defaults to the input datatype)."""
        odt = cast("str", self.get_nodeattr("outputDataType"))
        if odt == "":
            return self.get_input_datatype(ind)
        return DataType[odt]

    def get_instream_width(self, ind: int = 0) -> int:
        """Return the width of input stream `ind` in bits (0 for pad inputs)."""
        if ind != 0:
            return 0
        return self.get_input_datatype().bitwidth() * self.simd

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return the width of the output stream in bits."""
        return self.get_output_datatype().bitwidth() * self.simd

    def get_exp_cycles(self) -> int:
        """Return the expected cycle count (one per output stream word)."""
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    def _get_expanded_pad_values(self, context: dict[str, np.ndarray], ind: int) -> np.ndarray:
        """Return the pad values for input `ind` expanded to the full pad length."""
        pad_count = self._get_pad_count(ind)
        if pad_count == 0:
            return np.zeros((1, 0, self.num_channels), dtype=np.float32)

        if len(self.onnx_node.input) <= ind:
            raise FINNUserError(
                f"Pad1D {self._get_pad_side_name(ind)} padding requires input index {ind}"
            )

        pad_values = np.asarray(context[self.onnx_node.input[ind]], dtype=np.float32)
        self._validate_pad_shape(pad_values.shape, ind)
        if pad_values.shape[1] == 1 and pad_count > 1:
            return np.repeat(pad_values, pad_count, axis=1)
        return pad_values

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute the node in Python by concatenating pad values around the input tokens."""
        node = self.onnx_node
        inp = context[node.input[0]]

        values = [self._get_expanded_pad_values(context, 1), inp]
        values.append(self._get_expanded_pad_values(context, 2))

        result = np.concatenate(values, axis=1)
        oshape = self.get_normal_output_shape()
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(oshape)

    def bram_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Return the number of BRAMs used (none)."""
        return 0

    def lut_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Estimate the LUTs used for control logic and pad data storage."""
        pad_tokens = self.pad_left + self.pad_right
        return int(128 + self.num_channels * max(1, pad_tokens))

    def get_op_and_param_counts(self) -> dict[str, int]:
        """Return a dictionary with the number of stored pad token parameters."""
        pad_tokens = self.pad_left + self.pad_right
        return {"param_pad_tokens": int(self.num_channels * pad_tokens)}

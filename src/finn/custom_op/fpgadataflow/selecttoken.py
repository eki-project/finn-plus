# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Hardware abstraction layer for selecting one token from a token sequence."""

import numpy as np
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
class SelectToken(HWCustomOp):
    """Select one token vector from a sequence of token vectors."""

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return node attribute types, adding token, channel,
        index, SIMD and datatype attributes.
        """
        my_attrs: NodeAttrTypes = super().get_nodeattr_types()
        my_attrs.update(
            {
                "NumTokens": ("i", True, 0),
                "NumChannels": ("i", True, 0),
                "TokenIndex": ("i", True, 0),
                "SIMD": ("i", False, 1),
                "inputDataType": ("s", True, ""),
                "outputDataType": ("s", False, ""),
            }
        )
        return my_attrs

    @property
    def num_tokens(self) -> int:
        """Get the number of tokens in the input sequence."""
        return cast("int", self.get_nodeattr("NumTokens"))

    @property
    def num_channels(self) -> int:
        """Get the number of channels per token."""
        return cast("int", self.get_nodeattr("NumChannels"))

    @property
    def token_index(self) -> int:
        """Get the index of the selected token as given (may be negative)."""
        return cast("int", self.get_nodeattr("TokenIndex"))

    @property
    def simd(self) -> int:
        """Get the SIMD parallelism."""
        return cast("int", self.get_nodeattr("SIMD"))

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return the unfolded input shape (1, NumTokens, NumChannels)."""
        if ind != 0:
            raise FINNInternalError("SelectToken only has one input")
        return (1, self.num_tokens, self.num_channels)

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return the folded input shape."""
        normal_shape = self.get_normal_input_shape(ind)
        simd = self.simd
        num_channels = normal_shape[-1]
        if num_channels % simd != 0:
            raise FINNUserError("SIMD must divide NumChannels")
        return (*normal_shape[:-1], num_channels // simd, simd)

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the unfolded output shape (1, NumChannels)."""
        return (1, self.num_channels)

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return the folded output shape."""
        normal_shape = self.get_normal_output_shape(ind)
        simd = self.simd
        num_channels = normal_shape[-1]
        if num_channels % simd != 0:
            raise FINNUserError("SIMD must divide NumChannels")
        return (*normal_shape[:-1], num_channels // simd, simd)

    def make_shape_compatible_op(self, model: ModelWrapper) -> NodeProto:
        """Check the input shape and return a constant-shape op for the output."""
        ishape = model.get_tensor_shape(self.onnx_node.input[0])
        if ishape is None or tuple(ishape) != self.get_normal_input_shape():
            raise FINNUserError("Unexpected input shape for token sequence")
        return super().make_const_shape_op(self.get_normal_output_shape())

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer and set the input and output datatypes from the model tensors."""
        node = self.onnx_node
        attr_idt = None
        if self.get_nodeattr("inputDataType") != "":
            attr_idt = self.get_input_datatype()
        idt = model.get_tensor_datatype(node.input[0]) or attr_idt
        if idt is None:
            raise FINNUserError("SelectToken input datatype is not set")
        if attr_idt is not None and attr_idt != idt:
            log.warning(f"inputDataType changing for {node.name}: {attr_idt} -> {idt}")
        self.set_nodeattr("inputDataType", idt.name)
        self.set_nodeattr("outputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return the FINN DataType of the input."""
        return DataType[cast("str", self.get_nodeattr("inputDataType"))]

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:
        """Return the FINN DataType of the output (defaults to the input datatype)."""
        odt = cast("str", self.get_nodeattr("outputDataType"))
        return self.get_input_datatype(ind) if odt == "" else DataType[odt]

    def get_instream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return the width of the input stream in bits."""
        return self.get_input_datatype().bitwidth() * self.simd

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return the width of the output stream in bits."""
        return self.get_output_datatype().bitwidth() * self.simd

    def get_exp_cycles(self) -> int:
        """Return the expected cycle count (one per input stream word)."""
        return int(np.prod(self.get_folded_input_shape()[:-1]))

    def _resolved_token_index(self) -> int:
        """Return TokenIndex with a negative value resolved from the end of the sequence.

        Raises:
            FINNUserError: If the index does not select an existing token.

        """
        num_tokens = self.num_tokens
        token_index = self.token_index
        if token_index < 0:
            token_index += num_tokens
        if not 0 <= token_index < num_tokens:
            raise FINNUserError("TokenIndex must select an existing token")
        return token_index

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute the node in Python by slicing the selected token from the sequence."""
        node = self.onnx_node
        token_index = self._resolved_token_index()
        result = context[node.input[0]][:, token_index, :]
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(
            self.get_normal_output_shape()
        )

############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################
"""Parallel 2D transpose (inner shuffle) hardware custom operator."""

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


class InnerShuffle(HWCustomOp):
    """Abstraction layer for the parallel 2D transpose.

    Swaps the last two axes of the input tensor (``(..., a, b) -> (..., b, a)``).
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {
            "data_type": ("s", True, ""),
            "in_shape": ("ints", True, []),  # must have length 2
            "SIMD": ("i", False, 1),
            # Track original shuffle name/SIMD for SIMD config export
            "original_node_name": ("s", False, ""),
            "original_simd": ("i", False, 1),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    @property
    def in_shape(self) -> list[int]:
        """Get the input tensor shape."""
        return cast("list[int]", self.get_nodeattr("in_shape"))

    @property
    def simd(self) -> int:
        """Get the SIMD parallelism."""
        return cast("int", self.get_nodeattr("SIMD"))

    @property
    def dtype(self) -> BaseDataType:
        """Get the element data type."""
        return DataType[cast("str", self.get_nodeattr("data_type"))]

    def get_normal_input_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return normal input shape."""
        return self.in_shape

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return normal output shape."""
        ishape = tuple(self.get_normal_input_shape())
        return (*ishape[:-2], ishape[-1], ishape[-2])

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute node."""
        node = self.onnx_node
        input_data = context[node.input[0]]
        if len(input_data.shape) < 2:
            raise FINNInternalError(
                f"{node.name}: InnerShuffle requires at least 2D input, got {input_data.shape}"
            )
        # Transpose only the last two dimensions: (..., a, b) -> (..., b, a)
        axes = list(range(len(input_data.shape)))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        context[node.output[0]] = np.transpose(input_data, axes)

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return input datatype."""
        return self.dtype

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer node datatype."""
        node = self.onnx_node
        dt = model.get_tensor_datatype(node.input[0])
        if dt != self.get_input_datatype():
            log.warning(
                f"data_type changing for {node.name}: {self.get_input_datatype()!s} -> {dt!s}"
            )
        self.set_nodeattr("data_type", dt.name)
        model.set_tensor_datatype(node.output[0], dt)

    def get_instream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return instream width."""
        return self.get_input_datatype().bitwidth() * self.simd

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return outstream width."""
        return self.get_output_datatype().bitwidth() * self.simd

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return output datatype."""
        return self.dtype

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded output shape."""
        normal_oshape = list(self.get_normal_output_shape())
        if normal_oshape[-1] % self.simd != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: SIMD ({self.simd}) must divide the innermost "
                f"output dimension ({normal_oshape[-1]})"
            )
        fold = normal_oshape[-1] // self.simd
        return (*normal_oshape[:-1], fold, self.simd)

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded input shape."""
        normal_ishape = list(self.get_normal_input_shape())
        fold = int(np.prod(normal_ishape) // self.simd)
        return (fold, self.simd)

    def get_exp_cycles(self) -> int:
        """Estimate cycles for the double-buffered InnerShuffle RTL.

        The RTL uses two BRAM banks with page_size = I*J/SIMD. The first page
        must be fully written before reads can begin, adding one extra page of
        latency beyond the streaming throughput. Empirically verified to match
        cycles_rtlsim within atol=10.
        """
        in_shape = self.in_shape
        i_dim = in_shape[-2]
        j_dim = in_shape[-1]
        page_size = i_dim * j_dim // self.simd
        total_elems = int(np.prod(in_shape)) // self.simd
        return 2 * total_elems + page_size

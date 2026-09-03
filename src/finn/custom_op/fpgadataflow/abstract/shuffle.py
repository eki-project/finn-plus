############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

"""Generic transpose (shuffle) hardware custom operator.

Later lowered into ``InnerShuffle`` / ``OuterShuffle`` stages by
``transpose_decomposition``.
"""

import numpy as np
from onnx import NodeProto, helper
from operator import itemgetter
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
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
class Shuffle(HWCustomOp):
    """Abstraction layer for a generic transpose (rearrange + transpose).

    This operator is later transformed into ``InnerShuffle`` and
    ``OuterShuffle`` operations.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """The attributes for the Shuffle node capture the
        optional reshapes either side of the transpose.
        Below is a diagram indicating what tensors the
        attribute names are referring to.

              │ in_shape
              │
              │
        ┌─────▼──────┐
        │            │
        │ Reshape    │
        │            │
        └─────┬──────┘
              │
              │ transpose_in_shape
        ┌─────▼──────┐
        │            │
        │  Transpose │
        │            │
        └─────┬──────┘
              │  transpose_out_shape
        ┌─────▼──────┐
        │            │
        │  Reshape   │
        │            │
        └─────┬──────┘
              │
              │  out_shape
              ▼
        """  # noqa: D401
        my_attrs: NodeAttrTypes = {
            "data_type": ("s", True, ""),
            "transpose_in_shape": ("ints", True, []),
            "in_shape": ("ints", True, []),
            "transpose_out_shape": ("ints", True, []),
            "out_shape": ("ints", True, []),
            "perm": ("ints", True, []),
            "SIMD": ("i", False, 1),
            "NumChannels": ("i", False, 128),
            # Track original shuffle name/SIMD for SIMD config export
            "original_node_name": ("s", False, ""),
            "original_simd": ("i", False, 1),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    @property
    def dtype(self) -> BaseDataType:
        """Get the element data type."""
        return DataType[cast("str", self.get_nodeattr("data_type"))]

    @property
    def simd(self) -> int:
        """Get the SIMD parallelism."""
        return cast("int", self.get_nodeattr("SIMD"))

    @property
    def transpose_in_shape(self) -> list[int]:
        """Get the pre-transpose (reshaped) input shape."""
        return list(cast("list[int]", self.get_nodeattr("transpose_in_shape")))

    @property
    def perm(self) -> list[int]:
        """Get the axis permutation applied by the transpose."""
        return list(cast("list[int]", self.get_nodeattr("perm")))

    def get_normal_input_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return normal input shape."""
        return list(cast("list[int]", self.get_nodeattr("in_shape")))

    def get_normal_output_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return normal output shape."""
        return list(cast("list[int]", self.get_nodeattr("out_shape")))

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute node."""
        node = self.onnx_node
        input_data = context[node.input[0]]
        input_reshaped = input_data.reshape(self.transpose_in_shape)
        transposed = np.transpose(input_reshaped, axes=self.perm)
        context[node.output[0]] = transposed.reshape(self.get_normal_output_shape())

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

    def verify_node(self) -> list[str]:
        """Verify node."""
        raise NotImplementedError("This function is not yet implemented.")

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
        if normal_ishape[-1] % self.simd != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: SIMD ({self.simd}) must divide the innermost "
                f"input dimension ({normal_ishape[-1]})"
            )
        fold = normal_ishape[-1] // self.simd
        return (*normal_ishape[:-1], fold, self.simd)

    def get_exp_cycles(self) -> int:
        """Estimate cycles by decomposing into Inner/OuterShuffle stages.

        Decomposes the transpose into a sequence of hardware-constrained
        operations (inner_shuffle / outer_shuffle), creates temporary nodes
        for each stage, and returns the MAX of their cycle estimates
        (stages are pipelined, so throughput is limited by the slowest).
        """
        from finn.transformation.fpgadataflow.transpose_decomposition import (
            _is_inner_shuffle,
            decompose_transpose_with_constraints,
            shuffle_perfect_loopnest_coeffs,
        )

        transpose_in_shape = self.transpose_in_shape
        perm = self.perm
        simd = self.simd
        data_type = self.get_nodeattr("data_type")

        p_list, operation_types = decompose_transpose_with_constraints(
            perm, transpose_in_shape, simd
        )

        if len(p_list) == 0:
            return 0

        stage_cycles = []
        current_shape = list(transpose_in_shape)

        for step_idx, (p_perm, _op_type) in enumerate(zip(p_list, operation_types, strict=True)):
            # Note: p_perm must stay a list; _is_inner_shuffle compares it by
            # equality against a list and a tuple would never match.
            out_shape = list(itemgetter(*p_perm)(current_shape))

            if _is_inner_shuffle(p_perm, current_shape):
                # InnerShuffle: in_shape = current_shape
                tmp_node = helper.make_node(
                    "InnerShuffle",
                    ["tmp_in"],
                    ["tmp_out"],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    in_shape=current_shape,
                    data_type=data_type,
                    SIMD=simd,
                    name=f"tmp_inner_{step_idx}",
                )
            else:
                # OuterShuffle
                # Note: shuffle_perfect_loopnest_coeffs annotates its params as
                # ``tuple[int]`` but accepts any int sequence.
                loop_coeffs = shuffle_perfect_loopnest_coeffs(
                    shape=cast("tuple[int]", tuple(current_shape)),
                    perm=cast("tuple[int]", tuple(p_perm)),
                )
                tmp_node = helper.make_node(
                    "OuterShuffle",
                    ["tmp_in"],
                    ["tmp_out"],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    in_shape=current_shape,
                    transpose_in_shape=current_shape,
                    perm=p_perm,
                    out_shape=out_shape,
                    transpose_out_shape=out_shape,
                    data_type=data_type,
                    loop_coeffs=loop_coeffs,
                    SIMD=simd,
                    NumChannels=current_shape[-1],
                    name=f"tmp_outer_{step_idx}",
                )

            inst = cast("HWCustomOp", getCustomOp(tmp_node))
            stage_cycles.append(inst.get_exp_cycles())
            current_shape = out_shape

        return int(max(stage_cycles))

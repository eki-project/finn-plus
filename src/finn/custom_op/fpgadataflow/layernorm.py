###################################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright for portions of this file is held by AMD and Microsoft under
# MIT license as part of project Brainsmith.
# All other copyright is held by AMD and is provided under BSD-3-Clause license.
#
###################################################################################

"""Layer-normalization hardware custom operator."""

import numpy as np
import torch
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from torch.nn.functional import layer_norm
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


class LayerNorm(HWCustomOp):
    """Abstraction layer for HW implementation of LayerNorm.

    Normalizes each input vector over its innermost (channel) axis. The affine
    scale/bias of a full ``LayerNormalization`` are expected to have been split
    out into separate nodes beforehand, so this operator performs the
    zero-mean/unit-variance step only.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {
            "SIMD": ("i", True, 0),
            "ifm_dim": ("ints", True, []),
            "epsilon": ("f", True, 1e-5),
            # FINN DataTypes for inputs, outputs
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    @property
    def simd(self) -> int:
        """Get the SIMD parallelism."""
        return cast("int", self.get_nodeattr("SIMD"))

    @property
    def ifm_dim(self) -> list[int]:
        """Get the input feature-map shape."""
        return list(cast("list[int]", self.get_nodeattr("ifm_dim")))

    @property
    def epsilon(self) -> float:
        """Get the numerical-stability epsilon added to the variance."""
        return cast("float", self.get_nodeattr("epsilon"))

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute node.

        Functionally verified against the PyTorch ``layer_norm`` implementation
        (weight and bias are removed by an earlier transformation).
        """
        node = self.onnx_node
        in_values = context[node.input[0]]
        oshape = context[node.output[0]].shape
        in_act = torch.from_numpy(in_values)
        out_act = layer_norm(in_act, [in_values.shape[-1]], eps=self.epsilon)
        context[node.output[0]] = np.asarray(out_act, dtype=np.float32).reshape(oshape)

    def get_normal_input_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return normal input shape."""
        return self.ifm_dim

    def get_normal_output_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return normal output shape."""
        return self.get_normal_input_shape()

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded input shape."""
        normal_ishape = self.get_normal_input_shape()
        if normal_ishape[-1] % self.simd != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: SIMD ({self.simd}) must divide the innermost "
                f"input dimension ({normal_ishape[-1]})"
            )
        fold = normal_ishape[-1] // self.simd
        return (*normal_ishape[:-1], fold, self.simd)

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded output shape."""
        return self.get_folded_input_shape()

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:
        """Return FINN DataType of input."""
        if ind == 0:
            return DataType[cast("str", self.get_nodeattr("inputDataType"))]
        raise FINNInternalError(f"{self.onnx_node.name}: undefined input ind {ind} for LayerNorm")

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of output."""
        return DataType[cast("str", self.get_nodeattr("outputDataType"))]

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer node datatype."""
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            log.warning(
                f"inputDataType changing for {node.name}: "
                f"{self.get_input_datatype()!s} -> {idt!s}"
            )
        self.set_nodeattr("inputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], self.get_output_datatype())

    def get_instream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return instream width."""
        return self.get_input_datatype().bitwidth() * self.simd

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return outstream width."""
        return self.get_output_datatype().bitwidth() * self.simd

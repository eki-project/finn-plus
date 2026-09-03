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

"""Spatial cropping hardware custom operator."""

import numpy as np
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
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
class Crop(HWCustomOp):
    """Abstraction layer for Crop layers."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {
            "DataType": ("s", True, ""),
            "ImgDim": ("ints", True, []),  # [h, w]
            "NumChannels": ("i", True, 0),
            "CropNorth": ("i", True, []),
            "CropSouth": ("i", True, []),
            "CropWest": ("i", True, []),
            "CropEast": ("i", True, []),
            "SIMD": ("i", False, 1),
            "numInputVectors": ("ints", False, []),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    @property
    def dtype(self) -> BaseDataType:
        """Get the element data type."""
        return DataType[cast("str", self.get_nodeattr("DataType"))]

    @property
    def img_dim(self) -> list[int]:
        """Get the input image dimensions [h, w]."""
        return cast("list[int]", self.get_nodeattr("ImgDim"))

    @property
    def num_channels(self) -> int:
        """Get the number of channels."""
        return cast("int", self.get_nodeattr("NumChannels"))

    @property
    def crop_north(self) -> int:
        """Get the number of rows cropped from the top."""
        return cast("int", self.get_nodeattr("CropNorth"))

    @property
    def crop_south(self) -> int:
        """Get the number of rows cropped from the bottom."""
        return cast("int", self.get_nodeattr("CropSouth"))

    @property
    def crop_west(self) -> int:
        """Get the number of columns cropped from the left."""
        return cast("int", self.get_nodeattr("CropWest"))

    @property
    def crop_east(self) -> int:
        """Get the number of columns cropped from the right."""
        return cast("int", self.get_nodeattr("CropEast"))

    @property
    def simd(self) -> int:
        """Get the SIMD parallelism."""
        return cast("int", self.get_nodeattr("SIMD"))

    @property
    def num_input_vectors(self) -> list[int]:
        """Get the number of input vectors along the non-spatial axes."""
        return cast("list[int]", self.get_nodeattr("numInputVectors"))

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return normal input shape."""
        num_vec = self.num_input_vectors
        h, w = self.img_dim
        img_dim = [w] if h == 0 else [h, w]
        if num_vec != [0]:
            return (*num_vec, *img_dim, self.num_channels)
        return (*img_dim, self.num_channels)

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return normal output shape."""
        num_vec = self.num_input_vectors
        height, width = self.img_dim
        owidth = width - (self.crop_west + self.crop_east)
        oheight = height - (self.crop_north + self.crop_south)
        o_img_dim = [owidth] if oheight == 0 else [oheight, owidth]
        if num_vec != [0]:
            return (*num_vec, *o_img_dim, self.num_channels)
        return (*o_img_dim, self.num_channels)

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"
    ) -> None:  # noqa: ARG002
        """Execute node."""
        node = self.onnx_node
        h, w = self.img_dim
        crop_north = self.crop_north
        crop_east = self.crop_east
        crop_west = self.crop_west
        crop_south = self.crop_south
        inp = context[node.input[0]]
        if len(inp.shape) == 3:
            cropped_slice = inp[crop_north : h - crop_south, crop_west : w - crop_east, :]
        elif len(inp.shape) == 2:
            cropped_slice = inp[crop_west : w - crop_east, :]
        elif len(inp.shape) == 4:
            cropped_slice = inp[:, crop_north : h - crop_south, crop_west : w - crop_east, :]
        else:
            raise FINNInternalError(
                f"{node.name}: Crop execute_node only supports 2D-4D input tensors, "
                f"got shape {inp.shape}"
            )
        expected_shape = tuple(self.get_normal_output_shape())
        if cropped_slice.shape != expected_shape:
            raise FINNInternalError(
                f"{node.name}: cropped shape {cropped_slice.shape} does not match "
                f"expected output shape {expected_shape}"
            )
        context[node.output[0]] = cropped_slice

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
        self.set_nodeattr("DataType", dt.name)

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
        simd = self.simd
        if normal_oshape[-1] % simd != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: innermost dimension ({normal_oshape[-1]}) "
                f"must be divisible by SIMD ({simd})"
            )
        fold = normal_oshape[-1] // simd
        return (*normal_oshape[:-1], fold, simd)

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded input shape."""
        normal_ishape = list(self.get_normal_input_shape())
        simd = self.simd
        if normal_ishape[-1] % simd != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: innermost dimension ({normal_ishape[-1]}) "
                f"must be divisible by SIMD ({simd})"
            )
        fold = normal_ishape[-1] // simd
        return (*normal_ishape[:-1], fold, simd)

    def get_exp_cycles(self) -> int:
        """Return exp cycles."""
        simd = self.simd
        num_vec = self.num_input_vectors
        height, width = self.img_dim
        ch = self.num_channels
        if height == 0:
            # pretend that height is 1 for code generation
            height = 1

        if num_vec != [0]:
            return int(np.prod(num_vec)) * height * width * (ch // simd)
        return height * width * (ch // simd)

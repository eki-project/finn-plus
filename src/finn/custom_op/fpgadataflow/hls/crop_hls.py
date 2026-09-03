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

"""HLS backend implementation of the spatial cropping operator."""

import numpy as np
from typing import TYPE_CHECKING

from finn.custom_op.fpgadataflow.base.crop import Crop, NodeAttrTypes
from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

if TYPE_CHECKING:
    from onnx import GraphProto, NodeProto


@register_custom_op
class Crop_hls(Crop, HLSBackend):
    """Crop node with dynamically generated HLS."""

    def __init__(self, onnx_node: "NodeProto", **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(Crop.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self) -> None:
        """Return global includes."""
        self.code_gen_dict["$GLOBALS$"] = [
            '#include "crop.hpp"',
        ]

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Return defines."""
        simd = self.simd
        dtype = self.get_input_datatype()
        height, width = self.img_dim
        if height == 0:
            # pretend that height is 1 for code generation
            height = 1
        ch = self.num_channels
        self.code_gen_dict["$DEFINES$"] = [
            f"""
            constexpr unsigned  SIMD      = {simd};
            constexpr unsigned  H      = {height};
            constexpr unsigned  W      = {width};
            constexpr unsigned  CF     = {ch // simd};
            constexpr unsigned  CROP_N = {self.crop_north};
            constexpr unsigned  CROP_E = {self.crop_east};
            constexpr unsigned  CROP_S = {self.crop_south};
            constexpr unsigned  CROP_W = {self.crop_west};
            using  TV = hls::vector<{dtype.get_hls_datatype_str()}, SIMD>;
            """
        ]

    def docompute(self) -> None:
        """Return docompute."""
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """
            hls::stream<TV>  src0;
            hls::stream<TV>  dst0;
            #pragma HLS stream variable=src0 depth=2
            #pragma HLS stream variable=dst0 depth=2

            move(in0_V, src0);
            crop< H, W,	CF, CROP_N, CROP_E, CROP_S, CROP_W, TV>(src0, dst0);
            move(dst0, out0_V);
            """
        ]

    def blackboxfunction(self) -> None:
        """Return blackboxfunction."""
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"""
            void {self.onnx_node.name} (
                hls::stream<TV> &in0_V,
                hls::stream<TV> &out0_V
            )
            """
        ]

    def pragmas(self) -> None:
        """Return pragmas."""
        self.code_gen_dict["$PRAGMAS$"] = [
            """
            #pragma HLS interface AXIS port=in0_V
            #pragma HLS interface AXIS port=out0_V
            #pragma HLS aggregate variable=in0_V compact=bit
            #pragma HLS aggregate variable=out0_V compact=bit

            #pragma HLS interface ap_ctrl_none port=return
            #pragma HLS dataflow disable_start_propagation
            """
        ]

    def execute_node(self, context: dict[str, np.ndarray], graph: "GraphProto") -> None:
        """Execute node."""
        HLSBackend.execute_node(self, context, graph)

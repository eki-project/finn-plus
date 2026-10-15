############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

"""HLS backend implementation of the outer (rank-preserving) transpose."""

import math
import numpy as np
from onnx import NodeProto
from typing import TYPE_CHECKING

from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.outer_shuffle import NodeAttrTypes, OuterShuffle
from finn.util.exception import FINNUserError

if TYPE_CHECKING:
    from onnx import GraphProto


def auto_size_simd(i_dim: int, simd: int) -> int | None:
    """Return the smallest divisor d of ``i_dim`` such that d > ``simd``.

    If no such divisor exists, return None.
    """
    if i_dim <= 0:
        raise ValueError("i_dim must be a positive integer")
    if simd < 0:
        raise ValueError("simd must be a non-negative integer")

    candidates = []
    limit = math.isqrt(i_dim)
    for a in range(1, limit + 1):
        if i_dim % a == 0:
            b = i_dim // a
            if a > simd:
                candidates.append(a)
            if b > simd:
                candidates.append(b)

    if not candidates:
        return None

    return min(candidates)


@register_custom_op
class OuterShuffle_hls(OuterShuffle, HLSBackend):
    """HLS backend implementation of OuterShuffle.

    Uses the finn-hlslib ``input_gen`` streamtools function.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

        # check some constraints that it is a legal shuffle_hls
        last_dim = self.transpose_in_shape[-1]
        if last_dim % self.simd != 0:
            new_simd = auto_size_simd(last_dim, self.simd)
            if new_simd is None:
                raise FINNUserError(
                    f"{self.onnx_node.name}: unable to determine a SIMD value that divides "
                    f"the transpose dimension ({last_dim})"
                )
            self.set_nodeattr("SIMD", new_simd)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(OuterShuffle.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self) -> None:
        """Return global includes."""
        self.code_gen_dict["$GLOBALS$"] = [
            '#include "input_gen.hpp"',
            "#include <ap_int.h>",
            "#include <hls_vector.h>",
            "#include <hls_stream.h>",
        ]

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Return defines."""
        dtype = self.get_input_datatype()
        self.code_gen_dict["$DEFINES$"] = [
            f"""
            constexpr unsigned  SIMD = {self.simd};
            using  TE = {dtype.get_hls_datatype_str()};
            using  TV = hls::vector<TE, SIMD>;
            """
        ]

    def docompute(self) -> None:
        """Return docompute."""
        out_shape = self.transpose_out_shape
        out_shape[-1] = int(out_shape[-1] / self.simd)
        loop_coeffs = [1 if x == 1 else int(x / self.simd) for x in self.loop_coeffs]
        interleaved = [
            int(item) for pair in zip(out_shape, loop_coeffs, strict=False) for item in pair
        ]
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"""
            hls::stream<TV>  src0;
            hls::stream<TV>  dst0;
            #pragma HLS stream variable=src0 depth=2
            #pragma HLS stream variable=dst0 depth=2

            move(in0_V, src0);
            input_gen<-1,{np.prod(out_shape)},{",".join(map(str, interleaved))}>(src0, dst0);
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

    def timeout_value(self) -> None:
        """Set the timeout value for HLS functions defined for one clock cycle."""
        self.code_gen_dict["$TIMEOUT_VALUE$"] = [str(int(np.prod(self.get_normal_input_shape())))]

############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3 Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

"""HLS backend implementation of the hardware softmax operator."""

import numpy as np
from typing import TYPE_CHECKING

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.hwsoftmax import HWSoftmax, NodeAttrTypes

if TYPE_CHECKING:
    from onnx import GraphProto, NodeProto


class HWSoftmax_hls(HWSoftmax, HLSBackend):
    """Class for HW Softmax hls."""

    def __init__(self, onnx_node: "NodeProto", **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(HWSoftmax.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self) -> None:
        """Return global includes."""
        self.code_gen_dict["$GLOBALS$"] = [
            "#include <hls_vector.h>",
            '#include "softmax.hpp"',
            '#include "utils.hpp"',
        ]

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Return defines."""
        idtype = self.get_input_datatype()
        w = self.ifm_dim[-1]
        self.code_gen_dict["$DEFINES$"] = [
            f"""
            constexpr unsigned  SIMD = {self.simd};
            constexpr unsigned  W = {w};
            using  TI = {idtype.get_hls_datatype_str()};
            using  F = float;
           """
        ]

    def docompute(self) -> None:
        """Return docompute."""
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """
                static hls::stream<hls::vector<TI,SIMD>>  src0;
                static hls::stream<hls::vector<float,SIMD>>  dst0;

                move(in0_V, src0);
                static SoftMax<TI, float, W, SIMD> sm_inst;
                sm_inst.execute(src0, dst0);
                move(dst0, out0_V);
        """
        ]

    def blackboxfunction(self) -> None:
        """Return blackboxfunction."""
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"""
            void {self.onnx_node.name}(
                hls::stream<hls::vector<TI,SIMD>> &in0_V,
                hls::stream<hls::vector<float,SIMD>> &out0_V
                )
            """
        ]

    def pragmas(self) -> None:
        """Return pragmas."""
        self.code_gen_dict["$PRAGMAS$"] = [
            """
            #pragma HLS interface AXIS port=in0_V
            #pragma HLS interface AXIS port=out0_V
            #pragma HLS aggregate  variable=in0_V compact=bit
            #pragma HLS aggregate  variable=out0_V compact=bit

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

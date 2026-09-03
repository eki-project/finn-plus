# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""HLS backend implementation of the streaming channel-split operator."""

import numpy as np
from onnx import GraphProto, NodeProto

from finn.custom_op.fpgadataflow.base.split import NodeAttrTypes, StreamingSplit
from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend


@register_custom_op
class StreamingSplit_hls(StreamingSplit, HLSBackend):
    """Streaming split node with dynamically generated HLS.

    Only supports splitting along the last axis.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(StreamingSplit.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def execute_node(self, context: dict[str, np.ndarray], graph: GraphProto) -> None:
        """Execute node."""
        HLSBackend.execute_node(self, context, graph)

    def global_includes(self) -> None:
        """Return global includes."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "split.hpp"']

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Return defines."""
        self.code_gen_dict["$DEFINES$"] = []

    def docompute(self) -> None:
        """Return docompute."""
        n_outputs = self.get_n_outputs()
        out_stream_folds = ", ".join(
            str(self.get_folded_output_shape(i)[-2]) for i in range(n_outputs)
        )
        out_stream_names = ", ".join(f"out{i}_V" for i in range(n_outputs))
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"StreamingSplit<{out_stream_folds}>(in0_V, {out_stream_names});"
        ]

    def blackboxfunction(self) -> None:
        """Return blackboxfunction."""
        input_elem_hls_type = self.get_input_datatype().get_hls_datatype_str()
        simd = self.simd
        in_stream = f"hls::stream<hls::vector<{input_elem_hls_type}, {simd}>> &in0_V"
        out_streams = ", ".join(
            f"hls::stream<hls::vector<{input_elem_hls_type}, {simd}>> &out{i}_V"
            for i in range(self.get_n_outputs())
        )
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"void {self.onnx_node.name}({in_stream}, {out_streams})"
        ]

    def pragmas(self) -> None:
        """Return pragmas."""
        pragmas = ["#pragma HLS INTERFACE axis port=in0_V"]
        pragmas += [
            f"#pragma HLS INTERFACE axis port=out{i}_V" for i in range(self.get_n_outputs())
        ]
        pragmas.append("#pragma HLS INTERFACE ap_ctrl_none port=return")
        pragmas.append("#pragma HLS aggregate variable=in0_V compact=bit")
        pragmas += [
            f"#pragma HLS aggregate variable=out{i}_V compact=bit"
            for i in range(self.get_n_outputs())
        ]
        self.code_gen_dict["$PRAGMAS$"] = pragmas

    def timeout_condition(self) -> None:
        """Return timeout condition."""
        condition = " && ".join(f"out{i}_V.empty()" for i in range(self.get_n_outputs()))
        self.code_gen_dict["$TIMEOUT_CONDITION$"] = [condition]

    def timeout_read_stream(self) -> None:
        """Return timeout read stream."""
        self.code_gen_dict["$TIMEOUT_READ_STREAM$"] = [
            f"""if(!out{i}_V.empty()){{
                   strm{i} << out{i}_V.read();
                   }}"""
            for i in range(self.get_n_outputs())
        ]

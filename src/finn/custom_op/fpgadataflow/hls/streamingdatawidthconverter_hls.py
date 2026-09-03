# Copyright (C) 2023, Advanced Micro Devices, Inc.
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

"""HLS backend implementation of the streaming data-width converter."""

import numpy as np
from onnx import GraphProto, NodeProto
from typing import cast

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.streamingdatawidthconverter import (
    NodeAttrTypes,
    StreamingDataWidthConverter,
)
from finn.util.exception import FINNInternalError

# does not do anything at the ONNX node-by-node level, and input-output
# tensor shapes are the same. performs data width conversion at the rtlsim level


class StreamingDataWidthConverter_hls(StreamingDataWidthConverter, HLSBackend):
    """Corresponds to the finn-hlslib StreamingDataWidthConverter_Batch function."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(StreamingDataWidthConverter.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self) -> None:
        """Return global includes."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "streamtools.h"']

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Return defines."""
        num_reps = 1
        num_in_words = int(np.prod(self.get_folded_input_shape()[:-1]))
        in_width = cast("int", self.get_nodeattr("inWidth"))
        out_width = cast("int", self.get_nodeattr("outWidth"))
        self.code_gen_dict["$DEFINES$"] = [
            f"#define InWidth {in_width} ",
            f"#define OutWidth {out_width} ",
            f"#define NumInWords {num_in_words} ",
            f"#define numReps {num_reps}",
        ]
        if self.needs_lcm():
            lcm_width = self.get_iowidth_lcm()
            if num_in_words % (lcm_width / in_width) != 0:
                raise FINNInternalError(f"{self.onnx_node.name}: error in DWC LCM calculation")
            num_lcm_to_out = int(num_in_words // (lcm_width / in_width))
            self.code_gen_dict["$DEFINES$"].append(f"#define LCMWidth {lcm_width}")
            self.code_gen_dict["$DEFINES$"].append(f"#define NumLCMToOut {num_lcm_to_out}")

    def strm_decl(self) -> None:
        """Return strm decl."""
        self.code_gen_dict["$STREAMDECLARATIONS$"] = [
            f'hls::stream<ap_uint<{self.get_instream_width()}>> in0_V ("in0_V");',
            f'hls::stream<ap_uint<{self.get_outstream_width()}>> out0_V ("out0_V");',
        ]

    def docompute(self) -> None:
        """Return docompute."""
        op = "StreamingDataWidthConverter_Batch"
        if self.needs_lcm():
            self.code_gen_dict["$DOCOMPUTE$"] = [
                f'hls::stream<ap_uint<{self.get_iowidth_lcm()}>> intermediate ("intermediate");',
                f"{op}<InWidth, LCMWidth, NumInWords>(in0_V, intermediate, numReps);",
                f"{op}<LCMWidth, OutWidth, NumLCMToOut>(intermediate, out0_V, numReps);",
            ]
        else:
            self.code_gen_dict["$DOCOMPUTE$"] = [
                f"{op}<InWidth, OutWidth, NumInWords>(in0_V, out0_V, numReps);"
            ]

    def blackboxfunction(self) -> None:
        """Return blackboxfunction."""
        in_packed_hls_type = f"ap_uint<{self.get_instream_width()}>"
        out_packed_hls_type = f"ap_uint<{self.get_outstream_width()}>"
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"void {self.onnx_node.name}(hls::stream<{in_packed_hls_type} > &in0_V, "
            f"hls::stream<{out_packed_hls_type} > &out0_V)"
        ]

    def pragmas(self) -> None:
        """Return pragmas."""
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0_V",
            "#pragma HLS INTERFACE axis port=out0_V",
            "#pragma HLS INTERFACE ap_ctrl_none port=return",
        ]
        if self.needs_lcm():
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS DATAFLOW disable_start_propagation")

    def execute_node(self, context: dict[str, np.ndarray], graph: GraphProto) -> None:
        """Execute node."""
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            exp_shape = self.get_normal_input_shape()
            output = context[self.onnx_node.input[0]]
            output = np.asarray([output], dtype=np.float32).reshape(*exp_shape)
            context[self.onnx_node.output[0]] = output
        elif mode == "rtlsim":
            HLSBackend.execute_node(self, context, graph)

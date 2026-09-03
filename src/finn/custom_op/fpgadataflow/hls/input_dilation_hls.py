# Copyright (c) 2024, Advanced Micro Devices, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

"""HLS backend implementation of the input-dilation operator."""

import numpy as np
from typing import TYPE_CHECKING

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.input_dilation import InputDilation, NodeAttrTypes

if TYPE_CHECKING:
    from onnx import GraphProto, NodeProto


class InputDilation_hls(InputDilation, HLSBackend):
    """HLS backend implementation of input dilation.

    Uses the finn-hlslib ``FMPadding_Pixel_Nonsquare`` streamtools function.
    """

    def __init__(self, onnx_node: "NodeProto", **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(InputDilation.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self) -> None:
        """Return global includes."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "streamtools.h"']

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Return defines."""
        odim_h, odim_w = self.get_padded_odim()
        stride_h, stride_w = self.stride
        self.code_gen_dict["$DEFINES$"] = [
            f"""
            #define OutputDim_x {odim_w}\n
            #define OutputDim_y {odim_h}\n
            #define Stride_x {stride_w}\n
            #define Stride_y {stride_h}\n
            #define NumChannels {self.num_channels}\n
            #define SIMD {self.simd}\n
            """
        ]

    def docompute(self) -> None:
        """Return docompute."""
        in_t = self.get_input_datatype().get_hls_datatype_str()
        hls_call = "FMPadding_Pixel_Nonsquare"
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"""{hls_call}<OutputDim_x, OutputDim_y, Stride_x, Stride_y, NumChannels,
            SIMD, {in_t}> (in0_V, out0_V);"""
        ]

    def blackboxfunction(self) -> None:
        """Return blackboxfunction."""
        packed_hls_type = f"ap_uint<{self.get_instream_width()}>"
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"void {self.onnx_node.name}(hls::stream<{packed_hls_type} > &in0_V, "
            f"hls::stream<{packed_hls_type} > &out0_V)"
        ]

    def execute_node(self, context: dict[str, np.ndarray], graph: "GraphProto") -> None:
        """Execute node."""
        HLSBackend.execute_node(self, context, graph)

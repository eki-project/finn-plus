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

"""HLS backend implementation of the generic pooling operator."""

import numpy as np
from onnx import GraphProto, NodeProto

from finn.custom_op.fpgadataflow.base.pool import NodeAttrTypes, Pool
from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.exception import FINNUserError


@register_custom_op
class Pool_hls(Pool, HLSBackend):
    """Class that corresponds to the finn-hlslib ``Pool_batch`` function.

    Requires ``ConvolutionInputGenerator(depthwise == 1)`` to format its input.

    Input shape ``(BatchSize, OutImgDim, OutImgDim, TotalKernelSize * Channels)``
    Output shape ``(BatchSize, OutImgDim, OutImgDim, Channels)``

    Notes:
    * The input shape was chosen to be compatible with im2col (only true when there
      is not folding).
    * The actual data layout produced by the hlslib kernels is different
      for depthwise ops.

        * depthwise SWG: ``(1, OFMDim, OFMDim, IFMChannels/PE, K, K, PE)``

    Channels can be folded using PE (SIMD from the input perspective).
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Get dictionary of custom node attributes with their types and default values."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(Pool.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self) -> None:
        """List include directives for generated HLS code."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "pool.hpp"']

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Constant and type definitions for generated HLS code."""
        k = int(np.prod(self.kernel_size))
        cf = self.channels // self.pe
        osz = np.prod(self.out_img_dims)
        self.code_gen_dict["$DEFINES$"] = [
            f"constexpr unsigned  ISIZE = {osz * cf * k};",
            f"constexpr unsigned  K = {k};",
        ]

    def docompute(self) -> None:
        """Generate the computational part of the HLS C++ code."""
        pe = self.pe
        fxn = self.function
        idt = self.get_input_datatype()
        odt = self.get_output_datatype()
        o_hls_dt = f"hls::vector<{odt.get_hls_datatype_str()}, {pe}>"
        sign = "" if idt.signed() else "u"
        act_hls_dt = f"hls::vector<ap_{sign}int<{self.accum_bits}>, {pe}>"

        self.code_gen_dict["$DOCOMPUTE$"] = []
        if fxn == "MaxPool":
            self.code_gen_dict["$DOCOMPUTE$"] += [f"MaxPoolFunction<{o_hls_dt}> pool_fxn;"]
        elif fxn == "AccPool":
            self.code_gen_dict["$DOCOMPUTE$"] += [f"AccPoolFunction<{o_hls_dt}> pool_fxn;"]
        elif fxn == "AvgPool":
            n = np.prod(self.kernel_size)
            self.code_gen_dict["$DOCOMPUTE$"] += [
                f"AvgPoolFunction<{o_hls_dt},{act_hls_dt},{n}> pool_fxn;"
            ]
        elif fxn == "QuantAvgPool":
            self.code_gen_dict["$DOCOMPUTE$"] += [
                f"QuantAvgPoolFunction<{o_hls_dt},{act_hls_dt},{self.size}> pool_fxn;"
            ]
        else:
            raise FINNUserError(f"{self.onnx_node.name}: Pool_Batch does not support {fxn}")

        self.code_gen_dict["$DOCOMPUTE$"] += ["Pool_batch<ISIZE, K>(in0_V, out0_V, pool_fxn);"]

    def pragmas(self) -> None:
        """Generate HLS pragmas to apply to the HLS C++ code."""
        super().pragmas()
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS dataflow disable_start_propagation")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS aggregate variable=in0_V compact=bit")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS aggregate variable=out0_V compact=bit")

    def blackboxfunction(self) -> None:
        """Blackbox function interface from which the IP will be generated."""
        pe = self.pe
        idt = self.get_input_datatype()
        odt = self.get_output_datatype()
        i_hls_dt = f"hls::vector<{idt.get_hls_datatype_str()}, {pe}>"
        o_hls_dt = f"hls::vector<{odt.get_hls_datatype_str()}, {pe}>"

        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            (
                f"void {self.onnx_node.name}(hls::stream<{i_hls_dt}> &in0_V, "
                f"hls::stream<{o_hls_dt}> &out0_V)"
            )
        ]

    def execute_node(self, context: dict[str, np.ndarray], graph: GraphProto) -> None:
        """Execute the node in HLS C++ simulation."""
        HLSBackend.execute_node(self, context, graph)

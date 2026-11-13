# Copyright (C) 2023-2024, Advanced Micro Devices, Inc.
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
from typing import cast

from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.hwcustomop import NodeAttrTypes
from finn.custom_op.fpgadataflow.pool import Pool
from finn.util.data_packing import hls_datatype_str
from finn.util.exception import FINNUserError


@register_custom_op
class Pool_hls(Pool, HLSBackend):
    """Consolidated HLS pooling: fuses pad<> + input_gen<> + pool<F>()
    into a single HLS dataflow kernel.

    Input shape (BatchSize, IFMDim_H, IFMDim_W, Channels)  — raw feature map
    Output shape (BatchSize, OutImgDim_H, OutImgDim_W, Channels)

    Channels can be folded using PE.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(Pool.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def _padding(self) -> list[int]:
        """Return the padding as [north, west, south, east]."""
        return cast("list[int]", self.get_nodeattr("Padding"))

    def global_includes(self) -> None:
        """List include directives for generated HLS code."""
        includes = ['#include "pool.hpp"', '#include "input_gen.hpp"']
        if any(p != 0 for p in self._padding()):
            includes.append('#include "pad.hpp"')
        self.code_gen_dict["$GLOBALS$"] = includes

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Constant and type definitions for generated HLS code."""
        pe = self.pe
        cf = self.channels // pe
        kh, kw = self.kernel_size
        sh, sw = cast("list[int]", self.get_nodeattr("Stride"))
        ifm_h, ifm_w = cast("list[int]", self.get_nodeattr("IFMDim"))
        ofm_h, ofm_w = self.out_img_dims
        pad = self._padding()
        has_padding = any(p != 0 for p in pad)

        # Padded dimensions (what input_gen sees)
        ph = ifm_h + pad[0] + pad[2]
        pw = ifm_w + pad[1] + pad[3]

        defines = [
            f"static constexpr unsigned PE = {pe};",
            f"static constexpr unsigned CF = {cf};",
            f"static constexpr unsigned H = {ph};",
            f"static constexpr unsigned W = {pw};",
            f"static constexpr unsigned KH = {kh};",
            f"static constexpr unsigned KW = {kw};",
            f"static constexpr unsigned SH = {sh};",
            f"static constexpr unsigned SW = {sw};",
            f"static constexpr unsigned OH = {ofm_h};",
            f"static constexpr unsigned OW = {ofm_w};",
        ]
        if has_padding:
            defines += [
                f"static constexpr unsigned IFM_H = {ifm_h};",
                f"static constexpr unsigned IFM_W = {ifm_w};",
                f"static constexpr unsigned PAD_N = {pad[0]};",
                f"static constexpr unsigned PAD_W = {pad[1]};",
                f"static constexpr unsigned PAD_S = {pad[2]};",
                f"static constexpr unsigned PAD_E = {pad[3]};",
            ]
        self.code_gen_dict["$DEFINES$"] = defines

    def docompute(self) -> None:
        """Generate the function body of the HLS code."""
        pe = self.pe
        fxn = self.function
        idt = self.get_input_datatype()
        odt = self.get_output_datatype()
        i_tensor_dt = f"tensor<{hls_datatype_str(idt)}, {pe}>"
        o_tensor_dt = f"tensor<{hls_datatype_str(odt)}, {pe}>"
        sign = "" if idt.signed() else "u"
        act_tensor_dt = f"tensor<ap_{sign}int<{self.accum_bits}>, {pe}>"

        has_padding = any(p != 0 for p in self._padding())

        lines = []

        # Pool functor type (template parameter F of pool<F>())
        if fxn == "MaxPool":
            pool_type = f"MaxPool<{o_tensor_dt}>"
        elif fxn == "AccPool":
            pool_type = f"SumPool<{o_tensor_dt}>"
        elif fxn == "AvgPool":
            n = int(np.prod(self.kernel_size))
            pool_type = f"AvgPool<{o_tensor_dt},{act_tensor_dt},{n}>"
        elif fxn == "QuantAvgPool":
            pool_type = f"QuantAvgPool<{o_tensor_dt},{act_tensor_dt},{self.size}>"
        else:
            raise FINNUserError(f"{self.onnx_node.name}: Pool does not support {fxn}")

        # input_gen loop nest for depthwise 2D pooling with stride:
        #   for oh in OH: for ow in OW: for cf in CF: for kh in KH: for kw in KW:
        #     addr = (oh*SH+kh)*W*CF + (ow*SW+kw)*CF + cf
        # M=2: flit_t last marker fires upon completion of 2 innermost loops
        #       (KW and KH), i.e. every KH*KW = K elements — one kernel window.
        ig_call = "input_gen<2, H*W*CF, OH, SH*W*CF, OW, SW*CF, CF, 1, KH, W*CF, KW, CF>"
        flit_dt = f"flit_t<{i_tensor_dt}>"
        pool_call = f"pool<{pool_type}>"

        if has_padding:
            # Pad value expression
            pad_val = int(cast("int", self.get_nodeattr("PadValue")))
            lines.append(f'static hls::stream<{i_tensor_dt}>  padded("padded");')
            lines.append("#pragma HLS stream depth=2 variable=padded")
            lines.append(f'static hls::stream<{flit_dt}>  feed("feed");')
            lines.append("#pragma HLS stream depth=2 variable=feed")
            # Build pad value initializer (broadcasting constructor)
            lines.append(
                "pad<IFM_H, IFM_W, CF, PAD_N, PAD_E, PAD_S, PAD_W>"
                f"(in0_V, padded, {i_tensor_dt}({hls_datatype_str(idt)}({pad_val})));"
            )
            lines.append(f"{ig_call}(padded, feed);")
            lines.append(f"{pool_call}(feed, out0_V);")
        else:
            lines.append(f'static hls::stream<{flit_dt}>  feed("feed");')
            lines.append("#pragma HLS stream depth=2 variable=feed")
            lines.append(f"{ig_call}(in0_V, feed);")
            lines.append(f"{pool_call}(feed, out0_V);")

        self.code_gen_dict["$DOCOMPUTE$"] = lines

    def pragmas(self) -> None:
        """Pragma directives for generated HLS code."""
        super().pragmas()
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS dataflow disable_start_propagation")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS aggregate variable=in0_V compact=bit")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS aggregate variable=out0_V compact=bit")

    def timeout_value(self) -> None:
        """Set the number of consecutive calls without output after which cppsim stops.

        input_gen must buffer up to (KH-1) padded rows of CF folds before the first
        window completes, which exceeds the default of 1000 for wide/deeply folded
        inputs and left the final stream read blocking forever. Bound it by the
        expected cycle count instead.
        """
        self.code_gen_dict["$TIMEOUT_VALUE$"] = [str(self.get_exp_cycles() + 1000)]

    def ipgen_extra_directives(self) -> list[str]:
        """Return a list of extra tcl directives for HLS synthesis."""
        return ["config_compile -pipeline_style flp"]

    def blackboxfunction(self) -> None:
        """Blackbox function interface from which the IP will be generated."""
        pe = self.pe
        i_tensor_dt = f"tensor<{hls_datatype_str(self.get_input_datatype())}, {pe}>"
        o_tensor_dt = f"tensor<{hls_datatype_str(self.get_output_datatype())}, {pe}>"

        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            (
                f"void {self.onnx_node.name}(hls::stream<{i_tensor_dt}> &in0_V, "
                f"hls::stream<{o_tensor_dt}> &out0_V)"
            )
        ]

    def execute_node(self, context: dict[str, np.ndarray], graph: GraphProto) -> None:
        """Execute the node in HLS C++ simulation."""
        HLSBackend.execute_node(self, context, graph)

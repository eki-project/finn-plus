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

"""HLS backend implementation of the streaming FIFO (virtual FIFO for live sizing)."""

import numpy as np
from onnx import GraphProto, NodeProto
from pathlib import Path
from qonnx.core.datatype import DataType
from typing import cast

from finn.custom_op.fpgadataflow.base.streamingfifo import NodeAttrTypes, StreamingFIFO
from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.exception import FINNInternalError


@register_custom_op
class StreamingFIFO_hls(StreamingFIFO, HLSBackend):
    """HLS-based FIFO implementation.

    Currently only used as a virtual FIFO for live FIFO-sizing.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {
            # Only purpose of this CustomOp for now: virtual FIFO for live FIFO-sizing
            "impl_style": ("s", False, "virtual", {"virtual"}),
        }
        my_attrs.update(StreamingFIFO.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self) -> None:
        """Add the global include for the virtual FIFO implementation."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "virtual_fifo.hpp"']

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Return defines."""
        num_reps = 1
        width = self.get_instream_width()
        self.code_gen_dict["$DEFINES$"] = [
            f"#define Width {width} ",
            f"#define numReps {num_reps}",
        ]

    def strm_decl(self) -> None:
        """Return strm decl."""
        sname = self.hls_sname()
        self.code_gen_dict["$STREAMDECLARATIONS$"] = [
            f'hls::stream<ap_uint<{self.get_instream_width()}>> in0_{sname} ("in0_{sname}");',
            f'hls::stream<ap_uint<{self.get_outstream_width()}>> out0_{sname} ("out0_{sname}");',
        ]

    def docompute(self) -> None:
        """Return docompute."""
        sname = self.hls_sname()
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"""
            #pragma HLS dataflow disable_start_propagation

            static hls::stream<ap_uint<Width>> in_fifo;
            static hls::stream<Payload<ap_uint<Width>>::type> out_fifo;
            #pragma HLS stream variable=in_fifo depth=2
            #pragma HLS stream variable=out_fifo depth=2

            // AXI-Stream -> FIFO
            move(in0_{sname}, in_fifo);

            // Main
            VirtualFIFO<Width>(in_fifo, out_fifo, mode, depth, occupancy, max_occupancy);

            // FIFO -> AXI-Stream
            move(out_fifo, out0_{sname});
            """
        ]

    def blackboxfunction(self) -> None:
        """Return blackboxfunction."""
        sname = self.hls_sname()
        in_packed_hls_type = f"ap_uint<{self.get_instream_width()}>"
        out_packed_hls_type = f"ap_uint<{self.get_outstream_width()}>"
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"""void {self.onnx_node.name}(
            hls::stream<{in_packed_hls_type} > &in0_{sname},
            hls::stream<{out_packed_hls_type} > &out0_{sname}, ap_uint<32> mode,
            ap_uint<32> depth, ap_uint<32> &occupancy, ap_uint<32> &max_occupancy)"""
        ]

    def pragmas(self) -> None:
        """Return pragmas."""
        sname = self.hls_sname()
        self.code_gen_dict["$PRAGMAS$"] = [
            f"#pragma HLS INTERFACE axis port=in0_{sname}",
            f"#pragma HLS INTERFACE axis port=out0_{sname}",
            "#pragma HLS INTERFACE s_axilite port=mode",
            "#pragma HLS INTERFACE s_axilite port=depth",
            "#pragma HLS INTERFACE s_axilite port=occupancy",
            "#pragma HLS INTERFACE s_axilite port=max_occupancy",
            "#pragma HLS INTERFACE ap_ctrl_none port=return",
        ]

    def get_verilog_top_module_intf_names(self) -> dict[str, list[tuple[str, int]] | list[str]]:
        """Return verilog top module intf names (adds the AXI-lite control interface)."""
        intf_names = super().get_verilog_top_module_intf_names()
        intf_names["axilite"] = ["s_axi_control"]
        return intf_names

    def execute_node(
        self, context: dict[str, np.ndarray], graph: GraphProto  # noqa: ARG002
    ) -> None:
        """Execute node.

        Only ``cppsim`` (a shape-preserving no-op) is supported; the virtual
        HLS FIFO has no standalone RTL simulation model.
        """
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_shape = self.get_normal_input_shape()
        folded_ishape = self.get_folded_input_shape()

        if mode != "cppsim":
            raise FINNInternalError(
                f"{node.name}: exec_mode {mode} is not supported for the virtual HLS FIFO"
            )
        code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_cppsim")))

        inp = context[node.input[0]]
        if str(inp.dtype) != "float32":
            raise FINNInternalError(f"{node.name}: input datatype is not float32")
        if inp.shape != tuple(exp_shape):
            raise FINNInternalError(f"{node.name}: input shape {inp.shape} != {tuple(exp_shape)}")

        if self.get_input_datatype() == DataType["BIPOLAR"]:
            # store bipolar activations as binary
            inp = (inp + 1) / 2
        # reshape input into folded shape and make a copy before saving
        reshaped_input = inp.reshape(folded_ishape).copy()
        np.save(str(code_gen_dir / "input_0.npy"), reshaped_input)

        context[node.output[0]] = np.asarray([inp], dtype=np.float32).reshape(*exp_shape)
        # binary -> bipolar if needed
        if self.get_output_datatype() == DataType["BIPOLAR"]:
            context[node.output[0]] = 2 * context[node.output[0]] - 1
        if context[node.output[0]].shape != tuple(exp_shape):
            raise FINNInternalError(
                f"{node.name}: output shape {context[node.output[0]].shape} != {tuple(exp_shape)}"
            )

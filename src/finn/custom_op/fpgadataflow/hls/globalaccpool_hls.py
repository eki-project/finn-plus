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

"""HLS backend implementation of the global accumulate-pooling operator."""

import numpy as np
from typing import TYPE_CHECKING

from finn.custom_op.fpgadataflow.globalaccpool import GlobalAccPool
from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.hwcustomop import NodeAttrTypes
from finn.util.data_packing import hls_datatype_str
from finn.util.exception import FINNInternalError

if TYPE_CHECKING:
    from onnx import GraphProto, NodeProto


@register_custom_op
class GlobalAccPool_hls(GlobalAccPool, HLSBackend):
    """Class that corresponds to finn-hlslib pool_global function."""

    def __init__(self, onnx_node: "NodeProto", **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(GlobalAccPool.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def verify_node(self) -> list[str]:
        """Verify node."""
        info_messages = []
        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify that all necessary attributes exist
        try:
            self.get_nodeattr("code_gen_dir_cppsim")
            self.get_nodeattr("executable_path")
            self.get_nodeattr("NumChannels")
            self.get_nodeattr("PE")
            self.get_nodeattr("inputDataType")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append("The required GlobalAccPool attributes do not exist.")

        # verify that input data is 2D
        if len(self.num_input_vectors) != 3:
            raise FINNInternalError(
                f"{self.onnx_node.name}: GlobalAccPool requires 2D data input "
                f"(numInputVectors of length 3), got {self.num_input_vectors}"
            )

        return info_messages

    def execute_node(self, context: dict[str, np.ndarray], graph: "GraphProto") -> None:
        """Execute node."""
        HLSBackend.execute_node(self, context, graph)

    def global_includes(self) -> None:
        """Return global includes."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "pool.hpp"']

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Return defines."""
        pe = self.pe
        cf = self.num_channels // pe
        ishape = self.get_normal_input_shape()
        pixels = ishape[1] * ishape[2]
        self.code_gen_dict["$DEFINES$"] = [
            f"static constexpr unsigned PE = {pe};",
            f"static constexpr unsigned CF = {cf};",
            f"static constexpr unsigned PIXELS = {pixels};",
        ]

    def docompute(self) -> None:
        """Return docompute."""
        o_tensor_dt = f"tensor<{hls_datatype_str(self.get_output_datatype())}, {self.pe}>"
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"pool_global<SumPool<{o_tensor_dt}>, CF, PIXELS>(in0_V, out0_V);"
        ]

    def timeout_value(self) -> None:
        """Set the number of consecutive calls without output after which cppsim stops.

        pool_global is function-pipelined and processes one input per call.
        It needs CF * PIXELS calls before the first output, so the
        default timeout of 1000 is insufficient for large configurations.
        """
        ishape = self.get_normal_input_shape()
        pixels = ishape[1] * ishape[2]
        total_inputs = (self.num_channels // self.pe) * pixels
        self.code_gen_dict["$TIMEOUT_VALUE$"] = [str(total_inputs + 1000)]

    def ipgen_extra_directives(self) -> list[str]:
        """Return a list of extra tcl directives for HLS synthesis."""
        return ["config_compile -pipeline_style flp"]

    def blackboxfunction(self) -> None:
        """Return blackboxfunction."""
        pe = self.pe
        idt = hls_datatype_str(self.get_input_datatype())
        odt = hls_datatype_str(self.get_output_datatype())
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            (
                f"void {self.onnx_node.name}(hls::stream<tensor<{idt}, {pe}>> &in0_V, "
                f"hls::stream<tensor<{odt}, {pe}>> &out0_V)"
            )
        ]

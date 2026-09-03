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

from finn.custom_op.fpgadataflow.globalaccpool import GlobalAccPool, NodeAttrTypes
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.exception import FINNInternalError

if TYPE_CHECKING:
    from onnx import GraphProto, NodeProto


class GlobalAccPool_hls(GlobalAccPool, HLSBackend):
    """Class that corresponds to finn-hlslib AccPool_Batch function."""

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
            info_messages.append("The required GlobalAccPool_Batch attributes do not exist.")

        # verify that input data is 2D
        if len(self.num_input_vectors) != 3:
            raise FINNInternalError(
                f"{self.onnx_node.name}: GlobalAccPool_Batch requires 2D data input "
                f"(numInputVectors of length 3), got {self.num_input_vectors}"
            )

        return info_messages

    def execute_node(self, context: dict[str, np.ndarray], graph: "GraphProto") -> None:
        """Execute node."""
        HLSBackend.execute_node(self, context, graph)

    def global_includes(self) -> None:
        """Return global includes."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "maxpool.h"']

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Return defines."""
        self.code_gen_dict["$DEFINES$"] = []

    def docompute(self) -> None:
        """Return docompute."""
        img_dim = self.get_normal_input_shape()[1]
        in_hls_type = self.get_input_datatype().get_hls_datatype_str()
        out_hls_type = self.get_output_datatype().get_hls_datatype_str()
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"AccPool_Batch<{img_dim}, {self.num_channels}, {in_hls_type}, "
            f"{self.pe}, {out_hls_type}> (in0_V, out0_V, 1);"
        ]

    def blackboxfunction(self) -> None:
        """Return blackboxfunction."""
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"""void {self.onnx_node.name}(hls::stream<ap_uint<{self.get_instream_width()}>> &in0_V,
                hls::stream<ap_uint<{self.get_outstream_width()}>> &out0_V)"""
        ]

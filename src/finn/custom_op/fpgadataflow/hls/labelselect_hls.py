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

"""HLS backend implementation of the top-K label-selection operator."""

import numpy as np
from typing import TYPE_CHECKING

from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.hwcustomop import NodeAttrTypes
from finn.custom_op.fpgadataflow.labelselect import LabelSelect
from finn.util.data_packing import hls_datatype_str
from finn.util.exception import FINNInternalError

if TYPE_CHECKING:
    from onnx import GraphProto, NodeProto


@register_custom_op
class LabelSelect_hls(LabelSelect, HLSBackend):
    """Class that corresponds to finn-hlslib LabelSelect_Batch function."""

    def __init__(self, onnx_node: "NodeProto", **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(LabelSelect.get_nodeattr_types(self))
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
            self.get_nodeattr("Labels")
            self.get_nodeattr("PE")
            self.get_nodeattr("K")
            self.get_nodeattr("inputDataType")
            self.get_nodeattr("outputDataType")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append("The required LabelSelect_Batch attributes do not exist.")

        # verify that input data is 1D
        if len(self.num_input_vectors) > 1:
            raise FINNInternalError(
                f"{self.onnx_node.name}: LabelSelect_Batch requires 1D data input "
                f"(numInputVectors of length 1), got {self.num_input_vectors}"
            )

        return info_messages

    def execute_node(self, context: dict[str, np.ndarray], graph: "GraphProto") -> None:
        """Execute node."""
        HLSBackend.execute_node(self, context, graph)
        # TopK ind output normally uses TensorProto.INT64, which
        # can cause issues for the node-by-node simulation in FINN
        # (as the custom DataType system always assumes float containers)
        # so cast the output to int64
        outp = self.onnx_node.output[0]
        context[outp] = context[outp].astype(np.int64)

    def global_includes(self) -> None:
        """Return global includes."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "topk.hpp"']

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Return defines."""
        self.code_gen_dict["$DEFINES$"] = []

    def read_npy_data(self) -> None:
        """Return read npy data."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        npy_type = "float"
        npy_in = f"{code_gen_dir}/input_0.npy"
        self.code_gen_dict["$READNPYDATA$"] = [
            (f'npy2tensorstream<{npy_type}>("{npy_in}", in0_V);')
        ]

    def docompute(self) -> None:
        """Return docompute."""
        # PE and the input/output element types are inferred from the stream
        # types tensor<TI, PE> / tensor<TO, 1> declared by strm_decl/blackboxfunction
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"LabelSelect_Batch<{self.labels}, {self.k}>(in0_V, out0_V);"
        ]

    def timeout_value(self) -> None:
        """Set the number of consecutive calls without output after which cppsim stops.

        LabelSelect_Batch is function-pipelined and processes one input per call.
        It needs Labels/PE calls before the first output, so the default timeout
        of 1000 is insufficient for 1000 or more labels (e.g. ImageNet).
        """
        total_inputs = self.labels // self.pe
        self.code_gen_dict["$TIMEOUT_VALUE$"] = [str(total_inputs + 1000)]

    def ipgen_extra_directives(self) -> list[str]:
        """Return a list of extra tcl directives for HLS synthesis."""
        return ["config_compile -pipeline_style frp"]

    def blackboxfunction(self) -> None:
        """Return blackboxfunction."""
        idt = hls_datatype_str(self.get_input_datatype())
        odt = hls_datatype_str(self.get_output_datatype())
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            (
                f"void {self.onnx_node.name}(hls::stream<tensor<{idt}, {self.pe}>> &in0_V, "
                f"hls::stream<tensor<{odt}, 1>> &out0_V)"
            )
        ]

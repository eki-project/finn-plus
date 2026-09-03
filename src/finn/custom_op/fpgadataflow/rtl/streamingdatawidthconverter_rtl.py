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

"""RTL backend implementation of the streaming data-width converter."""

import numpy as np
import shutil
from onnx import GraphProto, NodeProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from typing import cast

from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.custom_op.fpgadataflow.streamingdatawidthconverter import (
    NodeAttrTypes,
    StreamingDataWidthConverter,
)
from finn.util.exception import FINNUserError
from finn.util.settings import get_settings

_RTL_SOURCES = ["dwc_axi.sv", "dwc.sv"]


class StreamingDataWidthConverter_rtl(StreamingDataWidthConverter, RTLBackend):
    """Corresponds to the finn-rtllib data-width converter module."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Get the attribute types for this node."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(StreamingDataWidthConverter.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def check_divisible_iowidths(self) -> None:
        """Check that input and output stream widths have an integer ratio.

        The RTL module only supports stream widths that are integer width
        ratios of one another.
        """
        iwidth = cast("int", self.get_nodeattr("inWidth"))
        owidth = cast("int", self.get_nodeattr("outWidth"))
        if iwidth % owidth != 0 and owidth % iwidth != 0:
            raise FINNUserError(
                f"{self.onnx_node.name}: the RTL DWC requires stream widths that are integer "
                f"ratios of each other, but inWidth={iwidth} and outWidth={owidth}"
            )

    def execute_node(self, context: dict[str, np.ndarray], graph: GraphProto) -> None:
        """Execute the node in the given context and graph for simulation."""
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            StreamingDataWidthConverter.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)

    def get_template_values(self) -> dict[str, str | int]:
        """Get the code generation template values for this node."""
        return {
            "IBITS": int(self.get_instream_width()),
            "OBITS": int(self.get_outstream_width()),
            "TOP_MODULE_NAME": self.get_verilog_top_module_name(),
        }

    def generate_hdl(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:  # noqa: ARG002
        """Generate the HDL code for this node."""
        rtlsrc = Path(get_settings().finn_rtllib) / "dwc" / "hdl"
        template_path = rtlsrc / "dwc_template.v"
        code_gen_dict = self.get_template_values()
        topname = self.get_verilog_top_module_name()
        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", topname)

        code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))
        template = template_path.read_text()
        for key_name, value in code_gen_dict.items():
            template = template.replace(f"${key_name}$", str(value))
        (code_gen_dir / f"{topname}.v").write_text(template)

        for sv_file in _RTL_SOURCES:
            shutil.copy(rtlsrc / sv_file, code_gen_dir)
        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", str(code_gen_dir))
        self.set_nodeattr("ip_path", str(code_gen_dir))

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        """Get list of RTL files required for this node."""
        if abspath:
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen")) + "/"
            rtllib_dir = str(Path(get_settings().finn_rtllib) / "dwc" / "hdl") + "/"
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        top_module = cast("str", self.get_nodeattr("gen_top_module"))
        return [rtllib_dir + f for f in _RTL_SOURCES] + [f"{code_gen_dir}{top_module}.v"]

    def code_generation_ipi(self) -> list[str]:
        """Construct and return the TCL for node instantiation in Vivado IPI."""
        code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))
        top_module = cast("str", self.get_nodeattr("gen_top_module"))
        sourcefiles = [str(code_gen_dir / f) for f in [*_RTL_SOURCES, f"{top_module}.v"]]

        cmd = [f"add_files -norecurse {f}" for f in sourcefiles]
        cmd += [f"create_bd_cell -type module -reference {top_module} {self.onnx_node.name}"]
        return cmd

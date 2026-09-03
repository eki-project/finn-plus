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

"""RTL implementation of FMPadding for feature map padding.

This module provides an RTL-based implementation of feature map padding using the
finn-rtllib fmpadding_axi component. Supports runtime reconfiguration of padding
amounts and spatial feature sizes via optional AXI-Lite interface.
"""

import math
import numpy as np
import shutil
from onnx import NodeProto
from pathlib import Path
from qonnx.core.datatype import BaseDataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import roundup_to_integer_multiple
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow.fmpadding import FMPadding, NodeAttrTypes
from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from onnx import GraphProto


@register_custom_op
class FMPadding_rtl(FMPadding, RTLBackend):
    """CustomOp wrapper for the finn-rtllib fmpadding_axi component.

    Supports adjusting the padding amount and spatial feature sizes at runtime.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize the RTL FMPadding component."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types, including dynamic_mode for runtime reconfiguration."""
        my_attrs: NodeAttrTypes = {
            # Enable reprogrammable implementation to change FM dimensions,
            # stride, or dilation during runtime
            "dynamic_mode": ("i", False, 0, {0, 1}),
        }
        my_attrs.update(FMPadding.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    @property
    def dynamic_mode(self) -> bool:
        """Get whether runtime-reprogrammable (dynamic) mode is enabled."""
        return bool(self.get_nodeattr("dynamic_mode"))

    def get_verilog_top_module_intf_names(self) -> dict[str, list[tuple[str, int]] | list[str]]:
        """Return Verilog top module interface names.

        Overloads the default implementation to add the optional AXI-Lite
        control interface when dynamic_mode is enabled.
        """
        intf_names = super().get_verilog_top_module_intf_names()
        if self.dynamic_mode:
            intf_names["axilite"] = ["s_axilite"]
        return intf_names

    def get_template_values(
        self,
        ifm_dims: list[int],
        pads: list[int],
        chans: int,
        simd: int,
        idt: BaseDataType,
    ) -> dict[str, str | int]:
        """Calculate template parameter values for HDL generation.

        Parameters
        ----------
        ifm_dims : list
            Input feature map dimensions [H, W]
        pads : list
            Padding amounts [top, left, bottom, right]
        chans : int
            Number of channels
        simd : int
            SIMD parallelism factor
        idt : DataType
            Input data type

        Returns
        -------
        dict
            Dictionary of template substitution values for HDL generation
        """
        dim_y, dim_x = ifm_dims
        pad_t, pad_l, pad_b, pad_r = pads
        y_counter_bits = math.ceil(math.log2(pad_t + dim_y + pad_b + 1))
        x_counter_bits = math.ceil(math.log2(pad_l + dim_x + pad_r + 1))
        topname = self.get_verilog_top_module_name()
        stream_bits = idt.bitwidth() * simd
        stream_bits = int(roundup_to_integer_multiple(stream_bits, 8))
        return {
            "XCOUNTER_BITS": x_counter_bits,
            "YCOUNTER_BITS": y_counter_bits,
            "NUM_CHANNELS": int(chans),
            "SIMD": int(simd),
            "ELEM_BITS": idt.bitwidth(),
            "TOP_MODULE_NAME": topname,
            "INIT_XON": int(pad_l),
            "INIT_XOFF": int(pad_l + dim_x),
            "INIT_XEND": int(pad_l + dim_x + pad_r - 1),
            "INIT_YON": int(pad_t),
            "INIT_YOFF": int(pad_t + dim_y),
            "INIT_YEND": int(pad_t + dim_y + pad_b - 1),
            "STREAM_BITS": int(stream_bits),
        }

    def get_dynamic_config(
        self, ifm_dims: list[int] | None = None, pads: list[int] | None = None
    ) -> dict[str, tuple[int, int]]:
        """Return a configuration dict to re-configure FM dimension and
        padding amounts during runtime.
        """
        if ifm_dims is None:
            ifm_dims = self.img_dim
        if pads is None:
            pads = self.padding
        code_gen_dict = self.get_template_values(
            ifm_dims, pads, self.num_channels, self.simd, self.get_input_datatype()
        )
        return {
            "XON": (0 * 4, int(code_gen_dict["INIT_XON"])),
            "XOFF": (1 * 4, int(code_gen_dict["INIT_XOFF"])),
            "XEND": (2 * 4, int(code_gen_dict["INIT_XEND"])),
            "YON": (3 * 4, int(code_gen_dict["INIT_YON"])),
            "YOFF": (4 * 4, int(code_gen_dict["INIT_YOFF"])),
            "YEND": (5 * 4, int(code_gen_dict["INIT_YEND"])),
        }

    def generate_hdl(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:  # noqa: ARG002
        """Generate HDL code from templates for this node."""
        rtlsrc = Path(get_settings().finn_rtllib) / "fmpadding/hdl"
        template_path = rtlsrc / "fmpadding_template.v"
        code_gen_dict = self.get_template_values(
            self.img_dim, self.padding, self.num_channels, self.simd, self.get_input_datatype()
        )
        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        # apply code generation to templates
        code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))
        template = template_path.read_text()
        for key_name, value in code_gen_dict.items():
            template = template.replace(f"${key_name}$", str(value))

        (code_gen_dir / f"{self.get_verilog_top_module_name()}.v").write_text(template)

        sv_files = ["fmpadding_axi.sv", "fmpadding.sv", "axi2we.sv"]
        for sv_file in sv_files:
            shutil.copy(rtlsrc / sv_file, code_gen_dir)
        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", str(code_gen_dir))
        self.set_nodeattr("ip_path", str(code_gen_dir))

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        """Return list of RTL files required for this node.

        The list is four files: fmpadding_axi.sv, fmpadding.sv, axi2we.sv and
        the generated .v file.
        """
        if abspath:
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen")) + "/"
            rtllib_dir = str(Path(get_settings().finn_rtllib) / "fmpadding/hdl") + "/"
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        gen_top_module = cast("str", self.get_nodeattr("gen_top_module"))
        return [
            rtllib_dir + "fmpadding_axi.sv",
            rtllib_dir + "fmpadding.sv",
            rtllib_dir + "axi2we.sv",
            code_gen_dir + gen_top_module + ".v",
        ]

    def code_generation_ipi(self) -> list[str]:
        """Construct and return the TCL for node instantiation in Vivado IPI."""
        code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))
        gen_top_module = cast("str", self.get_nodeattr("gen_top_module"))

        sourcefiles = [
            "fmpadding_axi.sv",
            "fmpadding.sv",
            "axi2we.sv",
            gen_top_module + ".v",
        ]
        sourcepaths = [str(code_gen_dir / f) for f in sourcefiles]

        cmd = [f"add_files -norecurse {f}" for f in sourcepaths]
        cmd += [f"create_bd_cell -type module -reference {gen_top_module} {self.onnx_node.name}"]
        return cmd

    def execute_node(self, context: dict[str, np.ndarray], graph: "GraphProto") -> None:
        """Execute this FMPadding node via C++ or RTL simulation."""
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            FMPadding.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)

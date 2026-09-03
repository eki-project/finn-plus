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

"""RTL backend implementation of the streaming FIFO (RTL / Vivado IP / virtual)."""

import numpy as np
import shutil
from onnx import GraphProto, NodeProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from typing import cast

from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.custom_op.fpgadataflow.streamingfifo import NodeAttrTypes, StreamingFIFO
from finn.util.exception import FINNInternalError
from finn.util.logging import log
from finn.util.settings import get_settings


@register_custom_op
class StreamingFIFO_rtl(StreamingFIFO, RTLBackend):
    """RTL implementation of a streaming FIFO for data buffering."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize the RTL streaming FIFO."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types, adding ``impl_style`` and ``fifo_id``."""
        my_attrs: NodeAttrTypes = {
            # Toggle between rtl or IPI implementation
            # rtl - use the rtl generated IP during stitching
            # vivado - use the AXI Infrastructure FIFO
            # virtual - use virtual rtl implementation for live fifo-sizing
            "impl_style": ("s", False, "rtl", {"rtl", "vivado", "virtual"}),
            # Unique FIFO ID for ring bus addressing (only for impl_style=virtual)
            "fifo_id": ("i", False, 0),
        }
        my_attrs.update(StreamingFIFO.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    @property
    def impl_style(self) -> str:
        """Get the implementation style (rtl/vivado/virtual)."""
        return cast("str", self.get_nodeattr("impl_style"))

    def get_adjusted_depth(self) -> int:
        """Return the FIFO depth, rounded up to a power of 2 for the vivado impl."""
        depth = self.depth
        if self.impl_style == "vivado":
            # round up depth to nearest power-of-2 (the Vivado FIFO impl may fail otherwise)
            adjusted = 1 << (depth - 1).bit_length()
            if adjusted != depth:
                log.warning(
                    f"{self.onnx_node.name}: rounding-up FIFO depth "
                    f"from {depth} to {adjusted} for impl_style=vivado"
                )
            return adjusted
        return depth

    def get_verilog_top_module_intf_names(self) -> dict[str, list[tuple[str, int]] | list[str]]:
        """Return the Verilog top-module interface names for this node."""
        ret = super().get_verilog_top_module_intf_names()
        if self.impl_style == "rtl" and self.depth_monitor == 1:
            ret["ap_none"] = ["maxcount"]
        if self.impl_style == "virtual":
            ret["ap_none"] = ["icfg", "ocfg"]
        return ret

    def is_sim_fifo_gauge(self) -> bool:
        """Return whether this FIFO uses the simulation-gauge implementation.

        RTL FIFOs with depth monitoring enabled use an infinite Verilog queue
        for simulation instead of ``Q_srl``.
        """
        return self.depth_monitor == 1 and self.impl_style == "rtl"

    def generate_hdl(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:  # noqa: ARG002
        """Generate HDL code from templates for this node."""
        code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))
        if self.impl_style == "virtual":
            # No HDL generation needed for virtual FIFOs
            self.set_nodeattr("ipgen_path", str(code_gen_dir))
            self.set_nodeattr("ip_path", str(code_gen_dir))
            return

        rtlsrc = Path(get_settings().finn_rtllib) / "fifo" / "hdl"
        template_path = rtlsrc / "fifo_template.v"

        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        topname = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", topname)

        # make instream width a multiple of 8 for the axi interface
        in_width = self.get_instream_width_padded()
        count_width = self.depth.bit_length()
        code_gen_dict = {
            "$TOP_MODULE_NAME$": topname,
            "$COUNT_WIDTH$": f"{count_width}",
            "$COUNT_RANGE$": f"[{count_width - 1}:0]",
            "$IN_RANGE$": f"[{in_width - 1}:0]",
            "$OUT_RANGE$": f"[{in_width - 1}:0]",
            "$WIDTH$": str(in_width),
            "$DEPTH$": str(self.depth),
        }

        template = template_path.read_text()
        for key, value in code_gen_dict.items():
            template = template.replace(key, str(value))
        (code_gen_dir / f"{topname}.v").write_text(template)

        shutil.copy(rtlsrc / "fifo_gauge.sv", code_gen_dir)
        shutil.copy(rtlsrc / "Q_srl.v", code_gen_dir)
        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", str(code_gen_dir))
        self.set_nodeattr("ip_path", str(code_gen_dir))

    def code_generation_ipi(self) -> list[str]:
        """Construct and return the TCL for node instantiation in Vivado IPI."""
        impl_style = self.impl_style
        node_name = self.onnx_node.name
        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))

        if impl_style == "rtl":
            top_module = cast("str", self.get_nodeattr("gen_top_module"))
            sourcefiles = [
                str(Path(code_gen_dir) / f) for f in ["fifo_gauge.sv", "Q_srl.v", f"{top_module}.v"]
            ]
            cmd = [f"add_files -norecurse {f}" for f in sourcefiles]
            cmd += [f"create_bd_cell -type module -reference {top_module} {node_name}"]
            return cmd

        if impl_style == "vivado":
            depth = self.get_adjusted_depth()
            ram_style = self.ram_style
            intf = self.get_verilog_top_module_intf_names()
            clk_name = intf["clk"][0]
            rst_name = intf["rst"][0]
            dout_name = intf["m_axis"][0][0]
            din_name = intf["s_axis"][0][0]
            tdata_num_bytes = int(np.ceil(self.get_outstream_width() / 8))
            return [
                f"create_bd_cell -type hier {node_name}",
                f"create_bd_pin -dir I -type clk /{node_name}/{clk_name}",
                f"create_bd_pin -dir I -type rst /{node_name}/{rst_name}",
                "create_bd_intf_pin -mode Master "
                f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/{dout_name}",
                "create_bd_intf_pin -mode Slave "
                f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/{din_name}",
                "create_bd_cell -type ip "
                f"-vlnv xilinx.com:ip:axis_data_fifo:2.0 /{node_name}/fifo",
                f"set_property -dict [list CONFIG.FIFO_DEPTH {{{depth}}}] "
                f"[get_bd_cells /{node_name}/fifo]",
                f"set_property -dict [list CONFIG.FIFO_MEMORY_TYPE {{{ram_style}}}] "
                f"[get_bd_cells /{node_name}/fifo]",
                f"set_property -dict [list CONFIG.TDATA_NUM_BYTES {{{tdata_num_bytes}}}] "
                f"[get_bd_cells /{node_name}/fifo]",
                f"connect_bd_intf_net [get_bd_intf_pins {node_name}/fifo/M_AXIS] "
                f"[get_bd_intf_pins {node_name}/{dout_name}]",
                f"connect_bd_intf_net [get_bd_intf_pins {node_name}/fifo/S_AXIS] "
                f"[get_bd_intf_pins {node_name}/{din_name}]",
                f"connect_bd_net [get_bd_pins {node_name}/{rst_name}] "
                f"[get_bd_pins {node_name}/fifo/s_axis_aresetn]",
                f"connect_bd_net [get_bd_pins {node_name}/{clk_name}] "
                f"[get_bd_pins {node_name}/fifo/s_axis_aclk]",
            ]

        if impl_style == "virtual":
            sourcefiles = self.get_rtl_file_list(abspath=True)
            fifo_id = self.get_nodeattr("fifo_id")
            width = int(self.get_instream_width_padded())
            fm_size = int(np.prod(self.get_folded_input_shape()[0:-1]))
            cmd = [f"add_files -norecurse {f}" for f in sourcefiles]
            cmd += [
                f"create_bd_cell -type module -reference fifo_gauge_wrapper {node_name}",
                f"set_property CONFIG.ID {fifo_id} [get_bd_cells {node_name}]",
                f"set_property CONFIG.DATA_WIDTH {width} [get_bd_cells {node_name}]",
                f"set_property CONFIG.FM_SIZE {fm_size} [get_bd_cells {node_name}]",
            ]
            return cmd

        raise FINNInternalError(
            f"{node_name}: FIFO implementation style {impl_style} not supported, "
            f"please use rtl or vivado"
        )

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        """Return the list of RTL files required for this node."""
        is_virtual = self.impl_style == "virtual"
        if abspath:
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen")) + "/"
            subdir = "fifo_virtual/hdl/" if is_virtual else "fifo/hdl/"
            rtllib_dir = str(Path(get_settings().finn_rtllib) / subdir.rstrip("/")) + "/"
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        if is_virtual:
            return [
                rtllib_dir + "fifo_gauge_pkg.sv",
                rtllib_dir + "fifo_gauge.sv",
                rtllib_dir + "fifo_gauge_wrapper.v",
            ]
        top_module = cast("str", self.get_nodeattr("gen_top_module"))
        return [
            rtllib_dir + "Q_srl.v",
            rtllib_dir + "fifo_gauge.sv",
            f"{code_gen_dir}{top_module}.v",
        ]

    def prepare_rtlsim(self, behav: bool = False) -> None:
        """Prepare this node for RTL simulation.

        Raises NotImplementedError if impl_style is not 'rtl'.
        """
        # TODO: Support simulation of vivado-style FIFOs,
        # or ensure node-by-node rtlsim is always skipped for FIFOs in general
        if self.impl_style != "rtl":
            log.warning(
                f"Trying to prepare rtlsim for {self.onnx_node.name}, but impl_style "
                "is set to vivado or virtual, which is not supported for simulation. Skipping. "
                "Simulation will fall back to Python simulation."
            )
            raise NotImplementedError
        return super().prepare_rtlsim(behav)

    def execute_node(self, context: dict[str, np.ndarray], graph: GraphProto) -> None:
        """Execute this FIFO node (Python no-op for cppsim/vivado/virtual, RTL sim otherwise)."""
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim" or self.impl_style in ("vivado", "virtual"):
            # Fall back to Python simulation (no-op) for vivado or virtual style FIFOs
            StreamingFIFO.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)

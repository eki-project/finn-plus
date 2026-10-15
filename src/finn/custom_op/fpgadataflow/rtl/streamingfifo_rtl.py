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
"""RTL backend implementation of the streaming FIFO.

This module provides an RTL-based implementation of streaming FIFOs for buffering
data between layers, backed by the shared ``finn-rtllib/fifo/hdl/fifo.sv`` module
(or the ``fifo_gauge`` behavioral model under ``FINN_SIMULATION``), as well as the
"virtual" FIFO gauge used for live (on-FPGA) FIFO sizing.
"""

import numpy as np
from onnx import GraphProto, NodeProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from typing import cast

from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.custom_op.fpgadataflow.streamingfifo import NodeAttrTypes, StreamingFIFO
from finn.util.basic import fifo_rtl_files
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log
from finn.util.settings import get_settings


@register_custom_op
class StreamingFIFO_rtl(StreamingFIFO, RTLBackend):
    """RTL implementation of a streaming FIFO for data buffering."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize the RTL streaming FIFO.

        Args:
            onnx_node: ONNX node to wrap.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types, adding ``impl_style`` and ``fifo_id``.

        ``impl_style`` chooses between the RTL FIFO and the virtual FIFO gauge used for
        live FIFO sizing.
        """
        my_attrs: NodeAttrTypes = {
            # Toggle between rtl or virtual implementation
            # rtl - use the rtl generated IP (fifo.sv) during stitching
            # virtual - use virtual rtl implementation for live fifo-sizing
            "impl_style": ("s", False, "rtl", {"rtl", "virtual"}),
            # Unique FIFO ID for ring bus addressing (only for impl_style=virtual)
            "fifo_id": ("i", False, 0),
        }
        my_attrs.update(StreamingFIFO.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    @property
    def impl_style(self) -> str:
        """Get the implementation style (rtl/virtual)."""
        return cast("str", self.get_nodeattr("impl_style"))

    def get_verilog_top_module_intf_names(self) -> dict[str, list[tuple[str, int]] | list[str]]:
        """Return the Verilog top-module interface names for this node.

        Includes the ring-bus config ports for virtual FIFOs and the optional maxcount
        output for depth monitoring.
        """
        ret = super().get_verilog_top_module_intf_names()
        if self.impl_style == "virtual":
            ret["ap_none"] = ["icfg", "ocfg"]
        elif self.depth_monitor == 1:
            ret["ap_none"] = ["maxcount"]
        return ret

    def is_sim_fifo_gauge(self) -> bool:
        """Return whether this FIFO uses the simulation-gauge implementation.

        RTL FIFOs with depth monitoring enabled use the ``fifo_gauge`` behavioral model
        (infinite queue) for simulation.
        """
        return self.depth_monitor == 1 and self.impl_style == "rtl"

    def generate_hdl(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:  # noqa: ARG002
        """Generate HDL code from templates for this node.

        Args:
            model: ONNX model wrapper.
            fpgapart: Target FPGA part number.
            clk: Target clock period in ns.
        """
        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
        if self.impl_style == "virtual":
            # No HDL generation needed for virtual FIFOs
            self.set_nodeattr("ipgen_path", code_gen_dir)
            self.set_nodeattr("ip_path", code_gen_dir)
            return

        rtlsrc = Path(get_settings().finn_rtllib) / "fifo" / "hdl"
        template_path = rtlsrc / "fifo_template.v"

        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        topname = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", topname)

        # make instream width a multiple of 8 for axi interface
        in_width = self.get_instream_width_padded()

        depth = int(self.depth)
        # fifo.sv will not elaborate below DEPTH 2; catch it here rather than in a
        # Vivado elaboration log
        if depth < 2:
            raise FINNInternalError(
                f"{self.onnx_node.name}: depth {depth} cannot be built, fifo.sv requires 2 or "
                "above. A FIFO this shallow should have been removed by RemoveShallowFIFOs."
            )
        ram_style = self.ram_style
        code_gen_dict = {
            "$TOP_MODULE_NAME$": topname,
            "$IN_RANGE$": f"[{in_width - 1}:0]",
            "$OUT_RANGE$": f"[{in_width - 1}:0]",
            "$WIDTH$": str(in_width),
            "$DEPTH$": str(depth),
            # fifo.sv's RAM_STYLE_EFF ladder still uses the legacy "shift" token for the
            # SRL backing; map the FINN-facing "srl" onto it at the RTL boundary
            "$RAM_STYLE$": "shift" if ram_style == "srl" else ram_style,
            "$DATA_LOGFILE$": cast("str", self.get_nodeattr("debug_log_path")),
        }
        # apply code generation to templates
        template = template_path.read_text()
        for key_name, value in code_gen_dict.items():
            template = template.replace(key_name, str(value))
        (Path(code_gen_dir) / f"{topname}.v").write_text(template)

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def code_generation_ipi(self) -> list[str]:
        """Generate the TCL commands for instantiating this IP in Vivado IPI.

        Returns:
            list[str]: TCL commands for IP instantiation.

        Raises:
            FINNUserError: If impl_style is neither rtl nor virtual.
        """
        impl_style = self.impl_style
        node_name = self.onnx_node.name
        if impl_style == "rtl":
            top_module = cast("str", self.get_nodeattr("gen_top_module"))
            cmd = [f"add_files -norecurse {f}" for f in self.get_rtl_file_list(abspath=True)]
            cmd += [f"create_bd_cell -type module -reference {top_module} {node_name}"]
            return cmd
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
        raise FINNUserError(
            f"FIFO implementation style {impl_style} not supported, please use rtl or virtual"
        )

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        """Get the list of RTL files required for this node.

        For impl_style=rtl this is the per-node wrapper written by generate_hdl() plus the
        shared FIFO sources (referenced in place so that the flat elaboration namespace only
        ever sees one declaration of module fifo).

        Args:
            abspath: If True, return absolute file paths; otherwise return bare file names.

        Returns:
            list[str]: The RTL file paths.
        """
        if self.impl_style == "virtual":
            rtllib_dir = (
                str(Path(get_settings().finn_rtllib) / "fifo_virtual" / "hdl") + "/"
                if abspath
                else ""
            )
            return [
                rtllib_dir + "fifo_gauge_pkg.sv",
                rtllib_dir + "fifo_gauge.sv",
                rtllib_dir + "fifo_gauge_wrapper.v",
            ]
        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen")) + "/" if abspath else ""
        top_module = cast("str", self.get_nodeattr("gen_top_module"))
        return [f"{code_gen_dir}{top_module}.v", *fifo_rtl_files(abspath, gauge=True)]

    def prepare_rtlsim(self, behav: bool = False) -> None:
        """Prepare this node for RTL simulation.

        Args:
            behav: Passed through to RTLBackend.prepare_rtlsim().

        Raises:
            NotImplementedError: If impl_style is 'virtual' (not supported for simulation).
        """
        if self.impl_style != "rtl":
            log.warning(
                f"Trying to prepare rtlsim for {self.onnx_node.name}, but impl_style "
                "is set to virtual, which is not supported for simulation. Skipping. "
                "Simulation will fall back to Python simulation."
            )
            raise NotImplementedError
        return super().prepare_rtlsim(behav)

    def execute_node(self, context: dict[str, np.ndarray], graph: GraphProto) -> None:
        """Execute this FIFO node.

        A FIFO only passes data through, so it is never simulated on its own: the
        passthrough is pinned ahead of RTLBackend's rtlsim implementation.

        Args:
            context: Dictionary mapping tensor names to numpy arrays.
            graph: ONNX graph containing this node.
        """
        StreamingFIFO.execute_node(self, context, graph)

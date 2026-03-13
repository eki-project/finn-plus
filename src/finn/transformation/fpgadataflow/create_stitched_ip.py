"""Create stitched IP from FINN dataflow graph.

This module provides transformations to create a Vivado IP Block Design project
from generated IPs in a FINN dataflow graph.
"""

# Copyright (c) 2020, Xilinx, Inc.
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

import json
import multiprocessing as mp
import os
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_num_default_workers
from shutil import copytree
from subprocess import CalledProcessError
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from onnx import NodeProto

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.templates import get_templates_folder
from finn.transformation.fpgadataflow.replace_verilog_relpaths import ReplaceVerilogRelPaths
from finn.util.basic import launch_process_helper, make_build_dir
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.util.logging import log


def is_external_input(model: ModelWrapper, node: "NodeProto", i: int) -> bool:
    """Check if input i of node should be made external.

    Returns True only if input is unconnected and has no initializer.
    Exception: second input of FC layers when mem_mode is external.
    """
    node_inst = getCustomOp(node)
    op_type = node.op_type
    producer = model.find_producer(node.input[i])
    if producer is None:
        if model.get_initializer(node.input[i]) is None:
            return True
        if op_type.startswith("MVAU") and node_inst.get_nodeattr("mem_mode") == "external":
            return True
    return False


def is_external_output(model: ModelWrapper, node: "NodeProto", i: int) -> bool:
    """Check if output i of node should be made external.

    Returns True only if output is unconnected.
    """
    # TODO should ideally check if tensor is in top-level outputs
    consumers = model.find_consumers(node.output[i])
    return consumers == []


class CreateStitchedIP(Transformation):
    """Create a Vivado IP Block Design project from all the generated IPs of a
    graph. All nodes in the graph must have the fpgadataflow backend attribute,
    and the PrepareIP transformation must have been previously run on
    the graph. The resulting block design is also packaged as IP. The
    transformation gets the fpgapart as a string.

    Outcome if successful: sets the vivado_stitch_proj attribute in the ONNX
    ModelProto's metadata_props field, with the created project dir as the
    value. A make_project.tcl script is also placed under the same folder,
    which is called to instantiate the per-layer IPs and stitch them together.
    The packaged block design IP can be found under the ip subdirectory.
    """

    def __init__(
        self,
        fpgapart: str,
        clk_ns: float,
        ip_name: str = "finn_design",
        vitis: bool = False,
        signature: list | None = None,
        functional_simulation: bool = False,
    ) -> None:
        """Initialize CreateStitchedIP transformation.

        Args:
            fpgapart: FPGA part identifier
            clk_ns: Clock period in nanoseconds
            ip_name: Name for the IP design
            vitis: Whether to target Vitis
            signature: Optional signature list [customer, application, version]
            functional_simulation: Whether to generate functional simulation wrapper
        """
        if signature is None:
            signature = []
        super().__init__()
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns
        self.ip_name = ip_name
        self.vitis = vitis
        self.signature = signature
        self.functional_simulation = functional_simulation
        self.has_aximm = False
        self.has_m_axis = False
        self.m_axis_idx = 0
        self.has_s_axis = False
        self.s_axis_idx = 0
        self.clock_reset_are_external = False
        self.clock2x_is_external = False
        self.create_cmds = []
        self.connect_cmds = []
        # keep track of top-level interface names
        self.intf_names = {
            "clk": [],
            "rst": [],
            "s_axis": [],
            "m_axis": [],
            "aximm": [],
            "axilite": [],
            "ap_none": [],
        }

    def is_double_pumped(self, node: "NodeProto") -> bool:
        """Check if node uses double-pumped compute or memory."""
        if node.op_type.startswith("MVAU"):
            inst = getCustomOp(node)
            try:
                pumped_compute = cast("int", inst.get_nodeattr("pumpedCompute"))
            except AttributeError:
                pumped_compute = 0
            return bool(pumped_compute or cast("int", inst.get_nodeattr("pumpedMemory")))
        return False

    def connect_clk_rst(self, node: "NodeProto") -> None:
        """Connect clock and reset signals for a node."""
        inst_name = node.name
        node_inst = getCustomOp(node)
        if not isinstance(node_inst, HWCustomOp):
            raise FINNInternalError(
                f"Node {node.name} is not an HWCustomOp, cannot connect AXI interfaces."
            )
        clock_intf_name = node_inst.get_verilog_top_module_intf_names()["clk"][0]
        reset_intf_name = node_inst.get_verilog_top_module_intf_names()["rst"][0]

        # make clock and reset external, if they aren't already
        if not self.clock_reset_are_external:
            self.connect_cmds.extend(
                [
                    f"make_bd_pins_external [get_bd_pins {inst_name}/{clock_intf_name}]",
                    "set_property name ap_clk [get_bd_ports ap_clk_0]",
                    f"make_bd_pins_external [get_bd_pins {inst_name}/{reset_intf_name}]",
                    "set_property name ap_rst_n [get_bd_ports ap_rst_n_0]",
                ]
            )
            self.clock_reset_are_external = True
            self.intf_names["clk"] = ["ap_clk"]
            self.intf_names["rst"] = ["ap_rst_n"]
        # otherwise connect clock and reset
        else:
            self.connect_cmds.extend(
                [
                    f"connect_bd_net [get_bd_ports ap_rst_n] "
                    f"[get_bd_pins {inst_name}/{reset_intf_name}]",
                    f"connect_bd_net [get_bd_ports ap_clk] "
                    f"[get_bd_pins {inst_name}/{clock_intf_name}]",
                ]
            )

        # make clk2x external, if it isn't already and connect clk2x
        if self.is_double_pumped(node):
            clock2x_intf_name = node_inst.get_verilog_top_module_intf_names()["clk2x"][0]
            if not self.clock2x_is_external:
                self.connect_cmds.extend(
                    [
                        f"make_bd_pins_external [get_bd_pins {inst_name}/{clock2x_intf_name}]",
                        "set_property name ap_clk2x [get_bd_ports ap_clk2x_0]",
                    ]
                )
                self.clock2x_is_external = True
                self.intf_names["clk2x"] = ["ap_clk2x"]
            # otherwise connect clk2x
            else:
                if self.is_double_pumped(node):
                    self.connect_cmds.append(
                        f"connect_bd_net [get_bd_ports ap_clk2x] "
                        f"[get_bd_pins {inst_name}/{clock2x_intf_name}]"
                    )

    def connect_axi(self, node: "NodeProto") -> None:
        """Connect AXI-Lite and AXI-MM interfaces for a node."""
        inst_name = node.name
        node_inst = getCustomOp(node)
        if not isinstance(node_inst, HWCustomOp):
            raise FINNInternalError(
                f"Node {node.name} is not an HWCustomOp, cannot connect AXI interfaces."
            )
        axilite_intf_name = node_inst.get_verilog_top_module_intf_names()["axilite"]
        aximm_intf_name = node_inst.get_verilog_top_module_intf_names()["aximm"]

        if len(axilite_intf_name) != 0:
            self.connect_cmds.append(
                f"make_bd_intf_pins_external [get_bd_intf_pins {inst_name}/{axilite_intf_name[0]}]"
            )
            ext_if_name = f"{axilite_intf_name[0]}_{len(self.intf_names['axilite'])}"
            self.intf_names["axilite"].append(ext_if_name)

        if len(aximm_intf_name) != 0:
            ext_if_name = f"m_axi_gmem{len(self.intf_names['aximm'])}"
            seg_name = f"{inst_name}/Data_m_axi_gmem/SEG_{ext_if_name}_Reg"

            self.connect_cmds.extend(
                [
                    f"make_bd_intf_pins_external "
                    f"[get_bd_intf_pins {inst_name}/{aximm_intf_name[0][0]}]",
                    f"set_property name {ext_if_name} [get_bd_intf_ports m_axi_gmem_0]",
                    "assign_bd_address",
                    f"set_property offset 0 [get_bd_addr_segs {{{seg_name}}}]",
                    f"set_property range 4G [get_bd_addr_segs {{{seg_name}}}]",
                ]
            )

            self.intf_names["aximm"] = [(ext_if_name, aximm_intf_name[0][1])]
            self.has_aximm = True

    def connect_m_axis_external(self, node: "NodeProto", idx: int | None = None) -> None:
        """Make AXI Stream master interface(s) external."""
        inst_name = node.name
        node_inst = getCustomOp(node)
        if not isinstance(node_inst, HWCustomOp):
            raise FINNInternalError(
                f"Node {node.name} is not an HWCustomOp, cannot connect AXI interfaces."
            )
        output_intf_names = node_inst.get_verilog_top_module_intf_names()["m_axis"]

        # make output axis external
        for i in range(len(output_intf_names)):
            if idx is not None and idx != i:
                continue
            output_intf_name = output_intf_names[i][0]

            self.connect_cmds.extend(
                [
                    f"make_bd_intf_pins_external [get_bd_intf_pins {inst_name}/{output_intf_name}]",
                    f"set_property name m_axis_{self.m_axis_idx} "
                    f"[get_bd_intf_ports {output_intf_name}_0]",
                ]
            )

            self.has_m_axis = True
            self.intf_names["m_axis"].append((f"m_axis_{self.m_axis_idx}", output_intf_names[i][1]))
            self.m_axis_idx += 1

    def connect_s_axis_external(self, node: "NodeProto", idx: int | None = None) -> None:
        """Make AXI Stream slave interface(s) external."""
        inst_name = node.name
        node_inst = getCustomOp(node)
        if not isinstance(node_inst, HWCustomOp):
            raise FINNInternalError(
                f"Node {node.name} is not an HWCustomOp, cannot connect AXI interfaces."
            )
        input_intf_names = node_inst.get_verilog_top_module_intf_names()["s_axis"]

        # make input axis external
        for i in range(len(input_intf_names)):
            if idx is not None and idx != i:
                continue
            input_intf_name = input_intf_names[i][0]

            self.connect_cmds.extend(
                [
                    f"make_bd_intf_pins_external [get_bd_intf_pins {inst_name}/{input_intf_name}]",
                    f"set_property name s_axis_{self.s_axis_idx} "
                    f"[get_bd_intf_ports {input_intf_name}_0]",
                ]
            )

            self.has_s_axis = True
            self.intf_names["s_axis"].append((f"s_axis_{self.s_axis_idx}", input_intf_names[i][1]))
            self.s_axis_idx += 1

    def connect_ap_none_external(self, node: "NodeProto") -> None:
        """Make ap_none interfaces external."""
        inst_name = node.name
        node_inst = getCustomOp(node)
        if not isinstance(node_inst, HWCustomOp):
            raise FINNInternalError(
                f"Node {node.name} is not an HWCustomOp, cannot connect AXI interfaces."
            )
        input_intf_names = node_inst.get_verilog_top_module_intf_names()["ap_none"]

        # make external
        for i in range(len(input_intf_names)):
            input_intf_name = input_intf_names[i]
            self.connect_cmds.extend(
                [
                    f"make_bd_pins_external [get_bd_pins {inst_name}/{input_intf_name}]",
                    f"set_property name {input_intf_name} [get_bd_ports {input_intf_name}_0]",
                ]
            )
            self.intf_names["ap_none"].append(input_intf_name)

    def insert_signature(self, checksum_count: int) -> None:
        """Insert AXI info signature component into the design."""
        signature_vlnv = "AMD:user:axi_info_top:1.0"
        signature_name = "axi_info_top0"
        fclk_mhz = 1 / (self.clk_ns * 0.001)
        fclk_hz = fclk_mhz * 1000000

        # Create signature cell and configure properties
        self.create_cmds.extend(
            [
                f"create_bd_cell -type ip -vlnv {signature_vlnv} {signature_name}",
                f"set_property -dict [list "
                f"CONFIG.SIG_CUSTOMER {{{self.signature[0]}}} "
                f"CONFIG.SIG_APPLICATION {{{self.signature[1]}}} "
                f"CONFIG.VERSION {{{self.signature[2]}}} "
                f"CONFIG.CHECKSUM_COUNT {{{checksum_count}}} "
                f"] [get_bd_cells {signature_name}]",
            ]
        )

        # Connect clocks, resets and configure AXI interface
        self.connect_cmds.extend(
            [
                f"connect_bd_net [get_bd_ports ap_clk] [get_bd_pins {signature_name}/ap_clk]",
                f"connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins {signature_name}/ap_rst_n]",
                f"set_property -dict [list "
                f"CONFIG.FREQ_HZ {{{fclk_hz}}} "
                f"CONFIG.CLK_DOMAIN {{ap_clk}} "
                f"] [get_bd_intf_pins {signature_name}/s_axi]",
                f"make_bd_intf_pins_external [get_bd_intf_pins {signature_name}/s_axi]",
                "set_property name s_axilite_info [get_bd_intf_ports s_axi_0]",
                "assign_bd_address",
            ]
        )

    def apply(self, model: "ModelWrapper") -> tuple[ModelWrapper, Literal[False]]:
        """Apply the CreateStitchedIP transformation to the model."""
        # ensure non-relative readmemh .dat files
        model = model.transform(ReplaceVerilogRelPaths())
        ip_dirs = ["list"]
        # add RTL streamer IP
        ip_dirs.append("$::env(FINN_RTLLIB)/memstream")
        if self.signature:
            ip_dirs.append("$::env(FINN_RTLLIB)/axi_info")
        if (
            model.graph.node[0].op_type not in ["StreamingFIFO_rtl", "IODMA_hls"]
            and self.functional_simulation is False
        ):
            log.warning(
                """First node is not StreamingFIFO or IODMA.
                You may experience incorrect stitched-IP rtlsim or hardware
                behavior. It is strongly recommended to insert FIFOs prior to
                calling CreateStitchedIP."""
            )
        if model.graph.node[0].op_type == "StreamingFIFO_rtl":
            firstfifo = getCustomOp(model.graph.node[0])
            if firstfifo.get_nodeattr("impl_style") == "vivado":
                log.warning(
                    """First FIFO has impl_style=vivado, which may cause
                    simulation glitches (e.g. dropping the first input sample
                    after reset)."""
                )
        for node in model.graph.node:
            # ensure that all nodes are fpgadataflow, and that IPs are generated
            if not is_hls_node(node) and not is_rtl_node(node):
                raise FINNUserError(
                    f"{node.name} is not an fpgadataflow node. Aborting stitching IP."
                )
            node_inst = getCustomOp(node)
            if not isinstance(node_inst, RTLBackend) and not isinstance(node_inst, HLSBackend):
                raise FINNInternalError(
                    f"Node {node.name} is not an RTL Node or HLS Node, "
                    "cannot connect AXI interfaces."
                )
            ip_dir_value = node_inst.get_nodeattr("ip_path")
            if type(ip_dir_value) is not str or ip_dir_value == "":
                raise FINNInternalError(f"ip_path has the wrong type in node {node.name}.")
            if not Path(ip_dir_value).is_dir():
                raise FINNInternalError(
                    f"IP generation directory doesn't exist in node {node.name}."
                )
            ip_dirs += [ip_dir_value]
            self.create_cmds += node_inst.code_generation_ipi()
            self.connect_clk_rst(node)
            self.connect_ap_none_external(node)
            self.connect_axi(node)
            for i in range(len(node.input)):
                if not is_external_input(model, node, i):
                    producer = model.find_producer(node.input[i])
                    if producer is None:
                        continue
                    j = list(producer.output).index(node.input[i])
                    prod = getCustomOp(producer)
                    if not isinstance(prod, HWCustomOp):
                        raise FINNInternalError(
                            f"Producer node {producer.name} is not an HWCustomOp, "
                            "cannot connect AXI interfaces."
                        )
                    src_intf_name = prod.get_verilog_top_module_intf_names()["m_axis"][j][0]
                    dst_intf_name = node_inst.get_verilog_top_module_intf_names()["s_axis"][i][0]
                    self.connect_cmds.append(
                        f"connect_bd_intf_net [get_bd_intf_pins {producer.name}/{src_intf_name}] "
                        f"[get_bd_intf_pins {node.name}/{dst_intf_name}]"
                    )

        # process external inputs and outputs in top-level graph input order
        for graph_input in model.graph.input:
            inp_name = graph_input.name
            inp_cons = model.find_consumers(inp_name)
            assert inp_cons != [], f"No consumer for input {inp_name}"
            assert len(inp_cons) == 1, f"Multiple consumers for input {inp_name}"
            node = inp_cons[0]
            node_inst = getCustomOp(node)
            for i in range(len(node.input)):
                if node.input[i] == inp_name:
                    self.connect_s_axis_external(node, idx=i)
        for output in model.graph.output:
            out_name = output.name
            node = model.find_producer(out_name)
            assert node is not None, f"No producer for output {out_name}"
            node_inst = getCustomOp(node)
            for i in range(len(node.output)):
                if node.output[i] == out_name:
                    self.connect_m_axis_external(node, idx=i)

        if self.signature:
            # extract number of checksum layer from graph
            checksum_layers = model.get_nodes_by_op_type("CheckSum_hls")
            self.insert_signature(len(checksum_layers))

        # create a temporary folder for the project
        prjname = "finn_vivado_stitch_proj"
        vivado_stitch_proj_dir = make_build_dir(prefix="vivado_stitch_proj_")
        model.set_metadata_prop("vivado_stitch_proj", vivado_stitch_proj_dir)
        # start building the tcl script
        tcl = []

        # Project setup
        ip_dirs_str = " ".join(ip_dirs)
        block_name = self.ip_name

        tcl.extend(
            [
                f"create_project {prjname} {vivado_stitch_proj_dir} -part {self.fpgapart}",
                "set_msg_config -id {[BD 41-1753]} -suppress",
                f"set_property ip_repo_paths [{ip_dirs_str}] [current_project]",
                "update_ip_catalog",
                f'create_bd_design "{block_name}"',
            ]
        )
        # Add commands and validate design
        tcl.extend(self.create_cmds)
        tcl.extend(self.connect_cmds)

        fclk_mhz = 1 / (self.clk_ns * 0.001)
        fclk_hz = fclk_mhz * 1000000

        # Configure clocks and validate design
        clock_config = [f"set_property CONFIG.FREQ_HZ {round(fclk_hz)} [get_bd_ports /ap_clk]"]
        if self.clock2x_is_external:
            clock_config.append(
                f"set_property CONFIG.FREQ_HZ {round(2 * fclk_hz)} [get_bd_ports /ap_clk2x]"
            )

        clock_config.extend(["validate_bd_design", "save_bd_design"])

        tcl.extend(clock_config)

        # Create wrapper HDL
        bd_base = f"{vivado_stitch_proj_dir}/{prjname}.srcs/sources_1/bd/{block_name}"
        bd_filename = f"{bd_base}/{block_name}.bd"
        wrapper_filename = f"{bd_base}/hdl/{block_name}_wrapper.v"

        tcl.extend(
            [
                f"make_wrapper -files [get_files {bd_filename}] -top",
                f"add_files -norecurse {wrapper_filename}",
                f"set_property top {block_name}_wrapper [current_fileset]",
            ]
        )

        model.set_metadata_prop("wrapper_filename", wrapper_filename)
        num_workers = get_num_default_workers()
        assert num_workers >= 0, "Number of workers must be nonnegative."
        if num_workers == 0:
            num_workers = mp.cpu_count()

        fifosim_wrapper_filename = None
        if self.functional_simulation:
            bd_base_sim = f"{vivado_stitch_proj_dir}/{prjname}.sim/sim_1/synth/func/xsim/"
            fifosim_wrapper_filename = f"{bd_base_sim}/fifosim_wrapper_func_synth.v"

            tcl.extend(
                [
                    f"launch_runs synth_1 -jobs {num_workers}",
                    "wait_on_run [get_runs synth_1]",
                    "open_run synth_1 -name synth_1",
                    "opt_design",
                    # "opt_design -muxf_remap -carry_remap -control_set_merge "
                    # "-merge_equivalent_drivers -mbufg_opt -dsp_register_opt "
                    # "-control_set_opt -remap -resynth_area -resynth_remap",
                    # "opt_design",
                    f"write_verilog -mode funcsim -force -file {fifosim_wrapper_filename}",
                ]
            )

            model.set_metadata_prop("wrapper_filename", fifosim_wrapper_filename)
        # Synthesize to DCP and export stub, DCP and constraints
        if self.vitis:
            tcl.extend(
                [
                    f"set_property SYNTH_CHECKPOINT_MODE Hierarchical [ get_files {bd_filename} ]",
                    "set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} "
                    "-value {-mode out_of_context} -objects [get_runs synth_1]",
                    f"launch_runs synth_1 -jobs {num_workers}",
                    "wait_on_run [get_runs synth_1]",
                    "open_run synth_1 -name synth_1",
                    f"write_verilog -force -mode synth_stub {block_name}.v",
                    f"write_checkpoint {block_name}.dcp",
                    f"write_xdc {block_name}.xdc",
                    f"report_utilization -hierarchical -hierarchical_depth 5 "
                    f"-file {block_name}_partition_util.rpt",
                ]
            )
        # Export block design itself as an IP core
        block_vendor = "xilinx_finn"
        block_library = "finn"
        block_vlnv = f"{block_vendor}:{block_library}:{block_name}:1.0"
        model.set_metadata_prop("vivado_stitch_vlnv", block_vlnv)
        model.set_metadata_prop("vivado_stitch_ifnames", json.dumps(self.intf_names))

        # Package IP and configure properties
        tcl.extend(
            [
                f"ipx::package_project -root_dir {vivado_stitch_proj_dir}/ip "
                f"-vendor {block_vendor} -library {block_library} -taxonomy /UserIP "
                f"-module {block_name} -import_files",
                "set_property ipi_drc {ignore_freq_hz true} [ipx::current_core]",
                "ipx::remove_segment -quiet m_axi_gmem0:APERTURE_0 "
                "[ipx::get_address_spaces m_axi_gmem0 -of_objects [ipx::current_core]]",
                f"set_property core_revision 2 [ipx::find_open_core {block_vlnv}]",
                f"ipx::create_xgui_files [ipx::find_open_core {block_vlnv}]",
                "set_property value_resolve_type user [ipx::get_bus_parameters "
                "-of [ipx::get_bus_interfaces -of [ipx::current_core ]]]",
            ]
        )
        # If targeting Vitis, add some properties to the IP
        if self.vitis:
            # Configure Vitis kernel properties
            tcl.extend(
                [
                    f"set_property sdx_kernel true [ipx::find_open_core {block_vlnv}]",
                    f"set_property sdx_kernel_type rtl [ipx::find_open_core {block_vlnv}]",
                    f"set_property supported_families {{}} [ipx::find_open_core {block_vlnv}]",
                    f"set_property xpm_libraries {{XPM_CDC XPM_MEMORY XPM_FIFO}} "
                    f"[ipx::find_open_core {block_vlnv}]",
                    f"set_property auto_family_support_level level_2 "
                    f"[ipx::find_open_core {block_vlnv}]",
                ]
            )

            # Remove all files from synthesis and sim groups and replace with DCP
            tcl.extend(
                [
                    "ipx::remove_all_file "
                    "[ipx::get_file_groups xilinx_anylanguagebehavioralsimulation]",
                    "ipx::remove_all_file [ipx::get_file_groups xilinx_anylanguagesynthesis]",
                    "ipx::remove_file_group "
                    "xilinx_anylanguagebehavioralsimulation [ipx::current_core]",
                    "ipx::remove_file_group xilinx_anylanguagesynthesis [ipx::current_core]",
                ]
            )

            # Setup file structure for DCP-based IP
            tcl.extend(
                [
                    f"file delete -force {vivado_stitch_proj_dir}/ip/sim",
                    f"file delete -force {vivado_stitch_proj_dir}/ip/src",
                    f"file mkdir {vivado_stitch_proj_dir}/ip/dcp",
                    f"file mkdir {vivado_stitch_proj_dir}/ip/impl",
                    f"file copy -force {block_name}.dcp {vivado_stitch_proj_dir}/ip/dcp",
                    f"file copy -force {block_name}.xdc {vivado_stitch_proj_dir}/ip/impl",
                ]
            )

            # Add implementation and checkpoint file groups
            tcl.extend(
                [
                    "ipx::add_file_group xilinx_implementation [ipx::current_core]",
                    f"ipx::add_file impl/{block_name}.xdc "
                    "[ipx::get_file_groups xilinx_implementation]",
                    f"set_property used_in [list implementation] "
                    f"[ipx::get_files impl/{block_name}.xdc "
                    f"-of_objects [ipx::get_file_groups xilinx_implementation]]",
                    "ipx::add_file_group xilinx_synthesischeckpoint [ipx::current_core]",
                    f"ipx::add_file dcp/{block_name}.dcp "
                    f"[ipx::get_file_groups xilinx_synthesischeckpoint]",
                    "ipx::add_file_group xilinx_simulationcheckpoint [ipx::current_core]",
                    f"ipx::add_file dcp/{block_name}.dcp "
                    f"[ipx::get_file_groups xilinx_simulationcheckpoint]",
                ]
            )
        # add a rudimentary driver mdd to get correct ranges in xparameters.h later on
        min_driver = get_templates_folder() / "ipcore_driver"
        copytree(min_driver, vivado_stitch_proj_dir + "/data")

        #####
        # Core Cleanup Operations
        tcl.append(
            """
set core [ipx::current_core]

# Add rudimentary driver
file copy -force data ip/
set file_group [ipx::add_file_group -type software_driver {} $core]
set_property type mdd       [ipx::add_file data/finn_design.mdd $file_group]
set_property type tclSource [ipx::add_file data/finn_design.tcl $file_group]

# Remove all XCI references to subcores
set impl_files [ipx::get_file_groups xilinx_implementation -of $core]
foreach xci [ipx::get_files -of $impl_files {*.xci}] {
    ipx::remove_file [get_property NAME $xci] $impl_files
}

# Construct a single flat memory map for each AXI-lite interface port
foreach port [get_bd_intf_ports -filter {CONFIG.PROTOCOL==AXI4LITE}] {
    set pin $port
    set awidth ""
    while { $awidth == "" } {
        set pins [get_bd_intf_pins -of [get_bd_intf_nets -boundary_type lower -of $pin]]
        set kill [lsearch $pins $pin]
        if { $kill >= 0 } { set pins [lreplace $pins $kill $kill] }
        if { [llength $pins] != 1 } { break }
        set pin [lindex $pins 0]
        set awidth [get_property CONFIG.ADDR_WIDTH $pin]
    }
    if { $awidth == "" } {
       puts "CRITICAL WARNING: Unable to construct address map for $port."
    } {
       set range [expr 2**$awidth]
       set range [expr $range < 4096 ? 4096 : $range]
       puts "INFO: Building address map for $port: 0+:$range"
       set name [get_property NAME $port]
       set addr_block [ipx::add_address_block Reg0 [ipx::add_memory_map $name $core]]
       set_property range $range $addr_block
       set_property slave_memory_map_ref $name [ipx::get_bus_interfaces $name -of $core]
    }
}

# Finalize and Save
ipx::update_checksums $core
ipx::save_core $core

# Remove stale subcore references from component.xml
file rename -force ip/component.xml ip/component.bak
set ifile [open ip/component.bak r]
set ofile [open ip/component.xml w]
set buf [list]
set kill 0
while { [eof $ifile] != 1 } {
    gets $ifile line
    if { [string match {*<spirit:fileSet>*} $line] == 1 } {
        foreach l $buf { puts $ofile $l }
        set buf [list $line]
    } elseif { [llength $buf] > 0 } {
        lappend buf $line

        if { [string match {*</spirit:fileSet>*} $line] == 1 } {
            if { $kill == 0 } { foreach l $buf { puts $ofile $l } }
            set buf [list]
            set kill 0
        } elseif { [string match {*<xilinx:subCoreRef>*} $line] == 1 } {
            set kill 1
        }
    } else {
        puts $ofile $line
    }
}
close $ifile
close $ofile
"""
        )

        # Export list of used Verilog files (for rtlsim later on)
        v_file_list = f"{vivado_stitch_proj_dir}/all_verilog_srcs.txt"
        tcl.extend(
            [
                "set all_v_files [get_files -filter {USED_IN_SYNTHESIS == 1 "
                "&& (FILE_TYPE == Verilog || FILE_TYPE == SystemVerilog "
                '|| FILE_TYPE =="Verilog Header")}]',
                f"set fp [open {v_file_list} w]",
                "foreach vf $all_v_files {puts $fp $vf}",
                "close $fp",
            ]
        )
        # write the project creator tcl script
        tcl_string = "\n".join(tcl) + "\n"
        with Path(f"{vivado_stitch_proj_dir}/make_project.tcl").open("w") as f:
            f.write(tcl_string)
        # create a shell script and call Vivado
        make_project_sh = f"{vivado_stitch_proj_dir}/make_project.sh"
        working_dir = Path.cwd()
        with Path(make_project_sh).open("w") as f:
            f.write("#!/bin/bash \n")
            f.write(f"cd {vivado_stitch_proj_dir}\n")
            f.write("vivado -mode batch -source make_project.tcl\n")
            f.write(f"cd {working_dir}\n")
        bash_command = ["bash", make_project_sh]

        try:
            launch_process_helper(bash_command, print_stdout=False)
        except CalledProcessError as e:
            raise FINNUserError(
                f"CreateStitchedIP: make_project.sh failed with a non-zero "
                f"exit code. Check previous logs and logs in "
                f"{vivado_stitch_proj_dir} to find out why it failed."
            ) from e

        if self.functional_simulation:
            with Path(v_file_list).open("a") as f:
                f.write(f"{fifosim_wrapper_filename}\n")

        # wrapper may be created in different location depending on Vivado version
        if not Path(wrapper_filename).is_file():
            # check in alternative location (.gen instead of .srcs)
            wrapper_filename_alt = wrapper_filename.replace(".srcs", ".gen")
            if Path(wrapper_filename_alt).is_file():
                if not self.functional_simulation:
                    model.set_metadata_prop("wrapper_filename", wrapper_filename_alt)
            else:
                raise FINNUserError(
                    f"""CreateStitchedIP failed, no wrapper HDL found \
                        under {wrapper_filename} or {wrapper_filename_alt}.
                    Please check logs under the parent directory."""
                )

        return (model, False)

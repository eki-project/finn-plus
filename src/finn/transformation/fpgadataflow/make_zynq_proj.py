# Copyright (C) 2020, Xilinx, Inc.
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

"""Transformation to create Zynq Vivado projects for FINN dataflow designs."""

import json
import math
import multiprocessing as mp
from onnx import NodeProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.util.basic import get_num_default_workers
from shutil import copy
from subprocess import CalledProcessError
from typing import Literal

from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.floorplan import Floorplan
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.instrumentation import GenerateInstrumentationIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import (
    launch_process_helper,
    make_build_dir,
    pynq_native_port_width,
    pynq_part_map,
)
from finn.util.exception import FINNError, FINNInternalError, FINNSynthesisError, FINNUserError
from finn.util.logging import log
from finn.util.settings import get_settings

from . import templates


def _build_sdp_kernel(args: tuple) -> tuple:
    """Worker function for parallel SDP kernel builds.

    Runs InsertFIFO (if needed), SpecializeLayers, GiveUniqueNodeNames,
    PrepareIP, HLSSynthIP, and CreateStitchedIP for a single
    StreamingDataflowPartition kernel. Saves the result back to disk.

    Args:
        args: tuple of (sdp_node_name, dataflow_model_filename, fpga_part,
              period_ns, enable_instrumentation)
    """
    sdp_node_name, dataflow_model_filename, fpga_part, period_ns, enable_instrumentation = args
    prefix = sdp_node_name + "_"
    kernel_model = ModelWrapper(dataflow_model_filename)

    if not kernel_model.get_nodes_by_op_type("IODMA_hls"):
        del kernel_model.model.graph.metadata_props[:]
        kernel_model.save(dataflow_model_filename)

    prcont = [
        n
        for n in kernel_model.graph.node
        if n.op_type == "NodeContainer"
        and getCustomOp(n).get_nodeattr("multi_dnn_type") == "partial_reconfiguration"
    ]
    if prcont:
        if not (
            kernel_model.graph.node[0].op_type == "NodeContainer"
            and getCustomOp(kernel_model.graph.node[0]).get_nodeattr("multi_dnn_type")
            == "partial_reconfiguration"
        ):
            raise FINNInternalError(
                "Expected NodeContainer in SDP when using partial reconfiguration"
            )
        if len(kernel_model.graph.node) != 1:
            raise FINNInternalError(
                "Only one NodeContainer per SDP when using partial reconfiguration"
            )
        pr_container_inst = getCustomOp(prcont[0])
        for body_idx in range(pr_container_inst.get_nodeattr("bodies")):
            body_model = pr_container_inst.get_nodeattr("body_" + str(body_idx))

            if not enable_instrumentation:
                body_model = body_model.transform(InsertFIFO())
            body_model = body_model.transform(SpecializeLayers(fpga_part))

            body_model.save(dataflow_model_filename)
            body_model = body_model.transform(PrepareIP(fpga_part, period_ns))
            # Do not try to parallelize HLSSynthIP here (mp.pool within an mp.pool not allowed)
            body_model = body_model.transform(HLSSynthIP(num_workers=1))
            body_model = body_model.transform(
                CreateStitchedIP(
                    fpga_part,
                    period_ns,
                    f"sdp_{pr_container_inst.onnx_node.name}_{body_idx}",
                    vitis=False,
                )
            )
            body_model.set_metadata_prop("platform", "zynq-iodma")
            pr_container_inst.set_nodeattr("body_" + str(body_idx), body_model)
            body_model.save(dataflow_model_filename)

        kernel_model.set_metadata_prop("platform", "zynq-iodma")
        kernel_model.save(dataflow_model_filename)
        return

    # InsertFIFO at this stage interferes with tLastMarker
    # TODO: is this really needed here at all?
    if not enable_instrumentation:
        kernel_model = kernel_model.transform(InsertFIFO())
    kernel_model = kernel_model.transform(SpecializeLayers(fpga_part))
    if not kernel_model.get_nodes_by_op_type("NodeContainer"):
        kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
    kernel_model.save(dataflow_model_filename)
    kernel_model = kernel_model.transform(PrepareIP(fpga_part, period_ns))
    # Do not try to parallelize HLSSynthIP here (mp.pool within an mp.pool not allowed)
    kernel_model = kernel_model.transform(HLSSynthIP(num_workers=1))
    kernel_model = kernel_model.transform(
        CreateStitchedIP(fpga_part, period_ns, sdp_node_name, False)
    )
    kernel_model.set_metadata_prop("platform", "zynq-iodma")
    kernel_model.save(dataflow_model_filename)


def _require_ip_dir(node: NodeProto, ip_dir_value: str) -> None:
    """Raise if the directory holding a node's generated IP blocks does not exist."""
    if not Path(ip_dir_value).is_dir():
        raise FINNInternalError(
            f"{node.name}: the directory that should contain the generated ip blocks "
            f"doesn't exist: {ip_dir_value}"
        )


def collect_ip_dirs(model: ModelWrapper, ipstitch_path: str | None) -> list[str]:
    """Collect list of all IP directories required by the design."""
    ip_dirs = []
    need_memstreamer = False
    for node in model.graph.node:
        node_inst = getCustomOp(node)
        if node.op_type == "NodeContainer":
            if node_inst.get_nodeattr("multi_dnn_type") == "partial_reconfiguration":
                for body_idx in range(node_inst.get_nodeattr("bodies")):
                    body_model = node_inst.get_nodeattr("body_" + str(body_idx))
                    a = collect_ip_dirs(body_model, None)
                    ip_dirs += a
            else:
                code_gen_dir = node_inst.get_nodeattr("code_gen_dir_ipgen")
                if code_gen_dir and Path(code_gen_dir).is_dir():
                    ip_dirs.append(code_gen_dir)
                ip_dir_value = node_inst.get_nodeattr("ip_path")
                _require_ip_dir(node, ip_dir_value)
                ip_dirs += [ip_dir_value]
        else:
            ip_dir_value = node_inst.get_nodeattr("ip_path")
            _require_ip_dir(node, ip_dir_value)
            ip_dirs += [ip_dir_value]
        if (
            node.op_type.startswith("MVAU") or node.op_type == "Thresholding_hls"
        ) and node_inst.get_nodeattr("mem_mode") == "internal_decoupled":
            need_memstreamer = True
    ip_dirs += [ipstitch_path + "/ip"] if ipstitch_path else []
    if need_memstreamer:
        # add RTL streamer IP
        ip_dirs.append("$::env(FINN_RTLLIB)/memstream")
    return ip_dirs


class MakeZYNQProject(Transformation):
    """Create a Vivado overlay project (including the shell infrastructure)
    from the already-stitched IP block for this graph.
    All nodes in the graph must have the fpgadataflow backend attribute,
    and the CreateStitchedIP transformation must have been previously run on
    the graph. This is functionally equivalent with MakePYNQProject but does
    not use Pynq infrastructure and instead creates a fully custom block design.
    However, this transform requires DMAs in the accelerator design.

    Outcome if successful: sets the vivado_pynq_proj attribute in the ONNX
    ModelProto's metadata_props field, with the created project dir as the
    value.
    """

    def __init__(
        self,
        platform: str,
        period_ns: float,
        enable_debug: bool = False,
        enable_finn_switch: bool = False,
        live_fifo_sizing: bool = False,
    ) -> None:
        """Initialize MakeZYNQProject with the target platform and clock period."""
        super().__init__()
        self.platform = platform
        self.period_ns = period_ns
        self.enable_finn_switch = enable_finn_switch
        self.live_fifo_sizing = live_fifo_sizing
        self.enable_debug = 1 if enable_debug else 0
        self.enable_gpio_reset = 0

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transformation to create a Zynq project."""
        config = []
        idma_idx = 0
        odma_idx = 0
        aximm_idx = 0
        nested_interconnect_count = 0
        master_axilite_idx = 0
        axilite_interconnect_idx = 0
        axilite_idx = 0
        instance_names = {}

        sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
        partial_reconfiguration = False
        for sdp_node in sdp_nodes:
            sdp_node = getCustomOp(sdp_node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            if any(
                n.op_type == "NodeContainer"
                and getCustomOp(n).get_nodeattr("multi_dnn_type") == "partial_reconfiguration"
                for n in kernel_model.graph.node
            ):
                partial_reconfiguration = True
                # Copy body_0 metadata to the SDP node
                pr_node = kernel_model.get_nodes_by_op_type("NodeContainer")[
                    0
                ]  # We can assume that we have only one NodeContainer
                pr_node_inst = getCustomOp(pr_node)
                body_model = pr_node_inst.get_nodeattr("body_0")
                kernel_model.set_metadata_prop(
                    "vivado_stitch_proj",
                    body_model.get_metadata_prop("vivado_stitch_proj"),
                )
                kernel_model.set_metadata_prop(
                    "wrapper_filename", body_model.get_metadata_prop("wrapper_filename")
                )
                kernel_model.set_metadata_prop(
                    "vivado_stitch_vlnv",
                    body_model.get_metadata_prop("vivado_stitch_vlnv"),
                )
                kernel_model.set_metadata_prop(
                    "vivado_stitch_ifnames",
                    body_model.get_metadata_prop("vivado_stitch_ifnames"),
                )
                kernel_model.save(dataflow_model_filename)

        sw_nodes = [
            getCustomOp(n)
            for sdp in sdp_nodes
            for n in ModelWrapper(getCustomOp(sdp).get_nodeattr("model")).graph.node
            if n.op_type == "NodeContainer"
            and getCustomOp(n).get_nodeattr("multi_dnn_type") == "selectable_weights"
        ]

        # instantiate instrumentation IP if it was generated
        instr_ip_dir = model.get_metadata_prop("instrumentation_ipgen")

        if self.enable_finn_switch:
            # TODO: Add -copy_to
            module_dir = Path(get_settings().finn_rtllib) / "finn_switch" / "hdl" / "switch.v"
            config.append(
                "add_files -copy_to [get_property DIRECTORY [current_project]] -norecurse "
                f"{module_dir}"
            )
            config.append("create_bd_cell -type module -reference finn_switch finn_switch")

        use_instrumentation = instr_ip_dir is not None and Path(instr_ip_dir).is_dir()
        if use_instrumentation:
            # instantiate GPIO IP to trigger reset
            self.enable_gpio_reset = 1
            # in the template this will connect to first port of interconnect_0
            master_axilite_idx += 1

            # update IP repository
            config.append(
                "set_property ip_repo_paths "
                f"[concat [get_property ip_repo_paths [current_project]] [list {instr_ip_dir}]] "
                "[current_project]"
            )
            config.append("update_ip_catalog -rebuild -scan_changes")
            # create instance
            config.append(
                "create_bd_cell -type ip -vlnv {} {}".format(
                    "xilinx.com:hls:instrumentation_wrapper:1.0",
                    "instrumentation_wrap_0",
                )
            )
            # connect clock % reset
            config.append(
                "connect_bd_net [get_bd_pins instrumentation_wrap_0/ap_clk] "
                "[get_bd_pins smartconnect_0/aclk]"
            )
            config.append(
                "connect_bd_net [get_bd_pins instrumentation_wrap_0/ap_rst_n] "
                "[get_bd_pins smartconnect_0/aresetn]"
            )
            # connect AXI-lite control interface
            config.append(
                "connect_bd_intf_net [get_bd_intf_pins instrumentation_wrap_0/s_axi_ctrl] "
                f"[get_bd_intf_pins axi_interconnect_0/M{master_axilite_idx:02d}_AXI]"
            )
            config.append("assign_axi_addr_proc instrumentation_wrap_0/s_axi_ctrl")
            master_axilite_idx += 1

        if self.live_fifo_sizing:
            # instantiate virtual FIFO controller
            rtl_path = get_settings().finn_rtllib
            files = [
                Path(rtl_path) / "axi/hdl/axilite.sv",
                Path(rtl_path) / "fifo_virtual/hdl/fifo_gauge_pkg.sv",
                Path(rtl_path) / "fifo_virtual/hdl/fifo_controller.sv",
                Path(rtl_path) / "fifo_virtual/hdl/fifo_controller_wrapper.v",
            ]
            for f in files:
                config.append(f"add_files -norecurse {f}")
            config.append(
                "create_bd_cell -type module -reference fifo_controller_wrapper fifo_controller_0"
            )

            # connect clock & reset
            config.append(
                "connect_bd_net [get_bd_pins fifo_controller_0/ap_clk] "
                "[get_bd_pins smartconnect_0/aclk]"
            )
            config.append(
                "connect_bd_net [get_bd_pins fifo_controller_0/ap_rst_n] "
                "[get_bd_pins smartconnect_0/aresetn]"
            )

            # connect AXI-lite control interface
            config.append(
                "connect_bd_intf_net [get_bd_intf_pins fifo_controller_0/s_axi] "
                f"[get_bd_intf_pins axi_interconnect_0/M{master_axilite_idx:02d}_AXI]"
            )
            # Do not use assign_axi_addr_proc here. It doesn't map the 32-bit aperture correctly.
            # Instead, let assign_bd_address command assign the address later.
            # TODO: Support 32-bit systems by making aperture smaller?
            # config.append("assign_axi_addr_proc fifo_controller_0/s_axi")
            master_axilite_idx += 1

        # instantiate nested AXI interconnects if required
        # only the nested interconnects and all interfaces connected before this line
        # will be connected to the original (master) interconnect
        total_axilite_count = 0
        for node in model.graph.node:
            sdp_node = getCustomOp(node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            ifnames = eval(kernel_model.get_metadata_prop("vivado_stitch_ifnames"))
            total_axilite_count += len(ifnames["axilite"])
        if total_axilite_count > (64 - master_axilite_idx):
            nested_interconnect_count = math.ceil(total_axilite_count / 64.0)
            for i in range(1, nested_interconnect_count + 1):
                # create instance
                config.append(
                    f"create_bd_cell -type ip -vlnv $interconnect_vlnv axi_interconnect_{i}"
                )
                # configure instance
                config.append(
                    f"set_property -dict [list CONFIG.NUM_MI {min(64, total_axilite_count)}] "
                    f"[get_bd_cells axi_interconnect_{i}]"
                )
                # connect to master interconnect
                config.append(
                    "connect_bd_intf_net [get_bd_intf_pins "
                    f"axi_interconnect_0/M{master_axilite_idx:02d}_AXI] "
                    f"-boundary_type upper [get_bd_intf_pins axi_interconnect_{i}/S00_AXI]"
                )
                # connect clocks/reset
                config.append(
                    "connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] "
                    f"[get_bd_pins axi_interconnect_{i}/ACLK]"
                )
                config.append(
                    "connect_bd_net [get_bd_pins proc_sys_reset_0/interconnect_aresetn] "
                    f"[get_bd_pins axi_interconnect_{i}/ARESETN]"
                )
                master_axilite_idx += 1
                total_axilite_count = max(0, total_axilite_count - 64)

            if total_axilite_count != 0:
                raise FINNInternalError(
                    f"Not all AXI-lite interfaces connected! ({total_axilite_count} left)"
                )

            # start populating the first nested interconnect
            axilite_interconnect_idx = 1
        else:
            axilite_idx = master_axilite_idx

        num_sdps = len(model.graph.node)
        prev_node_name = None
        for node in model.graph.node:
            if node.op_type != "StreamingDataflowPartition":
                raise FINNInternalError(f"Invalid link graph: unexpected node {node.op_type}")
            sdp_node = getCustomOp(node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            sdp_id = int(node.name.split("_")[-1])

            ipstitch_path = kernel_model.get_metadata_prop("vivado_stitch_proj")
            if ipstitch_path is None or (not Path(ipstitch_path).is_dir()):
                raise FINNInternalError(
                    f"No stitched IPI design found for {node.name}, apply CreateStitchedIP first."
                )

            vivado_stitch_vlnv = kernel_model.get_metadata_prop("vivado_stitch_vlnv")
            if vivado_stitch_vlnv is None:
                raise FINNInternalError(
                    f"No vlnv found for {node.name}, apply CreateStitchedIP first."
                )

            ip_dirs = ["list"]
            ip_dirs += collect_ip_dirs(kernel_model, ipstitch_path)
            ip_dirs_str = "[{}]".format(" ".join(ip_dirs))
            config.append(
                "set_property ip_repo_paths "
                f"[concat [get_property ip_repo_paths [current_project]] {ip_dirs_str}] "
                "[current_project]"
            )
            config.append("update_ip_catalog -rebuild -scan_changes")

            ifnames = eval(kernel_model.get_metadata_prop("vivado_stitch_ifnames"))

            # gather info on connectivity
            # assume each node connected to outputs/inputs is DMA:
            # has axis, aximm and axilite
            # everything else is axis-only
            # assume only one connection from each ip to the next
            # all aximm allocated to DDR[0]
            # all kernels allocated to SLR0
            producer = None if len(node.input) == 0 else model.find_producer(node.input[0])
            consumer = model.find_consumers(node.output[0])
            # define kernel instances
            # name kernels connected to graph inputs as idmaxx
            # name kernels connected to graph outputs as odmaxx
            # do not expect IDMA/ODMA when instrumentation is enabled
            if (not use_instrumentation or self.enable_finn_switch) and (
                (producer is None) or (consumer == [])
            ):
                # TODO not a good way of checking for external inp&out
                # should look at the list of top-level in/out instead
                if producer is None:
                    instance_names[node.name] = "idma" + str(idma_idx)
                    idma_idx += 1
                elif consumer == []:
                    instance_names[node.name] = "odma" + str(odma_idx)
                    odma_idx += 1
                config.append(
                    f"create_bd_cell -type ip -vlnv {vivado_stitch_vlnv} "
                    f"{instance_names[node.name]}"
                )
                config.append(
                    "connect_bd_intf_net [get_bd_intf_pins "
                    f"{instance_names[node.name]}/m_axi_gmem0] "
                    f"[get_bd_intf_pins smartconnect_0/S{aximm_idx:02d}_AXI]"
                )
                if len(ifnames["axilite"]) != 1:
                    raise FINNInternalError("Must have 1 AXI lite interface on IODMA nodes")
                axilite_intf_name = ifnames["axilite"][0]
                config.append(
                    "connect_bd_intf_net [get_bd_intf_pins "
                    f"{instance_names[node.name]}/{axilite_intf_name}] "
                    "[get_bd_intf_pins "
                    f"axi_interconnect_{axilite_interconnect_idx}/M{axilite_idx:02d}_AXI]"
                )
                # assign_bd_address with appropriate range/offset
                config.append(
                    f"assign_axi_addr_proc {instance_names[node.name]}/{axilite_intf_name}"
                )

                aximm_idx += 1
                axilite_idx += 1
                if axilite_idx == 64:
                    axilite_interconnect_idx += 1
                    axilite_idx = 0
                if axilite_interconnect_idx == 0:
                    master_axilite_idx += 1
            else:
                instance_names[node.name] = node.name
                config.append(
                    f"create_bd_cell -type ip -vlnv {vivado_stitch_vlnv} "
                    f"{instance_names[node.name]}"
                )

                for axilite_intf_name in ifnames["axilite"]:
                    config.append(
                        "connect_bd_intf_net [get_bd_intf_pins "
                        f"{instance_names[node.name]}/{axilite_intf_name}] "
                        "[get_bd_intf_pins "
                        f"axi_interconnect_{axilite_interconnect_idx}/M{axilite_idx:02d}_AXI]"
                    )
                    # assign_bd_address with appropriate range/offset
                    config.append(
                        f"assign_axi_addr_proc {instance_names[node.name]}/{axilite_intf_name}"
                    )
                    axilite_idx += 1
                    if axilite_idx == 64:
                        axilite_interconnect_idx += 1
                        axilite_idx = 0
                    if axilite_interconnect_idx == 0:
                        master_axilite_idx += 1
            sdp_node.set_nodeattr("instance_name", instance_names[node.name])

            config.append(
                f"connect_bd_net [get_bd_pins {instance_names[node.name]}/ap_clk] "
                "[get_bd_pins smartconnect_0/aclk]"
            )
            config.append(
                f"connect_bd_net [get_bd_pins {instance_names[node.name]}/ap_rst_n] "
                "[get_bd_pins smartconnect_0/aresetn]"
            )
            # connect streams
            if self.enable_finn_switch:
                for i in range(len(node.input)):
                    if producer is not None:
                        producer = model.find_producer(node.input[i])
                        j = list(producer.output).index(node.input[i])
                        producer_model = ModelWrapper(getCustomOp(producer).get_nodeattr("model"))
                        producer_idma = any(
                            s.name.startswith("IODMA") for s in producer_model.graph.output
                        )
                        # True when this node is the terminal output endpoint (ODMA or last SDP).
                        # Previously detected via TLastMarker inputs, but TLastMarker is no longer
                        # inserted in multi-DNN flows. Use the graph-level consumer check instead.
                        node_odma = consumer == []
                        if not (producer_idma or node_odma):
                            config.append(
                                "connect_bd_intf_net [get_bd_intf_pins "
                                f"{instance_names[node.name]}/s_axis_{i}] "
                                f"[get_bd_intf_pins {instance_names[producer.name]}/m_axis_{j}]"
                            )
                        elif producer_idma:
                            config.append(
                                "connect_bd_intf_net [get_bd_intf_pins "
                                f"{instance_names[producer.name]}/m_axis_{j}] "
                                "[get_bd_intf_pins finn_switch/A_IN0]"
                            )

                            config.append(
                                "connect_bd_intf_net [get_bd_intf_pins finn_switch/A_IN1] "
                                "[get_bd_intf_pins instrumentation_wrap_0/finnix]"
                            )

                            config.append(
                                "connect_bd_intf_net [get_bd_intf_pins "
                                f"{instance_names[node.name]}/s_axis_0] "
                                "[get_bd_intf_pins finn_switch/A_OUT]"
                            )

                            ifnames = kernel_model.get_metadata_prop("vivado_stitch_ifnames")
                            ifnames = json.loads(ifnames)
                            width = ifnames["s_axis"][0][1]
                            config.append(
                                f"set_property CONFIG.DATA_WIDTH_A {{{width}}} [get_bd_cells "
                                "finn_switch]"
                            )
                        else:
                            config.append(
                                "connect_bd_intf_net [get_bd_intf_pins "
                                f"{instance_names[node.name]}/s_axis_{i}] "
                                "[get_bd_intf_pins finn_switch/B_OUT0]"
                            )

                            config.append(
                                "connect_bd_intf_net [get_bd_intf_pins finn_switch/B_OUT1] "
                                "[get_bd_intf_pins instrumentation_wrap_0/finnox]"
                            )

                            config.append(
                                "connect_bd_intf_net [get_bd_intf_pins "
                                f"{instance_names[producer.name]}/m_axis_0] "
                                "[get_bd_intf_pins finn_switch/B_IN]"
                            )

                            ifnames = kernel_model.get_metadata_prop("vivado_stitch_ifnames")
                            ifnames = json.loads(ifnames)
                            width = ifnames["s_axis"][0][1]
                            config.append(
                                f"set_property CONFIG.DATA_WIDTH_B {{{width}}} [get_bd_cells "
                                "finn_switch]"
                            )
            else:
                for i in range(len(node.input)):
                    producer = model.find_producer(node.input[i])
                    if producer is not None:
                        j = list(producer.output).index(node.input[i])
                        config.append(
                            "connect_bd_intf_net [get_bd_intf_pins "
                            f"{instance_names[node.name]}/s_axis_{i}] "
                            f"[get_bd_intf_pins {instance_names[producer.name]}/m_axis_{j}]"
                        )

            # connect first/last dataflow partition to instrumentation wrapper
            if use_instrumentation and not self.enable_finn_switch:
                if producer is None:
                    config.append(
                        "connect_bd_intf_net [get_bd_intf_pins "
                        f"{instance_names[node.name]}/s_axis_0] "
                        "[get_bd_intf_pins instrumentation_wrap_0/finnix]"
                    )
                if consumer == []:
                    config.append(
                        "connect_bd_intf_net [get_bd_intf_pins "
                        f"{instance_names[node.name]}/m_axis_0] "
                        "[get_bd_intf_pins instrumentation_wrap_0/finnox]"
                    )

            # connect ring bus for live FIFO sizing
            if self.live_fifo_sizing:
                if "icfg" not in ifnames["ap_none"] or "ocfg" not in ifnames["ap_none"]:
                    raise FINNError(
                        "Live FIFO sizing requested but no icfg/ocfg interfaces found "
                        f"on SDP {node.name}"
                    )
                if sdp_id == 0:
                    # connect first SDP to fifo_controller
                    config.append(
                        "connect_bd_net [get_bd_pins fifo_controller_0/ocfg] "
                        f"[get_bd_pins {instance_names[node.name]}/icfg]"
                    )
                else:
                    # connect previous SDP to this SDP
                    config.append(
                        f"connect_bd_net [get_bd_pins {instance_names[prev_node_name]}/ocfg] "
                        f"[get_bd_pins {instance_names[node.name]}/icfg]"
                    )
                if sdp_id == num_sdps - 1:
                    # connect last SDP to fifo_controller
                    config.append(
                        f"connect_bd_net [get_bd_pins {instance_names[node.name]}/ocfg] "
                        "[get_bd_pins fifo_controller_0/icfg]"
                    )
                prev_node_name = node.name

        # TODO: WORKAROUND, do not instantiate smartconnect when not needed!
        if use_instrumentation and not self.enable_finn_switch:
            config.append("delete_bd_objs [get_bd_cells smartconnect_0]")
            aximm_idx = 1

        # finalize nested interconnect clock/reset
        for i in range(1, nested_interconnect_count + 1):
            config.append(
                "connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] "
                f"[get_bd_pins axi_interconnect_{i}/M*_ACLK]"
            )
            config.append(
                "connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] "
                f"[get_bd_pins axi_interconnect_{i}/M*_ARESETN]"
            )

        # create a temporary folder for the project
        vivado_pynq_proj_dir = make_build_dir(prefix="vivado_zynq_proj_")
        model.set_metadata_prop("vivado_pynq_proj", vivado_pynq_proj_dir)

        fclk_mhz = int(1 / (self.period_ns * 0.001))

        pr_config = self._generate_pr_flow(model) if (partial_reconfiguration or sw_nodes) else ""

        # create a TCL recipe for the project
        ipcfg = vivado_pynq_proj_dir + "/ip_config.tcl"
        config = "\n".join(config) + "\n"
        with Path(ipcfg).open("w") as f:
            f.write(
                (
                    templates.custom_zynq_shell_template
                    % (
                        fclk_mhz,
                        master_axilite_idx,
                        aximm_idx,
                        self.platform,
                        pynq_part_map[self.platform],
                        config,
                        self.enable_debug,
                        self.enable_gpio_reset,
                        self.enable_finn_switch,
                    )
                )
                .replace("$BOARDFILES$", str(get_settings().finn_deps / "board_files"))
                .replace("$PR_CONFIG$", pr_config)
            )

        # create a TCL recipe for the project
        synth_project_sh = vivado_pynq_proj_dir + "/synth_project.sh"
        working_dir = Path.cwd()
        with Path(synth_project_sh).open("w") as f:
            f.write("#!/bin/bash \n")
            f.write(f"cd {vivado_pynq_proj_dir}\n")
            f.write(f"vivado -mode batch -source {ipcfg}\n")
            f.write(f"cd {working_dir}\n")

        # call the synthesis script
        bash_command = ["bash", synth_project_sh]
        try:
            launch_process_helper(bash_command, print_stdout=False)
        except CalledProcessError as e:
            raise FINNSynthesisError(
                f"Synthesis failed. Check {vivado_pynq_proj_dir} for details.",
                Path(vivado_pynq_proj_dir) / "vivado.log",
            ) from e

        bitfile_name = vivado_pynq_proj_dir + "/finn_zynq_link.runs/impl_1/top_wrapper.bit"
        if not Path(bitfile_name).is_file():
            raise FINNSynthesisError(
                f"Synthesis failed, no bitfile found. Check logs under {vivado_pynq_proj_dir}",
                Path(vivado_pynq_proj_dir) / "vivado.log",
            )
        deploy_bitfile_name = vivado_pynq_proj_dir + "/resizer.bit"
        copy(bitfile_name, deploy_bitfile_name)

        model.set_metadata_prop("bitfile", deploy_bitfile_name)

        # Store in bitfile_output as well, for compatability reasons
        model.set_metadata_prop("bitfile_output", deploy_bitfile_name)

        hwh_name_alts = [
            vivado_pynq_proj_dir + "/finn_zynq_link.srcs/sources_1/bd/top/hw_handoff/top.hwh",
            vivado_pynq_proj_dir + "/finn_zynq_link.gen/sources_1/bd/top/hw_handoff/top.hwh",
        ]
        hwh_name = None
        for hwh_name_cand in hwh_name_alts:
            if Path(hwh_name_cand).is_file():
                hwh_name = hwh_name_cand
        if hwh_name is None or not Path(hwh_name).is_file():
            raise FINNSynthesisError(
                f"Synthesis failed, no bitfile found. Check logs under {vivado_pynq_proj_dir}",
                Path(vivado_pynq_proj_dir) / "vivado.log",
            )
        deploy_hwh_name = vivado_pynq_proj_dir + "/resizer.hwh"
        copy(hwh_name, deploy_hwh_name)
        model.set_metadata_prop("hw_handoff", deploy_hwh_name)
        # filename for the synth utilization report
        synth_report_filename = vivado_pynq_proj_dir + "/synth_report.xml"
        model.set_metadata_prop("vivado_synth_rpt", synth_report_filename)
        if partial_reconfiguration:
            partial_bs_dir = vivado_pynq_proj_dir + "/partial_bitstreams"
            if Path(partial_bs_dir).is_dir():
                model.set_metadata_prop("partial_bitfiles_dir", partial_bs_dir)
        return (model, False)

    def _generate_pr_flow(self, model: ModelWrapper) -> list[str]:
        """Generate partial reconfiguration hardware and bitstreams."""
        pr_config = []
        sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
        pr_sdp_nodes = []
        sw_sdp_nodes = []
        for sdp_node in sdp_nodes:
            sdp_node_inst = getCustomOp(sdp_node)
            dataflow_model_filename = sdp_node_inst.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            if any(
                n.op_type == "NodeContainer"
                and getCustomOp(n).get_nodeattr("multi_dnn_type") == "partial_reconfiguration"
                for n in kernel_model.graph.node
            ):
                pr_sdp_nodes.append(sdp_node)
            elif any(
                n.op_type == "NodeContainer"
                and getCustomOp(n).get_nodeattr("multi_dnn_type") == "selectable_weights"
                for n in kernel_model.graph.node
            ):
                sw_sdp_nodes.append(sdp_node)

        # Capture the current top-level BD design before any sub-design switches.
        # This is needed even when there are no PR SDPs (SW-only case).
        pr_config.append("set curdesign [current_bd_design]")

        for pr_sdp_node in pr_sdp_nodes:
            pr_sdp_node_inst = getCustomOp(pr_sdp_node)
            dataflow_model_filename = pr_sdp_node_inst.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            pr_node = next(
                n
                for n in kernel_model.graph.node
                if n.op_type == "NodeContainer"
                and getCustomOp(n).get_nodeattr("multi_dnn_type") == "partial_reconfiguration"
            )
            pr_node_inst = getCustomOp(pr_node)
            sdp_name = pr_sdp_node.name
            for body_idx in range(pr_node_inst.get_nodeattr("bodies")):
                body_model = pr_node_inst.get_nodeattr("body_" + str(body_idx))
                if body_idx == 0:
                    # Special case, as this block is in the main bd
                    pr_config.append(f"group_bd_cells Hier_{sdp_name} [get_bd_cells {sdp_name}]")

                    # Validate before creating a Block Design Container
                    pr_config.append("validate_bd_design")

                    pr_config.append("startgroup")
                    pr_config.append("set curdesign [current_bd_design]")
                    pr_config.append(
                        f"create_bd_design -cell [get_bd_cells /Hier_{sdp_name}] Hier_{sdp_name}"
                    )
                    pr_config.append("current_bd_design $curdesign")

                    pr_config.append(
                        "set new_cell "
                        f"[create_bd_cell -type container -reference Hier_{sdp_name} "
                        f"Hier_{sdp_name}_temp]"
                    )
                    pr_config.append(f"replace_bd_cell [get_bd_cells /Hier_{sdp_name}] $new_cell")

                    pr_config.append(f"catch {{delete_bd_objs [get_bd_cells /Hier_{sdp_name}]}}")
                    pr_config.append(f"set_property name Hier_{sdp_name} $new_cell")
                    pr_config.append("endgroup")

                    # Enable DFX on the BDC
                    pr_config.append(
                        f"set_property CONFIG.ENABLE_DFX {{true}} [get_bd_cells Hier_{sdp_name}]"
                    )
                else:
                    # For each additional body create a Reconfigurable Module BD
                    # boundary ports are pre-defined by the container
                    body_vlnv = body_model.get_metadata_prop("vivado_stitch_vlnv")
                    body_ipstitch_path = body_model.get_metadata_prop("vivado_stitch_proj")
                    body_ifnames = eval(body_model.get_metadata_prop("vivado_stitch_ifnames"))

                    body_ip_dirs = ["list"]
                    body_ip_dirs += collect_ip_dirs(body_model, body_ipstitch_path)
                    body_ip_dirs_str = "[{}]".format(" ".join(body_ip_dirs))

                    bd_name = f"Hier_{sdp_name}_{body_idx}"
                    instance_name = f"body_{body_idx}_ip"

                    pr_config.append(
                        "create_bd_design -boundary_from_container "
                        f"[get_bd_cells /Hier_{sdp_name}] {bd_name}"
                    )

                    pr_config.append(f"current_bd_design [get_bd_designs {bd_name}]")
                    pr_config.append(
                        "set_property ip_repo_paths "
                        "[concat [get_property ip_repo_paths [current_project]] "
                        f"{body_ip_dirs_str}] "
                        "[current_project]"
                    )
                    pr_config.append("update_ip_catalog -rebuild -scan_changes")
                    pr_config.append(f"create_bd_cell -type ip -vlnv {body_vlnv} {instance_name}")
                    pr_config.append(
                        f"connect_bd_net [get_bd_pins {instance_name}/ap_clk] "
                        "[get_bd_ports ap_clk]"
                    )
                    pr_config.append(
                        f"connect_bd_net [get_bd_pins {instance_name}/ap_rst_n] "
                        "[get_bd_ports ap_rst_n]"
                    )
                    for s_axis_name, _ in body_ifnames.get("s_axis", []):
                        pr_config.append(
                            f"connect_bd_intf_net [get_bd_intf_pins {instance_name}/{s_axis_name}] "
                            f"[get_bd_intf_ports {s_axis_name}]"
                        )
                    for m_axis_name, _ in body_ifnames.get("m_axis", []):
                        pr_config.append(
                            f"connect_bd_intf_net [get_bd_intf_pins {instance_name}/{m_axis_name}] "
                            f"[get_bd_intf_ports {m_axis_name}]"
                        )
                    for axilite_name in body_ifnames.get("axilite", []):
                        pr_config.append(
                            "connect_bd_intf_net [get_bd_intf_pins "
                            f"{instance_name}/{axilite_name}] "
                            f"[get_bd_intf_ports {axilite_name}]"
                        )
                    for aximm_name, _ in body_ifnames.get("aximm", []):
                        pr_config.append(
                            f"connect_bd_intf_net [get_bd_intf_pins {instance_name}/{aximm_name}] "
                            f"[get_bd_intf_ports {aximm_name}]"
                        )
                    pr_config.append("save_bd_design")
                    pr_config.append("validate_bd_design")
                    pr_config.append("current_bd_design $curdesign")

        # Switch back to top-level design and add all multi-DNN wrapper RTL files
        pr_config.append("current_bd_design $curdesign")
        rtllib = Path(get_settings().finn_rtllib)
        for wrapper_file in [
            rtllib / "dfx" / "dfx_wrapper" / "dfx_wrapper.sv",
            rtllib / "dfx" / "dfx_wrapper" / "dfx_wrapper_wrapper.v",
            rtllib / "dfx" / "dfx_tuser_passthrough" / "dfx_tuser_passthrough.sv",
            rtllib / "dfx" / "dfx_tuser_passthrough" / "dfx_tuser_passthrough_wrapper.v",
            rtllib / "dfx" / "sw_wrapper" / "sw_wrapper.sv",
            rtllib / "dfx" / "sw_wrapper" / "sw_wrapper_wrapper.v",
        ]:
            pr_config.append(
                "add_files -copy_to [get_property DIRECTORY [current_project]] -norecurse "
                f"{wrapper_file}"
            )

        if pr_sdp_nodes:
            # DFX Controller & ICAP (only needed for partial reconfiguration)
            for pr_file in [
                Path(get_settings().finn_rtllib) / "icap" / "icape3_wrapper.v",
            ]:
                pr_config.append(
                    "add_files -copy_to [get_property DIRECTORY [current_project]] -norecurse "
                    f"{pr_file}"
                )
            pr_config.append("create_bd_cell -type module -reference icape3_wrapper icape3_wrapper")
            pr_config.append(
                "create_bd_cell -type ip -vlnv xilinx.com:ip:dfx_controller:1.0 dfx_controller_0"
            )
            pr_config.append(
                "source [get_property REPOSITORY "
                "[get_ipdefs *dfx_controller:1.0]]"
                "/xilinx/dfx_controller_v1_0/tcl/api.tcl -notrace"
            )
            pr_config.append(
                "connect_bd_intf_net [get_bd_intf_pins dfx_controller_0/ICAP] "
                "[get_bd_intf_pins icape3_wrapper/ICAP]"
            )
            pr_config.append(
                "connect_bd_net [get_bd_pins icape3_wrapper/clk] "
                "[get_bd_pins clk_wiz_0/clk_out1]"
            )
            for pr_sdp in pr_sdp_nodes:
                pr_sdp_inst = getCustomOp(pr_sdp)
                pr_sdp_model = ModelWrapper(pr_sdp_inst.get_nodeattr("model"))
                pr_nodecontainer = next(
                    n
                    for n in pr_sdp_model.graph.node
                    if n.op_type == "NodeContainer"
                    and getCustomOp(n).get_nodeattr("multi_dnn_type") == "partial_reconfiguration"
                )
                pr_nodecontainer_inst = getCustomOp(pr_nodecontainer)
                num_bodies = pr_nodecontainer_inst.get_nodeattr("bodies")
                dfx_cont_vs_config = []

                vs_name = pr_sdp.name
                dfx_cont_vs_config.append(f"CONFIG.VS.{vs_name}.NUM_RMS_ALLOCATED {num_bodies}")
                for rm_idx in range(num_bodies):
                    dfx_cont_vs_config.append(f"CONFIG.VS.{vs_name}.RM.{rm_idx}.BS.0.ADDRESS 0x0")
                dfx_cont_vs_config.append(
                    f"CONFIG.VS.{vs_name}.NUM_TRIGGERS_ALLOCATED {num_bodies}"
                )
                dfx_cont_vs_config.append(f"CONFIG.VS.{vs_name}.NUM_HW_TRIGGERS {num_bodies}")
                for rm_idx in range(num_bodies):
                    dfx_cont_vs_config.append(f"CONFIG.VS.{vs_name}.TRIGGER{rm_idx}_TO_RM {rm_idx}")
                pr_config.append(
                    "dfx_controller_v1_0::set_property -dict [list {}] "
                    "[get_bd_cells dfx_controller_0]".format(" ".join(dfx_cont_vs_config))
                )

            pr_config.append("set_property CONFIG.PSU__USE__S_AXI_GP3 {1} [get_bd_cells zynq_ps]")

            # Create dedicated reset controller for DFX controller
            # (reset 1, independent of main system reset 0)
            pr_config.append(
                "create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 "
                "proc_sys_reset_dfx"
            )
            pr_config.append(
                "connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] "
                "[get_bd_pins proc_sys_reset_dfx/slowest_sync_clk]"
            )
            pr_config.append(
                "connect_bd_net [get_bd_pins clk_wiz_0/locked] "
                "[get_bd_pins proc_sys_reset_dfx/dcm_locked]"
            )

            pr_config.append(
                "set_property CONFIG.PSU__NUM_FABRIC_RESETS {2} [get_bd_cells zynq_ps]"
            )

            pr_config.append(
                "connect_bd_net [get_bd_pins zynq_ps/pl_resetn1] "
                "[get_bd_pins proc_sys_reset_dfx/ext_reset_in]"
            )

            # Connect DFX controller clock and reset
            pr_config.append(
                "connect_bd_net [get_bd_pins dfx_controller_0/clk] "
                "[get_bd_pins smartconnect_0/aclk]"
            )
            pr_config.append(
                "connect_bd_net [get_bd_pins dfx_controller_0/clk] "
                "[get_bd_pins dfx_controller_0/icap_clk]"
            )
            pr_config.append(
                "connect_bd_net [get_bd_pins dfx_controller_0/reset] "
                "[get_bd_pins proc_sys_reset_dfx/peripheral_aresetn]"
            )
            pr_config.append(
                "connect_bd_net [get_bd_pins dfx_controller_0/icap_reset] "
                "[get_bd_pins proc_sys_reset_dfx/peripheral_aresetn]"
            )

            # Connect DFX controller s_axi_reg to PS master via axi_interconnect_0
            # (extend axi_interconnect_0 with one extra master port)
            pr_config.append(
                "set dfx_mi_idx [get_property CONFIG.NUM_MI [get_bd_cells axi_interconnect_0]]"
            )
            pr_config.append(
                "set_property CONFIG.NUM_MI [expr {$dfx_mi_idx + 1}] "
                "[get_bd_cells axi_interconnect_0]"
            )
            pr_config.append(
                "connect_bd_intf_net [get_bd_intf_pins dfx_controller_0/s_axi_reg] "
                "[get_bd_intf_pins axi_interconnect_0/[format M%02d_AXI $dfx_mi_idx]]"
            )
            pr_config.append(
                "connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] "
                "[get_bd_pins axi_interconnect_0/[format M%02d_ACLK $dfx_mi_idx]]"
            )
            pr_config.append(
                "connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] "
                "[get_bd_pins axi_interconnect_0/[format M%02d_ARESETN $dfx_mi_idx]]"
            )
            pr_config.append(
                "assign_bd_address [get_bd_addr_segs {dfx_controller_0/s_axi_reg/Reg}]"
            )

            pr_config.append("save_bd_design")

            # SmartConnect to route dfx_controller AXI master → zynq_ps/S_AXI_HP1_FPD
            pr_config.append(
                "set smartconnect_dfx_vlnv "
                "[get_property VLNV [get_ipdefs xilinx.com:ip:smartconnect:*]]"
            )
            pr_config.append(
                "create_bd_cell -type ip -vlnv $smartconnect_dfx_vlnv smartconnect_dfx"
            )
            pr_config.append(
                "set_property -dict [list CONFIG.NUM_SI {1}] [get_bd_cells smartconnect_dfx]"
            )
            pr_config.append(
                "connect_bd_intf_net [get_bd_intf_pins dfx_controller_0/M_AXI_MEM] "
                "[get_bd_intf_pins smartconnect_dfx/S00_AXI]"
            )
            pr_config.append(
                "connect_bd_intf_net [get_bd_intf_pins smartconnect_dfx/M00_AXI] "
                "[get_bd_intf_pins zynq_ps/S_AXI_HP1_FPD]"
            )
            pr_config.append(
                "connect_bd_net [get_bd_pins smartconnect_dfx/aclk] "
                "[get_bd_pins smartconnect_0/aclk]"
            )
            pr_config.append(
                "connect_bd_net [get_bd_pins smartconnect_dfx/aresetn] "
                "[get_bd_pins smartconnect_0/aresetn]"
            )
            pr_config.append(
                "connect_bd_net [get_bd_pins zynq_ps/saxihp1_fpd_aclk] "
                "[get_bd_pins smartconnect_0/aclk]"
            )

            # Assign the DDR address segment to dfx_controller_0/M_AXI_MEM so it can
            # read bitstreams from DRAM via smartconnect_dfx → zynq_ps/S_AXI_HP1_FPD.
            # All SDP axilite segments are already assigned via assign_axi_addr_proc,
            # so this call picks up only the remaining HP1 DDR window.
            pr_config.append("assign_bd_address")

            # Source AMD dfx_decoupler API once (needed before per-PR instantiation loop)
            pr_config.append(
                "source [get_property REPOSITORY "
                "[get_ipdefs *dfx_decoupler:1.0]]"
                "/xilinx/dfx_decoupler_v1_0/tcl/api.tcl -notrace"
            )

            reset_aresetn_pin = "smartconnect_0/aresetn"
        else:
            # SW-only: connect to the existing reset net via smartconnect_0/aresetn.
            # The template connects that pin to the peripheral_aresetn of the
            # automation-created rst_zynq_ps_* cell, so it carries the same signal.
            reset_aresetn_pin = "smartconnect_0/aresetn"

        # Compute a single consistent tUSER width for the entire accelerator so that
        # all wrapper modules use the same TUSER_WIDTH and tUSER bits propagate
        # without truncation end-to-end.
        def _tuser_width_for_pr(pr_sdp: NodeProto) -> int:
            """Determine the tUSER width required by a partial reconfiguration SDP.

            Uses the NodeContainer's explicit tuser_width attribute if set, otherwise
            derives the width from the number of bodies that have to be distinguished.
            """
            inst = getCustomOp(pr_sdp)
            km = ModelWrapper(inst.get_nodeattr("model"))
            nc = next(
                n
                for n in km.graph.node
                if n.op_type == "NodeContainer"
                and getCustomOp(n).get_nodeattr("multi_dnn_type") == "partial_reconfiguration"
            )
            nc_inst = getCustomOp(nc)
            nb = nc_inst.get_nodeattr("bodies")
            attr = nc_inst.get_nodeattr("tuser_width")
            return attr if attr > 0 else max(math.ceil(math.log2(max(nb, 2))), 1)

        def _tuser_width_for_sw(sw_sdp: NodeProto) -> int:
            """Determine the tUSER width required by a selectable weights SDP.

            The width is derived from the number of weight sets that have to be
            distinguished by the tUSER signal.
            """
            inst = getCustomOp(sw_sdp)
            km = ModelWrapper(inst.get_nodeattr("model"))
            nc = next(
                n
                for n in km.graph.node
                if n.op_type == "NodeContainer"
                and getCustomOp(n).get_nodeattr("multi_dnn_type") == "selectable_weights"
            )
            nb = getCustomOp(nc).get_nodeattr("bodies")
            return max(math.ceil(math.log2(max(nb, 2))), 1)

        all_tuser_widths = [_tuser_width_for_pr(p) for p in pr_sdp_nodes] + [
            _tuser_width_for_sw(s) for s in sw_sdp_nodes
        ]
        global_tuser_width = max(all_tuser_widths) if all_tuser_widths else 1

        # Propagate the tUSER width to the finn_switch so the A path (instrumentation
        # → first SDP) carries tuser all the way to the dfx_wrapper input.
        if self.enable_finn_switch:
            pr_config.append(
                f"set_property CONFIG.TUSER_WIDTH_A {{{global_tuser_width}}} [get_bd_cells "
                "finn_switch]"
            )

        # Per-region DFX Wrapper and AMD DFX Decoupler instantiation.
        # Each PR SDP gets its own dfx_wrapper (static region controller) and
        # dfx_decoupler (output-side RP isolation), replacing the previous global
        # dfx_schedule + dfx_finn_decouple + dfx_decoupler architecture.
        for pr_sdp in pr_sdp_nodes:
            pr_sdp_inst = getCustomOp(pr_sdp)
            pr_sdp_model = ModelWrapper(pr_sdp_inst.get_nodeattr("model"))
            pr_nodecontainer = next(
                n
                for n in pr_sdp_model.graph.node
                if n.op_type == "NodeContainer"
                and getCustomOp(n).get_nodeattr("multi_dnn_type") == "partial_reconfiguration"
            )
            pr_nodecontainer_inst = getCustomOp(pr_nodecontainer)
            sdp_name = pr_sdp.name
            num_bodies = pr_nodecontainer_inst.get_nodeattr("bodies")
            # Use AXI-Stream-padded widths (multiples of 8) for both data paths.
            # The stitched BDC IP always uses padded widths; using the unpadded
            # get_instream_width() would cause a data-width mismatch in the BD.
            in_data_width = pr_nodecontainer_inst.get_instream_width_padded()
            out_data_width = pr_nodecontainer_inst.get_outstream_width_padded()

            body_0_model = pr_nodecontainer_inst.get_nodeattr("body_0")
            body_0_ifnames = eval(body_0_model.get_metadata_prop("vivado_stitch_ifnames"))
            s_axis_name = body_0_ifnames["s_axis"][0][0]
            m_axis_name = body_0_ifnames["m_axis"][0][0]

            # NUM_OUTPUT_BEATS: number of AXI-Stream beats per output frame.
            # Derived from the last node of the first PR body (all bodies are
            # functionally equivalent and share the same output shape).
            # Matches the formula used by dfx_tuser_passthrough and sw_wrapper.
            last_node_inst = getCustomOp(body_0_model.graph.node[-1])
            out_shape = last_node_inst.get_folded_output_shape()
            num_output_beats = int(math.prod(out_shape[1:-1]))

            # Create per-region DFX Wrapper (replaces global dfx_schedule + dfx_finn_decouple)
            pr_config.append(
                f"create_bd_cell -type module -reference dfx_wrapper_wrapper dfx_wrapper_{sdp_name}"
            )
            pr_config.append(
                "set_property -dict [list "
                f"CONFIG.IN_DATA_WIDTH {{{in_data_width}}} "
                f"CONFIG.OUT_DATA_WIDTH {{{out_data_width}}} "
                f"CONFIG.TUSER_WIDTH {{{global_tuser_width}}} "
                f"CONFIG.NUM_RM {{{num_bodies}}} "
                f"CONFIG.NUM_OUTPUT_BEATS {{{num_output_beats}}}] "
                f"[get_bd_cells dfx_wrapper_{sdp_name}]"
            )
            pr_config.append(
                f"connect_bd_net [get_bd_pins dfx_wrapper_{sdp_name}/aclk] "
                "[get_bd_pins smartconnect_0/aclk]"
            )
            pr_config.append(
                f"connect_bd_net [get_bd_pins dfx_wrapper_{sdp_name}/aresetn] "
                f"[get_bd_pins {reset_aresetn_pin}]"
            )

            # Create per-region AMD DFX Decoupler on the BDC output side
            pr_config.append(
                "create_bd_cell -type ip -vlnv xilinx.com:ip:dfx_decoupler:1.0 "
                f"dfx_decoupler_{sdp_name}"
            )
            pr_config.append(
                "dfx_decoupler_v1_0::set_property -dict "
                "[list CONFIG.INTF.intf_0.VLNV "
                "xilinx.com:interface:axis_rtl:1.0] "
                f"[get_bd_cells dfx_decoupler_{sdp_name}]"
            )

            # Wire DFX controller signals for this virtual socket (VS = sdp_name):
            #   vsm_<sdp_name>_hw_triggers <- dfx_wrapper controller_trigger (RM select)
            #   vsm_<sdp_name>_rm_decouple -> dfx_wrapper controller_decouple (decouple status)
            #   vsm_<sdp_name>_rm_decouple -> dfx_decoupler decouple (isolate BDC output)
            pr_config.append(
                f"connect_bd_net [get_bd_pins dfx_wrapper_{sdp_name}/controller_trigger] "
                f"[get_bd_pins dfx_controller_0/vsm_{sdp_name}_hw_triggers]"
            )
            pr_config.append(
                f"connect_bd_net [get_bd_pins dfx_controller_0/vsm_{sdp_name}_rm_decouple] "
                f"[get_bd_pins dfx_wrapper_{sdp_name}/controller_decouple]"
            )
            pr_config.append(
                f"connect_bd_net [get_bd_pins dfx_controller_0/vsm_{sdp_name}_rm_decouple] "
                f"[get_bd_pins dfx_decoupler_{sdp_name}/decouple]"
            )

            # Input side: find the upstream master, disconnect from BDC,
            # route through dfx_wrapper (s_axis -> rp_m_axis -> BDC input)
            pr_config.append(
                f"set upstream_master_{sdp_name} [get_bd_intf_pins -of_objects "
                f"[get_bd_intf_nets -of_objects [get_bd_intf_pins Hier_{sdp_name}/{s_axis_name}]] "
                "-filter {mode == Master}]"
            )
            pr_config.append(
                "delete_bd_objs [get_bd_intf_nets -of_objects "
                f"[get_bd_intf_pins Hier_{sdp_name}/{s_axis_name}]]"
            )
            pr_config.append(
                f"connect_bd_intf_net $upstream_master_{sdp_name} "
                f"[get_bd_intf_pins dfx_wrapper_{sdp_name}/s_axis]"
            )
            pr_config.append(
                f"connect_bd_intf_net [get_bd_intf_pins dfx_wrapper_{sdp_name}/rp_m_axis] "
                f"[get_bd_intf_pins Hier_{sdp_name}/{s_axis_name}]"
            )

            # Output side: find the downstream slave, disconnect from BDC,
            # route through dfx_decoupler (BDC output -> rp_intf_0 -> s_intf_0 -> rp_s_axis)
            # then through dfx_wrapper (m_axis -> downstream)
            pr_config.append(
                f"set downstream_slave_{sdp_name} [get_bd_intf_pins -of_objects "
                f"[get_bd_intf_nets -of_objects [get_bd_intf_pins Hier_{sdp_name}/{m_axis_name}]] "
                "-filter {mode == Slave}]"
            )
            pr_config.append(
                "delete_bd_objs [get_bd_intf_nets -of_objects "
                f"[get_bd_intf_pins Hier_{sdp_name}/{m_axis_name}]]"
            )
            pr_config.append(
                f"connect_bd_intf_net [get_bd_intf_pins Hier_{sdp_name}/{m_axis_name}] "
                f"[get_bd_intf_pins dfx_decoupler_{sdp_name}/rp_intf_0]"
            )
            pr_config.append(
                f"connect_bd_intf_net [get_bd_intf_pins dfx_decoupler_{sdp_name}/s_intf_0] "
                f"[get_bd_intf_pins dfx_wrapper_{sdp_name}/rp_s_axis]"
            )
            pr_config.append(
                f"connect_bd_intf_net [get_bd_intf_pins dfx_wrapper_{sdp_name}/m_axis] "
                f"$downstream_slave_{sdp_name}"
            )

            # Per-region reset: dfx_wrapper/accel_reset_n drives the BDC ap_rst_n directly,
            # replacing the global proc_sys_reset_accel approach.
            pr_config.append(
                f"set rst_net_hier_{sdp_name} "
                f"[get_bd_nets -of_objects [get_bd_pins Hier_{sdp_name}/ap_rst_n]]"
            )
            pr_config.append(
                f"if {{$rst_net_hier_{sdp_name} ne {{}}}} "
                f"{{ disconnect_bd_net $rst_net_hier_{sdp_name} [get_bd_pins "
                f"Hier_{sdp_name}/ap_rst_n] }}"
            )
            pr_config.append(
                f"connect_bd_net [get_bd_pins dfx_wrapper_{sdp_name}/accel_reset_n] "
                f"[get_bd_pins Hier_{sdp_name}/ap_rst_n]"
            )

        # Per-segment tUSER Passthrough wrapper instantiation.
        # Each static SDP (not PR, not SW) is wrapped in dfx_tuser_passthrough to
        # forward the tUSER side-channel and regenerate tLast at the output.
        static_sdp_nodes = [n for n in sdp_nodes if n not in pr_sdp_nodes and n not in sw_sdp_nodes]
        for non_pr_sdp in static_sdp_nodes:
            non_pr_sdp_inst = getCustomOp(non_pr_sdp)
            sdp_name = non_pr_sdp.name
            body_model = ModelWrapper(non_pr_sdp_inst.get_nodeattr("model"))

            body_ifnames = eval(body_model.get_metadata_prop("vivado_stitch_ifnames"))
            if not body_ifnames.get("s_axis") or not body_ifnames.get("m_axis"):
                # IDMA/ODMA endpoint nodes have no bidirectional stream interface;
                # they do not need a dfx_tuser_passthrough wrapper.
                continue
            s_axis_name = body_ifnames["s_axis"][0][0]
            m_axis_name = body_ifnames["m_axis"][0][0]

            # Separate padded widths for the input path (s_axis→rp_m_axis) and the
            # output path (rp_s_axis→m_axis); the wrapped static IP chain can change
            # the stream width (e.g. DWC inserted between nodes with different SIMD/PE).
            first_node_inst = getCustomOp(body_model.graph.node[0])
            last_node_inst = getCustomOp(body_model.graph.node[-1])
            in_data_width = first_node_inst.get_instream_width_padded()
            out_data_width = last_node_inst.get_outstream_width_padded()

            # NUM_OUTPUT_BEATS: number of AXI-Stream beats per output frame.
            # Derived from the folded output shape: product of all dimensions
            # except the outermost batch and the innermost element dimension,
            # matching the same formula used by InsertTLastMarker.
            out_shape = last_node_inst.get_folded_output_shape()
            num_output_beats = int(math.prod(out_shape[1:-1]))

            pr_config.append(
                "create_bd_cell -type module "
                "-reference dfx_tuser_passthrough_wrapper "
                f"dfx_tuser_passthrough_{sdp_name}"
            )
            pr_config.append(
                "set_property -dict [list "
                f"CONFIG.IN_DATA_WIDTH {{{in_data_width}}} "
                f"CONFIG.OUT_DATA_WIDTH {{{out_data_width}}} "
                f"CONFIG.TUSER_WIDTH {{{global_tuser_width}}} "
                f"CONFIG.NUM_OUTPUT_BEATS {{{num_output_beats}}}] "
                f"[get_bd_cells dfx_tuser_passthrough_{sdp_name}]"
            )
            pr_config.append(
                f"connect_bd_net [get_bd_pins dfx_tuser_passthrough_{sdp_name}/aclk] "
                "[get_bd_pins smartconnect_0/aclk]"
            )
            pr_config.append(
                f"connect_bd_net [get_bd_pins dfx_tuser_passthrough_{sdp_name}/aresetn] "
                f"[get_bd_pins {reset_aresetn_pin}]"
            )

            # Input side: find the upstream master, disconnect from SDP,
            # route through passthrough (s_axis -> rp_m_axis -> SDP input)
            pr_config.append(
                f"set upstream_master_{sdp_name} [get_bd_intf_pins -of_objects "
                f"[get_bd_intf_nets -of_objects [get_bd_intf_pins {sdp_name}/{s_axis_name}]] "
                "-filter {mode == Master}]"
            )
            pr_config.append(
                "delete_bd_objs [get_bd_intf_nets -of_objects "
                f"[get_bd_intf_pins {sdp_name}/{s_axis_name}]]"
            )
            pr_config.append(
                f"connect_bd_intf_net $upstream_master_{sdp_name} "
                f"[get_bd_intf_pins dfx_tuser_passthrough_{sdp_name}/s_axis]"
            )
            pr_config.append(
                "connect_bd_intf_net "
                f"[get_bd_intf_pins dfx_tuser_passthrough_{sdp_name}/rp_m_axis] "
                f"[get_bd_intf_pins {sdp_name}/{s_axis_name}]"
            )

            # Output side: find the downstream slave, disconnect from SDP,
            # route through passthrough (SDP output -> rp_s_axis -> m_axis -> downstream)
            pr_config.append(
                f"set downstream_slave_{sdp_name} [get_bd_intf_pins -of_objects "
                f"[get_bd_intf_nets -of_objects [get_bd_intf_pins {sdp_name}/{m_axis_name}]] "
                "-filter {mode == Slave}]"
            )
            pr_config.append(
                "delete_bd_objs [get_bd_intf_nets -of_objects "
                f"[get_bd_intf_pins {sdp_name}/{m_axis_name}]]"
            )
            pr_config.append(
                f"connect_bd_intf_net [get_bd_intf_pins {sdp_name}/{m_axis_name}] "
                f"[get_bd_intf_pins dfx_tuser_passthrough_{sdp_name}/rp_s_axis]"
            )
            pr_config.append(
                "connect_bd_intf_net "
                f"[get_bd_intf_pins dfx_tuser_passthrough_{sdp_name}/m_axis] "
                f"$downstream_slave_{sdp_name}"
            )

        # Per-SW-region SW Wrapper instantiation.
        # Each selectable_weights SDP is wrapped in sw_wrapper to send a set-selection
        # token (derived from the incoming tUSER) before each frame, then forward data.
        for sw_sdp in sw_sdp_nodes:
            sw_sdp_inst = getCustomOp(sw_sdp)
            sdp_name = sw_sdp.name
            body_model = ModelWrapper(sw_sdp_inst.get_nodeattr("model"))
            body_ifnames = eval(body_model.get_metadata_prop("vivado_stitch_ifnames"))

            # s_axis list contains both the data stream and the tap port (s_axis_tap_id_*).
            # Separate them by name prefix.
            s_axis_data = [
                (n, w) for n, w in body_ifnames["s_axis"] if not n.startswith("s_axis_tap")
            ]
            s_axis_tap_list = [
                (n, w) for n, w in body_ifnames["s_axis"] if n.startswith("s_axis_tap")
            ]
            if not s_axis_data:
                raise FINNInternalError(f"No data s_axis interface found on SW SDP {sdp_name}")
            if not s_axis_tap_list:
                raise FINNInternalError(f"No s_axis_tap interface found on SW SDP {sdp_name}")
            s_axis_name, data_in_width = s_axis_data[0]
            s_axis_tap_name = s_axis_tap_list[0][0]
            m_axis_name, data_out_width = body_ifnames["m_axis"][0]

            # Locate the selectable_weights NC to get num_sets and output beat count.
            sw_nc = next(
                n
                for n in body_model.graph.node
                if n.op_type == "NodeContainer"
                and getCustomOp(n).get_nodeattr("multi_dnn_type") == "selectable_weights"
            )
            sw_nc_inst = getCustomOp(sw_nc)
            num_sets = sw_nc_inst.get_nodeattr("bodies")

            last_node_inst = getCustomOp(body_model.graph.node[-1])
            out_shape = last_node_inst.get_folded_output_shape()
            num_output_beats = int(math.prod(out_shape[1:-1]))

            pr_config.append(
                f"create_bd_cell -type module -reference sw_wrapper_wrapper sw_wrapper_{sdp_name}"
            )
            pr_config.append(
                "set_property -dict [list "
                f"CONFIG.DATA_IN_WIDTH {{{data_in_width}}} "
                f"CONFIG.DATA_OUT_WIDTH {{{data_out_width}}} "
                f"CONFIG.TUSER_WIDTH {{{global_tuser_width}}} "
                f"CONFIG.NUM_SETS {{{num_sets}}} "
                f"CONFIG.NUM_OUTPUT_BEATS {{{num_output_beats}}}] "
                f"[get_bd_cells sw_wrapper_{sdp_name}]"
            )
            pr_config.append(
                f"connect_bd_net [get_bd_pins sw_wrapper_{sdp_name}/aclk] "
                "[get_bd_pins smartconnect_0/aclk]"
            )
            pr_config.append(
                f"connect_bd_net [get_bd_pins sw_wrapper_{sdp_name}/aresetn] "
                f"[get_bd_pins {reset_aresetn_pin}]"
            )

            # Input side: redirect upstream → sw_wrapper/s_axis → SDP/s_axis_name
            pr_config.append(
                f"set upstream_master_{sdp_name} [get_bd_intf_pins -of_objects "
                f"[get_bd_intf_nets -of_objects [get_bd_intf_pins {sdp_name}/{s_axis_name}]] "
                "-filter {mode == Master}]"
            )
            pr_config.append(
                "delete_bd_objs [get_bd_intf_nets -of_objects "
                f"[get_bd_intf_pins {sdp_name}/{s_axis_name}]]"
            )
            pr_config.append(
                f"connect_bd_intf_net $upstream_master_{sdp_name} "
                f"[get_bd_intf_pins sw_wrapper_{sdp_name}/s_axis]"
            )
            pr_config.append(
                "connect_bd_intf_net "
                f"[get_bd_intf_pins sw_wrapper_{sdp_name}/rp_m_axis] "
                f"[get_bd_intf_pins {sdp_name}/{s_axis_name}]"
            )

            # Output side: redirect SDP/m_axis_name → sw_wrapper/rp_s_axis → downstream
            pr_config.append(
                f"set downstream_slave_{sdp_name} [get_bd_intf_pins -of_objects "
                f"[get_bd_intf_nets -of_objects [get_bd_intf_pins {sdp_name}/{m_axis_name}]] "
                "-filter {mode == Slave}]"
            )
            pr_config.append(
                "delete_bd_objs [get_bd_intf_nets -of_objects "
                f"[get_bd_intf_pins {sdp_name}/{m_axis_name}]]"
            )
            pr_config.append(
                f"connect_bd_intf_net [get_bd_intf_pins {sdp_name}/{m_axis_name}] "
                f"[get_bd_intf_pins sw_wrapper_{sdp_name}/rp_s_axis]"
            )
            pr_config.append(
                "connect_bd_intf_net "
                f"[get_bd_intf_pins sw_wrapper_{sdp_name}/m_axis] "
                f"$downstream_slave_{sdp_name}"
            )

            # Set-selection side: sw_wrapper/m_axis_setsel → SDP/s_axis_tap_id_*
            pr_config.append(
                "connect_bd_intf_net "
                f"[get_bd_intf_pins sw_wrapper_{sdp_name}/m_axis_setsel] "
                f"[get_bd_intf_pins {sdp_name}/{s_axis_tap_name}]"
            )

        for pr_sdp in pr_sdp_nodes:
            pr_sdp_inst = getCustomOp(pr_sdp)
            pr_sdp_model = ModelWrapper(pr_sdp_inst.get_nodeattr("model"))
            pr_nodecontainer = next(
                n
                for n in pr_sdp_model.graph.node
                if n.op_type == "NodeContainer"
                and getCustomOp(n).get_nodeattr("multi_dnn_type") == "partial_reconfiguration"
            )
            pr_nodecontainer_inst = getCustomOp(pr_nodecontainer)
            sdp_name = pr_sdp.name
            num_bodies = pr_nodecontainer_inst.get_nodeattr("bodies")
            bd_list = ":".join(
                [f"Hier_{sdp_name}.bd"] + [f"Hier_{sdp_name}_{i}.bd" for i in range(1, num_bodies)]
            )
            pr_config.append(
                "set_property -dict [list "
                f"CONFIG.LIST_SIM_BD {{{bd_list}}} "
                f"CONFIG.LIST_SYNTH_BD {{{bd_list}}} "
                f"] [get_bd_cells Hier_{sdp_name}]"
            )

        pr_config.append("save_bd_design")
        pr_config.append("validate_bd_design")
        pr_config.append("make_wrapper -files [get_files top.bd] -import -fileset sources_1 -top")
        pr_config.append("set_property top top_wrapper [get_filesets sources_1]")
        pr_config.append("update_compile_order -fileset sources_1")
        pr_config.append("generate_target all [get_files top.bd]")

        if not pr_sdp_nodes:
            # SW-only mode: wrapper connections are complete. No PR configurations,
            # pblocks, or per-body impl runs needed; the outer Vivado flow handles
            # synthesis and implementation.
            pr_config = "\n".join(pr_config) + "\n"
            return pr_config

        # Enable Vivado DFX implementation flow only when there are actual
        # reconfigurable partitions. Setting this for SW-only builds would
        # cause Vivado to require PR configurations and pblocks that don't exist.
        pr_config.append("set_property PR_FLOW 1 [current_project]")

        pr_sdp_bodies = []
        pr_sdp_names = []
        for pr_sdp_node in pr_sdp_nodes:
            pr_sdp_inst = getCustomOp(pr_sdp_node)
            pr_sdp_model = ModelWrapper(pr_sdp_inst.get_nodeattr("model"))
            pr_nodecontainer_inst = getCustomOp(
                next(
                    n
                    for n in pr_sdp_model.graph.node
                    if n.op_type == "NodeContainer"
                    and getCustomOp(n).get_nodeattr("multi_dnn_type") == "partial_reconfiguration"
                )
            )
            pr_sdp_names.append(pr_sdp_node.name)
            pr_sdp_bodies.append(pr_nodecontainer_inst.get_nodeattr("bodies"))
        if not all(n == pr_sdp_bodies[0] for n in pr_sdp_bodies):
            raise FINNUserError("All NodeContainers must have the same number of bodies for pr")
        num_bodies = pr_sdp_bodies[0]

        for body_id in range(num_bodies):
            config_name = f"config_{body_id}"
            partitions = " ".join(
                f"top_i/Hier_{sdp_name}:Hier_{sdp_name}_inst_0"
                if body_id == 0
                else f"top_i/Hier_{sdp_name}:Hier_{sdp_name}_{body_id}_inst_0"
                for sdp_name in pr_sdp_names
            )
            pr_config.append(
                f"create_pr_configuration -name {config_name} -partitions [list {partitions}]"
            )
            if body_id == 0:
                pr_config.append("set_property PR_CONFIGURATION config_0 [get_runs impl_1]")
            else:
                impl_run = f"impl_body_{body_id}"
                pr_config.append(
                    f"create_run {impl_run} -parent_run impl_1 "
                    f"-flow {{Vivado Implementation 2020}} -pr_config {config_name}"
                )

        pr_config.append("launch_runs synth_1 -jobs 4")
        pr_config.append("wait_on_run synth_1")

        # Collect pblock info for every PR SDP before choosing the mode.
        # Each entry is (sdp_name, pblock_string_or_empty, pr_nodecontainer_inst).
        pr_sdp_pblock_info = []
        for pr_sdp in pr_sdp_nodes:
            pr_sdp_inst = getCustomOp(pr_sdp)
            sdp_name = pr_sdp.name
            pr_sdp_model = ModelWrapper(pr_sdp_inst.get_nodeattr("model"))
            pr_nodecontainer = next(
                n
                for n in pr_sdp_model.graph.node
                if n.op_type == "NodeContainer"
                and getCustomOp(n).get_nodeattr("multi_dnn_type") == "partial_reconfiguration"
            )
            pr_nodecontainer_inst = getCustomOp(pr_nodecontainer)
            pblock = pr_nodecontainer_inst.get_nodeattr("pblock")
            pr_sdp_pblock_info.append((sdp_name, pblock))

        pblocks_specified = [pblock for _, pblock in pr_sdp_pblock_info]
        all_empty = all(p == "" for p in pblocks_specified)
        all_specified = all(p != "" for p in pblocks_specified)

        if not all_empty and not all_specified:
            raise FINNError(
                "Mixed pblock specification: either ALL PR regions must have an explicit "
                "'pblock' string, or ALL must omit it (auto-floorplanning mode). "
                "Found a mix of specified and empty pblock attributes."
            )

        pr_config.append("open_run synth_1 -name synth_1")

        if all_empty:
            # ----------------------------------------------------------------
            # Auto-floorplanning mode: query per-cell resource usage from the
            # synthesised netlist and let generate_multi_dfx_pblocks size and
            # place the pblocks automatically.
            # ----------------------------------------------------------------
            dfx_tcl_path = str(
                Path(__file__).parent.parent.parent
                / "util"
                / "vivado_scripts"
                / "dfx_auto_floorplanning.tcl"
            )
            pr_config.append(f"source {{{dfx_tcl_path}}}")

            cell_names = [f"top_i/Hier_{sdp_name}" for sdp_name, _ in pr_sdp_pblock_info]
            pblock_names = [f"pblock_Hier_{sdp_name}" for sdp_name, _ in pr_sdp_pblock_info]

            pr_config.append(
                "auto_floorplan_from_synthesis "
                f"{{{' '.join(cell_names)}}} {{{' '.join(pblock_names)}}}"
            )
        else:
            # ----------------------------------------------------------------
            # Manual mode: use the pblock strings already set on each
            # NodeContainer (existing behaviour).
            # ----------------------------------------------------------------
            for sdp_name, pblock in pr_sdp_pblock_info:
                pblock_name = f"pblock_Hier_{sdp_name}"
                cell_path = f"top_i/Hier_{sdp_name}"
                pr_config.append(f"create_pblock {pblock_name}")
                pr_config.append(
                    f"add_cells_to_pblock [get_pblocks {pblock_name}] [get_cells {cell_path}]"
                )
                pr_config.append(f"resize_pblock [get_pblocks {pblock_name}] -add {{{pblock}}}")
                pr_config.append(f"set_property SNAPPING_MODE ON [get_pblocks {pblock_name}]")

        pr_config.append("save_constraints -force")
        pr_config.append("close_design")

        for body_id in range(num_bodies):
            run_name = "impl_1" if body_id == 0 else f"impl_body_{body_id}"
            pr_config.append(
                f"set_property STEPS.WRITE_BITSTREAM.ARGS.BIN_FILE true [get_runs {run_name}]"
            )

        pr_config.append("launch_runs impl_1 -to_step write_bitstream -jobs 4")
        pr_config.append("wait_on_run impl_1")

        if all_empty:
            # Auto mode: query post-implementation utilisation and write the JSON report.
            _pr_report_path = str(
                Path(model.get_metadata_prop("vivado_pynq_proj")) / "pr_region_resources.json"
            )
            model.set_metadata_prop("pr_region_resources_json", _pr_report_path)
            pr_config.append("open_run impl_1 -name impl_1")
            pr_config.append(
                "write_pr_resource_report "
                f"{{{' '.join(cell_names)}}} {{{' '.join(pblock_names)}}} "
                f"{{{_pr_report_path}}}"
            )
            pr_config.append("close_design")

        for body_id in range(1, num_bodies):
            impl_run = f"impl_body_{body_id}"
            pr_config.append(f"launch_runs {impl_run} -to_step write_bitstream -jobs 4")
            pr_config.append(f"wait_on_run {impl_run}")

        pr_config.append(
            "set partial_bs_dir "
            "[file join [get_property DIRECTORY [current_project]] partial_bitstreams]"
        )
        pr_config.append("file mkdir $partial_bs_dir")
        for body_id in range(num_bodies):
            impl_run = "impl_1" if body_id == 0 else f"impl_body_{body_id}"
            pr_config.append(
                "file copy -force "
                f"[file join [get_property DIRECTORY [get_runs {impl_run}]] top_wrapper.bit] "
                f"[file join $partial_bs_dir config_{body_id}.bit]"
            )
            for sdp_name in pr_sdp_names:
                if body_id == 0:
                    partial_bit_name = f"top_i_Hier_{sdp_name}_Hier_{sdp_name}_inst_0_partial.bit"
                else:
                    partial_bit_name = (
                        f"top_i_Hier_{sdp_name}_Hier_{sdp_name}_{body_id}_inst_0_partial.bit"
                    )
                pr_config.append(
                    "file copy -force "
                    f"[file join [get_property DIRECTORY [get_runs {impl_run}]] "
                    f"{partial_bit_name}] "
                    f"[file join $partial_bs_dir partial_{sdp_name}_{body_id}.bit]"
                )
                partial_bin_name = partial_bit_name.replace(".bit", ".bin")
                pr_config.append(
                    "file copy -force "
                    f"[file join [get_property DIRECTORY [get_runs {impl_run}]] "
                    f"{partial_bin_name}] "
                    f"[file join $partial_bs_dir partial_{sdp_name}_{body_id}.bin]"
                )
                pr_config.append(
                    "dfx_controller_v1_0::format_bin_for_icap "
                    "-bs 1 "
                    f"-i [file join $partial_bs_dir partial_{sdp_name}_{body_id}.bin] "
                    f"-o [file join $partial_bs_dir partial_{sdp_name}_{body_id}_icap.bin]"
                )

        pr_config.append("set pr_flow 1")
        pr_config.append("save_bd_design")
        # Re-validate now that all wrapper connections are in place
        pr_config.append("validate_bd_design")
        pr_config = "\n".join(pr_config) + "\n"
        return pr_config


class ZynqBuild(Transformation):
    """Best-effort attempt at building the accelerator for Zynq.
    It assumes the model has only fpgadataflow nodes.
    """

    def __init__(
        self,
        platform: str,
        period_ns: float,
        enable_debug: bool = False,
        enable_instrumentation: bool = False,
        instrumentation_no_dma: bool = False,
        instrumentation_avg_n: int = 64,
        live_fifo_sizing: bool = False,
        partition_model_dir: Path | str | None = None,
    ) -> None:
        """Initialize ZynqBuild with platform and build settings."""
        super().__init__()
        self.fpga_part = pynq_part_map[platform]
        self.axi_port_width = pynq_native_port_width[platform]
        self.period_ns = period_ns
        self.platform = platform
        self.enable_debug = enable_debug
        self.enable_instrumentation = enable_instrumentation
        self.instrumentation_no_dma = instrumentation_no_dma
        self.instrumentation_avg_n = instrumentation_avg_n
        self.live_fifo_sizing = live_fifo_sizing
        self.partition_model_dir = (
            str(partition_model_dir) if partition_model_dir is not None else None
        )

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, Literal[False]]:
        """Apply the ZynqBuild transformation to create a complete Zynq accelerator."""
        model = model.transform(InferDataLayouts())
        # prepare at global level, then break up into kernels
        enable_finn_switch = (
            self.enable_instrumentation
            and (not self.instrumentation_no_dma)
            and (not self.live_fifo_sizing)
        )
        if self.enable_instrumentation:
            if self.instrumentation_no_dma is True or self.live_fifo_sizing is True:
                prep_transforms = [
                    GenerateInstrumentationIP(
                        self.fpga_part, self.period_ns, self.instrumentation_avg_n
                    ),
                    Floorplan(),
                    CreateDataflowPartition(partition_model_dir=self.partition_model_dir),
                ]
            else:
                # DMA & Instrumentation Wrapper Case
                prep_transforms = [
                    GenerateInstrumentationIP(
                        self.fpga_part, self.period_ns, self.instrumentation_avg_n
                    ),
                    InsertIODMA(self.axi_port_width),
                    InsertDWC(),
                    SpecializeLayers(self.fpga_part),
                    Floorplan(),
                    CreateDataflowPartition(partition_model_dir=self.partition_model_dir),
                ]
        else:
            prep_transforms = [
                InsertIODMA(self.axi_port_width),
                InsertDWC(),
                SpecializeLayers(self.fpga_part),
                Floorplan(),
                CreateDataflowPartition(partition_model_dir=self.partition_model_dir),
            ]
        for trn in prep_transforms:
            model = model.transform(trn)
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())
        # Build each kernel individually (in parallel)
        sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
        worker_args = [
            (
                sdp_node.name,
                getCustomOp(sdp_node).get_nodeattr("model"),
                self.fpga_part,
                self.period_ns,
                self.enable_instrumentation,
            )
            for sdp_node in sdp_nodes
        ]
        num_workers = get_num_default_workers()
        if num_workers == 0:
            num_workers = mp.cpu_count()
        num_workers = min(num_workers, len(worker_args))
        log.info(f"Building {len(worker_args)} SDP kernels with {num_workers} workers...")
        if num_workers > 1:
            with mp.Pool(num_workers) as pool:
                pool.map(_build_sdp_kernel, worker_args, chunksize=1)
        else:
            for args in worker_args:
                _build_sdp_kernel(args)
        # Assemble design from IPs
        model = model.transform(
            MakeZYNQProject(
                self.platform,
                self.period_ns,
                enable_debug=self.enable_debug,
                enable_finn_switch=enable_finn_switch,
                live_fifo_sizing=self.live_fifo_sizing,
            )
        )

        # set platform attribute for correct remote execution
        model.set_metadata_prop("platform", "zynq-iodma")

        return (model, False)

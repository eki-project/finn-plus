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
from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveUnusedTensors,
)

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    FpgaMemoryType,
    MFCommunicationKernel,
    MFTopology,
    VitisOptStrategy,
)
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.floorplan import Floorplan
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.multifpga_network import AuroraNetworkMetadata
from finn.transformation.fpgadataflow.multifpga_utils import get_device_id
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import make_build_dir
from finn.util.deps import get_deps_path
from finn.util.logging import log

from . import templates


def _check_vitis_envvars():
    assert "XILINX_VITIS" in os.environ, "XILINX_VITIS must be set for Vitis"
    assert "PLATFORM_REPO_PATHS" in os.environ, "PLATFORM_REPO_PATHS must be set for Vitis"
    assert (
        "XILINX_XRT" in os.environ
    ), "XILINX_XRT must be set for Vitis, ensure the XRT env is sourced"


class CreateVitisXO(Transformation):
    """Create a Vitis object file from a stitched FINN ip.

    Outcome if successful: sets the vitis_xo attribute in the ONNX
    ModelProto's metadata_props field with the name of the object file as value.
    The object file can be found under the ip subdirectory.
    """

    def __init__(self, ip_name="finn_design"):
        super().__init__()
        self.ip_name = ip_name

    def apply(self, model):
        _check_vitis_envvars()
        vivado_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
        stitched_ip_dir = vivado_proj_dir + "/ip"
        interfaces = json.loads(model.get_metadata_prop("vivado_stitch_ifnames"))
        args_string = []
        arg_id = 0
        # NOTE: this assumes the graph is Vitis-compatible: max one axi lite interface
        # developed from instructions in UG1393 (v2019.2) and package_xo documentation
        # package_xo is responsible for generating the kernel xml
        assert len(interfaces["axilite"]) <= 1, "CreateVitisXO supports max 1 AXI lite interface"
        axilite_intf_name = None
        if len(interfaces["axilite"]) == 1:
            axilite_intf_name = interfaces["axilite"][0]
            if len(interfaces["aximm"]) > 0:
                args_string.append(
                    "{addr:1:%s:%s:0x8:0x10:ap_uint&lt;%s>*:0}"
                    % (
                        str(arg_id),
                        interfaces["aximm"][0][0],
                        str(interfaces["aximm"][0][1]),
                    )
                )
                arg_id += 1
                args_string.append(
                    "{numReps:0:%s:%s:0x4:0x1C:uint:0}" % (str(arg_id), axilite_intf_name)
                )
                arg_id += 1
            else:
                args_string.append(
                    "{numReps:0:%s:%s:0x4:0x10:uint:0}" % (str(arg_id), axilite_intf_name)
                )
                arg_id += 1
        for intf in interfaces["s_axis"] + interfaces["m_axis"]:
            stream_width = intf[1]
            stream_name = intf[0]
            args_string.append(
                "{%s:4:%s:%s:0x0:0x0:ap_uint&lt;%s>:0}"
                % (stream_name, str(arg_id), stream_name, str(stream_width))
            )
            arg_id += 1

        # save kernel xml then run package_xo
        xo_name = self.ip_name + ".xo"
        xo_path = vivado_proj_dir + "/" + xo_name
        model.set_metadata_prop("vitis_xo", xo_path)

        # generate the package_xo command in a tcl script
        package_xo_string = "package_xo -force -xo_path %s -kernel_name %s -ip_directory %s" % (
            xo_path,
            self.ip_name,
            stitched_ip_dir,
        )
        for arg in args_string:
            package_xo_string += " -kernel_xml_args " + arg
        with open(vivado_proj_dir + "/gen_xo.tcl", "w") as f:
            f.write(package_xo_string)

        # create a shell script and call Vivado
        package_xo_sh = vivado_proj_dir + "/gen_xo.sh"
        working_dir = os.environ["PWD"]
        with open(package_xo_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(vivado_proj_dir))
            f.write("vivado -mode batch -source gen_xo.tcl\n")
            f.write("cd {}\n".format(working_dir))
        bash_command = ["bash", package_xo_sh]
        process_compile = subprocess.Popen(
            bash_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        _, stderr_data = process_compile.communicate()
        stderr_stripped = stderr_data.decode().strip()
        if stderr_stripped != "" and stderr_stripped is not None:
            log.critical(stderr_stripped)  # Decode bytes and log as critical
        assert os.path.isfile(xo_path), (
            "Vitis .xo file not created, check logs under %s" % vivado_proj_dir
        )
        return (model, False)


class BuildAllXOs(Transformation):
    """Built from the former VitisBuild transformation. Seperated out for more modular use.
    Packages all StreamingDataflowPartitions into XO files, saves their path as a node attribute.
    These can then be used for linking. Also works with Multi-FPGA (_currently_ (!) assigns IODMA
    to the first and last SDP)"""

    # TODO: Rather pass the arguments as needed, not the entire config.
    def __init__(self, cfg: DataflowBuildConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        is_mulitfpga = False
        if self.cfg.partitioning_configuration is not None:
            if model.get_metadata_prop("is_multifpga") != "True":
                log.critical(
                    "A Multi-FPGA partitioning configuration was given, but the "
                    'model metadata prop "is_multifpga" is not set to true. '
                    "Proceeding with the single FPGA case."
                )
            else:
                is_mulitfpga = True
        _check_vitis_envvars()
        if is_mulitfpga:
            # Confirm the shape of the SDP graph (one line, one input, one output)
            bad_shape_found = False
            for i, node in enumerate(model.graph.node):
                if node.op_type != "StreamingDataflowPartition":
                    bad_shape_found = True
                    log.error(
                        f"Node {node.name} is not a StreamingDataflowPartition. "
                        f"Did you run all necessary steps first?"
                    )
                pre, suc = model.find_direct_predecessors(node), model.find_direct_successors(node)
                if i == 0 and pre is not None:
                    bad_shape_found = True
                    log.critical("Node 0 in the graph has unexpected predecessors!")
                elif i == len(model.graph.node) - 1 and suc is not None:
                    bad_shape_found = True
                    log.critical("The last node has unexpected successors!")
                if i not in [0, len(model.graph.node) - 1]:
                    if pre is None or len(pre) != 1:
                        bad_shape_found = True
                        log.critical(
                            f"Node {i} ({node.name}) has more or "
                            f"less than 1 predecessor. Expected exactly 1!"
                        )
                    if suc is None or len(suc) != 1:
                        bad_shape_found = True
                        log.critical(
                            f"Node {i} ({node.name}) has more or "
                            f"less than 1 successor. Expected exactly 1!"
                        )
            if bad_shape_found:
                raise Exception(
                    "Bad graph found. Cannot produce XOs. " "Please check the logs for errors!"
                )

            # Insert IODMAs
            log.debug("Inserting IODMAs")
            iodma_transforms = [
                GiveUniqueNodeNames(),
                SpecializeLayers(self.cfg.fpga_part),
                PrepareIP(self.cfg.fpga_part, self.cfg.synth_clk_period_ns),
                HLSSynthIP(),
            ]

            # Prepare
            first_node_path = getCustomOp(model.graph.node[0]).get_nodeattr("model")
            last_node_path = getCustomOp(model.graph.node[-1]).get_nodeattr("model")
            first_node_model = ModelWrapper(first_node_path)
            last_node_model = ModelWrapper(last_node_path)

            # Input
            first_node_model = first_node_model.transform(
                InsertIODMA(512, insert_input=True, insert_output=False)
            )
            for transform in iodma_transforms:
                first_node_model = first_node_model.transform(transform)

            # Output
            last_node_model = last_node_model.transform(
                InsertIODMA(512, insert_input=False, insert_output=True)
            )
            for transform in iodma_transforms:
                last_node_model = last_node_model.transform(transform)

            # Save changes
            first_node_model.save(first_node_path)
            last_node_model.save(last_node_path)

            # Do all other necessary steps on all SDPs
            for sdp_node in model.graph.node:
                log.debug(f"Creating XO for SDP: {sdp_node.name}")
                submodel_transforms = [
                    InsertDWC(),
                    GiveUniqueNodeNames(),
                    GiveReadableTensorNames(),
                    SpecializeLayers(self.cfg.fpga_part),
                    GiveUniqueNodeNames(),
                    GiveReadableTensorNames(),
                    InsertFIFO(),
                    SpecializeLayers(self.cfg.fpga_part),
                    RemoveUnusedTensors(),
                    GiveUniqueNodeNames(prefix=sdp_node.name + "_"),
                    PrepareIP(self.cfg.fpga_part, self.cfg.synth_clk_period_ns),
                    HLSSynthIP(),
                    CreateStitchedIP(
                        self.cfg.fpga_part, self.cfg.synth_clk_period_ns, sdp_node.name, vitis=True
                    ),
                    CreateVitisXO(sdp_node.name),
                ]
                submodel_path = getCustomOp(sdp_node).get_nodeattr("model")
                submodel = ModelWrapper(submodel_path)
                for transform in submodel_transforms:
                    submodel = submodel.transform(transform)
                submodel.set_metadata_prop("platform", "alveo")
                submodel.save(submodel_path)
        else:
            # TODO: Move over here from VitisBuild
            raise NotImplementedError()
        return model, False


class InvalidVitisLinkConfigError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class VitisLinkConfiguration:
    """Manages XO files, CU instantiations, stream connections,
    port connections, Vivado props, etc.
    It can output a linking configuration to pass to v++ and
    create a shell script to run it. Tries to be as strict and careful as possible,
    and depending on the issue raises an Exception, logs an error or warning
    or continues silently."""

    def __init__(self, platform: str, optimization_level: str, f_mhz: int) -> None:
        self.cu: list[str] = []
        self.nk: list[tuple[str, str]] = []
        self.sc: dict[str, list[str]] = {}
        self.sp: dict[str, str] = {}
        self.xo: list[Path] = []
        self.connects: list[tuple[str, str]] = []
        self.vivado_section: str = "[vivado]\n"
        self.connectivity_section: str = ""
        self.platform: str = platform
        self.optimization_level: str = optimization_level
        self.f_mhz: int = f_mhz

    def add_cu(self, kernel_name: str, cu_name: str) -> None:
        """Add a compute unit (instance of a kernel)"""
        if cu_name in self.cu:
            kern = next(kname for kname, cname in self.nk if cname == cu_name)
            raise InvalidVitisLinkConfigError(
                f"Tried creating CU {cu_name}, but a CU of this "
                f"name of kernel {kern} already exists!"
            )
        self.cu.append(cu_name)
        self.nk.append((kernel_name, cu_name))

    def add_sc(self, cu_sender: str, cu_receiver: str) -> None:
        """Add a Streaming Connection between two CUs:
        >>> lc = VitisLinkConfiguration("", "", 100)
        >>> lc.add_cu("A", "a")
        >>> lc.add_cu("B", "b")
        >>> lc.add_sc("a.out", "b.in")
        >>> lc.sc["a.out"]
        ['b.in']
        """
        # Check formatting
        for cu in [cu_sender, cu_receiver]:
            splits = cu.split(".")
            if len(splits) != 2:
                raise InvalidVitisLinkConfigError(
                    f"{cu} is incorrectly formatted. Required "
                    f"syntax to add a streaming connection from CU "
                    f'a on port out is "a.out".'
                )

        # Yield warning if the direction seems wrong
        sender_port = cu_sender.split(".")[1]
        receiver_port = cu_receiver.split(".")[1]
        if sender_port.lower() in ["s_axis", "in"] or receiver_port.lower() in ["m_axis", "out"]:
            log.error(
                f"Adding connection sc={cu_sender}:{cu_receiver}. The port "
                "names suggest that the order of sender and receiver might be "
                "swapped. Proceeding now."
            )

        # Add the connection
        if cu_sender not in self.sc.keys():
            self.sc[cu_sender] = []
        self.sc[cu_sender].append(cu_receiver)

    def add_sp(self, cu_port_name: str, mem_type: str) -> None:
        self.sp[cu_port_name] = mem_type

    def add_connect(self, a: str, b: str) -> None:
        """Add a connect assignment. Not to be confused with stream_connect (sc)!"""
        self.connects.append((a, b))

    def add_vivado_line(self, line: str) -> None:
        self.vivado_section += line

    def add_xo(self, xo_file: Path) -> None:
        """Add an XO file. This will emit an error if the XO file is not found, but it will
        NOT raise an exception."""
        if not xo_file.exists():
            log.error(
                f"Tried adding non-existing file {xo_file.absolute()}. "
                f"Continuing in case this is on purpose."
            )
        self.xo.append(xo_file)

    def add_connectivity(self, txt: str) -> None:
        """Add further lines to the connectivity section. For example to assign clocks or ports"""
        self.connectivity_section += txt

    def get_config_validation_errors(self) -> None | list[InvalidVitisLinkConfigError]:
        """Check the configuration and if errors are found, return them"""
        errors = []
        # All CUs in SCs exist and CU ports are correctly formatted
        for cu_sender, receivers in self.sc.items():
            for cu_receiver in receivers:
                sender_split = cu_sender.split(".")
                if len(sender_split) != 2:
                    errors.append(
                        InvalidVitisLinkConfigError(
                            f"SC {cu_sender}:{cu_receiver} "
                            f"incorrectly formatted. "
                            "Use the syntax CU.PORT"
                        )
                    )
                sender_name = sender_split[0]
                if sender_name not in self.cu:
                    errors.append(
                        InvalidVitisLinkConfigError(
                            f"SC {cu_sender}:{cu_receiver} uses the unknown CU {sender_name}"
                        )
                    )
                receiver_split = cu_receiver.split(".")
                if len(receiver_split) != 2:
                    errors.append(
                        InvalidVitisLinkConfigError(
                            f"SC {cu_sender}:{cu_receiver} "
                            f"incorrectly formatted. "
                            "Use the syntax CU.PORT"
                        )
                    )
                receiver_name = receiver_split[0]
                if receiver_name not in self.cu:
                    errors.append(
                        InvalidVitisLinkConfigError(
                            f"SC {cu_sender}:"
                            f"{cu_receiver} uses the unknown "
                            f"CU {receiver_name}"
                        )
                    )
        # No two same named CUs
        if len(set(self.cu)) != len(self.cu):
            errors.append(
                InvalidVitisLinkConfigError(
                    "It seems that there are one or more CUs with the same name!"
                )
            )
        for kernel, cu in self.nk:
            for kernel2, cu2 in self.nk:
                if cu == cu2 and kernel != kernel2:
                    errors.append(
                        InvalidVitisLinkConfigError(
                            f"There are 2 or more CUs named {cu} "
                            f"from different kernels ({kernel} "
                            f"and {kernel2})"
                        )
                    )
        if len(errors) > 0:
            return errors
        return None

    def generate_config(self, path: Path) -> None:
        """Write the complete config to the given path. Raises an error if the
        config is invalid"""
        errors = self.get_config_validation_errors()
        if errors is not None:
            for err in errors:
                log.error(f"{path}: {err}")
            raise errors[0]
        with path.open("w+") as f:
            f.write("[connectivity]\n")
            for kernel_name, cu_name in self.nk:
                f.write(f"nk={kernel_name}:1:{cu_name}\n")

            # origin_cu and target_cu already require the ports already being in the str
            for origin_cu in self.sc.keys():
                for target_cu in self.sc[origin_cu]:
                    f.write(f"sc={origin_cu}:{target_cu}\n")

            for sp_cu, sp_mem in self.sp.items():
                f.write(f"sp={sp_cu}:{sp_mem}\n")

            for a, b in self.connects:
                f.write(f"connect={a}:{b}\n")

            if self.connectivity_section != "":
                f.write(self.connectivity_section + "\n")

            f.write(self.vivado_section)

    def generate_run_script(self, config_path: Path, target: Path | None = None) -> None:
        """Generate a shell script to start v++ with the correct parameters.
        Produces the shell script next to the path of the config file
        unless a path is specified"""
        xo_string = " ".join([str(xo) for xo in self.xo])
        if not config_path.exists():
            log.error(
                f"Writing compilation / v++ script for non-existing configuration "
                f"in {config_path.absolute()}. Continuing in case this is on purpose."
            )
        runner_path = config_path.parent / "run_vitis_link.sh"
        if target is not None:
            runner_path = target
        with runner_path.open("w+") as f:
            f.write("#!/bin/bash\n")
            f.write(
                f"v++ --target hw --platform {self.platform} --link {xo_string} "
                f"--config {config_path} --optimize {self.optimization_level} "
                f"--report_level estimate --save-temps --kernel_frequency {self.f_mhz}"
            )


# TODO: Replace with generic variant, replace asserts with exceptions and logging
# Everything hardcoded in this transform will be made configurable / automatic
class AuroraChainVitisLink(Transformation):
    """Simple MultiFPGA Link transform. Will be replaced with the ported improved
    approach soon. Only for testing purposes.
    Valid configuration: AURORA Kernels + CHAIN topology, on a U280"""

    def __init__(self, cfg: DataflowBuildConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        # Aurora + Chain config enables the following assumptions:
        # - Each device is only visited once
        # - Every SDP is on a different device
        # - Aurora count per device

        # Testing related assertions
        assert self.cfg.board is not None
        assert self.cfg.partitioning_configuration is not None
        assert self.cfg.partitioning_configuration.topology == MFTopology.CHAIN
        assert (
            self.cfg.partitioning_configuration.communication_kernel == MFCommunicationKernel.AURORA
        )
        assert self.cfg.vitis_opt_strategy is not None

        # Build dummy kernels if necessary. Used to connect open
        # ports while Aurora is in Duplex mode
        dummy_kernel_dir = get_deps_path() / "vitis_dummy_kernel"
        rx_dummy = dummy_kernel_dir / "rx_dummy_kernel.xo"
        tx_dummy = dummy_kernel_dir / "tx_dummy_kernel.xo"
        if not rx_dummy.exists() or not tx_dummy.exists():
            subprocess.run(["make"], cwd=dummy_kernel_dir, stdout=subprocess.DEVNULL)

        # Load metadata
        metadata = AuroraNetworkMetadata(model)
        assert metadata.table != {}
        aurora_storage = model.get_metadata_prop("aurora_storage")
        assert aurora_storage is not None
        aurora_storage = Path(aurora_storage)

        # Configs
        configs: dict[int, VitisLinkConfiguration] = {}
        for i, sdp in enumerate(model.graph.node):
            this_device = get_device_id(sdp)
            assert this_device is not None

            # Create config
            configs[this_device] = VitisLinkConfiguration(
                self.cfg._resolve_vitis_platform(),  # noqa
                self.cfg.vitis_opt_strategy.value,
                round(1000 / self.cfg.synth_clk_period_ns),
            )
            this_config = configs[this_device]
            submodel = ModelWrapper(getCustomOp(sdp).get_nodeattr("model"))

            # Get aurora XOs and kernel names
            aurora_kernels = metadata.get_aurora_kernels(this_device)
            sdp_xo = submodel.get_metadata_prop("vitis_xo")
            assert sdp_xo is not None
            xos = [Path(sdp_xo)] + [
                aurora_storage / Path(kernel_name + ".xo") for kernel_name in aurora_kernels
            ]
            if i == 0:
                xos.append(rx_dummy)
                this_config.add_cu("rx_dummy_kernel", "rx_dummy_kernel")
            elif i == len(configs) - 1:
                xos.append(tx_dummy)
                this_config.add_cu("tx_dummy_kernel", "tx_dummy_kernel")
            else:
                xos += [rx_dummy, tx_dummy]
                this_config.add_cu("rx_dummy_kernel", "rx_dummy_kernel")
                this_config.add_cu("tx_dummy_kernel", "tx_dummy_kernel")
            for xo in xos:
                assert xo.exists()
                this_config.add_xo(xo)

            # Add CUs
            this_config.add_cu(sdp.name, sdp.name)
            this_config.add_cu("aurora_flow_0", "aurora_flow_0")
            this_config.add_connect("io_clk_qsfp0_refclkb_00", "aurora_flow_0/gt_refclk_0")
            this_config.add_connect("aurora_flow_0/gt_port", "io_gt_qsfp0_00")
            this_config.add_connect(
                "aurora_flow_0/init_clk", "ii_level0_wire/ulp_m_aclk_freerun_ref_00"
            )
            if i not in [0, len(configs) - 1]:
                this_config.add_cu("aurora_flow_1", "aurora_flow_1")
                this_config.add_connect("io_clk_qsfp1_refclkb_00", "aurora_flow_1/gt_refclk_1")
                this_config.add_connect("aurora_flow_1/gt_port", "io_gt_qsfp1_00")
                this_config.add_connect(
                    "aurora_flow_1/init_clk", "ii_level0_wire/ulp_m_aclk_freerun_ref_00"
                )

            # Add connections
            if i == 0:
                this_config.add_sc(sdp.name + ".m_axis_0", "aurora_flow_0.tx_axis")
                this_config.add_sc("aurora_flow_0.rx_axis", "rx_dummy_kernel.A")
            elif i == len(configs) - 1:
                this_config.add_sc("aurora_flow_0.rx_axis", sdp.name + ".s_axis_0")
                this_config.add_sc("tx_dummy_kernel.A", "aurora_flow_0.tx_axis")
            else:
                this_config.add_sc("aurora_flow_0.rx_axis", sdp.name + ".s_axis_0")
                this_config.add_sc(sdp.name + ".m_axis_0", "aurora_flow_1.tx_axis")
                this_config.add_sc("tx_dummy_kernel.A", "aurora_flow_0.tx_axis")
                this_config.add_sc("aurora_flow_1.rx_axis", "rx_dummy_kernel.A")

            if i in [0, len(configs) - 1]:
                this_config.add_sp(sdp.name + ".m_axi_gmem0", "HBM[0]")

            if self.cfg.vitis_opt_strategy == VitisOptStrategy.PERFORMANCE_BEST:
                this_config.add_vivado_line(
                    "[vivado]\n"
                    "prop=run.impl_1.STEPS.OPT_DESIGN.ARGS.DIRECTIVE=ExploreWithRemap\n"
                    "prop=run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=Explore\n"
                    "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=true\n"
                    "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=Explore\n"
                    "prop=run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=Explore\n"
                )
        # Start all the synthesis runs
        execute_synthesis_parallel(
            list(configs.values()), self.cfg.partitioning_configuration.parallel_synthesis_workers
        )
        return model, False


# TODO: Make this part of the new VitisLink
def execute_synthesis_parallel(configs: list[VitisLinkConfiguration], workers: int) -> None:
    """Execute the list of synthesis in parallel. Can be used for faster design space
    exploration or for Multi-FPGA applications.

    This creates the necessary temp dirs by itself also"""

    def run_link_config(config: VitisLinkConfiguration, index: int) -> None:
        link_dir = Path(make_build_dir(f"parallel_link{index}_"))
        config.generate_config(link_dir / "config.txt")
        config.generate_run_script(link_dir / "config.txt")
        subprocess.run("bash run_vitis_link.sh", shell=True, cwd=link_dir)

    with ThreadPoolExecutor(max_workers=workers) as tpe:
        tpe.map(run_link_config, configs, list(range(len(configs))))
    tpe.shutdown(wait=True)


class VitisLink(Transformation):
    """Create an XCLBIN with Vitis.

    Outcome if successful: sets the bitfile attribute in the ONNX
    ModelProto's metadata_props field with the XCLBIN full path as value.
    """

    def __init__(
        self,
        platform,
        f_mhz=200,
        strategy=VitisOptStrategy.PERFORMANCE,
        enable_debug=False,
        fpga_memory_type="default",
    ):
        super().__init__()
        self.platform = platform
        self.f_mhz = f_mhz
        self.strategy = strategy
        self.enable_debug = enable_debug
        self.fpga_memory_type = fpga_memory_type

    def apply(self, model):
        _check_vitis_envvars()
        # create a config file and empty list of xo files
        config = ["[connectivity]"]
        object_files = []
        idma_idx = 0
        odma_idx = 0
        mem_idx = 0
        instance_names = {}
        for node in model.graph.node:
            assert node.op_type == "StreamingDataflowPartition", "Invalid link graph"
            sdp_node = getCustomOp(node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            kernel_xo = kernel_model.get_metadata_prop("vitis_xo")
            object_files.append(kernel_xo)
            # gather info on connectivity
            # assume each node connected to outputs/inputs is DMA:
            # has axis, aximm and axilite
            # everything else is axis-only
            # assume only one connection from each ip to the next
            if len(node.input) == 0:
                producer = None
            else:
                producer = model.find_producer(node.input[0])
            consumer = model.find_consumers(node.output[0])
            # define kernel instances
            # name kernels connected to graph inputs as idmaxx
            # name kernels connected to graph inputs as odmaxx
            # TODO not a good way of checking for external in/out
            # check top-level in/out list instead
            if producer is None:
                instance_names[node.name] = "idma" + str(idma_idx)
                config.append("nk=%s:1:%s" % (node.name, instance_names[node.name]))
                idma_idx += 1
            elif consumer == []:
                instance_names[node.name] = "odma" + str(odma_idx)
                config.append("nk=%s:1:%s" % (node.name, instance_names[node.name]))
                odma_idx += 1
            else:
                instance_names[node.name] = node.name
                config.append("nk=%s:1:%s" % (node.name, instance_names[node.name]))
            sdp_node.set_nodeattr("instance_name", instance_names[node.name])
            # explicitly assign SLRs if the slr attribute is not -1
            node_slr = sdp_node.get_nodeattr("slr")
            if node_slr != -1:
                config.append("slr=%s:SLR%d" % (instance_names[node.name], node_slr))
            # assign memory banks
            if producer is None or consumer is None or consumer == []:
                node_mem_port = sdp_node.get_nodeattr("mem_port")
                if node_mem_port == "":
                    if self.fpga_memory_type == FpgaMemoryType.DEFAULT:
                        # configure good defaults based on board
                        if (
                            "u50" in self.platform
                            or "u280" in self.platform
                            or "u55c" in self.platform
                        ):
                            # Use HBM where available (also U50 does not have DDR)
                            mem_type = "HBM"
                            mem_idx = 0
                        elif "u200" in self.platform:
                            # Use DDR controller in static region of U200
                            mem_type = "DDR"
                            mem_idx = 1
                        elif "u250" in self.platform:
                            # Use DDR controller on the node's SLR if set, otherwise 0
                            mem_type = "DDR"
                            if node_slr == -1:
                                mem_idx = 0
                            else:
                                mem_idx = node_slr
                        else:
                            mem_type = "DDR"
                            mem_idx = 1
                    elif self.fpga_memory_type == FpgaMemoryType.HOST_MEM:
                        mem_type = "HOST"
                        mem_idx = 0
                    else:
                        raise RuntimeError(
                            "Unknown fpga memory type: "
                            + str(self.fpga_memory_type)
                            + ". Aborting!"
                        )
                    node_mem_port = "%s[%d]" % (mem_type, mem_idx)
                config.append("sp=%s.m_axi_gmem0:%s" % (instance_names[node.name], node_mem_port))
            # connect streams
            if producer is not None:
                for i in range(len(node.input)):
                    producer = model.find_producer(node.input[i])
                    if producer is not None:
                        j = list(producer.output).index(node.input[i])
                        config.append(
                            "stream_connect=%s.m_axis_%d:%s.s_axis_%d"
                            % (
                                instance_names[producer.name],
                                j,
                                instance_names[node.name],
                                i,
                            )
                        )

        # create a temporary folder for the project
        link_dir = make_build_dir(prefix="vitis_link_proj_")
        model.set_metadata_prop("vitis_link_proj", link_dir)

        # add Vivado physopt directives if desired
        if self.strategy == VitisOptStrategy.PERFORMANCE_BEST:
            config.append("[vivado]")
            config.append("prop=run.impl_1.STEPS.OPT_DESIGN.ARGS.DIRECTIVE=ExploreWithRemap")
            config.append("prop=run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=Explore")
            config.append("prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=true")
            config.append("prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=Explore")
            config.append("prop=run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=Explore")

        config = "\n".join(config) + "\n"
        with open(link_dir + "/config.txt", "w") as f:
            f.write(config)

        # create tcl script to generate resource report in XML format
        gen_rep_xml = templates.vitis_gen_xml_report_tcl_template
        gen_rep_xml = gen_rep_xml.replace("$VITIS_PROJ_PATH$", link_dir)
        with open(link_dir + "/gen_report_xml.tcl", "w") as f:
            f.write(gen_rep_xml)

        debug_commands = []
        if self.enable_debug:
            for inst in list(instance_names.values()):
                debug_commands.append("--dk chipscope:%s" % inst)

        # create a shell script and call Vitis
        script = link_dir + "/run_vitis_link.sh"
        working_dir = os.environ["PWD"]
        with open(script, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(link_dir))
            f.write(
                "v++ -t hw --platform %s --link %s"
                " --kernel_frequency %d --config config.txt --optimize %s"
                " --save-temps -R2 %s\n"
                % (
                    self.platform,
                    " ".join(object_files),
                    self.f_mhz,
                    self.strategy.value,
                    " ".join(debug_commands),
                )
            )
            f.write("cd {}\n".format(working_dir))
        bash_command = ["bash", script]
        process_compile = subprocess.Popen(
            bash_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        _, stderr_data = process_compile.communicate()
        stderr_stripped = stderr_data.decode().strip()
        if stderr_stripped != "" and stderr_stripped is not None:
            log.critical(stderr_stripped)  # Decode bytes and log as critical
        # TODO rename xclbin appropriately here?
        xclbin = link_dir + "/a.xclbin"
        assert os.path.isfile(xclbin), (
            "Vitis .xclbin file not created, check logs under %s" % link_dir
        )
        model.set_metadata_prop("bitfile", xclbin)

        # run Vivado to gen xml report
        gen_rep_xml_sh = link_dir + "/gen_report_xml.sh"
        working_dir = os.environ["PWD"]
        with open(gen_rep_xml_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(link_dir))
            f.write("vivado -mode batch -source %s\n" % (link_dir + "/gen_report_xml.tcl"))
            f.write("cd {}\n".format(working_dir))
        bash_command = ["bash", gen_rep_xml_sh]
        process_genxml = subprocess.Popen(
            bash_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        _, stderr_data = process_genxml.communicate()
        stderr_stripped = stderr_data.decode().strip()
        if stderr_stripped != "" and stderr_stripped is not None:
            log.critical(stderr_stripped)  # Decode bytes and log as critical
        # filename for the synth utilization report
        synth_report_filename = link_dir + "/synth_report.xml"
        model.set_metadata_prop("vivado_synth_rpt", synth_report_filename)
        return (model, False)


class VitisBuild(Transformation):
    """Best-effort attempt at building the accelerator with Vitis.
    It assumes the model has only fpgadataflow nodes

    :parameter fpga_part: string identifying the target FPGA
    :parameter period_ns: target clock period
    :parameter platform: target Alveo platform, one of ["U50", "U200", "U250", "U280"]
    :parameter strategy: Vitis optimization strategy
    :parameter enable_debug: add Chipscope to all AXI interfaces
    :parameter floorplan_file: path to a JSON containing a dictionary with
        SLR assignments for each node in the ONNX graph.
        Must be parse-able by the ApplyConfig transform.
    :parameter enable_link: enable linking kernels (.xo files),
        otherwise just synthesize them independently.
    :parameter fpga_memory_type: Specify whether Host or FPGA memory such as DDR/HBM should be used
    """

    def __init__(
        self,
        fpga_part,
        period_ns,
        platform,
        strategy=VitisOptStrategy.PERFORMANCE,
        enable_debug=False,
        floorplan_file=None,
        enable_link=True,
        partition_model_dir=None,
        fpga_memory_type=FpgaMemoryType.DEFAULT,
    ):
        super().__init__()
        self.fpga_part = fpga_part
        self.period_ns = period_ns
        self.platform = platform
        self.strategy = strategy
        self.enable_debug = enable_debug
        self.floorplan_file = floorplan_file
        self.enable_link = enable_link
        self.partition_model_dir = partition_model_dir
        self.fpga_memory_type = fpga_memory_type

    def apply(self, model):
        _check_vitis_envvars()
        # prepare at global level, then break up into kernels
        prep_transforms = [InsertIODMA(512), InsertDWC(), SpecializeLayers(self.fpga_part)]
        for trn in prep_transforms:
            model = model.transform(trn)
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())

        model = model.transform(Floorplan(floorplan=self.floorplan_file))

        model = model.transform(
            CreateDataflowPartition(partition_model_dir=self.partition_model_dir)
        )
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())

        # Build each kernel individually
        sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
        for sdp_node in sdp_nodes:
            prefix = sdp_node.name + "_"
            sdp_node = getCustomOp(sdp_node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            kernel_model = kernel_model.transform(InsertFIFO())
            kernel_model = kernel_model.transform(SpecializeLayers(self.fpga_part))
            kernel_model = kernel_model.transform(RemoveUnusedTensors())
            kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
            kernel_model.save(dataflow_model_filename)
            kernel_model = kernel_model.transform(PrepareIP(self.fpga_part, self.period_ns))
            kernel_model = kernel_model.transform(HLSSynthIP())
            kernel_model = kernel_model.transform(
                CreateStitchedIP(self.fpga_part, self.period_ns, sdp_node.onnx_node.name, True)
            )
            kernel_model = kernel_model.transform(CreateVitisXO(sdp_node.onnx_node.name))
            kernel_model.set_metadata_prop("platform", "alveo")
            kernel_model.save(dataflow_model_filename)
        # Assemble design from kernels
        if self.enable_link:
            model = model.transform(
                VitisLink(
                    self.platform,
                    round(1000 / self.period_ns),
                    strategy=self.strategy,
                    enable_debug=self.enable_debug,
                    fpga_memory_type=self.fpga_memory_type,
                )
            )
        # set platform attribute for correct remote execution
        model.set_metadata_prop("platform", "alveo")

        return (model, False)

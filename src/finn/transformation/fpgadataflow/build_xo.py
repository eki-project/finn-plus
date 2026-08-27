"""Build XO files for FINN IP core(s)."""

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
from pathlib import Path
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveReadableTensorNames, RemoveUnusedTensors
from subprocess import CalledProcessError
from typing import TYPE_CHECKING

from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.qonnx.give_unique_node_names_recursive import GiveUniqueNodeNamesRecursive
from finn.util.basic import launch_process_helper
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.fpgadataflow import check_all_sdp_nodes, get_submodel
from finn.util.logging import log
from finn.util.vivado import check_vitis_envvars

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


class CreateVitisXO(Transformation):
    """Create a Vitis object file from a stitched FINN IP.

    Outcome if successful: sets the vitis_xo attribute in the ONNX
    ModelProto's metadata_props field with the name of the object file as value.
    The object file can be found under the ip subdirectory.
    """

    # TODO: Make parallelizable. Currently not possible, since the models metadata props
    # are changed before returning it, making concurrent execution impossible.

    def __init__(self, ip_name: str = "finn_design") -> None:
        """Initialize CreateVitisXO transformation."""
        super().__init__()
        self.ip_name = ip_name

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply CreateVitisXO transformation to create Vitis object file."""
        check_vitis_envvars()

        # Locate the stitched IP
        vivado_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
        if vivado_proj_dir is None:
            raise FINNUserError(
                "Error while building xo: 'vivado_stitch_proj' was not set in the model."
            )
        vivado_proj_dir = Path(vivado_proj_dir)
        stitched_ip_dir = vivado_proj_dir / "ip"
        if not stitched_ip_dir.exists():
            raise FINNInternalError(f"Stitched IP directory does not exist: {stitched_ip_dir}")

        # Load the interface names
        ifnames = model.get_metadata_prop("vivado_stitch_ifnames")
        if ifnames is None:
            raise FINNInternalError("Error building xo: 'vivado_stitch_ifnames' was not set!")
        interfaces = json.loads(ifnames)

        # NOTE: this assumes the graph is Vitis-compatible: max one axi lite interface
        # developed from instructions in UG1393 (v2019.2) and package_xo documentation
        # package_xo is responsible for generating the kernel xml
        assert len(interfaces["axilite"]) <= 1, "CreateVitisXO supports max 1 AXI lite interface"
        if len(interfaces["axilite"]) > 1:
            raise FINNInternalError(
                f"Error building xo: cannot create vitis-compatible "
                f"xo because more than 1 AXI Lite interface was found. "
                f"Found: {len(interfaces['axilite'])} interfaces."
            )
        # Prepare
        args_string = []
        arg_id = 0
        axilite_intf_name = None

        # Build kernel XML arguments: {name:addressQualifier:id:port:size:offset:type:memSize}
        # (from UG1702, 2025.2)
        if len(interfaces["axilite"]) == 1:
            axilite_intf_name = interfaces["axilite"][0]
            if len(interfaces["aximm"]) > 0:
                # Address argument
                args_string.append(
                    f"{{addr:1:{arg_id}:{interfaces['aximm'][0][0]}"
                    f":0x8:0x10:ap_uint&lt;{interfaces['aximm'][0][1]}>*:0}}"
                )

                # NumReps argument
                arg_id += 1
                args_string.append(f"{{numReps:0:{arg_id}:{axilite_intf_name}:0x4:0x1C:uint:0}}")
                arg_id += 1
            else:
                # NumReps argument
                args_string.append(f"{{numReps:0:{arg_id}:{axilite_intf_name}:0x4:0x10:uint:0}}")
                arg_id += 1

        # IO AXI streams
        for intf in interfaces["s_axis"] + interfaces["m_axis"]:
            stream_width = intf[1]
            stream_name = intf[0]
            args_string.append(
                f"{{{stream_name}:4:{arg_id}:{stream_name}:0x0:0x0:"
                f"ap_uint&lt;{stream_width}>:0}}"
            )
            arg_id += 1

        # Save the xo location into the metadata prop, then run package_xo
        xo_path = vivado_proj_dir / (self.ip_name + ".xo")
        model.set_metadata_prop("vitis_xo", str(xo_path))

        # Generate the package_xo command in a tcl script
        package_xo_string = (
            f"package_xo -force -xo_path {xo_path} -kernel_name "
            f"{self.ip_name} -ip_directory {stitched_ip_dir} "
            + " ".join([f" -kernel_xml_args {arg}" for arg in args_string])
        )

        # Write the command into a Tcl script
        (vivado_proj_dir / "gen_xo.tcl").write_text(package_xo_string)

        # Create a shell script and call Vivado
        package_xo_sh = vivado_proj_dir / "gen_xo.sh"
        working_dir = Path.cwd()
        with package_xo_sh.open("w") as f:
            f.write("#!/bin/bash \n")
            f.write("set -e\n")
            f.write(f"cd {vivado_proj_dir}\n")
            f.write("vivado -mode batch -source gen_xo.tcl\n")
            f.write(f"cd {working_dir}\n")

        # Run the package command
        try:
            bash_command = ["bash", package_xo_sh]
            launch_process_helper(bash_command, print_stdout=False)
        except CalledProcessError as e:
            raise FINNUserError(
                f"An error ocurred while generating the XO file for "
                f"{self.ip_name}. Check {vivado_proj_dir} for further "
                f"details."
            ) from e
        if not xo_path.exists():
            raise FINNUserError(f"Vitis .xo file not created, check logs under {vivado_proj_dir}")
        return (model, False)


class BuildAllXOs(Transformation):
    """Build all XOs from SDPs in the graph using `CreateVitisXO`. Assigns IODMAs to nodes without
    successors or predecessors.
    """

    def __init__(self, fpga_part: str, synth_clk_period_ns: float) -> None:
        """Build all XOs."""
        super().__init__()
        self.fpga_part = fpga_part
        self.synth_clk_period_ns = synth_clk_period_ns

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Build all XOs."""
        check_vitis_envvars()

        # Confirm that the graph is only a line with (multiple) input and output nodes.
        # Every other graph at this point is invalid.
        # Every node must be an SDP node at this point.
        check_all_sdp_nodes(model)

        # Do all other necessary steps on all SDPs
        for i, sdp_node in enumerate(model.graph.node):
            log.info(f"Creating XO for SDP: {sdp_node.name} ({i}/{len(model.graph.node)})")
            submodel_transforms = [
                InsertDWC(),
                GiveUniqueNodeNamesRecursive(),
                GiveReadableTensorNames(),
                SpecializeLayers(self.fpga_part),
                GiveUniqueNodeNamesRecursive(),
                GiveReadableTensorNames(),
                InsertFIFO(),
                SpecializeLayers(self.fpga_part),
                RemoveUnusedTensors(),
                GiveUniqueNodeNamesRecursive(prefix=sdp_node.name),
                PrepareIP(self.fpga_part, self.synth_clk_period_ns),
                HLSSynthIP(),
                CreateStitchedIP(
                    self.fpga_part,
                    self.synth_clk_period_ns,
                    sdp_node.name,
                    vitis=True,
                ),
                CreateVitisXO(sdp_node.name),
            ]
            submodel, submodel_path = get_submodel(sdp_node)
            for transform in submodel_transforms:
                submodel = submodel.transform(transform)
            submodel.set_metadata_prop("platform", "alveo")
            submodel.save(submodel_path)
        return model, False

"""Module for post-synthesis resource analysis of FPGA dataflow models."""
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
import xml.etree.ElementTree as ET
from json import JSONDecodeError
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from typing import cast

from finn.util.exception import FINNDataflowError
from finn.util.fpgadataflow import get_device_id, is_hls_node, is_rtl_node


def post_synth_res(
    model: ModelWrapper, override_synth_report_filename: Path | str | None = None
) -> dict[int, dict]:
    """Extract the FPGA resource results from the Vivado synthesis.
    Ensure that all nodes have unique names (by calling the GiveUniqueNodeNames
    transformation) prior to calling this analysis pass to ensure all nodes are
    visible in the results.

    By default, the dict contains one entry per device. If `override_synth_report_filename` is used,
    it automatically gets assigned device ID 0.

    Returns {device_id: {node name : resources_dict}}.
    """
    if override_synth_report_filename is not None:
        return {
            0: _post_synth_res_single_file(
                model,
                Path(override_synth_report_filename),
                model.get_metadata_prop("platform") == "alveo",
                None,
            )
        }
    report_json_data = model.get_metadata_prop("vivado_synth_rpt")
    if report_json_data is None:
        raise FINNDataflowError(
            "Cannot generate resource reports - model metadata "
            "property 'vivado_synth_rpt' is empty!"
        )
    try:
        reports = json.loads(report_json_data)
    except JSONDecodeError:
        reports = {0: Path(report_json_data)}
    result = {}
    for device, file in reports.items():
        result[device] = _post_synth_res_single_file(
            model, file, model.get_metadata_prop("platform") == "alveo", device
        )
    return result


def _post_synth_res_single_file(  # noqa
    model: ModelWrapper, file: Path, is_vitis_flow: bool, device_id: int | None
) -> dict:
    """Extract FPGA resource results from the Vivado synthesis for a single file.

    Parameters
    ----------
        `model`: Model. Used to iterate over all StreamingDataflowPartitions.
        `file`: Path to the resource report to be read.
        `is_vitis_flow`: Set to true in case that the flow/shell is a Vitis flow.
        `device_id`: Device ID for this report. Used to check that the function does not try to
            read data from other devices' SDPs out of this report.
    """
    res_dict = {}
    if file.exists():
        tree = ET.parse(file.absolute())
        root = tree.getroot()
        all_cells = root.findall(".//tablecell")
        # strip all whitespace from table cell contents
        for cell in all_cells:
            cell.attrib["contents"] = cell.attrib["contents"].strip()
    else:
        raise FINNDataflowError(
            f"Could not read synthesis report at: {file}. Please run synthesis first."
        )

    restype_to_ind_default = {
        "LUT": 2,
        "SRL": 5,
        "FF": 6,
        "BRAM_36K": 7,
        "BRAM_18K": 8,
        "DSP": 10,
    }
    restype_to_ind_vitis = {
        "LUT": 4,
        "SRL": 7,
        "FF": 8,
        "BRAM_36K": 9,
        "BRAM_18K": 10,
        "URAM": 11,
        "DSP": 12,
    }

    # format: (human_readable_name_in_report, canonical_name)
    res_types_to_search = [
        ("Total LUTs", "LUT"),
        ("SRLs", "SRL"),
        ("FFs", "FF"),
        ("RAMB36", "BRAM_36K"),
        ("RAMB18", "BRAM_18K"),
        ("URAM", "URAM"),
        ("DSP Blocks", "DSP"),
    ]

    # try to infer resource type to table index by
    # looking at the names in headings
    header_row = root.findall(".//*[@contents='Instance']/..")
    if header_row != []:
        headers = [x.attrib["contents"] for x in list(header_row[0])]
        restype_to_ind = {}
        for res_type_name, res_type in res_types_to_search:
            if res_type_name in headers:
                restype_to_ind[res_type] = headers.index(res_type_name)
    else:
        # could not infer resource types from header
        # fall back to default indices
        restype_to_ind = restype_to_ind_vitis if is_vitis_flow else restype_to_ind_default

    def get_instance_stats(inst_name: str) -> dict[str, int] | None:
        """Return resource stats for a specific instance name."""
        row = root.findall(f".//*[@contents='{inst_name}']/..")
        if row != []:
            node_dict: dict[str, int] = {}
            row = list(row[0])
            for restype, ind in restype_to_ind.items():
                node_dict[restype] = int(row[ind].attrib["contents"])
            return node_dict
        return None

    # global (top-level) stats, including shell etc.
    top_dict = get_instance_stats("(top)")
    if top_dict is not None:
        res_dict["(top)"] = top_dict

    # stats for largest shell components (so we could subtract them later on)
    # IDMA/ODMA are modeled in ONNX SDPs (queried below)
    shell_components = [
        "top_instrumentation_wrap_0_0",
        "top_axi_interconnect_0_0",
        "top_smartconnect_0_0",
    ]
    for shell_comp in shell_components:
        shell_dict = get_instance_stats(shell_comp)
        if shell_dict is not None:
            res_dict[shell_comp] = shell_dict

    for node in model.graph.node:
        if node.op_type == "StreamingDataflowPartition":
            sdp_model = ModelWrapper(cast("str", getCustomOp(node).get_nodeattr("model")))
            sdp_device_id = get_device_id(node)
            if sdp_device_id is None and device_id is not None:
                raise FINNDataflowError(
                    f"Cannot create resource report for device {device_id}, since "
                    f"StreamingDataflowPartition {node.name} has no device ID!"
                )
            if sdp_device_id is not None and device_id is None:
                raise FINNDataflowError(
                    f"This seems to be a Single-FPGA flow (device_id was not specified), "
                    f"but StreamingDataflowPartition {node.name} has a device ID!"
                )
            if sdp_device_id is not None and device_id is not None and sdp_device_id != device_id:
                # Cant find info about this SDP node in this resource report, since its a different
                # devices' report.
                continue
            # Recurse directly into the single-file helper (instead of going through
            # post_synth_res()) so that results from multiple SDPs are merged by node
            # name instead of being wrapped under (and overwriting each other at) the
            # same device key.
            sdp_res_dict = _post_synth_res_single_file(
                sdp_model, file, sdp_model.get_metadata_prop("platform") == "alveo", sdp_device_id
            )
            # drop the nested "(top)" entry: it refers to the same top-level instance
            # already captured above and would otherwise just duplicate top_dict
            sdp_res_dict.pop("(top)", None)
            res_dict.update(sdp_res_dict)
        elif is_hls_node(node) or is_rtl_node(node):
            node_dict = get_instance_stats(node.name)
            if node_dict is not None:
                res_dict[node.name] = node_dict

    return res_dict

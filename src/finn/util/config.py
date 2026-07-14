"""Utility functions for extracting model configuration."""
############################################################################
# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

# Note: This files is migrated and extended from qonnx.util.config
# For more information on the git history of the file see here:
# https://github.com/fastmachinelearning/qonnx/blob/
# abb9eb12e0248014a805f505aacfaeb14d42409a/src/qonnx/util/config.py

import contextlib
import json
import onnx
from numpy import typing as npt
from pathlib import Path
from qonnx.custom_op.registry import getCustomOp, is_custom_op
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


# update this code to handle export configs from subgraphs
# where the subgraph is found in a node's attribute as a graph type
def extract_model_config(
    model: "ModelWrapper", subgraph_hier: str | None, attr_names_to_extract: list[str]
) -> dict[str, dict[str, int | float | str | bool | npt.NDArray | list[str | int | float] | None]]:
    """Create a dictionary with layer name -> attribute mappings extracted from the
    model. The created dictionary can be later applied on a model with
    finn.transform.general.ApplyConfig.

    Nodes in subgraphs are prefixed with their parent hierarchy using '_' as separator.
    For example, a node 'Conv_0' inside a subgraph of node 'IfNode_0' will be exported
    as 'IfNode_0_Conv_0' in the config."""
    cfg = {}
    cfg["Defaults"] = {}
    for n in model.graph.node:
        new_hier = n.name if subgraph_hier is None else str(subgraph_hier) + "_" + n.name

        # Check if this is a custom op and prepare to extract attributes
        layer_dict = {}
        is_custom = is_custom_op(n.domain, n.op_type)
        if is_custom:
            oi = getCustomOp(n)
            layer_dict = {}
            for attr in attr_names_to_extract:
                with contextlib.suppress(AttributeError):
                    layer_dict[attr] = oi.get_nodeattr(attr)

        # Process node attributes - handle both subgraphs and extractable attributes
        for attr in n.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                # If the attribute is a graph, extract configs from the subgraph recursively
                # Include the subgraph attribute name in the hierarchy
                subgraph_hier_with_attr = new_hier + "_" + attr.name
                cfg.update(
                    extract_model_config(
                        model.make_subgraph_modelwrapper(attr.g),
                        subgraph_hier_with_attr,
                        attr_names_to_extract,
                    )
                )

        # Add the node's config if we extracted any attributes
        if is_custom and len(layer_dict) > 0:
            cfg[new_hier] = layer_dict

    return cfg


def extract_model_config_to_json(
    model: "ModelWrapper", json_filename: Path, attr_names_to_extract: list[str]
) -> None:
    """Create a json file with layer name -> attribute mappings extracted from the
    model. The created json file can be later applied on a model with
    finn.transform.general.ApplyConfig."""
    with json_filename.open("w") as f:
        json.dump(
            extract_model_config(
                model, subgraph_hier=None, attr_names_to_extract=attr_names_to_extract
            ),
            f,
            indent=2,
        )


def extract_model_config_consolidate_shuffles(
    model: "ModelWrapper", output_file: Path, hw_attrs: list[str]
) -> None:
    """Export flow that takes into consideration how Shuffle operations have been decomposed."""
    extract_model_config_to_json(model, output_file, hw_attrs)

    with output_file.open() as f:
        config = json.load(f)

    shuffle_configs = {}
    nodes_to_remove = []

    for node in model.graph.node:
        if node.op_type in ["InnerShuffle_rtl", "OuterShuffle_hls"]:
            inst = getCustomOp(node)
            original_name = inst.get_nodeattr("original_node_name")
            original_simd = inst.get_nodeattr("original_simd")

            if original_name and node.name in config:
                if original_name not in shuffle_configs:
                    consolidated_config = config[node.name].copy()
                    if original_simd is not None:
                        consolidated_config["SIMD"] = original_simd
                    shuffle_configs[original_name] = consolidated_config
                nodes_to_remove.append(node.name)

    for node_name in nodes_to_remove:
        del config[node_name]

    config.update(shuffle_configs)

    with output_file.open("w") as f:
        json.dump(config, f, indent=2)

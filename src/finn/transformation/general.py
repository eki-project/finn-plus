"""Generally applicable transformations."""

############################################################################
# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import json
from collections.abc import Callable

# Protobuf onnx graph node type
from onnx import AttributeProto, GraphProto, NodeProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

# Note: This transformation is migrated and extended from qonnx.transformation.general
# For more information on the git history of the file see here:
# https://github.com/fastmachinelearning/qonnx/blob/
# abb9eb12e0248014a805f505aacfaeb14d42409a/src/qonnx/transformation/general.py
from finn.util.exception import FINNInternalError
from finn.util.logging import log


class ApplyConfig(Transformation):
    """Applies node properties (attributes) from either a config dict or its JSON
    representation given as a filename.
    The JSON file can specify default values for particular op_types, as well
    as values for nodes with particular names. Example dict::

        {
        # set kernel_size = 3 for all nodes with op_type=Im2Col
        "Defaults" : {"kernel_size" : [3, ["Im2Col"]]},
        # set kernel_size = 7 for the particular node with name Im2Col_0
        "Im2Col_0" : {"kernel_size" : 7}
        }

    """  # noqa

    def __init__(
        self, config: Path | str | dict, node_filter: Callable[[NodeProto], bool] = lambda _: True
    ) -> None:
        """Apply a JSON config file to the model."""
        super().__init__()
        if type(config) is not dict:
            self.config = Path(config)  # type: ignore
        else:
            self.config = config
        self.node_filter = node_filter
        self.used_configurations = ["Defaults"]
        self.missing_configurations = []

    def configure_network(
        self, target: GraphProto | ModelWrapper, model_config: dict, subgraph_hier: str | None
    ) -> None:
        """Configure network - target can be a GraphProto or ModelWrapper.
        If it's a ModelWrapper, get the graph.
        """
        if type(target) is ModelWrapper:
            graph: GraphProto = target.graph
        elif type(target) is GraphProto:
            graph: GraphProto = target
        else:
            raise FINNInternalError(
                f"Tried configuring target of type {type(target)}, "
                f"but a GraphProto or ModelWrapper is required."
            )

        for node in graph.node:
            if not self.node_filter(node):
                continue

            # Build the config key by prepending hierarchy
            config_key = (
                node.name if subgraph_hier is None else str(subgraph_hier) + "_" + node.name
            )

            try:
                node_config = model_config[config_key].copy()
            except KeyError:
                self.missing_configurations += [node.name]
                node_config = {}

            if node_config:
                self.used_configurations += [config_key]

            from qonnx.custom_op.registry import getCustomOp

            try:
                inst = getCustomOp(node)

                if "Defaults" in model_config.keys():
                    # set specified defaults
                    default_values = []
                    for key, value in model_config["Defaults"].items():
                        assert len(value) % 2 == 0
                        if key not in model_config:
                            for val, op in zip(value[::2], value[1::2], strict=True):
                                default_values.append((key, val, op))
                                assert not (op == "all" and len(value) > 2)
                    default_configs = {
                        key: val
                        for key, val, op in default_values
                        if op == "all" or node.op_type in op
                    }
                    for attr_name, value in default_configs.items():
                        inst.set_nodeattr(attr_name, value)

                # set node attributes from specified configuration
                for attr_name, value in node_config.items():
                    inst.set_nodeattr(attr_name, value)
            except Exception:
                # Node is not a custom op, but it might have subgraphs
                pass

            # Recursively handle nested subgraphs
            for attr in node.attribute:
                if attr.type == AttributeProto.GRAPH:
                    # Build the subgraph hierarchy including the attribute name
                    if subgraph_hier is None:
                        new_hier = node.name
                    else:
                        new_hier = str(subgraph_hier) + "_" + node.name
                    # Include the subgraph attribute name in the hierarchy
                    new_hier = new_hier + "_" + attr.name
                    self.configure_network(attr.g, model_config, subgraph_hier=new_hier)

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the config to the model."""
        if isinstance(self.config, dict):
            model_config = self.config
        else:
            with self.config.open("r") as f:
                model_config = json.load(f)

        # apply configuration on upper level
        self.configure_network(model.model.graph, model_config, subgraph_hier=None)

        # Configuration verification
        # Remove duplicates from missing_configurations
        # (can happen with shared subgraphs in If nodes)
        unique_missing = list(dict.fromkeys(self.missing_configurations))
        if len(unique_missing) > 0:
            log.warning("\nNo HW configuration for nodes: " + ", ".join(unique_missing))

        # Check for unused configs (top-level configs that weren't applied)
        unused_configs = [
            x for x in model_config if x not in self.used_configurations and x != "Defaults"
        ]
        if len(unused_configs) > 0:
            log.warning("\nUnused HW configurations: " + ", ".join(unused_configs))

        # one iteration is enough
        return (model, False)

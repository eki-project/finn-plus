############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

"""Module for set loop boundary."""

import onnx
from ast import literal_eval
from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from typing import Literal

from finn.util.exception import FINNInternalError


class SetLoopBoundary(Transformation):
    """Sets metadata attributes to nodes between defined node or tensor ranges in an ONNX model.

    :param node_metadata: Dictionary containing metadata attributes to set on the nodes.
    :param node_range: Tuple containing start and end node names (start_node, end_node).
    :param tensor_range: Tuple containing start and end tensor names (start_tensor, end_tensor).
    """

    def __init__(
        self,
        node_metadata: dict[str, str],
        node_range: tuple[NodeProto, NodeProto] | None = None,
        tensor_range: tuple[str, str] | None = None,
    ) -> None:
        """Initialize instance."""
        super().__init__()
        if (node_range is None and tensor_range is None) or (
            node_range is not None and tensor_range is not None
        ):
            raise FINNInternalError(
                "You must provide either a node_range or a tensor_range, but not both or none."
            )

        self.start_node: NodeProto | None = None
        self.end_node: NodeProto | None = None
        self.start_tensor: str | None = None
        self.end_tensor: str | None = None

        if node_range:
            self.start_node, self.end_node = node_range
        if tensor_range:
            self.start_tensor, self.end_tensor = tensor_range

        self.node_metadata = node_metadata

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, Literal[False]]:
        """Apply transformation."""
        graph = model.graph

        # Transformation can only be applied to cleaned up (const-folded) FINN-ONNX model
        # Check if any Quant or Const nodes exist and if yes, throw an error
        count = 0
        for op_type in ["BinaryQuant", "Quant", "Trunc", "IntQuant", "FloatQuant", "Constant"]:
            count += len(model.get_nodes_by_op_type(op_type))
        assert (
            count == 0
        ), """The model is either in QONNX format (Quant nodes present)
            or const folding was not applied yet. SetLoopBoundary can only be applied
            to cleaned up and const-folded FINN-ONNX model."""

        apply_metadata = False

        for node in graph.node:
            # Activate the metadata application once the start node or tensor is found
            if self.start_node and node.name == self.start_node.name:
                apply_metadata = True
            if self.start_tensor and (
                self.start_tensor in node.input or self.start_tensor in node.output
            ):
                apply_metadata = True

            # Apply metadata if within range and lambda condition is met
            if apply_metadata:
                for key, value in self.node_metadata.items():
                    node.metadata_props.append(onnx.StringStringEntryProto(key=key, value=value))

            # Apply dummy metadata to allow loop rolling to work correctly
            # If we don't do this, the loop extraction will assume
            # that the set metadata in the beginning applies to all nodes
            else:
                for key, value in self.node_metadata.items():
                    values = literal_eval(value)
                    node.metadata_props.append(
                        onnx.StringStringEntryProto(
                            key=key, value=f"['{values[0]}', '{values[1]}1']"
                        )
                    )

            # Deactivate the metadata application once the end node or tensor is reached
            if self.end_node and node.name == self.end_node.name:
                apply_metadata = False
            if self.end_tensor and (
                self.end_tensor in node.input or self.end_tensor in node.output
            ):
                apply_metadata = False

        return model, False

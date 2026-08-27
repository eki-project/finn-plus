"""Utility functions for working with fpgadataflow nodes in ONNX graphs."""
# Copyright (c) 2020 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

from onnx import NodeProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp, is_custom_op
from qonnx.util.basic import get_by_name
from typing import cast

from finn.util.exception import FINNInternalError, FINNUserError


def is_fpgadataflow_node(node: NodeProto | None) -> bool:
    """Return True if given node is fpgadataflow node. Otherwise False."""
    is_node = False
    if node is not None and is_custom_op(node.domain):
        n_backend = get_by_name(node.attribute, "backend")
        if n_backend is not None:
            backend_value = n_backend.s.decode("UTF-8")
            if backend_value == "fpgadataflow":
                is_node = True

    return is_node


def is_hls_node(node: NodeProto | None) -> bool:
    """Return True if given node is hls node. Otherwise False."""
    is_node = False
    if node is not None and node.domain == "finn.custom_op.fpgadataflow.hls":
        n_backend = get_by_name(node.attribute, "backend")
        if n_backend is not None:
            backend_value = n_backend.s.decode("UTF-8")
            if backend_value == "fpgadataflow":
                is_node = True

    return is_node


def is_rtl_node(node: NodeProto | None) -> bool:
    """Return True if given node is rtl node. Otherwise False."""
    is_node = False
    if node is not None and node.domain == "finn.custom_op.fpgadataflow.rtl":
        n_backend = get_by_name(node.attribute, "backend")
        if n_backend is not None:
            backend_value = n_backend.s.decode("UTF-8")
            if backend_value == "fpgadataflow":
                is_node = True

    return is_node


def get_submodel(node: NodeProto) -> tuple[ModelWrapper, Path]:
    """Try to retrieve the submodel (and its path) of a StreamingDataflowPartition
    node. If the node is not an SDP or the `model` metadata prop does not exist,
    or the path does not point to a file, an error is raised.
    """
    if node.op_type != "StreamingDataflowPartition":
        raise FINNInternalError(f"Cannot get model of non-SDP node: {node.name}")
    p = getCustomOp(node).get_nodeattr("model")
    if p is None:
        raise FINNInternalError(
            f"SDP node {node.name} has no 'model' metadata prop. " f"Cannot get model."
        )
    p = Path(str(p))
    if not p.exists():
        raise FINNInternalError(
            f"Cannot open model of SDP node {node.name}: " f"No file found at path: {p}"
        )
    return ModelWrapper(str(p)), p


def get_device_id(node: NodeProto) -> int | None:
    """Return the node's device ID. If no nodeattribute exists returns None."""
    try:
        return cast("int", (getCustomOp(node).get_nodeattr("device_id")))
    except ValueError:
        return None


def set_device_id(node: NodeProto, value: int) -> None:
    """Set the device_id nodeattribute of the given node."""
    getCustomOp(node).set_nodeattr("device_id", value)


def get_input_nodes(model: ModelWrapper) -> list[tuple[NodeProto, int]]:
    """Return a list of all input nodes (no predecessors) and their indices in the graph."""
    res = []
    for i, node in enumerate(model.graph.node):
        pre = model.find_direct_predecessors(node)
        if pre is None:
            res.append((node, i))
    return res


def get_output_nodes(model: ModelWrapper) -> list[tuple[NodeProto, int]]:
    """Return a list of all input nodes (no successors) and their indices in the graph."""
    res = []
    for i, node in enumerate(model.graph.node):
        suc = model.find_direct_successors(node)
        if suc is None:
            res.append((node, i))
    return res


def check_all_sdp_nodes(model: ModelWrapper) -> None:
    """Verify that all nodes are SDP nodes."""
    for node in model.graph.node:
        if node.op_type != "StreamingDataflowPartition":
            raise FINNUserError(
                f"Node {node.name} is not a StreamingDataflowPartition. "
                f"Make sure to run step_create_dataflow_partition (or "
                f"its Multi-FPGA equivalent) before."
            )


def get_vitis_xo(node: NodeProto) -> Path:
    """Get the path to the XO file of the submodel of the given node. Raises an error if the
    path does not point to an existing file or the metadata prop does not exist.
    The path to the xo must not necessarily point to an existing file.
    """
    try:
        sm_path = Path(str(getCustomOp(node).get_nodeattr("model")))
    except AttributeError as e:
        raise FINNUserError(f"Node {node.name} has no sub-model/graph!") from e
    if not sm_path.exists():
        raise FINNUserError(f"No file found for submodel/graph of node {node.name} at {sm_path}!")
    xo = ModelWrapper(str(sm_path)).get_metadata_prop("vitis_xo")
    if xo is None:
        raise FINNUserError(f"Submodel/graph of node {node.name} has no vitis_xo metadata!")
    return Path(xo)

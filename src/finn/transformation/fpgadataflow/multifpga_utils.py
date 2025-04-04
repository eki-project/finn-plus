from __future__ import annotations

from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp


def get_device_id(node: NodeProto) -> int | None:
    try:
        return getCustomOp(node).get_nodeattr("device_id")
    except ValueError:
        return None


def set_device_id(node: NodeProto, value: int) -> None:
    getCustomOp(node).set_nodeattr("device_id", value)


def get_last_submodel_node(sdp_node: NodeProto) -> NodeProto:
    return get_submodel(sdp_node).graph.node[-1]


def get_first_submodel_node(sdp_node: NodeProto) -> NodeProto:
    return get_submodel(sdp_node).graph.node[0]


def get_submodel(node: NodeProto) -> ModelWrapper:
    return ModelWrapper(getCustomOp(node).get_nodeattr("model"))


def get_all_branches(model: ModelWrapper) -> list[list[int]]:
    raise NotImplementedError()

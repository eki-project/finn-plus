from __future__ import annotations

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.multifpga_utils import (
    get_device_id,
    get_submodel,
    set_device_id,
)


class CreateMultiFPGAStreamingDataflowPartition(Transformation):
    """Operates like CreateDataflowPartition but using the nodes device id as a key. Additionally,
    two non consecutive instances on the same device create
    different SDPs (think for example about a there-and-back topology)

    IMPORTANT: Currently this assumes that every branch is split and joined on the same device"""

    def __init__(self) -> None:
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        current_device = get_device_id(model.graph.node[0])
        current_max = 0
        for node in model.graph.node:
            assert node.op_type not in ["StreamingDataflowPartition", "GenericPartition"]
            device = get_device_id(node)
            assert device is not None, f"Node {node.name} of type {node.op_type} does not have"
            "a device_id attribute"
            # TODO: Setting partition_id and calling CreateDataflowPartitions might not be
            # the best way to do it. Maybe change at some point
            if device != current_device:
                current_device = device
                current_max += 1
            getCustomOp(node).set_nodeattr("partition_id", current_max)

        model = model.transform(CreateDataflowPartition())

        # Set the SDP's device_id
        for node in model.graph.node:
            device_id = get_device_id(get_submodel(node).graph.node[0])
            assert device_id is not None
            set_device_id(node, device_id)
        return model, False

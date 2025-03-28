from __future__ import annotations

import json
from onnx import NodeProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.builder.build_dataflow_config import PartitioningConfiguration, PartitioningStrategy
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition


def get_device_id(node: NodeProto) -> int | None:
    try:
        return getCustomOp(node).get_nodeattr("device_id")
    except ValueError:
        return None


def set_device_id(node: NodeProto, value: int) -> None:
    getCustomOp(node).set_nodeattr("device_id", value)


def get_submodel(node: NodeProto) -> ModelWrapper:
    return ModelWrapper(getCustomOp(node).get_nodeattr("model"))


def get_all_branches(model: ModelWrapper) -> list[list[int]]:
    raise NotImplementedError()


class PartitionForMultiFPGA(Transformation):
    """
    Receive a model with only FPGADataflow nodes and partition it by assigning it's
    device node attribute. Partitioning is done with respect to the chosen strategy
    """

    def __init__(self, pcfg: PartitioningConfiguration, report_dir: Path) -> None:
        self.pcfg = pcfg
        self.report_dir = report_dir

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        if self.pcfg.num_fpgas <= 1:
            # Do nothing, since no multi fpga is used
            return model, False

        if self.pcfg.num_fpgas > len(model.graph.node):
            # TODO: Exchange for custom error exception class
            raise Exception(
                f"Cannot partition: number of FPGAs set to \
               {self.pcfg.num_fpgas},but only {len(model.graph.node)} layers are \
               in the model! (num_fpgas <= #Layers must be true)"
            )

        # Notify other components that we do a Multi-FPGA build
        model.set_metadata_prop("fpgamode", "multi")

        # Collecting groups
        # grouped_together_nodes = None  # get_grouped_layers(model)  # TODO: Implement

        # Trying to open resource estimates
        resource_estimates = None
        raw_estimates = None
        if self.pcfg.partition_strategy == PartitioningStrategy.RESOURCE_UTILIZATION:
            re_path = Path(self.report_dir) / "estimate_layer_resources.json"
            re_hls_path = Path(self.report_dir) / "estimate_layer_resources_hls.json"

            # TODO: Custom exceptions
            if not re_path.exists():
                raise Exception(
                    f"Cannot find layer resource estimate in {re_path}. \
                  Check to make sure step_generate_estimate_reports and \
                  the corresponding output are set!"
                )
            if not re_hls_path.exists():
                raise Exception(
                    f"Cannot find layer resource estimate in {re_hls_path}. \
                  Check to make sure step_generate_estimate_reports and \
                  the corresponding output are set!"
                )
            with re_path.open("r") as f:
                raw_estimates = json.load(f)
            with re_hls_path.open("r") as f:
                raw_estimates.update(json.load(f))

            resource_estimates = []
            for node in model.graph.node:
                if node.name not in raw_estimates.keys():
                    raise Exception(
                        f"Could not find resource estimate for node \
                     {node.name} - cannot partition without estimate while \
                     using RESOURCE_UTILIZATION strategy"
                    )
                resource_estimates.append(raw_estimates[node.name])

            # Converting all estimates to int
            for i in range(len(resource_estimates)):
                for key in resource_estimates[i].keys():
                    resource_estimates[i][key] = int(resource_estimates[i][key])

        return model, False


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


class CreateNetworkMetadata(Transformation):
    """Create metadata that describes how kernels have to be configured to achieve a given network
    topology"""

    def __init__(
        self, rx_channel_per_kernel: int, tx_channel_per_kernel: int, kernel_available: int
    ) -> None:
        self.kernel_rx = rx_channel_per_kernel
        self.kernel_tx = tx_channel_per_kernel
        self.kernels = kernel_available

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        return model, False


class CreateChainNetworkMetadata(CreateNetworkMetadata):
    def __init__(
        self, rx_channel_per_kernel: int, tx_channel_per_kernel: int, kernel_available: int
    ) -> None:
        super().__init__(rx_channel_per_kernel, tx_channel_per_kernel, kernel_available)
        assert self.kernel_rx * self.kernels >= 1
        assert self.kernel_tx * self.kernels >= 1

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        model = model.transform(GiveUniqueNodeNames())
        seen_devices = []
        for node in model.graph.node:
            dev = get_device_id(node)
            assert dev not in seen_devices, (
                "Cannot create networked chain, " "since the same device is visited more than once"
            )
            seen_devices.append(dev)

        # _table = {}
        # for i, n1 in enumerate(model.graph.node):
        #    if i == len(model.graph.node) - 1:
        #        break
        #    n2 = model.graph.node[i + 1]
        #    _d1 = get_device_id(n1)
        #    _d2 = get_device_id(n2)

        return model, False


class CreateReturnChainNetworkMetadata(CreateNetworkMetadata):
    def __init__(
        self, rx_channel_per_kernel: int, tx_channel_per_kernel: int, kernel_available: int
    ) -> None:
        super().__init__(rx_channel_per_kernel, tx_channel_per_kernel, kernel_available)
        assert self.kernel_rx * self.kernels >= 2
        assert self.kernel_tx * self.kernels >= 2

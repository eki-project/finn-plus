from __future__ import annotations

from abc import ABC, abstractmethod
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.builder.build_dataflow_config import DataflowBuildConfig, PartitioningStrategy
from finn.transformation.fpgadataflow.multifpga_network import Device
from finn.transformation.fpgadataflow.multifpga_utils import get_inseperable_nodes, set_device_id


class Partitioner(ABC):
    """Models a linear problem that can be used to solve Multi-FPGA partitioning. The idea to solve
    this in general using an LP was first devised by the AMD team for Elastic-DF and implemented as
    a prototype in finn-experimental.
    (https://github.com/Xilinx/finn-experimental/blob/main/src/finnexperimental/analysis/partitioning.py)

    We use a slightly different approach to modelling the problem and the objective function,
    however the partitioner from finn-experimental should be relativly easy to swap in
    if needed."""

    def __init__(
        self,
        strategy: PartitioningStrategy,
        inseperable_nodes: list[list[Device]],
        resource_estimates: dict | None = None,
    ) -> None:
        self.strategy = strategy
        self.inseperable_nodes = inseperable_nodes
        self.resource_estimates = resource_estimates

    @abstractmethod
    def solve(self) -> dict[int, Device] | None:
        """Try to optimize the objective function. If no feasible solution is found
        return None, otherwise return a mapping of nodes to their device"""


class AuroraPartitioner(Partitioner):
    def __init__(
        self,
        strategy: PartitioningStrategy,
        inseperable_nodes: list[list[int]],
        resource_estimates: dict | None = None,
    ) -> None:
        super().__init__(strategy, inseperable_nodes, resource_estimates)

    def solve(self) -> dict[int, Device] | None:
        return None


class PartitionForMultiFPGA(Transformation):
    """
    Receive a model with only FPGADataflow nodes and partition it by assigning it's
    device node attribute. Partitioning is done with respect to the chosen strategy.
    To determine how partitioning is done, pass the partitioner type yourself.
    """

    def __init__(
        self, cfg: DataflowBuildConfig, partitioner_type: type[Partitioner] = AuroraPartitioner
    ) -> None:
        self.cfg = cfg
        self.partitioner_type = partitioner_type

    def gather_resource_utilization(self, model: ModelWrapper) -> dict[str, dict[str, int]]:
        """Gather the resources of all layers based on the estimation values from
        the previous build steps.
        Returns a table like:
        {
            "layer_0": {
                "LUT": ...,
                "DSP": ...,
                ...
            },
            ...
        }"""
        # TODO: Check / Clean up all the various estimate functions
        model = model.transform(GiveUniqueNodeNames())
        estimates = res_estimation(model, self.cfg.fpga_part)
        hls_estimates = hls_synth_res_estimation(model)
        for layer in hls_estimates.keys():
            # Case 1: Only HLS estimate: Add it
            if layer not in estimates.keys():
                estimates[layer] = hls_estimates[layer]
            else:
                current_layer_estimates = hls_estimates[layer]
                for restype in current_layer_estimates.keys():
                    # Case 2: Res exists in both estimates: Take max
                    if restype in estimates[layer].keys():
                        estimates[layer][restype] = max(
                            estimates[layer][restype], current_layer_estimates[restype]
                        )
                    # Case 3: Res exists only in hls: Add it
                    else:
                        estimates[layer][restype] = current_layer_estimates[restype]
        return estimates

    def estimate_required_fpgas(self) -> int:
        """Use resource utilization to estimate how many FPGAs will be needed to
        partition the given model."""
        raise NotImplementedError()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        if self.cfg.partitioning_configuration is None:
            return model, False

        # Dont split during branches. Find all layers that should be on the same device.
        # TODO: Implementation
        inseperable_nodes = get_inseperable_nodes(model)

        # Create the partitioner itself
        partitioner = self.partitioner_type(
            strategy=self.cfg.partitioning_configuration.partition_strategy,
            inseperable_nodes=inseperable_nodes,
            resource_estimates=None,
        )
        if (
            self.cfg.partitioning_configuration.partition_strategy
            == PartitioningStrategy.RESOURCE_UTILIZATION
        ):
            estimates = self.gather_resource_utilization(model)
            for layer in estimates.keys():
                assert any(estimates[layer][res] > 0 for res in estimates[layer].keys()), (
                    f"Layer {layer} has an all-0 resource estimation. "
                    "Cannot partition faulty resource estimated layers."
                )
            partitioner.resource_estimates = estimates

        # Try and solve the model
        mapping = partitioner.solve()

        # TODO: Update with QoL
        assert mapping is not None, "Could not find a feasible solution for partitioning. Stopping."

        # TODO: Use index or name?
        for i, node in enumerate(model.graph.node):
            set_device_id(node, mapping[i])
        return model, False

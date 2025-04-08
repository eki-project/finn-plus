from __future__ import annotations

import mip
from abc import ABC, abstractmethod
from mip import Model, xsum
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    MultiFPGACommunicationScheme,
    PartitioningStrategy,
)
from finn.transformation.fpgadataflow.multifpga_network import Device
from finn.transformation.fpgadataflow.multifpga_utils import get_inseperable_nodes, set_device_id
from finn.util.platforms import Platform


class Partitioner(ABC):
    """Models a linear problem that can be used to solve Multi-FPGA partitioning. The idea to solve
    this in general using an LP was first devised by the AMD team for Elastic-DF and implemented as
    a prototype in finn-experimental.
    (https://github.com/Xilinx/finn-experimental/blob/main/src/finnexperimental/analysis/partitioning.py)

    We use a slightly different approach to modelling the problem and the objective function,
    however the partitioner from finn-experimental should be relativly easy to swap in
    if needed.

    Parameters:

    inseperable_nodes: Nodes that need to stay together because they are in a split

    considered_resources: What types of resources are used in the objective
        function to determine load."""

    def __init__(
        self,
        strategy: PartitioningStrategy,
        devices: int,
        nodes: int,
        inseperable_nodes: list[list[int]],
        resources_per_device: dict,
        resource_estimates: dict | None = None,
        considered_resources: list[str] | None = None,
    ) -> None:
        self.strategy = strategy
        self.inseperable_nodes = inseperable_nodes
        self.resource_estimates = resource_estimates
        self.resources_per_device = resources_per_device
        self.considered_resources = (
            ["LUT", "FF", "BRAM_18K", "DSP"]
            if considered_resources is None
            else considered_resources
        )
        self.device_count = devices
        self.node_count = nodes

    @abstractmethod
    def solve(self) -> dict[int, Device] | None:
        """Try to optimize the objective function. If no feasible solution is found
        return None, otherwise return a mapping of nodes to their device"""

    def total_resources_per_device(self) -> dict[str, int]:
        """Return the total resources per device. Normally, resources_per_device
        is split into the different SLRs if there are some."""
        if self.resources_per_device is None:
            return {}
        acc = {}
        for restype in self.considered_resources:
            acc[restype] = 0
            for slr in self.resources_per_device:
                acc[restype] += self.resources_per_device[slr][restype]
        return acc


class AuroraPartitioner(Partitioner):
    def __init__(
        self,
        communication_scheme: MultiFPGACommunicationScheme,
        network_ports_per_device: int,
        strategy: PartitioningStrategy,
        devices: int,
        nodes: int,
        inseperable_nodes: list[list[int]],
        resources_per_device: dict,
        resource_estimates: dict | None = None,
        considered_resources: list[str] | None = None,
        limit_nodes_per_device: int | None = None,
        max_utilization: float = 0.75,
    ) -> None:
        super().__init__(
            strategy,
            devices,
            nodes,
            inseperable_nodes,
            resources_per_device,
            resource_estimates,
            considered_resources,
        )
        self.limit_nodes_per_device = limit_nodes_per_device
        self.network_ports_per_device = network_ports_per_device
        self.communication_scheme = communication_scheme
        self.max_utilization = max_utilization
        self.model = Model()

        # self.devices[node][device] = 1: Node <node> is on device <device>
        self.devices = [
            [
                self.model.add_var(name=f"node{node}_on_device{device}", var_type=mip.BINARY)
                for device in range(self.device_count)
            ]
            for node in range(self.node_count)
        ]

        # Every layer can only be on one device
        for node in range(self.node_count):
            self.model += (
                xsum(self.devices[node][device] for device in range(len(self.devices[node]))) == 1
            )

        # Helper to see what device a node is on
        self.chosen_device = [
            self.model.add_var(name=f"lod_{node}", var_type=mip.INTEGER)
            for node in range(self.node_count)
        ]
        for node in range(self.node_count):
            self.model += self.chosen_device[node] == xsum(
                self.devices[node][device] * device  # type: ignore
                for device in range(self.device_count)
            )

        # Grouped nodes need to stay together
        for group in self.inseperable_nodes:
            for node in range(len(group) - 1):
                self.model += self.chosen_device[group[node]] == self.chosen_device[group[node + 1]]

        # Number of nodes having this device as their ID
        self.nodesperdevice = [
            self.model.add_var(name=f"lpd_{device}", var_type=mip.INTEGER)
            for device in range(self.device_count)
        ]
        for device in range(self.device_count):
            self.model += self.nodesperdevice[device] == xsum(
                self.devices[node][device] for node in range(self.device_count)
            )

        # Optionally limit number of nodes per device
        # (This may be necessary in some cases to avoid the maximum number of compute units allowed
        # on the FPGAs)
        if self.limit_nodes_per_device is not None:
            for device in range(self.device_count):
                self.model += (
                    self.nodesperdevice[device] <= self.limit_nodes_per_device  # type: ignore
                )

        # Connections that a device has with other devices
        self.connections_per_device_helper = [
            [
                self.model.add_var(
                    name=f"helper_connections_on_device_{device}_{node}", var_type=mip.INTEGER
                )
                for node in range(self.node_count)
            ]
            for device in range(self.device_count)
        ]
        self.connections_per_device = [
            self.model.add_var(name=f"connections_on_device_{device}", var_type=mip.INTEGER)
            for device in range(self.device_count)
        ]
        for device in range(self.device_count):
            for node in range(self.node_count - 1):
                self.model += (
                    self.connections_per_device_helper[device][node]
                    >= self.devices[node][device] - self.devices[node + 1][device]
                )
                self.model += (
                    self.connections_per_device_helper[device][node]
                    >= self.devices[node + 1][device] - self.devices[node][device]
                )
            self.model += self.connections_per_device[device] == xsum(
                self.connections_per_device_helper[device][node] for node in range(self.node_count)
            )

        # Limit the number of connections per device (depends on the FPGAs QSFP ports)
        for device in range(self.device_count):
            self.model += (
                self.connections_per_device[device] <= self.network_ports_per_device
            )  # type: ignore

        # Helper for device difference
        device_diff = []
        for i in range(self.node_count):
            device_diff.append(
                self.model.add_var(name=f"helper_consec_device_{i}", var_type=mip.INTEGER)
            )

        # Consecutive nodes must be on consecutive devices
        for node in range(self.node_count - 1):
            self.model += (
                device_diff[node] >= self.chosen_device[node] - self.chosen_device[node + 1]
            )
            self.model += (
                device_diff[node] >= self.chosen_device[node + 1] - self.chosen_device[node]
            )
            self.model += device_diff[node] <= 1
            self.model += device_diff[node] >= 0

        # Setting topology requirements
        match self.communication_scheme:
            case MultiFPGACommunicationScheme.AURORA_CHAIN:
                self.model += self.chosen_device[0] == 0
                self.model += self.chosen_device[-1] == self.device_count - 1

            case MultiFPGACommunicationScheme.AURORA_RETURNCHAIN:
                raise NotImplementedError()

            case _:
                raise AssertionError(
                    f"Invalid communication scheme "
                    f"for Aurora partitioner: {self.communication_scheme}"
                )

        # Objective Function
        if self.strategy == PartitioningStrategy.LAYER_COUNT:
            # Calculcate the difference to the "ideal" load
            # (All devices have the same number of layers)
            avg_diff = [
                self.model.add_var(var_type=mip.CONTINUOUS) for i in range(self.device_count)
            ]
            avg_ideal_load = self.node_count / self.device_count
            for i in range(self.device_count):
                self.model += avg_diff[i] >= self.nodesperdevice[i] - avg_ideal_load  # type: ignore
                self.model += avg_diff[i] >= avg_ideal_load - self.nodesperdevice[i]  # type: ignore

            # Get the largest of those differences
            max_diff = self.model.add_var(name="max_diff", var_type=mip.CONTINUOUS)
            for device in range(self.device_count):
                self.model += max_diff >= avg_diff[device]

            # Try to minimize the max difference to ideal
            self.model.objective = max_diff
            self.model.sense = mip.MINIMIZE

        elif self.strategy == PartitioningStrategy.RESOURCE_UTILIZATION:
            assert self.resource_estimates is not None

            # Collect the resource usage of all nodes on a device
            self.resource_use_int = {}
            for device in range(self.device_count):
                self.resource_use_int[device] = {}
                for resource_name in self.considered_resources:
                    self.resource_use_int[device][resource_name] = self.model.add_var(
                        f"resource_use_int_device{device}_resource{resource_name}",
                        var_type=mip.INTEGER,
                    )
                    self.model += self.resource_use_int[device][resource_name] == xsum(
                        [
                            self.devices[layer][device]
                            * self.resource_estimates[layer][resource_name]
                            for layer in range(self.device_count)
                            if resource_name in self.resource_estimates[layer].keys()
                        ]
                    )

            # Limit resource usage to (available resources * max usage in percent)
            total_per_device = self.total_resources_per_device()
            for device in range(self.device_count):
                for resource_name in self.considered_resources:
                    max_resources = int(total_per_device[resource_name] * self.max_utilization)
                    self.model += self.resource_use_int[device][resource_name] <= max_resources

            # Balance so that the maximum difference to the ideal load over all devices and
            # resources is as low as possible
            # Use relative values because resources are available at vastly different scales
            self.resource_diff = {}
            self.resource_use_relative = {}
            for device in range(self.device_count):
                self.resource_diff[device] = {}
                self.resource_use_relative[device] = {}
                for resource_name in self.considered_resources:
                    self.resource_diff[device][resource_name] = self.model.add_var(
                        name=f"resource_diff_to_ideal_device{device}_resource{resource_name}",
                        var_type=mip.CONTINUOUS,
                    )
                    self.resource_use_relative[device][resource_name] = self.model.add_var(
                        name=f"resource_use_cont_device{device}_resource{resource_name}",
                        var_type=mip.CONTINUOUS,
                    )

                    # TODO: Ignore linewidth linter rule in the future to make this more readable
                    # Resource util in relative terms (0-1)
                    self.model += self.resource_use_relative[device][resource_name] == (
                        self.resource_use_int[device][resource_name]
                        / total_per_device[resource_name]
                    )

                    # Convert to float and get diff
                    self.model += self.resource_diff[device][resource_name] >= (
                        self.resource_use_relative[device][resource_name]
                        - total_per_device["ideal_utilization"]
                    )
                    self.model += self.resource_diff[device][resource_name] >= (
                        total_per_device["ideal_utilization"]
                        - self.resource_use_relative[device][resource_name]
                    )

            # A device cannot be completely empty
            for device in range(self.device_count):
                self.model += (
                    xsum(
                        [
                            self.resource_use_relative[device][res]
                            for res in self.considered_resources
                        ]
                    )
                    >= 0.0001
                )  # type: ignore

            # The min resource diff to ideal on a device, regardless of resource type
            # (If ideal is 70%, and we have LUT: 61% and DSP: 32%,
            # then we use 61%, so diff is 70%-61%=9%)
            self.min_resource_diff = []
            for device in range(self.device_count):
                self.min_resource_diff.append(
                    self.model.add_var(f"min_resource_diff_device{device}", var_type=mip.CONTINUOUS)
                )
                for res in self.considered_resources:
                    self.model += self.min_resource_diff[device] <= self.resource_diff[device][res]

                    # If we dont specify this, it will stay at the initial value of 0,
                    # since 0 is smaller than all the resource_diffs
                    self.model += self.min_resource_diff[device] >= 0.0001

            # Maximum of the min resource diff of all devices
            max_diff = self.model.add_var("max_diff", var_type=mip.CONTINUOUS)
            for device in range(self.device_count):
                self.model += max_diff >= xsum(
                    [self.resource_diff[device][res] for res in self.considered_resources]
                )

            # Set objective function
            self.model.objective = max_diff
            self.model.sense = mip.MINIMIZE

        else:
            raise AssertionError(f"Unknown partitioning strategy: {self.strategy}")

        # TODO: Tests

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

    def resources_per_device(self, p: Platform) -> dict[int, dict[str, int]]:
        """Return the available resources as given by FINN platforms as a
        dictionary instead of nested lists. First by SLR, then by resource name"""
        assert p is not None
        res = p.compute_resources
        new = {}
        for slr in range(len(res)):
            new[slr] = {}
            for i, name in enumerate(["LUT", "FF", "BRAM_18K", "URAM", "DSP"]):
                new[slr][name] = res[slr][i]
        return new

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

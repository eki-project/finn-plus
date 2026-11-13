"""AuroraFlow partitioner for MultiFPGA."""

import mip
from math import ceil
from mip import xsum
from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from typing import Any

from finn.builder.build_dataflow_config import DataflowBuildConfig, MFTopology, PartitioningStrategy
from finn.transformation.fpgadataflow.multifpga.graph import get_inseparable_nodes
from finn.transformation.fpgadataflow.multifpga.partitioner import Partitioner
from finn.util.exception import (
    FINNMultiFPGAConfigError,
    FINNMultiFPGAError,
    FINNMultiFPGAPartitionerError,
)
from finn.util.logging import log
from finn.util.platforms import platforms
from finn.util.resources import available_resources_on_platform, get_estimated_model_resources


class AuroraPartitioner(Partitioner):
    """Concrete partitioner implementation for usage with AuroraFlow cores."""

    def get_successors(self, node: NodeProto) -> list[NodeProto]:
        """Return the list of direct successors."""
        s = self.modelwrapper.find_direct_successors(node)
        if s is None:
            return []
        return s

    def run_checks(self) -> None:
        """Run some checks on the model and configuration. This will warn or error,
        in case issues are detected.
        """
        if self.pcfg.num_fpgas > len(self.modelwrapper.graph.node):
            # Stop if there are more devices than nodes
            raise FINNMultiFPGAPartitionerError(
                f"Model infeasible: Cannot partition a model with "
                f"{len(self.modelwrapper.graph.node)} nodes to {self.pcfg.num_fpgas} devices!"
            )

        if self.pcfg.partition_strategy == PartitioningStrategy.RESOURCE_UTILIZATION:
            if self.pcfg.max_utilization > 1.0:
                raise FINNMultiFPGAPartitionerError(
                    f"Max utilization was set to {self.pcfg.max_utilization:2.2%} > "
                    f"100%. Decrease max_utilization to continue."
                )
            if self.pcfg.ideal_utilization > 1.0:
                raise FINNMultiFPGAPartitionerError(
                    f"Ideal utilization was set to {self.pcfg.ideal_utilization:2.2%} > "
                    f"100%. Decrease ideal_utilization to continue."
                )

            if self.pcfg.ideal_utilization > self.pcfg.max_utilization:
                raise FINNMultiFPGAPartitionerError(
                    "Cannot create Multi-FPGA partition if the requested ideal utilization"
                    "is greater than the requested max allowed utilization "
                    f"({self.pcfg.ideal_utilization:.2%} > {self.pcfg.max_utilization:.2%})"
                )

            # Warn about shell utilization
            if self.pcfg.max_utilization >= 0.85:
                log.warning(
                    f"Max utilization per device is set to {self.pcfg.max_utilization:2.2%}. "
                    f"Setting the max utilization too high might cause issues during P&R, "
                    f"due to the shell's own resource requirements, as well as difficulties "
                    f"during routing. Consider decreasing max utilization "
                    f"in case implementation fails."
                )

            # Check that all estimates exist
            missing_estimates = False
            for layer in self.resource_estimates.keys():
                if all(
                    self.resource_estimates[layer][res] <= 0
                    for res in self.resource_estimates[layer].keys()
                ):
                    missing_estimates = True
                    log.critical(
                        f"Layer {layer} has an all-0 resource estimation for all "
                        "resource types. Cannot partition using resource estimates!"
                    )
            if missing_estimates:
                raise FINNMultiFPGAError(
                    "Cannot partition with faulty resource estimations and "
                    "RESOURCE_UTILIZATION as PartitioningStrategy. Check logs to find information"
                    "about which layers have missing resource estimates!"
                )

            # Warn about total resources
            for restype in self.pcfg.considered_resources:
                total_required = sum([rv[restype] for rv in self.resource_estimates.values()])
                total_on_devices = (
                    self.pcfg.max_utilization * self.pcfg.num_fpgas * self.device_resources[restype]
                )
                if total_required > total_on_devices:
                    raise FINNMultiFPGAPartitionerError(
                        f"The model requires a total of "
                        f"{total_required} {restype}, but "
                        f"{self.pcfg.num_fpgas} devices combined only have "
                        f"{total_on_devices} {restype} in total "
                        f"(considering the current max utilization value configured)."
                    )
                factor = total_required / total_on_devices
                ceiled_factor = ceil(factor)
                if (ceiled_factor - factor) < 0.15 and int(ceiled_factor) == int(
                    self.pcfg.num_fpgas
                ):
                    log.warning(
                        f"The total resources of type {restype} required by the "
                        f"model are found on roughly {factor} devices. Since "
                        f"this is just below the number of "
                        f"available devices, partitioning might fail. Too coarse grained "
                        f"layers might make partitioning impossible, "
                        f"despite enough resources "
                        f"being available overall."
                    )

            # Give a warning if resource usage gets close to its maximum
            for node in self.modelwrapper.graph.node:
                for restype in self.resource_estimates[node.name].keys():
                    thresh_percentage = (
                        0.1 if self.pcfg.max_utilization >= 0.1 else self.pcfg.max_utilization
                    )
                    if restype not in self.device_resources.keys():
                        continue
                    warn_threshold = (
                        self.pcfg.max_utilization - thresh_percentage
                    ) * self.device_resources[restype]
                    max_util = self.device_resources[restype] * self.pcfg.max_utilization
                    if self.resource_estimates[node.name][restype] >= warn_threshold:
                        if self.resource_estimates[node.name][restype] < max_util:
                            log.warning(
                                f"Node {node}'s usage of {restype} is within "
                                f"{thresh_percentage:2.2%} of the maximum allowed utilization "
                                f"({self.resource_estimates[node.name][restype]} / "
                                f"{max_util}) "
                                "on a single device. Partitioning might fail!"
                            )
                        else:
                            raise FINNMultiFPGAPartitionerError(
                                f"Node {node}'s usage of {restype} is above "
                                f"the allowed utilization per device "
                                f"({self.resource_estimates[node.name][restype]} > "
                                f"{max_util}). "
                                "Theoretical max per device would "
                                f"be {self.device_resources[restype]}. "
                                "The node cannot fit on a device of this type."
                            )

    def __init__(self, cfg: DataflowBuildConfig, modelwrapper: ModelWrapper) -> None:
        """Create the solver model and run some checks to ensure partitioning is possible."""
        super().__init__(cfg)
        self.modelwrapper = modelwrapper  # (model = MIP Model, modelwrapper = ONNX Model)
        self.model.verbose = 0

        # The status will be set by the _solve method
        # This way we can catch if someone tries to request resource usage before
        # the model was solved
        self.status: mip.OptimizationStatus | None = None

        if self.pcfg.num_fpgas < 1:
            raise NotImplementedError()

        # We need to estimate how many resources the model will likely need
        self.resource_estimates = get_estimated_model_resources(
            self.modelwrapper,
            self.cfg._resolve_fpga_part(),  # noqa
            self.pcfg.considered_resources,
            True,
        )

        # What are the resources available per device?
        if self.cfg.board is None:
            raise FINNMultiFPGAConfigError(
                "Please specify the 'board' parameter in your dataflow config."
            )
        self.device_resources = available_resources_on_platform(
            platforms[self.cfg.board](), self.pcfg.considered_resources
        )

        # What nodes need to stay together?
        self.inseparable_nodes = None
        if self.pcfg.single_stream_network:
            self.inseparable_nodes = get_inseparable_nodes(modelwrapper)

        # Check that no issues are present
        self.run_checks()

        # ---------- LP Definition ----------

        # self.devices[node][device] = 1: Node <node> is on device <device>
        self.devices = {
            node.name: [
                self.model.add_var(name=f"node{node.name}_on_device{device}", var_type=mip.BINARY)
                for device in range(self.pcfg.num_fpgas)
            ]
            for node in modelwrapper.graph.node
        }

        # Every layer can only be on one device
        for node in modelwrapper.graph.node:
            self.model += (
                xsum(
                    self.devices[node.name][device]
                    for device in range(len(self.devices[node.name]))
                )
                == 1
            )

        # Helper to see what device a node is on
        self.chosen_device = {
            node.name: self.model.add_var(
                name=f"chosen_device_of_node_{node.name}", var_type=mip.INTEGER
            )
            for node in modelwrapper.graph.node
        }
        for node in modelwrapper.graph.node:
            self.model += self.chosen_device[node.name] == xsum(
                self.devices[node.name][device] * device for device in range(self.pcfg.num_fpgas)
            )

        # Custom constraints
        for name, device in self.pcfg.custom_partitioning_constraints.items():
            self.model += self.chosen_device[name] == device

        # Grouped nodes need to stay together
        # First check that no group is too large
        if self.inseparable_nodes is not None:
            nodecount = len(modelwrapper.graph.node)
            # Total number of nodes that are part of an inseparable group
            nodes_in_groups = sum(len(group) for group in self.inseparable_nodes)
            # Max number of devices possible: One per group and one per node not in a group
            max_devices_possible = nodecount - nodes_in_groups + len(self.inseparable_nodes)
            for i, group in enumerate(self.inseparable_nodes):
                # 1. Single group larger than the model itself
                if len(group) > nodecount:
                    raise FINNMultiFPGAPartitionerError(
                        f"Group {i} of inseparable nodes is larger than the total set of all "
                        f"nodes in the model. (Has {len(group)} nodes, but only {nodecount} "
                        "nodes in the graph!)"
                    )
                # 2. Num. nodes == Num. groups. Leads to atleast 1 empty device
                if len(group) == nodecount and self.pcfg.num_fpgas > 1:
                    raise FINNMultiFPGAPartitionerError(
                        f"Group {i} has the same number of nodes as the graph in total. However "
                        "since more than 1 device is used, this would result in one device "
                        "being completely empty, leading to an invalid partitioning model."
                    )
            # 3. Not enough devices to have this many nodes in groups
            if self.pcfg.num_fpgas > max_devices_possible:
                raise FINNMultiFPGAPartitionerError(
                    f"Requested number of FPGAs ({self.pcfg.num_fpgas})"
                    f" is larger than the number of "
                    f"devices possible. {nodecount - nodes_in_groups} nodes can be alone on a "
                    f"device, and {len(self.inseparable_nodes)} groups of nodes can be on a "
                    f"device. The largest possible device count partitioning would "
                    f"result in {max_devices_possible} devices"
                )

            # Nodes in groups stay together
            for group in self.inseparable_nodes:
                for i in range(len(group) - 1):
                    first = modelwrapper.graph.node[group[i]].name
                    second = modelwrapper.graph.node[group[i + 1]].name
                    self.model += self.chosen_device[first] == self.chosen_device[second]

        # Number of nodes having this device as their ID
        self.nodesperdevice = [
            self.model.add_var(name=f"lpd_{device}", var_type=mip.INTEGER)
            for device in range(self.pcfg.num_fpgas)
        ]
        for device in range(self.pcfg.num_fpgas):
            self.model += self.nodesperdevice[device] == xsum(
                self.devices[node.name][device] for node in modelwrapper.graph.node
            )

        # Connections that a device has with other devices
        # Helper variable that says whether A is on device D and A->B is a device switch
        self.device_switch: list[dict[str, dict[str, mip.Var]]] = [
            {
                node.name: {
                    suc.name: self.model.add_var(
                        name=f"device_switch_d{device}_from_{node.name}_to_{suc.name}",
                        var_type=mip.INTEGER,
                    )
                    for suc in self.get_successors(node)
                }
                for node in modelwrapper.graph.node
            }
            for device in range(self.pcfg.num_fpgas)
        ]

        # Condition A: NODE is on DEVICE
        # Condition B: SUCCESSOR has a different device than NODE
        # Variable C: If both are given, device_switch is constrained:
        #   C <= 1, C <= 1, C >= 1 (thus C = 1)
        # If one is not true, then
        #   C <= 0, C <= 1, C >= 0 (thus C = 0)
        for device in range(self.pcfg.num_fpgas):
            for node in modelwrapper.graph.node:
                for suc in self.get_successors(node):
                    self.model += (
                        self.device_switch[device][node.name][suc.name]
                        <= self.devices[node.name][device]
                    )
                    self.model += self.device_switch[device][node.name][suc.name] <= (
                        1 - self.devices[suc.name][device]
                    )
                    self.model += self.device_switch[device][node.name][suc.name] >= (
                        self.devices[node.name][device] + (1 - self.devices[suc.name][device]) - 1
                    )

        # Finally calculating the conections per device
        self.connections_per_device = [
            self.model.add_var(name=f"connections_on_device_{device}", var_type=mip.INTEGER)
            for device in range(self.pcfg.num_fpgas)
        ]
        for device in range(self.pcfg.num_fpgas):
            self.model += self.connections_per_device[device] == xsum(
                self.device_switch[device][node.name][suc.name]
                for suc in self.get_successors(node)
                for node in modelwrapper.graph.node
            )

        # Limit the number of connections per device (depends on the FPGAs QSFP ports)
        for device in range(self.pcfg.num_fpgas):
            self.model += self.connections_per_device[device] <= self.pcfg.ports_per_device

        # Consecutive nodes must be on consecutive devices
        self.device_diff: dict[str, dict[str, mip.Var]] = {}
        for node in modelwrapper.graph.node:
            self.device_diff[node.name] = {}
            for suc in self.get_successors(node):
                self.device_diff[node.name][suc.name] = self.model.add_var(
                    name=f"devicediff_{node.name}_{suc.name}", var_type=mip.INTEGER
                )
                self.model += (
                    self.device_diff[node.name][suc.name]
                    >= self.chosen_device[node.name] - self.chosen_device[suc.name]
                )
                self.model += (
                    self.device_diff[node.name][suc.name]
                    >= self.chosen_device[suc.name] - self.chosen_device[node.name]
                )
                self.model += self.device_diff[node.name][suc.name] <= 1
                self.model += self.device_diff[node.name][suc.name] >= 0

        # Setting topology requirements
        input_nodes = [
            node
            for node in modelwrapper.graph.node
            if modelwrapper.find_direct_predecessors(node) is None
        ]
        output_nodes = [
            node
            for node in modelwrapper.graph.node
            if modelwrapper.find_direct_successors(node) is None
        ]
        if len(input_nodes) > 1:
            log.warning(
                "There are multiple input nodes."
                "All input nodes will be constrained to the same "
                "device. To change this, "
                "pass custom device constraints to the partitioner."
            )
        if len(output_nodes) > 1:
            log.warning(
                "There are multiple output nodes."
                "All output nodes will be constrained to the same "
                "device. To change this, "
                "pass custom device constraints to the partitioner."
            )
        match self.pcfg.topology:
            case MFTopology.CHAIN:
                for node in input_nodes:
                    self.model += self.chosen_device[node.name] == 0
                for node in output_nodes:
                    self.model += self.chosen_device[node.name] == self.pcfg.num_fpgas - 1

                # We also need to make sure, that for a chain, every succeeding
                # node has the same or higher device id to prevent going back and forth
                for node in modelwrapper.graph.node:
                    for suc in self.get_successors(node):
                        self.model += self.chosen_device[suc.name] >= self.chosen_device[node.name]

            case MFTopology.RETURNCHAIN:
                for node in input_nodes:
                    self.model += self.chosen_device[node.name] == 0
                for node in output_nodes:
                    self.model += self.chosen_device[node.name] == 0

            case _:
                raise FINNMultiFPGAConfigError(
                    f"Invalid communication scheme for Aurora partitioner: {self.pcfg.topology}"
                )

        # Objective Function
        if self.pcfg.partition_strategy == PartitioningStrategy.LAYER_COUNT:
            # Calculcate the difference to the "ideal" load
            # (All devices have the same number of layers)
            avg_diff = [
                self.model.add_var(var_type=mip.CONTINUOUS) for i in range(self.pcfg.num_fpgas)
            ]
            avg_ideal_load = len(modelwrapper.graph.node) / self.pcfg.num_fpgas
            for i in range(self.pcfg.num_fpgas):
                self.model += avg_diff[i] >= self.nodesperdevice[i] - avg_ideal_load  # type: ignore
                self.model += avg_diff[i] >= avg_ideal_load - self.nodesperdevice[i]  # type: ignore

            # Get the largest of those differences
            max_diff = self.model.add_var(name="max_diff", var_type=mip.CONTINUOUS)
            for device in range(self.pcfg.num_fpgas):
                self.model += max_diff >= avg_diff[device]

            # Try to minimize the max difference to ideal
            self.model.objective = max_diff
            self.model.sense = mip.MINIMIZE

        elif self.pcfg.partition_strategy == PartitioningStrategy.RESOURCE_UTILIZATION:
            # Collect the resource usage of all nodes on a device
            self.resource_use_int = {}
            for device in range(self.pcfg.num_fpgas):
                self.resource_use_int[device] = {}
                for resource_name in self.pcfg.considered_resources:
                    self.resource_use_int[device][resource_name] = self.model.add_var(
                        f"resource_use_int_device{device}_resource{resource_name}",
                        var_type=mip.INTEGER,
                    )
                    self.model += self.resource_use_int[device][resource_name] == xsum(
                        [
                            self.devices[node.name][device]
                            * self.resource_estimates[node.name][resource_name]
                            for node in modelwrapper.graph.node
                            if resource_name in self.resource_estimates[node.name].keys()
                        ]
                    )

            # Limit resource usage to (available resources * max usage in percent)
            for device in range(self.pcfg.num_fpgas):
                for resource_name in self.pcfg.considered_resources:
                    max_resources = int(
                        self.device_resources[resource_name] * self.pcfg.max_utilization
                    )
                    self.model += self.resource_use_int[device][resource_name] <= max_resources

            # Balance so that the maximum difference to the ideal load over all devices and
            # resources is as low as possible
            # Use relative values because resources are available at vastly different scales
            self.resource_diff = {}
            self.resource_use_relative = {}
            for device in range(self.pcfg.num_fpgas):
                self.resource_diff[device] = {}
                self.resource_use_relative[device] = {}
                for resource_name in self.pcfg.considered_resources:
                    self.resource_diff[device][resource_name] = self.model.add_var(
                        name=f"resource_diff_to_ideal_device{device}_resource{resource_name}",
                        var_type=mip.CONTINUOUS,
                    )
                    self.resource_use_relative[device][resource_name] = self.model.add_var(
                        name=f"resource_use_cont_device{device}_resource{resource_name}",
                        var_type=mip.CONTINUOUS,
                    )

                    # Resource util in relative terms (0-1)
                    self.model += self.resource_use_relative[device][resource_name] == (
                        self.resource_use_int[device][resource_name]
                        / self.device_resources[resource_name]
                    )

                    # Convert to float and get diff
                    self.model += self.resource_diff[device][resource_name] >= (
                        self.resource_use_relative[device][resource_name]
                        - self.pcfg.ideal_utilization
                    )
                    self.model += self.resource_diff[device][resource_name] >= (
                        self.pcfg.ideal_utilization
                        - self.resource_use_relative[device][resource_name]
                    )

            # A device cannot be completely empty
            for device in range(self.pcfg.num_fpgas):
                self.model += (
                    xsum(
                        [
                            self.resource_use_relative[device][res]
                            for res in self.pcfg.considered_resources
                        ]
                    )
                    # Needs to be really small so the model is still valid for very small designs
                    >= 0.0000001
                )

            # The min resource diff to ideal on a device, regardless of resource type
            # (If ideal is 70%, and we have LUT: 61% and DSP: 32%,
            # then we use 61%, so diff is 70%-61%=9%)
            self.min_resource_diff = []
            for device in range(self.pcfg.num_fpgas):
                self.min_resource_diff.append(
                    self.model.add_var(f"min_resource_diff_device{device}", var_type=mip.CONTINUOUS)
                )
                for res in self.pcfg.considered_resources:
                    self.model += self.min_resource_diff[device] <= self.resource_diff[device][res]

                    # If we dont specify this, it will stay at the initial value of 0,
                    # since 0 is smaller than all the resource_diffs
                    self.model += self.min_resource_diff[device] >= 0.000000001

            # Maximum of the min resource diff of all devices
            max_diff = self.model.add_var("max_diff", var_type=mip.CONTINUOUS)
            for device in range(self.pcfg.num_fpgas):
                self.model += max_diff >= xsum(
                    [self.resource_diff[device][res] for res in self.pcfg.considered_resources]
                )

            # Set objective function
            self.model.objective = max_diff
            self.model.sense = mip.MINIMIZE

        else:
            raise FINNMultiFPGAConfigError(
                f"Unknown partitioning strategy: {self.pcfg.partition_strategy}"
            )

    def create_result(self) -> dict[str, int]:
        """Create a uniform result mapping from the internal model. Refer to baseclass docstring
        for more information.
        """
        for node in self.modelwrapper.graph.node:
            if self.chosen_device[node.name].x is None:
                raise FINNMultiFPGAError(
                    "Cannot create result "
                    "partition mapping. Partitioning either has not "
                    "been done or the model could not be solved."
                )
        return {
            node.name: int(self.chosen_device[node.name].x)  # type: ignore
            for node in self.modelwrapper.graph.node
        }

    def _get_resource_use_relative(self) -> dict[int, dict[str, Any]]:
        """Get relative resource usage per device and resource type."""
        if self.status is None:
            raise FINNMultiFPGAError(
                "Resource utilization per device was requested "
                "before the model was solved. Please call solve() first."
            )
        data = {}
        for device in range(self.pcfg.num_fpgas):
            data[device] = {}
            for restype in self.resource_use_relative[device].keys():
                data[device][restype] = self.resource_use_relative[device][restype].x
        return data

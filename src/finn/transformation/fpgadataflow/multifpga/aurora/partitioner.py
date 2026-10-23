"""AuroraFlow partitioner for MultiFPGA."""
# pyright: reportOperatorIssue=false

import mip
from math import ceil
from mip import xsum
from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING, cast, override

from finn.builder.build_dataflow_config import DataflowBuildConfig, MFTopology, PartitioningStrategy
from finn.transformation.fpgadataflow.multifpga.graph import get_inseparable_nodes
from finn.transformation.fpgadataflow.multifpga.milp_partitioner import MILPPartitioner
from finn.util.exception import (
    FINNInternalError,
    FINNMultiFPGAConfigError,
    FINNMultiFPGAError,
    FINNMultiFPGAPartitionerError,
    FINNMultiFPGAUnsolvableError,
)
from finn.util.logging import log
from finn.util.resources import PLATFORM_RESOURCE_TYPES, available_resources_on_platform

if TYPE_CHECKING:
    from numbers import Real


class PointToPointBidirectionalFixedPartitioner(MILPPartitioner):
    """Partitioner for point-to-point (potentially) bidirectional
    connections between fixed devices.

    Originally only for AuroraFlow.

    Assumes that every connection is potentially duplex, and
    that every connection has a fixed target (i.e. no online/dynamic routing).

    For similar previous work refer to Elastic-DF.

    This class assumes that `links_per_device` refers to bidirectional links.
    (1 link to a device can hold 2 connections, back and forth).
    """

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
        if self.mfcfg.num_fpgas > len(self.modelwrapper.graph.node):
            # Stop if there are more devices than nodes
            raise FINNMultiFPGAUnsolvableError(
                f"Model infeasible: Cannot partition a model with "
                f"{len(self.modelwrapper.graph.node)} nodes to {self.mfcfg.num_fpgas} devices!"
            )

        # Custom constraints
        node_names = {node.name for node in self.modelwrapper.graph.node}
        for name, device in self.mfcfg.custom_partitioning_constraints.items():
            # A node name that is not part of the model (e.g. a typo, or a name that changed
            # because the model was modified) would otherwise fail with a bare KeyError
            if name not in node_names:
                raise FINNMultiFPGAConfigError(
                    f"Custom partitioning constraint for node {name} cannot be applied, since the "
                    f"model contains no node of that name."
                )
            # A negative device ID can never be satisfied, regardless of the number of devices
            if device < 0:
                raise FINNMultiFPGAConfigError(
                    f"Custom partitioning constraint for node {name} assigns the invalid "
                    f"device {device}. Device IDs start at 0."
                )
            # A device ID that is too large for the current number of devices makes the model
            # unsolvable, but automatic partitioning can still try a higher number of devices.
            if device >= self.mfcfg.num_fpgas:
                raise FINNMultiFPGAUnsolvableError(
                    f"Custom partitioning constraint for node {name} assigns device {device}, but "
                    f"only {self.mfcfg.num_fpgas} devices are used."
                )

        # Inseparable groups: Check that no group is too large
        if self.inseparable_nodes is not None:
            nodecount = len(self.modelwrapper.graph.node)
            # Total number of nodes that are part of an inseparable group
            nodes_in_groups = sum(len(group) for group in self.inseparable_nodes)
            # Max number of devices possible: One per group and one per node not in a group
            max_devices_possible = nodecount - nodes_in_groups + len(self.inseparable_nodes)
            for i, group in enumerate(self.inseparable_nodes):
                # 1. Single group larger than the model itself
                if len(group) > nodecount:
                    raise FINNMultiFPGAUnsolvableError(
                        f"Group {i} of inseparable nodes is larger than the total set of all "
                        f"nodes in the model. (Has {len(group)} nodes, but only {nodecount} "
                        "nodes in the graph!)"
                    )
                # 2. A single group contains all nodes. Leads to at least 1 empty device
                if len(group) == nodecount and self.mfcfg.num_fpgas > 1:
                    raise FINNMultiFPGAUnsolvableError(
                        f"Group {i} has the same number of nodes as the graph in total. However "
                        "since more than 1 device is used, this would result in one device "
                        "being completely empty, leading to an invalid partitioning model."
                    )
            # 3. Not enough devices to have this many nodes in groups
            if self.mfcfg.num_fpgas > max_devices_possible:
                raise FINNMultiFPGAUnsolvableError(
                    f"Requested number of FPGAs ({self.mfcfg.num_fpgas})"
                    f" is larger than the number of "
                    f"devices possible. {nodecount - nodes_in_groups} nodes can be alone on a "
                    f"device, and {len(self.inseparable_nodes)} groups of nodes can be on a "
                    f"device. The largest possible device count partitioning would "
                    f"result in {max_devices_possible} devices"
                )

        if self.mfcfg.partition_strategy in [
            PartitioningStrategy.BALANCE_RESOURCE,
            PartitioningStrategy.BALANCE_RESOURCE_WEIGHTED,
            PartitioningStrategy.MINIMIZE_MAX_RESOURCE,
        ]:
            if self.mfcfg.max_utilization > 1.0:
                raise FINNMultiFPGAPartitionerError(
                    f"Max utilization was set to {self.mfcfg.max_utilization:2.2%} > "
                    f"100%. Decrease max_utilization to continue."
                )

            # Warn about shell utilization
            if self.mfcfg.max_utilization >= 0.85:
                log.warning(
                    f"Max utilization per device is set to {self.mfcfg.max_utilization:2.2%}. "
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
                    "Cannot partition with faulty resource estimations and the resource based "
                    f"partitioning strategy {self.mfcfg.partition_strategy.name}. Check logs to "
                    "find information about which layers have missing resource estimates!"
                )

            # Check that the devices combined have enough resources
            for restype in self.mfcfg.considered_resources:
                total_required = sum([rv[restype] for rv in self.resource_estimates.values()])
                total_on_devices = (
                    self.mfcfg.max_utilization
                    * self.mfcfg.num_fpgas
                    * self.device_resources[restype]
                )
                if total_required > total_on_devices:
                    raise FINNMultiFPGAUnsolvableError(
                        f"The model requires a total of "
                        f"{total_required} {restype}, but "
                        f"{self.mfcfg.num_fpgas} devices combined only have "
                        f"{total_on_devices} {restype} in total "
                        f"(considering the current max utilization value configured)."
                    )
                factor = total_required / total_on_devices
                ceiled_factor = ceil(factor)
                if (ceiled_factor - factor) < 0.15 and int(ceiled_factor) == int(
                    self.mfcfg.num_fpgas
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

            # Error about resources that are ignored by the solver, but heavily used by the model
            all_resources_available = available_resources_on_platform(self.board, None)
            for restype in PLATFORM_RESOURCE_TYPES:
                if restype in self.mfcfg.considered_resources:
                    continue
                total_required = sum(rv.get(restype, 0) for rv in self.resource_estimates.values())
                max_on_device = self.mfcfg.max_utilization * all_resources_available[restype]
                if total_required > 0.5 * max_on_device:
                    log.error(
                        f"The model requires a total of {total_required} {restype}, which is more "
                        f"than half of the allowed utilization of a single device "
                        f"({max_on_device} {restype} at {self.mfcfg.max_utilization:.2%} max "
                        f"utilization). However {restype} is not in considered_resources, so "
                        f"the partitioner ignores it and might overutilize {restype} on a "
                        f"device. Consider adding {restype} to considered_resources."
                    )

            # Give a warning if resource usage gets close to its maximum
            for node in self.modelwrapper.graph.node:
                for restype in self.resource_estimates[node.name].keys():
                    thresh_percentage = (
                        0.1 if self.mfcfg.max_utilization >= 0.1 else self.mfcfg.max_utilization
                    )
                    if restype not in self.device_resources.keys():
                        continue
                    warn_threshold = (
                        self.mfcfg.max_utilization - thresh_percentage
                    ) * self.device_resources[restype]
                    max_util = self.device_resources[restype] * self.mfcfg.max_utilization
                    if self.resource_estimates[node.name][restype] >= warn_threshold:
                        if self.resource_estimates[node.name][restype] < max_util:
                            log.warning(
                                f"Node {node.name}'s usage of {restype} is within "
                                f"{thresh_percentage:2.2%} of the maximum allowed utilization "
                                f"({self.resource_estimates[node.name][restype]} / "
                                f"{max_util}) "
                                "on a single device. Partitioning might fail!"
                            )
                        else:
                            raise FINNMultiFPGAUnsolvableError(
                                f"Node {node.name}'s usage of {restype} is above "
                                f"the allowed utilization per device "
                                f"({self.resource_estimates[node.name][restype]} > "
                                f"{max_util}). "
                                "Theoretical max per device would "
                                f"be {self.device_resources[restype]}. "
                                "The node cannot fit on a device of this type."
                            )

        # Chains and rings need one link to each neighbouring device. A link is duplex, so a
        # device with a single neighbour (2 devices, or the ends of a chain) needs only one link.
        if self.mfcfg.topology in [
            MFTopology.SIMPLEX_CHAIN,
            MFTopology.DUPLEX_CHAIN,
            MFTopology.RING,
        ]:
            required_links = min(self.mfcfg.num_fpgas - 1, 2)
            if self.mfcfg.links_per_device < required_links:
                raise FINNMultiFPGAConfigError(
                    f"The topology {self.mfcfg.topology.name} with {self.mfcfg.num_fpgas} devices "
                    f"requires {required_links} bidirectional links per device, "
                    f"but links_per_device is "
                    f"{self.mfcfg.links_per_device}."
                )

        # If we can expect a fork, some strategies may not yield the desired mapping
        if (
            self.mfcfg.topology
            in [MFTopology.SIMPLEX_CHAIN, MFTopology.DUPLEX_CHAIN, MFTopology.RING]
            and not self.mfcfg.single_stream_network
        ):
            log.error(
                f"The configuration allows for forks and joins across different devices "
                f"(single_stream_network=False). The topology {self.mfcfg.topology.name} "
                f"however does not allow forks/joins across different devices. It is highly "
                f"recommended to switch to another topology if the aforementioned "
                f"fork/join case should be allowed."
            )

    def __init__(self, cfg: DataflowBuildConfig, modelwrapper: ModelWrapper) -> None:
        """Run some checks to ensure partitioning is possible. The MILP itself is defined in
        `_build_model` once `solve()` is called.
        """
        # (model = MIP Model, modelwrapper = ONNX Model)
        super().__init__(modelwrapper, cfg)

        if self.mfcfg.num_fpgas < 1:
            raise FINNInternalError(
                "Number of FPGAs for MultiFPGA was set below 1 when trying to create the MIP model."
                " Automatic partitioning should have prevented this."
            )

        # What are the resources available per device?
        self.device_resources = available_resources_on_platform(
            self.board, self.mfcfg.considered_resources
        )

        # What nodes need to stay together?
        self.inseparable_nodes = None
        if self.mfcfg.single_stream_network:
            self.inseparable_nodes = get_inseparable_nodes(modelwrapper)

        # Check that no issues are present
        self.run_checks()

    # ---------- LP Definition ----------
    @override
    def _build_model(self) -> None:
        """Define the variables, constraints and the objective of the partitioning MILP."""
        self.model.verbose = 0

        # Define some helper lists
        input_nodes = [
            node
            for node in self.modelwrapper.graph.node
            if self.modelwrapper.find_direct_predecessors(node) is None
        ]
        output_nodes = [
            node
            for node in self.modelwrapper.graph.node
            if self.modelwrapper.find_direct_successors(node) is None
        ]
        if len(input_nodes) > 1:
            log.warning(
                "There are multiple input nodes. "
                "All input nodes will be constrained to the same "
                "device. To change this, "
                "pass custom device constraints to the partitioner."
            )
        if len(output_nodes) > 1:
            log.warning(
                "There are multiple output nodes. "
                "All output nodes will be constrained to the same "
                "device. To change this, "
                "pass custom device constraints to the partitioner."
            )

        # self.devices[node][device] = 1: Node <node> is on device <device>
        self.devices = {
            node.name: [
                self.model.add_var(name=f"node{node.name}_on_device{device}", var_type=mip.BINARY)
                for device in range(self.mfcfg.num_fpgas)
            ]
            for node in self.modelwrapper.graph.node
        }

        # Every layer can only be on one device
        for node in self.modelwrapper.graph.node:
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
            for node in self.modelwrapper.graph.node
        }
        for node in self.modelwrapper.graph.node:
            self.model += self.chosen_device[node.name] == xsum(
                self.devices[node.name][device] * device for device in range(self.mfcfg.num_fpgas)
            )

        # Custom constraints (validated in run_checks)
        for name, device in self.mfcfg.custom_partitioning_constraints.items():
            self.model += self.chosen_device[name] == device

        # Grouped nodes need to stay together (group sizes are validated in run_checks)
        if self.inseparable_nodes is not None:
            for group in self.inseparable_nodes:
                for i in range(len(group) - 1):
                    self.model += self.chosen_device[group[i]] == self.chosen_device[group[i + 1]]

        # Number of nodes having this device as their ID
        self.nodesperdevice = [
            self.model.add_var(name=f"lpd_{device}", var_type=mip.INTEGER)
            for device in range(self.mfcfg.num_fpgas)
        ]
        for device in range(self.mfcfg.num_fpgas):
            self.model += self.nodesperdevice[device] == xsum(
                self.devices[node.name][device] for node in self.modelwrapper.graph.node
            )

        # A device cannot be completely empty, regardless of topology and partitioning strategy
        for device in range(self.mfcfg.num_fpgas):
            self.model += self.nodesperdevice[device] >= 1

        # Count in/out edges, where one of the nodes is on the given device
        # self.edges[device][node a][node b] == 1 if a and b are on different devices,
        # one of them being <device>
        self.edges: dict[int, dict[str, dict[str, mip.Var]]] = {
            d: {} for d in range(self.mfcfg.num_fpgas)
        }
        self.connections_per_device = [
            self.model.add_var(name=f"connections_on_device_{device}", var_type=mip.INTEGER)
            for device in range(self.mfcfg.num_fpgas)
        ]
        for node in self.modelwrapper.graph.node:
            suc = self.modelwrapper.find_direct_successors(node)
            if suc is None:
                continue
            for successor in suc:
                for d in range(self.mfcfg.num_fpgas):
                    if node.name not in self.edges[d]:
                        self.edges[d][node.name] = {}
                    if successor.name not in self.edges[d][node.name]:
                        self.edges[d][node.name][successor.name] = self.model.add_var(
                            name=f"edge_{d}_{node.name}_{successor.name}", var_type=mip.BINARY
                        )

                    # 1)
                    # Lower bound: the absolute difference of the is-on-device-d variables of the
                    # two nodes. This is 1 if exactly one of the nodes is on device d, and 0 if
                    # both or neither are on d (also if they are on two different devices other
                    # than d)
                    self.model += self.edges[d][node.name][successor.name] >= (
                        self.devices[node.name][d] - self.devices[successor.name][d]
                    )
                    self.model += self.edges[d][node.name][successor.name] >= (
                        self.devices[successor.name][d] - self.devices[node.name][d]
                    )

                    # 2)
                    # Necessary check to upper bound the value to 0 if both nodes
                    # are on device d.
                    self.model += self.edges[d][node.name][successor.name] <= (
                        2 - self.devices[node.name][d] - self.devices[successor.name][d]
                    )

                    # 3)
                    # Upper bound to 0 if neither node is on device d
                    self.model += self.edges[d][node.name][successor.name] <= (
                        self.devices[node.name][d] + self.devices[successor.name][d]
                    )

                    # If a and b are on the same device (device d):
                    # e >= 0 (1), e <= 0 (2), e <= 2 (3), ===> e = 0
                    # If a and b are on different devices (but one is _d_):
                    # e >= 1 (1), e <= 1 (2), e <= 1 (3), ===> e = 1
                    # If neither a nor b are on device d:
                    # e >= 0 (1), e <= 2 (2), e <= 0 (3), ===> e = 0
                    #
                    # So since the variable is only 1, iff both are on different devices,
                    # one of them being d, we can count the total number of connections
                    # by summing over all edges

        # Finally calculating the connections per device
        # This refer to UNIDIRECTIONAL CONNECTIONS.
        # This means a device with connections_per_device == 2 and mfcfg.links_per_device == 1
        # work together fine, as long as both connections are to/from the same device.
        for device in range(self.mfcfg.num_fpgas):
            self.model += self.connections_per_device[device] == xsum(
                self.edges[device][a][b] for a in self.edges[device] for b in self.edges[device][a]
            )

        # Without a fixed topology, only the total can be limited: a duplex link carries at most
        # one connection per direction. This is necessary but not sufficient, since both
        # directions of a link must lead to the same partner device.
        if self.mfcfg.topology in [
            MFTopology.ARBITRARY,
            MFTopology.ARBITRARY_OPEN,
            MFTopology.ARBITRARY_CLOSED,
        ]:
            for device in range(self.mfcfg.num_fpgas):
                self.model += self.connections_per_device[device] <= 2 * self.mfcfg.links_per_device

        # Consecutive nodes must be on consecutive devices
        self.device_diff: dict[str, dict[str, mip.Var]] = {}
        for node in self.modelwrapper.graph.node:
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

                # Topology-specific constraints regarding device diffs
                match self.mfcfg.topology:
                    case MFTopology.SIMPLEX_CHAIN:
                        # In a simplex chain, the device order is strict, and no
                        # jumps or forks are allowed
                        self.model += self.device_diff[node.name][suc.name] <= 1
                        self.model += self.device_diff[node.name][suc.name] >= 0
                    case MFTopology.DUPLEX_CHAIN:
                        # In a duplex chain, the same goes, A <==> B may be 1 device ID apart
                        self.model += self.device_diff[node.name][suc.name] <= 1
                        self.model += self.device_diff[node.name][suc.name] >= 0
                    case MFTopology.RING:
                        # In a ring, just as in a simplex chain, the device difference is almost
                        # always 1: Only the last jump from N back to 0 is allowed to be larger
                        if suc.name not in [node.name for node in output_nodes]:
                            self.model += self.device_diff[node.name][suc.name] <= 1
                            self.model += self.device_diff[node.name][suc.name] >= 0
                    case (
                        MFTopology.ARBITRARY
                        | MFTopology.ARBITRARY_OPEN
                        | MFTopology.ARBITRARY_CLOSED
                    ):
                        # No constraints on the device difference (warned once below)
                        pass
                    case _:
                        raise FINNInternalError(
                            f"Device difference constraints for topology "
                            f"{self.mfcfg.topology.name} are not implemented!"
                        )

        # Setting topology requirements
        match self.mfcfg.topology:
            case MFTopology.SIMPLEX_CHAIN:
                for node in input_nodes:
                    self.model += self.chosen_device[node.name] == 0
                for node in output_nodes:
                    self.model += self.chosen_device[node.name] == self.mfcfg.num_fpgas - 1

                # We also need to make sure, that for a chain, every succeeding
                # node has the same or higher device id to prevent going back and forth
                for node in self.modelwrapper.graph.node:
                    for suc in self.get_successors(node):
                        self.model += self.chosen_device[suc.name] >= self.chosen_device[node.name]

                # Every inner device has 2 connections (one in, one out),
                # and every outer device only 1
                for device in range(self.mfcfg.num_fpgas):
                    if device in [0, self.mfcfg.num_fpgas - 1]:
                        self.model += self.connections_per_device[device] == 1
                    else:
                        self.model += self.connections_per_device[device] == 2

            case MFTopology.DUPLEX_CHAIN:
                for node in input_nodes:
                    self.model += self.chosen_device[node.name] == 0
                for node in output_nodes:
                    self.model += self.chosen_device[node.name] == 0

                # Here we limit the connections of inner devices to exactly 4, and
                # the connections of outer devices to 2. Together with the max 1 ID difference
                # between devices, this forces a duplex chain
                for device in range(self.mfcfg.num_fpgas):
                    if device in [0, self.mfcfg.num_fpgas - 1]:
                        self.model += self.connections_per_device[device] == 2
                    else:
                        self.model += self.connections_per_device[device] == 4

            case MFTopology.RING:
                for node in input_nodes:
                    self.model += self.chosen_device[node.name] == 0
                for node in output_nodes:
                    self.model += self.chosen_device[node.name] == 0

                # Same constraint as in SIMPLEX_CHAIN, with the exception of the last node
                for node in self.modelwrapper.graph.node:
                    for suc in self.get_successors(node):
                        if suc not in output_nodes:
                            self.model += (
                                self.chosen_device[suc.name] >= self.chosen_device[node.name]
                            )
                        else:
                            # The ring closes only from the last device. Otherwise, with forks,
                            # a branch could jump back to device 0 early (e.g. from device 1).
                            self.model += self.chosen_device[node.name] == self.mfcfg.num_fpgas - 1

                # A ring has 2 connections for every device
                for device in range(self.mfcfg.num_fpgas):
                    self.model += self.connections_per_device[device] == 2

            case MFTopology.ARBITRARY:
                log.warning(
                    "Topology ARBITRARY was chosen. The MILP will not be constrained to a "
                    "specific topology! Connections between devices are not necessarily "
                    "monotonic or fixed (e.g. 0 -> 12 -> 86 -> 1 is allowed). Be aware that "
                    "this may result in non-performant networking patterns."
                )

            case MFTopology.ARBITRARY_OPEN:
                log.warning(
                    f"Topology {self.mfcfg.topology.name} chosen. No "
                    f"constraints for inner connections will be set."
                )
                if self.mfcfg.single_stream_network:
                    log.warning(
                        f"Topology {self.mfcfg.topology.name} chosen. single_stream_network=True "
                        f"is set - more restrictive topologies could be used."
                    )
                for node in input_nodes:
                    self.model += self.chosen_device[node.name] == 0
                for node in output_nodes:
                    self.model += self.chosen_device[node.name] == self.mfcfg.num_fpgas - 1

            case MFTopology.ARBITRARY_CLOSED:
                log.warning(
                    f"Topology {self.mfcfg.topology.name} chosen. No "
                    f"constraints for inner connections will be set."
                )
                if self.mfcfg.single_stream_network:
                    log.warning(
                        f"Topology {self.mfcfg.topology.name} chosen. single_stream_network=True "
                        f"is set - more restrictive topologies could be used."
                    )
                for node in input_nodes:
                    self.model += self.chosen_device[node.name] == 0
                for node in output_nodes:
                    self.model += self.chosen_device[node.name] == 0

            case _:
                raise FINNMultiFPGAConfigError(
                    f"Invalid communication scheme for Aurora partitioner: {self.mfcfg.topology}"
                )

        # Objective Function
        if self.mfcfg.partition_strategy == PartitioningStrategy.LAYER_COUNT:
            # Calculate the difference to the "ideal" load
            # (All devices have the same number of layers)
            avg_diff = [
                self.model.add_var(var_type=mip.CONTINUOUS) for _ in range(self.mfcfg.num_fpgas)
            ]
            avg_ideal_load = len(self.modelwrapper.graph.node) / self.mfcfg.num_fpgas
            for i in range(self.mfcfg.num_fpgas):
                self.model += avg_diff[i] >= self.nodesperdevice[i] - avg_ideal_load  # type: ignore
                self.model += avg_diff[i] >= avg_ideal_load - self.nodesperdevice[i]  # type: ignore

            # Get the largest of those differences
            max_diff = self.model.add_var(name="max_diff", var_type=mip.CONTINUOUS)
            for device in range(self.mfcfg.num_fpgas):
                self.model += max_diff >= avg_diff[device]

            # Try to minimize the max difference to ideal
            self.model.objective = max_diff
            self.model.sense = mip.MINIMIZE

        elif self.mfcfg.partition_strategy in [
            PartitioningStrategy.BALANCE_RESOURCE,
            PartitioningStrategy.BALANCE_RESOURCE_WEIGHTED,
            PartitioningStrategy.MINIMIZE_MAX_RESOURCE,
        ]:
            # Collect the resource usage of all nodes on a device
            self.resource_use_int = {}
            for device in range(self.mfcfg.num_fpgas):
                self.resource_use_int[device] = {}
                for resource_name in self.mfcfg.considered_resources:
                    self.resource_use_int[device][resource_name] = self.model.add_var(
                        f"resource_use_int_device{device}_resource{resource_name}",
                        var_type=mip.INTEGER,
                    )
                    self.model += self.resource_use_int[device][resource_name] == xsum(
                        [
                            self.devices[node.name][device]
                            * self.resource_estimates[node.name][resource_name]
                            for node in self.modelwrapper.graph.node
                            if resource_name in self.resource_estimates[node.name].keys()
                        ]
                    )

            # Limit resource usage to (available resources * max usage in percent)
            for device in range(self.mfcfg.num_fpgas):
                for resource_name in self.mfcfg.considered_resources:
                    max_resources = int(
                        self.device_resources[resource_name] * self.mfcfg.max_utilization
                    )
                    self.model += self.resource_use_int[device][resource_name] <= max_resources

            # Balance using relative values, because resources are available at vastly different
            # scales
            self.resource_use_relative = {}
            for device in range(self.mfcfg.num_fpgas):
                self.resource_use_relative[device] = {}
                for resource_name in self.mfcfg.considered_resources:
                    self.resource_use_relative[device][resource_name] = self.model.add_var(
                        name=f"resource_use_cont_device{device}_resource{resource_name}",
                        var_type=mip.CONTINUOUS,
                    )

                    # Resource util in relative terms (0-1)
                    self.model += self.resource_use_relative[device][resource_name] == (
                        self.resource_use_int[device][resource_name]
                        / self.device_resources[resource_name]
                    )

            # Ideal resource utilization is I_r = (sum (d in D) R_rd) / |D|
            # (R_rd = relative resource usage of type r on device d)
            # (|D| = number of devices)
            self.ideals = {}
            for resource in self.mfcfg.considered_resources:
                self.ideals[resource] = self.model.add_var(
                    name=f"ideal_{resource}", var_type=mip.CONTINUOUS
                )
                self.model += self.ideals[resource] == (
                    xsum(
                        self.resource_use_relative[d][resource] for d in range(self.mfcfg.num_fpgas)
                    )
                    / self.mfcfg.num_fpgas
                )

            # Model variable that checks the difference in relative utilization compared to ideal
            # The difference taken to ideal is the max over all resource types and devices
            # t = max (over r,d) |R_rd - I_r|
            self._diff_to_ideal = {}
            self.diff_to_ideal = self.model.add_var(
                name="worst_diff_to_ideal", var_type=mip.CONTINUOUS
            )

            # Create helpers for each case (needed later again)
            for device in range(self.mfcfg.num_fpgas):
                self._diff_to_ideal[device] = {}
                for resource in self.mfcfg.considered_resources:
                    self._diff_to_ideal[device][resource] = self.model.add_var(
                        name=f"diff_to_ideal_{device}_{resource}", var_type=mip.CONTINUOUS
                    )
                    self.model += self._diff_to_ideal[device][resource] >= (
                        self.resource_use_relative[device][resource] - self.ideals[resource]
                    )
                    self.model += self._diff_to_ideal[device][resource] >= (
                        self.ideals[resource] - self.resource_use_relative[device][resource]
                    )

            # Take the max
            for device in range(self.mfcfg.num_fpgas):
                for resource in self.mfcfg.considered_resources:
                    self.model += self.diff_to_ideal >= self._diff_to_ideal[device][resource]

            # Similar to above with a small weighted sum added
            # Useful if the partition would otherwise be so granular, that smaller differences
            # to ideal resources would not be balanced anymore. This term causes the objective
            # to be reduced further.
            # 0.1 instead of 1.0 to clearly favor the worst difference as the deciding factor.
            weight_factor = 0.1 / (self.mfcfg.num_fpgas * len(self.mfcfg.considered_resources))
            self.diff_to_ideal_weighted = self.model.add_var(
                name="worst_diff_to_ideal_weighted", var_type=mip.CONTINUOUS
            )
            self.model += self.diff_to_ideal_weighted == self.diff_to_ideal + weight_factor * xsum(
                xsum(
                    self._diff_to_ideal[device][resource] for device in range(self.mfcfg.num_fpgas)
                )
                for resource in self.mfcfg.considered_resources
            )

            # max_util: the maximum relative use over all devices and resource types.
            self.max_util = self.model.add_var(
                "max_util", var_type=mip.CONTINUOUS, lb=cast("Real", 0.0)
            )
            for device in range(self.mfcfg.num_fpgas):
                for resource in self.mfcfg.considered_resources:
                    self.model += self.max_util >= self.resource_use_relative[device][resource]

            # Objective function
            # TODO: Connection costs for models that split/join across different devices
            # For detailed explanations refer to their docstrings (build_dataflow_config.py)
            match self.mfcfg.partition_strategy:
                case PartitioningStrategy.BALANCE_RESOURCE:
                    self.model.objective = self.diff_to_ideal
                case PartitioningStrategy.BALANCE_RESOURCE_WEIGHTED:
                    self.model.objective = self.diff_to_ideal_weighted
                case PartitioningStrategy.MINIMIZE_MAX_RESOURCE:
                    self.model.objective = self.max_util
                case _:
                    raise FINNInternalError("Unreachable path detected.")

            # Set sense
            self.model.sense = mip.MINIMIZE

        else:
            raise FINNMultiFPGAConfigError(
                f"Unknown partitioning strategy: {self.mfcfg.partition_strategy}"
            )

    @override
    def _create_result(self) -> dict[str, int]:
        """Create a uniform result mapping from the internal model. Refer to baseclass docstring
        for more information.
        """
        result = {}
        for node in self.modelwrapper.graph.node:
            value = self.chosen_device[node.name].x
            if value is None:
                raise FINNMultiFPGAError(
                    "Cannot create result "
                    "partition mapping. Partitioning either has not "
                    "been done or the model could not be solved."
                )
            # Solvers only keep integer variables integral up to a tolerance (e.g. 0.9999999)
            device = round(value)  # type: ignore
            if abs(device - value) > self.model.integer_tol:  # type: ignore
                raise FINNInternalError(
                    f"The solver assigned the non-integer device {value} to node {node.name}. "
                    f"This should not happen, since the chosen device is an INTEGER variable."
                )
            result[node.name] = device
        return result

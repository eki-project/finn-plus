from __future__ import annotations

import mip
from abc import ABC, abstractmethod
from mip import Model, xsum
from pathlib import Path
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from typing import TYPE_CHECKING, Any

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    MFCommunicationKernel,
    MFTopology,
    PartitioningStrategy,
)
from finn.transformation.fpgadataflow.multifpga_utils import (
    available_resources,
    get_estimated_model_resources,
    get_inseperable_nodes,
    is_single_in_out_model,
    set_device_id,
)
from finn.util.basic import make_build_dir
from finn.util.logging import log
from finn.util.platforms import platforms

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

    from finn.transformation.fpgadataflow.multifpga_network import Device


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
        function to determine load.

    To implement a custom partitioner, currently one needs to implement the following methods:
    __init__
    _solve
    _write_solution_data
    _get_resource_use_by_device"""

    def __init__(
        self,
        strategy: PartitioningStrategy,
        topology: MFTopology | None,
        devices: int,
        nodes: int,
        inseperable_nodes: list[list[int]],
        resources_per_device: dict,
        resource_estimates: dict | None = None,
        considered_resources: list[str] | None = None,
        network_ports_per_device: int = 2,
        max_utilization: float | None = None,
        ideal_utilization: float | None = None,
    ) -> None:
        self.strategy = strategy
        self.max_utilization = max_utilization
        self.topology = topology
        self.ideal_util = ideal_utilization
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
        self.network_ports_per_device = network_ports_per_device
        try:
            self.model = Model()
        except OSError:
            log.warning(
                "Creation of mip.Model failed. This might be known bug "
                "(LD_LIBRARY_PATH only modified at runtime to point to "
                "libgurobi instead of before). Falling back to CBC"
            )
            self.model = Model(solver_name=mip.CBC)
        self.latest_snapshot_path: Path | None = None
        log.info("Partitioner initialized.")
        log.info(f"Strategy: {self.strategy.name}")
        if self.topology is not None:
            log.info(f"Topology: {self.topology.name}")
        log.info(f"Devices: {self.device_count}")
        log.info(f"Nodes/Layers: {self.node_count}")
        log.info(f"Considered resources: {self.considered_resources}")
        log.info(f"Ideal / Max utilization: {self.ideal_util} / {self.max_utilization}")
        log.info(f"Groups of inseperable nodes: {len(self.inseperable_nodes)}")
        log.info(f"Network ports per device: {self.network_ports_per_device}")

        if (self.strategy == PartitioningStrategy.RESOURCE_UTILIZATION) and (
            None in [self.max_utilization, self.ideal_util, self.resource_estimates]
        ):
            log.error(
                f"One of the required partitioner parameters for the strategy "
                f"{self.strategy.name} was not found. Please provide max_utilization, "
                "ideal_utilization and resource_estimates!"
            )
            raise Exception("Missing partitioner paremeter!")  # TODO: Replace

    @abstractmethod
    def _solve(self, solver_timeout: int) -> dict[int, Device] | None:
        """The real solving function. Should be called indirectly via solve()"""

    @abstractmethod
    def _write_solution_data(self, p: Path, node_index_name_map: dict[int, str] | None) -> None:
        pass

    def solve(
        self,
        solver_timeout: int,
        snapshot_path: Path,
        solution_path: Path,
        node_index_name_map: dict[int, str] | None = None,
    ) -> dict[int, Device] | None:
        """Try to optimize the objective function. If no feasible solution is found
        return None, otherwise return a mapping of nodes to their device. After trying
        to solve, creates a snapshot description of the model in a temp build dir, as well
        as a solution in the same dir, if one was found"""
        result = self._solve(solver_timeout)
        self._create_model_snapshot(snapshot_path)
        if result is not None:
            self._write_solution_data(solution_path, node_index_name_map)
        return result

    def _create_model_snapshot(self, p: Path) -> None:
        """Create a snapshot of all model variables and conditions in a temporary
        build dir. This is useful for debugging and checking on why a model is
        infeasible for example."""
        constraints = ""
        for constr in self.model.constrs:
            constraints += f"\t{constr}\n"
        content = "Model Snapshot\n"
        content += f"Strategy: {self.strategy}\n"
        content += f"Max utilization: {self.max_utilization}\n"
        content += f"Ideal utilization: {self.ideal_util}\n"
        content += f"Considered resource types: {self.considered_resources}\n"
        content += f"Inseperable nodes: {self.inseperable_nodes}\n"
        content += f"Device count: {self.device_count}\n"
        content += f"Node count: {self.node_count}\n"
        content += f"Resources per device: {self.get_resource_use_by_device()}\n"
        if self.resource_estimates is not None:
            content += "Resource estimates per node:\n"
            for node in self.resource_estimates.keys():
                content += f"{node}\n"
                for restype in self.resource_estimates[node].keys():
                    if restype in self.considered_resources:
                        content += f"\t{restype}: {self.resource_estimates[node][restype]}\n"
        content += "Constraints:\n"
        for cons in self.model.constrs:
            content += f"{cons}\n"
        log.debug(f"Writing model snapshot data to {p.absolute()}")
        with p.open("w+") as f:
            f.write(content)

    @abstractmethod
    def _get_resource_use_by_device(self) -> dict[int, dict[str, Any]] | None:
        pass

    def get_resource_use_by_device(self) -> dict[int, dict[str, Any]] | None:
        """Return the resources used by a device. This only works if the optimization goal was
        resource usage. If no optimization was done, the dict will contain None's
        Actual implementation is left to the subclasses."""
        if self.strategy == PartitioningStrategy.RESOURCE_UTILIZATION:
            return self._get_resource_use_by_device()
        return None


class AuroraPartitioner(Partitioner):
    def __init__(
        self,
        network_ports_per_device: int,
        strategy: PartitioningStrategy,
        devices: int,
        nodes: int,
        inseperable_nodes: list[list[int]],
        resources_per_device: dict,
        topology: MFTopology,
        resource_estimates: dict | None = None,
        considered_resources: list[str] | None = None,
        limit_nodes_per_device: int | None = None,
        max_utilization: float | None = None,
        ideal_utilization: float | None = None,
    ) -> None:
        super().__init__(
            strategy,
            topology,
            devices,
            nodes,
            inseperable_nodes,
            resources_per_device,
            resource_estimates,
            considered_resources,
            network_ports_per_device,
            max_utilization,
            ideal_utilization,
        )
        self.limit_nodes_per_device = limit_nodes_per_device
        self.model.verbose = 0
        if (
            ideal_utilization is not None
            and max_utilization is not None
            and ideal_utilization > max_utilization
        ):
            msg = (
                f"Cannot create partition if ideal utilization is set greater than max "
                f"utilization ({ideal_utilization} > {max_utilization})"
            )
            log.critical(msg)
            raise Exception(msg)

        log.debug("Creating partitioning model")

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
            self.model.add_var(name=f"chosen_device_of_node_{node}", var_type=mip.INTEGER)
            for node in range(self.node_count)
        ]
        for node in range(self.node_count):
            self.model += self.chosen_device[node] == xsum(
                self.devices[node][device] * device  # type: ignore
                for device in range(self.device_count)
            )

        # Grouped nodes need to stay together
        # First check that no group is too large
        if len(self.inseperable_nodes) > 0:
            nodes_in_groups = sum(len(group) for group in self.inseperable_nodes)
            max_devices_possible = nodes - nodes_in_groups + len(self.inseperable_nodes)
            for i, group in enumerate(self.inseperable_nodes):
                # 1. Single group larger than the model itself
                if len(group) > nodes:
                    msg = (
                        f"Group {i} of inseperable nodes is larger than the total set of all "
                        f"nodes in the model. (Has {len(group)} nodes, but only {nodes} "
                        "nodes in the graph!)"
                    )
                    log.critical(msg)
                    raise Exception(msg)  # TODO
                # 2. Num. nodes == Num. groups. Leads to atleast 1 empty device
                if len(group) == nodes and devices > 1:
                    msg = (
                        f"Group {i} has the same number of nodes as the graph in total. However "
                        "since more than 1 device is used, this would result in one device "
                        "being completely empty, leading to an invalid partitioning model."
                    )
                    log.critical(msg)
                    raise Exception(msg)  # TODO
            # 3. Not enough devices to have this many nodes in groups
            if devices > max_devices_possible:
                msg = (
                    f"Requested number of FPGAs ({devices}) is larger than the number of "
                    f"devices possible. {nodes - nodes_in_groups} nodes can be alone on a "
                    f"device, and {len(self.inseperable_nodes)} groups of nodes can be on a "
                    f"device. The largest possible device count partitioning would "
                    f"result in {max_devices_possible} devices"
                )
                log.critical(msg)
                raise Exception(msg)

        # Nodes in groups stay together
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
                self.devices[node][device] for node in range(self.node_count)
            )

        # Optionally limit number of nodes per device
        # (This may be necessary in some cases to avoid the maximum number of compute units allowed
        # on the FPGAs)
        if self.limit_nodes_per_device is not None:
            log.info(f"Number of nodes per device limited to: {self.limit_nodes_per_device}")
            for device in range(self.device_count):
                self.model += (
                    self.nodesperdevice[device] <= self.limit_nodes_per_device  # type: ignore
                )

        # Connections that a device has with other devices
        self.connections_per_device_helper = [
            [
                self.model.add_var(
                    name=f"nodes_{node}_{node+1}_on_diff_devices", var_type=mip.INTEGER
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
                # Variable is 1, if the next node is on a different device
                self.model += (
                    self.connections_per_device_helper[device][node]
                    >= self.devices[node][device] - self.devices[node + 1][device]
                )
                self.model += (
                    self.connections_per_device_helper[device][node]
                    >= self.devices[node + 1][device] - self.devices[node][device]
                )

            # Number of nodes that leave a device
            self.model += self.connections_per_device[device] == xsum(
                self.connections_per_device_helper[device][node] for node in range(self.node_count)
            )

        # Limit the number of connections per device (depends on the FPGAs QSFP ports)
        for device in range(self.device_count):
            self.model += (
                self.connections_per_device[device] <= self.network_ports_per_device
            )  # type: ignore

        # Helper for device difference
        self.device_diff = []
        for i in range(self.node_count):
            self.device_diff.append(
                self.model.add_var(
                    name=f"device_difference_node{i}_to_node{i+1}", var_type=mip.INTEGER
                )
            )

        # Consecutive nodes must be on consecutive devices
        for node in range(self.node_count - 1):
            self.model += (
                self.device_diff[node] >= self.chosen_device[node] - self.chosen_device[node + 1]
            )
            self.model += (
                self.device_diff[node] >= self.chosen_device[node + 1] - self.chosen_device[node]
            )
            self.model += self.device_diff[node] <= 1
            self.model += self.device_diff[node] >= 0

        # Setting topology requirements
        match self.topology:
            case MFTopology.CHAIN:
                self.model += self.chosen_device[0] == 0
                self.model += self.chosen_device[-1] == self.device_count - 1

            case MFTopology.RETURNCHAIN:
                raise NotImplementedError()

            case _:
                raise AssertionError(
                    f"Invalid communication scheme for Aurora partitioner: {self.topology}"
                )  # TODO

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
                            self.devices[node][device]
                            * self.resource_estimates[node][resource_name]
                            for node in range(self.node_count)
                            if resource_name in self.resource_estimates[node].keys()
                        ]
                    )

            # Limit resource usage to (available resources * max usage in percent)
            total_per_device = self.resources_per_device
            for device in range(self.device_count):
                for resource_name in self.considered_resources:
                    assert (
                        self.max_utilization is not None
                    )  # Should be caught in constructor before
                    max_resources = int(total_per_device[resource_name] * self.max_utilization)
                    self.model += self.resource_use_int[device][resource_name] <= max_resources

            # Give a warning if resource usage gets close to its maximum
            for node in range(self.node_count):
                for restype in self.resource_estimates[node].keys():
                    assert self.max_utilization is not None
                    thresh_percentage = 0.1 if self.max_utilization >= 0.1 else self.max_utilization
                    if restype not in total_per_device.keys():
                        continue
                    warn_threshold = (self.max_utilization - thresh_percentage) * total_per_device[
                        restype
                    ]
                    max_util = total_per_device[restype] * self.max_utilization
                    if self.resource_estimates[node][restype] >= warn_threshold:
                        if self.resource_estimates[node][restype] < max_util:
                            log.warning(
                                f"Node {node}'s usage of {restype} is within "
                                f"{thresh_percentage:2.2%} of the maximum allowed utilization "
                                f"({self.resource_estimates[node][restype]} / "
                                f"{max_util}) "
                                "on a single device. Partitioning might fail!"
                            )
                        else:
                            log.error(
                                f"Node {node}'s usage of {restype} is above "
                                f"the allowed utilization "
                                f"({self.resource_estimates[node][restype]} > "
                                f"{max_util})."
                            )
                            log.error(
                                "Theoretical max per device would "
                                f"be {total_per_device[restype]} "
                                "on a single device. Partitioning will fail!"
                            )
                            # TODO: When we have a nice exiting mechanism, FINN should stop
                            # TODO: here already

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
                        self.resource_use_relative[device][resource_name] - self.ideal_util
                    )
                    self.model += self.resource_diff[device][resource_name] >= (
                        self.ideal_util - self.resource_use_relative[device][resource_name]
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
                    # Needs to be really small so the model is still valid for very small designs
                    >= 0.0000001
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
                    self.model += self.min_resource_diff[device] >= 0.000000001

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

    def _write_solution_data(self, p: Path, node_index_name_map: dict[int, str] | None) -> None:
        assert self.resource_estimates is not None
        sol = ""
        if node_index_name_map is None:
            node_index_name_map = {i: str(i) for i in range(self.node_count)}
        for node in range(self.node_count):
            sol += f"\n\nNode {node}: {node_index_name_map[node]}\n"
            sol += f"\t\tDevice: {self.chosen_device[node].x}\n"
            for res in self.considered_resources:
                if res not in self.resource_estimates[node].keys():
                    continue
                sol += f"\t\t{res}:" + "{:.2f}%".format(
                    100 * self.resource_estimates[node][res] / self.resources_per_device[res]
                )
        with p.open("w+") as f:
            f.write(sol)

    def _solve(self, solver_timeout: int) -> dict[int, Device] | None:
        status = self.model.optimize(solver_timeout)  # type: ignore
        assert status != mip.OptimizationStatus.ERROR, "There was an error when solving the model!"
        if status in [mip.OptimizationStatus.INFEASIBLE, mip.OptimizationStatus.NO_SOLUTION_FOUND]:
            return None

        # Some log outputs
        for device in range(self.device_count):
            log.debug(f"Device {device} has connections: {self.connections_per_device[device].x}")
        if self.strategy == PartitioningStrategy.RESOURCE_UTILIZATION:
            for device in range(self.device_count):
                resources = sorted(
                    self.resource_use_relative[device].items(),
                    key=lambda tpl: tpl[1].x,
                    reverse=True,
                )
                log.info(
                    f"Device {device} - largest resource type is "
                    f"{resources[0][0]} at {resources[0][1].x:2.2%}"
                )
        else:
            # TODO: Log number of layers per device
            pass

        mapping = {}
        for i in range(self.node_count):
            mapping[i] = self.chosen_device[i].x
        return mapping

    def _get_resource_use_by_device(self) -> dict[int, dict[str, Any]] | None:
        data = {}
        for device in range(self.device_count):
            data[device] = {}
            for restype in self.resource_use_relative[device].keys():
                data[device][restype] = self.resource_use_relative[device][restype].x
        return data


class PartitionForMultiFPGA(Transformation):
    """
    Receive a model with only FPGADataflow nodes and partition it by assigning it's
    device node attribute. Partitioning is done with respect to the chosen strategy.
    To determine how partitioning is done, pass the partitioner type yourself.
    """

    def __init__(self, cfg: DataflowBuildConfig) -> None:
        self.cfg = cfg
        # TODO: Replace assert
        assert self.cfg.partitioning_configuration is not None
        communication_kernel = self.cfg.partitioning_configuration.communication_kernel
        partitioners = {MFCommunicationKernel.AURORA: AuroraPartitioner}
        try:
            self.partitioner_type = partitioners[communication_kernel]
        except KeyError as ke:
            raise Exception(f"No partitioner type found for {communication_kernel}") from ke
        log.info(
            f"Based on the communication kernel, "
            f'partitioner "{self.partitioner_type.__name__}" was chosen!'
        )

    def estimate_required_fpgas(self) -> int:
        """Use resource utilization to estimate how many FPGAs will be needed to
        partition the given model."""
        raise NotImplementedError()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        if self.cfg.partitioning_configuration is None:
            return model, False

        # Dont split during branches. Find all layers that should be on the same device.
        log.debug("Gathering inseperable nodes...")
        if not is_single_in_out_model(model):
            msg = (
                "The model has either more than 1 input or more than 1 output nodes. "
                "This might cause issue during partitioning. Please check your ONNX file."
            )
            log.error(msg)
            raise Exception(msg)  # TODO
        inseperable_nodes = get_inseperable_nodes(model)

        # Number of devices to partition to
        if self.cfg.partitioning_configuration.num_fpgas < 0:
            devices = self.estimate_required_fpgas()
            raise NotImplementedError()
        else:  # noqa
            devices = self.cfg.partitioning_configuration.num_fpgas

        # Needed to resolve platform
        if self.cfg.board is None:
            log.critical('Parameter "board" is required in config for MultiFPGA partitioning')
            raise Exception('Parameter "board" is required in config for MultiFPGA partitioning')

        # Calculate estimates
        if self.cfg.fpga_part is None:
            self.cfg.fpga_part = self.cfg._resolve_fpga_part()  # noqa
        model = model.transform(GiveUniqueNodeNames())
        estimates = get_estimated_model_resources(model, self.cfg.fpga_part)
        if (
            self.cfg.partitioning_configuration.partition_strategy
            == PartitioningStrategy.RESOURCE_UTILIZATION
        ):
            missing_estimates = False
            for layer in estimates.keys():
                if all(estimates[layer][res] <= 0 for res in estimates[layer].keys()):
                    missing_estimates = True
                    log.critical(
                        f"Layer {layer} has an all-0 resource estimation for all "
                        "resource types. Cannot partition using resource estimates!"
                    )
            if missing_estimates:
                raise Exception(
                    "Cannot partition with faulty resource estimations and "
                    "RESOURCE_UTILIZATION as PartitioningStrategy!"
                )

        # Stop if there are more devices than nodes
        if devices > len(model.graph.node):
            msg = f"Cannot partition a model with {len(model.graph.node)} to {devices} devices!"
            log.error(msg)
            log.critical("Infeasible model. Stopping.")
            raise Exception(msg)  # TODO

        # Create the partitioner itself
        device_resources = available_resources(
            platforms[self.cfg.board](), self.cfg.partitioning_configuration.considered_resources
        )
        partitioner = self.partitioner_type(
            devices=devices,
            topology=self.cfg.partitioning_configuration.topology,
            strategy=self.cfg.partitioning_configuration.partition_strategy,
            inseperable_nodes=inseperable_nodes,
            nodes=len(model.graph.node),
            resources_per_device=device_resources,
            considered_resources=self.cfg.partitioning_configuration.considered_resources,
            resource_estimates=estimates,
            max_utilization=self.cfg.partitioning_configuration.max_utilization,
            ideal_utilization=self.cfg.partitioning_configuration.ideal_utilization,
            network_ports_per_device=self.cfg.partitioning_configuration.ports_per_device,
        )

        # Try and solve the model
        logdir = Path(make_build_dir("partition_solver_"))
        index_name_map = dict(enumerate([node.name for node in model.graph.node]))
        mapping = partitioner.solve(
            solver_timeout=self.cfg.partitioning_configuration.partition_solver_timeout,
            snapshot_path=logdir / "snapshot.txt",
            solution_path=logdir / "solution.txt",
            node_index_name_map=index_name_map,
        )
        if mapping is None and partitioner.latest_snapshot_path is not None:
            print(
                f"Model solver failed. Snapshot can be "
                f"found in {partitioner.latest_snapshot_path.absolute()}"
            )

        if mapping is None:
            log.critical("Could not find a feasible solution to the partition model.")
            # TODO: Custom errors
            raise Exception("Could not find a feasible solution to the partition model.")

        # TODO: Warning for very low resource usage

        # TODO: Use index or name?
        for i, node in enumerate(model.graph.node):
            # We have to cast to int here, because the solver returns a float
            # And if qonnx set_nodeattr sees a float in an int field, it sets it
            # to 0, regardless of the floats value
            log.info(f"Mapping node {node.name} to device {int(mapping[i])}")
            set_device_id(node, int(mapping[i]))
        return model, False

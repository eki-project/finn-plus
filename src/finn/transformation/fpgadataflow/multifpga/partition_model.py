"""MultiFPGA partitioning related functionality."""
import mip
import rich.box
import time
import yaml
from collections import Counter
from math import ceil
from pathlib import Path, PosixPath, PurePath, WindowsPath
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from rich.layout import Layout
from rich.table import Table

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    MFCommunicationKernel,
    MFVerbosity,
    PartitioningConfiguration,
)
from finn.transformation.fpgadataflow.multifpga.aurora.partitioner import AuroraPartitioner
from finn.transformation.fpgadataflow.multifpga.partitioner import Partitioner
from finn.util.basic import make_build_dir
from finn.util.exception import (
    FINNInternalError,
    FINNMultiFPGAConfigError,
    FINNMultiFPGAPartitionerError,
    FINNMultiFPGAUserError,
)
from finn.util.fpgadataflow import set_device_id
from finn.util.logging import LogDisabledConsole, log
from finn.util.platforms import platforms
from finn.util.resources import (
    available_resources_on_platform,
    get_estimated_model_resources,
    get_model_device_resource_factors,
)


class ApplyPartitioning(Transformation):
    """Apply partitioning from a YAML file to the graph. Can be used to load an existing
    configuration. Afterwards every node has their device_id node attribute set.

    This expects a YAML file:
    ```
    MVAU_hls_0: 1
    FMPadding_rtl_2: 19
    ...
    ```
    """

    def __init__(self, mapping: dict[str, int] | Path) -> None:
        """Load the mapping either directly or from the specified path."""
        super().__init__()
        self.mapping: dict[str, int] = {}
        if type(mapping) is dict:
            self.mapping = mapping
        elif type(mapping) in [PurePath, Path, PosixPath, WindowsPath]:
            mapping = Path(mapping)  # type: ignore
            if not mapping.exists():
                raise FINNMultiFPGAUserError(
                    f"Cannot read partitioning from {mapping}. No such file exists."
                )
            with mapping.open("r") as f:
                self.mapping = yaml.load(f, yaml.Loader)
        else:
            raise FINNMultiFPGAUserError(
                f"Cannot read partitioning config of type {type(mapping)}."
            )

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Assign the device ids to the nodes."""
        done = 0
        for nodename, device in self.mapping.items():
            node = model.get_node_from_name(nodename)
            if node is None:
                raise FINNMultiFPGAUserError(
                    f"Tried assigning node {nodename} "
                    f"to device {device}, but the model does "
                    f"not contain any nodes of that name. "
                    f"Is your partitioning config outdated?"
                )
            set_device_id(node, device)
            done += 1
        if done != len(model.graph.node):
            raise FINNInternalError(
                f"Something went wrong when partitioning. "
                f"Some nodes did not receive a device ID. Set "
                f"device IDs: {done} / Total expected: {len(model.graph.node)}"
            )
        return model, False


class PartitionForMultiFPGA(Transformation):
    """Receive a model with only FPGADataflow nodes and partition it by assigning it's
    device node attribute. Partitioning is done with respect to the chosen strategy.

    The resulting IDs are assigned to the nodes
    and additionally stored in .../output_dir/partitioning.yaml.
    """

    def __init__(self, cfg: DataflowBuildConfig, automatic_solution_attempts: int = 3) -> None:
        """Partition the model. `automatic_solution_attempts` is ignored,
        unless automatic partitioning is enabled.
        """
        self.auto_solution_attempts = automatic_solution_attempts
        self.cfg = cfg
        if self.cfg.partitioning_configuration is None:
            raise FINNMultiFPGAConfigError(
                "Partitioning config is None, but " "'PartitionForMultiFPGA' was called. "
            )
        self.pcfg: PartitioningConfiguration = cfg.partitioning_configuration  # type: ignore
        self.verbosity = self.pcfg.verbosity

        # Select the partitioner class based on the communication kernel
        partitioners = {MFCommunicationKernel.AURORA: AuroraPartitioner}
        try:
            self.partitioner_type = partitioners[self.pcfg.communication_kernel]
        except KeyError as ke:
            raise FINNMultiFPGAConfigError(
                f"There is currently no partitioner implementation "
                f"for usage with the communication kernel "
                f"{self.pcfg.communication_kernel.name}"
            ) from ke

        # These will be filled out after partitioning
        self.partitioner: Partitioner | None = None
        self.mapping: dict[str, int] | None = None

    def show_mapping(self, mapping: dict[str, int]) -> None:
        """Display mapping either as table or prints, depending on console size."""
        keys = list(mapping.keys())
        with LogDisabledConsole() as cons:
            required_tables = ceil(len(keys) / (cons.height - 5))
            allowed_tables = cons.width / 20
            if required_tables < allowed_tables:
                entries_per_table = (len(keys) // required_tables) + 1
                tables = [Table(box=rich.box.SIMPLE) for _ in range(required_tables)]
                layout = Layout()
                layout.split_row(*tables)
                for table in tables:
                    table.add_column("Index", justify="center", header_style="bold")
                    table.add_column("Node Name", justify="left", header_style="bold")
                    table.add_column("Dev", justify="left", header_style="bold", style="bold green")
                for i, key in enumerate(keys):
                    log.info(str(i) + ": " + str(i // entries_per_table))
                    tables[i // entries_per_table].add_row(str(i), key, str(int(mapping[key])))
                cons.print(layout)
                return
        for key in keys:
            log.info(f"Mapping {key} -> {mapping[key]}")
        return

    def _log_pre_solve_information(self, partitioner: Partitioner) -> None:
        """Log some information before starting to solve the LP."""

    def generate_partitioning_report(
        self, model: ModelWrapper, mapping: dict[str, int] | None, elapsed_seconds: int
    ) -> str:
        """Generate a report in which the required resources, the resources available per device,
        and much more are listed. Works even if no solution was found.
        """
        if self.partitioner is None or self.partitioner.status is None:
            raise FINNInternalError(
                "Cannot log post-solving information before the model was solved."
            )
        assert self.cfg.partitioning_configuration is not None
        assert self.cfg.board is not None

        s = ""

        # General information
        s += f"{' Model and Configuration ':=^80}\n"
        s += "=" * 80 + "\n"
        s += f"{'Layers: ' + str(len(model.graph.node)):<10}\n"
        s += f"{'Devices: ' + str(self.cfg.partitioning_configuration.num_fpgas):<10}\n"
        s += f"{'Time elapsed: ' + str(elapsed_seconds) + 's':<10}\n"

        # Resources
        resource_estimates = get_estimated_model_resources(
            model,
            self.cfg._resolve_fpga_part(),  # noqa
            self.cfg.partitioning_configuration.considered_resources,
            True,
        )
        device_resources = available_resources_on_platform(
            platforms[self.cfg.board](), self.cfg.partitioning_configuration.considered_resources
        )

        s += f"\n{' Required Resources by the Model ':=^80}\n"
        s += "=" * 80 + "\n"
        for restype in self.cfg.partitioning_configuration.considered_resources:
            total_required = sum([rv[restype] for rv in resource_estimates.values()])
            total_on_device = (
                self.cfg.partitioning_configuration.max_utilization * device_resources[restype]
            )
            factor = total_required / total_on_device
            s += (
                f"{restype:<15}{total_required:<15_}   ({factor:.2f}x  "
                f"on  {self.cfg.board}  at  "
                f"{self.cfg.partitioning_configuration.max_utilization:.2%}  max utilization)\n"
            )

        s += f"\n{' Available Resources on Devices at Utilization Percentages':=^80}\n"
        s += "=" * 80 + "\n"
        maxutil = self.cfg.partitioning_configuration.max_utilization
        maxutil_percent = f"{self.cfg.partitioning_configuration.max_utilization:.2%}"
        devices = self.cfg.partitioning_configuration.num_fpgas
        s += (
            " " * 15 + f"{'1x @ ' + maxutil_percent:^15}"
            f"{'1x @ 100%':^15}"
            f"{str(devices) + 'x @ ' + maxutil_percent + '(!)':^15}"
            f"{str(devices) + 'x @ 100%':^15}\n"
        )
        for restype in self.cfg.partitioning_configuration.considered_resources:
            res = int(device_resources[restype])
            s += (
                f"{restype:<15}"
                f"{int(res * maxutil):^15_}"
                f"{int(res):^15_}"
                f"{int(maxutil * res * devices):^15_}"
                f"{int(devices * res):^15_}\n"
            )

        s += f"\n{' Largest Nodes by Resource Type ':=^80}\n"
        s += "=" * 80 + "\n"
        for restype in self.cfg.partitioning_configuration.considered_resources:
            largest = ""
            largest_amount = 0
            for layer in resource_estimates.keys():
                if int(resource_estimates[layer][restype]) > largest_amount:
                    largest_amount = resource_estimates[layer][restype]
                    largest = layer
            s += f"{restype:<15}{largest_amount:<12_}{largest:<25}\n"

        # Return early if no solution was found
        if self.partitioner.model.status in [
            mip.OptimizationStatus.INFEASIBLE,
            mip.OptimizationStatus.NO_SOLUTION_FOUND,
        ]:
            return s

        assert mapping is not None
        s += f"\n{' Nodes per Device ':=^80}\n"
        s += "=" * 80 + "\n"
        counter = Counter(list(mapping.values()))
        for device in counter.keys():
            s += f"{'Device ' + str(device) + ': ':<10}{str(counter[device]) + ' nodes':<10}\n"

        actual_resources = self.partitioner.get_resource_use_relative()
        if actual_resources is not None:
            s += f"\n{' Resources per Device ':=^80}\n"
            for device in actual_resources.keys():
                s += f"{' Device ' + str(device) + ' ':-^40}\n"
                if actual_resources[device] is None:
                    continue
                for restype in actual_resources[device].keys():
                    if actual_resources[device][restype] is None:
                        continue
                    s += f"{restype:<15}{actual_resources[device][restype]:<15.2%}\n"

        return s

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Partition the model."""
        if self.cfg.partitioning_configuration is None or self.cfg.board is None:
            raise FINNMultiFPGAConfigError("No Multi-FPGA partitioning config or board given.")

        # Try solving with these device counts
        device_counts = [self.cfg.partitioning_configuration.num_fpgas]
        if self.cfg.partitioning_configuration.num_fpgas <= 0:
            # Calculate all factors and use the biggest one. Since 2.4x is not a valid device
            # count, ceil to the next value. This will be the lower bound for the number of
            # devices to try out.
            factors = get_model_device_resource_factors(
                model,
                platforms[self.cfg.board](),
                self.cfg._resolve_fpga_part(),  # noqa
                self.cfg.partitioning_configuration.considered_resources,
                self.cfg.partitioning_configuration.max_utilization,
            )
            if factors is None:
                raise FINNMultiFPGAUserError(
                    "Cannot partition the model automatically: One of the "
                    "considered resources is not available in either the "
                    "model or device estimates!"
                )
            expected_devices = ceil(max(factors.values()))
            device_counts = list(
                range(expected_devices, expected_devices + self.auto_solution_attempts)
            )
            if expected_devices == 1:
                log.warning(
                    "The number of required devices (lower bound) "
                    "was set to 1! To explicitly force partitioning across multiple "
                    "devices, set a fixed 'num_fpgas' number in your "
                    "partitioning configuration."
                )

        solution_found = False
        for i, device_count in enumerate(device_counts):
            log.info(
                f"Trying to partition model... ({device_count} devices) "
                f"[Attempt {i+1}/{len(device_counts)}]"
            )

            # Set the device count
            self.cfg.partitioning_configuration.num_fpgas = device_count

            # Create the partitioner
            self.partitioner = self.partitioner_type(self.cfg, model)

            # Solve the model (timed)
            start = time.time()
            self.mapping = self.partitioner.solve(solver_timeout=self.pcfg.partition_solver_timeout)
            elapsed_seconds = time.time() - start

            # Minimal logging
            if self.partitioner.status is None:
                raise FINNInternalError(
                    "No optimization status found despite previous solving attempt?"
                )
            if self.verbosity.value > MFVerbosity.LOW.value:
                log.info(
                    f"Solver done. Took {elapsed_seconds:.3f} seconds. "
                    f"Optimization Status: {self.partitioner.status.name}"
                )
            if self.mapping is not None:
                solution_found = True
                break

        assert self.partitioner is not None  # for the type checker

        # Store the model definition for debugging - only store the last try
        logdir = Path(make_build_dir("partitioning_model_data_"))
        model_definition_file = logdir / "model.lp"
        self.partitioner.model.write(str(model_definition_file))

        # Generate report, regardless of whether partitioning was successful
        report = self.generate_partitioning_report(model, self.mapping, int(elapsed_seconds))

        # Display / save report
        if self.verbosity.value == self.verbosity.EXTRA_HIGH.value:
            log.info("\n" + report, extra={"highlighter": None, "markup": False})
        report_path = Path(self.cfg.output_dir) / "report" / "partitioning_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)

        # If partitioning failed, return now
        if self.mapping is None or not solution_found:
            raise FINNMultiFPGAPartitionerError(
                f"Partitioning failed. Status: "
                f"{self.partitioner.model.status.name}.\n"
                f"A detailed report can be viewed at: {report_path.absolute()}\n"
                f"The model definition can be found at: {model_definition_file.absolute()}"
            )

        # Apply results back to the model
        model = model.transform(ApplyPartitioning(self.mapping))
        log.info("Partitioning successful.")

        # Write results to build dir and log dir
        self.partitioner.write_results(logdir / "partitioning.yaml")
        self.partitioner.write_results(Path(self.cfg.output_dir) / "report" / "partitioning.yaml")
        return model, False

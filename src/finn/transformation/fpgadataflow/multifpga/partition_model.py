"""MultiFPGA partitioning related functionality."""
import yaml
from math import ceil
from pathlib import Path, PurePath
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from typing import TYPE_CHECKING, cast

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    MFVerbosity,
    MultiFPGAConfiguration,
)
from finn.transformation.fpgadataflow.multifpga.backend import get_backend
from finn.transformation.fpgadataflow.multifpga.utils import (
    ensure_board_set,
    ensure_multifpga_config,
)
from finn.util.exception import (
    FINNMultiFPGAConfigError,
    FINNMultiFPGAUnsolvableError,
    FINNMultiFPGAUserError,
)
from finn.util.fpgadataflow import set_device_id
from finn.util.logging import log
from finn.util.resources import get_model_device_resource_factors

if TYPE_CHECKING:
    from finn.transformation.fpgadataflow.multifpga.partitioner import Partitioner


class ApplyDeviceMapping(Transformation):
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
        # isinstance instead of comparing the exact type, so that subclasses (e.g. an OrderedDict
        # or the platform specific PosixPath / WindowsPath) are accepted as well
        if isinstance(mapping, dict):
            self.mapping = mapping
        elif isinstance(mapping, PurePath):
            mapping = Path(mapping)
            if not mapping.exists():
                raise FINNMultiFPGAUserError(
                    f"Cannot read partitioning from {mapping}. No such file exists."
                )
            with mapping.open("r") as f:
                loaded = yaml.safe_load(f)
            # Anything but a mapping of node names to devices (e.g. an empty file or a list)
            # would otherwise fail later with an unrelated error
            if not isinstance(loaded, dict):
                raise FINNMultiFPGAUserError(
                    f"Cannot read partitioning from {mapping}. Expected a mapping of node names "
                    f"to device IDs, but got {type(loaded).__name__}."
                )
            self.mapping = loaded
        else:
            raise FINNMultiFPGAUserError(
                f"Cannot read partitioning config of type {type(mapping)}."
            )

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Assign the device ids to the nodes."""
        # Validate the complete mapping before assigning anything, so that an invalid mapping
        # does not leave the model partially partitioned
        model_nodes = {node.name: node for node in model.graph.node}
        unknown = [name for name in self.mapping if name not in model_nodes]
        if len(unknown) > 0:
            raise FINNMultiFPGAUserError(
                f"Tried assigning the node(s) {', '.join(unknown)} to a device, but the model does "
                f"not contain any nodes of that name. Is your partitioning config outdated?"
            )
        # An incomplete mapping is a mistake in the user provided partitioning, not an
        # internal error
        missing = [name for name in model_nodes if name not in self.mapping]
        if len(missing) > 0:
            raise FINNMultiFPGAUserError(
                f"The partitioning does not assign a device to the node(s) "
                f"{', '.join(sorted(missing))}. Every node needs a device ID."
            )

        # Apply the mapping by assigning the devices
        for nodename, device in self.mapping.items():
            set_device_id(model_nodes[nodename], device)
        return model, False


class FindMultiFPGADeviceMapping(Transformation):
    """Receive a model with only FPGADataflow nodes and partition it by assigning its
    device node attribute. Partitioning is done with respect to the chosen strategy.

    The resulting IDs are assigned to the nodes and additionally stored in device_mapping.yaml
    in the report directory, next to the partitioning report.
    """

    @ensure_multifpga_config
    @ensure_board_set
    def __init__(self, cfg: DataflowBuildConfig, automatic_solution_attempts: int = 3) -> None:
        """Partition the model. `automatic_solution_attempts` is ignored,
        unless automatic partitioning is enabled.
        """
        self.auto_solution_attempts = automatic_solution_attempts
        self.cfg = cfg
        self.mfcfg: MultiFPGAConfiguration = cast(
            "MultiFPGAConfiguration", cfg.multifpga_configuration
        )
        self.verbosity = self.mfcfg.verbosity

        # Select the partitioner based on the communication kernel
        self.partitioner_type = get_backend(self.mfcfg.network_backend).partitioner

        # These will be filled out after partitioning
        self.partitioner: Partitioner | None = None
        self.mapping: dict[str, int] | None = None

    def generate_device_counts(self, model: ModelWrapper) -> list[int]:
        """Generate a list of device counts to try. If the number is set
        in the config, only this number is returned. Otherwise the estimates are used
        to judge the number of devices needed."""
        device_counts = [self.mfcfg.num_fpgas]
        if self.mfcfg.num_fpgas <= 0:
            # Calculate all factors and use the biggest one. Since 2.4x is not a valid device
            # count, ceil to the next value. This will be the lower bound for the number of
            # devices to try out.
            factors = get_model_device_resource_factors(
                model,
                self.cfg.board,  # type: ignore
                self.cfg._resolve_fpga_part(),  # noqa
                self.mfcfg.considered_resources,
                self.mfcfg.max_utilization,
            )
            if factors is None:
                raise FINNMultiFPGAUserError(
                    "Cannot partition the model automatically: One of the "
                    "considered resources is not available in the model estimates!"
                )
            # A model without any usage of the considered resources would give 0 devices
            expected_devices = max(1, ceil(max(factors.values())))
            device_counts = list(
                range(expected_devices, expected_devices + self.auto_solution_attempts)
            )
            if expected_devices == 1:
                log.warning(
                    "The number of required devices (lower bound) "
                    "was set to 1! To explicitly force partitioning across multiple "
                    "devices, set a fixed 'num_fpgas' number in your "
                    "Multi-FPGA configuration."
                )
        return device_counts

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Partition the model."""
        if self.cfg.multifpga_configuration is None or self.cfg.board is None:
            raise FINNMultiFPGAConfigError("No Multi-FPGA partitioning config or board given.")

        # Try solving with all different device counts
        device_counts = self.generate_device_counts(model)
        last_error: FINNMultiFPGAUnsolvableError | None = None
        for i, device_count in enumerate(device_counts):
            log.info(
                f"Trying to partition model... ({device_count} devices) "
                f"[Attempt {i+1}/{len(device_counts)}]"
            )
            self.cfg.multifpga_configuration.num_fpgas = device_count
            # Reset first, so that the report below is not made from a previous attempt if
            # creating the partitioner fails
            self.partitioner = None
            self.mapping = None
            try:
                self.partitioner = self.partitioner_type(self.cfg, model)
                self.mapping = self.partitioner.solve()
                break
            except FINNMultiFPGAUnsolvableError as e:
                last_error = e
                log.debug(f"Partitioning with {device_count} devices is unsolvable: {e}")

        # Creating the partitioner fails already if its checks show that the model is unsolvable.
        # Then there is no partitioner to write a report.
        if self.partitioner is None:
            raise FINNMultiFPGAUnsolvableError(
                f"Could not solve the device mapping: {last_error}"
            ) from last_error

        # Write the report regardless of fail. If high verbosity also directly print it
        report = self.partitioner.write_report()
        if self.verbosity.value == MFVerbosity.EXTRA_HIGH.value:
            log.info(
                "\n" + self.partitioner.generate_partitioning_report(),
                extra={"highlighter": None, "markup": False},
            )

        # Fail if finally no mapping was found
        if self.mapping is None:
            raise FINNMultiFPGAUnsolvableError(
                f"Could not solve the device mapping: {last_error} Take a look at: {report}"
            ) from last_error
        self.partitioner.write_mapping()

        # Log time required for solving the model
        if self.verbosity.value > MFVerbosity.LOW.value:
            log.info(f"Solver done. Took {self.partitioner.solve_seconds:.3f} seconds.")

        # Apply results back to the model
        model = model.transform(ApplyDeviceMapping(self.mapping))
        log.info("Partitioning successful.")
        return model, False

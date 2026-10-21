"""Prepare and package AuroraFlow kernels."""

import os
import shlex
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from finn.builder.build_dataflow_config import MFVerbosity, MultiFPGAConfiguration
from finn.transformation.fpgadataflow.multifpga.aurora.metadata import (
    AuroraLinkEndpoint,
    AuroraNetworkMetadata,
)
from finn.util.basic import get_vivado_version, make_build_dir
from finn.util.exception import (
    FINNInternalError,
    FINNMultiFPGAConfigError,
    FINNMultiFPGAError,
    FINNMultiFPGAUserError,
    FINNUserError,
)
from finn.util.logging import log
from finn.util.settings import get_settings


class PrepareAuroraFlow(Transformation):
    """Prepares all AuroraFlow kernel XO files and stores them in the network
    metadata of the ModelWrapper.

    If the Aurora Version is passed, the given one is used, otherwise we try to figure out
    the version that best matches the used Vivado version automatically.

    Requires: A metadata creation transformation needs to be run beforehand, so that this transform
    knows, which kernels it needs to package.

    Afterwards: The models metadata contains only valid paths to AuroraFlow kernel XOs.
    """

    def __init__(
        self,
        platform: str,
        part: str,
        multifpga_configuration: MultiFPGAConfiguration,
        aurora_version: str | None = None,
    ) -> None:
        """Prepare AuroraFlow."""
        super().__init__()
        # TODO: non-vitis platforms?
        self.platform = platform
        self.part = part
        self.verbosity = multifpga_configuration.verbosity
        self.make_args = " ".join(
            f"{k}={v}" for k, v in multifpga_configuration.communication_kernel_arguments.items()
        )
        self.ports = multifpga_configuration.links_per_device
        self.aurora_storage = Path(make_build_dir("aurora_storage_")).absolute()
        self.aurora_path = get_settings().finn_deps / "AuroraFlow"
        if not self.aurora_path.exists():
            raise FINNMultiFPGAConfigError(
                "Could not find AuroraFlow in FINN+'s dependency "
                "directory. Are all dependencies downloaded "
                f"and installed? (Searched at {self.aurora_path})."
            )

        if aurora_version is None:
            # The Aurora IP Core is version locked, so we figure out which Vivado version we use
            vivado_path = shutil.which("vivado")
            if vivado_path is None:
                raise FINNUserError("Vivado is not available in PATH!")

            # The PATH entry may be a wrapper or symlink, so the version is taken from
            # XILINX_VIVADO, which works for both the <tool>/<version> and the
            # <version>/<tool> install layout
            vivado_root = os.environ.get("XILINX_VIVADO")
            if vivado_root is None:
                raise FINNUserError("XILINX_VIVADO was not set but is required.")
            vivado_version = get_vivado_version()
            if vivado_version is None:
                raise FINNUserError(
                    f"Could not find a version number in XILINX_VIVADO: {vivado_root}"
                )
            year, _release = vivado_version
            self.aurora_version = ""
            if year < 2024:
                self.aurora_version = "12.0"
            else:
                # TODO: This might need to be updated for future versions.
                self.aurora_version = "13.0"
        else:
            self.aurora_version = aurora_version
        log.debug(f"Using Aurora IP Version: {self.aurora_version}")

        # Check in case a more complicated matching ever overlooks a version
        if self.aurora_version == "":
            raise FINNInternalError("Aurora IP Core Version undetermined!")

    def package_single(self, args: str, device: int, index: int) -> Path:
        """Package a single AuroraFlow kernel. Stored in a subdirectory of the
        aurora_storage directory,
        which is saved in the metadata prop of the model.
        The directory is named 'aurora_device_<device>_index_<index>'.
        Inside are the packaged xo files.

        >>> from finn.builder.build_dataflow_config import MFNetworkBackendType
        >>> p = MultiFPGAConfiguration(
        ...     num_fpgas=2, links_per_device=2, network_backend=MFNetworkBackendType.AURORAFLOW
        ... )
        >>> t = PrepareAuroraFlow("xilinx_u55c_gen3x16_xdma_3_202210_1", "xcu55c-fsvh2892-2L-e", p)
        >>> xo = t.package_single("", device=0, index=0)
        >>> xo.exists()
        True
        >>> xo.name
        'aurora_flow_hw_0.xo'
        >>> xo.parent.parent.name
        'auroraflow_build_dev0_ind0'
        """
        if index >= self.ports:
            raise FINNMultiFPGAConfigError(
                f"Cannot create aurora kernel number {index+1} "
                f"(index {index}), since the configuration/device "
                f"has only {self.ports} networking ports."
            )
        if self.verbosity.value >= MFVerbosity.HIGH.value:
            log.info(f"Packaging AuroraFlow core (Device: {device}, Index: {index})")

        # Copy the AuroraFlow project into a build directory
        build_dir = self.aurora_storage / f"auroraflow_build_dev{device}_ind{index}"
        shutil.copytree(self.aurora_path, build_dir, dirs_exist_ok=True)

        # Create the aurora kernel xo file
        if self.make_args != "" and self.verbosity.value > MFVerbosity.LOW.value:
            log.info(
                f"Packaging AuroraFlow kernel with additional make arguments: {self.make_args}"
            )
        result = subprocess.run(
            shlex.split(
                f"make aurora_hw AURORA_VERSION={self.aurora_version} "
                f"PART={self.part} PLATFORM={self.platform} "
                f"{args} {self.make_args}"
            ),
            cwd=build_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise FINNMultiFPGAUserError(
                f"Error during creation of the AuroraFlow "
                f"kernels. Check build directory at: "
                f"{build_dir.absolute()}\nSTDERR:\n{result.stderr}"
            )
        xo_path = build_dir / "build" / f"aurora_flow_hw_{index}.xo"
        if not xo_path.exists():
            raise FINNMultiFPGAError(
                f"Packaging AuroraFlow failed. Expected "
                f"kernel at path {xo_path}. Check logs in {build_dir.absolute()}"
            )
        return xo_path.absolute()

    def package_all_from_metadata(self, metadata: AuroraNetworkMetadata) -> None:
        """Package all AuroraFlow kernels required by the metadata. If a path is missing or
        doesn't point to a valid object, it is generated and filled out.
        """
        if self.verbosity == MFVerbosity.HIGH:
            log.info(f"Packaging kernels with PART={self.part} and PLATFORM={self.platform}")

        # Package all Aurora kernels concurrently. The kernel is specific to the QSFP port, so it
        # is packaged for the link index of the endpoint, not for its position in the metadata.
        futures: list[Future] = []
        endpoints: list[AuroraLinkEndpoint] = []
        with ThreadPoolExecutor(max_workers=get_settings().num_default_workers) as tpe:
            for device, endpoint in metadata.get_unprepared_aurora_kernels():
                index = endpoint.link_index
                if index is None:
                    raise FINNInternalError(
                        f"Cannot package AuroraFlow kernel on device {device} connected to device "
                        f"{endpoint.partner_device}: the link index of the endpoint is missing."
                    )
                if self.verbosity.value > MFVerbosity.LOW.value:
                    log.info(f"Packaging AuroraFlow core number {index} for device {device}.")
                futures.append(
                    tpe.submit(
                        self.package_single,
                        "",
                        device,
                        index,
                    )
                )
                endpoints.append(endpoint)
            tpe.shutdown()

        # Store results in the metadata
        for endpoint, future in zip(endpoints, futures, strict=True):
            endpoint.aurora_xo = future.result()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Package all aurora kernels required by the model in parallel.
        Sets the "aurora_storage" metadata prop to the path where the xo files
        are stored.
        """
        # Load the metadata from the model's metadata prop
        metadata = AuroraNetworkMetadata.load_from_model(model)

        # Store the location of the aurora kernels in the model as well - just in case
        model.set_metadata_prop("aurora_storage", str(self.aurora_storage.absolute()))

        # Package the kernels and modify the metadata to contain the paths
        # of the packaged kernels
        self.package_all_from_metadata(metadata)

        # Store the updated metadata
        metadata.save()
        return model, False

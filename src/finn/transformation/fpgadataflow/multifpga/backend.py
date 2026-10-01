"""Registry of the communication kernels supported by the Multi-FPGA flow. Each communication
kernel registers one backend that bundles everything the flow needs for this kernel: The
partitioner, the metadata class and the transformations to prepare the kernels and to add them to
the linking configurations. Steps of the flow look up the backend with `get_backend()` instead of
dispatching on the communication kernel themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    MFCommunicationKernel,
    MultiFPGAConfiguration,
)
from finn.transformation.fpgadataflow.multifpga.aurora.link_config_transform import (
    AddAuroraToLinkConfig,
)
from finn.transformation.fpgadataflow.multifpga.aurora.metadata import AuroraNetworkMetadata
from finn.transformation.fpgadataflow.multifpga.aurora.partitioner import AuroraPartitioner
from finn.transformation.fpgadataflow.multifpga.aurora.prepare_aurora import PrepareAuroraFlow
from finn.util.exception import FINNInternalError, FINNMultiFPGAConfigError

if TYPE_CHECKING:
    from collections.abc import Callable
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.transformation.base import Transformation

    from finn.transformation.fpgadataflow.multifpga.metadata import NetworkMetadata
    from finn.transformation.fpgadataflow.multifpga.partitioner import Partitioner


@dataclass(frozen=True)
class CommunicationBackend:
    """Everything the Multi-FPGA flow needs to know about one communication kernel.
    Create and register backends with `register_backend()`.
    """

    kernel: MFCommunicationKernel
    """The communication kernel this backend implements."""

    partitioner: Callable[[DataflowBuildConfig, ModelWrapper], Partitioner]
    """Create the partitioner for the given model. Used by `PartitionForMultiFPGA`."""

    metadata_type: type[NetworkMetadata]
    """Metadata class. Used by `CreateNetworkMetadata` and to load the metadata later on."""

    prepare_kernels: Callable[[DataflowBuildConfig], Transformation]
    """Create the transformation that prepares (e.g. packages) the communication kernels."""

    modify_link_config: Callable[[DataflowBuildConfig], Transformation]
    """Create the transformation that adds the communication kernels to the linking configs."""


_BACKENDS: dict[MFCommunicationKernel, CommunicationBackend] = {}


def register_backend(
    kernel: MFCommunicationKernel,
    *,
    partitioner: Callable[[DataflowBuildConfig, ModelWrapper], Partitioner],
    metadata_type: type[NetworkMetadata],
    prepare_kernels: Callable[[DataflowBuildConfig], Transformation],
    modify_link_config: Callable[[DataflowBuildConfig], Transformation],
) -> CommunicationBackend:
    """Create and register the backend for the given communication kernel and return it.
    Raises an error if the kernel already has a backend.
    """
    if kernel in _BACKENDS:
        raise FINNInternalError(
            f"A backend for communication kernel {kernel.name} is already registered."
        )
    backend = CommunicationBackend(
        kernel=kernel,
        partitioner=partitioner,
        metadata_type=metadata_type,
        prepare_kernels=prepare_kernels,
        modify_link_config=modify_link_config,
    )
    _BACKENDS[kernel] = backend
    return backend


def get_backend(kernel: MFCommunicationKernel) -> CommunicationBackend:
    """Return the backend of the given communication kernel."""
    try:
        return _BACKENDS[kernel]
    except KeyError as e:
        raise FINNMultiFPGAConfigError(
            f"Communication kernel {kernel.name} has no Multi-FPGA backend. Available "
            f"communication kernels: {', '.join(k.name for k in _BACKENDS) or 'none'}"
        ) from e


def _get_multifpga_configuration(cfg: DataflowBuildConfig) -> MultiFPGAConfiguration:
    """Return the Multi-FPGA configuration, which must be set when using a backend."""
    if cfg.multifpga_configuration is None:
        raise FINNInternalError(
            "A Multi-FPGA backend was used, but no Multi-FPGA configuration is set."
        )
    return cfg.multifpga_configuration


def _get_board(cfg: DataflowBuildConfig) -> str:
    """Return the board, which must be set for Multi-FPGA builds."""
    if cfg.board is None:
        raise FINNMultiFPGAConfigError("Cannot do Multi-FPGA without 'board' being specified.")
    return cfg.board


# AuroraFlow
register_backend(
    MFCommunicationKernel.AURORA,
    partitioner=AuroraPartitioner,
    metadata_type=AuroraNetworkMetadata,
    prepare_kernels=lambda cfg: PrepareAuroraFlow(
        cfg._resolve_vitis_platform(),  # noqa: SLF001
        cfg._resolve_fpga_part(),  # noqa: SLF001
        _get_multifpga_configuration(cfg),
    ),
    modify_link_config=lambda cfg: AddAuroraToLinkConfig(
        _get_board(cfg),
        cfg._resolve_fpga_part(),  # noqa: SLF001
    ),
)

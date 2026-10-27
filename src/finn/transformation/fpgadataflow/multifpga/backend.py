"""Registry of the network backends (communication kernels) supported by the Multi-FPGA flow.
Each communication kernel registers one backend that bundles everything the flow needs for this
kernel: The partitioner, the metadata class and the transformations to prepare the kernels and to
add them to the linking configurations. Steps of the flow look up the backend with `get_backend()`
instead of dispatching on the communication kernel themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    MFNetworkBackendType,
    MultiFPGAConfiguration,
)
from finn.transformation.fpgadataflow.multifpga.aurora.link_config_transform import (
    AddAuroraToLinkConfig,
)
from finn.transformation.fpgadataflow.multifpga.aurora.metadata import AuroraNetworkMetadata
from finn.transformation.fpgadataflow.multifpga.aurora.partitioner import (
    PointToPointBidirectionalFixedPartitioner,
)
from finn.transformation.fpgadataflow.multifpga.aurora.prepare_aurora import PrepareAuroraFlow
from finn.util.contracts import ensure_board_set, ensure_multifpga_config
from finn.util.exception import FINNInternalError, FINNMultiFPGAConfigError

if TYPE_CHECKING:
    from collections.abc import Callable
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.transformation.base import Transformation

    from finn.transformation.fpgadataflow.multifpga.metadata import NetworkMetadata
    from finn.transformation.fpgadataflow.multifpga.partitioner import Partitioner


@dataclass(frozen=True)
class NetworkBackend:
    """Everything the Multi-FPGA flow needs to know about one communication kernel.
    Create and register backends with `register_backend()`.
    """

    type: MFNetworkBackendType
    """The type of network backend (communication kernel) this entry implements."""

    partitioner: Callable[[DataflowBuildConfig, ModelWrapper], Partitioner]
    """Create the partitioner for the given model. Used by `FindMultiFPGADeviceMapping`."""

    metadata_type: type[NetworkMetadata]
    """Metadata class. Used by `CreateNetworkMetadata` and to load the metadata later on."""

    prepare_kernels: Callable[[DataflowBuildConfig], Transformation]
    """Create the transformation that prepares (e.g. packages) the communication kernels."""

    modify_link_config: Callable[[DataflowBuildConfig], Transformation]
    """Create the transformation that adds the communication kernels to the linking configs."""


_BACKENDS: dict[MFNetworkBackendType, NetworkBackend] = {}


def register_backend(
    backend_type: MFNetworkBackendType,
    *,
    partitioner: Callable[[DataflowBuildConfig, ModelWrapper], Partitioner],
    metadata_type: type[NetworkMetadata],
    prepare_kernels: Callable[[DataflowBuildConfig], Transformation],
    modify_link_config: Callable[[DataflowBuildConfig], Transformation],
) -> NetworkBackend:
    """Create and register the entry for the given network backend type and return it.
    Raises an error if the network backend type is already registered.
    """
    if backend_type in _BACKENDS:
        raise FINNInternalError(f"Network backend {backend_type.name} is already registered.")
    entry = NetworkBackend(
        type=backend_type,
        partitioner=partitioner,
        metadata_type=metadata_type,
        prepare_kernels=prepare_kernels,
        modify_link_config=modify_link_config,
    )
    _BACKENDS[backend_type] = entry
    return entry


def get_backend(backend_type: MFNetworkBackendType) -> NetworkBackend:
    """Return the registered entry of the given network backend type."""
    try:
        return _BACKENDS[backend_type]
    except KeyError as e:
        raise FINNMultiFPGAConfigError(
            f"Network backend {backend_type.name} is not registered for Multi-FPGA. Registered "
            f"network backends: {', '.join(b.name for b in _BACKENDS) or 'none'}"
        ) from e


@ensure_multifpga_config
def _get_multifpga_configuration(cfg: DataflowBuildConfig) -> MultiFPGAConfiguration:
    """Return the Multi-FPGA configuration, which must be set when using a backend."""
    return cast("MultiFPGAConfiguration", cfg.multifpga_configuration)


@ensure_board_set
def _get_board(cfg: DataflowBuildConfig) -> str:
    """Return the board, which must be set for Multi-FPGA builds."""
    return cast("str", cfg.board)


# AuroraFlow
register_backend(
    MFNetworkBackendType.AURORAFLOW,
    partitioner=PointToPointBidirectionalFixedPartitioner,
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

"""Contains the base NetworkMetadata class, as well as creators
for metadatas for various kinds of topologies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

# Needs to be outside the type-checking block, so that subclasses using mashumaro can serialize it
from finn.builder.build_dataflow_config import MFTopology
from finn.util.basic import get_metadata_prop_path
from finn.util.logging import log

if TYPE_CHECKING:
    from pathlib import Path
    from qonnx.core.modelwrapper import ModelWrapper
    from typing import Self

    from finn.builder.build_dataflow_config import MultiFPGAConfiguration


class DataDirection(str, Enum):
    """Data movement direction."""

    TX = "TX"
    RX = "RX"
    BIDIRECTIONAL = "BIDIRECTIONAL"


@dataclass
class NetworkMetadata(ABC):
    """Metadata baseclass for storage of Multi-FPGA connections. Defines connections between
    devices, as well as which nodes on the devices are responsible for communication.
    """

    topology: MFTopology = MFTopology.SIMPLEX_CHAIN
    """Topology the devices were partitioned for. Used e.g. to choose the layout in `plot()`."""

    @classmethod
    def create_from_multifpga_configuration(cls, mfcfg: MultiFPGAConfiguration) -> NetworkMetadata:
        """Create an empty metadata object. Subclasses can override this to read the settings
        they need from the Multi-FPGA configuration.
        """
        return cls(topology=mfcfg.topology)

    @classmethod
    def load_from_model(cls, model: ModelWrapper) -> Self:
        """Load the metadata from a modelwrapper. The `network_metadata` metadata prop of the
        model points to the location the metadata was saved to (see `CreateNetworkMetadata`).
        How the metadata is stored there is up to the subclass, which reads it in `load()`.
        Must be called on the metadata class of the communication kernel that created the
        metadata, e.g. `get_backend(kernel).metadata_type.load_from_model(model)`.
        """
        p = get_metadata_prop_path(
            model,
            "network_metadata",
            must_exist=True,
            custom_error=(
                "Cannot load metadata from model, since no 'network_metadata' prop was set. "
                "Did you forget to run the 'CreateNetworkMetadata' transformation before this?"
            ),
        )
        return cls.load(p)

    @abstractmethod
    def __getitem__(self, key: Any) -> Any:
        """Getter base method for internal data."""
        pass  # noqa

    @abstractmethod
    def __setitem__(self, key: Any, value: Any) -> Any:
        """Setter base method for internal data."""
        pass  # noqa

    @abstractmethod
    def add_connection(
        self,
        sender_device: int,
        sender_node: str,
        sender_tensor_index: int,
        receiver_device: int,
        receiver_node: str,
        receiver_tensor_index: int,
    ) -> None:
        """Add a connection to the metadata."""
        pass  # noqa

    @abstractmethod
    def save(self, p: Path | None) -> None:
        """Save the internal data to the given path. If no path is given,
        the metadata is stored at the location it was loaded from before.
        """

    @classmethod
    @abstractmethod
    def load(cls, p: Path) -> Self:
        """Load the metadata from the given path, as written by `save()`."""

    @abstractmethod
    def node_is_sender(self, node: str) -> bool:
        """Return whether the given node acts as a sender (TX) on its device."""

    @abstractmethod
    def node_is_receiver(self, node: str) -> bool:
        """Return whether the given node acts as a receiver (RX) on its device."""

    @abstractmethod
    def get_partner_node(
        self, node: str, stream_index: int, direction: DataDirection
    ) -> tuple[int, str, int] | None:
        """Search through all nodes and return the one that acts as a communication partner
        of the given stream of `node` in the given direction, as (partner device, partner node,
        partner stream index). For TX, `stream_index` is the index of the `m_axis` stream of
        `node`, for RX the index of its `s_axis` stream. If A:m_axis_0 (device 0) -> B:s_axis_1
        (device 1), then get_partner_node(A, 0, TX) returns (1, B, 1) and
        get_partner_node(B, 1, RX) returns (0, A, 0). If nothing is connected to the stream in the
        given direction, None is returned.
        """

    def plot_dot(
        self,
        path: Path,  # noqa: ARG002
        model: ModelWrapper | None = None,  # noqa: ARG002
    ) -> None:
        """Write the metadata as a Graphviz DOT graph to the given path. If Graphviz' `dot` is
        available, also render it next to the DOT file. If the model (with SDPs and their device
        IDs) is given, the SDPs on each device and the data flow between them are included.
        Overwrite this method if possible."""
        log.warning("Cannot create a DOT graph of the metadata for the chosen backend!")

    @abstractmethod
    def export(self, path: Path) -> None:
        """Export the metadata to the given path. In many cases this can be
        a NetworkX graph, serialized as a JSON via node_link_data, but it might also
        be in any other format. The exported file should be used as an information
        source for further non-FINN MultiFPGA related tasks, such as cabling, switch setup,
        etc."""

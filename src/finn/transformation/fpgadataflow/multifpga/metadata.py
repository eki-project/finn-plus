"""Contains the base NetworkMetadata class, as well as creators
for metadatas for various kinds of topologies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from finn.util.basic import get_metadata_prop_path

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

    @classmethod
    def create_from_multifpga_configuration(
        cls, mfcfg: MultiFPGAConfiguration  # noqa: ARG003
    ) -> NetworkMetadata:
        """Create an empty metadata object. Subclasses can override this to read the settings
        they need from the Multi-FPGA configuration.
        """
        return cls()

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
        receiver_device: int,
        receiver_node: str,
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
    def get_partner_node(self, node: str, direction: DataDirection) -> str | None:
        """Search through all nodes and return the one that acts as a communication partner
        in the given direction. If A <-> B, then get_partner_node(A, TX) returns B and
        get_partner_node(B, RX) returns A. (get_partner_node(A, RX) would return None,
        since A is not a receiving from any node.)
        """  # noqa

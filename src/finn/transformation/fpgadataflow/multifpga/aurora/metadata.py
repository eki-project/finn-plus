"""Manage metadata for connecting multiple FPGAs using different technologies and topologies."""

from __future__ import annotations

from dataclasses import dataclass, field
from mashumaro.mixins.yaml import DataClassYAMLMixin

# Needs to be outside the type-checking block, so that DataClassYAMLMixin can pick it up
# for automatic serialization.
from pathlib import Path
from typing import TYPE_CHECKING, cast

from finn.transformation.fpgadataflow.multifpga.metadata import DataDirection, NetworkMetadata
from finn.util.basic import get_metadata_prop_path
from finn.util.exception import FINNInternalError, FINNMultiFPGAConfigError

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


@dataclass
class AuroraNetworkKernelMetadata:
    """Keeps track of a single kernel on a device. It stores the path to
    the packaged kernel, with which device this kernel communicates, and which kernels on both
    devices are responsible for communication in both directions.
    """

    aurora_xo: Path | None
    """Path to the kernel XO file. If it has not been set yet, this is left as None."""

    partner_device: int
    """ID of the partner device."""

    connecting_kernels: dict[DataDirection, tuple[str, str] | None]
    """Connections with other devices. One entry for RX, one for TX.
    The value can be None, in which case no connection for this direction
    was established yet, or it can contain a tuple of the names of the kernels
    to which the Aurora kernels are connected.

    TX: (SDP0, SDP1) would mean that SDP0 (this device) sends to SDP1 (other device)
    RX: (SDP1, SDP0) would mean that SDP0 (other device) sends to SDP1 (this device)

    The first element is always the kernel matching the communication direction: For TX it's
    the sender, for RX it's the receiver.

    A complete connection requires matching pairs of TX and RX data on the two devices.
    """


@dataclass
class AuroraNetworkMetadata(NetworkMetadata, DataClassYAMLMixin):
    """Defines an AuroraFlow based network. On each device is a list of
    AuroraNetworkKernelMetadata objects which describe the configuration of
    a single AuroraFlow kernel.
    """

    data: dict[int, list[AuroraNetworkKernelMetadata]] = field(default_factory=dict)
    ports_per_device: int = 2
    loaded_from_path: Path | None = None

    @staticmethod
    def from_model(model: ModelWrapper) -> AuroraNetworkMetadata:
        """Load metadata from the model."""
        p = get_metadata_prop_path(
            model,
            "network_metadata",
            must_exist=True,
            custom_error=(
                "Cannot load metadata from model, since no 'network_metadata' prop was set. "
                "Did you forget to run the 'AssignMetadata' transformation before this?"
            ),
        )
        meta = AuroraNetworkMetadata.load(p)
        meta.loaded_from_path = p
        return meta

    @staticmethod
    def load(p: Path) -> AuroraNetworkMetadata:
        """Load from a YAML file."""
        if not p.exists():
            raise FINNInternalError(f"Tried loading Aurora metadata from non-existing path: {p}")
        return AuroraNetworkMetadata.from_yaml(p.read_text())

    def save(self, p: Path | None = None) -> None:
        """Store data at the given path. If none is given, store to the location
        where the data was loaded from.
        """
        if p is None:
            if self.loaded_from_path is None:
                raise FINNInternalError(
                    "Cannot store Aurora metadata: No "
                    "path given and the metadata does "
                    "not seem to be loaded from an existing path."
                )
            p = self.loaded_from_path
        p.write_text(str(self.to_yaml()))

    def __getitem__(self, key: int) -> list[AuroraNetworkKernelMetadata]:
        """Get kernels on the given device."""
        if key not in self.data:
            raise FINNInternalError(
                f"Cannot get AuroraFlow kernels for device {key}. "
                f"Such a device does not exist in the metadata of "
                f"this design. Was the metadata object "
                f"instantiated and loaded properly?"
            )
        return self.data[key]

    def __setitem__(self, key: int, value: list[AuroraNetworkKernelMetadata]) -> None:
        """Set the specified kernel data."""
        self.data[key] = value

    def _has_free_connection(
        self, kernel: AuroraNetworkKernelMetadata, direction: DataDirection, partner: int
    ) -> bool:
        """Return whether the given kernel has a free spot/kernel
        in the given direction with the given partner device."""
        return kernel.partner_device == partner and kernel.connecting_kernels[direction] is None

    def _add_single_connection(
        self,
        on_device: int,
        on_node: str,
        other_device: int,
        other_node: str,
        direction: DataDirection,
        create_devices_implicitly: bool = True,
    ) -> None:
        """Add a single connection to the table (unidirectional)."""
        # First check that the devices exist or create them if necessary
        if on_device not in self.data.keys():
            if create_devices_implicitly:
                self.data[on_device] = []
            else:
                raise FINNInternalError(
                    f"Device {on_device} does not exist "
                    f"in the metadata and is not supposed "
                    f"to be created implicitly."
                )
        if other_device not in self.data:
            if create_devices_implicitly:
                self.data[other_device] = []
            else:
                raise FINNInternalError(
                    f"Device {other_device} does not exist "
                    f"in the metadata and is not supposed "
                    f"to be created implicitly."
                )

        # Check if there is a kernel that still has an unused connection in our direction
        for aurora_kernel in self.data[on_device]:
            if self._has_free_connection(aurora_kernel, direction, other_device):
                aurora_kernel.connecting_kernels[direction] = (on_node, other_node)
                return

        # Check if we already used all ports on this device. If yes, we cannot map
        # the partitioning.
        current_ports_used = len(self.data[on_device])
        if current_ports_used == self.ports_per_device:
            raise FINNMultiFPGAConfigError(
                f"Could not add a connection between devices "
                f"{on_device} and {other_device}, because "
                f"{on_device} already uses all of it's "
                f"{self.ports_per_device} communication ports. "
                f"Unable to map this model to this setup."
            )

        # Leave the path as None for now - this will be initialized by the AuroraFlow preparation
        # Instantiate new kernel for this device
        self.data[on_device].append(
            AuroraNetworkKernelMetadata(
                aurora_xo=None,
                partner_device=other_device,
                connecting_kernels={DataDirection.TX: None, DataDirection.RX: None},
            )
        )

        # Add the connection
        self.data[on_device][-1].connecting_kernels[direction] = (on_node, other_node)

    def add_connection(
        self,
        sender_device: int,
        sender_node: str,
        receiver_device: int,
        receiver_node: str,
    ) -> None:
        """Add a connection to the metadata. This modifies the internal
        representation to add kernels on both devices
        and store which nodes connect to which other nodes on which devices.
        If a kernel has an unused channel, it is
        utilized before creating a new kernel.
        """
        self._add_single_connection(
            sender_device, sender_node, receiver_device, receiver_node, DataDirection.TX
        )
        self._add_single_connection(
            receiver_device, receiver_node, sender_device, sender_node, DataDirection.RX
        )

    def node_is_sender(self, node: str) -> bool:
        """Return whether the given node acts as a sender (TX) on its device."""
        for device in self.data.keys():
            for i in range(len(self.data[device])):
                nodes = self.data[device][i].connecting_kernels[DataDirection.TX]
                if nodes is not None and nodes[0] == node:
                    return True
        return False

    def node_is_receiver(self, node: str) -> bool:
        """Return whether the given node acts as a receiver (RX) on its device."""
        for device in self.data.keys():
            for i in range(len(self.data[device])):
                nodes = self.data[device][i].connecting_kernels[DataDirection.RX]
                if nodes is not None and nodes[0] == node:
                    return True
        return False

    def get_partner_node(self, node: str, direction: DataDirection) -> str | None:
        """Search through all nodes and return the one that acts as a communication partner
        in the given direction. If A <-> B, then get_partner_node(A, TX) returns B and
        get_partner_node(B, RX) returns A. (get_partner_node(A, RX) would return None,
        since A is not a receiving from any node.)
        """  # noqa
        for device_data in self.data.values():
            for i in range(len(device_data)):
                node_tuple = device_data[i].connecting_kernels[direction]
                if node_tuple is not None and node_tuple[0] == node:
                    return node_tuple[1]
        return None

    def get_unprepared_aurora_kernels(self) -> list[tuple[int, int]]:
        """Return a list of all device:kernel_index combinations which do not yet have
        an associated AuroraFlow kernel XO file. This can be used for the packaging transformations.
        This also lists paths pointing to nowhere.
        """
        results = []
        for device in self.data.keys():
            for index in range(len(self[device])):
                if self[device][index].aurora_xo is None:
                    results.append((device, index))
                else:
                    if not cast("Path", self[device][index].aurora_xo).exists():
                        results.append((device, index))
        return results

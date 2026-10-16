"""Manage metadata for connecting multiple FPGAs using different technologies and topologies."""

from __future__ import annotations

from dataclasses import dataclass, field
from mashumaro.mixins.yaml import DataClassYAMLMixin

# Needs to be outside the type-checking block, so that DataClassYAMLMixin can pick it up
# for automatic serialization.
from pathlib import Path
from typing import TYPE_CHECKING, cast

from finn.transformation.fpgadataflow.multifpga.metadata import DataDirection, NetworkMetadata
from finn.util.exception import FINNInternalError, FINNMultiFPGAConfigError

if TYPE_CHECKING:
    from typing import Self

    from finn.builder.build_dataflow_config import MultiFPGAConfiguration


@dataclass
class AuroraLinkEndpoint:
    """Dataclass to keep track of a connection between two devices, considering one
    of the two endpoints. It tracks the AuroraFlow kernel XO file used for this endpoint,
    which device it connects to, which link/port-index this endpoint and the other connected
    endpoint use, as well as which kernels communicate in each direction."""

    aurora_xo: Path | None
    """Path to the kernel XO file. If it has not been set yet, this is left as None."""

    partner_device: int
    """ID of the partner device."""

    link_index: int | None
    """Index of the QSFP port / link of the FPGA that this endpoint sits at."""

    partner_link_index: int | None
    """Index of the QSFP port / link that the other endpoint occupies."""

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
    AuroraLinkEndpoint objects which describe the configuration of
    a single AuroraFlow kernel.
    """

    data: dict[int, list[AuroraLinkEndpoint]] = field(default_factory=dict)
    links_per_device: int = 2
    # Not serialized, since it is only valid for the file it was loaded from. Set by load().
    loaded_from_path: Path | None = field(default=None, metadata={"serialize": "omit"})

    @classmethod
    def create_from_multifpga_configuration(
        cls, mfcfg: MultiFPGAConfiguration
    ) -> AuroraNetworkMetadata:
        """Create an empty metadata object, using the configured number of links per device."""
        return cls(links_per_device=mfcfg.links_per_device)

    @classmethod
    def load(cls, p: Path) -> Self:
        """Load from a YAML file. Afterwards, `save()` without a path stores to the same file."""
        if not p.exists():
            raise FINNInternalError(f"Tried loading Aurora metadata from non-existing path: {p}")
        meta = cls.from_yaml(p.read_text())
        meta.loaded_from_path = p
        return meta

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

    def __getitem__(self, key: int) -> list[AuroraLinkEndpoint]:
        """Get kernels on the given device."""
        if key not in self.data:
            raise FINNInternalError(
                f"Cannot get AuroraFlow kernels for device {key}. "
                f"Such a device does not exist in the metadata of "
                f"this design. Was the metadata object "
                f"instantiated and loaded properly?"
            )
        return self.data[key]

    def __setitem__(self, key: int, value: list[AuroraLinkEndpoint]) -> None:
        """Set the specified kernel data."""
        self.data[key] = value

    def _has_free_connection(
        self, kernel: AuroraLinkEndpoint, direction: DataDirection, partner: int
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
    ) -> int:
        """Add a single connection to the table (unidirectional).
        Returns the QSFP port / link that was used for this connection."""
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
                if aurora_kernel.link_index is None:
                    raise FINNInternalError(
                        f"Link index of existing connection is not set! ({on_device}) "
                        f"{on_node} -> ({other_device}) {other_node}"
                    )
                return aurora_kernel.link_index

        # Check if we already used all ports on this device. If yes, we cannot map
        # the partitioning.
        current_ports_used = len(self.data[on_device])
        if current_ports_used == self.links_per_device:
            raise FINNMultiFPGAConfigError(
                f"Could not add a connection between devices "
                f"{on_device} and {other_device}, because "
                f"{on_device} already uses all of it's "
                f"{self.links_per_device} communication links. "
                f"Unable to map this model to this setup."
            )

        # Leave the path as None for now - this will be initialized by the AuroraFlow preparation
        # Instantiate new kernel for this device
        utilized_port_count = len(self.data[on_device])
        self.data[on_device].append(
            AuroraLinkEndpoint(
                aurora_xo=None,
                partner_device=other_device,
                connecting_kernels={DataDirection.TX: None, DataDirection.RX: None},
                link_index=utilized_port_count,
                partner_link_index=None,
            )
        )

        # Add the connection
        self.data[on_device][-1].connecting_kernels[direction] = (on_node, other_node)

        # Return which port this connection sits on
        link_index = self.data[on_device][-1].link_index
        assert link_index is not None
        return link_index

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
        sender_link = self._add_single_connection(
            sender_device, sender_node, receiver_device, receiver_node, DataDirection.TX
        )
        receiver_link = self._add_single_connection(
            receiver_device, receiver_node, sender_device, sender_node, DataDirection.RX
        )

        # Both endpoints sit at the two ends of the same cable, so each one has to point to the
        # link of the other. An endpoint whose other direction is already in use must already
        # point to the same partner link, otherwise the two devices are out of sync.
        for endpoint, partner_link in (
            (
                self.get_endpoint(
                    sender_device, sender_node, receiver_device, receiver_node, DataDirection.TX
                ),
                receiver_link,
            ),
            (
                self.get_endpoint(
                    receiver_device, receiver_node, sender_device, sender_node, DataDirection.RX
                ),
                sender_link,
            ),
        ):
            if endpoint.partner_link_index not in (None, partner_link):
                raise FINNInternalError(
                    f"Cannot connect {sender_node} [{sender_device}, link {sender_link}] -> "
                    f"{receiver_node} [{receiver_device}, link {receiver_link}]: the endpoint at "
                    f"link {endpoint.link_index} is already cabled to link "
                    f"{endpoint.partner_link_index} of device {endpoint.partner_device}."
                )
            endpoint.partner_link_index = partner_link

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

    def get_endpoint(
        self,
        device: int,
        node: str,
        partner_device: int,
        partner_node: str,
        direction: DataDirection,
    ) -> AuroraLinkEndpoint:
        """Return the endpoint on `device` that connects `node` with `partner_node` on
        `partner_device` in the given direction. The node order follows `connecting_kernels`:
        for TX, `node` sends to `partner_node`, for RX, `node` receives from `partner_node`.
        The endpoint is searched by its content, not by its position in the list of
        endpoints of the device.
        """
        matches = [
            endpoint
            for endpoint in self[device]
            if endpoint.partner_device == partner_device
            and endpoint.connecting_kernels[direction] == (node, partner_node)
        ]
        if len(matches) != 1:
            raise FINNInternalError(
                f"Expected exactly one AuroraFlow endpoint on device {device} for the "
                f"{direction.name} connection {node} [{device}] <-> {partner_node} "
                f"[{partner_device}], but found {len(matches)}."
            )
        return matches[0]

    def get_unprepared_aurora_kernels(self) -> list[tuple[int, AuroraLinkEndpoint]]:
        """Return a list of all (device, endpoint) combinations which do not yet have
        an associated AuroraFlow kernel XO file. This can be used for the packaging transformations.
        This also lists paths pointing to nowhere.
        """
        results = []
        for device in self.data.keys():
            for endpoint in self[device]:
                if endpoint.aurora_xo is None or not cast("Path", endpoint.aurora_xo).exists():
                    results.append((device, endpoint))
        return results

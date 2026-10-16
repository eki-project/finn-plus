"""Manage metadata for connecting multiple FPGAs using different technologies and topologies."""

from __future__ import annotations

import contextlib
import json
import networkx as nx
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from mashumaro.mixins.yaml import DataClassYAMLMixin
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Needs to be outside the type-checking block, so that DataClassYAMLMixin can pick it up
# for automatic serialization.
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from finn.builder.build_dataflow_config import MFTopology
from finn.transformation.fpgadataflow.multifpga.metadata import DataDirection, NetworkMetadata
from finn.util.exception import FINNInternalError, FINNMultiFPGAConfigError, FINNUserError
from finn.util.fpgadataflow import get_device_id
from finn.util.logging import log
from finn.util.platforms import platforms

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper
    from typing import Self

    from finn.builder.build_dataflow_config import DataflowBuildConfig, MultiFPGAConfiguration


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
        """Create an empty metadata object, using the configured topology and number of links
        per device."""
        return cls(topology=mfcfg.topology, links_per_device=mfcfg.links_per_device)

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

    def export(self, path: Path) -> None:
        """Export the metadata using NetworkX node_link_data.
        Can be used to configure cabling and switch."""
        g = nx.Graph()
        for device, endpoints in self.data.items():
            for endpoint in endpoints:
                if endpoint.connecting_kernels[DataDirection.TX] is not None:
                    g.add_edge(
                        (device, endpoint.link_index),
                        (endpoint.partner_device, endpoint.partner_link_index),
                    )
        path.write_text(json.dumps(nx.node_link_data(g), indent=4))
        log.info(f"Exported Multi-FPGA node-link data to {path}")

    def plot(
        self,
        path: Path,
        model: ModelWrapper | None = None,
        cfg: DataflowBuildConfig | None = None,
    ) -> None:
        """Draw the cabling of the AuroraFlow network. The image format is given by the suffix
        of `path` (e.g. `.svg` or `.png`).

        Every FPGA is a box and every used QSFP port a square `Q<link index>` on its edge. A cable
        is a thick line between two ports, with one arrow per used direction, labelled with the
        sending and receiving SDP. If `model` is given, arrows inside the boxes show how the data
        passes through the SDPs of each FPGA, from the port (or host) it arrives at to the port
        (or host) it leaves through. If `cfg` is given, the platform interfaces of the ports are
        listed.
        """
        if not self.data:
            log.warning("Cannot plot the Multi-FPGA network: the metadata contains no devices.")
            return
        devices = sorted(self.data)
        box = np.array([5.2, 3.2])  # Width and height of an FPGA box
        pitch = box[0] + 3.8  # Distance between the centers of neighbouring boxes
        port_size, lane = 0.56, 0.14  # Side of a port square, offset of the TX/RX arrows
        colors = {
            "background": "#0b0f19",
            "fpga": "#111827",
            "fpga_edge": "#3b82f6",
            "port": "#1d4ed8",
            "port_edge": "#93c5fd",
            "cable": "#1e3a8a",
            "data": "#38bdf8",
            "compute": "#818cf8",
            "compute_fill": "#1e1b4b",
            "host": "#64748b",
            "text": "#e5e7eb",
            "muted": "#94a3b8",
        }

        def short(node: str) -> str:
            """Shorten the default SDP node names."""
            return node.replace("StreamingDataflowPartition", "SDP")

        # 1. Place the FPGAs (their box centers) according to the topology
        match self.topology:
            case MFTopology.SIMPLEX_CHAIN | MFTopology.DUPLEX_CHAIN:
                # Chains pass the devices in the order of their IDs, so a row from left to right
                # follows the data. The way back of a duplex chain uses the same cables and shows
                # up as a second arrow on each of them, so it needs no other arrangement.
                centers = {d: np.array([i * pitch, 0.0]) for i, d in enumerate(devices)}
            case MFTopology.RING:
                # Clockwise on a circle, starting on the left. The radius keeps neighbouring
                # boxes `pitch` apart, so the cable closing the ring is as long as the others.
                radius = pitch / (2 * np.sin(np.pi / len(devices))) if len(devices) > 1 else 0
                angles = np.pi - 2 * np.pi * np.arange(len(devices)) / len(devices)
                centers = {
                    d: radius * np.array([np.cos(a), np.sin(a)])
                    for d, a in zip(devices, angles, strict=True)
                }
            case _:
                log.error(
                    f"Cannot plot the Multi-FPGA network: no layout algorithm is implemented for "
                    f"the topology '{self.topology.value}' yet."
                )
                return

        # 2. Collect the cables, each as the sorted pair of the (device, link) ports it connects
        cables = sorted(
            {
                tuple(sorted([(d, e.link_index), (e.partner_device, e.partner_link_index)]))
                for d in devices
                for e in self[d]
            }
        )
        cable_number = {port: i for i, cable in enumerate(cables) for port in cable}

        # 3. Put every port on the side of its box that faces the partner device. A side is
        # given by its outward normal, e.g. (1, 0) for the right side.
        endpoints_per_side: dict[
            tuple[int, tuple[int, int]], list[AuroraLinkEndpoint]
        ] = defaultdict(list)
        for device in devices:
            for e in self[device]:
                dx, dy = centers[e.partner_device] - centers[device]
                normal = (int(np.sign(dx)), 0) if abs(dx) >= abs(dy) else (0, int(np.sign(dy)))
                endpoints_per_side[(device, normal)].append(e)
        ports = {}
        for (device, normal), endpoints in endpoints_per_side.items():
            along = np.array([-normal[1], normal[0]])  # Direction along the side
            # Order the ports by the position of their partner along the side, so that cables do
            # not cross. Several cables to the same device are ordered by their number, mirrored
            # on the two devices (`along` points the other way there), so that they run parallel.
            endpoints.sort(
                key=lambda e, device=device, along=along: (
                    centers[e.partner_device] @ along,
                    cable_number[(device, e.link_index)] * along.sum(),
                )
            )
            # Spread the ports evenly along the side
            for i, e in enumerate(endpoints):
                offset = (i + 1) / (len(endpoints) + 1) - 0.5
                ports[(device, e.link_index)] = (
                    centers[device] + (np.array(normal) / 2 + along * offset) * box
                )

        def pin(device: int, e: AuroraLinkEndpoint, direction: DataDirection) -> np.ndarray:
            """Return where the arrow of one direction of a cable meets the port of `e`. The
            two directions use parallel lanes, each shifted to the left of its data direction.
            """
            p, q = ports[(device, e.link_index)], ports[(e.partner_device, e.partner_link_index)]
            left = np.array([p[1] - q[1], q[0] - p[0]]) / np.linalg.norm(q - p)
            return p + left * (lane if direction == DataDirection.TX else -lane)

        # 4. Set up a dark figure. Its size follows the extent of the layout, with room for the
        # title above and for the host connections and notes below.
        corners = np.array([c + s * box / 2 for c in centers.values() for s in (-1, 1)])
        low = corners.min(axis=0) - [1.5, 3.4]
        high = corners.max(axis=0) + np.array([1.5, 1.4])
        fig = Figure(figsize=tuple((high - low) * 0.45), facecolor=colors["background"])
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.set_axis_off()

        def arrow(start: np.ndarray, end: np.ndarray, color: str, **kwargs: Any) -> None:
            """Draw an arrow. `patchA` and `patchB` cut it off at the outline of these shapes."""
            ax.add_patch(
                FancyArrowPatch(
                    tuple(start),
                    tuple(end),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=1.5,
                    color=color,
                    zorder=3,
                    **kwargs,
                )
            )

        # 5. Draw the FPGAs and their ports
        for device, center in centers.items():
            ax.add_patch(
                FancyBboxPatch(
                    tuple(center - box / 2),
                    *box,
                    boxstyle="round,pad=0,rounding_size=0.35",
                    facecolor=colors["fpga"],
                    edgecolor=colors["fpga_edge"],
                    linewidth=1.6,
                    zorder=1,
                )
            )
            # Title in the top left corner, out of the way of ports in the middle of the top side
            ax.text(
                *(center + np.array([-box[0] / 2 + 0.35, box[1] / 2 - 0.45])),
                f"FPGA {device}",
                color=colors["text"],
                fontsize=11,
                fontweight="bold",
                ha="left",
                va="center",
            )
        port_patches = {}
        for (device, link), point in ports.items():
            port_patches[(device, link)] = ax.add_patch(
                FancyBboxPatch(
                    tuple(point - port_size / 2),
                    port_size,
                    port_size,
                    boxstyle="round,pad=0,rounding_size=0.08",
                    facecolor=colors["port"],
                    edgecolor=colors["port_edge"],
                    linewidth=1.2,
                    zorder=4,
                )
            )
            ax.text(
                *point,
                f"Q{link}",
                color="white",
                fontsize=7,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=5,
            )

        # 6. Draw the cables: a thick line per cable, and an arrow per used direction from the
        # sending to the receiving port, labelled with the SDPs on both sides
        for a, b in cables:
            ax.plot(
                *np.array([ports[a], ports[b]]).T,
                color=colors["cable"],
                linewidth=10,
                solid_capstyle="round",
                zorder=2,
            )
        for device in devices:
            for e in self[device]:
                nodes = e.connecting_kernels[DataDirection.TX]
                if nodes is None:
                    continue
                partner = self.get_endpoint(
                    e.partner_device, nodes[1], device, nodes[0], DataDirection.RX
                )
                start = pin(device, e, DataDirection.TX)
                end = pin(e.partner_device, partner, DataDirection.RX)
                arrow(
                    start,
                    end,
                    colors["data"],
                    patchA=port_patches[(device, e.link_index)],
                    patchB=port_patches[(e.partner_device, partner.link_index)],
                )
                # Put the label on the outer side of the lane. If it would be upside down, turn
                # it, and then also reverse the text, so that its arrow still follows the data.
                dx, dy = end - start
                angle = np.degrees(np.arctan2(dy, dx))
                label = f"{short(nodes[0])} → {short(nodes[1])}"
                if not -90 < angle <= 90:
                    angle -= 180 * np.sign(angle)
                    label = f"{short(nodes[1])} ← {short(nodes[0])}"
                outward = (start - ports[(device, e.link_index)]) / lane
                ax.text(
                    *((start + end) / 2 + outward * 0.3),
                    label,
                    color=colors["data"],
                    fontsize=7,
                    rotation=angle,
                    rotation_mode="anchor",
                    ha="center",
                    va="center",
                )

        # 7. With the model: follow the data through the SDPs of each FPGA, from where it enters
        # the FPGA (a port or the host) to where it leaves it, and draw every such path as two
        # arrows through a compute block that names the first and the last SDP on the path
        if model is not None:
            # Where data enters an FPGA: (device, first SDP, point, port square or None)
            entries: list[tuple[int, str, np.ndarray, FancyBboxPatch | None]] = []
            # Where data leaves an FPGA, per SDP: (point, port square or None)
            exits: dict[str, list[tuple[np.ndarray, FancyBboxPatch | None]]] = defaultdict(list)
            for device in devices:
                for e in self[device]:
                    port_patch = port_patches[(device, e.link_index)]
                    if (rx := e.connecting_kernels[DataDirection.RX]) is not None:
                        entries.append(
                            (device, rx[0], pin(device, e, DataDirection.RX), port_patch)
                        )
                    if (tx := e.connecting_kernels[DataDirection.TX]) is not None:
                        exits[tx[0]].append((pin(device, e, DataDirection.TX), port_patch))

            # The host is connected to the bottom of the FPGAs with the model inputs and outputs
            host_x = box[0] / 2 - 0.5
            for tensor in model.graph.input:
                for node in model.find_consumers(tensor.name) or []:
                    device = cast("int", get_device_id(node))
                    point = centers[device] + np.array([-host_x, -box[1] / 2])
                    arrow(point - [0, 1.0], point, colors["host"])
                    ax.text(
                        *(point - [0, 1.3]),
                        "host in",
                        color=colors["muted"],
                        fontsize=7,
                        ha="center",
                        va="center",
                    )
                    entries.append((device, node.name, point, None))
            for tensor in model.graph.output:
                node = model.find_producer(tensor.name)
                device = cast("int", get_device_id(node))
                point = centers[device] + np.array([host_x, -box[1] / 2])
                arrow(point, point - [0, 1.0], colors["host"])
                ax.text(
                    *(point - [0, 1.3]),
                    "host out",
                    color=colors["muted"],
                    fontsize=7,
                    ha="center",
                    va="center",
                )
                exits[node.name].append((point, None))

            # Search forward from every entry through the SDPs on the same FPGA. Every exit on
            # the way gives a path: (start, start port, end, end port, label of the block).
            paths: dict[int, list[tuple]] = defaultdict(list)
            for device, first, start, start_patch in entries:
                todo, seen = [first], set()
                while todo:
                    name = todo.pop()
                    if name in seen:
                        continue
                    seen.add(name)
                    label = short(first) if name == first else f"{short(first)} … {short(name)}"
                    paths[device] += [
                        (start, start_patch, end, end_patch, label)
                        for end, end_patch in exits[name]
                    ]
                    successors = model.find_direct_successors(model.get_node_from_name(name))
                    todo += [s.name for s in successors or [] if get_device_id(s) == device]

            for device, device_paths in paths.items():
                # Stack the blocks of an FPGA, ordered by the height of the ports they connect,
                # so that the arrows of different paths do not cross
                device_paths.sort(key=lambda path: -(path[0][1] + path[2][1]))
                step = min(0.85, (box[1] - 1.3) / len(device_paths))
                for i, (start, start_patch, end, end_patch, label) in enumerate(device_paths):
                    height = -0.25 + ((len(device_paths) - 1) / 2 - i) * step
                    center = centers[device] + np.array([0, height])
                    ax.add_patch(
                        FancyBboxPatch(
                            tuple(center - [1.3, 0.3]),
                            2.6,
                            0.6,
                            boxstyle="round,pad=0,rounding_size=0.15",
                            facecolor=colors["compute_fill"],
                            edgecolor=colors["compute"],
                            linewidth=1.2,
                            zorder=4,
                        )
                    )
                    ax.text(
                        *center,
                        f"⚙ {label}",
                        color=colors["text"],
                        fontsize=7,
                        ha="center",
                        va="center",
                        zorder=5,
                    )
                    # The data enters the block slightly above its middle, on the side facing
                    # where it comes from, and leaves it slightly below, on the side facing where
                    # it goes. This keeps a path that returns through the same port readable.
                    side_in = np.sign(start[0] - center[0]) or -1
                    side_out = np.sign(end[0] - center[0]) or 1
                    block_in = center + np.array([side_in * 1.3, 0.12])
                    block_out = center + np.array([side_out * 1.3, -0.12])
                    arrow(
                        start,
                        block_in,
                        colors["compute"],
                        patchA=start_patch,
                        connectionstyle="arc3,rad=0.1",
                    )
                    arrow(
                        block_out,
                        end,
                        colors["compute"],
                        patchB=end_patch,
                        connectionstyle="arc3,rad=0.1",
                    )

        # 8. Title, and notes on how to read the diagram
        ax.text(
            low[0] + 0.6,
            high[1] - 0.7,
            f"AuroraFlow network: {len(devices)} FPGAs, {self.topology.value.replace('_', ' ')}",
            color=colors["text"],
            fontsize=13,
            fontweight="bold",
            va="center",
        )
        notes = [
            "Thick lines: cables between the QSFP ports Q<link index>. "
            "Arrows: data, from the sending to the receiving SDP."
        ]
        if model is not None:
            notes.append("⚙: SDPs the data passes on an FPGA (first … last).")
        if cfg is not None and cfg.board in platforms:
            # Platforms without known QSFP interface names raise, they are left out then
            with contextlib.suppress(FINNUserError):
                platform = platforms[cfg.board]()
                notes.append(
                    f"Board {cfg.board}: "
                    + ", ".join(
                        f"Q{link} = {platform.get_qsfp_gt_interface(link)}"
                        for link in sorted({link for _, link in ports})
                    )
                )
        ax.text(
            low[0] + 0.6,
            low[1] + 0.5,
            "\n".join(notes),
            color=colors["muted"],
            fontsize=8,
            va="bottom",
            linespacing=1.6,
        )

        fig.savefig(path, facecolor=fig.get_facecolor())
        log.info(f"Saved a diagram of the Multi-FPGA network to {path}")

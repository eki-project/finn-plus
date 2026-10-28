"""Manage metadata for connecting multiple FPGAs using different technologies and topologies."""

from __future__ import annotations

import json
import networkx as nx
import shutil
from dataclasses import dataclass, field
from mashumaro.mixins.yaml import DataClassYAMLMixin

# Needs to be outside the type-checking block, so that DataClassYAMLMixin can pick it up
# for automatic serialization.
from pathlib import Path
from typing import TYPE_CHECKING, cast

from finn.transformation.fpgadataflow.multifpga.metadata import DataDirection, NetworkMetadata
from finn.util.basic import launch_process_helper
from finn.util.exception import FINNInternalError, FINNMultiFPGAConfigError
from finn.util.fpgadataflow import get_device_id
from finn.util.logging import log

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper
    from typing import Self

    from finn.builder.build_dataflow_config import MultiFPGAConfiguration


@dataclass
class AuroraUnidirectionalConnection:
    """Dataclass storing the fact that we need to connect
    <A>:m_axis_<A_index>  to  <B>:s_axis_<B_index>."""

    source_kernel: str
    source_kernel_stream_index: int
    target_kernel: str
    target_kernel_stream_index: int


@dataclass
class AuroraLinkEndpoint:
    """Dataclass to keep track of a connection between two devices, considering one
    of the two endpoints. It tracks the AuroraFlow kernel XO file used for this endpoint,
    which device it connects to, which QSFP port index this endpoint and the other connected
    endpoint use, as well as which kernels communicate in each direction. The QSFP port index is
    not the index of an SDP stream (`m_axis_<i>` / `s_axis_<j>`), which the connections store as
    stream index."""

    aurora_xo: Path | None
    """Path to the kernel XO file. If it has not been set yet, this is left as None."""

    partner_device: int
    """ID of the partner device."""

    port_index: int
    """Index of the QSFP port of the FPGA that this endpoint sits at. Always known when the
    endpoint is created."""

    partner_port_index: int | None
    """Index of the QSFP port that the other endpoint occupies."""

    tx: AuroraUnidirectionalConnection | None
    rx: AuroraUnidirectionalConnection | None
    """Connections with other devices. One entry for RX, one for TX.
    The value can be None, in which case no connection for this direction
    was established yet, or it contains the sending and receiving kernels and their stream indices.

    TX: SDP0:m_axis_0 -> SDP1:s_axis_1 would mean that SDP0 (this device) sends to SDP1 (other
        device)
    RX: SDP0:m_axis_0 -> SDP1:s_axis_1 would mean that SDP0 (other device) sends to SDP1 (this
        device)

    I.e.: For TX the source_kernel is on THIS device, for RX the target_kernel is on THIS device.

    A complete connection requires the same connection as TX on the sending and as RX on the
    receiving device.
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

    def _has_free_tx_connection(self, endpoint: AuroraLinkEndpoint, partner: int) -> bool:
        """Return whether the given kernel has a free spot/kernel
        in the given direction with the given partner device."""
        return endpoint.partner_device == partner and endpoint.tx is None

    def _has_free_rx_connection(self, endpoint: AuroraLinkEndpoint, partner: int) -> bool:
        """Return whether the given kernel has a free spot/kernel
        in the given direction with the given partner device."""
        return endpoint.partner_device == partner and endpoint.rx is None

    def _has_free_connection(
        self, endpoint: AuroraLinkEndpoint, direction: DataDirection, partner: int
    ) -> bool:
        """Return whether the given kernel has a free spot/kernel
        in the given direction with the given partner device."""
        match direction:
            case DataDirection.TX:
                return self._has_free_tx_connection(endpoint, partner)
            case DataDirection.RX:
                return self._has_free_rx_connection(endpoint, partner)
            case _:
                raise FINNInternalError(
                    f"Unknown direction {direction} when searching a free connection."
                )

    def _add_single_connection(
        self,
        on_device: int,
        on_node: str,
        on_tensor_index: int,
        other_device: int,
        other_node: str,
        other_tensor_index: int,
        direction: DataDirection,
        create_devices_implicitly: bool = True,
    ) -> int:
        """Add a single connection to the table (unidirectional).
        Returns the index of the QSFP port that was used for this connection."""
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

        # The connection to be added. It always points from the sender to the receiver, so that
        # both endpoints of a cable store the same connection
        match direction:
            case DataDirection.TX:
                connection = AuroraUnidirectionalConnection(
                    on_node, on_tensor_index, other_node, other_tensor_index
                )
            case DataDirection.RX:
                connection = AuroraUnidirectionalConnection(
                    other_node, other_tensor_index, on_node, on_tensor_index
                )
            case _:
                raise FINNInternalError(
                    "Unknown direction encountered while creating metadata connection."
                )

        # Check if there is a kernel that still has an unused connection in our direction
        for aurora_kernel in self.data[on_device]:
            if not self._has_free_connection(aurora_kernel, direction, other_device):
                continue
            match direction:
                case DataDirection.TX:
                    aurora_kernel.tx = connection
                case DataDirection.RX:
                    aurora_kernel.rx = connection
                case _:
                    raise FINNInternalError(
                        "Unknown direction encountered while creating metadata connection."
                    )
            return aurora_kernel.port_index

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
                tx=None,
                rx=None,
                port_index=utilized_port_count,
                partner_port_index=None,
            )
        )

        # Add the connection
        match direction:
            case DataDirection.TX:
                self.data[on_device][-1].tx = connection
            case DataDirection.RX:
                self.data[on_device][-1].rx = connection
            case _:
                raise FINNInternalError(
                    "Unknown direction encountered while creating metadata connection."
                )

        # Return which port this connection sits on
        return self.data[on_device][-1].port_index

    def add_connection(
        self,
        sender_device: int,
        sender_node: str,
        sender_tensor_index: int,
        receiver_device: int,
        receiver_node: str,
        receiver_tensor_index: int,
    ) -> None:
        """Add a connection to the metadata. This modifies the internal
        representation to add kernels on both devices
        and store which nodes connect to which other nodes on which devices.
        If a kernel has an unused channel, it is
        utilized before creating a new kernel.
        """
        sender_port = self._add_single_connection(
            sender_device,
            sender_node,
            sender_tensor_index,
            receiver_device,
            receiver_node,
            receiver_tensor_index,
            DataDirection.TX,
        )
        receiver_port = self._add_single_connection(
            receiver_device,
            receiver_node,
            receiver_tensor_index,
            sender_device,
            sender_node,
            sender_tensor_index,
            DataDirection.RX,
        )

        # Both endpoints sit at the two ends of the same cable, so each one has to point to the
        # QSFP port of the other. An endpoint whose other direction is already in use must already
        # point to the same partner port, otherwise the two devices are out of sync.
        connection = AuroraUnidirectionalConnection(
            sender_node, sender_tensor_index, receiver_node, receiver_tensor_index
        )
        for endpoint, partner_port in (
            (self.get_endpoint(sender_device, connection, DataDirection.TX), receiver_port),
            (self.get_endpoint(receiver_device, connection, DataDirection.RX), sender_port),
        ):
            if endpoint.partner_port_index not in (None, partner_port):
                raise FINNInternalError(
                    f"Cannot connect {sender_node} [{sender_device}, port {sender_port}] -> "
                    f"{receiver_node} [{receiver_device}, port {receiver_port}]: the endpoint at "
                    f"port {endpoint.port_index} is already cabled to port "
                    f"{endpoint.partner_port_index} of device {endpoint.partner_device}."
                )
            endpoint.partner_port_index = partner_port

    def node_is_sender(self, node: str) -> bool:
        """Return whether the given node acts as a sender (TX) on its device."""
        for device in self.data.keys():
            for endpoint in self.data[device]:
                if endpoint.tx is not None and endpoint.tx.source_kernel == node:
                    return True
        return False

    def node_is_receiver(self, node: str) -> bool:
        """Return whether the given node acts as a receiver (RX) on its device."""
        for device in self.data.keys():
            for endpoint in self.data[device]:
                if endpoint.rx is not None and endpoint.rx.target_kernel == node:
                    return True
        return False

    def get_partner_node(
        self, node: str, stream_index: int, direction: DataDirection
    ) -> tuple[int, str, int] | None:
        """Search through all nodes and return the one that acts as a communication partner
        of the given stream of `node` in the given direction, as (partner device, partner node,
        partner stream index). For TX, `stream_index` is the index of the `m_axis` stream of
        `node`, for RX the index of its `s_axis` stream.
        If A:m_axis_0 (device 0) -> B:s_axis_1 (device 1),
        then get_partner_node(A, 0, TX) returns (1, B, 1) and get_partner_node(B, 1, RX) returns
        (0, A, 0). If nothing is connected to the stream in the given direction, None is returned.
        """
        for device_data in self.data.values():
            for endpoint in device_data:
                match direction:
                    case DataDirection.TX:
                        if (
                            endpoint.tx is not None
                            and endpoint.tx.source_kernel == node
                            and endpoint.tx.source_kernel_stream_index == stream_index
                        ):
                            return (
                                endpoint.partner_device,
                                endpoint.tx.target_kernel,
                                endpoint.tx.target_kernel_stream_index,
                            )
                    case DataDirection.RX:
                        if (
                            endpoint.rx is not None
                            and endpoint.rx.target_kernel == node
                            and endpoint.rx.target_kernel_stream_index == stream_index
                        ):
                            return (
                                endpoint.partner_device,
                                endpoint.rx.source_kernel,
                                endpoint.rx.source_kernel_stream_index,
                            )
                    case _:
                        raise FINNInternalError(
                            f"Unknown direction {direction} when searching the partner node."
                        )
        return None

    def get_endpoint(
        self,
        device: int,
        connection: AuroraUnidirectionalConnection,
        direction: DataDirection,
    ) -> AuroraLinkEndpoint:
        """Return the endpoint on `device` that carries `connection` in the given direction.
        For TX, `device` is the device of the sending kernel, for RX the device of the
        receiving kernel. The endpoint is searched by its content, not by its position in the
        list of endpoints of the device.
        """
        match direction:
            case DataDirection.TX:
                matches = [endpoint for endpoint in self[device] if endpoint.tx == connection]
            case DataDirection.RX:
                matches = [endpoint for endpoint in self[device] if endpoint.rx == connection]
            case _:
                raise FINNInternalError(
                    f"Unknown direction {direction} when searching the endpoint."
                )
        if len(matches) != 1:
            raise FINNInternalError(
                f"Expected exactly one AuroraFlow endpoint on device {device} for the "
                f"{direction.name} connection {connection.source_kernel}:m_axis_"
                f"{connection.source_kernel_stream_index} -> {connection.target_kernel}:s_axis_"
                f"{connection.target_kernel_stream_index}, but found {len(matches)}."
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
                if endpoint.tx is not None:
                    g.add_edge(
                        (device, endpoint.port_index),
                        (endpoint.partner_device, endpoint.partner_port_index),
                    )
        path.write_text(json.dumps(nx.node_link_data(g), indent=4))
        log.info(f"Exported Multi-FPGA node-link data to {path}")

    @staticmethod
    def short_node_name(node: str) -> str:
        """Shorten the default SDP node names for diagrams."""
        return node.replace("StreamingDataflowPartition", "SDP")

    def plot_dot(self, path: Path, model: ModelWrapper | None = None) -> None:
        """Write the AuroraFlow network as a Graphviz DOT graph to `path`. If Graphviz' `dot` is
        found, the graph is also rendered as an SVG with the same name next to it.

        Every FPGA is a cluster with its used QSFP ports `Q<port index>`. Every used direction of
        a cable is an edge from the sending to the receiving port, labelled with the sending and
        receiving SDP. If `model` is given, the SDPs are placed in the cluster of their FPGA and
        connected to each other, to the ports they send and receive through, and to the host.
        """
        short = self.short_node_name

        def quote(name: str) -> str:
            """Return `name` as a quoted DOT ID."""
            return '"' + name.replace("\\", "\\\\").replace('"', '\\"') + '"'

        def port(device: int, port_index: int | None) -> str:
            """Return the DOT ID of a QSFP port."""
            return quote(f"FPGA{device}_Q{port_index}")

        title = (
            f"AuroraFlow network: {len(self.data)} FPGAs, "
            f"{self.topology.value.replace('_', ' ')}"
        )
        lines = [
            "digraph multifpga {",
            f"    label={quote(title)}; labelloc=t; rankdir=LR;",
            "    node [fontname=Helvetica, fontsize=10]; edge [fontname=Helvetica, fontsize=8];",
        ]

        # Every FPGA is a cluster with its ports and, if the model is given, its SDPs
        for device in sorted(self.data):
            lines.append(f"    subgraph cluster_fpga{device} {{")
            lines.append(f'        label="FPGA {device}"; style=rounded;')
            lines += [
                f"        {port(device, e.port_index)} "
                f'[label="Q{e.port_index}", shape=square, style=filled, fillcolor=lightblue];'
                for e in self[device]
            ]
            if model is not None:
                lines += [
                    f"        {quote(node.name)} [label={quote(short(node.name))}, shape=box];"
                    for node in model.graph.node
                    if get_device_id(node) == device
                ]
            lines.append("    }")

        # One edge per used direction of a cable, from the sending to the receiving port. With the
        # model also from the sending SDP to its port, and from the port to the receiving SDP.
        for device in sorted(self.data):
            for e in self[device]:
                if e.tx is not None:
                    label = f"{short(e.tx.source_kernel)} → {short(e.tx.target_kernel)}"
                    lines.append(
                        f"    {port(device, e.port_index)} -> "
                        f"{port(e.partner_device, e.partner_port_index)} "
                        f"[label={quote(label)}, penwidth=2];"
                    )
                    if model is not None:
                        lines.append(
                            f"    {quote(e.tx.source_kernel)} -> {port(device, e.port_index)};"
                        )
                if model is not None and e.rx is not None:
                    lines.append(
                        f"    {port(device, e.port_index)} -> {quote(e.rx.target_kernel)};"
                    )

        # With the model: the data between the SDPs of the same FPGA, and from and to the host
        if model is not None:
            for node in model.graph.node:
                lines += [
                    f"    {quote(node.name)} -> {quote(successor.name)};"
                    for successor in model.find_direct_successors(node) or []
                    if get_device_id(successor) == get_device_id(node)
                ]
            lines.append('    host_in [label="host in", shape=ellipse];')
            lines.append('    host_out [label="host out", shape=ellipse];')
            for tensor in model.graph.input:
                lines += [
                    f"    host_in -> {quote(node.name)};"
                    for node in model.find_consumers(tensor.name) or []
                ]
            for tensor in model.graph.output:
                if (producer := model.find_producer(tensor.name)) is not None:
                    lines.append(f"    {quote(producer.name)} -> host_out;")
        lines.append("}")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info(f"Saved a DOT graph of the Multi-FPGA network to {path}")

        dot = shutil.which("dot")
        if dot is None:
            return
        svg = path.with_suffix(".svg")
        launch_process_helper([dot, "-Tsvg", path, "-o", svg], print_stdout=False)
        log.info(f"Rendered the DOT graph of the Multi-FPGA network to {svg}")

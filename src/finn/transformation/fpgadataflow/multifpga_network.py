from __future__ import annotations

import yaml
from abc import ABC, abstractmethod
from enum import Enum
from onnx import NodeProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from typing import Callable

from finn.transformation.fpgadataflow.multifpga import (
    get_device_id,
    get_first_submodel_node,
    get_last_submodel_node,
)
from finn.util.basic import make_build_dir

CommunicationKernelName = str
Device = int
NodeName = str


class DataDirection(str, Enum):
    TX = "TX"
    RX = "RX"
    BIDIRECTIONAL = "BIDIRECTIONAL"


class NetworkMetadata(ABC):
    def __init__(self) -> None:
        self.table = {}

    @abstractmethod
    def add_connection(
        self,
        sender_device: Device,
        sender_node: NodeName,
        receiver_device: Device,
        receiver_node: NodeName,
    ) -> None:
        pass

    def save(self, p: Path) -> None:
        with p.open("w+") as f:
            yaml.dump(self.table, f, yaml.Dumper)

    def load(self, p: Path) -> None:
        with p.open("r") as f:
            self.table = yaml.load(f, yaml.Loader)


class AuroraNetworkMetadata(NetworkMetadata):
    AuroraTableType = dict[
        Device,
        dict[
            CommunicationKernelName,
            dict[str | DataDirection, None | Device | tuple[NodeName, NodeName]],
        ],
    ]

    # TODO: Remove default value
    def __init__(self, ports_per_device: int = 2) -> None:
        super().__init__()
        self.ports_per_device = ports_per_device
        self.table: AuroraNetworkMetadata.AuroraTableType = {}

    def _add_single_connection(
        self,
        on_device: Device,
        on_node: NodeName,
        other_device: Device,
        other_node: NodeName,
        direction: DataDirection,
    ) -> None:
        found_free_spot = False
        for aurora_table in self.table[on_device].values():
            if aurora_table["partner"] == other_device and aurora_table[direction] is None:
                if direction == DataDirection.TX:
                    aurora_table[direction] = (on_node, other_node)
                elif direction == DataDirection.RX:
                    aurora_table[direction] = (other_node, on_node)
                found_free_spot = True

        if not found_free_spot:
            # TODO: Replace with exceptions when we have the infrastructure
            assert (
                len(self.table[on_device]) < self.ports_per_device
            ), f"Too many kernels / ports required for this device ({on_device})!"
            new_aurora = f"aurora_flow_{len(self.table[on_device])}_dev{on_device}"
            self.table[on_device][new_aurora] = {
                "partner": other_device,
                DataDirection.TX: None,
                DataDirection.RX: None,
            }
            if direction == DataDirection.TX:
                self.table[on_device][new_aurora][direction] = (on_node, other_node)
            elif direction == DataDirection.RX:
                self.table[on_device][new_aurora][direction] = (other_node, on_node)

    def add_connection(
        self,
        sender_device: Device,
        sender_node: NodeName,
        receiver_device: Device,
        receiver_node: NodeName,
    ) -> None:
        """Add a connection between sender_device and receiver_device. This creates both the
        TX and RX endpoints.

        If we look at the node name tuple (n0, n1):
            If this is in the TX field: n0 is on our current device and sends to n1
            If this is in the RX field: n1 is on our current device and receives from n0"""
        if sender_device not in self.table:
            self.table[sender_device] = {}
        if receiver_device not in self.table:
            self.table[receiver_device] = {}
        self._add_single_connection(
            sender_device, sender_node, receiver_device, receiver_node, DataDirection.TX
        )
        self._add_single_connection(
            receiver_device, receiver_node, sender_device, sender_node, DataDirection.RX
        )

    def connections_with(self, d1: Device, d2: Device) -> int:
        """Return how many connections there are between the two devices"""
        if d1 not in self.table.keys():
            return 0
        s = 0
        for aurora in self.table[d1].values():
            if aurora["partner"] == d2:
                s += 1
        return s

    def has_connection(
        self, d1: Device, n1: NodeName, d2: Device, n2: NodeName, direction: DataDirection
    ) -> bool:
        """Return whether this metadata contains a connection between given devices and nodes"""
        if d1 not in self.table.keys():
            return False
        for aurora in self.table[d1].values():
            if (
                aurora["partner"] == d2
                and aurora[direction] is not None
                and aurora[direction] == (n1, n2)
            ):
                return True
        return False

    def get_kernel_names_for(
        self, sdp1: NodeProto, sdp2: NodeProto
    ) -> tuple[CommunicationKernelName, CommunicationKernelName] | None:
        """Check if a connection between the SDP nodes exists, and if so returns the names
        of the matching aurora kernels"""
        lnode = get_last_submodel_node(sdp1).name
        fnode = get_first_submodel_node(sdp2).name
        d1 = get_device_id(sdp1)
        d2 = get_device_id(sdp2)
        assert d1 is not None
        assert d2 is not None
        if not self.has_connection(
            d1, lnode, d2, fnode, DataDirection.TX
        ) or not self.has_connection(d2, lnode, d1, fnode, DataDirection.RX):
            return None

        t1 = None
        t2 = None
        for device in self.table.keys():
            if device == d1:
                for aurora_name, aurora in self.table[device].items():
                    if aurora["partner"] == d2 and aurora[DataDirection.TX] == (lnode, fnode):
                        t1 = aurora_name

            if device == d2:
                for aurora_name, aurora in self.table[device].items():
                    if aurora["partner"] == d1 and aurora[DataDirection.RX] == (lnode, fnode):
                        t2 = aurora_name
        assert t1 is not None
        assert t2 is not None
        return (t1, t2)


class CreateNetworkMetadata(ABC):
    """Run this transformation to create metadata necessary for Multi-FPGA settings. Pass the type
    of metadata as a type to the constructor, or a factory producing one."""

    def __init__(
        self, save_as_format: type[NetworkMetadata] | Callable[[], NetworkMetadata]
    ) -> None:
        super().__init__()
        self.metadata_type = save_as_format
        self.metadata = self.metadata_type()

    def save_metadata(self, model: ModelWrapper) -> Path:
        metadata_dir = Path(make_build_dir("network_metadata")).absolute()
        metadata_path = metadata_dir / "metadata.yaml"
        self.metadata.save(metadata_path)
        model.set_metadata_prop("network_metadata", str(metadata_path))
        return metadata_path

    @abstractmethod
    def create_metadata(self, model: ModelWrapper) -> None:
        """Create the metadata and assign it to the object variable.
        When creating a new type of network metadata this has to be implemented"""
        raise NotImplementedError()


class CreateChainNetworkMetadata(CreateNetworkMetadata):
    """Create a simple network with FPGAs connected in a simple line"""

    def __init__(self, save_as_format: type[NetworkMetadata]) -> None:
        super().__init__(save_as_format)

    def create_metadata(self, model: ModelWrapper) -> None:
        # Build graph
        for i, n1 in enumerate(model.graph.node):
            if i == len(model.graph.node) - 1:
                break
            n2 = model.graph.node[i + 1]
            d1 = get_device_id(n1)
            d2 = get_device_id(n2)
            assert d1 is not None
            assert d2 is not None
            if d1 != d2:
                self.metadata.add_connection(
                    int(d1),
                    get_last_submodel_node(n1).name,
                    int(d2),
                    get_first_submodel_node(n2).name,
                )


class CreateReturnChainNetworkMetadata(CreateNetworkMetadata):
    pass


class AssignNetworkMetadata(Transformation):
    def __init__(
        self,
        metadata_type: type[NetworkMetadata] | Callable[[], NetworkMetadata],
        creator_type: type[CreateNetworkMetadata],
    ) -> None:
        super().__init__()
        self.creator = creator_type(save_as_format=metadata_type)

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        self.creator.create_metadata(model)
        self.creator.save_metadata(model)
        return model, False

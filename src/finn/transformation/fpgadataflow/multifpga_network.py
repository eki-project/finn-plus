from __future__ import annotations

import yaml
from enum import Enum
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


class NetworkMetadata:
    def __init__(self) -> None:
        self.table = {}

    def add_connection(
        self, sender_device: int, sender_node: str, receiver_device: int, receiver_node: str
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
        on_device: int,
        on_node: str,
        other_device: int,
        other_node: str,
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
            new_aurora = f"aurora_flow_{len(self.table[on_device])}"
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
        self, sender_device: int, sender_node: str, receiver_device: int, receiver_node: str
    ) -> None:
        """Add a connection between sender_device and receiver_device. This creates both the
        TX and RX endpoints."""
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


class CreateNetworkMetadata(Transformation):
    """Run this transformation to create metadata necessary for Multi-FPGA settings. Pass the type
    of metadata as a type to the constructor, or a factory producing one."""

    def __init__(
        self, save_as_format: type[NetworkMetadata] | Callable[[], NetworkMetadata]
    ) -> None:
        super().__init__()
        self.metadata_type = save_as_format
        self.metadata = self.metadata_type()


class CreateChainNetworkMetadata(CreateNetworkMetadata):
    """Create a simple network with FPGAs connected in a simple line"""

    def __init__(self, save_as_format: type[NetworkMetadata]) -> None:
        super().__init__(save_as_format)

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
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

        # Save results for usage in later steps
        metadata_dir = Path(make_build_dir("network_metadata")).absolute()
        metadata_path = metadata_dir / "metadata.yaml"
        self.metadata.save(metadata_path)
        model.set_metadata_prop("network_metadata", str(metadata_path))

        return model, False


class CreateReturnChainNetworkMetadata(CreateNetworkMetadata):
    pass

from __future__ import annotations

import yaml
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path, PosixPath, PurePath, WindowsPath
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from typing import Any, Callable

from finn.transformation.fpgadataflow.multifpga_utils import get_device_id
from finn.util.basic import make_build_dir
from finn.util.exception import FINNMultiFPGAConfigError, FINNMultiFPGAError
from finn.util.logging import log

CommunicationKernelName = str
Device = int
NodeName = str


class DataDirection(str, Enum):
    TX = "TX"
    RX = "RX"
    BIDIRECTIONAL = "BIDIRECTIONAL"


class NetworkMetadata(ABC):
    def __init__(self, load_from: Path | ModelWrapper | None = None) -> None:
        self.table = {}
        if load_from is not None:
            if type(load_from) is ModelWrapper:
                p = load_from.get_metadata_prop("network_metadata")
                assert p is not None
                p = Path(p)
                assert p.exists()
                self.load(p)
            elif issubclass(type(load_from), PurePath):
                # Additional assert required because PyLance cannot detect the issubclass entry
                # condition and wants an extra assert that load_from is NOT a modelwrapper
                assert isinstance(load_from, (PurePath, PosixPath, WindowsPath))
                assert load_from.exists(), (
                    f"Could not load NetworkMetadata from {load_from}, "
                    f"since no such file exists!"
                )
                self.load(load_from)
            else:
                raise FINNMultiFPGAError(
                    f"Could not load NetworkMetadata from unknown type: {type(load_from)}"
                )

    @abstractmethod
    def __getitem__(self, key: Any) -> Any:
        pass

    @abstractmethod
    def __setitem__(self, key: Any, value: Any) -> Any:
        pass

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
    """Defines an Aurora Network. Structure is:
    >>> table = {1: {"aurora0": {"partner": 2, DataDirection.TX: ("sdp1", "sdp2"), DataDirection.RX: ("sdp1", "sdp3")}}}

    In this table, device 1 sends to device 2 from sdp1 to sdp2. It also receives from device 2:
    sdp1 receives data from sdp3. The first tuple element is ON the device, while the second is on the PARTNER device.

    add_connection does both endpoints on both devices:
    >>> am = AuroraNetworkMetadata()
    >>> am.add_connection(0, "sdp0", 1, "sdp1")
    >>> am[0]
    {'aurora_flow_0_dev0': {'partner': 1, <DataDirection.TX: 'TX'>: ('sdp0', 'sdp1'), <DataDirection.RX: 'RX'>: None}}
    >>> am[1]
    {'aurora_flow_0_dev1': {'partner': 0, <DataDirection.TX: 'TX'>: None, <DataDirection.RX: 'RX'>: ('sdp1', 'sdp0')}}

    You can access the data easily. For example to get the receiving kernel on device 1:
    >>> am[1, "aurora_flow_0_dev1", DataDirection.RX, 0]
    'sdp1'

    You can also set data this way if necessary:
    >>> am[1, "aurora_flow_0_dev1", DataDirection.RX] = ("sdp4", "sdp0")
    >>> am[1, "aurora_flow_0_dev1", DataDirection.RX]
    ('sdp4', 'sdp0')
    """  # noqa

    # TODO: Remove default value
    def __init__(
        self, load_from: Path | ModelWrapper | None = None, ports_per_device: int = 2
    ) -> None:
        super().__init__(load_from)
        self.ports_per_device = ports_per_device

    def __getitem__(self, key: tuple) -> Any | None:
        if type(key) is int:
            return self.table[key]
        elif type(key) is tuple:  # noqa
            data = self.table
            for k in key:
                data = data[k]
            return data
        return None

    def __setitem__(self, key: Any, value: Any) -> None:
        if type(key) is int:
            self.table[key] = value
        elif type(key) is tuple:
            data = self.table
            for k in key[:-1]:
                data = data[k]
            data[key[-1]] = value

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
                aurora_table[direction] = (on_node, other_device)
                found_free_spot = True

        if not found_free_spot:
            current_ports_used = len(self.table[on_device])
            if current_ports_used > self.ports_per_device:
                raise FINNMultiFPGAConfigError(
                    f"Could not add a connection between devices "
                    f"{on_device} and {other_device}, because "
                    f"{on_device} already uses all of it's "
                    f"{self.ports_per_device} communication ports. "
                    f"Cannot map this model to this setup."
                )
            new_aurora = f"aurora_flow_{len(self.table[on_device])}_dev{on_device}"
            self.table[on_device][new_aurora] = {
                "partner": other_device,
                DataDirection.TX: None,
                DataDirection.RX: None,
            }
            self.table[on_device][new_aurora][direction] = (on_node, other_node)

    def add_connection(
        self,
        sender_device: Device,
        sender_node: NodeName,
        receiver_device: Device,
        receiver_node: NodeName,
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

    def get_aurora_kernels(self, device: Device) -> list[CommunicationKernelName]:
        """Return all aurora kernel names for this device. Good for packaging.
        >>> am = AuroraNetworkMetadata()
        >>> am.add_connection(0, "sdp0", 1, "sdp1")
        >>> am.add_connection(0, "sdp2", 2, "sdp3")
        >>> am.get_aurora_kernels(0)
        ['aurora_flow_0_dev0', 'aurora_flow_1_dev0']"""
        if device not in self.table:
            return []
        return list(self.table[device].keys())

    def get_connections(self, d1: Device, d2: Device) -> int:
        """Return the number of connections between d1 and d2.
        >>> am = AuroraNetworkMetadata()
        >>> am.add_connection(0, "sdp0", 1, "sdp1")
        >>> am.add_connection(1, "sdp1", 2, "sdp2")
        >>> am.get_connections(0, 1)
        1
        >>> am.get_connections(1, 0)
        1
        >>> am.get_connections(1, 2)
        1"""
        if d1 not in self.table or d2 not in self.table:
            return 0
        return len(list(filter(lambda aurora: aurora[1]["partner"] == d2, self.table[d1].items())))

    def get_devices(self) -> list[Device]:
        """Return all devices used in this network metadata"""
        return list(self.table.keys())

    def sends_to_aurora(self, sdp_name: str, device: int) -> list[str]:
        """Return the names of aurora kernels that this SDP kernel will output data to
        >>> am = AuroraNetworkMetadata()
        >>> am.add_connection(0, "sdp0", 1, "sdp1")
        >>> am.add_connection(1, "sdp1", 2, "sdp2")
        >>> am.sends_to_aurora("sdp1", 1)
        ['aurora_flow_1_dev1']
        """
        kernels = []
        if device not in self.table.keys():
            raise FINNMultiFPGAError(f"There is no such device ({device}) in the metadata table!")
        for aurora_kernel, connection in self.table[device].items():
            if connection[DataDirection.TX] is None:
                continue
            if connection[DataDirection.TX][0] == sdp_name:
                kernels.append(aurora_kernel)
        return kernels

    def receives_from_aurora(self, sdp_name: str, device: int) -> list[str]:
        """Return the names of aurora kernels that this SDP kernel will receive data from
        >>> am = AuroraNetworkMetadata()
        >>> am.add_connection(0, "sdp0", 1, "sdp1")
        >>> am.add_connection(1, "sdp1", 2, "sdp2")
        >>> am.receives_from_aurora("sdp1", 1)
        ['aurora_flow_0_dev1']
        """
        kernels = []
        if device not in self.table.keys():
            raise FINNMultiFPGAError(f"There is no such device ({device}) in the metadata table!")
        for aurora_kernel, connection in self.table[device].items():
            if connection[DataDirection.RX] is None:
                continue
            if connection[DataDirection.RX][0] == sdp_name:
                kernels.append(aurora_kernel)
        return kernels

    def get_open_duplex_connections(
        self, direction: DataDirection, on_device: int | None = None
    ) -> list[str]:
        """Get a list of all Aurora kernels, that, in the given direction, have an open port.
        (Unused duplex port). If on_device is None, its checked for all devices
        >>> am = AuroraNetworkMetadata()
        >>> am.add_connection(0, "sdp0", 1, "sdp1")
        >>> am.add_connection(1, "sdp1", 2, "sdp2")
        >>> am.get_open_duplex_connections(DataDirection.RX)
        ['aurora_flow_0_dev0', 'aurora_flow_1_dev1']
        >>> am.get_open_duplex_connections(DataDirection.TX)
        ['aurora_flow_0_dev1', 'aurora_flow_0_dev2']
        >>> am.get_open_duplex_connections(DataDirection.TX, on_device=2)
        ['aurora_flow_0_dev2']
        """
        kernels = []
        if on_device is not None and on_device not in self.table.keys():
            raise FINNMultiFPGAError(
                f"Tried checking for open duplex connections on device "
                f"{on_device}, which is not to be found in the metadata "
                "table!"
            )
        for device in self.table.keys():
            if on_device is not None and on_device != device:
                continue
            for aurora_kernel, connection in self.table[device].items():
                if connection[direction] is None:
                    kernels.append(aurora_kernel)
        return kernels


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
                    n1.name,
                    int(d2),
                    n2.name,
                )
                log.debug(
                    f"Created connection in metadata between {n1.name} "
                    f"(device {d1}) and {n2.name} (device {d2})"
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

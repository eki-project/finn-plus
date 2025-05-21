from __future__ import annotations

import pytest

from pathlib import Path
from test_multifpga_sdp_creation import create_sdp_ready_model
from typing import Any

from finn.transformation.fpgadataflow.multifpga_create_sdp import (
    CreateMultiFPGAStreamingDataflowPartition,
)
from finn.transformation.fpgadataflow.multifpga_kernel_preparation import PrepareAuroraFlow
from finn.transformation.fpgadataflow.multifpga_network import (
    AssignNetworkMetadata,
    AuroraNetworkMetadata,
    CreateChainNetworkMetadata,
)


@pytest.mark.multifpga
@pytest.mark.slow
@pytest.mark.parametrize("device_node_combinations", [(1, 2), (1, 3), (2, 2), (5, 10), (5, 11)])
@pytest.mark.parametrize("assignment_type", ["random", "equal"])
@pytest.mark.parametrize("metadata_creator", [CreateChainNetworkMetadata])
@pytest.mark.parametrize("shuffle_devices", [True, False])
def test_aurora_packaging_integrated(
    device_node_combinations: tuple[int, int],
    assignment_type: str,
    metadata_creator: type,
    shuffle_devices: bool,
) -> None:
    devices, nodes = device_node_combinations
    assignment_order = ""
    if metadata_creator is CreateChainNetworkMetadata:
        assignment_order = "linear"

    model = create_sdp_ready_model(
        nodes, devices, assignment_type, assignment_order, shuffle_devices
    )
    prepare_aurora = PrepareAuroraFlow()
    model = model.transform(CreateMultiFPGAStreamingDataflowPartition())
    model = model.transform(AssignNetworkMetadata(AuroraNetworkMetadata, metadata_creator))
    model = model.transform(prepare_aurora)

    meta = AuroraNetworkMetadata(model)

    # Check the paths from the nodes
    aurora_storage = model.get_metadata_prop("aurora_storage")
    assert aurora_storage is not None
    aurora_storage = Path(aurora_storage)
    assert aurora_storage.exists()

    # Here we only check if the kernels all got packaged, nothing else
    for device in meta.get_devices():
        for kernel in meta.get_aurora_kernels(device):
            assert (aurora_storage / (kernel + "xo")).exists()


@pytest.fixture
def create_aurora_metadata(request: Any) -> AuroraNetworkMetadata | None:
    match request.param:
        case "empty":
            return AuroraNetworkMetadata()
        case "chain":
            m_chain = AuroraNetworkMetadata()
            m_chain.add_connection(0, "sdp0", 1, "sdp1")
            m_chain.add_connection(1, "sdp1", 2, "sdp2")
            m_chain.add_connection(2, "sdp2", 3, "sdp3")
            return m_chain
        case "returnchain":
            m_returnchain = AuroraNetworkMetadata()
            m_returnchain.add_connection(0, "sdp0", 1, "sdp1")
            m_returnchain.add_connection(1, "sdp1", 2, "sdp2")
            m_returnchain.add_connection(2, "sdp2", 3, "sdp3")
            m_returnchain.add_connection(3, "sdp3", 2, "sdp4")
            m_returnchain.add_connection(2, "sdp4", 1, "sdp5")
            m_returnchain.add_connection(1, "sdp5", 0, "sdp6")
            return m_returnchain
        case _:
            return None


@pytest.mark.multifpga
@pytest.mark.slow
@pytest.mark.parametrize("create_aurora_metadata", ["empty", "chain", "returnchain"], indirect=True)
def test_aurora_packaging_isolated(create_aurora_metadata: AuroraNetworkMetadata | None) -> None:
    if create_aurora_metadata is None:
        raise AssertionError("Invalid testconfig: Unknown metadata type")
    prep = PrepareAuroraFlow()
    prep.package_all_from_metadata(create_aurora_metadata)
    for device in create_aurora_metadata.table.keys():
        for aurora_name in create_aurora_metadata.table[device].keys():
            assert (prep.aurora_storage / (aurora_name + ".xo")).exists()

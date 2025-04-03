import pytest

import yaml
from pathlib import Path
from test_multifpga_sdp_creation import create_sdp_ready_model

from finn.transformation.fpgadataflow.multifpga import CreateMultiFPGAStreamingDataflowPartition
from finn.transformation.fpgadataflow.multifpga_network import (
    AssignNetworkMetadata,
    AuroraNetworkMetadata,
    CreateChainNetworkMetadata,
    CreateNetworkMetadata,
    DataDirection,
    NetworkMetadata,
    get_device_id,
    get_first_submodel_node,
    get_last_submodel_node,
)


@pytest.mark.multifpga
@pytest.mark.parametrize(
    "device_node_combinations", [(2, 2), (5, 10), (40, 100), (50, 50), (1, 10)]
)
@pytest.mark.parametrize("assignment_type", ["random", "equal"])
@pytest.mark.parametrize("communication_metadata_type", [AuroraNetworkMetadata])
@pytest.mark.parametrize("communication_type", [CreateChainNetworkMetadata])
@pytest.mark.parametrize("shuffle_devices", [True, False])
@pytest.mark.parametrize("assignment_order", ["linear"])
def test_chain_metadata(
    device_node_combinations: tuple[int, int],
    assignment_type: str,
    assignment_order: str,
    communication_metadata_type: type[NetworkMetadata],
    communication_type: type[CreateNetworkMetadata],
    shuffle_devices: bool,
) -> None:
    device_count, node_count = device_node_combinations
    # TODO: Not ideal, better pass an own, self constructed model
    model = create_sdp_ready_model(
        node_count, device_count, assignment_type, assignment_order, shuffle_devices
    )
    model = model.transform(CreateMultiFPGAStreamingDataflowPartition())

    # Create the metadata
    model = model.transform(AssignNetworkMetadata(communication_metadata_type, communication_type))

    # Check the data exists
    raw_path = model.get_metadata_prop("network_metadata")
    assert raw_path is not None
    metadata_path = Path(raw_path)
    assert metadata_path.exists()

    # Check that its readable
    data = None
    with metadata_path.open("r") as f:
        data = yaml.load(f, yaml.Loader)
    assert data is not None

    # Check that the assignments worked as expected
    m = AuroraNetworkMetadata()
    m.table = data
    for i, n1 in enumerate(model.graph.node):
        if i == len(model.graph.node) - 1:
            break
        n2 = model.graph.node[i + 1]
        d1 = get_device_id(n1)
        d2 = get_device_id(n2)
        assert d1 is not None
        assert d2 is not None
        if d1 != d2:
            # Specific for line connections
            assert m.connections_with(d1, d2) == 1  # d1 sends
            assert m.connections_with(d2, d1) == 1  # d2 receives
        assert m.has_connection(
            d1,
            n1.name,
            d2,
            n2.name,
            DataDirection.TX,
        )
        assert m.has_connection(
            d2,
            n1.name,
            d1,
            n2.name,
            DataDirection.RX,
        )
        if i > 0:
            assert m.get_kernel_names_for(n1, n2) == (
                f"aurora_flow_1_dev{d1}",
                f"aurora_flow_0_dev{d2}",
            )
        else:
            assert m.get_kernel_names_for(n1, n2) == (
                f"aurora_flow_0_dev{d1}",
                f"aurora_flow_0_dev{d2}",
            )


@pytest.mark.multifpga
def test_return_chain_metadata() -> None:
    pass

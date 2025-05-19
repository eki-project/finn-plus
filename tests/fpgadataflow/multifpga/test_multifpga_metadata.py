import pytest

import yaml
from pathlib import Path
from test_multifpga_sdp_creation import create_sdp_ready_model

from finn.transformation.fpgadataflow.multifpga_create_sdp import (
    CreateMultiFPGAStreamingDataflowPartition,
)
from finn.transformation.fpgadataflow.multifpga_network import (
    AssignNetworkMetadata,
    AuroraNetworkMetadata,
    CreateChainNetworkMetadata,
    CreateNetworkMetadata,
    DataDirection,
    NetworkMetadata,
    get_device_id,
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
def test_aurora_chain_metadata(
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
    model = model.transform(AssignNetworkMetadata(communication_metadata_type, communication_type))

    # Check that the assignments worked as expected
    metadata_path = model.get_metadata_prop("network_metadata")
    assert metadata_path is not None
    metadata_path = Path(metadata_path)
    m1 = AuroraNetworkMetadata(load_from=model)
    m2 = AuroraNetworkMetadata(load_from=metadata_path)
    for m in [m1, m2]:
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
                assert m.get_connections(d1, d2) == 1
                assert m.get_connections(d2, d1) == 1
                # TODO: Add case for the last node
                if i > 0:
                    assert m[d1, f"aurora_flow_1_dev{d1}", DataDirection.TX] == (n1.name, n2.name)
                else:
                    assert m[d1, f"aurora_flow_0_dev{d1}", DataDirection.TX] == (n1.name, n2.name)
                assert m[d2, f"aurora_flow_0_dev{d2}", DataDirection.RX] == (n2.name, n1.name)

import pytest

from pathlib import Path
from test_multifpga_sdp_creation import create_sdp_ready_model

from finn.builder.build_dataflow_config import MultiFPGACommunicationScheme
from finn.transformation.fpgadataflow.multifpga import CreateMultiFPGAStreamingDataflowPartition
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
@pytest.mark.parametrize("communication_scheme", [MultiFPGACommunicationScheme.AURORA_CHAIN])
@pytest.mark.parametrize("shuffle_devices", [True, False])
def test_aurora_packaging_integrated(
    device_node_combinations: tuple[int, int],
    assignment_type: str,
    communication_scheme: MultiFPGACommunicationScheme,
    shuffle_devices: bool,
) -> None:
    devices, nodes = device_node_combinations
    assignment_order = None
    if communication_scheme == MultiFPGACommunicationScheme.AURORA_CHAIN:
        assignment_order = "linear"
        cr_type = CreateChainNetworkMetadata
    else:
        raise NotImplementedError()

    model = create_sdp_ready_model(
        nodes, devices, assignment_type, assignment_order, shuffle_devices
    )
    prepare_aurora = PrepareAuroraFlow()
    model = model.transform(CreateMultiFPGAStreamingDataflowPartition())
    model = model.transform(AssignNetworkMetadata(AuroraNetworkMetadata, cr_type))
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


m_empty = AuroraNetworkMetadata()
m_chain = AuroraNetworkMetadata()
m_returnchain = AuroraNetworkMetadata()

m_chain.add_connection(0, "sdp0", 1, "sdp1")
m_chain.add_connection(1, "sdp1", 2, "sdp2")
m_chain.add_connection(2, "sdp2", 3, "sdp3")

m_returnchain.add_connection(0, "sdp0", 1, "sdp1")
m_returnchain.add_connection(1, "sdp1", 2, "sdp2")
m_returnchain.add_connection(2, "sdp2", 3, "sdp3")
m_returnchain.add_connection(3, "sdp3", 2, "sdp4")
m_returnchain.add_connection(2, "sdp4", 1, "sdp5")
m_returnchain.add_connection(1, "sdp5", 0, "sdp6")


@pytest.mark.multifpga
@pytest.mark.slow
@pytest.mark.parametrize("metadata", [m_empty, m_chain, m_returnchain])
def test_aurora_packaging_isolated(metadata: AuroraNetworkMetadata) -> None:
    prep = PrepareAuroraFlow()
    prep.package_all_from_metadata(metadata)
    for device in metadata.table.keys():
        for aurora_name in metadata.table[device].keys():
            assert (prep.aurora_storage / (aurora_name + ".xo")).exists()

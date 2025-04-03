import pytest

from pathlib import Path
from qonnx.custom_op.registry import getCustomOp
from test_multifpga_sdp_creation import create_sdp_ready_model

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    MultiFPGACommunicationScheme,
    PartitioningConfiguration,
)
from finn.builder.build_dataflow_steps import (
    step_create_multifpga_sdp,
    step_prepare_network_infrastructure,
)
from finn.transformation.fpgadataflow.multifpga import (
    CreateMultiFPGAStreamingDataflowPartition,
    get_device_id,
)
from finn.transformation.fpgadataflow.multifpga_kernel_preparation import PrepareAuroraFlow
from finn.transformation.fpgadataflow.multifpga_network import (
    AssignNetworkMetadata,
    AuroraNetworkMetadata,
    CreateChainNetworkMetadata,
)
from finn.util.basic import make_build_dir


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
    for sdp in model.graph.node:
        assert sdp.op_type == "StreamingDataflowPartition"
        assert get_device_id(sdp) is not None

    model = model.transform(AssignNetworkMetadata(AuroraNetworkMetadata, cr_type))
    model = model.transform(prepare_aurora)

    metadata_path = model.get_metadata_prop("network_metadata")
    assert metadata_path is not None
    metadata_path = Path(metadata_path)
    assert metadata_path.exists()
    meta = AuroraNetworkMetadata(metadata_path)

    # Check the paths from the nodes
    aurora_storage = model.get_metadata_prop("aurora_storage")
    assert aurora_storage is not None
    aurora_storage = Path(aurora_storage)
    assert aurora_storage.exists()

    for sdp in model.graph.node:
        connections = meta.get_connections_of_node(sdp)
        found_connections = getCustomOp(sdp).get_nodeattr("network_connections")
        for kernel_name, _, _, direction in connections:
            xo_path = aurora_storage / (kernel_name + ".xo")
            assert f"{kernel_name}:{direction.value}:{xo_path}" in found_connections
            assert xo_path.exists()


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

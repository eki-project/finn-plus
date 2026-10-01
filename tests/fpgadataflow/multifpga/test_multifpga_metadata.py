import pytest

import onnx.helper as oh
from copy import deepcopy
from onnx import TensorProto
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model
from typing import cast

from finn.builder.build_dataflow_config import (
    MFCommunicationKernel,
    MFTopology,
    MFVerbosity,
    MultiFPGAConfiguration,
)
from finn.transformation.fpgadataflow.multifpga.aurora.metadata import AuroraNetworkMetadata
from finn.transformation.fpgadataflow.multifpga.backend import get_backend
from finn.transformation.fpgadataflow.multifpga.create_network_metadata import CreateNetworkMetadata
from finn.transformation.fpgadataflow.multifpga.metadata import DataDirection
from finn.util.basic import get_metadata_prop_path, make_build_dir
from finn.util.exception import FINNError, FINNMultiFPGAConfigError
from finn.util.fpgadataflow import get_device_id, set_device_id
from tests.fpgadataflow.test_set_folding import make_multi_fclayer_model


def make_mfcfg(
    communication_kernel: MFCommunicationKernel, ports_per_device: int = 2
) -> MultiFPGAConfiguration:
    """Return a Multi-FPGA configuration for metadata creation, without logging output."""
    return MultiFPGAConfiguration(
        communication_kernel=communication_kernel,
        ports_per_device=ports_per_device,
        verbosity=MFVerbosity.NONE,
    )


def sdp_model(partition_topology: MFTopology) -> ModelWrapper:  # noqa
    """Return an SDP only graph for testing of metadata functions. SDP don't contain any
    actual submodels. Node attribute `device_id` is set afterwards.

    Arguments:
    ---------
        `partition_topology`: In which topology the `device_id`s should be assigned to the nodes.
            This simulates the partitioning transformation of the Multi-FGPA flow.

    Returns:
    -------
        `ModelWrapper`
    """

    def device_id(i: int, nodes: int) -> int:
        if partition_topology == MFTopology.CHAIN:
            return i
        elif partition_topology == MFTopology.RETURNCHAIN:  # noqa
            return max(i, abs(nodes - i))
        raise NotImplementedError()

    node_count = 200
    assert node_count > 10

    # Make the main graph
    tensors = [
        oh.make_tensor_value_info(f"tensor_{i}", TensorProto.FLOAT, [1, 3])
        for i in range(node_count)
    ]
    nodes = [
        oh.make_node(
            op_type="StreamingDataflowPartition",
            domain="finn.custom_op.fpgadataflow",
            inputs=[f"tensor_{i}"],
            outputs=[f"tensor_{i+1}"],
            name=f"SDP_{i}",
            device_id=device_id(i, node_count) if i % 2 == 0 else device_id(i - 1, node_count),
        )
        for i in range(node_count - 1)
    ]
    graph = oh.make_graph(nodes=nodes, name="graph", inputs=[tensors[0]], outputs=[tensors[-1]])
    return ModelWrapper(qonnx_make_model(graph))


@pytest.mark.multifpga
@pytest.mark.auroraflow
@pytest.mark.parametrize("nodes", [1, 10, 100])
@pytest.mark.parametrize("communication_kernel", [MFCommunicationKernel.AURORA])
def test_metadata_sdp_only(nodes: int, communication_kernel: MFCommunicationKernel) -> None:
    """Test that the network metadata assignment only accepts SDP-only graphs."""
    model: ModelWrapper = make_multi_fclayer_model(
        3, DataType["BINARY"], DataType["BINARY"], DataType["BINARY"], nodes
    )
    with pytest.raises(FINNError):
        _ = model.transform(CreateNetworkMetadata(make_mfcfg(communication_kernel)))


@pytest.mark.multifpga
@pytest.mark.auroraflow
@pytest.mark.parametrize("communication_kernel", [MFCommunicationKernel.AURORA])
@pytest.mark.parametrize("topology", [MFTopology.CHAIN, MFTopology.RETURNCHAIN])
def test_metadata(
    communication_kernel: MFCommunicationKernel,
    topology: MFTopology,
) -> None:
    """Test that metadata for a model is created correctly."""
    model = sdp_model(topology)
    model = model.transform(CreateNetworkMetadata(make_mfcfg(communication_kernel)))
    path = get_metadata_prop_path(model, "network_metadata", must_exist=True)

    # Metadata file exists
    assert path.exists()

    # Load from model
    meta = get_backend(communication_kernel).metadata_type.load_from_model(model)

    # Check per connection
    for node in model.graph.node:
        device_this = get_device_id(node)
        successors = model.find_direct_successors(node)
        if successors is None:
            continue
        for successor in successors:
            device_other = get_device_id(successor)
            if device_this != device_other:
                assert meta.node_is_sender(node.name)
                assert meta.node_is_receiver(successor.name)
                assert meta.get_partner_node(node.name, DataDirection.TX) == successor.name
                assert meta.get_partner_node(successor.name, DataDirection.RX) == node.name

    # Aurora specific tests
    if communication_kernel == MFCommunicationKernel.AURORA:
        meta = cast("AuroraNetworkMetadata", meta)
        devices = {get_device_id(node) for node in model.graph.node}

        # Are all devices in the metadata?
        assert len(devices) == len(meta.data.keys())
        for device in devices:
            assert device in meta.data

        # Count connections between devices both in graph and in metadata
        combinations_metadata = {}
        for device in meta.data.keys():
            for kernel_data in meta.data[device]:
                key = None
                if kernel_data.connecting_kernels[DataDirection.TX] is not None:
                    key = (device, kernel_data.partner_device)
                    if key not in combinations_metadata:
                        combinations_metadata[key] = 0
                    combinations_metadata[key] += 1
                if kernel_data.connecting_kernels[DataDirection.RX] is not None:
                    key = (kernel_data.partner_device, device)
                    if key not in combinations_metadata:
                        combinations_metadata[key] = 0
                    combinations_metadata[key] += 1
        combinations_graph = {}
        for node in model.graph.node:
            device_this = get_device_id(node)
            successors = model.find_direct_successors(node)
            if successors is None:
                continue
            for successor in successors:
                device_other = get_device_id(successor)
                if device_this != device_other:
                    key = (device_this, device_other)
                    if key not in combinations_graph:
                        combinations_graph[key] = 0
                    combinations_graph[key] += 1
        assert len(list(combinations_metadata.keys())) == len(list(combinations_graph.keys()))
        assert set(combinations_graph.keys()) == set(combinations_metadata.keys())
        for key in combinations_graph.keys():
            # This connection must be twice in the metadata,
            # since it records it both from the TX and RX side.
            assert combinations_graph[key] * 2 == combinations_metadata[key], (
                f"Connection {key} found {combinations_graph[key]} time(s) in the "
                f"graph and {combinations_metadata[key]} time(s) in the metadata. "
            )
    else:
        raise NotImplementedError(
            f"No checks implemented for communication kernel {communication_kernel.name}"
        )


@pytest.mark.multifpga
@pytest.mark.auroraflow
@pytest.mark.parametrize("communication_kernel", [MFCommunicationKernel.AURORA])
@pytest.mark.parametrize("ports_per_device", [2, 3, 4])
def test_metadata_ports_per_device(
    communication_kernel: MFCommunicationKernel, ports_per_device: int
) -> None:
    """Test that the configured number of ports per device is respected by the metadata."""
    # Six SDPs alternating between device 0 and 1 cross devices five times. Both devices take
    # part in all five crossings, which needs three kernels (two directions each) per device.
    node_count = 6
    tensors = [
        oh.make_tensor_value_info(f"t_{i}", TensorProto.FLOAT, [1, 1])
        for i in range(node_count + 1)
    ]
    nodes = [
        oh.make_node(
            "StreamingDataflowPartition",
            [tensors[i].name],
            [tensors[i + 1].name],
            name=f"n{i}",
            domain="finn.custom_op.fpgadataflow",
            device_id=i % 2,
        )
        for i in range(node_count)
    ]
    model = ModelWrapper(
        qonnx_make_model(
            oh.make_graph(nodes, inputs=[tensors[0]], outputs=[tensors[-1]], name="graph")
        )
    )
    transform = CreateNetworkMetadata(make_mfcfg(communication_kernel, ports_per_device))
    match communication_kernel:
        case MFCommunicationKernel.AURORA:
            # Adding a connection that needs a new kernel on a device whose ports are all in use
            # raises in AuroraNetworkMetadata._add_single_connection
            if ports_per_device < 3:
                with pytest.raises(FINNMultiFPGAConfigError, match="communication ports"):
                    _ = model.transform(transform)
                return
            model = model.transform(transform)
            meta = AuroraNetworkMetadata.load_from_model(model)
            assert meta.ports_per_device == ports_per_device
            assert len(meta.data[0]) == 3
            assert len(meta.data[1]) == 3
        case _:
            raise NotImplementedError(
                f"No checks implemented for communication kernel {communication_kernel.name}"
            )


@pytest.mark.multifpga
@pytest.mark.auroraflow
def test_aurora_metadata_path_serialization() -> None:
    """Test that the load path is not stored in the file, but set again when loading."""
    build_dir = Path(make_build_dir("test_aurora_metadata_path_"))
    path = build_dir / "metadata.yaml"
    meta = AuroraNetworkMetadata(ports_per_device=4)
    meta.add_connection(0, "n0", 1, "n1")
    meta.save(path)
    assert "loaded_from_path" not in path.read_text()

    # Loading sets the path, so saving without a path writes back to the same file
    loaded = AuroraNetworkMetadata.load(path)
    assert loaded.loaded_from_path == path
    assert loaded.ports_per_device == 4
    assert loaded.data == meta.data
    loaded.add_connection(1, "n1", 2, "n2")
    loaded.save()
    assert AuroraNetworkMetadata.load(path).data == loaded.data

    # A moved file is loaded from (and saved to) its new location
    moved = build_dir / "moved.yaml"
    path.rename(moved)
    assert AuroraNetworkMetadata.load(moved).loaded_from_path == moved


@pytest.mark.multifpga
@pytest.mark.auroraflow
@pytest.mark.parametrize("communication_kernel", [MFCommunicationKernel.AURORA])
def test_metadata_small(communication_kernel: MFCommunicationKernel) -> None:
    """Test metadata creation on hand-crafted small models."""
    # Create the base model
    tensors = [oh.make_tensor_value_info(f"t_{i}", TensorProto.FLOAT, [1, 1]) for i in range(4)]
    nodes = [
        oh.make_node(
            "StreamingDataflowPartition",
            [tensors[i].name],
            [tensors[i + 1].name],
            name=f"n{i}",
            domain="finn.custom_op.fpgadataflow",
        )
        for i in range(3)
    ]
    model = ModelWrapper(
        qonnx_make_model(
            oh.make_graph(nodes, inputs=[tensors[0]], outputs=[tensors[-1]], name="graph")
        )
    )

    # Chain
    chain_model = deepcopy(model)
    c_nodes = chain_model.graph.node
    set_device_id(c_nodes[0], 0)
    set_device_id(c_nodes[1], 1)
    set_device_id(c_nodes[2], 2)
    metadata_type = get_backend(communication_kernel).metadata_type
    chain_model = chain_model.transform(CreateNetworkMetadata(make_mfcfg(communication_kernel)))
    meta = metadata_type.load_from_model(chain_model)
    assert meta.node_is_sender(c_nodes[0].name)
    assert meta.node_is_sender(c_nodes[1].name)
    assert not meta.node_is_sender(c_nodes[2].name)
    assert not meta.node_is_receiver(c_nodes[0].name)
    assert meta.node_is_receiver(c_nodes[1].name)
    assert meta.node_is_receiver(c_nodes[2].name)
    assert meta.get_partner_node(c_nodes[0].name, DataDirection.RX) is None
    assert meta.get_partner_node(c_nodes[-1].name, DataDirection.TX) is None

    # Returnchain (This also checks that a kernel with an open connection in
    # the other direction is utilized instead of making a new kernel.)
    rchain_model = deepcopy(model)
    rc_nodes = rchain_model.graph.node
    set_device_id(rc_nodes[0], 0)
    set_device_id(rc_nodes[1], 1)
    set_device_id(rc_nodes[2], 0)
    rchain_model = rchain_model.transform(CreateNetworkMetadata(make_mfcfg(communication_kernel)))
    meta = metadata_type.load_from_model(rchain_model)
    assert meta.node_is_sender(rc_nodes[0].name)
    assert meta.node_is_sender(rc_nodes[1].name)
    assert not meta.node_is_sender(rc_nodes[2].name)
    assert not meta.node_is_receiver(rc_nodes[0].name)
    assert meta.node_is_receiver(rc_nodes[1].name)
    assert meta.node_is_receiver(rc_nodes[2].name)
    assert meta.get_partner_node(rc_nodes[0].name, DataDirection.TX) == rc_nodes[1].name
    assert meta.get_partner_node(rc_nodes[1].name, DataDirection.TX) == rc_nodes[2].name
    assert meta.get_partner_node(rc_nodes[1].name, DataDirection.RX) == rc_nodes[0].name
    assert meta.get_partner_node(rc_nodes[2].name, DataDirection.RX) == rc_nodes[1].name
    if communication_kernel == MFCommunicationKernel.AURORA:
        meta = cast("AuroraNetworkMetadata", meta)
        assert len(meta.data[0]) == 1
        assert len(meta.data[1]) == 1
        assert meta.data[0][0].connecting_kernels[DataDirection.TX] == (
            rc_nodes[0].name,
            rc_nodes[1].name,
        ), str(meta.data[1])
        assert meta.data[0][0].connecting_kernels[DataDirection.RX] == (
            rc_nodes[2].name,
            rc_nodes[1].name,
        ), str(meta.data[0])
    else:
        raise NotImplementedError(
            f"No checks implemented for communication kernel {communication_kernel.name}"
        )

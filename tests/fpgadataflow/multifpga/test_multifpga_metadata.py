import pytest

import json
import networkx as nx
import onnx.helper as oh
import shutil
from copy import deepcopy
from onnx import TensorProto
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model
from typing import cast

from finn.builder.build_dataflow_config import (
    MFNetworkBackendType,
    MFTopology,
    MFVerbosity,
    MultiFPGAConfiguration,
)
from finn.transformation.fpgadataflow.multifpga.aurora.metadata import (
    AuroraLinkEndpoint,
    AuroraNetworkMetadata,
)
from finn.transformation.fpgadataflow.multifpga.backend import get_backend
from finn.transformation.fpgadataflow.multifpga.create_network_metadata import CreateNetworkMetadata
from finn.transformation.fpgadataflow.multifpga.metadata import DataDirection
from finn.util.basic import get_metadata_prop_path, make_build_dir
from finn.util.exception import FINNError, FINNInternalError, FINNMultiFPGAConfigError
from finn.util.fpgadataflow import get_device_id, set_device_id
from tests.fpgadataflow.multifpga.utils import chain_model, make_chain_partitioning_cfg
from tests.fpgadataflow.test_set_folding import make_multi_fclayer_model


def make_mfcfg(
    network_backend: MFNetworkBackendType, links_per_device: int = 2
) -> MultiFPGAConfiguration:
    """Return a Multi-FPGA configuration for metadata creation, without logging output."""
    return MultiFPGAConfiguration(
        network_backend=network_backend,
        links_per_device=links_per_device,
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
        if partition_topology == MFTopology.SIMPLEX_CHAIN:
            return i
        elif partition_topology == MFTopology.DUPLEX_CHAIN:  # noqa
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
@pytest.mark.parametrize("network_backend", [MFNetworkBackendType.AURORAFLOW])
def test_metadata_sdp_only(nodes: int, network_backend: MFNetworkBackendType) -> None:
    """Test that the network metadata assignment only accepts SDP-only graphs."""
    model: ModelWrapper = make_multi_fclayer_model(
        3, DataType["BINARY"], DataType["BINARY"], DataType["BINARY"], nodes
    )
    with pytest.raises(FINNError):
        _ = model.transform(CreateNetworkMetadata(make_mfcfg(network_backend)))


@pytest.mark.multifpga
@pytest.mark.auroraflow
@pytest.mark.parametrize("network_backend", [MFNetworkBackendType.AURORAFLOW])
@pytest.mark.parametrize("topology", [MFTopology.SIMPLEX_CHAIN, MFTopology.DUPLEX_CHAIN])
def test_metadata(
    network_backend: MFNetworkBackendType,
    topology: MFTopology,
) -> None:
    """Test that metadata for a model is created correctly."""
    model = sdp_model(topology)
    model = model.transform(CreateNetworkMetadata(make_mfcfg(network_backend)))
    path = get_metadata_prop_path(model, "network_metadata", must_exist=True)

    # Metadata file exists
    assert path.exists()

    # Load from model
    meta = get_backend(network_backend).metadata_type.load_from_model(model)

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
    if network_backend == MFNetworkBackendType.AURORAFLOW:
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
            f"No checks implemented for communication kernel {network_backend.name}"
        )


@pytest.mark.multifpga
@pytest.mark.auroraflow
@pytest.mark.parametrize("network_backend", [MFNetworkBackendType.AURORAFLOW])
@pytest.mark.parametrize("links_per_device", [2, 3, 4])
def test_metadata_links_per_device(
    network_backend: MFNetworkBackendType, links_per_device: int
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
    transform = CreateNetworkMetadata(make_mfcfg(network_backend, links_per_device))
    match network_backend:
        case MFNetworkBackendType.AURORAFLOW:
            # Adding a connection that needs a new kernel on a device whose links are all in use
            # raises in AuroraNetworkMetadata._add_single_connection
            if links_per_device < 3:
                with pytest.raises(FINNMultiFPGAConfigError, match="communication links"):
                    _ = model.transform(transform)
                return
            model = model.transform(transform)
            meta = AuroraNetworkMetadata.load_from_model(model)
            assert meta.links_per_device == links_per_device
            assert len(meta.data[0]) == 3
            assert len(meta.data[1]) == 3
        case _:
            raise NotImplementedError(
                f"No checks implemented for communication kernel {network_backend.name}"
            )


@pytest.mark.multifpga
@pytest.mark.auroraflow
def test_aurora_metadata_path_serialization() -> None:
    """Test that the load path is not stored in the file, but set again when loading."""
    build_dir = Path(make_build_dir("test_aurora_metadata_path_"))
    path = build_dir / "metadata.yaml"
    meta = AuroraNetworkMetadata(links_per_device=4)
    meta.add_connection(0, "n0", 1, "n1")
    meta.save(path)
    assert "loaded_from_path" not in path.read_text()

    # Loading sets the path, so saving without a path writes back to the same file
    loaded = AuroraNetworkMetadata.load(path)
    assert loaded.loaded_from_path == path
    assert loaded.links_per_device == 4
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
@pytest.mark.parametrize(
    "connections",
    [
        [(0, "n0", 1, "n1"), (1, "n1", 2, "n2")],
        [(0, "n0", 1, "n1"), (1, "n1", 2, "n2"), (2, "n2", 1, "n3"), (1, "n3", 0, "n4")],
    ],
    ids=["simplex_chain", "duplex_chain"],
)
def test_aurora_metadata_partner_links(connections: list[tuple[int, str, int, str]]) -> None:
    """Test that both endpoints of a cable point to each other's link. In both chains the
    middle device connects to device 0 on link 0 and to device 2 on link 1, so the cabling
    is not symmetric.
    """
    meta = AuroraNetworkMetadata(links_per_device=2)
    for connection in connections:
        meta.add_connection(*connection)
    cabling = {
        (device, endpoint.link_index): (endpoint.partner_device, endpoint.partner_link_index)
        for device, endpoints in meta.data.items()
        for endpoint in endpoints
    }
    assert cabling == {(0, 0): (1, 0), (1, 0): (0, 0), (1, 1): (2, 0), (2, 0): (1, 1)}

    # The endpoints are found by their content
    for sender_device, sender_node, receiver_device, receiver_node in connections:
        tx = meta.get_endpoint(
            sender_device, sender_node, receiver_device, receiver_node, DataDirection.TX
        )
        rx = meta.get_endpoint(
            receiver_device, receiver_node, sender_device, sender_node, DataDirection.RX
        )
        assert tx.partner_link_index == rx.link_index
        assert rx.partner_link_index == tx.link_index


@pytest.mark.multifpga
@pytest.mark.auroraflow
def test_aurora_metadata_get_endpoint_missing() -> None:
    """Test that looking up a connection that does not exist raises."""
    meta = AuroraNetworkMetadata(links_per_device=2)
    meta.add_connection(0, "n0", 1, "n1")
    # Wrong direction: n0 only sends to n1
    with pytest.raises(FINNInternalError, match="found 0"):
        meta.get_endpoint(0, "n0", 1, "n1", DataDirection.RX)


@pytest.mark.multifpga
@pytest.mark.auroraflow
def test_aurora_metadata_partner_link_mismatch() -> None:
    """Test that a connection is rejected if it reuses an endpoint whose partner link does not
    match the link used on the other device.
    """
    meta = AuroraNetworkMetadata(links_per_device=2)
    meta.add_connection(0, "n0", 1, "n1")
    # Simulate devices that are out of sync: the reverse connection reuses link 0 on both
    # devices, but the endpoint on device 1 claims to be cabled to link 1 of device 0
    meta.data[1][0].partner_link_index = 1
    with pytest.raises(FINNInternalError, match="already cabled to link 1 of device 0"):
        meta.add_connection(1, "n1", 0, "n2")


@pytest.mark.multifpga
@pytest.mark.auroraflow
@pytest.mark.parametrize(
    ("topology", "device_ids"),
    [
        (MFTopology.SIMPLEX_CHAIN, [0, 1, 1, 2]),
        (MFTopology.DUPLEX_CHAIN, [0, 1, 2, 2, 1, 0]),
        (MFTopology.RING, [0, 1, 2, 0]),
        (MFTopology.ARBITRARY, [0, 1, 2]),
    ],
)
@pytest.mark.parametrize("with_model", [True, False])
def test_aurora_metadata_plot(
    topology: MFTopology, device_ids: list[int], with_model: bool
) -> None:
    """Test that the network diagram is written for the topologies with a layout, and skipped
    for the others.
    """
    model = chain_model(len(device_ids))
    for node, device in zip(model.graph.node, device_ids, strict=True):
        set_device_id(node, device)
    cfg = make_chain_partitioning_cfg(max(device_ids) + 1, topology=topology)
    model = model.transform(
        CreateNetworkMetadata(cast("MultiFPGAConfiguration", cfg.multifpga_configuration))
    )
    meta = AuroraNetworkMetadata.load_from_model(model)
    assert meta.topology == topology

    path = Path(make_build_dir("test_aurora_metadata_plot_")) / "network_cabling.png"
    meta.plot(path, model if with_model else None, cfg)
    match topology:
        case MFTopology.SIMPLEX_CHAIN | MFTopology.DUPLEX_CHAIN | MFTopology.RING:
            assert path.stat().st_size > 0
        case MFTopology.ARBITRARY:
            assert not path.exists()
        case _:
            raise NotImplementedError(f"No checks implemented for topology {topology.name}")


@pytest.mark.multifpga
@pytest.mark.auroraflow
@pytest.mark.parametrize("with_model", [True, False])
def test_aurora_metadata_plot_dot(with_model: bool) -> None:
    """Test that the DOT graph contains one edge per used cable direction, and, with the model,
    the data flow inside the FPGAs and to and from the host. It is rendered only if Graphviz is
    available.
    """
    # Duplex chain: n0 (0) -> n1 (1) -> n2 (2) -> n3 (2) -> n4 (1) -> n5 (0)
    model = chain_model(6)
    for node, device in zip(model.graph.node, [0, 1, 2, 2, 1, 0], strict=True):
        set_device_id(node, device)
    cfg = make_chain_partitioning_cfg(3, topology=MFTopology.DUPLEX_CHAIN)
    model = model.transform(
        CreateNetworkMetadata(cast("MultiFPGAConfiguration", cfg.multifpga_configuration))
    )
    meta = AuroraNetworkMetadata.load_from_model(model)

    path = Path(make_build_dir("test_aurora_metadata_plot_dot_")) / "network_graph.dot"
    meta.plot_dot(path, model if with_model else None)
    dot = path.read_text(encoding="utf-8")
    assert dot.startswith("digraph")
    # Four cable directions: 0 -> 1 -> 2 and back
    assert dot.count("penwidth=2") == 4
    for device in range(3):
        assert f"subgraph cluster_fpga{device}" in dot
    if with_model:
        assert 'host_in -> "n0"' in dot
        assert '"n5" -> host_out' in dot
        # Only the SDPs on the same FPGA are connected directly
        assert '"n2" -> "n3"' in dot
        assert '"n0" -> "n1"' not in dot
    else:
        assert "host_in" not in dot
    assert path.with_suffix(".svg").exists() == (shutil.which("dot") is not None)


@pytest.mark.multifpga
@pytest.mark.auroraflow
def test_aurora_metadata_export() -> None:
    """Test that the exported node-link JSON contains exactly one undirected edge without
    attributes per cable between the (device, link) ports, regardless of whether the cable is
    used in one or both directions. The ports must follow the link indices, not the order of
    the endpoints in the metadata.
    """

    def endpoint(
        link: int,
        partner: int,
        partner_link: int,
        tx: tuple[str, str] | None,
        rx: tuple[str, str] | None,
    ) -> AuroraLinkEndpoint:
        return AuroraLinkEndpoint(
            aurora_xo=None,
            partner_device=partner,
            link_index=link,
            partner_link_index=partner_link,
            connecting_kernels={DataDirection.TX: tx, DataDirection.RX: rx},
        )

    # 0 -> 1 -> 2 -> 1 -> 0 -> 1: the cable between 0 and 1 carries data in both directions, the
    # cable between 1 and 2 only from 1 to 2. The link indices do not match the positions of the
    # endpoints in the lists.
    meta = AuroraNetworkMetadata(
        data={
            0: [endpoint(1, 1, 1, tx=("n0", "n1"), rx=("n5", "n4"))],
            1: [
                endpoint(0, 2, 1, tx=("n1", "n2"), rx=None),
                endpoint(1, 0, 1, tx=("n4", "n5"), rx=("n1", "n0")),
            ],
            2: [endpoint(1, 1, 0, tx=None, rx=("n2", "n1"))],
        }
    )

    path = Path(make_build_dir("test_aurora_metadata_export_")) / "cabling.json"
    meta.export(path)
    g = nx.node_link_graph(json.loads(path.read_text()))
    assert not g.is_directed()
    assert not g.is_multigraph()
    # The edges are undirected, so compare the ports of each cable regardless of their order
    assert sorted(tuple(sorted((u, v))) for u, v in g.edges) == [
        ((0, 1), (1, 1)),
        ((1, 0), (2, 1)),
    ]
    assert all(not d for _, _, d in g.edges(data=True))


@pytest.mark.multifpga
@pytest.mark.auroraflow
@pytest.mark.parametrize("network_backend", [MFNetworkBackendType.AURORAFLOW])
def test_metadata_small(network_backend: MFNetworkBackendType) -> None:
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
    metadata_type = get_backend(network_backend).metadata_type
    chain_model = chain_model.transform(CreateNetworkMetadata(make_mfcfg(network_backend)))
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
    rchain_model = rchain_model.transform(CreateNetworkMetadata(make_mfcfg(network_backend)))
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
    if network_backend == MFNetworkBackendType.AURORAFLOW:
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
            f"No checks implemented for communication kernel {network_backend.name}"
        )

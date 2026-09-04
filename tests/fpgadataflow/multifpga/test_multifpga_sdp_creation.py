"""Test that Multi-FPGA SDPs are created correctly based on a given partitioning."""
import pytest

import networkx as nx
import onnx.helper as oh
import os
import random
from copy import deepcopy
from onnx import TensorProto
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model
from random import randint
from typing import cast

from finn.builder.build_dataflow_config import MFVerbosity
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.multifpga.create_multi_sdp import (
    ClusterByNodeattribute,
    CreateMultiFPGAStreamingDataflowPartition,
    get_device_id,
)
from finn.transformation.qonnx.give_unique_node_names_recursive import GiveUniqueNodeNamesRecursive
from finn.util.basic import make_build_dir
from finn.util.fpgadataflow import get_submodel, set_device_id
from tests.fpgadataflow.test_set_folding import make_multi_fclayer_model


def make_testing_model_from_nx(g: nx.DiGraph) -> ModelWrapper:
    """Create a testing node only model from a DiGraph. Sets all data attributes
    of the DiGraph as node attributes on the ONNX node.
    """
    # Prepare all tensors / edges
    tensors = {
        (i, j): oh.make_tensor_value_info(f"t_{i}{j}", TensorProto.FLOAT, [1]) for i, j in g.edges
    }

    # Create all nodes (without Inputs/Outputs for now)
    nodes = {
        i: oh.make_node(
            "TestingNode",
            [],
            [],
            backend="fpgadataflow",
            domain="finn.custom_op.fpgadataflow",
            **data,
        )
        for i, data in g.nodes(data=True)
    }

    # Connect all nodes
    for (i, j), tensor in tensors.items():
        nodes[i].output.append(tensor.name)
        nodes[j].input.append(tensor.name)

    # Create inputs and outputs for the graph
    for i in g.nodes:
        if g.in_degree(i) == 0:
            tensors[(None, i)] = oh.make_tensor_value_info(f"in_{i}", TensorProto.FLOAT, [1])
            nodes[i].input.append(tensors[(None, i)].name)
        if g.out_degree(i) == 0:
            tensors[(i, None)] = oh.make_tensor_value_info(f"out_{i}", TensorProto.FLOAT, [1])
            nodes[i].output.append(tensors[(i, None)].name)

    # Build graph and model
    graph = oh.make_graph(
        nodes=list(nodes.values()),
        name="graph",
        inputs=[tensor for key, tensor in tensors.items() if key[0] is None],
        outputs=[tensor for key, tensor in tensors.items() if key[1] is None],
        value_info=[
            tensor for key, tensor in tensors.items() if key[0] is not None and key[1] is not None
        ],
    )
    model = qonnx_make_model(graph)
    return ModelWrapper(model)


# Nodes contain as first element their ID, as second element their device ID
# Edges contain a source and target node
# Expected partitions contain sets of nodes that should have the same partition ID
test_graphs = {
    # Basic test
    "simple": {"nodes": [(0, 0), (1, 1)], "edges": [(0, 1)], "expected_partitions": [{0}, {1}]},
    # This graph tests that the partitions are correctly created, if the devices
    # switch while in a branch
    #                   / 5(1) - 6(1) - 8(2) \
    # 0(0) - 1(0) - 2(0)                      9(2) - 10(2)
    #                   \ 3(0) - 4(0) - 7(2) /
    "branches1": {
        "nodes": [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 1),
            (6, 1),
            (7, 2),
            (8, 2),
            (9, 2),
            (10, 2),
        ],
        "edges": [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 7),
            (2, 5),
            (5, 6),
            (6, 8),
            (7, 9),
            (8, 9),
            (9, 10),
        ],
        "expected_partitions": [{0, 1, 2, 3, 4}, {5, 6}, {7, 8, 9, 10}],
    },
    # This graph tests that switches between devices are mapped to separate SDPs
    # It also tests multi IO
    # 0(0) \    / 3(3) - 4(3) - 5(3) -------\      / 11(7)
    #       2(2)                              10(6)
    # 1(1) /    \ 6(4) - 7(5) - 8(4) - 9(5) /      \ 12(8)
    "branches2": {
        "nodes": [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 3),
            (5, 3),
            (6, 4),
            (7, 5),
            (8, 4),
            (9, 5),
            (10, 6),
            (11, 7),
            (12, 8),
        ],
        "edges": [
            (0, 2),
            (1, 2),
            (2, 3),
            (2, 6),
            (3, 4),
            (4, 5),
            (5, 10),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (10, 12),
        ],
        "expected_partitions": [{0}, {1}, {2}, {3, 4, 5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}],
    },
    # This graph tests multiple branches on their own devices, with a long skip connection back
    # to the initial device
    #      / - 1(1) - 10(1) - 11(1) - 12(1) \
    #     / - 2(2) - 20(2) - 21(2) --------- \
    # 0(0) - 3(3) - 30(3) ------------------- 4(0)
    #    \ --------------------------------- /
    "branches3": {
        "nodes": [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (10, 1),
            (11, 1),
            (12, 1),
            (20, 2),
            (21, 2),
            (30, 3),
            (4, 0),
        ],
        "edges": [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 10),
            (10, 11),
            (11, 12),
            (12, 4),
            (2, 20),
            (20, 21),
            (21, 4),
            (3, 30),
            (30, 4),
            (0, 4),
        ],
        "expected_partitions": [{0}, {4}, {1, 10, 11, 12}, {2, 20, 21}, {3, 30}],
    },
    # This graph tests multi IO and a single device. Everything should be grouped together
    "single_device_multi_io": {
        "nodes": [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)],
        "edges": [(0, 4), (1, 4), (2, 4), (3, 4), (4, 5), (4, 6), (4, 7)],
        "expected_partitions": [{0, 1, 2, 3, 4, 5, 6, 7}],
    },
}


def digraph_from_graph_definition(
    graph: dict[str, list[tuple[int, int]] | list[set[int]]]
) -> nx.DiGraph:
    """Convert a graph definition as used in this test into a DiGraph."""
    g = nx.DiGraph()
    for node, device in graph["nodes"]:
        g.add_node(node, device_id=device, partition_id=0, original_index=node)
    for source, target in graph["edges"]:
        g.add_edge(source, target)
    return g


@pytest.mark.multifpga
@pytest.mark.parametrize(
    "graph",
    [pytest.param(graph_data, id=graph_name) for graph_name, graph_data in test_graphs.items()],
)
def test_sdp_creation(
    graph: dict[str, list[tuple[int, int]] | list[set[int]]], request: pytest.FixtureRequest
) -> None:
    """Test SDP creation on all the given graphs."""
    name = request.node.callspec.id
    # Create graph and assign device ID to nodes. Also assigns "original_index", which stores the
    # nx graph node that the eventual ONNX node will be made from. This is used to later check that
    # the partitions are as expected. Since the ordering of nodes in the ONNX might not
    # yield the same indexes as the original node names/ids, we store them as a node attribute.
    # This way we can in the modelwrapper still identify which node "6" originally was.
    g: nx.DiGraph = digraph_from_graph_definition(graph)
    model = make_testing_model_from_nx(g)

    # Test clustering and SDP creation separately. First, clustering:
    pmodel = model.transform(
        ClusterByNodeattribute(resolve_circular_dependencies=True, compare_attribute="device_id")
    )
    groups = {}
    for node in pmodel.graph.node:
        pid = getCustomOp(node).get_nodeattr("partition_id")
        original_index = getCustomOp(node).get_nodeattr("original_index")
        if pid not in groups.keys():
            groups[pid] = []
        groups[pid].append(original_index)
    for group in groups.values():
        assert set(group) in graph["expected_partitions"], f"Groups: {groups}"
    assert len(groups) == len(graph["expected_partitions"])

    # Test SDP creation now (on "model", instead of "pmodel")
    model = model.transform(
        CreateMultiFPGAStreamingDataflowPartition(
            separate_iodmas=True,
            dataflow_partition_directory=Path(make_build_dir(f"test_sdp_creation_{name}_")),
            verbosity=MFVerbosity.NONE,
        )
    )
    assert len(model.graph.node) == len(graph["expected_partitions"])
    for i in range(len(model.graph.node) - 1):
        assert get_device_id(model.graph.node[i]) != get_device_id(model.graph.node[i + 1])
    for sdp in model.graph.node:
        for node in get_submodel(sdp)[0].graph.node:
            assert get_device_id(node) == get_device_id(sdp)


def sdp_overview(model: ModelWrapper) -> str:
    """Give an overview of a modelwrapper consisting of SDP nodes. Used for error messages."""
    s = "Nodes (not necessarily in order!)\n"
    for node in model.graph.node:
        s += f"{node.name}\n"
        if "StreamingDataflowPartition" not in node.op_type:
            continue
        submodel, _ = get_submodel(node)
        for snode in submodel.graph.node:
            s += f"\t{snode.name}\n"
    return s


@pytest.mark.multifpga
@pytest.mark.parametrize(
    "graph",
    [pytest.param(graph_data, id=graph_name) for graph_name, graph_data in test_graphs.items()],
)
def test_iodma_separation(
    graph: dict[str, list[tuple[int, int]] | list[set[int]]], request: pytest.FixtureRequest
) -> None:
    """Test that IODMAs receive their own partition ID if requested."""
    name = request.node.callspec.id
    g = digraph_from_graph_definition(graph)
    model = make_testing_model_from_nx(g)
    model = model.transform(InsertIODMA())
    model = model.transform(GiveUniqueNodeNamesRecursive())
    model = model.transform(
        CreateMultiFPGAStreamingDataflowPartition(
            separate_iodmas=True,
            dataflow_partition_directory=Path(make_build_dir(f"test_sdp_iodma_separation_{name}_")),
            verbosity=MFVerbosity.EXTRA_HIGH,
        )
    )

    # Count input and output nodes, each should now have an IODMA
    io_nodes = sum([1 for n in g.nodes if g.in_degree(n) == 0 or g.out_degree(n) == 0])
    expected_cluster_partitions = len(graph["expected_partitions"])
    total_nodes = len(model.graph.node)
    assert total_nodes == expected_cluster_partitions + io_nodes, (
        f"Expected {expected_cluster_partitions} (expected device ID based clusters)"
        f" + {io_nodes} (separate partitions for single IODMA nodes), but got "
        f"{total_nodes} StreamingDataflowPartitions instead of the expected "
        f"{expected_cluster_partitions + io_nodes}.\n" + sdp_overview(model)
    )

    # If there is an SDP with only one node, and that node is an IODMA,
    # consequently this node was the only one with this specific
    # partition id
    for node in model.graph.node:
        pre = model.find_direct_predecessors(node)
        suc = model.find_direct_successors(node)
        submodel, _ = get_submodel(node)
        if pre is None or suc is None:
            assert len(submodel.graph.node) == 1
            assert "IODMA" in submodel.graph.node[0].op_type


@pytest.mark.multifpga
@pytest.mark.parametrize(
    "graph",
    [pytest.param(graph_data, id=graph_name) for graph_name, graph_data in test_graphs.items()],
)
def test_iodma_same_device(
    graph: dict[str, list[tuple[int, int]] | list[set[int]]], request: pytest.FixtureRequest
) -> None:
    """Test that inserted IODMAs have the same device id as the node
    that they are the successor/predecessor of.

    This test relies on the fact that some of the example graphs dont
    start/end on device 0 (which is the default value for device_id)."""
    name = request.node.callspec.id
    g = digraph_from_graph_definition(graph)
    model = make_testing_model_from_nx(g)
    model = model.transform(InsertIODMA())
    model = model.transform(GiveUniqueNodeNamesRecursive())
    for node in model.graph.node:
        if "IODMA" not in node.op_type:
            continue

        pre = model.find_direct_predecessors(node)
        pre = pre if pre is not None else []
        suc = model.find_direct_successors(node)
        suc = suc if suc is not None else []
        all_adjacent = pre + suc
        iodma_device = get_device_id(node)

        # This assumes that IODMAs only have one output or input, otherwise it
        # is not clear which device the IODMA should be on.
        for adjacent_node in all_adjacent:
            adjacent_device = get_device_id(adjacent_node)
            assert adjacent_device == iodma_device, (
                f"Graph {name}: Node {adjacent_node.name} is adjacent to IODMA {node.name}, "
                f"but their device IDs are different: {adjacent_device} and {iodma_device}."
            )


def equal_device_assignment(devices: int, nodes: int) -> list[int]:
    """Take a number of devices and nodes, and return a list of ints. The 0th
    index holds the number of nodes that device 0 should contain, etc.

    If the number of nodes cannot be divided by the number of devices, the
    leftover nodes are assigned to a random device.
    """
    leftover = nodes % devices
    nodes_used = nodes - leftover
    temp = []
    for _ in range(devices):
        temp.append(int(nodes_used / devices))
    temp[randint(0, len(temp) - 1)] += leftover
    return temp


def random_device_assignment(devices: int, nodes: int) -> list[int]:
    """Take a number of devices and nodes, and return a list of ints. The 0th
    index holds the number of nodes that device 0 should contain, etc.

    Randomly assign nodes to devices until no nodes are left. After being assigned
    a number of nodes, the device is removed from the pool of candidates.
    """
    assert nodes >= devices
    nodes_left = nodes
    temp = [0] * devices
    not_set = list(range(devices))  # Contains the indices of the devices
    for _ in range(devices):
        chosen_index = random.choice(not_set)
        not_set.remove(chosen_index)  # Assign to each device only once
        if len(not_set) == 0:
            temp[chosen_index] = nodes_left
        else:
            temp[chosen_index] = randint(1, nodes_left - len(not_set))
        nodes_left -= temp[chosen_index]
    # Temp is <devices> long and every bucket contains the nr of nodes
    # that the device has
    return temp


def create_sdp_ready_model_no_branches(
    node_count: int,
    device_count: int,
    assignment_type: str,
    shuffle_devices: bool = False,
) -> ModelWrapper:
    """Create a simple SDP ready model without branches. The device_id is
    set according to the passed assignment arguments.

    Args:
        node_count: Number of nodes. Nodes will be numbered 0-node_count.
        device_count: Number of devices to map the nodes to.
        assignment_type: How to assign which nodes belong to devices. Can be 'random'
            to assign a random number of nodes to a device, or 'equal' to distribute nodes
            equally across all devices.
        shuffle_devices: If True, the devices are not assigned from 0 to N but in random order.
    """
    # Create a simple chained model without branches
    model = make_multi_fclayer_model(
        3, DataType["BINARY"], DataType["BINARY"], DataType["BINARY"], node_count
    )
    for i, node in enumerate(model.graph.node):
        node.name = f"node_{i}"

    # Create assignment numbers
    if assignment_type == "random":
        assignment = random_device_assignment(device_count, node_count)
    elif assignment_type == "equal":
        assignment = equal_device_assignment(device_count, node_count)
    else:
        raise AssertionError()

    # Distribute the numbers
    assert sum(assignment) == len(
        model.graph.node
    ), f"Assignment length doesnt match model node count. Assignment: {assignment}"

    # Assign node-devices linearly
    overall_node_index = 0
    device_list = list(range(len(assignment)))
    if shuffle_devices:
        random.shuffle(device_list)

    for current_device in device_list:
        while assignment[current_device] > 0:
            set_device_id(model.graph.node[overall_node_index], current_device)
            overall_node_index += 1
            assignment[current_device] -= 1
    model = model.transform(GiveUniqueNodeNames())
    return model


@pytest.mark.multifpga
@pytest.mark.parametrize("devices", [5, 10, 1, 10, 2])
@pytest.mark.parametrize("nodes", [20, 50, 10, 10, 2])
def test_random_device_assign_util(devices: int, nodes: int) -> None:
    """Test that the random device assignment function does not accept more devices than
    nodes, and that the assignment itself works.
    """
    if devices > nodes:
        with pytest.raises(AssertionError):
            random_device_assignment(devices, nodes)
    else:
        assignment = random_device_assignment(devices, nodes)
        assert sum(assignment) == nodes
        assert all(x > 0 for x in assignment)
        assert len(assignment) == devices


@pytest.mark.multifpga
@pytest.mark.parametrize(
    "device_node_combinations", [(2, 2), (10, 20), (100, 200), (1, 2), (2, 13)]
)
@pytest.mark.parametrize("assignment_type", ["random", "equal"])
@pytest.mark.parametrize("shuffle_devices", [True, False])
def test_multi_sdp_creation_linear(
    device_node_combinations: tuple[int, int],
    assignment_type: str,
    shuffle_devices: bool,
) -> None:
    """Test that creating SDPs based on their device id works as expected."""
    device_count, node_count = device_node_combinations
    model = create_sdp_ready_model_no_branches(
        node_count, device_count, assignment_type, shuffle_devices
    )
    devices_counted = len({get_device_id(node) for node in model.graph.node})

    # Creation of the SDPs
    original_model = deepcopy(model)
    model = model.transform(
        CreateMultiFPGAStreamingDataflowPartition(
            separate_iodmas=True,
            dataflow_partition_directory=Path(make_build_dir("test_multi_sdp")),
            verbosity=MFVerbosity.NONE,
        )
    )
    sdp_test_dir = make_build_dir("test_sdp_creation")
    model.save(os.path.join(sdp_test_dir, "sdp_model.onnx"))  # noqa

    # Check that the number of SDPs is atleast as large as
    # the number of devices
    sdp_counted = len(
        [node for node in model.graph.node if node.op_type == "StreamingDataflowPartition"]
    )
    assert sdp_counted >= devices_counted

    # Check that all nodes in the parent graph are now SDPs
    for node in model.graph.node:
        assert node.op_type == "StreamingDataflowPartition"

    # Check that the order of nodes along the graph is kept
    partitioned_order = []
    order = [n.name for n in original_model.graph.node]
    for node in model.graph.node:
        submodel_path = getCustomOp(node).get_nodeattr("model")
        assert submodel_path is not None
        submodel = ModelWrapper(cast("str", submodel_path))
        for snode in submodel.graph.node:
            partitioned_order.append(snode.name)
    assert partitioned_order == order

    # Parent model shouldnt have any branches
    for node in model.graph.node:
        sucs = model.find_direct_successors(node)
        assert (sucs is None) or len(sucs) == 1

    # Check that all submodels' nodes have the same device ID
    for node in model.graph.node:
        submodel_path = getCustomOp(node).get_nodeattr("model")
        assert submodel_path is not None
        submodel = ModelWrapper(cast("str", submodel_path))
        devices_found = [get_device_id(n) for n in submodel.graph.node]
        assert len(set(devices_found)) == 1

    # Check that no two SDPs are on the same device after another
    for node in model.graph.node:
        node_a_device = get_device_id(node)
        sucs = model.find_direct_successors(node)
        if sucs is None:
            continue
        assert len(sucs) == 1, "Currently (!) SDPs can only have one successor."
        node_b_device = get_device_id(sucs[0])
        assert (
            node_a_device != node_b_device
        ), f"Consecutive SDPs with the same device: {[get_device_id(x) for x in model.graph.node]}"

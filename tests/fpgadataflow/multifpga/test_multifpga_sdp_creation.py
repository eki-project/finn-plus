import pytest

import os
import random
from copy import deepcopy
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from random import randint

from finn.transformation.fpgadataflow.multifpga import (
    CreateMultiFPGAStreamingDataflowPartition,
    get_device_id,
)
from finn.util.basic import make_build_dir
from tests.fpgadataflow.test_set_folding import make_multi_fclayer_model


# TODO: Tests for this util function
def equal_device_assignment(devices: int, nodes: int) -> list[int]:
    leftover = nodes % devices
    nodes_used = nodes - leftover
    temp = []
    for _ in range(devices):
        temp.append(int(nodes_used / devices))
    temp[randint(0, len(temp) - 1)] += leftover
    return temp


def random_device_assignment(devices: int, nodes: int) -> list[int]:
    """Randomly assign number of devices to nodes"""
    assert nodes >= devices
    nodes_left = nodes
    temp = [0] * devices
    not_set = list(range(devices))
    for _ in range(devices):
        chosen_index = random.choice(not_set)
        not_set.remove(chosen_index)
        if len(not_set) == 0:
            temp[chosen_index] = nodes_left
        else:
            temp[chosen_index] = randint(1, nodes_left - len(not_set))
        nodes_left -= temp[chosen_index]
    return temp


@pytest.mark.multifpga
@pytest.mark.parametrize("devices", [5, 10, 1, 10, 2])
@pytest.mark.parametrize("nodes", [20, 50, 10, 10, 2])
def test_random_device_assign_util(devices: int, nodes: int) -> None:
    if devices > nodes:
        with pytest.raises(AssertionError):
            random_device_assignment(devices, nodes)
    else:
        assignment = random_device_assignment(devices, nodes)
        assert sum(assignment) == nodes
        assert all(x > 0 for x in assignment)
        assert len(assignment) == devices


@pytest.mark.multifpga
@pytest.mark.parametrize("device_node_combinations", [(2, 2), (10, 20), (100, 200), (1, 2)])
@pytest.mark.parametrize("assignment_type", ["random", "equal"])
@pytest.mark.parametrize("device_assignment", ["linear"])
def test_sdp_creation(
    device_node_combinations: tuple[int, int], assignment_type: str, device_assignment: str
) -> None:
    device_count, node_count = device_node_combinations

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

    if device_assignment == "linear":
        overall_node_index = 0
        for current_device in range(len(assignment)):
            while assignment[current_device] > 0:
                getCustomOp(model.graph.node[overall_node_index]).set_nodeattr(
                    "device_id", current_device
                )
                overall_node_index += 1
                assignment[current_device] -= 1
    else:
        raise NotImplementedError()

    # Creation of the SDPs
    original_model = deepcopy(model)
    model = model.transform(CreateMultiFPGAStreamingDataflowPartition())
    sdp_test_dir = make_build_dir("test_sdp_creation")
    model.save(os.path.join(sdp_test_dir, "sdp_model.onnx"))  # noqa

    # Check that all nodes in the parent graph are now SDPs
    for node in model.graph.node:
        assert node.op_type == "StreamingDataflowPartition"

    # Check that the order of nodes along the graph is kept
    partitioned_order = []
    order = [n.name for n in original_model.graph.node]
    for node in model.graph.node:
        submodel = ModelWrapper(getCustomOp(node).get_nodeattr("model"))
        for snode in submodel.graph.node:
            partitioned_order.append(snode.name)
    assert partitioned_order == order

    # Parent model shouldnt have any branches
    for node in model.graph.node:
        sucs = model.find_direct_successors(node)
        assert (sucs is None) or len(sucs) == 1

    # Check that all submodels' nodes have the same device ID
    for node in model.graph.node:
        submodel = ModelWrapper(getCustomOp(node).get_nodeattr("model"))
        devices_found = [get_device_id(n) for n in submodel.graph.node]
        assert len(set(devices_found)) == 1

    # Check that no two SDPs are on the same device after another
    for i in range(len(model.graph.node) - 1):
        node_a_device = get_device_id(model.graph.node[i])
        node_b_device = get_device_id(model.graph.node[i + 1])
        assert (
            node_a_device != node_b_device
        ), f"Consecutive SDPs with the same device: {[get_device_id(x) for x in model.graph.node]}"


@pytest.mark.multifpga
def test_fail_on_split_branch_nodes() -> None:
    raise AssertionError()

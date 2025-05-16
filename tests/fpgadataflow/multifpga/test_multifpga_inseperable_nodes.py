from __future__ import annotations

import pytest

import onnx.helper as oh
from networkx import DiGraph, articulation_points
from qonnx.core.modelwrapper import ModelWrapper

from finn.transformation.fpgadataflow.multifpga_utils import (
    _get_end_nodes,
    _get_split_nodes,
    _get_start_nodes,
    _split_nodes_from,
    get_inseperable_nodes,
)


def list_contains_all_elements(this: list, other: list) -> bool:
    return all(n in this for n in other)


def networkx_to_onnx(g: DiGraph) -> ModelWrapper:
    nodes = [oh.make_node("StreamingDataflowPartition", [], [], n) for n in g.nodes]
    get_node_by_name = lambda name: [n for n in nodes if n.name == name][0]  # noqa
    for i, edge in enumerate(g.edges):
        source_node = get_node_by_name(edge[0])
        target_node = get_node_by_name(edge[1])
        source_node.output.append(f"edge_{i}")
        target_node.input.append(f"edge_{i}")
    # TODO: Make graph inputs and outputs
    graph = oh.make_graph(nodes, "graph", [], [])
    model = oh.make_model(graph)
    return ModelWrapper(model)


# Graphs and what the expected results are. If None, the function should crash
graphs = {
    "single-unequal-weighted-branch": (
        DiGraph(
            [
                ("A", "B"),
                ("B", "C"),
                ("C", "D"),
                ("C", "E"),
                ("D", "D1"),
                ("D1", "D2"),
                ("E", "E1"),
                ("E1", "E2"),
                ("E2", "E3"),
                ("E3", "F"),
                ("D2", "F"),
                ("F", "G"),
            ]
        ),
        [["C", "E", "E1", "E2", "E3", "D", "D1", "D2", "F"]],
    ),
    "small-diamonds": (
        DiGraph(
            [
                ("A", "B"),
                ("B", "D"),
                ("A", "C"),
                ("C", "D"),
                ("D", "E"),
                ("E", "F"),
                ("F", "G"),
                ("F", "H"),
                ("G", "I"),
                ("H", "I"),
            ]
        ),
        [["A", "B", "C", "D"], ["F", "G", "H", "I"]],
    ),
    "no-branches": (DiGraph([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]), []),
    "all-one-branch": (
        DiGraph(
            [
                ("A", "B"),
                ("A", "C"),
                ("B", "B1"),
                ("B1", "B2"),
                ("C", "C1"),
                ("C1", "C2"),
                ("C2", "C3"),
                ("C3", "D"),
                ("B2", "D"),
            ]
        ),
        [["A", "B", "C", "D", "B1", "B2", "C1", "C2", "C3"]],
    ),
    "two_input_graph": (DiGraph([("A", "C"), ("B", "C"), ("C", "D"), ("D", "E")]), []),
    "two_output_graph": (DiGraph([("A", "B"), ("B", "C"), ("C", "D"), ("C", "E")]), []),
}


# TODO: Doesnt yet work for two inputs. This should be caught by the transformation instead,
# but it still should be well defined in the function.
@pytest.mark.parametrize(
    "graph_data",
    [
        graphs["single-unequal-weighted-branch"],
        graphs["small-diamonds"],
        graphs["no-branches"],
        graphs["all-one-branch"],
        graphs["two_input_graph"],
        graphs["two_output_graph"],
    ],
)
def test_find_split_nodes(graph_data: tuple[DiGraph, list[list[str]]]) -> None:
    """Test that all splits are found. Check on a networkx graph directly"""
    g, expected_splits = graph_data
    art_points = list(articulation_points(g.to_undirected())) + _get_end_nodes(g)
    all_splits = [_split_nodes_from(g, splitter, art_points) for splitter in _get_split_nodes(g)]
    assert len(expected_splits) == len(all_splits)
    for expected_split_list in expected_splits:
        found_and_correct = False
        for found_split_list in all_splits:
            if list_contains_all_elements(expected_split_list, found_split_list) and len(
                found_split_list
            ) == len(expected_split_list):
                found_and_correct = True
        assert found_and_correct, (
            f"Did not find inseperable node list: {expected_split_list}. "
            f"Available lists were: {all_splits}."
            f" Splitters were: {_get_split_nodes(g)} "
            f"Cut vertices: {list(articulation_points(g.to_undirected()))}"
        )


def test_correct_input_count() -> None:
    assert len(_get_start_nodes(graphs["two_input_graph"][0])) == 2
    assert len(_get_end_nodes(graphs["two_input_graph"][0])) == 1


def test_correct_output_count() -> None:
    assert len(_get_start_nodes(graphs["two_output_graph"][0])) == 1
    assert len(_get_end_nodes(graphs["two_output_graph"][0])) == 2


def test_onnx_to_networkx() -> None:
    raise NotImplementedError()


# TODO: Doesnt yet work for two inputs. This should be caught by the transformation instead,
# but it still should be well defined in the function.
@pytest.mark.parametrize(
    "graph_data",
    [
        graphs["single-unequal-weighted-branch"],
        graphs["small-diamonds"],
        graphs["no-branches"],
        graphs["all-one-branch"],
        graphs["two_input_graph"],
        graphs["two_output_graph"],
    ],
)
def test_inseperable_nodes(graph_data: tuple[DiGraph, list[list[str]]]) -> None:
    g, expected_splits = graph_data
    model = networkx_to_onnx(g)
    indices = {}
    for i, node in enumerate(model.graph.node):
        indices[node.name] = i
    found_splits = get_inseperable_nodes(model)
    assert len(found_splits) == len(expected_splits)
    for expected_split_list in expected_splits:
        found_and_correct = False
        expected_split_list_int = [indices[n] for n in expected_split_list]
        for found_split_list in found_splits:
            if list_contains_all_elements(expected_split_list_int, found_split_list) and len(
                found_split_list
            ) == len(expected_split_list):
                found_and_correct = True
        assert found_and_correct, (
            f"Did not find inseperable node list: {expected_split_list}. "
            f"Available lists were: {found_splits}."
            f" Splitters were: {_get_split_nodes(g)} "
            f"Cut vertices: {list(articulation_points(g.to_undirected()))}"
        )


def test_resnet50_examples_inseperable_nodes() -> None:
    raise NotImplementedError()

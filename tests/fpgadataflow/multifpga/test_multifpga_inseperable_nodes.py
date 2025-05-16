import pytest

from networkx import DiGraph, articulation_points

from finn.transformation.fpgadataflow.multifpga_utils import (
    _get_end_nodes,
    _get_split_nodes,
    _split_nodes_from,
)


def list_contains_all_elements(this: list, other: list) -> bool:
    return all(n in this for n in other)


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
}


@pytest.mark.parametrize(
    "graph_data",
    [
        graphs["single-unequal-weighted-branch"],
        graphs["small-diamonds"],
        graphs["no-branches"],
        graphs["all-one-branch"],
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


def test_onnx_to_networkx() -> None:
    raise NotImplementedError()


def test_inseperable_nodes() -> None:
    raise NotImplementedError()


def test_resnet50_examples_inseperable_nodes() -> None:
    raise NotImplementedError()

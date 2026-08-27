"""Test the graph utility functions regarding inseparable nodes, which is important for the current
Multi-FPGA implementation.
"""

from __future__ import annotations

import pytest

from networkx import DiGraph, articulation_points

from finn.builder.build_dataflow_config import DataflowBuildConfig, ShellFlowType
from finn.transformation.fpgadataflow.multifpga.graph import (
    _get_end_nodes_nx,
    _get_inseparable_nodes_nx,
    _get_split_nodes_nx,
    _get_start_nodes_nx,
    get_inseparable_nodes,
    onnx_to_networkx,
)
from finn.util.basic import make_build_dir
from tests.fpgadataflow.multifpga.utils import (
    get_model,
    list_contains_all_elements,
    networkx_to_onnx,
)

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
    "two_input_graph_unbalanced": (
        DiGraph([("A", "B"), ("B", "C"), ("X", "C"), ("C", "D"), ("D", "E")]),
        [["A", "B", "X", "C"]],
    ),
    "two_output_graph_unbalanced": (
        DiGraph([("A", "B"), ("B", "C"), ("C", "D"), ("C", "E"), ("E", "F")]),
        [["C", "D", "E", "F"]],
    ),
    "two_input_graph": (
        DiGraph([("A", "C"), ("B", "C"), ("C", "D"), ("D", "E")]),
        [["A", "B", "C"]],
    ),
    "two_output_graph": (
        DiGraph([("A", "B"), ("B", "C"), ("C", "D"), ("C", "E")]),
        [["C", "D", "E"]],
    ),
}


@pytest.mark.parametrize(
    "graph_data",
    [
        graphs["single-unequal-weighted-branch"],
        graphs["small-diamonds"],
        graphs["no-branches"],
        graphs["all-one-branch"],
        graphs["two_input_graph"],
        graphs["two_output_graph"],
        graphs["two_input_graph_unbalanced"],
        graphs["two_output_graph_unbalanced"],
    ],
)
def test_find_split_nodes_networkx(graph_data: tuple[DiGraph, list[list[str]]]) -> None:
    """Test that all splits in a networkx graph are found."""
    g, expected_splits = graph_data
    all_splits = _get_inseparable_nodes_nx(g)
    assert len(expected_splits) == len(all_splits), (
        f"Expected {len(expected_splits)} but found {len(all_splits)} splits."
        f"The splits found were: {all_splits}. "
        f"The expected splits were: {expected_splits}."
    )
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
            f" Splitters were: {_get_split_nodes_nx(g)} "
            f"Cut vertices: {list(articulation_points(g.to_undirected()))}"
        )


def test_correct_input_count() -> None:
    """Internal test."""
    assert len(_get_start_nodes_nx(graphs["two_input_graph"][0])) == 2
    assert len(_get_end_nodes_nx(graphs["two_input_graph"][0])) == 1


def test_correct_output_count() -> None:
    """Internal test."""
    assert len(_get_start_nodes_nx(graphs["two_output_graph"][0])) == 1
    assert len(_get_end_nodes_nx(graphs["two_output_graph"][0])) == 2


@pytest.mark.parametrize(
    "model_type",
    [
        ("CNV", 1, 1, True),
        ("CNV", 1, 2, True),
        ("CNV", 2, 2, True),
        ("LFC", 1, 1, True),
        ("LFC", 1, 2, True),
        ("SFC", 1, 2, True),
        ("SFC", 2, 2, True),
        ("TFC", 1, 1, True),
        ("TFC", 1, 2, True),
        ("mobilenetv1", 4, 4, True),
        ("resnet18", 4, 4, True),
    ],
)
def test_onnx_to_networkx(
    model_type: tuple[str, int, int, bool], pytestconfig: pytest.Config
) -> None:
    """Test that the conversion between a modelwrapper and a networkx graph is done correctly."""
    model_name, wbits, abits, pretrained = model_type
    cfg = DataflowBuildConfig(
        output_dir=make_build_dir("test_onnx_nx_"),
        board="U55C",
        shell_flow_type=ShellFlowType.VITIS_ALVEO,
        target_fps=1000,
        synth_clk_period_ns=5.0,
        standalone_thresholds=True,
    )
    model, _ = get_model(
        model_name,
        wbits,
        abits,
        pretrained,
        "step_set_fifo_depths",
        True,
        cfg,
        pytestconfig,
        "onnx_to_nx",
    )
    g = onnx_to_networkx(model)

    # First check (and count) edges in the QONNX graph and see if they exist in the NX graph
    edges_model = 0
    for node in model.graph.node:
        sucs = model.find_direct_successors(node)
        if sucs is None:
            continue
        for suc in sucs:
            edges_model += 1
            assert (node.name, suc.name) in g.edges

    # Check that there are no other edges in the NX graph
    assert len(g.edges) == edges_model

    # Check node-equivalence
    assert set(g.nodes) == {node.name for node in model.graph.node}
    for _, data in g.nodes(data=True):
        assert data["onnx_node"] in model.graph.node


@pytest.mark.parametrize(
    "graph_data",
    [
        graphs["single-unequal-weighted-branch"],
        graphs["small-diamonds"],
        graphs["no-branches"],
        graphs["all-one-branch"],
        graphs["two_input_graph"],
        graphs["two_output_graph"],
        graphs["two_input_graph_unbalanced"],
        graphs["two_output_graph_unbalanced"],
    ],
)
def test_inseparable_nodes_qonnx(graph_data: tuple[DiGraph, list[list[str]]]) -> None:
    """Check that the inseparable node function finds the correct node groups by checking against
    pre-defined examples. The networkx graph is first converted to an ONNX graph.
    """
    g, expected_splits = graph_data
    model = networkx_to_onnx(g)
    indices = {}
    for i, node in enumerate(model.graph.node):
        indices[node.name] = i
    found_splits = get_inseparable_nodes(model)
    assert len(found_splits) == len(expected_splits), (
        f"Expected {len(expected_splits)} but found {len(found_splits)} splits. "
        f"The splits found were: {found_splits}. "
        f"The expected splits were: {expected_splits}."
    )
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
            f" Splitters were: {_get_split_nodes_nx(g)} "
            f"Cut vertices: {list(articulation_points(g.to_undirected()))}"
        )


def test_resnet18_examples_inseparable_nodes(pytestconfig: pytest.Config) -> None:
    """Test that the expected number of inseparable-node groups and
    group-sizes are found for the Resnet18.
    """
    cfg = DataflowBuildConfig(
        output_dir=make_build_dir("rn18_insep_nodes_"),
        board="U55C",
        shell_flow_type=ShellFlowType.VITIS_ALVEO,
        target_fps=100,
    )
    model, _ = get_model(
        "resnet18", 4, 4, False, "step_set_fifo_depths", True, cfg, pytestconfig, "rn18_insep_nodes"
    )
    groups = get_inseparable_nodes(model)
    assert len(groups) == 8
    assert max(len(group) for group in groups) == 10
    assert min(len(group) for group in groups) == 8

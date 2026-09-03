"""Graoh utils (mostly networkx based) for Multi-FPGA purposes."""
import itertools
import networkx as nx
from qonnx.core.modelwrapper import ModelWrapper
from typing import Any

from finn.util.exception import FINNMultiFPGAError


def onnx_to_networkx(model: ModelWrapper) -> nx.DiGraph:
    """Naively build a directed networkx graph from an ONNX graph.

    This differs from onnx-passes.utils.networkx.onnx_to_nx in that this function does _not_
    consider graph inputs and outputs as nodes, since in this context we only care about
    HW-synthesizable nodes which can have a designated device ID.
    """
    nxg = nx.DiGraph()
    for node in model.graph.node:
        nxg.add_node(node.name, onnx_node=node)
    for node in model.graph.node:
        pre = model.find_direct_predecessors(node)
        if pre is None:
            pre = []
        suc = model.find_direct_successors(node)
        if suc is None:
            suc = []
        for predecessor in pre:
            nxg.add_edge(predecessor.name, node.name)
        for successor in suc:
            nxg.add_edge(node.name, successor.name)
    return nxg


def _get_split_nodes_nx(g: nx.DiGraph) -> list[str]:
    """Return all nodes which have more than 1 successor (split the graph).

    >>> g = nx.DiGraph([(0,1), (0,2), (1,3), (2,3), (2,4)])
    >>> _get_split_nodes_nx(g)
    [0, 2]
    """
    return list(filter(lambda n: len(g.out_edges(n)) > 1, g.nodes))


def _split_nodes_from_nx(g: nx.DiGraph, source_node_name: str, art_points: list[str]) -> list[str]:
    """From the source vertex, find the first cut vertex, which is the joining node. Collect all
    nodes from source to cut vertex and return them.

    >>> g = nx.DiGraph([(0,1), (1,2), (2,3), (2,4), (3,5), (5,6), (6,7), (4,8), (8,9), (9,6), (6,10)])
    >>> sorted(_split_nodes_from_nx(g, 2, nx.articulation_points(g.to_undirected())))
    [2, 3, 4, 5, 6, 8, 9]
    """  # noqa
    assert len(g.out_edges(source_node_name)) > 1
    ap = list(art_points)
    for node in nx.dfs_preorder_nodes(g, source_node_name):
        if node in ap and node != source_node_name:
            return list(
                set(
                    itertools.chain.from_iterable(nx.node_disjoint_paths(g, source_node_name, node))
                )
            )
    return []


def _get_end_nodes_nx(g: nx.DiGraph) -> list[str]:
    """Return all nx DiGraph nodes that are end points (no outgoing edges, atleast
    one incoming edge).

    >>> g = nx.DiGraph([(0,1), (0,2), (10,11), (11,12), (10,13), (12,14), (13,14)])
    >>> sorted(_get_end_nodes_nx(g))
    [1, 2, 14]
    """
    return [n for n in g.nodes() if g.in_degree(n) > 0 and g.out_degree(n) == 0]


def _get_start_nodes_nx(g: nx.DiGraph) -> list[str]:
    """Return all start nodes (> 0 outgoing, 0 incoming edges).

    >>> g = nx.DiGraph([(0,1), (0,2), (10,11), (11,12), (10,13), (12,14), (13,14)])
    >>> sorted(_get_start_nodes_nx(g))
    [0, 10]
    """
    return [n for n in g.nodes() if g.in_degree(n) == 0 and g.out_degree(n) > 0]


def is_single_in_out_model(model: ModelWrapper) -> bool:
    """Return whether the given model has only one input and one output."""
    g = onnx_to_networkx(model)
    return len(_get_start_nodes_nx(g)) == 1 and len(_get_end_nodes_nx(g)) == 1


def _convert_to_index_groups(model: ModelWrapper, split_names: list[list[str]]) -> list[list[int]]:
    """Convert all groups of names to their indices in the graph."""
    idxs = {}
    for i, node in enumerate(model.graph.node):
        # TODO: Eventually remove this requirement
        if node.name in idxs.keys():
            raise FINNMultiFPGAError(
                "Cannot properly collect inseperable nodes - nodes don't have unique names!"
            )
        idxs[node.name] = i
    return [[idxs[nodename] for nodename in insep_nodes] for insep_nodes in split_names]


def _get_inseparable_nodes_nx(g: nx.DiGraph) -> list[list[str]]:
    """Return a list of all nodes that need to stay together during
    partitioning. (Operate on the nx graph).

    IMPORTANT:
    ---------
        This function might deliver several smaller groups for a single branch/join section. Since
        these groups share nodes however, for the partitioner there is no difference between
        `[A, B], [B, C]` and `[A, B, C]`.

    >>> g = nx.DiGraph([(0,1), (1,2), (1,3), (2,4), (3,4), (4,5), (5,6), (5,7),
    ...     (6,8), (8,9), (9,10), (7,11), (11,12), (12,13), (12,14), (13,10), (14,10)])
    >>> nodelist = _get_inseparable_nodes_nx(g)
    >>> nodelist = [sorted(nl) for nl in nodelist]
    >>> nodelist
    [[1, 2, 3, 4], [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
    """  # noqa
    # Also count last nodes so that a graph ending in a join node also is processed correctly
    art_points = list(nx.articulation_points(g.to_undirected())) + _get_end_nodes_nx(g)
    start_nodes = _get_start_nodes_nx(g)
    end_nodes = _get_end_nodes_nx(g)
    result = []
    for splitter in _get_split_nodes_nx(g):
        path_nodes: list[Any] = _split_nodes_from_nx(g, splitter, art_points)
        if path_nodes != []:
            result.append(path_nodes)

    # if we have multiple inputs, find the first merge for every input, and put all nodes
    # between them into one large group. This can be done, since even if one input joins
    # much later, we still don't allow a device crossing with two parallel branches,
    # so we need to group every node until the "furthest" join.
    start_group = []
    if len(start_nodes) > 1:
        for start_node in start_nodes:
            for target_node in nx.dfs_preorder_nodes(g, source=start_node):
                if g.in_degree(target_node) > 1:
                    for path in nx.node_disjoint_paths(g, start_node, target_node):
                        start_group += path
                    break
    if start_group != []:
        result.append(list(set(start_group)))

    # Same for the output nodes
    end_group = []
    if len(end_nodes) > 1:
        for end_node in end_nodes:
            # We have to change g to undirected here, since we
            # search "upwards" in the graph until the first fork
            for target_node in nx.dfs_preorder_nodes(g.to_undirected(), end_node):
                if g.out_degree(target_node) > 1:
                    for path in nx.node_disjoint_paths(g, target_node, end_node):
                        end_group += path
    if end_group != []:
        result.append(list(set(end_group)))

    # Merge connected groups
    for i in range(len(result)):
        found_merge = True
        while found_merge:
            found_merge = False
            for j in range(i + 1, len(result)):
                if any(node in result[j] for node in result[i]):
                    result[i] = list(set(result[i] + result.pop(j)))
                    found_merge = True
                    break
    return result


def get_inseparable_nodes(model: ModelWrapper) -> list[list[int]]:
    """Return a list of all nodes (indices) that need to stay together during
    partitioning.

    IMPORTANT:
    ---------
        This function will currently, when getting nested branches, use the smallest branches possible
        as groups, as can be seen in the doctest (_get_inseparable_nodes_nx). Since however, one side of the nested branch _must_
        belong to the other side of the main branch, both groups share nodes. Because the partitioner ILP
        requires then (e.g.) A and B to be on the same device, and B and C, it effectively groups together A,
        B and C.
    """  # noqa
    g = onnx_to_networkx(model)
    return _convert_to_index_groups(model, _get_inseparable_nodes_nx(g))

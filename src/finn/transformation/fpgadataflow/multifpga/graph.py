"""Graph utils (mostly networkx based) for Multi-FPGA purposes."""
import networkx as nx
from qonnx.core.modelwrapper import ModelWrapper
from typing import Any

from finn.util.exception import FINNMultiFPGAError, FINNMultiFPGAUserError


def onnx_to_networkx(model: ModelWrapper) -> nx.DiGraph:
    """Naively build a directed networkx graph from an ONNX graph.

    This differs from onnx-passes.utils.networkx.onnx_to_nx in that this function does _not_
    consider graph inputs and outputs as nodes, since in this context we only care about
    HW-synthesizable nodes which can have a designated device ID.

    Raises FINNMultiFPGAError if node names are not unique, since they are the networkx nodes.
    """
    nxg = nx.DiGraph()
    for node in model.graph.node:
        if node.name in nxg:
            raise FINNMultiFPGAError(
                f"Cannot build the networkx graph: node name {node.name} is not unique!"
            )
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


def _find_branch_nodes_until_common_join(g: nx.DiGraph, start_nodes: list[Any]) -> list[Any]:
    """Return all nodes from the start nodes up to their first common joins (both included), or
    an empty list if the start nodes never all join.
    """
    # Get the nodes reachable from each start node. A start node counts as reachable from itself,
    # since one branch of a fork can lead directly into the join (e.g. a residual connection)
    reachable = [{start_node} | nx.descendants(g, start_node) for start_node in start_nodes]

    # Their intersection are the nodes reachable from all start nodes
    common = set.intersection(*reachable)

    # If a node has no ancestor in this set, it means it was reachable from every start node,
    # but none of its ancestors were. These first nodes reachable from every start node are
    # the join nodes
    join_nodes = {node for node in common if nx.ancestors(g, source=node).isdisjoint(common)}

    # The group are the join nodes and their ancestors, as far as they are reachable from a start
    # node. For the branches of a fork, this excludes the fork and everything before it
    reachable_from_any = set.union(*reachable)
    group = set(join_nodes)
    for join_node in join_nodes:
        group |= nx.ancestors(g, source=join_node) & reachable_from_any
    return list(group)


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


def _get_inseparable_nodes_nx(g: nx.DiGraph) -> list[list[str]]:
    """Return the groups of nodes that need to stay on the same device during partitioning, so
    that every device crossing carries a single stream. (Operate on the nx graph).

    Grouped together are:
        - every fork with its branches, up to the node where they join again,
        - with several inputs: all nodes from the inputs up to their first common merges,
        - with several outputs: all nodes from the last common ancestors down to the outputs.
    Groups that share a node are merged, so the returned groups are disjoint.

    Raises FINNMultiFPGAUserError if the inputs have no common merge, or the outputs no common
    ancestor.

    >>> def groups(edges):
    ...     return sorted(sorted(group) for group in _get_inseparable_nodes_nx(nx.DiGraph(edges)))

    A chain needs no group:

    >>> groups([(0, 1), (1, 2)])
    []

    A fork and its join, but not the nodes before and after them:

    >>> groups([(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
    [[1, 2, 3, 4]]

    A residual connection, where one branch of the fork 1 leads directly into the join 3:

    >>> groups([(0, 1), (1, 2), (1, 3), (2, 3), (3, 4)])
    [[1, 2, 3]]

    A fork with three branches, where 2 and 3 join in 5 first and 4 only joins in 6:

    >>> groups([(0, 1), (1, 2), (1, 3), (1, 4), (2, 5), (3, 5), (5, 6), (4, 6), (6, 7)])
    [[1, 2, 3, 4, 5, 6]]

    Two separate fork/join sections give two groups:

    >>> groups([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (5, 7), (6, 7)])
    [[0, 1, 2, 3], [4, 5, 6, 7]]

    Several inputs, grouped up to their common merge 2:

    >>> groups([(0, 2), (1, 2), (2, 3)])
    [[0, 1, 2]]

    Inputs 0 and 1 merge in 2, input 4 only joins in 5, so node 3 between the merges is grouped too:

    >>> groups([(0, 2), (1, 2), (2, 3), (3, 5), (4, 5), (5, 6)])
    [[0, 1, 2, 3, 4, 5]]

    Several outputs, grouped from their last common ancestor 1 on:

    >>> groups([(0, 1), (1, 2), (1, 3)])
    [[1, 2, 3]]

    Nested forks before the outputs:

    >>> groups([(0, 1), (1, 2), (1, 3), (2, 4), (2, 5)])
    [[1, 2, 3, 4, 5]]

    Inputs 0 and 1 merge in 3, inputs 1 and 2 in 4, but no node merges all three:

    >>> groups([(0, 3), (1, 3), (1, 4), (2, 4)])  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    FINNMultiFPGAUserError: Not all inputs of the model (0, 1, 2) merge into a common node, ...

    Outputs 2 and 3 share the ancestor 0, outputs 3 and 4 the ancestor 1, but no node is an
    ancestor of all three:

    >>> groups([(0, 2), (0, 3), (1, 3), (1, 4)])  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    FINNMultiFPGAUserError: Not all outputs of the model (2, 3, 4) have a common ancestor, ...
    """
    # Define some helpers
    start_nodes = _get_start_nodes_nx(g)
    end_nodes = _get_end_nodes_nx(g)

    # Collect all groups
    result = []

    # INNER GROUPS
    # Group every splitting node with all nodes of its branches up to their common join. Branches
    # that never join again (e.g. separate outputs) are covered by the end group
    for splitter in _get_split_nodes_nx(g):
        branch_nodes = _find_branch_nodes_until_common_join(g, list(g.successors(splitter)))
        if branch_nodes != []:
            result.append([splitter, *branch_nodes])

    # START GROUP
    # Collect all nodes from the start nodes to their first common joins
    if len(start_nodes) > 1:
        start_group = _find_branch_nodes_until_common_join(g, start_nodes)
        if start_group == []:
            raise FINNMultiFPGAUserError(
                f"Not all inputs of the model ({', '.join(map(str, start_nodes))}) merge into a "
                "common node, so no single-stream cut separates them. Either set "
                "single_stream_network=False (with an ARBITRARY* topology), or split the model "
                "into separate accelerators."
            )
        result.append(start_group)

    # END GROUP
    # Same for the output nodes. Here we need to reverse the direction of all edges
    # so that the networkx functions do the correct thing (i.e. you need to swap the meaning
    # of "ancestors" and "descendants")
    if len(end_nodes) > 1:
        end_group = _find_branch_nodes_until_common_join(g.reverse(copy=False), end_nodes)
        if end_group == []:
            raise FINNMultiFPGAUserError(
                f"Not all outputs of the model ({', '.join(map(str, end_nodes))}) have a common "
                "ancestor, so no single-stream cut separates them. Either set "
                "single_stream_network=False (with an ARBITRARY* topology), or split the model "
                "into separate accelerators."
            )
        result.append(end_group)

    # MERGE
    # Merge connected groups (i.e. groups that have at least one node in common -
    # inseparable-ness is transitive)
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


def get_inseparable_nodes(model: ModelWrapper) -> list[list[str]]:
    """Return the groups of nodes (names) that need to stay on the same device during
    partitioning. The groups are disjoint. See `_get_inseparable_nodes_nx` for the grouping rules.
    """
    return _get_inseparable_nodes_nx(onnx_to_networkx(model))

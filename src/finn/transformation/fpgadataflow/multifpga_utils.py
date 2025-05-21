from __future__ import annotations

import itertools
import networkx as nx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from typing import TYPE_CHECKING

from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.util.exception import FINNMultiFPGAError

if TYPE_CHECKING:
    from onnx import NodeProto

    from finn.util.platforms import Platform


def get_device_id(node: NodeProto) -> int | None:
    try:
        return getCustomOp(node).get_nodeattr("device_id")
    except ValueError:
        return None


def set_device_id(node: NodeProto, value: int) -> None:
    getCustomOp(node).set_nodeattr("device_id", value)


def get_last_submodel_node(sdp_node: NodeProto) -> NodeProto:
    return get_submodel(sdp_node).graph.node[-1]


def get_first_submodel_node(sdp_node: NodeProto) -> NodeProto:
    return get_submodel(sdp_node).graph.node[0]


def get_submodel(node: NodeProto) -> ModelWrapper:
    return ModelWrapper(getCustomOp(node).get_nodeattr("model"))


def onnx_to_networkx(model: ModelWrapper) -> nx.DiGraph:
    """Naively build a directed networkx graph from an ONNX graph"""
    nxg = nx.DiGraph()
    for node in model.graph.node:
        nxg.add_node(node.name, d=node)
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


def _get_split_nodes(g: nx.DiGraph) -> list[str]:
    """Return all nodes which have more than 1 successor (split the graph)"""
    return list(filter(lambda n: len(g.out_edges(n)) > 1, g.nodes))


def _split_nodes_from(g: nx.DiGraph, source_node_name: str, art_points: list[str]) -> list[str]:
    """From the source vertex, find the first cut vertex, which is the joining node. Collect all
    nodes from source to cut vertex and return them."""
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


def _get_end_nodes(g: nx.DiGraph) -> list[str]:
    """Return all nx DiGraph nodes that are end points (no outgoing edges, atleast
    one incoming edge)"""
    return [n for n in g.nodes() if g.in_degree(n) > 0 and g.out_degree(n) == 0]


def _get_start_nodes(g: nx.DiGraph) -> list[str]:
    """Return all start nodes (> 0 outgoing, 0 incoming edges)"""
    return [n for n in g.nodes() if g.in_degree(n) == 0 and g.out_degree(n) > 0]


def is_single_in_out_model(model: ModelWrapper) -> bool:
    """Return whether the given model has only one input and one output"""
    g = onnx_to_networkx(model)
    return len(_get_start_nodes(g)) == 1 and len(_get_end_nodes(g)) == 1


def _convert_to_index_groups(model: ModelWrapper, split_names: list[list[str]]) -> list[list[int]]:
    """Convert all groups of names to their indices in the graph"""
    idxs = {}
    for i, node in enumerate(model.graph.node):
        # TODO: Eventually remove this requirement
        if node.name in idxs.keys():
            raise FINNMultiFPGAError(
                "Cannot properly collect inseperable nodes - nodes don't have unique names!"
            )
        idxs[node.name] = i
    return [[idxs[nodename] for nodename in insep_nodes] for insep_nodes in split_names]


def get_inseperable_nodes(model: ModelWrapper) -> list[list[int]]:
    """Return a list of all nodes that need to stay together during
    partitioning"""
    # TODO: Convert / check for cases where the branches have branches themselves
    g = onnx_to_networkx(model)

    # Also count last nodes so that a graph ending in a join node also is processed correctly
    art_points = list(nx.articulation_points(g.to_undirected())) + _get_end_nodes(g)
    all_splits = [_split_nodes_from(g, splitter, art_points) for splitter in _get_split_nodes(g)]
    return _convert_to_index_groups(model, all_splits)


def get_estimated_model_resources(model: ModelWrapper, fpga_part: str) -> dict[int, dict[str, int]]:
    """Gather the resources of all layers based on the estimation values from
    the previous build steps. Return them by the enumerated number of the node

    IMPORTANT: If the ordering or number of nodes in the graph changes, this becomes invalid!
    Returns a table like:
    {
        0: {
            "LUT": ...,
            "DSP": ...,
            ...
        },
        ...
    }"""
    # TODO: Check / Clean up all the various estimate functions
    estimates = res_estimation(model, fpga_part)
    hls_estimates = hls_synth_res_estimation(model)
    for layer in hls_estimates.keys():
        # Case 1: Only HLS estimate: Add it
        if layer not in estimates.keys():
            estimates[layer] = hls_estimates[layer]
        else:
            current_layer_estimates = hls_estimates[layer]
            for restype in current_layer_estimates.keys():
                # Case 2: Res exists in both estimates: Take max
                if restype in estimates[layer].keys():
                    estimates[layer][restype] = max(
                        estimates[layer][restype], current_layer_estimates[restype]
                    )
                # Case 3: Res exists only in hls: Add it
                else:
                    estimates[layer][restype] = current_layer_estimates[restype]
    est_by_index = {}
    for i, node in enumerate(model.graph.node):
        est_by_index[i] = estimates[node.name]
    return est_by_index


def _resources_per_device_per_slr(p: Platform) -> dict[int, dict[str, int]]:
    """Return the available resources as given by FINN platforms as a
    dictionary instead of nested lists. First by SLR, then by resource name"""
    assert p is not None
    assert p.compute_resources is not None
    res = p.compute_resources
    new = {}
    for slr in range(len(res)):
        new[slr] = {}
        for i, name in enumerate(["LUT", "FF", "BRAM_18K", "URAM", "DSP"]):
            new[slr][name] = res[slr][i]
    return new


def available_resources(p: Platform, considered_resources: list[str]) -> dict[str, int]:
    """Return the total resources per device. Normally,
    these values are split by SLR"""
    resources_per_device = _resources_per_device_per_slr(p)
    if resources_per_device is None:
        return {}
    acc = {}
    for restype in considered_resources:
        acc[restype] = 0
        for slr in resources_per_device:
            acc[restype] += resources_per_device[slr][restype]
    return acc

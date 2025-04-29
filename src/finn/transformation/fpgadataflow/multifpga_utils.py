from __future__ import annotations

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from typing import TYPE_CHECKING

from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.analysis.fpgadataflow.res_estimation import res_estimation

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


def get_inseperable_nodes(model: ModelWrapper) -> list[list[int]]:
    """Return a list of all nodes that need to stay together during
    partitioning"""
    raise NotImplementedError()


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

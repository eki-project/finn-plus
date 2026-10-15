"""Resource utility functions."""

from qonnx.core.modelwrapper import ModelWrapper

from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.util.exception import FINNConfigurationError, FINNUserError
from finn.util.logging import log
from finn.util.platforms import Platform, platforms

ResourceEstimates = dict[str, dict[str, int | float]]
"""Short alias for a resource dict."""

ResourceEstimatesByIndex = dict[int, dict[str, int | float]]
"""Short alias for a resource dict."""

PLATFORM_RESOURCE_TYPES = ["LUT", "FF", "BRAM_18K", "URAM", "DSP"]
"""Resource types listed per SLR in the platforms' `compute_resources` (in that order)."""


def _merge_resource_estimations(
    finn_estimates: ResourceEstimates, hls_estimates: ResourceEstimates
) -> ResourceEstimates:
    """Merge two resource estimates (e.g. FINN and HLS).

    Strategy:
    --------
        For a given node/resource-type combination:
        1. If only one of the estimates is available, use it.
        2. If an estimate exists in both, take the larger one.
    """
    result: ResourceEstimates = {}
    all_layers: set[str] = set(list(finn_estimates) + list(hls_estimates))
    for layer in all_layers:
        # Layer estimates only in A
        if (layer in finn_estimates) and (layer not in hls_estimates):
            result[layer] = finn_estimates[layer]

        # Layer estimates only in B
        elif (layer not in finn_estimates) and (layer in hls_estimates):
            result[layer] = hls_estimates[layer]

        # Layer estimates in both
        else:
            result[layer] = {}
            all_res_types = set(list(finn_estimates[layer]) + list(hls_estimates[layer]))
            for restype in all_res_types:
                # Resource estimates only in A
                if (restype in finn_estimates[layer]) and (restype not in hls_estimates[layer]):
                    result[layer][restype] = finn_estimates[layer][restype]

                # Resource estimates only in B
                elif (restype not in finn_estimates[layer]) and (restype in hls_estimates[layer]):
                    result[layer][restype] = hls_estimates[layer][restype]

                # Resource estimate in both
                else:
                    result[layer][restype] = max(
                        finn_estimates[layer][restype], hls_estimates[layer][restype]
                    )
    return result


def get_estimated_model_resources(  # noqa
    model: ModelWrapper,
    fpga_part: str,
    considered_resources: list[str],
    add_missing_resources: bool,
) -> ResourceEstimates:
    """Gather resources of all layers based on estimates both from FINNs HWCustomOp implementation,
    as well as the HLS reports. These are then merged to produce
    the worst case resource estimates and returned.

    Arguments:
    ---------
        `model`: The model to estimate.
        `fpga_part`: FPGA Part identifier.
        `considered_resources`: A list of resource types to consider.
        `add_missing_resources`: Determines behaviour in case a resource type
            from `considered_resources` is not found: If `True`, then missing
            resource types are set to 0. If, for example, a layer has no `FF` estimate
            in either FINN or HLS estimation, FF: 0 is entered for these layers. If set
            to `False`, an error is raised instead.
    """
    estimates: ResourceEstimates = res_estimation(model, fpga_part)
    hls_estimates: ResourceEstimates = hls_synth_res_estimation(model)
    result: ResourceEstimates = _merge_resource_estimations(estimates, hls_estimates)
    resource_missing = False

    # Check if resource types are missing (and add them)
    for layer in result:
        for restype in considered_resources:
            if restype not in result[layer]:
                if add_missing_resources:
                    result[layer][restype] = 0
                    log.debug(f"Added missing resource estimation on layer {layer} ({restype}: 0)")
                else:
                    resource_missing = True
                    log.error(f"Node {layer} has no resource estimation for resource {restype}!")

    # Check that estimates for all layers are available
    layer_missing = False
    for node in model.graph.node:
        if node.name not in result:
            layer_missing = True
            # TODO: Move this out of the function? (Should this better be checked by the caller?)
            log.error(f"No resource estimations were found for node {node.name}!")
    if layer_missing or resource_missing:
        raise FINNUserError(
            "At least one node is missing one or more resource estimation numbers.\n" + str(result)
        )
    return result


def _resources_per_device_per_slr(p: Platform) -> dict[int, dict[str, int]]:
    """Return the available resources as given by FINN platforms as a
    dictionary instead of nested lists. First by SLR, then by resource name.
    """
    assert p is not None
    assert p.compute_resources is not None
    res = p.compute_resources
    new = {}
    for slr in range(len(res)):
        new[slr] = {}
        # TODO: As soon as platforms.py uses dicts instead of lists this can be removed
        for i, name in enumerate(PLATFORM_RESOURCE_TYPES):
            new[slr][name] = res[slr][i]
    return new


def available_resources_on_platform(
    board: str, considered_resources: list[str] | None
) -> dict[str, int]:
    """Return the total resources of a single device of the given board (summed over its SLRs)
    for all considered resource types. Raises an error if a considered resource type does not
    exist on the board, since callers would otherwise divide by the zero available resources of
    that type. If `considered_resources` is None, all resource types are returned (with 0 for
    the ones that do not exist on the board) without raising.
    """
    resources_per_slr = _resources_per_device_per_slr(platforms[board]())
    available = {
        restype: sum(resources_per_slr[slr][restype] for slr in resources_per_slr)
        for restype in PLATFORM_RESOURCE_TYPES
    }
    if considered_resources is None:
        return available
    missing = [res for res in considered_resources if available.get(res, 0) <= 0]
    if len(missing) > 0:
        existing = {res: amount for res, amount in available.items() if amount > 0}
        raise FINNConfigurationError(
            f"The resource type(s) {', '.join(missing)} do not exist on board {board}. Resource "
            f"types available on this board: {existing}. For Multi-FPGA partitioning, remove "
            f"the missing resource type(s) from 'considered_resources'."
        )
    return {res: available[res] for res in considered_resources}


def get_model_device_resource_factors(
    model: ModelWrapper,
    board: str,
    fpga_part: str,
    considered_resources: list[str],
    max_utilization: float,
) -> dict[str, float] | None:
    """For the given model and resources, return how many devices are required
    per resource to fit the model. For example, {"LUT": 2.4} would mean that
    2.4 of the given devices are required to fit the model (lower bound).

    In case that the model estimate does not have the requested resource, None is returned.
    If the device does not have it, an error is raised.
    """
    estimated = get_estimated_model_resources(model, fpga_part, considered_resources, True)
    total_estimated = dict.fromkeys(considered_resources, 0)
    for resource in estimated.values():
        for restype in considered_resources:
            if restype not in resource:
                return None
            total_estimated[restype] += resource[restype]
    device = available_resources_on_platform(board, considered_resources)
    return {
        restype: total_estimated[restype] / (device[restype] * max_utilization)
        for restype in considered_resources
    }

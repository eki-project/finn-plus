"""Per-placement-unit resource sums for the automatic floorplanner.

This is the pre-synthesis area input to both levels of the floorplanner: Level-1 SLR
assignment constrains the sum of a unit's resources against a per-SLR budget, and
Level-2 packing sizes a unit's pblock from the same numbers.

Deliberately built on the live `res_estimation` analysis pass (which aggregates each
node's `node_res_estimation()`) and, optionally, on the merged FINN + Vitis-HLS
estimates from `finn.util.resources`. It is explicitly *not* built on
`finn.analysis.fpgadataflow.partitioning.res_estimation_complete`, which is orphaned
dead code and hardcodes FF usage to zero.
"""

import numpy as np
from functools import partial
from math import ceil
from numpy.typing import NDArray
from qonnx.core.modelwrapper import ModelWrapper

from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log
from finn.util.resources import ResourceEstimates, get_estimated_model_resources

#: Resource types the floorplanner constrains by default.
#:
#: FF is deliberately absent: FINN's analytical cost models (`node_res_estimation()`)
#: provide no FF estimate at all, so constraining FF against a per-SLR budget would
#: compare a budget against a column of zeros. FF numbers only exist once Vitis HLS
#: has run, i.e. with `use_hls_reports=True`; pass FF explicitly in that case.
DEFAULT_FLOORPLAN_RESOURCES: tuple[str, ...] = ("LUT", "BRAM_18K", "URAM", "DSP")

#: Index of each resource type in `Platform.compute_resources`/`guide_resources` rows.
PLATFORM_RESOURCE_ORDER: tuple[str, ...] = ("LUT", "FF", "BRAM_18K", "URAM", "DSP")

#: Default headroom applied on top of the estimate when sizing a placement unit,
#: matching the 1.1x convention already used elsewhere in the compiler.
DEFAULT_RESOURCE_MARGIN: float = 1.1

#: Default fraction of a pblock's SLICEM LUTs that LUTRAM and shift registers may fill;
#: see `DataflowBuildConfig.floorplan_slicem_utilization` for why it exists.
DEFAULT_SLICEM_UTILIZATION: float = 0.8

#: Default fraction of a pblock's slices that plain logic may fill; see
#: `DataflowBuildConfig.floorplan_logic_utilization`.
DEFAULT_LOGIC_UTILIZATION: float = 0.75


def get_node_resource_estimates(
    model: ModelWrapper,
    fpga_part: str,
    resource_types: tuple[str, ...] | list[str] = DEFAULT_FLOORPLAN_RESOURCES,
    use_hls_reports: bool = False,
) -> ResourceEstimates:
    """Return a per-node resource estimate restricted to `resource_types`.

    Args:
        model: The model to estimate. All nodes must have unique names.
        fpga_part: FPGA part identifier, needed by the DSP cost models.
        resource_types: Resource types to keep. Types a node has no estimate for are
            reported as 0.
        use_hls_reports: When True, merge in the post-HLS-synthesis numbers from the
            Vitis HLS reports (taking the larger of the two per resource type). Only
            meaningful once `HLSSynthIP` has run; it warns per node otherwise.

    Returns:
        Maps node name to a resource dict holding exactly `resource_types`. Nodes
        without a hardware backend (no HLS/RTL variant) are absent.
    """
    if use_hls_reports:
        estimates = get_estimated_model_resources(model, fpga_part, list(resource_types), True)
    else:
        estimates = model.analysis(partial(res_estimation, fpgapart=fpga_part))
    return {
        name: {restype: res.get(restype, 0) for restype in resource_types}
        for name, res in estimates.items()
    }


def sum_group_resources(
    node_resources: ResourceEstimates,
    groups: dict[int, list[str]],
    resource_types: tuple[str, ...] | list[str] = DEFAULT_FLOORPLAN_RESOURCES,
    margin: float = DEFAULT_RESOURCE_MARGIN,
) -> dict[int, dict[str, int]]:
    """Sum per-node estimates into per-placement-unit demands, with margin applied.

    Args:
        node_resources: Per-node estimates, as returned by `get_node_resource_estimates`.
        groups: Maps placement unit id to the node names it contains.
        resource_types: Resource types to sum.
        margin: Multiplicative headroom applied to each summed value. Must be >= 1.0;
            the result is rounded up to the next integer.

    Returns:
        Maps placement unit id to its summed, margin-applied resource demand.

    Raises:
        FINNUserError: If `margin` is below 1.0, which would size units below their
            own estimate.
        FINNInternalError: If a group references a node with no resource estimate,
            which means grouping and estimation disagree about the node set.
    """
    if margin < 1.0:
        raise FINNUserError(
            f"Floorplan resource margin must be >= 1.0 to leave headroom, got {margin}."
        )
    summed: dict[int, dict[str, int]] = {}
    for group_id, node_names in groups.items():
        totals = dict.fromkeys(resource_types, 0.0)
        for node_name in node_names:
            if node_name not in node_resources:
                raise FINNInternalError(
                    f"Placement unit {group_id} contains node {node_name}, which has no "
                    "resource estimate. Grouping and resource estimation disagree about "
                    "which nodes exist."
                )
            for restype in resource_types:
                totals[restype] += node_resources[node_name][restype]
        summed[group_id] = {restype: ceil(totals[restype] * margin) for restype in resource_types}
    return summed


def platform_slr_capacity(
    guide_resources: list[list[int]],
    res_limits: NDArray[np.float64] | list[float],
    resource_types: tuple[str, ...] | list[str] = DEFAULT_FLOORPLAN_RESOURCES,
) -> dict[int, dict[str, int]]:
    """Turn a platform's per-SLR resource rows into a usable per-SLR budget.

    `Platform.guide_resources` already nets out the memory-controller and network
    infrastructure cost per SLR; `Platform.res_limits` holds AMD's recommended
    per-resource-type utilization ceilings for timing closure. The budget the
    floorplanner packs against is the product of the two.

    Args:
        guide_resources: Per-SLR rows in `PLATFORM_RESOURCE_ORDER` order, i.e.
            `Platform.guide_resources`.
        res_limits: Per-resource-type utilization limits in the same order, i.e.
            `Platform.res_limits`.
        resource_types: Resource types to report.

    Returns:
        Maps SLR index to its usable budget per resource type.

    Raises:
        FINNInternalError: If a requested resource type is not one the platform
            model knows about.
    """
    capacity: dict[int, dict[str, int]] = {}
    for restype in resource_types:
        if restype not in PLATFORM_RESOURCE_ORDER:
            raise FINNInternalError(
                f"Resource type {restype} is not modeled by finn.util.platforms "
                f"(known types: {', '.join(PLATFORM_RESOURCE_ORDER)})."
            )
    for slr, row in enumerate(guide_resources):
        capacity[slr] = {}
        for restype in resource_types:
            idx = PLATFORM_RESOURCE_ORDER.index(restype)
            budget = int(row[idx] * res_limits[idx])
            if budget <= 0:
                log.debug(
                    f"SLR {slr} has no usable {restype} budget "
                    f"({row[idx]} available, limit {res_limits[idx]})."
                )
            capacity[slr][restype] = max(budget, 0)
    return capacity

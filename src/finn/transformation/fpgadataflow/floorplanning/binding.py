"""Collect the placement units a Vivado run should bind pblocks to (Phase B).

Phase A decided the floorplan and recorded it in nodeattrs. This module turns that
decision into the concrete unit list the Vivado-side geometry engine consumes, and
records the resulting pblock name back onto the nodes.

It has to look inside `StreamingDataflowPartition` children as well as the top
graph: by the time a build reaches the point where cells exist to constrain, the
per-layer model has usually been re-partitioned into SDP kernels, leaving the
floorplan attrs in the child models rather than the one in hand.
"""

from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from typing import cast

from finn.transformation.fpgadataflow.floorplanning.grouping import (
    UNASSIGNED_GROUP,
    is_placeable_node,
)
from finn.transformation.fpgadataflow.floorplanning.resources import (
    DEFAULT_FLOORPLAN_RESOURCES,
    DEFAULT_RESOURCE_MARGIN,
    get_node_resource_estimates,
    sum_group_resources,
)
from finn.transformation.fpgadataflow.floorplanning.tcl import (
    FloorplanUnitSpec,
    demands_from_estimate,
)
from finn.util.fpgadataflow import get_submodel
from finn.util.logging import log

#: Prefix of every pblock the automatic floorplanner creates. Distinct from the
#: DFX flow's `pblock_Hier_*` names so the two can coexist in one project.
PBLOCK_PREFIX: str = "pblock_finn_unit_"

#: Value of `floorplan_slr` for a node the floorplanner has not assigned.
UNASSIGNED_SLR: int = -1


def pblock_name_for_unit(unit_id: int) -> str:
    """Return the pblock name used for a placement unit."""
    return f"{PBLOCK_PREFIX}{unit_id}"


def iter_floorplanned_models(model: ModelWrapper) -> list[ModelWrapper]:
    """Return the model plus every SDP child model that carries floorplan attrs.

    Only one level of nesting is walked, which is what the Zynq and Alveo flows
    produce: a parent graph of SDP nodes, each wrapping one flat layer graph.
    """
    models = [model]
    for node in model.graph.node:
        if node.op_type != "StreamingDataflowPartition":
            continue
        try:
            submodel, _ = get_submodel(node)
        except Exception as error:
            log.warning(f"Floorplanner: cannot read the submodel of {node.name}: {error}")
            continue
        models.append(submodel)
    return models


def _units_in_model(model: ModelWrapper) -> dict[int, list[NodeProto]]:
    """Group a model's hardware nodes by `floorplan_group_id`, in graph order.

    A placeable node without a group -- one inserted after the floorplan was
    assigned, such as the FIFO `InsertFIFO` adds to a partition's boundary while
    the kernels are built -- gets a unit of its own, placed in graph order among
    the others, and the SLR every assigned node of the model shares. Left
    unconstrained, two such FIFOs of mobilenetv1 went wherever the placer liked:
    one ended 565 tiles from the layer feeding it, the worst path of the kernel
    after the reset (13 Oct 2026). A model whose assigned nodes span several SLRs
    cannot say where such a node belongs; it stays unconstrained, with a warning.
    """
    units: dict[int, list[NodeProto]] = {}
    late: list[NodeProto] = []
    for node in model.graph.node:
        if not is_placeable_node(node):
            continue
        inst = getCustomOp(node)
        group_id = cast("int", inst.get_nodeattr("floorplan_group_id"))
        if group_id == UNASSIGNED_GROUP:
            late.append(node)
            continue
        units.setdefault(group_id, []).append(node)
    if not late or not units:
        return units
    slrs = {
        cast("int", getCustomOp(n).get_nodeattr("floorplan_slr"))
        for nodes in units.values()
        for n in nodes
    } - {UNASSIGNED_SLR}
    if len(slrs) != 1:
        log.warning(
            f"Floorplanner: {len(late)} node(s) without a placement unit stay unconstrained; "
            f"the model's units span SLRs {sorted(slrs)}: {[n.name for n in late]}"
        )
        return units
    slr = slrs.pop()
    next_id = max(units) + 1
    for node in late:
        inst = getCustomOp(node)
        inst.set_nodeattr("floorplan_group_id", next_id)
        inst.set_nodeattr("floorplan_slr", slr)
        units[next_id] = [node]
        next_id += 1
    log.info(
        f"Floorplanner: {len(late)} node(s) inserted after the floorplan get units of "
        f"their own in SLR {slr}: {[n.name for n in late]}"
    )
    # Graph order, so the packer's dataflow order and the unit numbering follow it.
    position = {n.name: i for i, n in enumerate(model.graph.node)}
    return dict(sorted(units.items(), key=lambda kv: min(position[n.name] for n in kv[1])))


def collect_floorplan_units(
    model: ModelWrapper,
    fpga_part: str,
    resource_types: tuple[str, ...] | list[str] = DEFAULT_FLOORPLAN_RESOURCES,
    margin: float = DEFAULT_RESOURCE_MARGIN,
    use_hls_reports: bool = False,
    record_pblock_names: bool = True,
) -> list[FloorplanUnitSpec]:
    """Turn a floorplanned model into the unit list the Vivado engine takes.

    Args:
        model: A model `step_generate_floorplan_assignment` has run over, or whose
            SDP children it has run over.
        fpga_part: FPGA part identifier, needed by the DSP cost models.
        resource_types: Resource types to size units by.
        margin: Headroom applied to each unit's estimate.
        use_hls_reports: Whether to merge Vitis HLS synthesis numbers in.
        record_pblock_names: Whether to write the chosen pblock name back onto each
            node's `floorplan_pblock` attr. Turn off for a dry run.

    Returns:
        One spec per placement unit, ordered by unit id -- which is dataflow order,
        the order the packer's serpentine walk depends on. Empty when the model
        carries no floorplan, so a caller can act on that without a separate check.

    Note:
        Cells are named by their FINN node name rather than a hierarchical path.
        Where Vivado puts a node depends on how the design was stitched, packaged
        and instantiated, and differs between flows; the Tcl side resolves the name
        against the elaborated hierarchy instead.
    """
    specs: list[FloorplanUnitSpec] = []
    for submodel in iter_floorplanned_models(model):
        units = _units_in_model(submodel)
        if not units:
            continue
        estimates = get_node_resource_estimates(
            submodel, fpga_part, resource_types, use_hls_reports
        )
        summed = sum_group_resources(
            estimates,
            {unit_id: [n.name for n in nodes] for unit_id, nodes in units.items()},
            resource_types,
            margin,
        )
        for unit_id, nodes in units.items():
            slrs = {cast("int", getCustomOp(n).get_nodeattr("floorplan_slr")) for n in nodes}
            if slrs == {UNASSIGNED_SLR}:
                log.warning(
                    f"Floorplanner: placement unit {unit_id} has no SLR assignment; "
                    "not creating a pblock for it."
                )
                continue
            slrs.discard(UNASSIGNED_SLR)
            if len(slrs) > 1:
                # Grouping runs before assignment precisely so this cannot happen.
                log.warning(
                    f"Floorplanner: placement unit {unit_id} spans SLRs {sorted(slrs)}; "
                    f"binding it to SLR {min(slrs)}."
                )
            name = pblock_name_for_unit(len(specs))
            specs.append(
                FloorplanUnitSpec(
                    name=name,
                    slr=min(slrs),
                    demands=demands_from_estimate(summed[unit_id]),
                    cells=[n.name for n in nodes],
                )
            )
            if record_pblock_names:
                for node in nodes:
                    getCustomOp(node).set_nodeattr("floorplan_pblock", name)

    if specs:
        log.info(f"Floorplanner: {len(specs)} placement unit(s) ready for pblock binding")
    return specs

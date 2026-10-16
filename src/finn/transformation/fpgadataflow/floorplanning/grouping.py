"""Level-2 placement unit computation for the automatic floorplanner.

A *placement unit* is the atomic object the floorplanner places: it gets exactly one
SLR assignment (Level 1) and exactly one pblock (Level 2). This module computes those
units and records them in the `floorplan_group_id` nodeattr.

Grouping runs **before** Level-1 SLR assignment, not after. That ordering is what
guarantees a unit can never straddle two SLRs (and, on Alveo, never straddle two
independently elaborated Vitis kernel sessions, which the post-system-link hook has no
way to express). It also means an anchored node drags the glue logic grouped with it to
the same anchored SLR, instead of leaving that glue to an independent assignment.

Group ids are dense and assigned in topological order, so iterating units by ascending
id walks the dataflow graph front to back. The Level-2 packer consumes that order and
decides the geometric (serpentine) traversal itself; this module only fixes the
sequence, not the geometry.
"""

from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from typing import cast

from finn.builder.build_dataflow_config import FloorplanGranularity
from finn.util.exception import FINNInternalError
from finn.util.fpgadataflow import get_device_id, is_hls_node, is_rtl_node
from finn.util.logging import log

#: Op-type prefixes treated as "glue" logic under `NODE_WITH_ADJACENT_GLUE`: small,
#: never externally anchored nodes that exist only to connect two compute layers.
FIFO_OP_PREFIX: str = "StreamingFIFO"
DWC_OP_PREFIX: str = "StreamingDataWidthConverter"

#: Value of `floorplan_group_id` for a node that is not part of any placement unit.
UNASSIGNED_GROUP: int = -1


def is_placeable_node(node: NodeProto) -> bool:
    """Return whether a node is something the floorplanner can place.

    Narrower than `is_fpgadataflow_node`, deliberately. That also matches abstract
    nodes -- a `StreamingDataflowPartition`, or an `MVAU` that `SpecializeLayers`
    has not turned into `MVAU_hls`/`MVAU_rtl` yet. Neither has an area: the
    resource-estimation passes only cover nodes with an HLS or RTL backend, so
    grouping one would produce a placement unit that cannot be sized.
    """
    return is_hls_node(node) or is_rtl_node(node)


def _is_glue_node(node: NodeProto, group_dwcs: bool) -> bool:
    """Return whether a node is glue logic eligible for merging into a neighbour's unit."""
    if node.op_type.startswith(FIFO_OP_PREFIX):
        return True
    return group_dwcs and node.op_type.startswith(DWC_OP_PREFIX)


def _can_merge(node: NodeProto, neighbour: NodeProto | None) -> bool:
    """Return whether `node` may join `neighbour`'s placement unit.

    A merge is only allowed between two hardware nodes on the same device. The
    `device_id` check is a hard constraint enforced here, before merging, rather than
    left for the (per-device) Level-1 assignment to discover afterwards: a neighbour on
    another device is a physically separate chip, not just a different die region, so
    adjacency in the dataflow graph does not make it eligible.
    """
    if neighbour is None or not is_placeable_node(neighbour):
        return False
    return get_device_id(node) == get_device_id(neighbour)


def compute_placement_units(
    model: ModelWrapper,
    granularity: FloorplanGranularity = FloorplanGranularity.NODE_WITH_ADJACENT_GLUE,
    group_dwcs: bool = True,
) -> dict[int, list[str]]:
    """Compute the Level-2 placement units of a model.

    Args:
        model: The model to group. Nodes must have unique names, and the graph must be
            in topological order (which ONNX guarantees).
        granularity: `PER_NODE` gives every hardware node its own unit.
            `NODE_WITH_ADJACENT_GLUE` merges each FIFO -- and, when `group_dwcs` is set,
            each DataWidthConverter -- into the unit of the neighbour it is attached to.
        group_dwcs: Whether DWCs count as glue. Only meaningful for
            `NODE_WITH_ADJACENT_GLUE`. On by default: DWCs sit at essentially every
            width-changing boundary, so excluding them would undercut the mode's purpose
            of capping pblock count. Configurable off for A/B testing.

    Returns:
        Maps placement unit id to the node names in it, with ids dense and ascending in
        topological order. Nodes without a hardware backend are not part of any unit and
        do not appear.

    Raises:
        FINNInternalError: If the model contains duplicate node names, which would make
            the returned mapping ambiguous.
    """
    hw_nodes = [node for node in model.graph.node if is_placeable_node(node)]
    names = [node.name for node in hw_nodes]
    if len(set(names)) != len(names):
        raise FINNInternalError(
            "Cannot compute floorplan placement units: node names are not unique. "
            "Run GiveUniqueNodeNames first."
        )
    skipped = len(model.graph.node) - len(hw_nodes)
    if skipped:
        log.debug(
            f"Floorplan grouping skipped {skipped} node(s) with no HLS or RTL backend "
            "(abstract or structural nodes have no area to place)."
        )

    group_of: dict[str, int] = {}
    next_group = 0

    if granularity == FloorplanGranularity.PER_NODE:
        for node in hw_nodes:
            group_of[node.name] = next_group
            next_group += 1
    elif granularity == FloorplanGranularity.NODE_WITH_ADJACENT_GLUE:
        glue_nodes = []
        # First pass: every non-glue node opens its own unit, walking the graph front to
        # back so that a glue node's producer is already assigned when we reach it.
        for node in hw_nodes:
            if _is_glue_node(node, group_dwcs):
                glue_nodes.append(node)
                continue
            group_of[node.name] = next_group
            next_group += 1
        # Second pass: attach glue to its producer's unit. Chains of glue resolve left to
        # right, so a FIFO -> DWC -> FIFO run ends up in the unit of the compute node
        # feeding it.
        deferred = []
        for node in glue_nodes:
            producer = model.find_producer(node.input[0])
            if _can_merge(node, producer) and cast("NodeProto", producer).name in group_of:
                group_of[node.name] = group_of[cast("NodeProto", producer).name]
            else:
                deferred.append(node)
        # Third pass: glue with no usable producer (a graph input, or a producer on
        # another device) attaches to its consumer instead. Walk back to front so chains
        # resolve right to left here.
        for node in reversed(deferred):
            consumer = model.find_consumer(node.output[0])
            if _can_merge(node, consumer) and cast("NodeProto", consumer).name in group_of:
                group_of[node.name] = group_of[cast("NodeProto", consumer).name]
            else:
                # Isolated glue (no eligible neighbour on either side) stays its own unit.
                group_of[node.name] = next_group
                next_group += 1
    else:
        raise FINNInternalError(f"Unknown floorplan granularity: {granularity}")

    # Renumber so ids are dense and ascend with the topological position of each unit's
    # earliest member, which is the order the Level-2 packer walks units in.
    first_seen: dict[int, int] = {}
    for position, node in enumerate(hw_nodes):
        first_seen.setdefault(group_of[node.name], position)
    renumbered = {
        old: new for new, old in enumerate(sorted(first_seen, key=lambda g: first_seen[g]))
    }
    units: dict[int, list[str]] = {new: [] for new in renumbered.values()}
    for node in hw_nodes:
        units[renumbered[group_of[node.name]]].append(node.name)
    return units


def get_placement_units(model: ModelWrapper) -> dict[int, list[str]]:
    """Read placement units back from the `floorplan_group_id` nodeattrs.

    Args:
        model: A model that `ComputeFloorplanGroups` has already been applied to.

    Returns:
        Maps placement unit id to the node names in it, ids ascending.

    Raises:
        FINNInternalError: If a hardware node has no placement unit, i.e.
            `ComputeFloorplanGroups` has not run (or ran before nodes were added).
    """
    units: dict[int, list[str]] = {}
    for node in model.graph.node:
        if not is_placeable_node(node):
            continue
        group_id = cast("int", getCustomOp(node).get_nodeattr("floorplan_group_id"))
        if group_id == UNASSIGNED_GROUP:
            raise FINNInternalError(
                f"Node {node.name} has no floorplan_group_id. ComputeFloorplanGroups must "
                "run before any pass that consumes placement units."
            )
        units.setdefault(group_id, []).append(node.name)
    return dict(sorted(units.items()))


class ComputeFloorplanGroups(Transformation):
    """Compute Level-2 placement units and record them in `floorplan_group_id`.

    Preconditions:
        - Node names are unique (`GiveUniqueNodeNames`).
        - The graph shape is final: layers are specialized and all FIFOs/DWCs/IODMAs
          that the design will have have been inserted. Running earlier groups a node
          set that no longer matches the one that gets placed.
        - Multi-FPGA device partitioning, if used, has already run, so `device_id` is
          final. Grouping reads it to refuse merges across a device boundary.

    Postconditions:
        - Every hardware node carries a `floorplan_group_id` in `[0, n_units)`.
        - Ids are dense and ascend in topological order; the Level-2 packer walks units
          in that order.
        - No unit spans more than one `device_id`, so per-device Level-1 SLR assignment
          can treat each unit as one atomic variable.
        - Nodes without a hardware backend keep `floorplan_group_id == -1`.

    This transformation computes the grouping from scratch on every run and overwrites
    any previous value, so changing the granularity setting and re-running is safe.
    """

    def __init__(
        self,
        granularity: FloorplanGranularity = FloorplanGranularity.NODE_WITH_ADJACENT_GLUE,
        group_dwcs: bool = True,
    ) -> None:
        """Initialize the transformation.

        Args:
            granularity: Which granularity mode to group at, see
                `compute_placement_units`.
            group_dwcs: Whether DWCs count as glue under `NODE_WITH_ADJACENT_GLUE`.
        """
        super().__init__()
        self.granularity = granularity
        self.group_dwcs = group_dwcs

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Assign a placement unit to every hardware node in the model."""
        units = compute_placement_units(model, self.granularity, self.group_dwcs)
        group_of = {name: group_id for group_id, names in units.items() for name in names}
        for node in model.graph.node:
            if not is_placeable_node(node):
                continue
            getCustomOp(node).set_nodeattr("floorplan_group_id", group_of[node.name])
        log.info(
            f"Floorplanner: {len(group_of)} node(s) grouped into {len(units)} placement "
            f"unit(s) at {self.granularity.value} granularity."
        )
        return (model, False)

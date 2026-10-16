"""Bridge the floorplanner's SLR assignment onto the legacy `slr` nodeattr.

The floorplanner keeps its own `floorplan_slr` attr rather than writing `slr`
directly, so its output can be inspected and A/B-tested without disturbing the
passes that already consume `slr` (the `Floorplan` transform's DWC/FIFO
SLR-crossing logic, multi-FPGA SDP clustering, Vitis SLR pinning). This module is
the one explicit place where the two meet.

Copying is normally what a build wants, for two reasons:

- `Floorplan` inserts SLR-crossing-safe DWCs and FIFOs off the `slr` attr. Without
  the copy, nothing pipelines the SLR boundaries the floorplanner just created,
  and a WNS measurement of the result says little about the floorplan itself.
- On Alveo the copy is load-bearing rather than merely useful: `step_prepare_synthesis`
  creates one Vitis kernel per `Floorplan`-derived `partition_id`, so without it the
  floorplanner has no effect on kernel splitting at all.
"""

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from typing import cast

from finn.transformation.fpgadataflow.floorplanning.grouping import is_placeable_node
from finn.util.exception import FINNInternalError
from finn.util.logging import log

#: Value of `floorplan_slr` for a node the floorplanner has not assigned.
UNASSIGNED_SLR: int = -1


class BridgeFloorplanToSLR(Transformation):
    """Copy `floorplan_slr` onto the legacy `slr` nodeattr.

    Preconditions:
        - `AssignFloorplanSLRs` has run, so hardware nodes carry a `floorplan_slr`.

    Postconditions:
        - Every hardware node with a `floorplan_slr` has the same value in `slr`.
        - `floorplan_slr` is left in place, so the floorplanner's own decision stays
          readable after `Floorplan` has rewritten `slr` and `partition_id`.

    Run `Floorplan` after this, not before: it reads `slr` to decide DWC/FIFO
    placement and to derive `partition_id`, and would otherwise see the default
    values.
    """

    def __init__(self, overwrite_existing: bool = True) -> None:
        """Initialize the transformation.

        Args:
            overwrite_existing: Whether to overwrite an `slr` value that is already
                set to something other than the unassigned default. False keeps a
                pre-existing assignment (from a user's floorplan JSON, say) and only
                fills in the nodes that have none.
        """
        super().__init__()
        self.overwrite_existing = overwrite_existing

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Copy the floorplanner's SLR choice onto the legacy attr."""
        copied = 0
        kept = 0
        unassigned = 0
        for node in model.graph.node:
            if not is_placeable_node(node):
                continue
            inst = getCustomOp(node)
            floorplan_slr = cast("int", inst.get_nodeattr("floorplan_slr"))
            if floorplan_slr == UNASSIGNED_SLR:
                unassigned += 1
                continue
            if not self.overwrite_existing and inst.get_nodeattr("slr") != UNASSIGNED_SLR:
                kept += 1
                continue
            inst.set_nodeattr("slr", floorplan_slr)
            copied += 1

        if copied == 0 and unassigned > 0:
            raise FINNInternalError(
                f"BridgeFloorplanToSLR found no assigned node ({unassigned} unassigned). "
                "AssignFloorplanSLRs must run before the bridge."
            )
        message = f"Floorplanner: bridged {copied} node(s) from floorplan_slr to slr"
        if kept:
            message += f", kept {kept} existing assignment(s)"
        if unassigned:
            message += f", {unassigned} node(s) had no assignment"
        log.info(message)
        return (model, False)

"""SLR anchor constraints for externally connected nodes.

Not every node is freely placeable. A node with a direct AXI-MM port to HBM, DDR or
host memory has to land in an SLR that actually reaches that resource, and a node that
feeds or drains an off-board link should sit in the SLR holding the transceivers. This
module derives those constraints from the platform model and the graph, so that Level-1
SLR assignment can treat them as hard constraints rather than objective terms.

The rule for external memory is deliberately mechanism-based rather than a
hand-maintained op-type list: any node whose `get_verilog_top_module_intf_names()`
reports a non-empty `aximm` entry is externally memory connected. That is the same API
the legacy `Floorplan` transform already queries for its `axilite`/`ap_none` checks, and
it means a future op that grows an AXI-MM port is anchored automatically instead of
being silently missed.
"""

from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.builder.build_dataflow_config import FpgaMemoryType
from finn.transformation.fpgadataflow.floorplanning.grouping import is_placeable_node
from finn.util.exception import FINNUserError
from finn.util.fpgadataflow import get_device_id
from finn.util.logging import log
from finn.util.platforms import Platform

#: Boards whose external memory is HBM rather than DDR. Kept consistent with the board
#: switch in `BuildBasicVitisLinkConfig`, which decides the matching `add_sp` system
#: port string; the two must not drift apart.
HBM_BOARDS: frozenset[str] = frozenset({"u50", "u280", "u55c"})

#: The SLR indices a node may be placed in. A node with no entry in an anchor mapping
#: is unconstrained; an entry is always a non-empty list of allowed indices.
AnchorSet = list[int]


def node_has_external_memory_interface(node: NodeProto) -> bool:
    """Return whether a node drives its own AXI-MM port to external memory.

    Nodes that read external memory through a companion IODMA (external-weight MVAU/VVAU,
    for instance, whose weights arrive over AXI-Stream) deliberately do *not* match: the
    IODMA is the node that needs the anchor, and it gets one on its own.
    """
    return len(getCustomOp(node).get_verilog_top_module_intf_names().get("aximm", [])) > 0


def resolve_memory_anchor(
    platform: Platform,
    board: str | None,
    fpga_memory: FpgaMemoryType,
) -> AnchorSet:
    """Return the SLRs an externally memory connected node may be placed in.

    Args:
        platform: Platform model of the device the node sits on.
        board: Board name, used to decide HBM vs. DDR. May be None when only
            `fpga_part` was configured.
        fpga_memory: Build-wide memory choice. This is a global config flag rather than
            a per-node property, so it is read once for the whole model.

    Returns:
        The allowed SLR indices, or an empty list when the platform carries no usable
        external-memory location and the node should be left unanchored.
    """
    if fpga_memory == FpgaMemoryType.HOST_MEM:
        return [platform.get_pcie_slr()]
    if board is not None and board.lower() in HBM_BOARDS:
        if platform.hbm_slr >= 0:
            return [platform.hbm_slr]
        log.warning(
            f"Board {board} is an HBM board but its platform model records no hbm_slr; "
            "leaving externally memory connected nodes unanchored."
        )
        return []
    if platform.ddr_slr:
        # More than one DDR bank location is common (U250 has one per SLR). All of them
        # are valid, so the LP is free to pick whichever costs least.
        return list(platform.ddr_slr)
    if platform.hbm_slr >= 0:
        # Unknown board name, but the platform model does know where HBM lives.
        return [platform.hbm_slr]
    log.warning(
        "No external memory location is recorded for this platform; leaving externally "
        "memory connected nodes unanchored."
    )
    return []


def _qsfp_slr_index(platform: Platform) -> int | None:
    """Return the SLR index holding the QSFP transceivers, or None if not modeled.

    `Platform.qsfp_slr` is stored as a Vivado-style name ("SLR1") rather than an index,
    matching what `VitisLinkConfiguration.add_slr` consumes.
    """
    if platform.qsfp_slr is None:
        return None
    name = platform.qsfp_slr.strip().upper()
    if not name.startswith("SLR") or not name[3:].isdigit():
        log.warning(f"Cannot interpret qsfp_slr value {platform.qsfp_slr!r} as an SLR index.")
        return None
    return int(name[3:])


def _link_facing_nodes(model: ModelWrapper) -> set[str]:
    """Return the nodes on either side of a device boundary.

    In a multi-FPGA build these are exactly the nodes that will end up feeding or
    draining a communication (Aurora) kernel once those kernels are inserted, which is
    what makes them worth pinning to the transceiver SLR. In a single-device build there
    are no such edges and the result is empty.
    """
    link_facing: set[str] = set()
    for node in model.graph.node:
        if not is_placeable_node(node):
            continue
        successors = model.find_direct_successors(node) or []
        for successor in successors:
            if not is_placeable_node(successor):
                continue
            if get_device_id(node) != get_device_id(successor):
                link_facing.add(node.name)
                link_facing.add(successor.name)
    return link_facing


def derive_slr_anchors(
    model: ModelWrapper,
    platform: Platform,
    board: str | None = None,
    fpga_memory: FpgaMemoryType = FpgaMemoryType.DEFAULT,
    anchor_link_neighbours: bool = True,
) -> dict[str, AnchorSet]:
    """Derive per-node SLR anchor constraints for a model.

    Args:
        model: The model to analyse, with its graph shape final.
        platform: Platform model of the target device. Anchors are resolved against it,
            so in a multi-FPGA build it must be the platform of the device the node is
            on (see `AssignFloorplanSLRs`, which solves one problem per device).
        board: Board name, used to decide HBM vs. DDR.
        fpga_memory: Build-wide memory choice.
        anchor_link_neighbours: Whether to pin the nodes on either side of a
            device-to-device boundary to the transceiver SLR. The link kernel itself is
            already pinned elsewhere, by `AddAuroraToLinkConfig`; this extends the same
            rationale one hop into the FINN graph.

    Returns:
        Maps node name to the SLR indices it may be placed in. Nodes with no constraint
        are absent from the mapping rather than mapped to "all SLRs".
    """
    if platform.nslr <= 1:
        # Single-SLR part: there is nothing to choose, so no anchor is meaningful and
        # the unconfirmed-PCIe-SLR fallback warning would be pure noise.
        return {}

    anchors: dict[str, AnchorSet] = {}
    memory_anchor = resolve_memory_anchor(platform, board, fpga_memory)
    for node in model.graph.node:
        if not is_placeable_node(node):
            continue
        if memory_anchor and node_has_external_memory_interface(node):
            anchors[node.name] = list(memory_anchor)

    if anchor_link_neighbours:
        qsfp_slr = _qsfp_slr_index(platform)
        link_facing = _link_facing_nodes(model)
        if link_facing and qsfp_slr is None:
            log.warning(
                "Model crosses a device boundary but the platform records no qsfp_slr; "
                "link-facing nodes will not be anchored to the transceiver SLR."
            )
        elif qsfp_slr is not None:
            for name in link_facing:
                existing = anchors.get(name)
                if existing is None:
                    anchors[name] = [qsfp_slr]
                elif qsfp_slr in existing:
                    # Both constraints agree: take the more restrictive one.
                    anchors[name] = [qsfp_slr]
                else:
                    # An external-memory anchor is physical connectivity and cannot be
                    # given up; pinning a link-facing node next to the transceivers is
                    # only an optimization, so the memory anchor wins.
                    log.warning(
                        f"Node {name} is both externally memory connected (SLR "
                        f"{existing}) and adjacent to a device boundary (transceivers "
                        f"in SLR {qsfp_slr}). Keeping the memory anchor; the link will "
                        "cross an SLR boundary."
                    )

    return anchors


def resolve_mem_port(
    platform: Platform,
    board: str | None,
    fpga_memory: FpgaMemoryType,
    slr: int,
    channel: int,
) -> str:
    """Return the Vitis system-port string an externally connected node should use.

    This is the value the floorplanner writes into the `mem_port` nodeattr, which
    `BuildBasicVitisLinkConfig` then reads instead of deriving a bank index itself.
    The floorplanner is the party that knows both the SLR the node landed in and
    which external resource it was anchored to, so it is the right place to decide.

    Args:
        platform: Platform model of the node's device.
        board: Board name, used to decide HBM vs. DDR.
        fpga_memory: Build-wide memory choice.
        slr: The SLR the node was assigned to.
        channel: A per-device running index, used where the choice of channel is
            free (HBM) rather than fixed by the SLR (DDR).

    Returns:
        A string such as `"HBM[3]"`, `"DDR[1]"` or `"HOST[0]"`, or `""` when the
        platform records no external memory to point at.
    """
    if fpga_memory == FpgaMemoryType.HOST_MEM:
        return "HOST[0]"
    if (board is not None and board.lower() in HBM_BOARDS) or (
        board is None and platform.hbm_slr >= 0
    ):
        # Every HBM pseudo-channel reaches the same SLR, so the channel index is a
        # free choice; what matters for placement is the anchor that already put the
        # node in that SLR.
        return f"HBM[{channel}]"
    if platform.ddr_slr:
        # DDR banks are per-SLR, so the node's own SLR picks the bank. Fall back to
        # the nearest modeled bank if it somehow landed elsewhere.
        bank = slr if slr in platform.ddr_slr else min(platform.ddr_slr, key=lambda b: abs(b - slr))
        return f"DDR[{bank}]"
    return ""


def _intersect(left: AnchorSet, right: AnchorSet) -> AnchorSet:
    """Return the intersection of two anchor sets, preserving the left-hand order."""
    return [slr for slr in left if slr in right]


def lift_anchors_to_units(
    anchors: dict[str, AnchorSet],
    units: dict[int, list[str]],
) -> dict[int, AnchorSet]:
    """Lift per-node anchors onto the placement units the nodes belong to.

    Level-1 assignment works on units, not raw nodes, so a unit's allowed SLR set is
    the intersection of its members' constraints. In practice at most one member of a
    unit is anchored (glue grouping only merges small, never-anchored FIFOs and DWCs
    into a neighbour), so this normally reduces to "the anchor of whichever member has
    one".

    Args:
        anchors: Per-node anchors from `derive_slr_anchors`.
        units: Placement units from `compute_placement_units`.

    Returns:
        Maps unit id to its allowed SLR indices. Unconstrained units are absent.

    Raises:
        FINNUserError: If two members of one unit carry disjoint anchors, which makes
            the unit unplaceable. This is the degenerate case of glue sitting directly
            between two differently anchored compute nodes.
    """
    lifted: dict[int, AnchorSet] = {}
    for unit_id, node_names in units.items():
        allowed: AnchorSet | None = None
        for name in node_names:
            node_anchor = anchors.get(name)
            if node_anchor is None:
                continue
            allowed = list(node_anchor) if allowed is None else _intersect(allowed, node_anchor)
            if not allowed:
                raise FINNUserError(
                    f"Placement unit {unit_id} (nodes: {', '.join(node_names)}) has members "
                    "anchored to disjoint sets of SLRs, so it cannot be placed anywhere. "
                    "Use PER_NODE floorplan granularity, or exclude DWCs from glue "
                    "grouping, to keep these nodes in separate placement units."
                )
        if allowed is not None:
            lifted[unit_id] = allowed
    return lifted

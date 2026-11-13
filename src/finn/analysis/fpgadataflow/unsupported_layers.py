"""Analysis for detecting unsupported layers in FINN dataflow models.

This module provides functionality to validate that FINN models have a valid
partition structure where FPGA-supported operations form a single contiguous
section in the dataflow graph.
"""

from collections import deque
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from onnx import NodeProto


def unsupported_layers(
    model: ModelWrapper,
) -> tuple[Literal[False], "NodeProto"] | tuple[Literal[True], None]:
    """Check if all sink nodes are only reachable by paths with at most one
    connected section of nodes which are supported by the FPGA.
    """

    def is_supported_node(node: "NodeProto") -> bool:
        """Check if a node is supported by (= mapped to) the FPGA backend."""
        return node.domain.startswith("finn.custom_op.fpgadataflow")

    # Find source and sink nodes in the model
    source_nodes: list[NodeProto] = []
    sink_nodes: list[NodeProto | None] = []

    inputs = model.graph.input
    for inp in inputs:
        # Nodes are not supported by python sets, so needing to do deduplication manually
        for n in model.find_consumers(inp.name):
            if n not in source_nodes:
                source_nodes.append(n)

    outputs = model.graph.output
    for out in outputs:
        n = model.find_producer(out.name)
        if n not in sink_nodes:
            sink_nodes.append(n)

    # BFS to check paths
    queue: deque[tuple[NodeProto, bool, bool]] = deque()
    # Track (node_id, in_green_section, has_seen_complete_green_section)
    visited = []

    # Initialize BFS with all source nodes
    for source in source_nodes:
        is_supported = is_supported_node(source)
        queue.append((source, is_supported, False))
        visited.append((source, is_supported, False))

    while queue:
        node, in_fpga_sec, seen_complete_fpga_section = queue.popleft()

        # Process all successors
        successors = model.find_direct_successors(node)
        if successors is not None:
            for successor in successors:
                sucessor_supported = is_supported_node(successor)
                fpga_section_end = seen_complete_fpga_section

                # If transitioning from FPGA to Host, we've completed a FPGA section
                if in_fpga_sec and not sucessor_supported:
                    fpga_section_end = True

                # If we've already seen a complete FPGA section and are about to start a new one
                if seen_complete_fpga_section and not in_fpga_sec and sucessor_supported:
                    # This path would create a second FPGA section
                    return False, node

                # Handle cycles
                state = (successor, sucessor_supported, fpga_section_end)
                if state not in visited:
                    visited.append(state)
                    queue.append((successor, sucessor_supported, fpga_section_end))

    return True, None


def unmapped_layer_report(model: ModelWrapper) -> list[dict[str, str]]:
    """Report every node that is not mapped to a hardware layer, i.e. whose domain does not
    start with ``finn.custom_op.fpgadataflow``, together with its position relative to the
    hardware layers.

    Returns one dict per unmapped node, in graph order, with the keys ``name``, ``op_type``,
    ``domain`` and ``position``. The position considers all paths through the node, where an
    ancestor (descendant) is any node from which the node can be reached (that can be reached
    from the node) along the edges of the graph:

    * ``"interior"``: at least one hardware layer is an ancestor and at least one hardware
      layer is a descendant of the node. The node lies on a path between two hardware layers,
      even if other paths through it do not, e.g. if it also feeds a graph output.
    * ``"start"``: no hardware layer is an ancestor of the node. This includes nodes that have
      no hardware layer on any path through them, e.g. in a branch that bypasses the
      accelerator or in a graph without hardware layers.
    * ``"end"``: at least one hardware layer is an ancestor, but none is a descendant.

    The ``"interior"`` nodes are exactly the nodes that prevent the hardware layers from being
    extracted into a single dataflow partition: ``unsupported_layers`` rejects a model if and
    only if this report contains an ``"interior"`` node (given that all hardware layers are
    reachable from the graph inputs, which is what ``unsupported_layers`` traverses).
    ``"start"`` and ``"end"`` nodes stay outside of the partition and have to be implemented
    separately, e.g. in software.
    """

    def is_supported_node(node: "NodeProto") -> bool:
        """Check if a node is supported by (= mapped to) the FPGA backend."""
        return node.domain.startswith("finn.custom_op.fpgadataflow")

    nodes = list(model.graph.node)
    # Direct successors and predecessors by node index, derived from the tensor names
    producer_of: dict[str, int] = {}
    for idx, node in enumerate(nodes):
        for output_name in node.output:
            producer_of[output_name] = idx
    successors: list[list[int]] = [[] for _ in nodes]
    predecessors: list[list[int]] = [[] for _ in nodes]
    for idx, node in enumerate(nodes):
        for input_name in node.input:
            producer = producer_of.get(input_name)
            if producer is not None:
                successors[producer].append(idx)
                predecessors[idx].append(producer)

    def reachable_from_supported(edges: list[list[int]]) -> set[int]:
        """Return the indices of all nodes reachable from a supported node along edges."""
        reached: set[int] = set()
        queue = deque(idx for idx, node in enumerate(nodes) if is_supported_node(node))
        while queue:
            for neighbor in edges[queue.popleft()]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
        return reached

    has_supported_ancestor = reachable_from_supported(successors)
    has_supported_descendant = reachable_from_supported(predecessors)

    report = []
    for idx, node in enumerate(nodes):
        if is_supported_node(node):
            continue
        if idx not in has_supported_ancestor:
            position = "start"
        elif idx in has_supported_descendant:
            position = "interior"
        else:
            position = "end"
        report.append(
            {
                "name": node.name,
                "op_type": node.op_type,
                "domain": node.domain,
                "position": position,
            }
        )
    return report

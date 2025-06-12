from collections import deque
from qonnx.core.modelwrapper import ModelWrapper


def unsupported_layers(model: ModelWrapper):
    """
    Check if all sink nodes are only reachable by paths with at most one
    connected section of nodes which are supported by the FPGA.
    """

    def is_supported_node(node):
        return node.domain == "finn.custom_op.fpgadataflow"

    # Find source and sink nodes in the model
    source_nodes = []
    sink_nodes = []

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
    queue = deque()
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

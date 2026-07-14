"""Helpers for manipulating ONNX Script IR graphs in FINN."""

import ast
from collections.abc import Iterable
from onnx_ir import _enums
from onnxscript import ir
from onnxscript.rewriter._pattern_ir import GraphPattern, NodeOutputPattern, ValuePattern
from onnxscript.rewriter._rewrite_rule import ReplacementPatternFunction, ReplacementSubgraph
from onnxscript.rewriter.pattern import (
    MatchResult,
    OpsetPatternBuilder,
    RewriterContext,
    pattern_builder,
)
from qonnx.custom_op.registry import is_custom_op
from typing import Literal, cast

from finn.util.exception import FINNInternalError


class SubGraphView(ir.GraphView):
    """Create a read-only view of a subgraph defined by a set of nodes.

    Args:
        graph (ir.Graph): The parent graph containing the nodes.
        name (str): Name of the subgraph.
        nodes (List[ir.Node]): List of nodes that make up the subgraph.
        include_initializers (bool): Whether to include initializers connected to the
            subgraph nodes as part of the subgraph.
    """

    def __init__(
        self, graph: ir.Graph, name: str, nodes: list[ir.Node], include_initializers: bool = False
    ) -> None:
        """Initialize a subgraph view over the selected nodes."""
        self._assert_graph_subset(graph, nodes)
        self.include_initializers = include_initializers
        super().__init__(
            name=name,
            inputs=self._identify_inputs(nodes),
            initializers=self._identify_initializers(nodes),
            outputs=self._identify_outputs(nodes),
            nodes=nodes,
        )

    def _assert_graph_subset(self, graph: ir.Graph, nodes: list[ir.Node]) -> None:
        """Validate that all nodes belong to the supplied graph."""
        for node in nodes:
            if node.graph != graph:
                raise FINNInternalError("All nodes must belong to the same graph")

    def _identify_inputs(self, nodes: list[ir.Node]) -> list[ir.Value]:
        """Return the external input values for the subgraph."""
        inputs = set()
        for node in nodes:
            for inp in node.inputs:
                if inp is not None and (inp.is_graph_input() or inp.producer() not in nodes):
                    inputs.add(inp)
        return list(inputs)

    def _identify_initializers(self, nodes: list[ir.Node]) -> list[ir.Value]:
        """Return initializers connected to the subgraph when enabled."""
        initializers = set()
        if self.include_initializers:
            for node in nodes:
                for inp in node.inputs:
                    if inp is not None and inp.is_initializer():
                        initializers.add(inp)
        return list(initializers)

    def _identify_outputs(self, nodes: list[ir.Node]) -> list[ir.Value]:
        """Return values that exit the subgraph boundary."""
        outputs = set()
        for node in nodes:
            for output in node.outputs:
                if output.is_graph_output():
                    outputs.add(output)
                else:
                    for consumer in output.consumers():
                        if consumer not in nodes:
                            outputs.add(output)
        return list(outputs)


class PytorchMetadataNode:
    """Wrap an ONNX IR node and expose PyTorch exporter metadata.

    The Torch ONNX exporter stores per-node metadata describing the originating
    module instance hierarchy and class names. This helper parses the serialized
    metadata strings into Python objects and provides convenience accessors for
    querying instance/class names at different nesting depths.
    """

    def __init__(self, node: ir.Node) -> None:
        """Wrap a node and parse exporter metadata when present."""
        self._node = node

        if self.check_node_metadata_exists():
            self.instance_metadata = ast.literal_eval(
                self._node.metadata_props["pkg.torch.onnx.name_scopes"]
            )
            self.class_metadata = ast.literal_eval(
                self._node.metadata_props["pkg.torch.onnx.class_hierarchy"]
            )

    def check_node_metadata_exists(self) -> bool:
        """Return True if the required PyTorch metadata keys are present."""
        return bool(
            "pkg.torch.onnx.name_scopes" in self._node.metadata_props
            and "pkg.torch.onnx.class_hierarchy" in self._node.metadata_props
        )

    def is_last_level(self, level: int) -> bool:
        """Return True if the provided level is the last metadata entry."""
        return len(self.instance_metadata) - 1 == level

    def get_instance_name(self, depth: int = 0) -> str | None:
        """Return the instance name at the given depth, if available."""
        if depth >= len(self.instance_metadata):
            return None
        return self.instance_metadata[depth]

    def get_class_name(self, depth: int = 0) -> str | None:
        """Return the class name at the given depth, if available."""
        if depth >= len(self.instance_metadata):
            return None
        return self.class_metadata[depth]


class PytorchHierarchyNode:
    """Represent a node in the hierarchy reconstructed from PyTorch metadata.

    Each instance mirrors a PyTorch module captured by the exporter. It stores
    child modules plus the wrapped ONNX nodes and exposes helpers that let
    callers traverse or query the reconstructed module tree.

    Example::

        root = PytorchHierarchyNode()
        for ir_node in graph._nodes:
            root.add_node(ir_node)

        root.print_hierarchy()
        target_path = ["top_module", "encoder", "layer_0"]
        ir_nodes = root.get_nodes(target_path)

    Notes
    -----
    ``add_node`` can be called in any order because the structure is built
    incrementally. ``get_nodes`` performs prefix matching so supplying
    ``["top_module", "encoder"]`` returns every descendant of that subtree.
    Nodes that are missing exporter metadata are ignored, and the maximum
    depth matches the length of the serialized ``name_scopes`` list.
    """

    def __init__(self) -> None:
        """Initialize an empty hierarchy node."""
        self.instance_name = None
        self.module_type = None
        self.children = []
        self.nodes = []

    def print_hierarchy(self, instance_hierarchy: list[str] | None = None) -> None:
        """Print the module hierarchy and nodes to stdout."""
        if instance_hierarchy is None:
            instance_hierarchy = []
        if self.instance_name is not None:
            instance_hierarchy.append(self.instance_name)

        for child in self.children:
            child.print_hierarchy(list(instance_hierarchy))

        for node in self.nodes:
            print(
                f"Node: {node._node.name}, "  # noqa: SLF001
                f"Instance: {'/'.join(instance_hierarchy)},"
                f" Module: {self.module_type}"
            )

    def get_unwrapped_nodes(self) -> list[ir.Node]:
        """Return the underlying IR nodes stored in this hierarchy node."""
        # Return _node from self._nodes
        return [node._node for node in self.nodes]  # noqa: SLF001

    # Checks if the search hierarchy matches the instance hierarchy
    def hierarchy_matches(
        self, search_hierarchy: list[str], instance_hierarchy: list[str] | None = None
    ) -> bool:
        """Return True if the instance path matches the search prefix."""
        if instance_hierarchy is None:
            instance_hierarchy = []
        search_length = min(len(search_hierarchy), len(instance_hierarchy))
        return all(search_hierarchy[i] == instance_hierarchy[i] for i in range(search_length))

    # Return all nodes from the given name hierarchy on down
    def get_nodes(
        self, search_hierarchy: list[str], instance_hierarchy: list[str] | None = None
    ) -> list[ir.Node]:
        """Return all IR nodes under the matched hierarchy path."""
        if instance_hierarchy is None:
            instance_hierarchy = []

        nodes_to_return = []
        # base case for recursion
        # 1 - search_hierarchy does not match instance_hierarchy
        if self.instance_name is not None:
            instance_hierarchy.append(self.instance_name)

        if not self.hierarchy_matches(search_hierarchy, instance_hierarchy):
            return []

        for child in self.children:
            child_nodes = child.get_nodes(search_hierarchy, list(instance_hierarchy))
            nodes_to_return.extend(child_nodes)

        if len(instance_hierarchy) >= len(search_hierarchy):
            nodes_to_return.extend(self.get_unwrapped_nodes())

        return nodes_to_return

    def add_node(self, node: PytorchMetadataNode | ir.Node, level: int = 0) -> bool:
        """Insert a node into the hierarchy, creating children as needed."""
        if not isinstance(node, PytorchMetadataNode):
            node = PytorchMetadataNode(node)
            if node.check_node_metadata_exists() is False:
                return False

        if self.instance_name is None:
            self.instance_name = node.get_instance_name(level)
        if self.module_type is None:
            self.module_type = node.get_class_name(level)

        # check that instance name and module type match
        if self.instance_name != node.get_instance_name(level):
            return False
        if self.module_type != node.get_class_name(level):
            return False
        # if this is the last level of the hierarchy, add the node to this node
        # otherwise find the child node that matches the next level of the hierarchy
        # and add the node to that child
        if node.is_last_level(level):
            self.nodes.append(node)
            return True
        for child in self.children:
            if child.instance_name == node.get_instance_name(level + 1):
                return child.add_node(node, level + 1)

        # if no child matches the next level of the hierarchy, create a new child node
        new_child = PytorchHierarchyNode()
        new_child.instance_name = node.get_instance_name(level + 1)
        new_child.module_type = node.get_class_name(level + 1)
        self.children.append(new_child)
        return new_child.add_node(node, level + 1)


def direct_convert_ir_graph_to_pattern(graph: ir.Graph) -> GraphPattern:
    """Convert an IR graph into an ONNX Script ``GraphPattern``.

    The conversion walks nodes in order, mapping each IR ``Value`` to the
    corresponding ``ValuePattern``/``NodeOutputPattern`` produced by the
    pattern builder. The resulting pattern preserves input/output ordering and
    captures every constructed operator so it can later drive rewrite rules.
    """
    # Transform IR values to ValuePatterns

    vmap = {}
    for inp in graph.inputs:
        vmap[inp] = ValuePattern(inp.name)

    for init in graph.initializers:
        vmap[init] = ValuePattern(init)

    for node in graph._nodes:  # noqa: SLF001
        if node.op_type == "Constant":
            vmap[node.outputs[0]] = ValuePattern(node.outputs[0].name)

    builder = OpsetPatternBuilder("", record=True)

    with pattern_builder(builder):
        for node in graph._nodes:  # noqa: SLF001
            ninputs = []
            for ninput in node.inputs:
                ninputs.append(vmap[ninput])

            vp_outputs = builder.__getattr__(node.op_type)(
                *ninputs, _domain=node.domain, _outputs=len(node.outputs)
            )

            if isinstance(vp_outputs, NodeOutputPattern):
                vp_outputs = [vp_outputs]

            for vp_output in iter(vp_outputs):
                vmap[node.outputs[vp_output.output_index]] = vp_output

    pinputs = []
    for inp in graph.inputs:
        pinputs.append(vmap[inp])

    # build graph outputs
    poutputs = []
    for output in graph.outputs:
        poutputs.append(vmap[output])

    return GraphPattern(inputs=pinputs, outputs=poutputs, nodes=builder.nodes())


def remove_input_from_node(node: ir.Node, inp: ir.Value) -> None:
    """Remove a single input value from a node and update usages."""
    index = None
    for i, ninput in enumerate(node.inputs):
        if ninput == inp:
            index = i
            break
    if index is None:
        raise FINNInternalError("Input value not found in node inputs")
    node._inputs = tuple([x for x in node._inputs if x is not inp])  # noqa: SLF001
    inp._remove_usage(node, index)  # noqa: SLF001


def same(input_list: tuple[ir.Value | None, ...]) -> bool:
    """Return True if all values in the tuple are identical."""
    return len(set(input_list)) == 1


def vdisconnect(value: ir.Value) -> ir.Value:
    """Clear graph connectivity metadata from a value."""
    value._uses = {}  # noqa: SLF001
    value._producer = None  # noqa: SLF001
    value._index = None  # noqa: SLF001
    value._graph = None  # noqa: SLF001
    return value


def is_fpgadataflow_onnxir_node(node: ir.Node) -> bool:
    """Return True if given node is fpgadataflow node. Otherwise False."""
    is_node = False
    if node is not None and is_custom_op(node.domain) and "backend" in node.attributes:
        backend_value = node.attributes["backend"].as_string()
        if backend_value == "fpgadataflow":
            is_node = True

    return is_node


class ReplacementPatternGraph(ReplacementPatternFunction):
    """Instantiate a replacement pattern graph from an ONNX Script IR graph.

    The class adapts an ``ir.Graph`` into the replacement side of a rewrite
    rule: when the pattern matches, ``get_replacement`` materialises the stored
    graph inside the active rewrite context while remapping bound values to the
    match result.
    """

    def __init__(self, ir_graph: ir.Graph) -> None:
        """Store the IR graph to materialize during rewrite."""
        self._graph = ir_graph

    def get_replacement(self, match: MatchResult) -> ReplacementSubgraph | None:
        """Build the replacement subgraph for a successful match."""
        context = RewriterContext()
        # ``match.bindings`` maps ``value_name`` (str) from the replacement
        # subgraph pattern to actual IR values.
        vvmap = {}  # Maps pattern values to the values that will populate the replacement

        for value in self._graph.inputs:
            if value.name in match.bindings:
                vvmap[value] = match.bindings[value.name]
            else:
                vvmap[value] = value

        for node in self._graph._nodes:  # noqa: SLF001
            ninputs = []
            for ninput in node.inputs:
                ninputs.append(vvmap[ninput])

            coutput = context.__getattr__(node.op_type)(
                *ninputs,
                **node.attributes,
                _outputs=len(node.outputs),
                _domain=node.domain,
                _version=node.version,
            )
            if not isinstance(coutput, Iterable):
                coutput = [coutput]

            for i, cout in enumerate(coutput):
                cout._type = node.outputs[i].type  # noqa: SLF001
                cout._shape = node.outputs[i].shape  # noqa: SLF001
                for key in node.outputs[i].meta:
                    cout.meta[key] = node.outputs[i].meta[key]
                vvmap[node.outputs[cout.index()]] = cout

        new_outputs = [vvmap[x] for x in self._graph.outputs]
        return ReplacementSubgraph(
            match, new_outputs, context.nodes, context.initializers, context.used_opsets
        )


def find_nodes_of_optype(graph: ir.Graph, layername: str) -> list[ir.Node]:
    """Return all nodes matching the requested op type."""
    nodes = []
    for node in ir.traversal.RecursiveGraphIterator(graph):
        if node.op_type == layername:
            nodes.append(node)
    return nodes


def build_constant_from_tensor(name: str, tensor: ir.Tensor) -> ir.Node:
    """Create a Constant node holding the provided tensor."""
    value_attribute = ir.Attr(name="value", type=ir.AttributeType.TENSOR, value=tensor)
    ir_value_out = ir.Value(name=name + "_out", type=ir.TensorType(tensor.dtype))
    return ir.Node(
        "", "Constant", name=name, inputs=[], outputs=[ir_value_out], attributes=[value_attribute]
    )


def build_concat_node_from_inputs(inputs: tuple[ir.Value | None, ...]) -> ir.Node:
    """Build a Concat node that joins the provided inputs along axis 0."""
    axis = ir.Attr(name="axis", type=ir.AttributeType.INT, value=0)
    if inputs[0] is None:
        raise FINNInternalError("First input to concat cannot be None")
    if inputs[0].shape is None:
        raise FINNInternalError("Input to concat must have known shape")
    ndim = len(inputs) * cast("int", inputs[0].shape.dims[0])
    output_shape = ir.Shape([ndim, *inputs[0].shape.dims[1:]])
    output = ir.Value(name=f"{inputs[0].name}_concat", shape=output_shape, type=inputs[0].type)
    return ir.Node("", "Concat", inputs=inputs, attributes=[axis], outputs=[output])


def build_reshape_node(inp: ir.Value, reshape_shape: ir.Value) -> ir.Node:
    """Build a Reshape node using the provided shape value."""
    reshape_out = ir.Value(name=f"{inp.name}_reshape", type=inp.type)
    return ir.Node("", "Reshape", inputs=[inp, reshape_shape], outputs=[reshape_out])


def tensor_type_to_finn_datatype_string(
    tensor_type: ir.TensorType,
) -> Literal[
    "FLOAT32", "INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16", "UINT32", "UINT64", "BOOL"
]:
    """Map an ONNX Script tensor type to a FINN datatype string."""
    if tensor_type == ir.TensorType(_enums.DataType.FLOAT):
        return "FLOAT32"
    if tensor_type == ir.TensorType(_enums.DataType.INT8):
        return "INT8"
    if tensor_type == ir.TensorType(_enums.DataType.INT16):
        return "INT16"
    if tensor_type == ir.TensorType(_enums.DataType.INT32):
        return "INT32"
    if tensor_type == ir.TensorType(_enums.DataType.INT64):
        return "INT64"
    if tensor_type == ir.TensorType(_enums.DataType.UINT8):
        return "UINT8"
    if tensor_type == ir.TensorType(_enums.DataType.UINT16):
        return "UINT16"
    if tensor_type == ir.TensorType(_enums.DataType.UINT32):
        return "UINT32"
    if tensor_type == ir.TensorType(_enums.DataType.UINT64):
        return "UINT64"
    if tensor_type == ir.TensorType(_enums.DataType.BOOL):
        return "BOOL"
    raise FINNInternalError(f"Unsupported tensor type: {tensor_type}")

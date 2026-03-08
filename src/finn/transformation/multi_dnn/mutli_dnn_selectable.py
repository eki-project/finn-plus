from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation


class ExtractSelectableWeights(Transformation):
    def __init__(self, models: list[str]):
        super().__init__()
        self.models = models  # First model is always the "master model"

    def apply(self, model: ModelWrapper) -> ModelWrapper:
        if len(self.models) < 2:
            return model, False

        dnn_nodes = [getCustomOp(node) for node in model.get_nodes_by_op_type("DNNContainer")]
        dnn_nodes_and_bodies = [(op, op.get_nodeattr("body")) for op in dnn_nodes]
        dnn_nodes_and_bodies = [
            (op, body) for op, body in dnn_nodes_and_bodies if body.graph.name in self.models
        ]
        dnn_node_bodies = [body for _, body in dnn_nodes_and_bodies]

        # We currently only support this when the graphs are identical in structure
        # and only differ in weights
        fm = dnn_node_bodies[0]
        for body in dnn_node_bodies[1:]:
            for node_idx, fm_node in enumerate(fm.graph.node):
                body_node = body.graph.node[node_idx]
                if fm_node.op_type != body_node.op_type:
                    raise Exception(
                        f"The graphs differ in op_type for at least one node pair, "
                        f"cannot extract selectable weights: {fm_node.name} ({fm_node.op_type})"
                        f" vs {body_node.name} ({body_node.op_type})"
                    )
                body_op = getCustomOp(body_node)
                fm_op = getCustomOp(fm_node)
                for attr in fm_op.get_nodeattr_types().keys():
                    fm_attr = fm_op.get_nodeattr(attr)
                    body_attr = body_op.get_nodeattr(attr)
                    if fm_attr != body_attr:
                        print(
                            f"The graphs differ in attribute {attr} for at least one node pair, "
                            f"cannot extract selectable weights: {fm_node.name} ({fm_attr})"
                            f" vs {body_node.name} ({body_attr})"
                        )
                        raise Exception(
                            f"The graphs differ in attribute {attr} for at least one node pair, "
                            f"cannot extract selectable weights: {fm_node.name} ({fm_attr})"
                            f" vs {body_node.name} ({body_attr})"
                        )

        # If we are here, the graphs are identical in structure and only differ in weights,
        # we can extract selectable weights
        for node_idx, fm_node in enumerate(fm.graph.node):
            # Tested for MVAU_rtl, Thresholding_rtl,
            ops = ["MVAU_rtl", "Thresholding_rtl"]  # add more
            # TODO: Check that memmode is valid
            # TODO: Make sure the names are identical (?)
            if fm_node.op_type in ops or fm_node.op_type.startswith("Elementwise"):
                num_bodies = len(dnn_node_bodies)

                # Build a minimal single-node GraphProto for each body's node at this index
                def _make_single_node_graph(node, parent_model: ModelWrapper):
                    """Build a GraphProto containing only this node, with its initializers."""
                    parent_initializer_map = {
                        init.name: init for init in parent_model.graph.initializer
                    }
                    node_initializers = [
                        parent_initializer_map[inp]
                        for inp in node.input
                        if inp != "" and inp in parent_initializer_map
                    ]
                    initializer_names = {init.name for init in node_initializers}
                    inputs = [
                        helper.make_tensor_value_info(
                            inp, TensorProto.FLOAT, parent_model.get_tensor_shape(inp) or [1]
                        )
                        for inp in node.input
                        if inp != "" and inp not in initializer_names
                    ]
                    outputs = [
                        helper.make_tensor_value_info(
                            out, TensorProto.FLOAT, parent_model.get_tensor_shape(out) or [1]
                        )
                        for out in node.output
                        if out != ""
                    ]
                    graph = helper.make_graph(
                        nodes=[node],
                        name=node.name,
                        inputs=inputs,
                        outputs=outputs,
                        initializer=node_initializers,
                    )
                    # Copy quantization annotations for all tensors (edges) of this node
                    relevant_tensors = {inp for inp in node.input if inp != ""} | {
                        out for out in node.output if out != ""
                    }
                    for annotation in parent_model.graph.quantization_annotation:
                        if annotation.tensor_name in relevant_tensors:
                            graph.quantization_annotation.append(annotation)
                    return graph

                body_graphs = [
                    _make_single_node_graph(
                        dnn_node_bodies[b_idx].graph.node[node_idx],
                        dnn_node_bodies[b_idx],
                    )
                    for b_idx in range(num_bodies)
                ]

                # Determine input/output datatypes from the fm node
                fm_op = getCustomOp(fm_node)
                idt = fm_op.get_input_datatype().name
                odt = fm_op.get_output_datatype().name

                # Build the body_i keyword arguments for make_node
                body_kwargs = {f"body_{i}": body_graphs[i] for i in range(num_bodies)}

                initializer_names = {init.name for init in fm.model.graph.initializer}
                nc_inputs = [inp for inp in fm_node.input if inp not in initializer_names]

                node_container = helper.make_node(
                    "NodeContainer",
                    inputs=nc_inputs,
                    outputs=list(fm_node.output),
                    domain="finn.custom_op.fpgadataflow.rtl",
                    backend="fpgadataflow",
                    name="node_container_" + fm_node.name,
                    multi_dnn_type="selectable_weights",
                    bodies=num_bodies,
                    inputDataType=idt,
                    outputDataType=odt,
                    **body_kwargs,
                )

                # Replace fm_node in the fm graph with the NodeContainer
                fm_node_index = list(fm.graph.node).index(fm_node)
                fm.graph.node.remove(fm_node)
                fm.graph.node.insert(fm_node_index, node_container)

        # Write the modified fm back into its DNNContainer's body attribute
        fm_dnn_op = dnn_nodes_and_bodies[0][0]
        fm_dnn_op.set_nodeattr("body", fm)

        # Remove all DNNContainer nodes that contributed bodies (all except the first/fm one)
        for op, _ in dnn_nodes_and_bodies[1:]:
            model.graph.node.remove(op.onnx_node)

        return model, False

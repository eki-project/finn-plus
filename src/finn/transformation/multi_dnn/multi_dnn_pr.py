"""Transformation to apply partial reconfiguration to a multi-DNN model."""
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation


class ApplyPartialReconfiguration(Transformation):
    """Restructure DNNContainer nodes into a partial reconfiguration NodeContainer."""

    def __init__(self, **kwargs):
        """Initialize with reference_model_name and pr_regions keyword arguments."""
        super().__init__()
        self.reference_model_name = kwargs.get("reference_model_name", None)
        self.pr_regions = kwargs.get("pr_regions", None)

    def apply(self, model: ModelWrapper) -> ModelWrapper:
        """Apply partial reconfiguration restructuring to the model."""
        reference_model_name = self.reference_model_name
        pr_regions = self.pr_regions

        dnn_nodes = [getCustomOp(node) for node in model.get_nodes_by_op_type("DNNContainer")]
        dnn_nodes_and_bodies = {
            op.get_nodeattr("body").graph.name: op.get_nodeattr("body") for op in dnn_nodes
        }
        dnn_op_map = {op.get_nodeattr("body").graph.name: op for op in dnn_nodes}
        reference_model = dnn_nodes_and_bodies[reference_model_name]

        for pr_region_name in pr_regions:
            subgraphs = []
            ref_input_names = []
            ref_output_names = []
            for model_name, pr_nodes in pr_regions[pr_region_name].items():
                if model_name == "pblock":
                    continue
                graph = dnn_nodes_and_bodies[model_name].graph
                pr_node_set = set(pr_nodes)

                subgraph_nodes = [node for node in graph.node if node.name in pr_node_set]
                internal_outputs = {out for node in subgraph_nodes for out in node.output if out}
                initializer_names = {init.name for init in graph.initializer}
                seen = set()
                subgraph_input_names = []
                for node in subgraph_nodes:
                    for inp in node.input:
                        if (
                            inp
                            and inp not in internal_outputs
                            and inp not in initializer_names
                            and inp not in seen
                        ):
                            subgraph_input_names.append(inp)
                            seen.add(inp)

                outside_inputs = {
                    inp
                    for node in graph.node
                    if node.name not in pr_node_set
                    for inp in node.input
                    if inp
                }
                graph_output_names = {out.name for out in graph.output}
                subgraph_output_names = [
                    name
                    for name in internal_outputs
                    if name in outside_inputs or name in graph_output_names
                ]
                subgraph_node_inputs = {inp for node in subgraph_nodes for inp in node.input if inp}
                subgraph_initializers = [
                    init for init in graph.initializer if init.name in subgraph_node_inputs
                ]

                value_info_map = {vi.name: vi for vi in graph.value_info}
                value_info_map.update({vi.name: vi for vi in graph.input})
                value_info_map.update({vi.name: vi for vi in graph.output})

                subgraph_inputs = [
                    value_info_map[n] for n in subgraph_input_names if n in value_info_map
                ]
                subgraph_outputs = [
                    value_info_map[n] for n in subgraph_output_names if n in value_info_map
                ]
                intermediate_names = internal_outputs - set(subgraph_output_names)
                subgraph_value_info = [
                    value_info_map[n] for n in intermediate_names if n in value_info_map
                ]

                subgraph = helper.make_graph(
                    nodes=subgraph_nodes,
                    name=f"{pr_region_name}_{model_name}",
                    inputs=subgraph_inputs,
                    outputs=subgraph_outputs,
                    initializer=subgraph_initializers,
                    value_info=subgraph_value_info,
                )

                subgraph_tensor_names = subgraph_node_inputs | internal_outputs
                for annotation in graph.quantization_annotation:
                    if annotation.tensor_name in subgraph_tensor_names:
                        subgraph.quantization_annotation.append(annotation)
                subgraphs.append(subgraph)
                if model_name == reference_model_name:
                    ref_input_names = subgraph_input_names
                    ref_output_names = subgraph_output_names

            ref_pr_nodes_set = set(pr_regions[pr_region_name].get(reference_model_name, []))
            ref_graph = reference_model.graph
            ref_subgraph_nodes = [node for node in ref_graph.node if node.name in ref_pr_nodes_set]

            num_bodies = len(subgraphs)
            body_kwargs = {f"body_{i}": subgraphs[i] for i in range(num_bodies)}
            node_container = helper.make_node(
                "NodeContainer",
                inputs=ref_input_names,
                outputs=ref_output_names,
                domain="finn.custom_op.fpgadataflow.rtl",
                backend="fpgadataflow",
                name=f"node_container_{pr_region_name}",
                multi_dnn_type="partial_reconfiguration",
                bodies=num_bodies,
                pblock=pr_regions[pr_region_name].get("pblock", ""),
                **body_kwargs,
            )

            pr_node_indices = [
                i for i, n in enumerate(ref_graph.node) if n.name in ref_pr_nodes_set
            ]
            insert_idx = min(pr_node_indices) if pr_node_indices else len(ref_graph.node)
            for node in ref_subgraph_nodes:
                ref_graph.node.remove(node)
            ref_graph.node.insert(insert_idx, node_container)

        dnn_op_map[reference_model_name].set_nodeattr("body", reference_model)
        ref_dnn_node = dnn_op_map[reference_model_name].onnx_node
        for op in dnn_nodes:
            if op.onnx_node is not ref_dnn_node:
                model.graph.node.remove(op.onnx_node)

        return model, False

import copy
import json
import numpy as np
import qonnx.util.basic as util
from onnx import NodeProto, TensorProto, ValueInfoProto, helper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation

from finn.custom_op.fpgadataflow.dnncontainer import DNNContainer


class MultiDNNWrapperExposeIO(Transformation):
    # Call this before doing any further transformations. This transformation is used to map the IO
    def apply(self, model):
        for node in model.graph.node:
            if node.op_type == "DNNContainer":
                # Check if this node is not connected to anything else
                if model.find_direct_predecessors(node) or model.find_direct_successors(node):
                    continue

                dnn_custom_op = getCustomOp(node)
                assert isinstance(dnn_custom_op, DNNContainer)
                body_model = dnn_custom_op.get_nodeattr("body")
                body_graph = body_model.graph
                graph_name = body_graph.name

                for inp_name in node.input:
                    util.remove_by_name(model.graph.input, inp_name)
                for out_name in node.output:
                    util.remove_by_name(model.graph.output, out_name)

                new_input_names = {}
                for body_inp in body_graph.input:
                    new_inp_name = graph_name + "_" + body_inp.name
                    new_inp = helper.make_tensor_value_info(
                        new_inp_name, TensorProto.FLOAT, body_model.get_tensor_shape(body_inp.name)
                    )
                    model.graph.input.append(new_inp)
                    model.set_tensor_datatype(
                        new_inp_name, body_model.get_tensor_datatype(body_inp.name)
                    )
                    new_input_names.update({body_inp.name: new_inp_name})

                new_output_names = {}
                for body_out in body_graph.output:
                    new_out_name = graph_name + "_" + body_out.name
                    new_out = helper.make_tensor_value_info(
                        new_out_name, TensorProto.FLOAT, body_model.get_tensor_shape(body_out.name)
                    )
                    model.graph.output.append(new_out)
                    model.set_tensor_datatype(
                        new_out_name, body_model.get_tensor_datatype(body_out.name)
                    )
                    new_output_names.update({body_out.name: new_out_name})

                dnn_custom_op.set_nodeattr("io_map", json.dumps(new_input_names | new_output_names))

                node.input[:] = list(new_input_names.values())
                node.output[:] = list(new_output_names.values())

        return (model, False)


class CollapseModels(Transformation):
    def apply(self, model):
        inital_nodes = copy.deepcopy(model.graph.node)
        for node in inital_nodes:
            if node.op_type == "DNNContainer":
                dnn_custom_op = getCustomOp(node)
                assert isinstance(dnn_custom_op, DNNContainer)
                body_model = dnn_custom_op.get_nodeattr("body")
                io_map = json.loads(dnn_custom_op.get_nodeattr("io_map"))
                model.graph.node.remove(node)

                mod = body_model
                all_nodes = []
                all_initializers = []
                all_inputs = []
                all_outputs = []
                all_value_info = []

                prefix = mod.graph.name + "_"

                for node in mod.graph.node:
                    new_node = NodeProto()
                    new_node.CopyFrom(node)
                    new_node.name = prefix + node.name
                    new_node.input[:] = [prefix + inp for inp in node.input]
                    new_node.output[:] = [prefix + out for out in node.output]
                    all_nodes.append(new_node)

                # Process initializers
                for initializer in mod.graph.initializer:
                    new_initializer = TensorProto()
                    new_initializer.CopyFrom(initializer)
                    new_initializer.name = prefix + initializer.name
                    all_initializers.append(new_initializer)

                # Process inputs
                for input_tensor in mod.graph.input:
                    new_input = ValueInfoProto()
                    new_input.CopyFrom(input_tensor)
                    new_input.name = io_map[input_tensor.name]
                    all_inputs.append(new_input)

                # Process outputs
                for output_tensor in mod.graph.output:
                    new_output = ValueInfoProto()
                    new_output.CopyFrom(output_tensor)
                    new_output.name = io_map[output_tensor.name]
                    all_outputs.append(new_output)

                # Process value_info
                for value_info in mod.graph.value_info:
                    new_value_info = ValueInfoProto()
                    new_value_info.CopyFrom(value_info)
                    new_value_info.name = prefix + value_info.name
                    all_value_info.append(new_value_info)

                model.graph.node.extend(all_nodes)
                model.graph.initializer.extend(all_initializers)
                model.graph.value_info.extend(all_value_info)

                for qnt_annotation in mod.graph.quantization_annotation:
                    old_tensor_name = qnt_annotation.tensor_name
                    if old_tensor_name in io_map:
                        new_tensor_name = io_map[old_tensor_name]
                    else:
                        new_tensor_name = prefix + old_tensor_name
                    datatype = body_model.get_tensor_datatype(old_tensor_name)
                    if datatype is not None:
                        model.set_tensor_datatype(new_tensor_name, datatype)
                    layout = body_model.get_tensor_layout(old_tensor_name)
                    if layout is not None:
                        model.set_tensor_layout(new_tensor_name, layout)
                    sparsity = body_model.get_tensor_sparsity(old_tensor_name)
                    if sparsity is not None:
                        model.set_tensor_sparsity(new_tensor_name, sparsity)

        return (model, False)


class CombineInputsChannelwise(Transformation):
    def apply(self, model):
        inputs = model.graph.input
        if len(inputs) < 2:
            return model, False

        # Create new combined input
        shapes = np.array([model.get_tensor_shape(inp.name) for inp in inputs])
        if not np.all(shapes[:, :-1] == shapes[0, :-1]):
            raise ValueError("All inputs must have the same shape except for channel dimension")

        input_shape = [int(d) for d in shapes[0]]
        channelsize = [int(c) for c in shapes[:, -1]]
        input_shape[-1] = sum(channelsize)

        combined_input_name = "combined_input"
        combined_input = helper.make_tensor_value_info(
            combined_input_name, inputs[0].type.tensor_type.elem_type, input_shape
        )

        split_sizes_name = "split_sizes"
        split_sizes_tensor = helper.make_tensor(
            split_sizes_name, TensorProto.INT64, [len(channelsize)], channelsize
        )
        model.graph.initializer.append(split_sizes_tensor)

        # Add Split node to split the combined input along channel dimension
        split_outputs = [inp.name for inp in inputs]
        split_node = helper.make_node(
            "Split",
            inputs=[combined_input_name, split_sizes_name],
            outputs=split_outputs,
            axis=-1,
            name="input_split",
        )
        node_index = 0
        model.graph.node.insert(node_index, split_node)

        original_datatypes = {}
        original_shapes = {}
        for inp in inputs:
            dt = model.get_tensor_datatype(inp.name)
            if dt is not None:
                original_datatypes[inp.name] = dt

            orig_shape = []
            for dim in inp.type.tensor_type.shape.dim:
                orig_shape.append(dim.dim_value if dim.HasField("dim_value") else -1)
            original_shapes[inp.name] = orig_shape

            split_output_info = helper.make_tensor_value_info(
                inp.name, inp.type.tensor_type.elem_type, orig_shape
            )
            model.graph.value_info.append(split_output_info)

        input_datatypes = [model.get_tensor_datatype(inp.name) for inp in inputs]
        if not all(dt == input_datatypes[0] for dt in input_datatypes if dt is not None):
            raise ValueError("All inputs must have the same datatype")

        first_input_dt = model.get_tensor_datatype(inputs[0].name)
        if first_input_dt is not None:
            model.set_tensor_datatype(combined_input_name, first_input_dt)

        del model.graph.input[:]
        model.graph.input.append(combined_input)

        for tensor_name, dt in original_datatypes.items():
            model.set_tensor_datatype(tensor_name, dt)

        return model, False


class CombineOutputsChannelwise(Transformation):
    def apply(self, model):
        outputs = model.graph.output
        if len(outputs) < 2:
            return model, False

        # Add Concat node to combine all outputs along channel dimension
        combined_output_name = "combined_output"
        concat_inputs = [output.name for output in outputs]
        concat_node = helper.make_node(
            "Concat",
            inputs=concat_inputs,
            outputs=[combined_output_name],
            axis=-1,
            name="output_concat",
        )
        model.graph.node.append(concat_node)

        output_shape = []
        shapes = np.array([model.get_tensor_shape(outp.name) for outp in outputs])
        if not np.all(shapes[:, :-1] == shapes[0, :-1]):
            raise ValueError("All inputs must have the same shape except for channel dimension")

        output_shape = [int(d) for d in shapes[0]]
        channelsize = [int(c) for c in shapes[:, -1]]
        output_shape[-1] = sum(channelsize)

        combined_output = helper.make_tensor_value_info(
            combined_output_name, outputs[0].type.tensor_type.elem_type, output_shape
        )

        output_datatypes = [model.get_tensor_datatype(outp.name) for outp in outputs]
        if not all(dt == output_datatypes[0] for dt in output_datatypes if dt is not None):
            raise ValueError("All outputs must have the same datatype")

        first_output_dt = output_datatypes[0]
        if first_output_dt is not None:
            model.set_tensor_datatype(combined_output_name, first_output_dt)

        del model.graph.output[:]
        model.graph.output.append(combined_output)

        return model, False

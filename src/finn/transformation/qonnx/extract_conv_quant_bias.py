import numpy as np
import qonnx.core.onnx_exec as oxe
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.base import Transformation


class ExtractConvQuantBias(Transformation):
    """Scans the network for Conv nodes, which have a bias that is being
    initialized by a Quant node. These need extra handling as FoldQuantWeights
    would handle them incorrectly.
    NOTE: This is heavily based on the FoldQuantWeights transformation
    """
    def apply(self, model : ModelWrapper):
        graph = model.graph
        node_ind = 0
        execution_context = model.make_empty_exec_context()
        graph_modified = False

        for n in graph.node:
            node_ind += 1

            if n.op_type == "Quant":
                target_node = model.find_direct_successors(n)
                target_node = target_node[0]
                node_out = n.output[0]

                if target_node.op_type != "Conv":
                    continue
                    
                if len(target_node.input) != 3:
                    continue

                if node_out != target_node.input[2]:
                    continue

                oxe.execute_node(n, execution_context, graph)
                q_node_output = execution_context[node_out]

                # For both mul and Add:
                # Move the scale factor behind the next operator
                scale = model.get_initializer(n.input[1])
                new_initializer = q_node_output / scale

                # Round, to correct for floating point errors
                new_initializer = np.round(new_initializer)
                
                # Reshape scale for Conv if required
                target_output_shape = model.get_tensor_shape(target_node.output[0])
                if target_node.op_type == "Conv" and len(scale.shape) > 0:
                    conv_out_shape = [1] * len(target_output_shape)
                    # only support per-output channel scaling
                    # (i.e. all scale shape elems besides 0th must be 1s)
                    if len(scale.shape) > 1:
                        assert (
                            np.prod(scale.shape[1:]) == 1
                        ), "Can't fold scale beyond per-out-channel granularity"
                    # collect all scaling in channels dim (since we constrain)
                    conv_out_shape[1] = -1
                    scale = scale.reshape(conv_out_shape)
                    new_initializer = new_initializer.reshape(conv_out_shape)

                if scale.shape == (1,):
                    scale = scale[0]
                    mul_shape = tuple()
                else:
                    mul_shape = scale.shape

                #add tensor for div node initializer
                div_tensor = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    mul_shape,
                )
                graph.value_info.append(div_tensor)
                model.set_initializer(div_tensor.name, scale)

                #add tensor for add node initializer
                add_tensor = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    mul_shape,
                )
                graph.value_info.append(add_tensor)
                model.set_initializer(add_tensor.name, new_initializer)

                #add tensor for mul node initializer
                mul_tensor = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    mul_shape,
                )
                graph.value_info.append(mul_tensor)
                model.set_initializer(mul_tensor.name, scale)

                
                successor = model.find_consumers(node_out)
                if successor == []:
                    raise RuntimeError(
                        "Can only constant fold scaled Quant weights "
                        "if a successor exists."
                    )
                assert len(successor) == 1, "Only implemented for a single consumer"
                successor = successor[0]
                succ_output_name = successor.output[0]
                
                #add mul node
                act_mul_tensor = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    None,
                )
                graph.value_info.append(act_mul_tensor)

                mul_node = helper.make_node(
                    "Mul",
                    [act_mul_tensor.name, mul_tensor.name],
                    [succ_output_name]
                )
                graph.node.insert(node_ind, mul_node)
                
                #add add node
                act_add_tensor = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    None,
                )
                graph.value_info.append(act_add_tensor)

                add_node = helper.make_node(
                    "Add",
                    [act_add_tensor.name, add_tensor.name],
                    [act_mul_tensor.name]
                )
                graph.node.insert(node_ind, add_node)
                
                #add div node
                act_div_tensor = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    None,
                )
                graph.value_info.append(act_div_tensor)
                successor.output[0] = act_div_tensor.name

                #TODO might aswell add a mul node directly, spares using the ConvertDivToMul transformation
                div_node = helper.make_node(
                    "Div",
                    [act_div_tensor.name, div_tensor.name],
                    [act_add_tensor.name],
                )

                successor.output[0] = act_div_tensor.name
                graph.node.insert(node_ind, div_node)
                
                graph_modified = True
                graph.node.remove(n)
                target_node.input.remove(target_node.input[2]) #remove bias initializer from conv

        cleanup_model(model)
        return (model, graph_modified)
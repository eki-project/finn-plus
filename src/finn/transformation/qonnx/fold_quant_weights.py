# Copyright (c) 2021, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import qonnx.core.onnx_exec as oxe
from onnx import TensorProto, helper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.transformation.remove import remove_node_and_rewire


class FoldQuantWeights(Transformation):
    """Merges Quant nodes, which are used as weights into the initializer
    of the weight tensor.
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        execution_context = model.make_empty_exec_context()
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Quant" or n.op_type == "BipolarQuant":
                node_inp_inits = list(map(lambda x: model.get_initializer(x), n.input))
                node_inp_dyn = list(filter(lambda x: x is None, node_inp_inits))
                node_out = n.output[0]
                is_all_constant_inputs = len(node_inp_dyn) == 0
                ishape = model.get_tensor_shape(n.input[0])
                is_const_shape = (n.op_type == "Shape") and (ishape is not None)
                if is_all_constant_inputs or is_const_shape:
                    # Check node validity
                    if n.op_type == "Quant" and not model.get_initializer(n.input[2]) == 0:
                        raise ValueError(
                            "Only Quant nodes with zero-point == 0 " "are currently supported."
                        )
                    if model.is_fork_node(n):
                        raise ValueError(
                            "Weights quantized with the Quant node are not "
                            "allowed to be fork nodes node."
                        )
                    target_node = model.find_direct_successors(n)
                    if target_node is None:
                        raise RuntimeError(
                            "Weights quantized with the Quant node must have " "a successor node."
                        )
                    else:
                        target_node = target_node[0]
                    # If there is a DebugMarker in the weight path,
                    # then the DebugMarker needs to be removed before any further
                    # action is taken. Because this node interferes
                    # with how the constant folding tries to determine how to
                    # apply scale factors and in any case the DebugMarker would not
                    # return useful information after folding.
                    if target_node.op_type == "DebugMarker":
                        remove_node_and_rewire(model, target_node)
                        model = model.transform(FoldTransposeIntoQuantInit())
                        return model, True

                    # Continue with constant folding the quant node
                    scale = model.get_initializer(n.input[1])
                    unity_scale = (scale.flatten() == 1.0).all()
                    # this node has no dynamic inputs, only constant ones -- so we can
                    # do constant folding.
                    oxe.execute_node(n, execution_context, graph)
                    q_node_output = execution_context[node_out]
                    # Check we can directly constant fold
                    if unity_scale:
                        # use the execution result as an initializer
                        model.set_initializer(node_out, q_node_output)
                    else:
                        # Check next operator type
                        mul_like_nodes = [
                            "Mul",
                            "Div",
                            "Conv",
                            "MatMul",
                            "Gather",
                            "ConvTranspose",
                        ]
                        add_like_nodes = ["Add", "Sub"]
                        all_supported_ops = mul_like_nodes.copy()
                        all_supported_ops.extend(add_like_nodes)

                        if target_node.op_type not in all_supported_ops:
                            raise ValueError(
                                f"Can't constant fold Quant weight node "
                                f"into node type {target_node.op_type} "
                                f"at node: {target_node}."
                            )

                        # For both mul and Add:
                        # Move the scale factor behind the next operator
                        scale = model.get_initializer(n.input[1])
                        new_initializer = q_node_output / scale
                        # Round, to correct for floating point errors
                        new_initializer = np.round(new_initializer)
                        model.set_initializer(node_out, new_initializer)
                        q_inst = getCustomOp(n)
                        new_dtype = q_inst.get_integer_datatype(model)
                        model.set_tensor_datatype(node_out, new_dtype)

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

                        if scale.shape == (1,):
                            scale = scale[0]
                            mul_shape = tuple()
                        else:
                            mul_shape = scale.shape
                        mul_tensor = helper.make_tensor_value_info(
                            model.make_new_valueinfo_name(),
                            TensorProto.FLOAT,
                            mul_shape,  # Note: This shape is known exactly as
                            # it is an initializer with known shape
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

                        # output_shape = model.get_tensor_shape(successor.output[0])
                        act_mul_tensor = helper.make_tensor_value_info(
                            model.make_new_valueinfo_name(),
                            TensorProto.FLOAT,
                            None,  # Note: Explicitly delete the shape
                            # annotation to be redone by the next shape
                            # inference
                        )
                        graph.value_info.append(act_mul_tensor)
                        successor.output[0] = act_mul_tensor.name

                        mul_node = helper.make_node(
                            "Mul",
                            [act_mul_tensor.name, mul_tensor.name],
                            [succ_output_name],
                        )
                        graph.node.insert(node_ind, mul_node)

                        if target_node.op_type in add_like_nodes:
                            # Move the scale factor behind also in-front of
                            # the next operator
                            div_tensor = helper.make_tensor_value_info(
                                model.make_new_valueinfo_name(),
                                TensorProto.FLOAT,
                                None,  # Note: Explicitly delete the shape
                                # annotation to be redone by the next shape
                                # inference
                            )
                            graph.value_info.append(div_tensor)
                            model.set_initializer(div_tensor.name, scale)

                            # Detect which input of the add-like successor is
                            # fed by the quantizer node to select the other
                            # branch to insert the scale factor
                            if successor.input[0] == node_out:
                                succ_input_name = successor.input[1]
                            else:
                                succ_input_name = successor.input[0]

                            act_mul_tensor = helper.make_tensor_value_info(
                                model.make_new_valueinfo_name(),
                                TensorProto.FLOAT,
                                None,  # Note: Explicitly delete the shape
                                # annotation to be redone by the next shape
                                # inference
                            )
                            graph.value_info.append(act_mul_tensor)

                            # Detect which input of the add-like successor is
                            # fed by the quantizer node to select the other
                            # branch to insert the scale factor
                            if successor.input[0] == node_out:
                                successor.input[1] = act_mul_tensor.name
                            else:
                                successor.input[0] = act_mul_tensor.name

                            div_node = helper.make_node(
                                "Div",
                                [succ_input_name, div_tensor.name],
                                [act_mul_tensor.name],
                            )
                            graph.node.insert(node_ind, div_node)

                    # remove old node
                    graph.node.remove(n)
                    graph_modified = True
                    # Note: Running shape inference is necessary as shape
                    # annotations have been deleted above
                    model = model.transform(InferShapes())
                    return (model, graph_modified)
        return (model, graph_modified)

# Copyright (C) 2023-2024, Advanced Micro Devices, Inc.
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

"""Convert activation functions (GELU, SiLU, Sigmoid, Tanh, Erf-GELU) into PWPolyF HW layers."""

from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name


class InferPWPolyFLayer(Transformation):
    """Convert activations to piecewise polynomial HW layers."""

    _SINGLE_OP_MAP = {"Gelu": "gelu", "Tanh": "tanh"}

    def __init__(self):
        """Initialize instance."""
        super().__init__()

    @staticmethod
    def _is_const_scalar(model, tensor_name, value, tol=1e-3):
        """Check if *tensor_name* is a constant initializer equal to *value*."""
        init = model.get_initializer(tensor_name)
        if init is None:
            return False
        return init.size == 1 and abs(float(init.flat[0]) - value) < tol

    def _match_erf_gelu(self, model, erf_node):
        """Match Erf-based GELU: Div(x,sqrt(2))→Erf→Add(_,1)→Mul(0.5,_)→Mul(x,_).
        Returns (pwp_input, pwp_output, nodes_to_remove) or None."""
        # backward: Erf input must come from Div(x, sqrt(2))
        div_node = model.find_producer(erf_node.input[0])
        if div_node is None or div_node.op_type != "Div":
            return None
        if self._is_const_scalar(model, div_node.input[1], 1.4142135):
            gelu_input = div_node.input[0]
        elif self._is_const_scalar(model, div_node.input[0], 1.4142135):
            gelu_input = div_node.input[1]
        else:
            return None

        # forward: Erf → Add(_, 1)
        erf_consumers = model.find_consumers(erf_node.output[0])
        if len(erf_consumers) != 1 or erf_consumers[0].op_type != "Add":
            return None
        add_node = erf_consumers[0]
        other_add = [i for i in add_node.input if i != erf_node.output[0]]
        if len(other_add) != 1 or not self._is_const_scalar(model, other_add[0], 1.0):
            return None

        # Add → Mul(0.5, _)
        add_consumers = model.find_consumers(add_node.output[0])
        if len(add_consumers) != 1 or add_consumers[0].op_type != "Mul":
            return None
        mul_half = add_consumers[0]
        other_mul_half = [i for i in mul_half.input if i != add_node.output[0]]
        if len(other_mul_half) != 1 or not self._is_const_scalar(model, other_mul_half[0], 0.5):
            return None

        # Mul(0.5,_) → Mul(x, _)
        half_consumers = model.find_consumers(mul_half.output[0])
        if len(half_consumers) != 1 or half_consumers[0].op_type != "Mul":
            return None
        mul_x = half_consumers[0]
        other_mul_x = [i for i in mul_x.input if i != mul_half.output[0]]
        if len(other_mul_x) != 1 or other_mul_x[0] != gelu_input:
            return None

        nodes_to_remove = [div_node, erf_node, add_node, mul_half, mul_x]
        return (gelu_input, mul_x.output[0], nodes_to_remove)

    @staticmethod
    def _make_pwpolyf_node(pwp_input, pwp_output, func, in_shape, idt, name, K=3, degree=2):
        """Create a PWPolyF node with PE=1 for the given function, shape and datatype."""
        num_channels = in_shape[-1]
        return helper.make_node(
            "PWPolyF",
            [pwp_input],
            [pwp_output],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            func=func,
            K=K,
            degree=degree,
            NumChannels=num_channels,
            PE=1,
            inputDataType=idt.name,
            outputDataType=idt.name,
            numInputVectors=list(in_shape[:-1]),
            name=name,
        )

    def apply(self, model):
        """Replace PWPolyFunction ops and matched activation patterns by PWPolyF nodes."""
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1

            # Case 1: explicit PWPolyFunction custom op from PWPolyFActivation export
            if node.op_type == "PWPolyFunction":
                pwp_input = node.input[0]
                pwp_output = node.output[0]
                pwp_in_shape = model.get_tensor_shape(pwp_input)
                idt = model.get_tensor_datatype(pwp_input)

                func = get_by_name(node.attribute, "func").s.decode("utf-8")
                K_attr = get_by_name(node.attribute, "K")
                K = K_attr.i if K_attr is not None else 3
                degree_attr = get_by_name(node.attribute, "degree")
                degree = degree_attr.i if degree_attr is not None else 2

                new_node = self._make_pwpolyf_node(
                    pwp_input,
                    pwp_output,
                    func,
                    pwp_in_shape,
                    idt,
                    "PWPolyF_" + node.name,
                    K,
                    degree,
                )
                graph.node.insert(node_ind, new_node)
                graph.node.remove(node)
                graph_modified = True

            # Case 2: single-node standard ONNX activations (Gelu, Tanh)
            elif node.op_type in self._SINGLE_OP_MAP:
                pwp_input = node.input[0]
                pwp_output = node.output[0]
                pwp_in_shape = model.get_tensor_shape(pwp_input)
                if pwp_in_shape is None or len(pwp_in_shape) < 1:
                    continue
                idt = model.get_tensor_datatype(pwp_input)
                if idt != DataType["FLOAT32"]:
                    continue

                func = self._SINGLE_OP_MAP[node.op_type]
                new_node = self._make_pwpolyf_node(
                    pwp_input,
                    pwp_output,
                    func,
                    pwp_in_shape,
                    idt,
                    "PWPolyF_" + node.name,
                )
                graph.node.insert(node_ind, new_node)
                graph.node.remove(node)
                graph_modified = True

            # Case 3: Sigmoid — standalone or part of SiLU pattern
            elif node.op_type == "Sigmoid":
                sig_input = node.input[0]
                sig_output = node.output[0]
                pwp_in_shape = model.get_tensor_shape(sig_input)
                if pwp_in_shape is None or len(pwp_in_shape) < 1:
                    continue
                idt = model.get_tensor_datatype(sig_input)
                if idt != DataType["FLOAT32"]:
                    continue

                nodes_to_remove = [node]
                func = "sigmoid"
                pwp_output = sig_output

                # Probe for SiLU: Sigmoid feeds a Mul whose other input
                # is the same tensor x that enters the Sigmoid.
                sig_consumers = model.find_consumers(sig_output)
                if len(sig_consumers) == 1:
                    mul_cand = sig_consumers[0]
                    if mul_cand.op_type == "Mul":
                        mul_inputs = list(mul_cand.input)
                        other_idx = 1 if mul_inputs[0] == sig_output else 0
                        if mul_inputs[other_idx] == sig_input:
                            func = "silu"
                            pwp_output = mul_cand.output[0]
                            nodes_to_remove.append(mul_cand)

                new_node = self._make_pwpolyf_node(
                    sig_input,
                    pwp_output,
                    func,
                    pwp_in_shape,
                    idt,
                    "PWPolyF_" + node.name,
                )
                graph.node.insert(node_ind, new_node)
                for nd in nodes_to_remove:
                    graph.node.remove(nd)
                graph_modified = True

            # Case 4: Erf-based GELU (dynamo=True / opset < 20)
            # Div(x, sqrt(2)) → Erf → Add(_, 1) → Mul(0.5, _) → Mul(x, _)
            elif node.op_type == "Erf":
                match = self._match_erf_gelu(model, node)
                if match is None:
                    continue
                pwp_input, pwp_output, nodes_to_remove = match
                pwp_in_shape = model.get_tensor_shape(pwp_input)
                if pwp_in_shape is None or len(pwp_in_shape) < 1:
                    continue
                idt = model.get_tensor_datatype(pwp_input)
                if idt != DataType["FLOAT32"]:
                    continue

                new_node = self._make_pwpolyf_node(
                    pwp_input,
                    pwp_output,
                    "gelu",
                    pwp_in_shape,
                    idt,
                    "PWPolyF_" + node.name,
                )
                graph.node.insert(node_ind, new_node)
                for nd in nodes_to_remove:
                    graph.node.remove(nd)
                graph_modified = True

        return (model, graph_modified)

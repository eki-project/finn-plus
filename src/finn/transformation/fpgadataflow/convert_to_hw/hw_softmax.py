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

"""Convert Softmax nodes to HWSoftmax HW layers."""

from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from typing import Literal

from finn.util.exception import FINNInternalError


class InferHWSoftmax(Transformation):
    """Infers a regular softmax node without merging the multithreshold
    and setting the softmax to perform the quantisation.
    """

    def __init__(self) -> None:
        """Infers a regular softmax node without merging the multithreshold
        and setting the softmax to perform the quantisation.
        """
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, Literal[False]]:
        """Apply the transformation."""
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            if n.op_type == "Softmax":
                if (input_shape := model.get_tensor_shape(n.input[0])) is None:
                    raise FINNInternalError(
                        "Expected shape information to be available during convert to hw."
                    )

                idt0 = model.get_tensor_datatype(n.input[0])
                odt0 = model.get_tensor_datatype(n.output[0])
                new_node = helper.make_node(
                    "HWSoftmax",
                    [n.input[0]],  # input tensor(s)
                    [n.output[0]],  # output tensor(s)
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    ifm_dim=input_shape,
                    input_data_type=idt0.name,
                    output_data_type=odt0.name,
                    name=n.name,
                    SIMD=1,
                    NumChannels=input_shape[-1],
                    cpp_interface="hls_vector",
                    hls_style="freerunning",
                )
                graph.node.insert(node_ind, new_node)
                graph.node.remove(n)
        return (model, graph_modified)

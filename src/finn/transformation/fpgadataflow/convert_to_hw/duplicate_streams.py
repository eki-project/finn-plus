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

"""Insert DuplicateStreams HW layers for tensors with fanout >= 2."""

from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

from finn.util.exception import FINNInternalError


class InferDuplicateStreamsLayer(Transformation):
    """Insert a DuplicateStreams HW layer for any tensor with fanout >= 2."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transformation to insert DuplicateStreams HW layers where needed."""
        graph = model.graph
        graph_modified = False
        # check first if global input is split
        successors = model.find_consumers(graph.input[0].name)
        dt = model.get_tensor_datatype(graph.input[0].name)
        if successors is not None and len(successors) >= 2:
            output_tensor = graph.input[0].name
            n_outputs = len(successors)

            # create clone tensors
            if (out_shape := model.get_tensor_shape(output_tensor)) is None:
                raise FINNInternalError("Expected shape information on output")
            out_tensor_clones = []
            for _ in range(n_outputs):
                clone = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, out_shape
                )
                model.graph.value_info.append(clone)
                out_tensor_clones += [clone.name]

            num_ch = int(out_shape[-1])
            vecs = out_shape[:-1]

            # create node with no parallelization first
            pe = 1

            dup_node = helper.make_node(
                "DuplicateStreams",
                [output_tensor],
                out_tensor_clones,
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                NumChannels=num_ch,
                PE=pe,
                inputDataType=dt.name,
                numInputVectors=vecs,
                NumOutputStreams=n_outputs,
                outFIFODepths=[2] * n_outputs,
                name="DuplicateStreams_" + output_tensor,
                cpp_interface="hls_vector",
                hls_style="freerunning",
            )

            graph.node.insert(0, dup_node)

            # connect successors to out tensor clone
            clone_idx = 0
            for successor in successors:
                for i, succ_input in enumerate(successor.input):
                    if succ_input == output_tensor:
                        successor.input[i] = out_tensor_clones[clone_idx]
                        clone_idx += 1
                        # if one node has multiple connections to the same output
                        # find_direct_successors will return one node per input
                        # so break the inner loop will result in correct behaviour
                        break
            graph_modified = True

        for node_ind, node in enumerate(graph.node, start=1):
            for output_tensor in node.output:
                successors = model.find_consumers(output_tensor)
                # check if this tensor is also a global output
                is_global_output = any(out.name == output_tensor for out in graph.output)
                # determine total number of consumers (successors + global output)
                num_successors = len(successors) if successors is not None else 0
                total_consumers = num_successors + (1 if is_global_output else 0)

                if total_consumers >= 2:
                    new_global_output_tensor = None
                    n_outputs = total_consumers

                    dt = model.get_tensor_datatype(output_tensor)

                    # create clone tensors
                    out_shape = model.get_tensor_shape(output_tensor)
                    if (out_shape := model.get_tensor_shape(output_tensor)) is None:
                        raise FINNInternalError("Expected shape information on output")
                    out_tensor_clones = []
                    for i in range(n_outputs):
                        clone = helper.make_tensor_value_info(
                            model.make_new_valueinfo_name(), TensorProto.FLOAT, out_shape
                        )
                        # if one is a global output reserve
                        # the last out tensor clone for that connection
                        if i == (n_outputs - 1) and is_global_output:
                            new_global_output_tensor = clone
                        # else add it to the value info container
                        else:
                            model.graph.value_info.append(clone)
                        out_tensor_clones += [clone.name]

                    num_ch = int(out_shape[-1])
                    vecs = out_shape[:-1]

                    # create node with no parallelization first
                    pe = 1

                    dup_node = helper.make_node(
                        "DuplicateStreams",
                        [output_tensor],
                        out_tensor_clones,
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        NumChannels=num_ch,
                        PE=pe,
                        inputDataType=dt.name,
                        numInputVectors=vecs,
                        NumOutputStreams=n_outputs,
                        outFIFODepths=[2] * n_outputs,
                        name="DuplicateStreams_" + node.name,
                        cpp_interface="hls_vector",
                        hls_style="freerunning",
                    )

                    graph.node.insert(node_ind, dup_node)

                    # connect successors to out tensor clone
                    clone_idx = 0
                    for successor in successors:
                        for i, succ_input in enumerate(successor.input):
                            if succ_input == output_tensor:
                                successor.input[i] = out_tensor_clones[clone_idx]
                                clone_idx += 1
                                # if one node has multiple connections to the same output
                                # find_direct_successors will return one node per input
                                # so break the inner loop will result in correct behaviour
                                break

                    # if the tensor is a global output, connect the last clone to it
                    if is_global_output:
                        for i, graph_out in enumerate(graph.output):
                            if graph_out.name == output_tensor:
                                if new_global_output_tensor is None:
                                    raise FINNInternalError(
                                        "Cant copy from new_global_output_tensor, because not set."
                                    )
                                graph.output[i].CopyFrom(new_global_output_tensor)
                                break

                    graph_modified = True

        if graph_modified:
            model = model.transform(SortGraph())
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

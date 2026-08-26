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

"""Infer Pool HW layers from lowered pooling, i.e., Im2Col+Reduce."""

from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name
from typing import TYPE_CHECKING, Literal, cast

from finn.util.exception import FINNInternalError

if TYPE_CHECKING:
    import numpy as np


class InferPoolFromReduce(Transformation):
    """Infer pooling hardware from lowered pooling, i.e., Im2Col+Reduce."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, Literal[False]]:
        """Apply transformation to convert lowered pooling to hardware."""
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False

        # Enumerate all node in the graph and check for standalone standard ONNX
        # padding operators
        for index, node in enumerate(graph.node):
            if node.op_type in {"ReduceMax", "ReduceSum", "ReduceMean"}:
                # Reduction axes must be constants to turn this into hardware
                if (axes := model.get_initializer(node.input[1])) is None:
                    continue

                # The input to the reduction must be produced by a Reshape
                # operator unpacking the channel axis from the kernel shape
                if (reshape := model.find_producer(node.input[0])) is None:
                    continue

                if reshape.op_type != "Reshape":
                    continue

                # The reshape must be static, i.e., the shape parameter is
                # constant
                if (
                    shape := cast("np.ndarray|None", model.get_initializer(reshape.input[1]))
                ) is None:
                    continue

                # Reduction must operate on the second to last axis, which is
                # the (spatial) extent of the pooling window
                if list(axes) != [-2] and list(axes) != [len(shape) - 2]:
                    continue

                # The overall input must be produced from a sliding window input
                # generators, i.e., Im2Col operator
                if (im2col := model.find_producer(reshape.input[0])) is None:
                    continue

                if im2col.op_type != "Im2Col":
                    continue

                # Get the current input datatype annotation (Im2Col and Reshape
                # do not modify the datatype)
                idt = model.get_tensor_datatype(im2col.input[0])

                # Fallback: Assume output to be the same as the input and the
                # accumulator to be zero-sized
                odt, accum_bits = idt, 0

                # Simple type inference depending on the matched reduction
                # operator: Could be refined by minimize_accumulator_width
                if node.op_type == "ReduceMax":
                    accum_bits = 0
                    odt = idt

                if node.op_type == "ReduceSum":
                    # Minimum and maximum accumulated value to expect from
                    # reducing the input type over the reduction axis
                    minimum = shape[-2] * idt.min()
                    maximum = shape[-2] * idt.max()
                    # The output datatype must be able to fit the larger
                    # magnitude of the two
                    if abs(minimum) > abs(maximum):
                        odt = DataType.get_smallest_possible(minimum)
                    else:
                        odt = DataType.get_smallest_possible(maximum)
                    # Accumulator size is the same as the output size
                    accum_bits = odt.bitwidth()

                if node.op_type == "ReduceMean":
                    # Minimum and maximum accumulated value to expect from
                    # reducing the input type over the reduction axis
                    minimum = shape[-2] * idt.min()
                    maximum = shape[-2] * idt.max()
                    # The accumulator datatype must be able to fit the larger
                    # magnitude of the two
                    if abs(minimum) > abs(maximum):
                        acc = DataType.get_smallest_possible(minimum)
                    else:
                        acc = DataType.get_smallest_possible(maximum)
                    # Accumulator size if the bitwidth of this accumulator type
                    accum_bits = acc.bitwidth()
                    # The output type is the same as the input, as it is
                    # averaged over, i.e., divided by, the kernel size
                    odt = idt

                # Annotate the output to use the inferred type instead of the
                # current type annotation
                model.set_tensor_datatype(node.output[0], odt)

                # Lookup the pooling backend function corresponding to this
                # reduction operator
                pool_fxn = {
                    "ReduceMax": "MaxPool",
                    "ReduceSum": "AccPool",
                    "ReduceMean": "AvgPool",
                }[node.op_type]

                # This is indeed a supported pooling operator in its lowered
                # form: Keep Im2Col but replace Reshape+Reduce by HW operator
                attr = get_by_name(im2col.attribute, "kernel_size")
                if attr is None:
                    raise FINNInternalError(f"{node.name}: kernel_size not known for Im2Col.")
                pool = helper.make_node(
                    # This is the pooling custom hardware operator from the FINN
                    # custom domain
                    op_type="Pool",
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    # Connect the new operator to make use of the old inputs and
                    # outputs
                    inputs=[im2col.output[0]],
                    outputs=[node.output[0]],
                    # Pooling needs to know the input/output dimensions and the
                    # size of the pooling window
                    Channels=shape[-1],
                    KernelSize=attr.ints,
                    OutImgDims=shape[1:-2],
                    # Select the pooling backend implementation
                    Function=pool_fxn,
                    # Set parallelism to cover all input (also output) channels
                    PE=shape[-1],
                    # Configure the size of the internal accumulator if
                    # applicable (MaxPool ignores this)
                    AccumBits=accum_bits,
                    # Set the names of the input/output datatype
                    InputDataType=idt.name,
                    OutputDataType=odt.name,
                    # Pooling backend already uses the new hls::vector interface
                    cpp_interface="hls_vector",
                )

                # The input generator needs to be switched into depthwise mode,
                # as pooling does not reduce/expand along the channels
                getCustomOp(im2col).set_nodeattr("depthwise", 1)

                # Insert the pooling node into the graph, but do not remove the
                # old nodes as they might still have other consumers
                graph.node.insert(index, pool)

                # The reduction can always be removed, all consumers are rewired
                # to use the pooling output
                graph.node.remove(node)

                # If the reshape has only a single consumer, we can remove this
                # from the graph
                if len(model.find_consumers(reshape.output[0])) <= 1:
                    graph.node.remove(reshape)

        # Re-do shape and data type annotations after potential changes to the
        # model graph
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified

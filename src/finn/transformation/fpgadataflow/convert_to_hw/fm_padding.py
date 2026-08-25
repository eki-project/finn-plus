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

"""Convert standalone ONNX Pad layers to FMPadding HW layers."""

from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name

from finn.util.exception import FINNInternalError


class InferFMPadding(Transformation):
    """Convert Pad layers to FMPadding layers."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transformation to the entire model graph."""
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False

        # Enumerate all node in the graph and check for standalone standard ONNX
        # padding operators
        for index, node in enumerate(graph.node):
            if node.op_type == "Pad":
                # FMPadding only implements constant padding
                if (mode := get_by_name(node.attribute, "mode")) is not None and mode.s.decode(
                    "ascii"
                ) != "constant":
                    continue
                if (ishape := model.get_tensor_shape(node.input[0])) is None:
                    raise FINNInternalError(
                        "Expected shape information to exist during convert to hw."
                    )

                # Input shape must describe 4d image layout to be compatible
                # with the FMPadding operator
                if len(ishape) != 4:
                    continue

                # Padding axes must be constant initializer tensors, we cannot
                # do runtime dynamic behavior
                if (axes := model.get_initializer(node.input[3])) is None:
                    continue

                # Assuming NHWC layout as expected by FMPadding, the axes must
                # be the first two (HW) following the batch dimension
                if list(axes) != [1, 2]:
                    continue

                # FMPadding only implements constant zero padding at the moment
                if (model.get_initializer(node.input[2])) != 0:
                    continue

                # Padding amount for each dimension must be constant and match
                # the HW dimensions
                if (pads := model.get_initializer(node.input[1])) is None:
                    continue

                if len(pads) != 4:
                    continue

                # Configure the FINN CustomOp replacement of the pad operator
                padding = helper.make_node(
                    "FMPadding",
                    [node.input[0]],
                    [*node.output],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    ImgDim=ishape[1:3],
                    Padding=list(pads),
                    NumChannels=ishape[-1],
                    inputDataType=model.get_tensor_datatype(node.input[0]).name,
                    SIMD=ishape[-1],
                    name="FMPadding_" + node.name,
                )

                graph.node.insert(index, padding)
                graph.node.remove(node)

                # Consider the graph to be modified, triggering exhaustive
                # re-application of this transformation
                graph_modified = True
                # Exiting here triggers type and shape inference and cleanup
                # after each transformed node. This helps QONNX to behave
                # better/more consistent in certain cases...
                break

        # Re-do shape and data type annotations after potential changes to the
        # model graph
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified

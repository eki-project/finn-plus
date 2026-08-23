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

"""Convert the ONNX Reshape operator to the corresponding FINN HW custom operation."""

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import remove_by_name

# Base class for all FINN custom ops, here just used for type-hinting
from typing import TYPE_CHECKING

from finn.util.basic import getHWCustomOp

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class InferReshape(Transformation):
    """Converts ONNX Reshape operator to the corresponding HWCustomOp."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transform to convert Reshape operations hardware."""
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False

        # Transplant each Reshape operator into the custom domain
        for _index, node in enumerate(graph.node):
            if node.op_type == "Reshape":
                # Skip already converted nodes
                if node.domain == "finn.custom_op.fpgadataflow":
                    continue

                # Skip if this is not a static-shape reshape operation
                if (out_shape := model.get_initializer(node.input[1])) is None:
                    continue

                # Skip if any of the axes is set to 0 to avoid handling
                # differences between allowzero=0 and allowzero=1...
                # TODO: This could be supported by implementing simple shape
                #  inference logic here or by normalizing shapes via streamline
                if np.any(out_shape == 0):
                    continue

                # Skip if the input shape is not statically known
                if (inp_shape := model.get_tensor_shape(node.input[0])) is None:
                    continue

                # Skip if there are more than one axes set to -1
                if list(out_shape).count(-1) > 1:
                    continue

                # The shape is allowed to contain at most one -1, in which case
                # the value is inferred from the size of the tensor and the
                # remaining dimensions
                if list(out_shape).count(-1) == 1:
                    # The following expression assumes target to be a list which
                    # allows item assignment (might be a tuple depending on the
                    # origin of target)
                    out_shape = list(out_shape)
                    # Replace the inferred dimension by the number of elements
                    # missing from the target dimension compared to the input
                    # dimension
                    out_shape[out_shape.index(-1)] = int(
                        np.prod(inp_shape) / np.abs(np.prod(out_shape))
                    )

                # Transplant from the standard ONNX domain into the FINN domain
                node.domain = "finn.custom_op.fpgadataflow"
                # Now we can get the CustomOp wrapper instance providing easier
                # attribute access
                reshape: HWCustomOp = getHWCustomOp(node)
                # Set the backend attribute to mark this an operation supported
                # to be implemented on an FPGA by FINN
                reshape.set_nodeattr("backend", "fpgadataflow")

                # Remove the old allowzero attribute which is not handled by the
                # hardware operator
                remove_by_name(node.attribute, "allowzero")

                # Cast to python-int to match QONNX's type checking
                # This might fail, since technically, out_shape can be of type "Any"
                out_shape = [int(x) for x in out_shape]  # type: ignore

                # The reshape hardware operator statically annotates both, the
                # input and output shape, as node attributes
                reshape.set_nodeattr("inp_shape", list(inp_shape))
                reshape.set_nodeattr("out_shape", list(out_shape))

                # The input and output type of the reshaped tensor is the same
                # and needs to be statically annotated as an attribute
                reshape.set_nodeattr("dtype", model.get_tensor_datatype(node.input[0]).name)

                # Finally, there is no second shape input to the hardware
                # operator, thus delete this from the node input list
                node.input.pop(1)

        # Re-do shape and data type annotations after potential changes to the
        # model graph
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified

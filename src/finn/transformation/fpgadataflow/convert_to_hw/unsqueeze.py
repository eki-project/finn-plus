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

"""Convert the ONNX Unsqueeze operation to the corresponding FINN HW custom operation."""

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from typing import cast

from finn.util.basic import getHWCustomOp
from finn.util.exception import FINNUserError


class InferUnsqueeze(Transformation):
    """Convert the Unsqueeze operation to the corresponding FINN custom operation."""

    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper):  # noqa
        """Apply the transform to convert Unsqueeze operations to FINN custom ops."""
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for _index, node in enumerate(graph.node):
            # Handles Squeeze ONNX operations
            if node.op_type == "Unsqueeze":
                # Skip already converted nodes
                if node.domain == "finn.custom_op.fpgadataflow":
                    # Skip without warning
                    continue
                # Transplant this operator into our FINN domain
                node.domain = "finn.custom_op.fpgadataflow"  # noqa: Duplicate
                # Now we can get the CustomOp wrapper instance providing easier
                # attribute access
                inst = getHWCustomOp(node)
                # Set the backend attribute to mark this an operation supported
                # to be implemented on an FPGA by FINN
                inst.set_nodeattr("backend", "fpgadataflow")
                # Ge the input and output tensor names
                inp, out = node.input[0], node.output[0]
                ishape = model.get_tensor_shape(inp)
                oshape = model.get_tensor_shape(out)
                if ishape is None or oshape is None:
                    raise FINNUserError(
                        f"{node.name}: Expected shape information to be available "
                        "on input and output."
                    )
                # Set input/output shape and datatype node attributes required
                # by FINN custom op
                inst.set_nodeattr("inp_dtype", str(model.get_tensor_datatype(inp)))
                inst.set_nodeattr("inp_shape", ishape)  # type: ignore
                inst.set_nodeattr("out_dtype", str(model.get_tensor_datatype(out)))
                inst.set_nodeattr("out_shape", oshape)  # type: ignore
                if len(node.input) > 1:
                    axes = cast("np.ndarray|None", model.get_initializer(node.input[1]))
                    if axes is None:
                        raise FINNUserError(
                            f"{node.name}: For more than one input the axes input has "
                            "to be set and needs to be static."
                        )
                    if np.ndim(axes) == 0:
                        # Fix axes input initializer by converting from scalar (0D) to 1D array
                        axes = np.array([axes])
                        model.set_initializer(node.input[1], axes)
                        model.set_tensor_shape(node.input[1], axes.shape)
                    # Set axes attribute (used by older opsets) even if axes is provided as input
                    inst.set_nodeattr("axes", list(axes))
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

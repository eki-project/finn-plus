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

"""Convert Upsample and Resize nodes to UpsampleNearestNeighbour HW layers."""

import qonnx.core.data_layout as data_layout
from onnx import AttributeProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.util.basic import get_by_name
from typing import TYPE_CHECKING, cast

from finn.util.exception import FINNUserError
from finn.util.logging import log

if TYPE_CHECKING:
    import numpy as np


class InferUpsample(Transformation):
    """Convert Upsample and Resize nodes to UpsampleNearestNeighbour nodes."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transformation to infer UpsampleNearestNeighbour nodes."""
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Upsample" or n.op_type == "Resize":
                # Extract mode and scales and input shape
                mode = cast("AttributeProto", get_by_name(n.attribute, "mode")).s.decode("ascii")
                if n.op_type == "Upsample":
                    scales = cast("np.ndarray | None", model.get_initializer(n.input[1]))
                elif len(n.input) == 2:
                    # Resize version 10
                    scales = cast("np.ndarray | None", model.get_initializer(n.input[1]))
                elif len(n.input) == 3:
                    # Resize version 11 and up (no size input)
                    scales = cast("np.ndarray | None", model.get_initializer(n.input[2]))
                elif len(n.input) == 4:
                    # Resize version 11 and up
                    scales_init = cast("np.ndarray | None", model.get_initializer(n.input[2]))
                    sizes_init = cast("np.ndarray | None", model.get_initializer(n.input[3]))
                    scales_exists = scales_init is not None and len(scales_init) != 0
                    sizes_exists = sizes_init is not None and len(sizes_init) != 0
                    if not (scales_exists ^ sizes_exists):
                        raise FINNUserError(
                            f"{n.name}: Either scales or the target output size must "
                            "be specified. Specifying both is prohibited."
                        )
                    if scales_exists:
                        # Scales input
                        scales = scales_init
                    else:
                        # Convert sizes to scales
                        if (data_input_size := model.get_tensor_shape(n.input[0])) is None:
                            raise FINNUserError(
                                f"{n.name}: Expected shape information to be available on input."
                            )
                        scales = cast("np.ndarray", sizes_init) / data_input_size
                else:
                    raise FINNUserError(f"{n.name}: Unsupported number of inputs for Resize.")
                if scales is None:
                    raise FINNUserError(
                        f"{n.name}: Cannot infer UpsampleNearestNeighbour, scales are not static."
                    )
                if (in_shape := model.get_tensor_shape(n.input[0])) is None:
                    raise FINNUserError(
                        f"{n.name}: Expected shape information to be available on input."
                    )

                dt = model.get_tensor_datatype(n.input[0])
                if not dt.is_integer():
                    log.warning(f"{n.name}: Input not int. Can't infer UpsampleNearestNeighbour.")
                    continue

                if model.get_tensor_layout(n.input[0]) != data_layout.NHWC:
                    log.warning(f"{n.name}: Input not NHWC. Can't infer UpsampleNearestNeighbour.")
                    continue

                # Check that the parameters are okay
                if mode != "nearest":
                    raise FINNUserError(
                        f"{n.name}: Upsampling is only supported for the mode nearest."
                    )
                if len(in_shape) != 4:
                    raise FINNUserError(f"{n.name}: Upsampling is only supported for 4D inputs.")
                if scales.shape != (4,):
                    raise FINNUserError(f"{n.name}: Upsampling is only supported for 4D scales.")
                if not (scales >= 1).all():
                    raise FINNUserError(
                        f"{n.name}: Upsampling is only supported for scales "
                        "which are larger or equal 1 in all dimensions."
                    )

                # Assumes nhwc layout for scales and input
                if not (scales[0] == scales[3] == 1):
                    raise FINNUserError(
                        f"{n.name}: Upsampling is only supported for scales with "
                        "the first and last dimensions being 1 in NHWC."
                    )

                # Extract information for HW node
                hi = in_shape[1]
                wi = in_shape[2]
                ho = round(hi * scales[1])
                wo = round(wi * scales[2])
                num_channels = in_shape[-1]
                batch_size = in_shape[0]
                input_data_type = dt.name

                # Insert the HWCustomOp node
                upsample_hw_node = helper.make_node(
                    "UpsampleNearestNeighbour",
                    [n.input[0]],
                    [n.output[0]],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    SIMD=1,
                    HO=ho,
                    WO=wo,
                    HI=hi,
                    WI=wi,
                    NumChannels=num_channels,
                    inputDataType=input_data_type,
                    batchSize=batch_size,
                    name="UpsampleNearestNeighbour_" + n.name,
                    cpp_interface="hls_vector",
                    hls_style="freerunning",
                )

                # Remove the old node
                graph.node.insert(node_ind, upsample_hw_node)
                # remove old nodes
                graph.node.remove(n)
                graph_modified = True
        return (model, graph_modified)

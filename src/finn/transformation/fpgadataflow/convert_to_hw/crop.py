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

"""Convert Gather layers describing a crop into Crop HW layers."""

import numpy as np
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name
from typing import cast

from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log


def elements_are_consecutive(indices: "np.ndarray") -> bool:
    """Are elements consecutive (max diff. 1 between all adjacent elements)?."""
    if indices.size == 1:
        return True
    indices.sort()
    return bool(np.all(np.diff(indices) == 1))


class InferCrop(Transformation):
    """Find gather layers that can be converted into a Crop layer
    and replace them with a Crop layer.
    """

    def __init__(self) -> None:
        """Find gather layers that can be converted into a Crop layer
        and replace them with a Crop layer.
        """
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transformation."""
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Gather":
                # ensure that the indices input is an initializer
                if model.get_initializer(n.input[1]) is None:
                    continue

                # ensure that the axis is among the two innermost dimensions
                if (input_shape := model.get_tensor_shape(n.input[0])) is None:
                    log.warning(f"{n.name}: Expected input tensor shape to be set; Skipping node.")
                    continue
                if len(input_shape) <= 1:
                    raise FINNUserError(
                        f"{n.name}: Input shape needs to be at least 2D to be converted to Crop."
                    )

                max_index = len(input_shape) - 1
                attr = get_by_name(n.attribute, "axis")
                if attr is None:
                    raise FINNUserError(f"{n.name}: Attribute 'axis' not set for Gather operation.")
                axis = attr.i
                if len(input_shape) >= 3:
                    if axis not in [max_index - 1, max_index - 2]:
                        raise FINNUserError(
                            f"{n.name}: Crop operates on height and width of the input, "
                            "assuming (N)HWC layout."
                        )
                else:
                    if axis != max_index - 1:
                        raise FINNUserError(
                            f"{n.name}: Crop operates on width of the input, for 2D input "
                            "assuming WC layout."
                        )
                is_vertical = axis == max_index  # otherwise horizontal
                if is_vertical:
                    raise FINNInternalError(
                        f"{n.name}: axis validation above should already rule out vertical crops"
                    )

                # assume that the indices input is an int64 scalar or array
                indices = cast("np.ndarray | None", model.get_initializer(n.input[1]))
                if indices is None:
                    raise FINNUserError(f"{n.name}: Initializer for node not set on input 1.")
                indices = cast("np.ndarray", indices)
                if indices.dtype != np.int64:
                    raise FINNInternalError(f"{n.name}: Indices must be int64.")
                # Handle both scalar (0-d) and array cases; a scalar index is always consecutive
                indices_to_check = np.array([indices.item()]) if indices.ndim == 0 else indices
                if not elements_are_consecutive(indices_to_check):
                    raise FINNInternalError(f"{n.name}: Indices must be consecutive.")

                idt0 = model.get_tensor_datatype(n.input[0])

                crop_north = 0
                crop_east = 0
                crop_west = 0
                crop_south = 0
                num_inp_vec = [0]

                if len(input_shape) >= 3:
                    height_ind = len(input_shape) - 3
                    width_ind = len(input_shape) - 2
                    channels_ind = len(input_shape) - 1

                    height = input_shape[height_ind]
                    width = input_shape[width_ind]
                    channels = input_shape[channels_ind]
                    # save other dimensions in numInpVectors
                    if len(input_shape) > 3:
                        num_inp_vec = list(input_shape[:height_ind])

                    crop_min = int(np.min(indices_to_check))
                    crop_max = input_shape[axis] - int(np.max(indices_to_check)) - 1

                    if axis == height_ind:
                        crop_north = crop_min
                        crop_south = crop_max
                    elif axis == width_ind:
                        crop_west = crop_min
                        crop_east = crop_max

                elif len(input_shape) == 2:
                    # if there are only two dimensions, assume
                    height = 0
                    width_ind = len(input_shape) - 2
                    channels_ind = len(input_shape) - 1
                    width = input_shape[width_ind]
                    channels = input_shape[channels_ind]

                    # axis is on width dimension
                    crop_west = int(np.min(indices_to_check))
                    crop_east = input_shape[axis] - int(np.max(indices_to_check)) - 1
                else:
                    raise FINNUserError(
                        f"{n.name}: One dimensional input for Crop node is not supported"
                    )

                # create and insert new node
                new_node = helper.make_node(
                    "Crop",
                    [n.input[0]],  # input tensor(s)
                    [n.output[0]],  # output tensor(s)
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    DataType=idt0.name,
                    name="Crop" + n.name,
                    SIMD=1,
                    ImgDim=[height, width],
                    NumChannels=channels,
                    CropNorth=crop_north,
                    CropEast=crop_east,
                    CropWest=crop_west,
                    CropSouth=crop_south,
                    numInputVectors=num_inp_vec,
                    cpp_interface="hls_vector",
                    hls_style="freerunning",
                )
                graph.node.insert(node_ind, new_node)
                graph.node.remove(n)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

# Copyright (C) 2023, Advanced Micro Devices, Inc.
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

"""Nearest-neighbour upsampling hardware custom operator."""

import numpy as np
import onnxruntime as rt
from onnx import NodeProto, TensorProto, helper
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow import register_custom_op
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.logging import log

if TYPE_CHECKING:
    from onnx import GraphProto

# Type of the dictionary returned by get_nodeattr_types: maps attribute names to
# their (dtype, required, default[, allowed_values]) specification tuples
NodeAttrTypes = dict[
    str,
    tuple[str, bool, int | float | str | bool | np.ndarray | list]
    | tuple[str, bool, int | float | str | bool | np.ndarray | list, set | None],
]


@register_custom_op
class UpsampleNearestNeighbour(HWCustomOp):
    """Abstraction layer for HW implementation of UpsampleNearestNeighbour."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {
            "SIMD": ("i", True, 0),
            # Height, width of the output feature map
            "HO": ("i", True, 0),
            "WO": ("i", True, 0),
            # Height, width of the input feature map
            "HI": ("i", True, 0),
            "WI": ("i", True, 0),
            # Amount of channels of the input feature map
            "NumChannels": ("i", True, 0),
            # FINN input datatype
            "inputDataType": ("s", True, ""),
            # Batch size
            "batchSize": ("i", False, 1),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    @property
    def simd(self) -> int:
        """Get the SIMD parallelism."""
        return cast("int", self.get_nodeattr("SIMD"))

    @property
    def num_channels(self) -> int:
        """Get the number of input feature map channels."""
        return cast("int", self.get_nodeattr("NumChannels"))

    @property
    def hi(self) -> int:
        """Get the input feature map height."""
        return cast("int", self.get_nodeattr("HI"))

    @property
    def wi(self) -> int:
        """Get the input feature map width."""
        return cast("int", self.get_nodeattr("WI"))

    @property
    def ho(self) -> int:
        """Get the output feature map height."""
        return cast("int", self.get_nodeattr("HO"))

    @property
    def wo(self) -> int:
        """Get the output feature map width."""
        return cast("int", self.get_nodeattr("WO"))

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return cast("int", self.get_nodeattr("batchSize"))

    def get_exp_cycles(self) -> int:
        """Return exp cycles."""
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return normal input shape."""
        return (self.batch_size, self.hi, self.wi, self.num_channels)

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return normal output shape."""
        return (self.batch_size, self.ho, self.wo, self.num_channels)

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded input shape."""
        spatial_shape = list(self.get_normal_input_shape())[:-1]
        folds = self.num_channels // self.simd
        return (*spatial_shape, folds, self.simd)

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded output shape."""
        spatial_shape = list(self.get_normal_output_shape())[:-1]
        folds = self.num_channels // self.simd
        return (*spatial_shape, folds, self.simd)

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer node datatype."""
        node = self.onnx_node
        # data type stays the same
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            log.warning(
                f"inputDataType changing for {node.name}: {self.get_input_datatype()!s} -> {idt!s} "
            )
        self.set_nodeattr("inputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of input."""
        return DataType[cast("str", self.get_nodeattr("inputDataType"))]

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of output (same as input datatype)."""
        return self.get_input_datatype()

    def get_instream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return instream width."""
        return self.get_input_datatype().bitwidth() * self.simd

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return outstream width."""
        return self.get_output_datatype().bitwidth() * self.simd

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute node."""
        # create a standard resize node to help calculate the result
        node = self.onnx_node
        inp_values = context[node.input[0]]
        ishape = inp_values.shape
        scales_val = [1, round(self.ho / self.hi), round(self.wo / self.wi), 1]
        oshape = context[node.output[0]].shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        scales = helper.make_tensor_value_info("scales", TensorProto.FLOAT, [4])
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)
        node_resize = helper.make_node(
            "Resize",
            inputs=[node.input[0], "", "scales"],
            outputs=[node.output[0]],
            mode="nearest",
        )
        graph_resize = helper.make_graph(
            nodes=[node_resize],
            name="single-resize-exec",
            inputs=[inp, scales],
            outputs=[outp],
        )

        opset_imports = [helper.make_opsetid("", 13)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_resize = qonnx_make_model(graph_resize, **onnx_kwargs)
        idict = {node.input[0]: inp_values, "scales": scales_val}
        sess = rt.InferenceSession(model_resize.SerializeToString())
        result = sess.run(None, idict)
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(oshape)

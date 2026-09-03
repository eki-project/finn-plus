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

"""Top-K label-selection hardware custom operator."""

import numpy as np
import onnxruntime as rt
from onnx import NodeProto, TensorProto, helper
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model, roundup_to_integer_multiple
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.exception import FINNInternalError

if TYPE_CHECKING:
    from onnx import GraphProto

# Type of the dictionary returned by get_nodeattr_types: maps attribute names to
# their (dtype, required, default[, allowed_values]) specification tuples
NodeAttrTypes = dict[
    str,
    tuple[str, bool, int | float | str | bool | np.ndarray | list]
    | tuple[str, bool, int | float | str | bool | np.ndarray | list, set | None],
]


class LabelSelect(HWCustomOp):
    """Abstraction layer for HW implementation of LabelSelect.

    Emits the indices (labels) of the ``K`` largest input values.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)
        odt_name = cast("str", self.get_nodeattr("outputDataType"))
        if odt_name == "":
            # If not provided compute min size
            labels = cast("int", self.get_nodeattr("Labels"))
            odt = DataType.get_smallest_possible(labels - 1)
            # ensure a datatype divisible by 8-bits in case this is the last node
            bw = roundup_to_integer_multiple(odt.bitwidth(), 8)
            new_odt_name = odt.name.replace(str(odt.bitwidth()), str(bw))
            self.set_nodeattr("outputDataType", DataType[new_odt_name].name)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {
            "Labels": ("i", True, 0),
            "PE": ("i", True, 0),
            "K": ("i", True, 0),
            # FINN DataTypes for input
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", False, ""),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    @property
    def labels(self) -> int:
        """Get the number of input labels/classes."""
        return cast("int", self.get_nodeattr("Labels"))

    @property
    def pe(self) -> int:
        """Get the PE parallelism."""
        return cast("int", self.get_nodeattr("PE"))

    @property
    def k(self) -> int:
        """Get the number of top labels to select."""
        return cast("int", self.get_nodeattr("K"))

    @property
    def num_input_vectors(self) -> list[int]:
        """Get the number of input vectors along the non-label axes."""
        return list(cast("list[int]", self.get_nodeattr("numInputVectors")))

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return normal input shape."""
        return (*self.num_input_vectors, self.labels)

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded input shape."""
        if self.labels % self.pe != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: PE ({self.pe}) must divide Labels ({self.labels})"
            )
        folds = self.labels // self.pe
        return (*self.num_input_vectors, folds, self.pe)

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return normal output shape."""
        return (*self.num_input_vectors, self.k)

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded output shape."""
        return (*self.num_input_vectors, self.k, 1)

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer node datatype."""
        node = self.onnx_node
        # check input datatype against property
        idt = model.get_tensor_datatype(node.input[0])
        self.set_nodeattr("inputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], self.get_output_datatype())

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of input."""
        return DataType[cast("str", self.get_nodeattr("inputDataType"))]

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of output."""
        return DataType[cast("str", self.get_nodeattr("outputDataType"))]

    def get_instream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return input stream width."""
        return self.pe * self.get_input_datatype().bitwidth()

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return output stream width."""
        return self.get_output_datatype().bitwidth()

    def get_number_output_values(self) -> int:
        """Return number output values."""
        return self.k

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute node.

        Uses an ONNX Runtime ``TopK`` node to compute the result.
        """
        node = self.onnx_node
        inp_values = context[node.input[0]]
        oshape = context[node.output[0]].shape
        ishape = inp_values.shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        k_inp = helper.make_tensor_value_info("k_inp", TensorProto.INT64, [1])
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.INT64, oshape)
        val_outp = helper.make_tensor_value_info("val_outp", TensorProto.FLOAT, oshape)
        node_topk = helper.make_node(
            "TopK",
            inputs=[node.input[0], "k_inp"],
            outputs=["val_outp", node.output[0]],
        )
        graph_topk = helper.make_graph(
            nodes=[node_topk],
            name="single-add-exec",
            inputs=[inp, k_inp],
            outputs=[val_outp, outp],
        )

        model_topk = qonnx_make_model(graph_topk)
        idict = {node.input[0]: inp_values, "k_inp": [self.k]}
        sess = rt.InferenceSession(model_topk.SerializeToString())
        result = sess.run(None, idict)
        context[node.output[0]] = np.asarray(result[1], dtype=np.float32).reshape(oshape)

    def get_exp_cycles(self) -> int:
        """Return exp cycles."""
        return int(self.labels / self.pe)

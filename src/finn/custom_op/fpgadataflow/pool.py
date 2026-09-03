# Copyright (C) 2024, Advanced Micro Devices, Inc.
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

"""Generic pooling hardware custom operator (MaxPool, AvgPool, AccPool, QuantAvgPool)."""

import numpy as np
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.exception import FINNInternalError, FINNUserError

if TYPE_CHECKING:
    from onnx import GraphProto

# Type of the dictionary returned by get_nodeattr_types: maps attribute names to
# their (dtype, required, default[, allowed_values]) specification tuples
NodeAttrTypes = dict[
    str,
    tuple[str, bool, int | float | str | bool | np.ndarray | list]
    | tuple[str, bool, int | float | str | bool | np.ndarray | list, set | None],
]

_SUPPORTED_FUNCTIONS = ("MaxPool", "AvgPool", "AccPool", "QuantAvgPool")


class Pool(HWCustomOp):
    """Abstraction layer for HW implementation of Pool.

    Requires ``ConvolutionInputGenerator(depthwise == 1)`` to format its input.

    Input shape ``(BatchSize, OutImgDim, OutImgDim, TotalKernelSize * Channels)``
    Output shape ``(BatchSize, OutImgDim, OutImgDim, Channels)``

    Notes:
    * The input shape was chosen to be compatible with im2col (only true when there
      is not folding).
    * The actual data layout produced by the hlslib kernels is different
      for depthwise ops.

        * depthwise SWG: ``(1, OFMDim, OFMDim, IFMChannels/PE, K, K, PE)``

    Channels can be folded using PE (SIMD from the input perspective).
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Get dictionary of custom node attributes with their types and default values."""
        my_attrs: NodeAttrTypes = {
            "Channels": ("i", True, 0),
            "PE": ("i", True, 1),
            "KernelSize": ("ints", True, []),
            # Pooling function to use corresponding to hlslib functions
            "Function": ("s", True, "", set(_SUPPORTED_FUNCTIONS)),
            "OutImgDims": ("ints", True, []),
            # FINN DataTypes for inputs/outputs
            "InputDataType": ("s", True, ""),
            "OutputDataType": ("s", True, ""),
            "AccumBits": ("i", False, 0),
            "Size": ("i", False, 1),
            "BatchSize": ("i", False, 1),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    @property
    def channels(self) -> int:
        """Get the number of channels."""
        return cast("int", self.get_nodeattr("Channels"))

    @property
    def pe(self) -> int:
        """Get the PE (channel) parallelism."""
        return cast("int", self.get_nodeattr("PE"))

    @property
    def kernel_size(self) -> list[int]:
        """Get the pooling kernel size per spatial axis."""
        return list(cast("list[int]", self.get_nodeattr("KernelSize")))

    @property
    def out_img_dims(self) -> list[int]:
        """Get the output feature-map spatial dimensions."""
        return list(cast("list[int]", self.get_nodeattr("OutImgDims")))

    @property
    def function(self) -> str:
        """Get the pooling function name."""
        return cast("str", self.get_nodeattr("Function"))

    @property
    def accum_bits(self) -> int:
        """Get the accumulator bit width (AvgPool/QuantAvgPool)."""
        return cast("int", self.get_nodeattr("AccumBits"))

    @property
    def size(self) -> int:
        """Get the quantization shift amount (QuantAvgPool)."""
        return cast("int", self.get_nodeattr("Size"))

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return cast("int", self.get_nodeattr("BatchSize"))

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of input."""
        return DataType[cast("str", self.get_nodeattr("InputDataType"))]

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of output."""
        fxn = self.function
        odt = DataType[cast("str", self.get_nodeattr("OutputDataType"))]

        if fxn == "MaxPool":
            # Same as input
            if odt != self.get_input_datatype():
                raise FINNUserError(
                    f"{self.onnx_node.name}: input datatype must equal output datatype for MaxPool"
                )
        elif fxn in ("AccPool", "AvgPool"):
            pass
        elif fxn == "QuantAvgPool":
            if self.get_input_datatype().signed() != odt.signed():
                raise FINNUserError(
                    f"{self.onnx_node.name}: QuantAvgPool cannot mix signed and unsigned datatypes"
                )
        else:
            raise FINNUserError(f"{self.onnx_node.name}: Pool_Batch does not support {fxn}")

        return odt

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return shape of the input tensor."""
        k_prod = int(np.prod(self.kernel_size))
        return (self.batch_size, *self.out_img_dims, k_prod * self.channels)

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return shape of the folded input tensor."""
        normal_ishape = self.get_normal_input_shape()
        if self.channels % self.pe != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: PE ({self.pe}) must divide Channels ({self.channels})"
            )
        fold = normal_ishape[-1] // self.pe
        return (*normal_ishape[:-1], fold, self.pe)

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return shape of the output tensor."""
        return (self.batch_size, *self.out_img_dims, self.channels)

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return shape of the folded output tensor."""
        if self.channels % self.pe != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: PE ({self.pe}) must divide Channels ({self.channels})"
            )
        fold = self.channels // self.pe
        return (self.batch_size, *self.out_img_dims, fold, self.pe)

    def get_exp_cycles(self) -> int:
        """Return estimation of expected cycles for set folding."""
        # (Channels * kernel * kernel) / PE * odim * odim * batch_size
        k_prod = int(np.prod(self.kernel_size))
        exp_cycles = (
            ((self.channels * k_prod) / self.pe) * np.prod(self.out_img_dims) * self.batch_size
        )
        return int(exp_cycles)

    def get_instream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Width of the input stream."""
        return int(self.get_input_datatype().bitwidth() * self.pe)

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Width of the output stream."""
        return int(self.get_output_datatype().bitwidth() * self.pe)

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer the datatype of the output from the node attribute."""
        node = self.onnx_node
        new_dtype = model.get_tensor_datatype(node.input[0])
        self.set_nodeattr("InputDataType", new_dtype.name)
        # data type stays the same
        model.set_tensor_datatype(node.output[0], self.get_output_datatype())

    def verify_node(self) -> list[str]:
        """Verify the node configuration attributes."""
        info_messages = []
        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify the number of inputs
        if len(self.onnx_node.input) == 1:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("""Pool_Batch needs 1 data input""")

        # check supported function
        if self.function in _SUPPORTED_FUNCTIONS:
            info_messages.append("Attribute Function contains a supported pool function")
        else:
            info_messages.append("Attribute Function contains an unsupported pool function")
        return info_messages

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute the node with inputs from context, writing outputs to context."""
        # simulate behavior with Python functionality
        node = self.onnx_node
        fxn = self.function
        k = self.kernel_size
        ch = self.channels
        k2 = k[0] * k[1]

        inp_values = context[node.input[0]]
        ishape = inp_values.shape
        # reshape array to apply max or avg function only on kernel
        tmp_values = inp_values.reshape((*ishape[:-1], k2, ch))
        if fxn == "MaxPool":
            result = np.max(tmp_values, axis=3)
        elif fxn == "AccPool":
            result = np.sum(tmp_values, axis=3)
        elif fxn == "AvgPool":
            result = np.mean(tmp_values, axis=3)
        elif fxn == "QuantAvgPool":
            # determine bits to shift
            ibits = self.get_input_datatype().bitwidth()
            obits = self.get_output_datatype().bitwidth()
            max_value = (2**ibits - 1) * k2
            max_bit_width = int(max_value).bit_length()
            shift_bits = max(max_bit_width - obits, 0)
            result = np.sum(tmp_values, axis=3)
            result = np.right_shift(result.astype(int), shift_bits)
        else:
            raise FINNInternalError(f"{self.onnx_node.name}: Pool_Batch does not support {fxn}")
        oshape = context[node.output[0]].shape
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(oshape)

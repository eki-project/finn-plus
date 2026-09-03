# Copyright (c) 2024, Advanced Micro Devices, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

"""Interior zero-padding (input dilation) hardware custom operator.

This is neither an ONNX ``Pad``-style operator nor an ``Upsample``/``Resize``
operator - it is the input-dilation step of a strided ``ConvTranspose``. See
the ``InputDilation`` class docstring for details.
"""

import numpy as np
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.exception import FINNInternalError, FINNUserError
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


class InputDilation(HWCustomOp):
    """Interior zero-padding (input dilation) of a feature map.

    This operator inserts ``Stride - 1`` rows/columns of zeros *between* every
    pair of adjacent input pixels (per spatial axis), producing an output of
    size ``odim = idim + (idim - 1) * (Stride - 1)``. The original pixels are
    written at strided positions ``out[b, h * s_h, w * s_w, :] = in[b, h, w, :]``
    and every gap is left at zero. The datatype is unchanged.

    Relation to ONNX operators
    --------------------------
    The input-dilation (a.k.a. "fractionally-strided" or
    "zero-insertion") is the first step of a strided ``ConvTranspose``. A stride-s
    transposed convolution is equivalent to dilating the input with ``s - 1``
    zeros between pixels and then running a stride-1 regular convolution. FINN
    only produces this node when lowering ``ConvTranspose``:
    ``InferPixelPaddingDeconv`` (in
    ``finn.transformation.fpgadataflow.infer_pixel_padding_deconv``) rewrites a
    ``ConvTranspose`` (NCHW) into ``InputDilation + Im2Col + MatMul`` (NHWC).

    Node attributes
    ---------------
    ``ImgDim``
        Input spatial size ``[H, W]``.
    ``Stride``
        Pixel spacing ``[H, W]``; ``Stride - 1`` zeros are inserted between
        pixels along each axis. ``Stride = [1, 1]`` is the identity.
    ``NumChannels`` / ``SIMD``
        Channel count and channel-axis parallelism (``SIMD`` must divide
        ``NumChannels``).
    ``inputDataType``
        FINN datatype of the elements; must be able to represent 0 since the
        inserted values are zeros.
    ``numInputVectors``
        Batch size (input vectors processed per execution).
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {
            # spatial size of input images
            "ImgDim": ("ints", True, []),
            # stride to apply, can be non-square
            "Stride": ("ints", True, []),
            # number of channels in input image
            "NumChannels": ("i", True, 0),
            # SIMD Input parallelism
            "SIMD": ("i", False, 1),
            # FINN input datatype
            "inputDataType": ("s", True, ""),
            # shape describing input vecs per execution
            "numInputVectors": ("i", False, 1),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    @property
    def img_dim(self) -> list[int]:
        """Get the input image spatial dimensions [H, W]."""
        return cast("list[int]", self.get_nodeattr("ImgDim"))

    @property
    def stride(self) -> list[int]:
        """Get the pixel-padding stride [H, W]."""
        return cast("list[int]", self.get_nodeattr("Stride"))

    @property
    def num_channels(self) -> int:
        """Get the number of channels."""
        return cast("int", self.get_nodeattr("NumChannels"))

    @property
    def simd(self) -> int:
        """Get the SIMD parallelism."""
        return cast("int", self.get_nodeattr("SIMD"))

    @property
    def num_input_vectors(self) -> int:
        """Get the batch size (input vectors per execution)."""
        return cast("int", self.get_nodeattr("numInputVectors"))

    def get_padded_odim(self) -> list[int]:
        """Return the padded spatial size of the output."""
        idim_h, idim_w = self.img_dim
        stride_h, stride_w = self.stride
        odim_h = idim_h + (idim_h - 1) * (stride_h - 1)
        odim_w = idim_w + (idim_w - 1) * (stride_w - 1)
        return [odim_h, odim_w]

    def get_exp_cycles(self) -> int:
        """Return exp cycles."""
        odim_h, odim_w = self.get_padded_odim()
        exp_cycles = (self.num_channels / self.simd) * self.num_input_vectors * odim_h * odim_w
        return int(exp_cycles)

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return normal input shape."""
        idim_h, idim_w = self.img_dim
        return (1, idim_h, idim_w, self.num_channels)

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return normal output shape."""
        odim_h, odim_w = self.get_padded_odim()
        return (1, odim_h, odim_w, self.num_channels)

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded input shape."""
        normal_ishape = list(self.get_normal_input_shape())
        if self.num_channels % self.simd != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: SIMD ({self.simd}) must divide "
                f"input channels ({self.num_channels})"
            )
        fold = normal_ishape[-1] // self.simd
        return (*normal_ishape[:-1], fold, self.simd)

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded output shape."""
        normal_oshape = list(self.get_normal_output_shape())
        if self.num_channels % self.simd != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: SIMD ({self.simd}) must divide "
                f"input channels ({self.num_channels})"
            )
        fold = normal_oshape[-1] // self.simd
        return (*normal_oshape[:-1], fold, self.simd)

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer node datatype."""
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            log.warning(
                f"inputDataType changing for {node.name}: "
                f"{self.get_input_datatype()!s} -> {idt!s} "
            )
        self.set_nodeattr("inputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of input."""
        ret = DataType[cast("str", self.get_nodeattr("inputDataType"))]
        # the hlslib op always pads with zeros, so ensure that the DataType
        # is able to represent zeros
        if not ret.allowed(0):
            raise FINNUserError(
                f"{self.onnx_node.name}: InputDilation DataType ({ret}) must support zero"
            )
        return ret

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
        """Execute node.

        Simulates the behavior with plain Python.
        """
        node = self.onnx_node
        s_h, s_w = self.stride
        inp_values = context[node.input[0]]
        ishape = inp_values.shape
        result = np.zeros(
            (
                ishape[0],
                ishape[1] + (ishape[1] - 1) * (s_h - 1),
                ishape[2] + (ishape[2] - 1) * (s_w - 1),
                ishape[3],
            )
        )
        for b in range(ishape[0]):
            for h in range(ishape[1]):
                for w in range(ishape[2]):
                    oh = h * s_h
                    ow = w * s_w
                    result[b, oh, ow, :] = inp_values[b, h, w, :]
        oshape = context[node.output[0]].shape
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(oshape)

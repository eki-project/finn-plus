# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Hardware abstraction layer for piecewise polynomial activation functions."""

import math
import numpy as np
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow import register_custom_op
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.custom_op.general.pwpolyfunction import (
    CLAMP_CFG,
    NUM_OCTAVES,
    SUPPORTED_FUNCS,
    _fit_coefficients,
    _horner_eval_numpy,
    _segment_index_numpy,
)
from finn.util.exception import FINNUserError

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
class PWPolyF(HWCustomOp):
    """HW op for piecewise polynomial activations (GELU, SiLU, Sigmoid, Tanh).

    Element-wise FP32, coefficients baked into RTL.  No weights.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return node attribute types for the function, segmentation, folding and datatypes."""
        my_attrs: NodeAttrTypes = {
            # activation function: gelu, silu, sigmoid, tanh
            "func": ("s", True, ""),
            # top-mantissa subdivision bits (K=3 gives 81 segments)
            "K": ("i", False, 3),
            # parallelism; elements processed per cycle
            "PE": ("i", True, 0),
            # number of channels (last dimension of input tensor)
            "NumChannels": ("i", True, 0),
            # FINN DataTypes for inputs, outputs (always FLOAT32)
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # polynomial degree (number of FMA stages per PE)
            "degree": ("i", False, 2),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    @property
    def func(self) -> str:
        """Get the name of the approximated activation function."""
        return cast("str", self.get_nodeattr("func"))

    @property
    def k(self) -> int:
        """Get the number of top-mantissa subdivision bits (attribute ``K``)."""
        return cast("int", self.get_nodeattr("K"))

    @property
    def pe(self) -> int:
        """Get the PE parallelism."""
        return cast("int", self.get_nodeattr("PE"))

    @property
    def num_channels(self) -> int:
        """Get the number of channels (last dimension of the input tensor)."""
        return cast("int", self.get_nodeattr("NumChannels"))

    @property
    def degree(self) -> int:
        """Get the polynomial degree."""
        return cast("int", self.get_nodeattr("degree"))

    @property
    def num_input_vectors(self) -> list[int]:
        """Get the number of input vectors along the non-channel axes."""
        return cast("list[int]", self.get_nodeattr("numInputVectors"))

    def get_num_segments(self) -> int:
        """Return the number of polynomial segments derived from K."""
        return 1 + 2 * NUM_OCTAVES * (1 << self.k)

    def make_shape_compatible_op(self, model: ModelWrapper) -> NodeProto:  # noqa: ARG002
        """Return a constant-shape op for the output."""
        oshape = self.get_normal_output_shape()
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Check for FLOAT32 input and set the input and output datatypes accordingly."""
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != DataType["FLOAT32"]:
            raise FINNUserError(f"{node.name}: PWPolyF requires FLOAT32 input, got {idt}")
        self.set_nodeattr("inputDataType", idt.name)
        self.set_nodeattr("outputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def verify_node(self) -> list[str]:
        """Verify the backend, function, folding and datatype attributes.

        Returns:
            A list of info messages describing the verification results.

        """
        info_messages = []

        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        func = self.func
        if func in SUPPORTED_FUNCS:
            info_messages.append("Attribute func is set correctly")
        else:
            info_messages.append(f"Attribute func must be one of {SUPPORTED_FUNCS}, got {func}")

        pe = self.pe
        nch = self.num_channels
        if pe > 0 and nch > 0 and nch % pe == 0:
            info_messages.append("PE divides NumChannels")
        else:
            info_messages.append("PE must divide NumChannels evenly")

        idt = self.get_nodeattr("inputDataType")
        if idt != "FLOAT32":
            info_messages.append(f"PWPolyF requires FLOAT32 input, got {idt}")
        odt = self.get_nodeattr("outputDataType")
        if odt != "FLOAT32":
            info_messages.append(f"PWPolyF requires FLOAT32 output, got {odt}")

        return info_messages

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of input."""
        return DataType[cast("str", self.get_nodeattr("inputDataType"))]

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of output."""
        return DataType[cast("str", self.get_nodeattr("outputDataType"))]

    def get_instream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return the width of the input stream in bits."""
        return self.get_input_datatype().bitwidth() * self.pe

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return the width of the output stream in bits."""
        return self.get_output_datatype().bitwidth() * self.pe

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the folded input shape."""
        pe = self.pe
        fold = self.num_channels // pe
        return (*self.num_input_vectors, fold, pe)

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the folded output shape (same as input)."""
        return self.get_folded_input_shape()

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the unfolded input shape."""
        return (*self.num_input_vectors, self.num_channels)

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the unfolded output shape (same as input)."""
        return self.get_normal_input_shape()

    def get_exp_cycles(self) -> int:
        """Return the expected cycle count (one per output stream word)."""
        # II=1, latency amortised over stream length
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    def lut_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Estimate the LUTs used per FMA stage and PE."""
        return 100 * self.degree * self.pe

    def bram_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Estimate the RAMB18s used for the per-stage coefficient ROMs."""
        pe = self.pe
        degree = self.degree
        num_segs = self.get_num_segments()

        if degree <= 1:
            return 0

        # Stages after the first use a registered dynamic coefficient lookup
        # for the DSP C input. Vivado infers this as one 32-bit wide ROM per
        # stage and PE, backed by RAMB18 for the default K=3 table depth.
        coeff_width = 32
        if coeff_width <= 18 or num_segs > 512:
            bram18_per_coeff_rom = math.ceil(num_segs / 1024) * math.ceil(coeff_width / 18)
        else:
            bram18_per_coeff_rom = math.ceil(num_segs / 512) * math.ceil(coeff_width / 36)
        return pe * (degree - 1) * bram18_per_coeff_rom

    def uram_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Return the number of URAMs used (none)."""
        return 0

    def dsp_estimation(self, fpgapart: str | None = None) -> int:  # noqa: ARG002
        """Estimate the DSPs used (one per FMA stage and PE)."""
        return self.degree * self.pe

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute the node in Python via segment lookup, Horner evaluation and clamping."""
        node = self.onnx_node
        inp = context[node.input[0]]

        func = self.func
        degree = self.degree
        cfg = CLAMP_CFG[func]

        coeffs = _fit_coefficients(func, self.k, degree=degree)
        orig_shape = inp.shape
        x_flat = inp.flatten().astype(np.float32)

        seg_idx, is_neg_clamp, is_pos_clamp = _segment_index_numpy(x_flat, self.k)
        y = _horner_eval_numpy(x_flat, coeffs, seg_idx, degree, func)

        if cfg["pos_passthrough"]:
            y = np.where(is_pos_clamp, x_flat, y)
        else:
            y = np.where(is_pos_clamp, cfg["pos_clamp"], y)
        y = np.where(is_neg_clamp, cfg["neg_clamp"], y)

        context[node.output[0]] = y.reshape(orig_shape).astype(np.float32)

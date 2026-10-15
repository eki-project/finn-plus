# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Hardware abstraction layer for the elementwise Where operation."""

import math
import numpy as np
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow import register_custom_op
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


@register_custom_op
class HWWhere(HWCustomOp):
    """Elementwise ONNX Where with multidirectional broadcasting."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return node attribute types, adding shape, rank, PE,
        datatype and ram_style attributes.
        """
        my_attrs: NodeAttrTypes = super().get_nodeattr_types()
        my_attrs.update(
            {
                "Shape": ("ints", True, []),
                "CondShape": ("ints", False, []),
                "XShape": ("ints", False, []),
                "YShape": ("ints", False, []),
                "CondRank": ("i", False, -1),
                "XRank": ("i", False, -1),
                "YRank": ("i", False, -1),
                "PE": ("i", False, 1),
                "conditionDataType": ("s", False, "BINARY"),
                "inputDataType": ("s", True, ""),
                "outputDataType": ("s", False, ""),
                "ram_style": (
                    "s",
                    False,
                    "auto",
                    {"auto", "block", "distributed", "ultra"},
                ),
                # Where has three streaming inputs, unlike the base class default.
                "inFIFODepths": ("ints", False, [2, 2, 2]),
            }
        )
        return my_attrs

    @property
    def pe(self) -> int:
        """Get the PE parallelism of the output stream."""
        return cast("int", self.get_nodeattr("PE"))

    @property
    def ram_style(self) -> str:
        """Get the memory style of the input replay buffers."""
        return cast("str", self.get_nodeattr("ram_style"))

    def _shape(self) -> tuple[int, ...]:
        """Return the (broadcast) output shape as a tuple."""
        return tuple(cast("list[int]", self.get_nodeattr("Shape")))

    def _input_shape(self, ind: int) -> tuple[int, ...]:
        """Return the shape of input `ind`, falling back to the output shape if not set."""
        if ind == 0:
            attr_name, rank_name = "CondShape", "CondRank"
        elif ind == 1:
            attr_name, rank_name = "XShape", "XRank"
        elif ind == 2:
            attr_name, rank_name = "YShape", "YRank"
        else:
            raise FINNInternalError("Where has exactly three inputs")

        rank = cast("int", self.get_nodeattr(rank_name))
        shape = tuple(cast("list[int]", self.get_nodeattr(attr_name)))
        if rank >= 0:
            if len(shape) != rank:
                raise FINNUserError(f"{attr_name} length must match {rank_name}")
            return shape
        if len(shape) != 0:
            return shape
        return self._shape()

    def _rtl_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Return the shape as a tuple, mapping a scalar (rank 0) to (1,)."""
        if len(shape) == 0:
            return (1,)
        return tuple(shape)

    def _input_stream_pe(self, ind: int) -> int:
        """Return the stream parallelism of input `ind` (1 if its innermost dim is broadcast)."""
        shape = self._rtl_shape(self.get_normal_input_shape(ind))
        if shape[-1] == 1:
            return 1
        return self._output_stream_pe()

    def _output_stream_pe(self) -> int:
        """Return the output stream parallelism (1 if the innermost output dim is 1, else PE)."""
        shape = self._rtl_shape(self.get_normal_output_shape())
        if shape[-1] == 1:
            return 1
        return self.pe

    def _folded_shape(self, shape: tuple[int, ...], stream_pe: int) -> tuple[int, ...]:
        """Return the shape with its innermost dimension folded by `stream_pe`."""
        rtl_shape = self._rtl_shape(shape)
        *outer, channels = rtl_shape
        if channels % stream_pe != 0:
            raise FINNUserError("Stream PE must divide the innermost dimension")
        return (*outer, channels // stream_pe, stream_pe)

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return the unfolded shape of input `ind`."""
        if ind not in [0, 1, 2]:
            raise FINNInternalError("Where has exactly three inputs")
        return self._input_shape(ind)

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return the folded shape of input `ind`."""
        return self._folded_shape(self.get_normal_input_shape(ind), self._input_stream_pe(ind))

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return the unfolded output shape."""
        if ind != 0:
            raise FINNInternalError("Where has exactly one output")
        return self._shape()

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return the folded output shape."""
        return self._folded_shape(self.get_normal_output_shape(ind), self._output_stream_pe())

    def make_shape_compatible_op(self, model: ModelWrapper) -> NodeProto:
        """Check the input shapes and return a constant-shape op for the output."""
        for i, inp in enumerate(self.onnx_node.input):
            ishape = model.get_tensor_shape(inp)
            if ishape is None or tuple(ishape) != self.get_normal_input_shape(i):
                raise FINNUserError(f"Unexpected input shape for Where input {i}.")
        return super().make_const_shape_op(self.get_normal_output_shape())

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer and set the condition, input and output datatypes from the model tensors."""
        node = self.onnx_node

        cond_dt = model.get_tensor_datatype(node.input[0])
        if cond_dt is None:
            cond_dt = self.get_condition_datatype()
            model.set_tensor_datatype(node.input[0], cond_dt)
        if cond_dt != DataType["BINARY"]:
            raise FINNUserError("Where condition datatype must be BINARY")
        self.set_nodeattr("conditionDataType", cond_dt.name)

        attr_idt = None
        if self.get_nodeattr("inputDataType") != "":
            attr_idt = self.get_input_datatype(1)

        x_dt = model.get_tensor_datatype(node.input[1])
        y_dt = model.get_tensor_datatype(node.input[2])
        idt = x_dt if x_dt is not None else attr_idt
        if idt is None:
            raise FINNUserError("Where input datatype is not set")
        if y_dt is None:
            model.set_tensor_datatype(node.input[2], idt)
        elif y_dt != idt:
            raise FINNUserError("Where X and Y datatypes must match")
        if x_dt is None:
            model.set_tensor_datatype(node.input[1], idt)

        if attr_idt is not None and attr_idt != idt:
            log.warning(f"inputDataType changing for {node.name}: {attr_idt!s} -> {idt!s}")
        self.set_nodeattr("inputDataType", idt.name)

        attr_odt = cast("str", self.get_nodeattr("outputDataType"))
        if attr_odt != "" and DataType[attr_odt] != idt:
            log.warning(
                f"outputDataType changing for {node.name}: {DataType[attr_odt]!s} -> {idt!s}"
            )
        self.set_nodeattr("outputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def get_condition_datatype(self) -> BaseDataType:
        """Return the FINN DataType of the condition input."""
        return DataType[cast("str", self.get_nodeattr("conditionDataType"))]

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:
        """Return the FINN DataType of input `ind` (condition for 0, inputDataType for 1 and 2)."""
        if ind == 0:
            return self.get_condition_datatype()
        if ind in [1, 2]:
            return DataType[cast("str", self.get_nodeattr("inputDataType"))]
        raise FINNInternalError("Where has exactly three inputs")

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return the FINN DataType of the output (defaults to the X input datatype)."""
        odt = cast("str", self.get_nodeattr("outputDataType"))
        if odt == "":
            return self.get_input_datatype(1)
        return DataType[odt]

    def get_instream_width(self, ind: int = 0) -> int:
        """Return the width of input stream `ind` in bits."""
        if ind == 0:
            return self._input_stream_pe(ind)
        if ind in [1, 2]:
            return self.get_input_datatype(ind).bitwidth() * self._input_stream_pe(ind)
        return 0

    def get_outstream_width(self, ind: int = 0) -> int:
        """Return the width of the output stream in bits."""
        return self.get_output_datatype(ind).bitwidth() * self._output_stream_pe()

    def get_number_output_values(self) -> int:
        """Return the number of output stream words per frame."""
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    def get_exp_cycles(self) -> int:
        """Return the expected cycle count from the longest input stream plus the output stream."""
        input_cycles = max(int(np.prod(self.get_folded_input_shape(ind)[:-1])) for ind in range(3))
        output_cycles = self.get_number_output_values()
        return input_cycles + output_cycles + 4

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute the node in Python using numpy.where."""
        node = self.onnx_node
        cond = context[node.input[0]]
        xval = context[node.input[1]]
        yval = context[node.input[2]]

        result = np.where(cond.astype(bool), xval, yval)
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(
            self.get_normal_output_shape()
        )

    def _input_gen_buffer_specs(self) -> list[tuple[int, int]]:
        """Return the (width, depth) of each RTL input_gen buffer.

        The depth here must match the buffer size that ``input_gen.sv`` derives
        at elaboration from its FM_SIZE/DIMS/COEFS parameters (see where.sv's
        INIT_FM_SIZE/INIT_COEFS, which feed those parameters). Resource
        estimation has no access to the elaborated RTL, so this recomputes the
        same occupancy bound in Python; keep it in sync with input_gen.sv.
        """
        out_shape = self._rtl_shape(self.get_normal_output_shape())
        out_pe = self._output_stream_pe()
        out_dims = list(out_shape)
        out_dims[-1] //= out_pe
        specs = []

        for ind in range(3):
            in_shape = self._rtl_shape(self.get_normal_input_shape(ind))
            if len(in_shape) > len(out_shape):
                raise FINNUserError("Input rank must not exceed output rank")
            aligned_shape = (1,) * (len(out_shape) - len(in_shape)) + in_shape

            # Size of each axis in stream words: the innermost non-broadcast dim
            # is folded by PE.
            word_dims = [
                dim // out_pe if axis == len(out_shape) - 1 and dim != 1 else dim
                for axis, dim in enumerate(aligned_shape)
            ]

            fm_size = int(np.prod(word_dims))
            coefs = []
            for axis, dim in enumerate(aligned_shape):
                # Replay (coef 0) only when the operand dim is 1 and the output
                # dim is >1, matching where.sv's INIT_COEFS. When the output dim
                # is also 1 the RTL keeps a real stride, so mirror that here.
                if dim == 1 and out_dims[axis] > 1:
                    coefs.append(0)
                else:
                    coefs.append(int(np.prod(word_dims[axis + 1 :])))

            # Mirror the occupancy recurrence input_gen.sv evaluates at
            # elaboration to size its circular replay buffer.
            weights = [fm_size, *coefs]
            free_flags = [True]
            for axis, coef in enumerate(coefs):
                free_flags.append(
                    free_flags[-1] and coef > 0 and coef * out_dims[axis] <= weights[axis]
                )

            max_occupancy = 0
            read_rewind = 0
            free_rewind = 0
            for axis in range(len(out_shape) - 1, -1, -1):
                inner_read_rewind = read_rewind
                inner_free_rewind = free_rewind
                read_rewind = (out_dims[axis] - 1) * coefs[axis] + read_rewind
                free_rewind = (
                    (out_dims[axis] - 1) * coefs[axis] + free_rewind if free_flags[axis + 1] else 0
                )
                max_occupancy = max(max_occupancy, read_rewind - free_rewind)
                if free_flags[axis]:
                    burst = max(weights[axis] - free_rewind, 0)
                    required = inner_read_rewind - inner_free_rewind + burst
                    max_occupancy = max(max_occupancy, required)

            # input_gen adds one write-pointer delay and two safety entries,
            # then rounds the buffer depth up to a power of two.
            required_depth = max_occupancy + 3
            buffer_depth = 1 << (required_depth - 1).bit_length()
            specs.append((self.get_instream_width(ind), buffer_depth))

        return specs

    @staticmethod
    def _bram18_estimation(width: int, depth: int) -> int:
        """Return the number of RAMB18 blocks needed for a buffer of the given width and depth."""
        if width == 1:
            return math.ceil(depth / 16384)
        if width == 2:
            return math.ceil(depth / 8192)
        if width <= 4:
            return math.ceil(depth / 4096) * math.ceil(width / 4)
        if width <= 9:
            return math.ceil(depth / 2048) * math.ceil(width / 9)
        if width <= 18 or depth > 512:
            return math.ceil(depth / 1024) * math.ceil(width / 18)
        return math.ceil(depth / 512) * math.ceil(width / 36)

    def bram_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Estimate the number of BRAMs used by the input replay buffers."""
        ram_style = self.ram_style
        if ram_style == "block":
            buffer_specs = self._input_gen_buffer_specs()
        elif ram_style == "auto":
            # Vivado maps small buffers to distributed RAM; only buffers of at
            # least ~1 kbit are expected to end up in BRAM.
            buffer_specs = [
                (width, depth)
                for width, depth in self._input_gen_buffer_specs()
                if width * depth >= 1024
            ]
        else:
            return 0
        return int(sum(self._bram18_estimation(width, depth) for width, depth in buffer_specs))

    def uram_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Estimate the number of URAMs used by the input replay buffers."""
        if self.ram_style != "ultra":
            return 0
        return int(
            sum(
                math.ceil(width / 72) * math.ceil(depth / 4096)
                for width, depth in self._input_gen_buffer_specs()
            )
        )

    def bram_efficiency_estimation(self, fpgapart: str) -> float:
        """Return the fraction of estimated BRAM capacity actually used by the buffers."""
        bram_estimate = self.bram_estimation(fpgapart)
        if bram_estimate == 0:
            return 1
        buffer_specs = self._input_gen_buffer_specs()
        if self.ram_style == "auto":
            # Match bram_estimation's ~1 kbit distributed-vs-BRAM split.
            buffer_specs = [
                (width, depth) for width, depth in buffer_specs if width * depth >= 1024
            ]
        used_bits = sum(width * depth for width, depth in buffer_specs)
        return used_bits / (bram_estimate * 36 * 512)

    def uram_efficiency_estimation(self, fpgapart: str) -> float:
        """Return the fraction of estimated URAM capacity actually used by the buffers."""
        uram_estimate = self.uram_estimation(fpgapart)
        if uram_estimate == 0:
            return 1
        used_bits = sum(width * depth for width, depth in self._input_gen_buffer_specs())
        return used_bits / (uram_estimate * 72 * 4096)

    def lut_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Estimate the LUTs used for selection logic and LUTRAM-backed buffers."""
        selection_luts = 64 + self.pe * self.get_output_datatype().bitwidth()
        ram_style = self.ram_style
        if ram_style == "distributed":
            buffer_specs = self._input_gen_buffer_specs()
        elif ram_style == "auto":
            # Complement of bram_estimation: buffers below ~1 kbit stay in LUTRAM.
            buffer_specs = [
                (width, depth)
                for width, depth in self._input_gen_buffer_specs()
                if width * depth < 1024
            ]
        else:
            buffer_specs = []
        if buffer_specs:
            buffer_luts = sum(width * math.ceil(depth / 64) for width, depth in buffer_specs)
        else:
            buffer_luts = 0
        return int(selection_luts + buffer_luts)

    def get_op_and_param_counts(self) -> dict[str, int]:
        """Return a dictionary with the number of where operations per inference."""
        return {"op_where": int(np.prod(self.get_normal_output_shape()))}

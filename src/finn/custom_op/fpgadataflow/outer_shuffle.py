############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

"""Outer (rank-preserving) transpose hardware custom operator."""

import numpy as np
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow import register_custom_op
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp, NodeAttrTypes
from finn.custom_op.fpgadataflow.shuffle import auto_size_simd
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.input_gen import input_gen_buffer_depth
from finn.util.logging import log

if TYPE_CHECKING:
    from onnx import GraphProto


class _NestSim:
    """Simulate the ``Nest<R, W, N, C, V...>`` HLS template from input_gen.hpp.

    Models the read-pointer and free-pointer update logic of the HLS reorder
    buffer, including loop termination and counter reset behavior. The argument
    names mirror the C++ template parameters.
    """

    def __init__(self, r: bool, w: int, *rest: int) -> None:
        """Initialize the nested loop simulation state."""
        self.R = r
        self.W = w
        self.is_terminal = len(rest) == 0
        if self.is_terminal:
            self._rp_rewind = 0
            self._fp_rewind = 0
            self.max_rp_retract = 0
        else:
            self.N = rest[0]
            self.C = rest[1]
            # As in finn-hlslib input_gen.hpp: the inner loop may advance the free
            # pointer in lockstep if its furthest advance, (N-1)*C plus its own
            # (hypothetical) free-pointer rewind, never carries it beyond W.
            hypothetical_fp_rewind = _NestSim(True, self.C, *rest[2:])._fp_rewind  # noqa: SLF001
            self.R_INNER = (
                r and (self.C > 0) and ((self.N - 1) * self.C + hypothetical_fp_rewind <= w)
            )
            self.inner = _NestSim(self.R_INNER, self.C, *rest[2:])
            inner_rp_rewind = self.inner._rp_rewind  # noqa: SLF001
            inner_fp_rewind = self.inner._fp_rewind  # noqa: SLF001
            self._rp_rewind = (self.N - 1) * self.C + inner_rp_rewind
            self._fp_rewind = (self.N - 1) * self.C + inner_fp_rewind if self.R_INNER else 0
            self.terminal_rp_inc = w - self._rp_rewind
            self.cnt = self.N - 2
            self.max_rp_retract = max(-self.terminal_rp_inc, self.inner.max_rp_retract)

    def tick(self) -> tuple[int, int, bool]:
        """Advance the simulation by one step and return increments."""
        if self.is_terminal:
            return self.W, (self.W if self.R else 0), True
        rp_inc, fp_inc, term = self.inner.tick()
        if term:
            if self.cnt < 0:
                rp_inc = self.terminal_rp_inc
                if self.R:
                    fp_inc = self.W - self._fp_rewind
                self.cnt = self.N - 2
                return rp_inc, fp_inc, True
            self.cnt -= 1
            return rp_inc, fp_inc, False
        return rp_inc, fp_inc, False


@register_custom_op
class OuterShuffle(HWCustomOp):
    """Abstraction layer for HW implementation of an outer (rank-preserving) transpose.

    Rearranges the outer axes of a tensor according to ``perm``. Only
    permutations that leave the innermost dimension in place are feasible.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize the OuterShuffle custom op wrapper."""
        super().__init__(onnx_node, **kwargs)

        # SIMD must divide the innermost axis, which an outer shuffle cannot move.
        last_dim = self.transpose_in_shape[-1]
        if last_dim % self.simd != 0:
            new_simd = auto_size_simd(last_dim, self.simd)
            if new_simd is None:
                raise FINNUserError(
                    f"{self.onnx_node.name}: unable to determine a SIMD value that divides "
                    f"the innermost dimension ({last_dim})"
                )
            self.set_nodeattr("SIMD", new_simd)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return the node attribute schema for OuterShuffle."""
        my_attrs: NodeAttrTypes = {
            "data_type": ("s", True, ""),
            "transpose_in_shape": ("ints", True, []),
            "in_shape": ("ints", True, []),
            "transpose_out_shape": ("ints", True, []),
            "out_shape": ("ints", True, []),
            "loop_coeffs": ("ints", True, []),
            "perm": ("ints", True, []),
            "SIMD": ("i", False, 1),
            "NumChannels": ("i", False, 128),
            # Track original shuffle name/SIMD for SIMD config export
            "original_node_name": ("s", False, ""),
            "original_simd": ("i", False, 1),
            # Storage of the input_gen reorder buffer. "auto" leaves the choice to
            # HLS, which binds buffers deeper than 262144 words to URAM by itself.
            # Affects resources only, not the II (see get_input_gen_ii).
            "ram_style": ("s", False, "auto", {"auto", "block", "distributed", "ultra"}),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    @property
    def dtype(self) -> BaseDataType:
        """Get the element data type."""
        return DataType[cast("str", self.get_nodeattr("data_type"))]

    @property
    def simd(self) -> int:
        """Get the SIMD parallelism."""
        return cast("int", self.get_nodeattr("SIMD"))

    @property
    def num_channels(self) -> int:
        """Get the channel count."""
        return cast("int", self.get_nodeattr("NumChannels"))

    @property
    def transpose_in_shape(self) -> list[int]:
        """Get the pre-transpose (reshaped) input shape."""
        return list(cast("list[int]", self.get_nodeattr("transpose_in_shape")))

    @property
    def in_shape(self) -> list[int]:
        """Get the streaming input shape."""
        return list(cast("list[int]", self.get_nodeattr("in_shape")))

    @property
    def transpose_out_shape(self) -> list[int]:
        """Get the post-transpose (pre-reshape) output shape."""
        return list(cast("list[int]", self.get_nodeattr("transpose_out_shape")))

    @property
    def out_shape(self) -> list[int]:
        """Get the streaming output shape."""
        return list(cast("list[int]", self.get_nodeattr("out_shape")))

    @property
    def loop_coeffs(self) -> list[int]:
        """Get the permuted input strides driving the reorder buffer."""
        return list(cast("list[int]", self.get_nodeattr("loop_coeffs")))

    @property
    def perm(self) -> list[int]:
        """Get the axis permutation."""
        return list(cast("list[int]", self.get_nodeattr("perm")))

    def get_normal_input_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return the non-folded input shape."""
        return self.in_shape

    def get_normal_output_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return the non-folded output shape."""
        return self.out_shape

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute the outer shuffle using numpy reshape/transpose."""
        node = self.onnx_node
        input_data = context[node.input[0]]
        input_reshaped = input_data.reshape(self.transpose_in_shape)
        transposed = np.transpose(input_reshaped, axes=self.perm)
        context[node.output[0]] = transposed.reshape(self.out_shape)

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return the input datatype."""
        return self.dtype

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer and propagate the node datatype."""
        node = self.onnx_node
        dt = model.get_tensor_datatype(node.input[0])
        if dt != self.get_input_datatype():
            log.warning(
                f"data_type changing for {node.name}: {self.get_input_datatype()!s} -> {dt!s}"
            )
        self.set_nodeattr("data_type", dt.name)
        model.set_tensor_datatype(node.output[0], dt)

    def verify_node(self) -> list[str]:
        """Validate node attributes and shapes (not implemented)."""
        raise NotImplementedError("This function is not yet implemented.")

    def get_instream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return the input stream width in bits."""
        return self.get_input_datatype().bitwidth() * self.simd

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return the output stream width in bits."""
        return self.get_output_datatype().bitwidth() * self.simd

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return the output datatype."""
        return self.dtype

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the folded output shape for SIMD streaming."""
        normal_oshape = list(self.get_normal_output_shape())
        if normal_oshape[-1] % self.simd != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: SIMD ({self.simd}) must divide the innermost "
                f"output dimension ({normal_oshape[-1]})"
            )
        fold = normal_oshape[-1] // self.simd
        return (*normal_oshape[:-1], fold, self.simd)

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the folded input shape for SIMD streaming."""
        normal_ishape = list(self.get_normal_input_shape())
        if normal_ishape[-1] % self.simd != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: SIMD ({self.simd}) must divide the innermost "
                f"input dimension ({normal_ishape[-1]})"
            )
        fold = normal_ishape[-1] // self.simd
        return (*normal_ishape[:-1], fold, self.simd)

    def get_input_gen_nest(self) -> tuple[int, list[int], list[int]]:
        """Return the loop nest the HLS kernel instantiates input_gen<> with.

        Returns (num_words, extents, coeffs): words per frame after SIMD folding,
        the output loop extents and their input-side coefficients, derived from
        transpose_in_shape, perm and SIMD exactly like the HLS backend's docompute.
        """
        simd = self.simd
        in_shape = list(self.transpose_in_shape)
        perm = list(self.perm)
        extents = [in_shape[p] for p in perm]
        adjusted = [*in_shape, 1]
        input_strides = [int(np.prod(adjusted[i + 1 :])) for i in range(len(in_shape))]
        coeffs = [input_strides[p] for p in perm]
        # Apply SIMD folding to innermost dimension
        extents[-1] = int(extents[-1] / simd)
        coeffs = [1 if x == 1 else int(x / simd) for x in coeffs]
        return int(np.prod(extents)), extents, coeffs

    def get_buffer_depth(self) -> int:
        """Return the depth (BUF_SIZE, in words) of the input_gen reorder buffer.

        Mirrors the stall-free sizing finn-hlslib input_gen() dimensions it with.
        """
        num_words, extents, coeffs = self.get_input_gen_nest()
        return input_gen_buffer_depth(num_words, extents, coeffs)

    def get_input_gen_ii(self) -> int:
        """Return the pipeline II the HLS kernel achieves for its reorder buffer.

        Always 1: buffers up to 262144 words read with latency 1, and deeper
        ones, which HLS maps to a latency-pipelined memory cascade (URAM), are
        read ahead into a small queue by input_gen() (finn-hlslib 5f0aec0+), so
        the cascade latency no longer enters the loop recurrence. Before that,
        deep buffers dropped to II=3 (Vivado 2024.2) or 1 + depth/131072
        (2025.1+). Explicit storage (ram_style) does not change the II either.
        """
        return 1

    def get_exp_cycles(self) -> int:
        """Estimate cycles by simulating the input_gen HLS pipeline.

        Derives the loop nest from transpose_in_shape, perm and SIMD
        (get_input_gen_nest), the buffer depth from the stall-free sizing of
        finn-hlslib input_gen() (get_buffer_depth) and its pipeline II
        (get_input_gen_ii).

        The HLS pipeline has three stall sources:
        1. wp_delay (=4): write-pointer pipeline latency before reads begin
        2. Read stalls: consumer waits for data (rp >= wp_delayed)
        3. Write stalls: producer blocked by full buffer (wp - fp >= buf_size)
        """
        total_elems, extents, coeffs = self.get_input_gen_nest()
        # Build the Nest args: Nest<true, IFM_SIZE, N0, C0, N1, C1, ..., Nn, Cn>
        interleaved = [int(item) for pair in zip(extents, coeffs, strict=True) for item in pair]
        nest = _NestSim(True, total_elems, *tuple(interleaved))
        wp_delay = 4
        buf_size = self.get_buffer_depth()
        pipeline_ii = self.get_input_gen_ii()

        # Simulate the input_gen pipeline at II=1.
        # Models the wp delay pipeline, finite buffer backpressure,
        # and the Nest-driven read pointer pattern.
        wp = [0] * wp_delay
        rp = 0
        fp = 0
        ovld = False
        input_consumed = 0
        output_produced = 0
        cycle = 0

        while output_produced < total_elems and cycle < total_elems * 10:
            cycle += 1

            # Shift write pointer delay pipeline
            for i in range(wp_delay - 1, 0, -1):
                wp[i] = wp[i - 1]

            # Write into buffer if space available
            if wp[0] - fp < buf_size and input_consumed < total_elems:
                wp[0] += 1
                input_consumed += 1

            # Drain output buffer
            if ovld:
                output_produced += 1
                ovld = False

            # Refill output buffer via Nest tick
            if not ovld and rp < wp[wp_delay - 1]:
                rp_inc, fp_inc, _ = nest.tick()
                rp += rp_inc
                fp += fp_inc
                ovld = True

        return cycle * pipeline_ii

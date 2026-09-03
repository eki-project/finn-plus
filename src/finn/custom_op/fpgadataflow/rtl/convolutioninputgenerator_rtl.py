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

"""RTL implementation of ConvolutionInputGenerator (Sliding Window Generator).

This module provides an RTL-based implementation of the ConvolutionInputGenerator,
generating sliding windows for convolution operations on FPGA. Supports non-square,
1D, strided, dilated, and depthwise convolutions with configurable buffer implementations.
"""

import math
import numpy as np
import shutil
from onnx import NodeProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general import im2col
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.util.basic import roundup_to_integer_multiple
from typing import TYPE_CHECKING, Literal, cast

from finn.custom_op.fpgadataflow.convolutioninputgenerator import (
    ConvolutionInputGenerator,
    NodeAttrTypes,
)
from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from onnx import GraphProto

# RTL Convolution Input Generator / Sliding Window Generator (SWG)
# Matches and extends the functionality of all ConvolutionInputGenerator_* functions
# in finn-hlslib by generating HDL code for two different implementation styles:
# - Addressable cyclic buffer: to be used when out_width <= in_width
# - Parallel registers + line buffers: to be used when out_width > in_width
# Supports non-square, 1D, strided, dilated, and depthwise convolutions.
# Note: the actual data layout produced is different for depthwise and non-depthwise:
# * non-depthwise SWG: (1, OFMDim_H, OFMDim_W, K_H, K_W, IFMChannels/SIMD, SIMD)
# * depthwise SWG: (1, OFMDim_H, OFMDim_W, IFMChannels/SIMD, K_H, K_W, SIMD)

# NOTE: "Parallel" implementation style not yet implemented in this version!


@register_custom_op
class ConvolutionInputGenerator_rtl(ConvolutionInputGenerator, RTLBackend):
    """Class that corresponds to finn-rtllib swg module.

    Generates an RTL ConvolutionInputGenerator implementation
    based on (System-)Verilog templates, defined in finn-rtllib/swg.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize the RTL ConvolutionInputGenerator."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {
            # additional parallelization parameter - not yet implemented
            "M": ("i", False, 1),
        }
        my_attrs.update(ConvolutionInputGenerator.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    @property
    def m_par(self) -> int:
        """Get the additional parallelization parameter M."""
        return cast("int", self.get_nodeattr("M"))

    def get_number_input_values(self) -> int:
        """Return the number of expected input values."""
        folded_ishape = self.get_folded_input_shape()
        return int(np.prod(folded_ishape[:-1]))

    def use_parallel_window_output(self) -> bool:
        """Return whether the parallel window output mode is enabled."""
        return self.parallel_window

    def get_buffer_depth(self) -> int:
        """Return total depth of the internal buffer, depending on
        implementation style.
        """
        ifm_ch = self.ifm_channels
        k_h, k_w = self.kernel_dim
        _h, w = self.ifm_dim
        stride_h, stride_w = self.stride
        dilation_h, dilation_w = self.dilation
        simd = self.simd

        mmv_in = 1
        mmv_out = 1
        channel_factor = int(ifm_ch / simd)
        impl_style = self.select_impl_style()
        if impl_style == "default":
            buffer_min_size = (
                (k_h - 1) * dilation_h * w + (k_w - 1) * dilation_w + 1
            ) * channel_factor
            # add additional buffer space in case of stride > 1
            # this minimizes cycle count as it allows an earlier pre-load of inputs
            buffer_depth = (
                buffer_min_size
                + max(
                    0,
                    ((stride_w - 1) - (int(mmv_out * k_h * k_w / mmv_in))) * channel_factor,
                )
                + max(
                    0,
                    ((stride_h - 1) * w - (int(mmv_out * k_h * k_w / mmv_in))) * channel_factor,
                )
            )
        else:
            buffer_min_size = (
                (k_h - 1) * dilation_h * w + (k_w - 1) * dilation_w
            ) * channel_factor + 1
            buffer_depth = buffer_min_size + 1
        return buffer_depth

    def get_exp_cycles(self) -> int:
        """Return expected number of clock cycles for one inference."""
        impl_style = self.select_impl_style()

        if impl_style == "parallel":
            exp_cycles = self.get_number_input_values() + 2
        else:
            simd = self.simd
            ifm_ch = self.ifm_channels
            k_h, k_w = self.kernel_dim
            ifm_dim_h, ifm_dim_w = self.ifm_dim
            ofm_dim_h, ofm_dim_w = self.ofm_dim
            stride_h, stride_w = self.stride
            dilation_h, dilation_w = self.dilation
            depthwise = self.depthwise

            channel_factor = int(ifm_ch / simd)
            if ifm_dim_h == 1 or ifm_dim_w == 1:
                # 1D case
                (
                    ifm_ch,
                    [ifm_dim_h, ifm_dim_w],
                    [ofm_dim_h, ofm_dim_w],
                    [k_h, k_w],
                    [stride_h, stride_w],
                    [dilation_h, dilation_w],
                ) = self.get_1d_conv_attrs_normalized()

                if depthwise:
                    exp_cycles = (
                        +ofm_dim_w * k_w * channel_factor
                        + channel_factor * (k_w - 1) * (stride_w - 1)
                        - (k_w - 1)
                        + 2
                    )
                else:
                    exp_cycles = ofm_dim_w * k_w * channel_factor + 2
            else:
                # 2D case
                buffer_min_size = (
                    (k_h - 1) * dilation_h * ifm_dim_w + (k_w - 1) * dilation_w + 1
                ) * channel_factor
                cycles_write_block = ofm_dim_w * k_w * k_h * channel_factor
                cycles_read_block = stride_w * ifm_dim_w * channel_factor
                max_cycles = max(cycles_write_block, cycles_read_block)
                if depthwise:
                    max_cycles += ofm_dim_w * (stride_w - 1) * (channel_factor - 1)
                exp_cycles = buffer_min_size + ofm_dim_h * max_cycles
                if depthwise:
                    exp_cycles += (stride_h - 1) * ifm_dim_w * channel_factor

        return int(exp_cycles)

    def bram_estimation(self) -> int:
        """Estimate Block RAM (BRAM) resource usage."""
        simd = self.simd
        ram_style = self.ram_style
        impl_style = self.select_impl_style()
        k_h, k_w = self.kernel_dim
        ifm_dim_h, ifm_dim_w = self.ifm_dim
        dilation_h, dilation_w = self.dilation

        if ram_style in ("block", "auto"):
            buffer_width = simd * self.get_input_datatype().bitwidth()
            if impl_style == "default":
                buffer_depth = self.get_buffer_depth()
                buffer_count = 1
            else:
                if ifm_dim_h == 1 or ifm_dim_w == 1:
                    return 0  # 1D case (no line buffers needed)
                kernel_width = (k_w - 1) * dilation_w + 1
                buffer_depth = (ifm_dim_w - kernel_width) + ifm_dim_w * (dilation_h - 1)
                buffer_count = k_h - 1

            # NOTE: Actual BRAM usage might be lower in some cases
            # due to imperfect modeling of Vivado behavior
            if buffer_depth <= 512:
                ram_width = 36
            elif buffer_depth <= 1024:
                ram_width = 18
            elif buffer_depth <= 2048:
                ram_width = 9
            elif buffer_depth <= 4096:
                ram_width = 4
            elif buffer_depth <= 8192:
                ram_width = 2
            else:
                ram_width = 1

            ram_cascade_depth = math.ceil(buffer_depth / 16384)
            ram_cascade_width = math.ceil(buffer_width / ram_width)
            cascade_savings = 0
            if buffer_depth > 16384:
                remainder_depth = buffer_depth % 16384
                if remainder_depth <= 512:
                    remainder_width = 36
                elif remainder_depth <= 1024:
                    remainder_width = 18
                elif remainder_depth <= 2048:
                    remainder_width = 9
                elif remainder_depth <= 4096:
                    remainder_width = 4
                elif remainder_depth <= 8192:
                    remainder_width = 2
                else:
                    remainder_width = 1

                remainder_cascade_width = math.ceil(buffer_width / remainder_width)
                cascade_savings = ram_cascade_width - remainder_cascade_width

            return int((ram_cascade_depth * ram_cascade_width - cascade_savings) * buffer_count)
        return 0

    def lut_estimation(self) -> int:
        """Estimate LUT resource usage."""
        simd = self.simd
        ram_style = self.ram_style
        buffer_width = simd * self.get_input_datatype().bitwidth()
        buffer_depth = self.get_buffer_depth()
        if ram_style == "distributed":
            ram_luts = int(buffer_width * math.ceil(buffer_depth / 38))
        else:
            ram_luts = 0
        return 300 + ram_luts

    def uram_estimation(self) -> int:
        """Estimate UltraRAM (URAM) resource usage."""
        simd = self.simd
        ram_style = self.ram_style
        impl_style = self.select_impl_style()
        k_h, k_w = self.kernel_dim
        ifm_dim_h, ifm_dim_w = self.ifm_dim
        dilation_h, dilation_w = self.dilation

        if ram_style == "ultra":
            buffer_width = simd * self.get_input_datatype().bitwidth()
            if impl_style == "default":
                buffer_depth = self.get_buffer_depth()
                buffer_count = 1
            else:
                if ifm_dim_h == 1 or ifm_dim_w == 1:
                    return 0  # 1D case (no line buffers needed)
                kernel_width = (k_w - 1) * dilation_w + 1
                buffer_depth = (ifm_dim_w - kernel_width) + ifm_dim_w * (dilation_h - 1)
                buffer_count = k_h - 1

            ram_depth = 4096
            ram_width = 72
            ram_cascade_depth = math.ceil(buffer_depth / ram_depth)
            ram_cascade_width = math.ceil(buffer_width / ram_width)
            return int(ram_cascade_depth * ram_cascade_width * buffer_count)
        return 0

    def execute_node(self, context: dict[str, np.ndarray], graph: "GraphProto") -> None:
        """Execute this ConvolutionInputGenerator node.

        Performs sliding window generation for convolution operations.
        """
        mode = self.get_nodeattr("exec_mode")

        if mode == "cppsim":
            ConvolutionInputGenerator.execute_node(self, context, graph)
            # if depthwise = 1
            # interleave channels such that cppsim of ConvolutionInputGenerator_rtl
            # has a notion of SIMD parallelism. Subsequent VVAU_{hls/rtl} expects
            # the channels to be interleaved (i.e. to match their PE parallelism).
            if self.depthwise:
                node = self.onnx_node
                im2col_out = context[node.output[0]]
                simd = self.simd
                ofm_h, ofm_w = self.ofm_dim
                k_h, k_w = self.kernel_dim
                ifm_ch = self.ifm_channels
                im2col_out = im2col_out.reshape(1, ofm_h, ofm_w, k_h * k_w, ifm_ch // simd, simd)
                im2col_out = im2col_out.transpose(0, 1, 2, 4, 3, 5)
                im2col_out = im2col_out.reshape(1, ofm_h, ofm_w, ifm_ch * k_h * k_w)
                context[node.output[0]] = im2col_out
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)

    def prepare_codegen_default(self) -> tuple[Path, dict[str, list[str]]]:
        """Fill code generation dict for the default implementation style by computing
        the incremental addressing scheme for the circular buffer.
        """
        if self.dynamic_mode:
            template_select = "swg/swg_template_default_dynamic.sv"
        else:
            template_select = "swg/swg_template_default.sv"
        template_path = Path(get_settings().finn_rtllib) / template_select
        code_gen_dict: dict[str, list[str]] = {}

        ifm_ch = self.ifm_channels
        k_h, k_w = self.kernel_dim
        h, w = self.ifm_dim
        stride_h, stride_w = self.stride
        dilation_h, dilation_w = self.dilation
        depthwise = self.depthwise
        simd = self.simd

        pad = [0, 0, 0, 0]  # padding happens in separate padding node for now
        pad_h = pad[0] + pad[2]
        pad_w = pad[1] + pad[3]
        out_dim_h = im2col.compute_conv_output_dim(h, k_h, stride_h, pad_h, dilation_h)
        out_dim_w = im2col.compute_conv_output_dim(w, k_w, stride_w, pad_w, dilation_w)
        mmv_in = 1
        mmv_out = 1
        channel_factor = int(ifm_ch / simd)

        # compute minimal buffer length (assuming it holds 1 complete window)
        buffer_min_size = ((k_h - 1) * dilation_h * w + (k_w - 1) * dilation_w + 1) * channel_factor

        buffer_actual_size = self.get_buffer_depth()
        code_gen_dict["$BUF_ELEM_TOTAL$"] = [str(buffer_actual_size)]

        # compute some intermediate values, e.g., kernel "width" = k_w incl. dilation
        # or cols/rows that are skipped due to imperfect stride<->dim combination
        kernel_width = (k_w - 1) * dilation_w + 1
        kernel_height = (k_h - 1) * dilation_h + 1
        skip_columns = w % (kernel_width + (out_dim_w - 1) * stride_w)
        skip_rows = h % (kernel_height + (out_dim_h - 1) * stride_h)

        # compute address increment values for 5-loop nest
        addr_incr_end_simd = 1
        addr_incr_end_window_elem = (dilation_w - 1) * channel_factor + 1
        addr_incr_end_window_row = (
            ((w - kernel_width) * channel_factor)  # remaining line
            + ((dilation_h - 1) * w * channel_factor)  # skip lines
            + 1  # wrap-around of minimally sized buffer
        )
        addr_incr_end_window = -buffer_min_size + stride_w * channel_factor + 1
        addr_incr_end_row = (
            -buffer_min_size
            + ((skip_columns + kernel_width) * channel_factor)  # remaining line
            + ((stride_h - 1) * w * channel_factor)  # skip lines
            + 1
        )

        # re-use same controller structure -> re-assign address increments
        if depthwise:
            addr_incr_end_window_elem = dilation_w * channel_factor
            addr_incr_end_window_row = (
                channel_factor
                + (w - kernel_width) * channel_factor
                + (dilation_h - 1) * w * channel_factor
            )
            addr_incr_end_simd = -buffer_min_size + (channel_factor + 1)

        # sanity check for wrap logic
        if abs(addr_incr_end_window) > buffer_actual_size:
            raise FINNUserError(
                f"{self.onnx_node.name}: W increment > buffer size, "
                "try setting parallel_window=1"
            )
        if abs(addr_incr_end_row) > buffer_actual_size:
            raise FINNUserError(
                f"{self.onnx_node.name}: H increment > buffer size, "
                "try setting parallel_window=1"
            )

        # set certain threshold indices to detect when reading/writing finishes
        code_gen_dict["$LAST_READ_ELEM$"] = [str(h * w * channel_factor - 1)]
        code_gen_dict["$LAST_WRITE_ELEM$"] = [
            str(((h - skip_rows - 1) * w + (w - skip_columns)) * channel_factor - 1)
        ]

        # default controller loop structure: # iterations (counters) map directly
        loop_h_iterations = out_dim_h
        loop_w_iterations = out_dim_w
        loop_kh_iterations = k_h
        loop_kw_iterations = k_w
        loop_simd_iterations = channel_factor

        if depthwise and channel_factor > 1:
            # re-arrange existing controller loop structure for depthwise convolutions
            loop_kh_iterations = channel_factor
            loop_kw_iterations = k_h
            loop_simd_iterations = k_w
            addr_incr_end_simd_ = addr_incr_end_simd
            addr_incr_end_simd = addr_incr_end_window_elem
            addr_incr_end_window_elem = addr_incr_end_window_row
            addr_incr_end_window_row = addr_incr_end_simd_
            elem_per_window = k_h * k_w

            tail_incr_w = addr_incr_end_window + buffer_min_size - channel_factor
            tail_incr_h = addr_incr_end_row + buffer_min_size - channel_factor
            tail_incr_last_window = buffer_min_size - 1
            code_gen_dict["$IS_DEPTHWISE$"] = ["1"]
        else:
            # depthwise output format is equivalent to non-depthwise if SIMD=C
            elem_per_window = k_h * k_w * channel_factor

            tail_incr_w = addr_incr_end_window + buffer_min_size - 1
            tail_incr_h = addr_incr_end_row + buffer_min_size - 1
            tail_incr_last_window = buffer_min_size - 1
            code_gen_dict["$IS_DEPTHWISE$"] = ["0"]

        # support SIMD = IFMChannels and k_w = 1 cases
        # for k = [k_h, k_w] = [1, k_w], no adjustment is needed
        # for k = [k_h, k_w] = [1, 1], do not use this impl. style (mmv_out=K=1)
        # innermost loop is executed at least once -> adjust if needed
        if loop_simd_iterations == 1:
            # skip innermost SIMD loop completely
            if loop_kw_iterations == 1:
                # skip innermost KW loop completely
                code_gen_dict["$INNERMOST_STATE$"] = ["STATE_LOOP_KH"]
                loop_kh_iterations -= 1  # -1 because state is initial state
            else:
                code_gen_dict["$INNERMOST_STATE$"] = ["STATE_LOOP_KW"]
                loop_kw_iterations -= 1  # -1 because state is initial state
        else:
            code_gen_dict["$INNERMOST_STATE$"] = ["STATE_LOOP_SIMD"]
            loop_simd_iterations -= 1  # -1 because state is initial state

        cntr_bitwidth = math.ceil(
            math.log2(
                max(
                    loop_h_iterations - 2 + 1,
                    loop_w_iterations - 2 + 1,
                    loop_kh_iterations - 2 + 1,
                    loop_kw_iterations - 2 + 1,
                    loop_simd_iterations - 2 + 1,
                )
            )
        )
        code_gen_dict["$CNTR_BITWIDTH$"] = [str(cntr_bitwidth)]
        code_gen_dict["$LOOP_H_ITERATIONS$"] = [str(loop_h_iterations - 2)]
        code_gen_dict["$LOOP_W_ITERATIONS$"] = [str(loop_w_iterations - 2)]
        code_gen_dict["$LOOP_KH_ITERATIONS$"] = [str(loop_kh_iterations - 2)]
        code_gen_dict["$LOOP_KW_ITERATIONS$"] = [str(loop_kw_iterations - 2)]
        code_gen_dict["$LOOP_SIMD_ITERATIONS$"] = [str(loop_simd_iterations - 2)]

        incr_bitwidth = 1 + math.ceil(
            math.log2(
                max(
                    abs(addr_incr_end_simd) + 1,
                    abs(addr_incr_end_window_elem) + 1,
                    abs(addr_incr_end_window_row) + 1,
                    abs(addr_incr_end_window) + 1,
                    abs(addr_incr_end_row) + 1,
                    abs(tail_incr_w) + 1,
                    abs(tail_incr_h) + 1,
                    abs(tail_incr_last_window) + 1,
                )
            )
        )
        code_gen_dict["$INCR_BITWIDTH$"] = [str(incr_bitwidth)]
        code_gen_dict["$HEAD_INCR_SIMD$"] = [str(addr_incr_end_simd)]
        code_gen_dict["$HEAD_INCR_KW$"] = [str(addr_incr_end_window_elem)]
        code_gen_dict["$HEAD_INCR_KH$"] = [str(addr_incr_end_window_row)]
        code_gen_dict["$HEAD_INCR_W$"] = [str(addr_incr_end_window)]
        code_gen_dict["$HEAD_INCR_H$"] = [str(addr_incr_end_row)]
        code_gen_dict["$TAIL_INCR_W$"] = [str(tail_incr_w)]
        code_gen_dict["$TAIL_INCR_H$"] = [str(tail_incr_h)]
        code_gen_dict["$TAIL_INCR_LAST$"] = [str(tail_incr_last_window)]

        code_gen_dict["$ELEM_PER_WINDOW$"] = [str(elem_per_window)]
        code_gen_dict["$SIMD$"] = [str(simd)]
        code_gen_dict["$MMV_IN$"] = [str(mmv_in)]
        code_gen_dict["$MMV_OUT$"] = [str(mmv_out)]

        return template_path, code_gen_dict

    def prepare_codegen_parallel(self) -> tuple[Path, dict[str, list[str]]]:
        """Fill code generation dict for the parallel implementation style by computing
        the loop controller configuration and partitioning the fixed buffer into
        shift-registers (for parallel read access) and line buffers (for efficient
        LUTRAM/BRAM/URAM implementation).
        """
        template_path = Path(get_settings().finn_rtllib) / "swg/swg_template_parallel.sv"
        code_gen_dict: dict[str, list[str]] = {}

        ifm_ch = self.ifm_channels
        k_h, k_w = self.kernel_dim
        h, w = self.ifm_dim
        stride_h, stride_w = self.stride
        dilation_h, dilation_w = self.dilation
        simd = self.simd
        m_par = self.m_par

        pad = [0, 0, 0, 0]  # padding happens in separate padding node for now
        pad_h = pad[0] + pad[2]
        pad_w = pad[1] + pad[3]
        out_dim_h = im2col.compute_conv_output_dim(h, k_h, stride_h, pad_h, dilation_h)
        out_dim_w = im2col.compute_conv_output_dim(w, k_w, stride_w, pad_w, dilation_w)
        mmv_in = m_par * 1
        mmv_out = m_par * k_h * k_w
        channel_factor = int(ifm_ch / simd)

        # compute minimal buffer length (assuming it holds 1 complete window)
        buffer_min_size = ((k_h - 1) * dilation_h * w + (k_w - 1) * dilation_w) * channel_factor + 1

        buffer_actual_size = self.get_buffer_depth()
        code_gen_dict["$BUF_ELEM_TOTAL$"] = [str(buffer_actual_size)]

        # compute some intermediate values, e.g., kernel "width" = k_w incl. dilation
        # or cols/rows that are skipped due to imperfect stride<->dim combination
        kernel_width = (k_w - 1) * dilation_w + 1
        kernel_height = (k_h - 1) * dilation_h + 1
        skip_columns = w % (kernel_width + (out_dim_w - 1) * stride_w)
        skip_rows = h % (kernel_height + (out_dim_h - 1) * stride_h)

        # set certain threshold indices to detect when reading/writing finishes
        code_gen_dict["$LAST_READ_ELEM$"] = [str(h * w * channel_factor - 1)]
        code_gen_dict["$LAST_WRITE_ELEM$"] = [
            str(((h - skip_rows - 1) * w + (w - skip_columns)) * channel_factor - 1)
        ]

        # re-use default controller loop structure
        loop_h_iterations = out_dim_h
        loop_w_iterations = out_dim_w
        loop_kh_iterations = channel_factor
        loop_kw_iterations = 1
        loop_simd_iterations = 1

        if loop_kh_iterations == 1:
            if loop_w_iterations == 1:
                code_gen_dict["$INNERMOST_STATE$"] = ["STATE_LOOP_H"]
                loop_h_iterations -= 1  # -1 because state is initial state
            else:
                code_gen_dict["$INNERMOST_STATE$"] = ["STATE_LOOP_W"]
                loop_w_iterations -= 1  # -1 because state is initial state
        else:
            code_gen_dict["$INNERMOST_STATE$"] = ["STATE_LOOP_KH"]
            loop_kh_iterations -= 1  # -1 because state is initial state

        # set head address increment values
        addr_incr_end_simd = 1
        addr_incr_end_window_elem = 1
        addr_incr_end_window_row = 1
        addr_incr_end_window = (stride_w - 1) * channel_factor + 1
        addr_incr_end_row = ((skip_columns + (kernel_width - 1)) * channel_factor + 1) + (
            (stride_h - 1) * w * channel_factor
        )

        # add init value for CURRENT_ELEM counter = last elem of first window
        code_gen_dict["$FIRST_WRITE_ELEM$"] = [str(buffer_min_size - 1)]

        cntr_bitwidth = math.ceil(
            math.log2(
                max(
                    loop_h_iterations - 2 + 1,
                    loop_w_iterations - 2 + 1,
                    loop_kh_iterations - 2 + 1,
                    loop_kw_iterations - 2 + 1,
                    loop_simd_iterations - 2 + 1,
                )
            )
        )
        code_gen_dict["$CNTR_BITWIDTH$"] = [str(cntr_bitwidth)]
        code_gen_dict["$LOOP_H_ITERATIONS$"] = [str(loop_h_iterations - 2)]
        code_gen_dict["$LOOP_W_ITERATIONS$"] = [str(loop_w_iterations - 2)]
        code_gen_dict["$LOOP_KH_ITERATIONS$"] = [str(loop_kh_iterations - 2)]
        code_gen_dict["$LOOP_KW_ITERATIONS$"] = [str(loop_kw_iterations - 2)]
        code_gen_dict["$LOOP_SIMD_ITERATIONS$"] = [str(loop_simd_iterations - 2)]

        incr_bitwidth = 1 + math.ceil(
            math.log2(
                max(
                    abs(addr_incr_end_simd) + 1,
                    abs(addr_incr_end_window_elem) + 1,
                    abs(addr_incr_end_window_row) + 1,
                    abs(addr_incr_end_window) + 1,
                    abs(addr_incr_end_row) + 1,
                )
            )
        )
        code_gen_dict["$INCR_BITWIDTH$"] = [str(incr_bitwidth)]
        code_gen_dict["$HEAD_INCR_SIMD$"] = [str(addr_incr_end_simd)]
        code_gen_dict["$HEAD_INCR_KW$"] = [str(addr_incr_end_window_elem)]
        code_gen_dict["$HEAD_INCR_KH$"] = [str(addr_incr_end_window_row)]
        code_gen_dict["$HEAD_INCR_W$"] = [str(addr_incr_end_window)]
        code_gen_dict["$HEAD_INCR_H$"] = [str(addr_incr_end_row)]
        # not used, set to zero:
        code_gen_dict["$TAIL_INCR_W$"] = ["0"]
        code_gen_dict["$TAIL_INCR_H$"] = ["0"]
        code_gen_dict["$TAIL_INCR_LAST$"] = ["0"]
        code_gen_dict["$IS_DEPTHWISE$"] = ["0"]

        code_gen_dict["$SIMD$"] = [str(simd)]
        code_gen_dict["$MMV_IN$"] = [str(mmv_in)]
        code_gen_dict["$MMV_OUT$"] = [str(mmv_out)]

        # prepare buffer partitioning into "reg_fifos" and "bram_fifos"
        # use normalized ([H,W]=[1,W]) dimensions for 1D case
        (
            ifm_ch,
            [_ifm_dim_h, _ifm_dim_w],
            [_ofm_dim_h, _ofm_dim_w],
            [k_h, k_w],
            [stride_h, stride_w],
            [dilation_h, dilation_w],
        ) = self.get_1d_conv_attrs_normalized()

        reg_fifos = []
        bram_fifos_depth = []

        px_idx = 0
        for ky in range(k_h):
            reg_fifo = []
            for kx in range(k_w):
                for c in range(channel_factor):
                    if c < (channel_factor - 1):
                        if not (ky == 0 and kx == 0):
                            reg_fifo.append(-1)
                            px_idx += 1
                    else:
                        reg_fifo.append(px_idx)
                        px_idx += 1
                if kx < (k_w - 1):
                    reg_fifo.extend([-1] * ((dilation_w - 1) * channel_factor))
                    px_idx += (dilation_w - 1) * channel_factor
            reg_fifos.append(reg_fifo)

            if ky < (k_h - 1):
                line_buffer_len = ((w - kernel_width) + w * (dilation_h - 1)) * channel_factor
                bram_fifos_depth.append(line_buffer_len)
                px_idx += line_buffer_len

        code_gen_dict["$GENERATE_REG_FIFOS$"] = []
        for i, reg_fifo in enumerate(reg_fifos):
            reg_fifo_len = len(reg_fifo)
            code_gen_dict["$GENERATE_REG_FIFOS$"].append(
                f"""
                wire [IN_WIDTH-1:0] reg_fifo_{i}_in;
                wire [IN_WIDTH-1:0] reg_fifo_{i}_out;
                wire [IN_WIDTH*{reg_fifo_len}-1:0] reg_fifo_{i};
                swg_reg_buffer
                #(
                .WIDTH(IN_WIDTH),
                .DEPTH({reg_fifo_len})
                )
                reg_buffer_inst_{i}
                (
                    .clk(clk),
                    .shift_enable(shift_enable),
                    .shift_in(reg_fifo_{i}_in),
                    .shift_out(reg_fifo_{i}_out),
                    .data_out(reg_fifo_{i})
                );"""
            )

        code_gen_dict["$GENERATE_BRAM_FIFOS$"] = []
        ram_style = self.ram_style
        for i, bram_fifo_depth in enumerate(bram_fifos_depth):
            code_gen_dict["$GENERATE_BRAM_FIFOS$"].append(
                f"""
                wire [IN_WIDTH-1:0] bram_fifo_{i}_in;
                wire [IN_WIDTH-1:0] bram_fifo_{i}_out;
                swg_ram_buffer
                #(
                .WIDTH(IN_WIDTH),
                .DEPTH({bram_fifo_depth}),
                .RAM_STYLE("{ram_style}")
                )
                ram_buffer_inst_{i}
                (
                    .clk(clk),
                    .rst_n(rst_n),
                    .shift_enable(shift_enable),
                    .shift_in(bram_fifo_{i}_in),
                    .shift_out(bram_fifo_{i}_out)
                );"""
            )

        code_gen_dict["$GENERATE_OUTPUT_MAPPING$"] = []
        out_idx = mmv_out - 1
        for fifo_id, reg_fifo in enumerate(reg_fifos):
            for _fifo_idx, access_idx in enumerate(reg_fifo):
                if access_idx != -1:
                    code_gen_dict["$GENERATE_OUTPUT_MAPPING$"].append(
                        f"""assign data_out[OUT_ELEM_WIDTH*{out_idx}+:OUT_ELEM_WIDTH]
                        = reg_fifo_{fifo_id}[
                        {len(reg_fifo) - 1 - int((max(reg_fifo) - access_idx) / m_par)}
                        *{m_par}*OUT_ELEM_WIDTH+
                        OUT_ELEM_WIDTH*{(max(reg_fifo) - access_idx) % m_par}+:OUT_ELEM_WIDTH];"""
                    )
                    # reversal: out_idx=0 -> oldest buffer element -> highest access_idx
                    out_idx = out_idx - 1
        if out_idx != -1:
            raise FINNInternalError(
                f"{self.onnx_node.name}: not all output vector elements connected"
            )

        code_gen_dict["$GENERATE_BUFFER_CONNECTION$"] = []
        for i in range(len(reg_fifos)):
            if i == 0:
                # first FIFO containing newest elements -> input comes from input reg
                code_gen_dict["$GENERATE_BUFFER_CONNECTION$"].append(
                    f"""assign reg_fifo_{i}_in = data_in;"""
                )
            else:
                # other REG FIFOs -> input comes from connected BRAM FIFO (line buffer)
                input_fifo_id = i - 1
                code_gen_dict["$GENERATE_BUFFER_CONNECTION$"].append(
                    f"""assign reg_fifo_{i}_in = bram_fifo_{input_fifo_id}_out;
                    """
                )
        for i in range(len(bram_fifos_depth)):
            input_fifo_id = i
            code_gen_dict["$GENERATE_BUFFER_CONNECTION$"].append(
                f"""assign bram_fifo_{i}_in = reg_fifo_{input_fifo_id}_out;
                """
            )

        return template_path, code_gen_dict

    def select_impl_style(self) -> Literal["parallel", "default"]:
        """Select implementation style based on folding configuration."""
        simd = self.simd
        m_par = self.m_par
        depthwise = self.depthwise
        ifm_ch = self.ifm_channels
        ifm_dim_h, ifm_dim_w = self.ifm_dim
        stride_h, stride_w = self.stride
        dilation_h, dilation_w = self.dilation
        k_h, k_w = self.kernel_dim
        kernel_width = (k_w - 1) * dilation_w + 1  # incl. dilation
        kernel_height = (k_h - 1) * dilation_h + 1  # incl. dilation

        # check for valid configuration
        if not (
            kernel_height <= ifm_dim_h
            and kernel_width <= ifm_dim_w
            and stride_h <= ifm_dim_h
            and stride_w <= ifm_dim_w
        ):
            raise FINNUserError(
                f"{self.onnx_node.name}: illegal conv configuration, "
                "kernel or stride > FM dimension"
            )

        # init folding config
        if self.parallel_window:
            # mmv_in = M * 1
            mmv_out = m_par * k_h * k_w
        else:
            # mmv_in = 1
            mmv_out = 1
            if ifm_ch % simd != 0:
                raise FINNUserError(
                    f"{self.onnx_node.name}: SIMD ({simd}) must divide IFMChannels ({ifm_ch})"
                )

        # choose implementation style
        if mmv_out > 1 or (k_h == 1 and k_w == 1):
            impl_style = "parallel"
            if depthwise or (k_h == 1 and k_w == 1):
                # allow SIMD < IFM_CH in depthwise mode (VVAU supports the resulting data layout)
                # also allowed for 1x1 kernel since depthwise and non-depthwise are equivalent
                if ifm_ch % simd != 0:
                    raise FINNUserError(
                        f"{self.onnx_node.name}: SIMD ({simd}) must divide "
                        f"IFMChannels ({ifm_ch})"
                    )
            elif ifm_ch != simd:
                raise FINNUserError(
                    f"{self.onnx_node.name}: SIMD ({simd}) must be equal to IFMChannels ({ifm_ch})"
                )
        else:
            impl_style = "default"

        return impl_style

    def generate_hdl(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:  # noqa: ARG002
        """Generate HDL code and wrapper for the IP, depending on required
        implementation style.
        """
        impl_style = self.select_impl_style()

        # prepare code generation by filling out dictionaries
        if impl_style == "default":
            template_path, code_gen_dict = self.prepare_codegen_default()
        else:
            template_path, code_gen_dict = self.prepare_codegen_parallel()
            if self.dynamic_mode:
                raise FINNUserError(
                    f"{self.onnx_node.name}: dynamic mode is not compatible with parallel_window"
                )

        # add general parameters to dictionary
        code_gen_dict["$TOP_MODULE_NAME$"] = [self.get_verilog_top_module_name()]
        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())
        code_gen_dict["$BIT_WIDTH$"] = [str(self.get_input_datatype().bitwidth())]
        code_gen_dict["$IN_WIDTH_PADDED$"] = [
            str(roundup_to_integer_multiple(self.get_instream_width(), 8))
        ]
        code_gen_dict["$OUT_WIDTH_PADDED$"] = [
            str(roundup_to_integer_multiple(self.get_outstream_width(), 8))
        ]
        ram_style = self.ram_style
        code_gen_dict["$RAM_STYLE$"] = [f'"{ram_style}"']

        # apply code generation to templates
        code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))
        template = template_path.read_text()
        if self.dynamic_mode:
            template_select = "swg/swg_template_wrapper_dynamic.v"
        else:
            template_select = "swg/swg_template_wrapper.v"
        template_wrapper = (Path(get_settings().finn_rtllib) / template_select).read_text()
        template_axilite = (
            Path(get_settings().finn_rtllib) / "swg/swg_template_axilite.v"
        ).read_text()
        for key in code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(code_gen_dict[key])
            template = template.replace(key, code_gen_line)
            template_wrapper = template_wrapper.replace(key, code_gen_line)
            template_axilite = template_axilite.replace(key, code_gen_line)
        gen_top_module = cast("str", self.get_nodeattr("gen_top_module"))
        (code_gen_dir / f"{gen_top_module}_impl.sv").write_text(template)
        (code_gen_dir / f"{gen_top_module}_wrapper.v").write_text(template_wrapper)

        # AXI-Lite reg. file component is only needed for dynamic mode
        if self.dynamic_mode:
            (code_gen_dir / f"{gen_top_module}_axilite.v").write_text(template_axilite)

        # Copy static source file for common core components
        rtllib_dir = Path(get_settings().finn_rtllib)
        shutil.copy2(rtllib_dir / "swg/swg_common.sv", code_gen_dir)
        shutil.copy2(rtllib_dir / "swg/swg_pkg.sv", code_gen_dir)

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", str(code_gen_dir))
        self.set_nodeattr("ip_path", str(code_gen_dir))

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        """Return list of RTL files required for this node."""
        if abspath:
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen")) + "/"
            rtllib_dir = str(Path(get_settings().finn_rtllib) / "swg") + "/"
        else:
            code_gen_dir = ""
            rtllib_dir = ""
        gen_top_module = cast("str", self.get_nodeattr("gen_top_module"))
        verilog_files = [
            rtllib_dir + "swg_pkg.sv",
            code_gen_dir + gen_top_module + "_wrapper.v",
            code_gen_dir + gen_top_module + "_impl.sv",
            rtllib_dir + "swg_common.sv",
        ]
        if self.dynamic_mode:
            verilog_files.append(code_gen_dir + gen_top_module + "_axilite.v")

        return verilog_files

    def code_generation_ipi(self) -> list[str]:
        """Construct and return the TCL for node instantiation in Vivado IPI."""
        code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))
        gen_top_module = cast("str", self.get_nodeattr("gen_top_module"))

        sourcefiles = [
            "swg_pkg.sv",
            gen_top_module + "_wrapper.v",
            gen_top_module + "_impl.sv",
            "swg_common.sv",
        ]

        if self.dynamic_mode:
            sourcefiles += [gen_top_module + "_axilite.v"]

        sourcepaths = [str(code_gen_dir / f) for f in sourcefiles]

        cmd = [f"add_files -norecurse {f}" for f in sourcepaths]
        cmd += [f"create_bd_cell -type module -reference {gen_top_module} {self.onnx_node.name}"]
        return cmd

    def get_verilog_top_module_intf_names(self) -> dict[str, list[tuple[str, int]] | list[str]]:
        """Return a dict of names of input and output interfaces.

        Overloads the default HLSCustomOp implementation to add the axilite
        control interface. The keys reflect the protocols each interface
        implements: 'clk', 'rst', 'm_axis', 's_axis', 'aximm', 'axilite'.
        Values are lists of tuples (axis, aximm) or names (axilite): 'axis'
        tuples correspond to the list of node inputs in order, each tuple is
        (interface_name, interface_width_bits). axilite always assumed to be
        32 bits and is not a tuple (name only). Each block must have at most
        one aximm and one axilite.
        """
        intf_names = super().get_verilog_top_module_intf_names()
        if self.dynamic_mode:
            intf_names["axilite"] = ["s_axilite"]
        return intf_names

    def get_dynamic_config(
        self,
        ifm_dim: list[int] | None = None,
        stride: list[int] | None = None,
        dilation: list[int] | None = None,
    ) -> dict[str, tuple[int, int]]:
        """Return a configuration dict to re-configure FM dimension during runtime.

        Stride and dilation can also be changed. Certain restrictions apply
        (e.g. component must be synthesized for largest buffer size).
        """
        # NOTE: For better driver integration, this functionality could be packaged
        # as a standalone function in the future
        if self.select_impl_style() != "default":
            raise FINNUserError(
                f"{self.onnx_node.name}: impl. style is incompatible with dynamic mode"
            )

        if ifm_dim is None:
            ifm_dim = self.ifm_dim
        k = self.kernel_dim
        if stride is None:
            stride = self.stride
        if dilation is None:
            dilation = self.dilation

        k_h, k_w = k
        stride_h, stride_w = stride
        dilation_h, dilation_w = dilation
        ifm_dim_h, ifm_dim_w = ifm_dim
        ofm_dim_h = compute_conv_output_dim(ifm_dim_h, k_h, stride_h, 0, dilation_h)
        ofm_dim_w = compute_conv_output_dim(ifm_dim_w, k_w, stride_w, 0, dilation_w)
        ofm_dim = [ofm_dim_h, ofm_dim_w]

        # update attributes and perform sanity check
        original_buffer_depth = self.get_buffer_depth()
        self.set_nodeattr("IFMDim", cast("list[str | int | float]", list(ifm_dim)))
        self.set_nodeattr("OFMDim", cast("list[str | int | float]", ofm_dim))
        self.set_nodeattr("Stride", cast("list[str | int | float]", stride))
        self.set_nodeattr("Dilation", cast("list[str | int | float]", dilation))
        if self.get_buffer_depth() > original_buffer_depth:
            raise FINNUserError(
                f"{self.onnx_node.name}: requested dynamic configuration does not fit "
                "in generated buffer implementation"
            )

        # (re-)call codegen and extract new values
        # each setting is mapped to an axi-lite register address
        _template_path, code_gen_dict = self.prepare_codegen_default()
        return {
            "cfg_wren": (0 * 4, 1),
            "cfg_cntr_simd": (1 * 4, int(code_gen_dict["$LOOP_SIMD_ITERATIONS$"][0])),
            "cfg_cntr_kw": (2 * 4, int(code_gen_dict["$LOOP_KW_ITERATIONS$"][0])),
            "cfg_cntr_kh": (3 * 4, int(code_gen_dict["$LOOP_KH_ITERATIONS$"][0])),
            "cfg_cntr_w": (4 * 4, int(code_gen_dict["$LOOP_W_ITERATIONS$"][0])),
            "cfg_cntr_h": (5 * 4, int(code_gen_dict["$LOOP_H_ITERATIONS$"][0])),
            "cfg_incr_head_simd": (6 * 4, int(code_gen_dict["$HEAD_INCR_SIMD$"][0])),
            "cfg_incr_head_kw": (7 * 4, int(code_gen_dict["$HEAD_INCR_KW$"][0])),
            "cfg_incr_head_kh": (8 * 4, int(code_gen_dict["$HEAD_INCR_KH$"][0])),
            "cfg_incr_head_w": (9 * 4, int(code_gen_dict["$HEAD_INCR_W$"][0])),
            "cfg_incr_head_h": (10 * 4, int(code_gen_dict["$HEAD_INCR_H$"][0])),
            "cfg_incr_tail_w": (11 * 4, int(code_gen_dict["$TAIL_INCR_W$"][0])),
            "cfg_incr_tail_h": (12 * 4, int(code_gen_dict["$TAIL_INCR_H$"][0])),
            "cfg_incr_tail_last": (13 * 4, int(code_gen_dict["$TAIL_INCR_LAST$"][0])),
            "cfg_last_read": (14 * 4, int(code_gen_dict["$LAST_READ_ELEM$"][0])),
            "cfg_last_write": (15 * 4, int(code_gen_dict["$LAST_WRITE_ELEM$"][0])),
        }

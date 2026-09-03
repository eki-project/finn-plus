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

"""Vector-Vector Activation Unit (VVAU) implementation for FPGA dataflow.

This module contains the VVAU class which provides hardware abstraction for
vector-vector activation layers in FPGA implementations. The VVAU performs
convolutional operations with thresholding activation functions.
"""

import math
import numpy as np
import onnx.numpy_helper as np_helper
import textwrap
from onnx import NodeProto
from pathlib import Path
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.util.basic import (
    calculate_matvec_accumulator_range,
    interleave_matrix_outer_dim_from_partitions,
    roundup_to_integer_multiple,
)
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.data_packing import numpy_to_hls_code, pack_innermost_dim_as_hex_string
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from onnx import GraphProto

# Type of the dictionary returned by get_nodeattr_types: maps attribute names to
# their (dtype, required, default[, allowed_values]) specification tuples
NodeAttrTypes = dict[
    str,
    tuple[str, bool, int | float | str | bool | np.ndarray | list]
    | tuple[str, bool, int | float | str | bool | np.ndarray | list, set | None],
]


class VVAU(HWCustomOp):
    """Abstraction layer for HW implementation of VectorVectorActivation layers."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize the VVAU (Vector-Vector Activation Unit) instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Get the dictionary of node attribute types for VVAU."""
        my_attrs: NodeAttrTypes = {
            "PE": ("i", True, 0),
            "SIMD": ("i", False, 1),
            "Dim": ("ints", True, []),  # [H, W]
            "Channels": ("i", True, 0),
            "Kernel": ("ints", True, []),  # [H, W]
            "resType": ("s", False, "auto", {"auto", "lut", "dsp"}),
            "ActVal": ("i", False, 0),
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # FINN DataType for accumulator -- auto-computed and updated
            "accDataType": ("s", False, "INT32"),
            # no-activation mode (produce accumulators)
            "noActivation": ("i", False, 0, {0, 1}),
            # memory mode for the layer weights
            # internal_embedded -- embedded weights, long compile/synth times
            # internal_decoupled -- default, streaming weights with streamer packaged inside IP
            # external -- streaming weights with external streamer
            "mem_mode": (
                "s",
                False,
                "internal_decoupled",
                {"internal_embedded", "internal_decoupled", "external"},
            ),
            # (mem_mode = internal_decoupled only) whether weights will be writable through
            # an AXI-lite interface during runtime
            # 1 for enabled, 0 for disabled.
            # see finn-rtllib/memstream/doc/README for more about the memory
            # address map used for writable weights
            # IMPORTANT: After using AXI lite to either read or write the weights,
            # always "flush" the accelerator by first passing a dummy input
            # vector through the accelerator. This will get rid of any old
            # weight data from the weight FIFOs.
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
            # FPGA resource type for memories in internal_decoupled mode
            # auto -- let Vivado decide
            # block -- use BRAM
            # distributed -- use LUTRAM
            # ultra -- use UltraRAM (URAM), must have runtime_writeable_weights=1
            # see also https://www.xilinx.com/support/answers/38070.html
            "ram_style": (
                "s",
                False,
                "auto",
                {"auto", "block", "distributed", "ultra"},
            ),
            # use xnor-popcount for binary weights/inputs, thus treating them
            # as bipolar
            "binaryXnorMode": ("i", False, 0, {0, 1}),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    @property
    def pe(self) -> int:
        """Get the PE (output-channel) parallelism."""
        return cast("int", self.get_nodeattr("PE"))

    @property
    def simd(self) -> int:
        """Get the SIMD (kernel) parallelism."""
        return cast("int", self.get_nodeattr("SIMD"))

    @property
    def channels(self) -> int:
        """Get the number of channels."""
        return cast("int", self.get_nodeattr("Channels"))

    @property
    def dim(self) -> list[int]:
        """Get the feature map dimensions [H, W]."""
        return list(cast("list[int]", self.get_nodeattr("Dim")))

    @property
    def kernel(self) -> list[int]:
        """Get the kernel dimensions [H, W]."""
        return list(cast("list[int]", self.get_nodeattr("Kernel")))

    @property
    def res_type(self) -> str:
        """Get the requested multiplier resource type (auto/lut/dsp)."""
        return cast("str", self.get_nodeattr("resType"))

    @property
    def act_val(self) -> int:
        """Get the activation bias applied by the thresholding step."""
        return cast("int", self.get_nodeattr("ActVal"))

    @property
    def mem_mode(self) -> str:
        """Get the weight memory mode (internal_embedded/internal_decoupled/external)."""
        return cast("str", self.get_nodeattr("mem_mode"))

    @property
    def ram_style(self) -> str:
        """Get the FPGA resource type for the weight memory."""
        return cast("str", self.get_nodeattr("ram_style"))

    @property
    def no_activation(self) -> int:
        """Get whether the node runs without thresholding (0/1)."""
        return cast("int", self.get_nodeattr("noActivation"))

    @property
    def binary_xnor_mode(self) -> int:
        """Get whether xnor-popcount (bipolar) mode is enabled (0/1)."""
        return cast("int", self.get_nodeattr("binaryXnorMode"))

    @property
    def runtime_writeable_weights(self) -> int:
        """Get whether weights are writeable at runtime via AXI-lite (0/1)."""
        return cast("int", self.get_nodeattr("runtime_writeable_weights"))

    def instantiate_ip(self, cmd: list[str]) -> None:
        """Append the backend-specific IP instantiation TCL to ``cmd``."""
        raise NotImplementedError

    def _infer_sparse_weight_tensor(
        self, w_conv: np.ndarray, k_h: int, k_w: int, channels: int
    ) -> np.ndarray:
        """Convert dense convolution weights to sparse weight tensor format."""
        w_sparse = np.zeros((channels, channels, k_h, k_w), dtype=np.float32)
        for ch in range(channels):
            w_sparse[ch][ch] = w_conv[ch][0]
        w_conv = w_sparse.astype(np.float32)
        w_matmul = w_conv.transpose(0, 2, 3, 1)
        w_matmul = w_matmul.reshape(channels, channels * k_h * k_w)
        return w_matmul.T

    def execute_node(self, context: dict[str, np.ndarray], graph: "GraphProto") -> None:
        """Execute the VVAU node operation.

        Performs the vector-vector activation computation including matrix
        multiplication and optional thresholding activation.
        """
        node = self.onnx_node
        in_act = context[node.input[0]]
        (_, dim_h, dim_w, _) = in_act.shape
        (k_h, k_w) = self.kernel
        channels = self.channels
        producer = next((x for x in graph.node if x.output[0] == node.input[0]), None)
        if producer is not None and producer.op_type in (
            "Im2Col",
            "ConvolutionInputGenerator",
        ):
            pe = channels
        else:
            pe = self.pe

        # Reorder the input activations. Note that PE gets interleaved by the SWG,
        # so we have to untangle and for simplicity of computation assume pe=1.
        # Note that PE has no effect on the QONNX node
        in_act = in_act.reshape(1, dim_h, dim_w, channels // pe, k_h * k_w, pe)
        in_act = in_act.transpose(0, 1, 2, 4, 3, 5)
        in_act = in_act.reshape(1, dim_h, dim_w, channels * k_h * k_w)
        # Reshape weights in appropriate format
        vvau_w_init = next(x for x in graph.initializer if x.name == node.input[1])
        vvau_w = np_helper.to_array(vvau_w_init)
        vvau_w_onnx = self._infer_sparse_weight_tensor(vvau_w, k_h, k_w, channels)

        # result is in [N, H, W, C] format
        result = np.matmul(in_act, vvau_w_onnx)
        if (
            self.get_nodeattr("inputDataType") == "BIPOLAR"
            and self.get_nodeattr("weightDataType") == "BIPOLAR"
        ):
            result = (result + k_h * k_w) / 2

        if self.no_activation == 0:
            vvau_thr_init = next(x for x in graph.initializer if x.name == node.input[2])
            vvau_thr = np_helper.to_array(vvau_thr_init)
            odt_is_bipolar = self.get_nodeattr("outputDataType") == "BIPOLAR"
            out_scale = 2 if odt_is_bipolar else 1
            out_bias = -1 if odt_is_bipolar else self.act_val
            # NHWC to NCHW for multithreshold node
            result = result.transpose((0, 3, 1, 2))
            result = multithreshold(result, vvau_thr, out_scale, out_bias)
            # NCHW to NHWC
            result = result.transpose((0, 2, 3, 1))

        context[node.output[0]] = result

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer and set the node's data types based on the model."""
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype(0):
            log.warning(
                f"inputDataType changing for {node.name}: "
                f"{self.get_input_datatype(0)!s} -> {idt!s}"
            )
        self.set_nodeattr("inputDataType", idt.name)
        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:
        """Return FINN DataType of input."""
        # when performing FIFO insertion on an FC layer with ext weights, the ind
        # parameter can be > 0 (referring to the weights) so handle that here
        if ind == 0:
            return DataType[cast("str", self.get_nodeattr("inputDataType"))]
        if ind == 1:
            return DataType[cast("str", self.get_nodeattr("weightDataType"))]
        raise FINNInternalError(f"{self.onnx_node.name}: undefined input ind {ind} for this layer")

    def get_accumulator_datatype(self) -> BaseDataType:
        """Return FINN DataType of accumulator."""
        return DataType[cast("str", self.get_nodeattr("accDataType"))]

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of output."""
        return DataType[cast("str", self.get_nodeattr("outputDataType"))]

    def get_instream_width(self, ind: int = 0) -> int:
        """Return the input stream width in bits for the specified input."""
        if ind == 0:
            i_bits = self.get_input_datatype(ind).bitwidth()
            return i_bits * self.simd * self.pe
        if ind == 1:
            if self.mem_mode in ("internal_decoupled", "external"):
                wp = self.get_input_datatype(1).bitwidth()
                return self.simd * self.pe * wp
            return 0
        if ind == 2:
            # check if integrated thresholding and return 0
            # because threshold values are always embedded
            # or raise exception if there shouldn't be a third input to the node
            if not self.no_activation:
                return 0
            raise FINNInternalError(f"{self.onnx_node.name}: input ind 2 out of range")
        raise FINNInternalError(f"{self.onnx_node.name}: undefined input ind {ind} for this layer")

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return the output stream width in bits."""
        o_bits = self.get_output_datatype().bitwidth()
        return o_bits * self.pe

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return the folded input shape for hardware implementation."""
        k_h, k_w = self.kernel
        dim_h, dim_w = self.dim
        ch = self.channels
        simd = self.simd
        pe = self.pe
        kernel_2 = k_h * k_w
        if kernel_2 % simd != 0:
            raise FINNInternalError("Requirement kernel (k_h * k_w) divisable by SIMD is violated.")
        sf = kernel_2 // simd
        if ch % pe != 0:
            raise FINNInternalError("Requirement Channels divisable by PE is violated.")
        nf = ch // pe

        if ind == 0:
            # calculate shape of input 0
            return (1, dim_h, dim_w, sf * nf, simd * pe)
        if ind == 1 and self.mem_mode == "external":
            # calculate shape of input 1 (weights)
            return (1, sf * nf, pe)
        raise FINNInternalError(f"{self.onnx_node.name}: undefined input shape for requested input")

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the folded output shape for hardware implementation."""
        ch = self.channels
        pe = self.pe
        nf = ch // pe
        dim_h, dim_w = self.dim
        return (1, dim_h, dim_w, nf, pe)

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the normal (unfolded) input shape."""
        dim_h, dim_w = self.dim
        ch = self.channels
        k_h, k_w = self.kernel
        return (1, dim_h, dim_w, k_h * k_w * ch)

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the normal (unfolded) output shape."""
        ch = self.channels
        dim_h, dim_w = self.dim
        return (1, dim_h, dim_w, ch)

    def calc_wmem(self) -> int:
        """Calculate and return WMEM."""
        ch = self.channels
        k_h, k_w = self.kernel
        return (k_h * k_w * ch // self.pe) // self.simd

    def calc_tmem(self) -> int:
        """Calculate and return TMEM."""
        if self.no_activation == 1:
            return 0
        return self.channels // self.pe

    def uram_estimation(self) -> int:
        """Estimate UltraRAM (URAM) usage for this layer."""
        p = self.pe
        q = self.simd
        w = self.get_input_datatype(1).bitwidth()
        omega = self.calc_wmem()
        mem_width = q * w * p
        mmode = self.mem_mode
        mstyle = self.ram_style
        if (
            (mmode == "internal_decoupled" and mstyle != "ultra")
            or (mmode == "internal_embedded")
            or (mmode == "external")
        ):
            return 0
        width_multiplier = math.ceil(mem_width / 72)
        depth_multiplier = math.ceil(omega / 4096)
        return width_multiplier * depth_multiplier

    def bram_estimation(self) -> int:
        """Calculate resource estimation for BRAM."""
        # TODO add in/out FIFO contributions
        p = self.pe
        q = self.simd
        w = self.get_input_datatype(1).bitwidth()
        omega = self.calc_wmem()
        mem_width = q * w * p
        # assuming SDP mode RAMB18s (see UG573 Table 1-10)
        # since this is HLS memory, not using the full width of a BRAM
        # assuming memories up to 128 deep get implemented in LUTs
        mmode = self.mem_mode
        mstyle = self.ram_style
        if (
            (mmode == "internal_decoupled" and mstyle in ["distributed", "ultra"])
            or (mstyle == "auto" and self.calc_wmem() <= 128)
            or (mmode == "internal_embedded" and self.calc_wmem() <= 128)
            or (mmode == "external")
        ):
            return 0

        if mem_width == 1:
            return math.ceil(omega / 16384)
        if mem_width == 2:
            return math.ceil(omega / 8192)
        if mem_width <= 4:
            return (math.ceil(omega / 4096)) * (math.ceil(mem_width / 4))
        if mem_width <= 9:
            return (math.ceil(omega / 2048)) * (math.ceil(mem_width / 8))
        if mem_width <= 18 or omega > 512:
            return (math.ceil(omega / 1024)) * (math.ceil(mem_width / 16))
        return (math.ceil(omega / 512)) * (math.ceil(mem_width / 32))

    def bram_efficiency_estimation(self) -> float:
        """Estimate BRAM efficiency (utilization) for this layer."""
        p = self.pe
        w = self.get_input_datatype(1).bitwidth()
        omega = self.calc_wmem()
        bram16_est = self.bram_estimation()
        if bram16_est == 0:
            return 1.0
        wbits = w * p * omega
        bram16_est_capacity = bram16_est * 36 * 512
        return wbits / bram16_est_capacity

    def uram_efficiency_estimation(self) -> float:
        """Estimate URAM efficiency: parameter storage needed / allocated URAM storage."""
        w = self.get_input_datatype(1).bitwidth()
        d_in = int(np.prod(self.kernel))
        d_out = self.channels
        uram_est = self.uram_estimation()
        if uram_est == 0:
            return 1.0
        wbits = w * d_in * d_out
        uram_est_capacity = uram_est * 72 * 4096
        return wbits / uram_est_capacity

    def get_exp_cycles(self) -> int:
        """Get the expected number of execution cycles for this layer."""
        pe = self.pe
        simd = self.simd
        ch = self.channels
        dim_h, dim_w = self.dim
        k_h, k_w = self.kernel
        # currently FINN supports for vvau a batch size of 1
        batch_size = 1
        # since mmv != 1 is not supported yet, we set mmv for now to 1
        mmv = 1
        exp_cycles = ((ch * k_h * k_w) / pe / simd) * batch_size * (dim_h * dim_w) / mmv
        return int(exp_cycles)

    def minimize_accumulator_width(self, model: ModelWrapper) -> BaseDataType:
        """Minimize the accumulator bit width.

        The width is derived from the weight values, input data types, and the
        size of the dot product.
        """
        weights = model.get_initializer(self.onnx_node.input[1])
        if not isinstance(weights, np.ndarray):
            raise FINNInternalError(f"{self.onnx_node.name}: weight initializer is missing")
        k_h, k_w = self.kernel
        fm = self.channels
        # put weights into the shape expected by calculate_matvec_accumulator_range
        weights = weights.reshape(fm, k_h * k_w).transpose()
        # since in the calculation the values of the weight matrix are used,
        # for the bipolar case they need to be converted to bipolar
        if self.binary_xnor_mode:
            weights = 2 * weights - 1

        idt = self.get_input_datatype(0)

        # if runtime-writeable weights or mem_mode=external, then the values of the weights can
        # change and we need to use the worst-case values from the datatypes
        if self.runtime_writeable_weights or self.mem_mode == "external":
            wdt = self.get_input_datatype(1)
            lower_worst = wdt.min() * np.ones((k_h * k_w, fm))
            lower_range = calculate_matvec_accumulator_range(lower_worst, idt)
            upper_worst = wdt.max() * np.ones((k_h * k_w, fm))
            upper_range = calculate_matvec_accumulator_range(upper_worst, idt)
            acc_min = min(min(lower_range), min(upper_range))
            acc_max = max(max(lower_range), max(upper_range))
        else:
            (acc_min, acc_max) = calculate_matvec_accumulator_range(weights, idt)

        # if the acc_range is always greater than 0, then acc_max <= 2^P - 1
        if acc_min >= 0:
            acc_bit_width = math.ceil(np.log2(acc_max + 1))
            adt = DataType[f"UINT{acc_bit_width}"]
        # if the acc_range is signed, then acc_min >= -2^{P-1} and acc_max <=
        # 2^{P - 1} - 1, which means 2^{P - 1} >= max(-acc_min, 1 + acc_max)
        else:
            acc_max_signed = max(-acc_min, 1 + acc_max)
            acc_bit_width = math.ceil(np.log2(acc_max_signed) + 1)
            adt = DataType[f"INT{acc_bit_width}"]

        # Note: Thresholds may not fit in the accumulator datatype at this point.
        # They will be clipped to the accumulator range by RoundAndClipThresholds transformation.

        # if no activation, output and accumulator datatypes are the same
        if self.no_activation:
            # if this is the last node in the graph, then ensure the datatype is
            # divisibly by 8 bits
            if model.find_direct_successors(self.onnx_node) is None:
                bw = roundup_to_integer_multiple(adt.bitwidth(), 8)
                new_adt_name = adt.name.replace(str(adt.bitwidth()), str(bw))
                adt = DataType[new_adt_name]
            # for no-activation nodes, output dt = acc dt
            self.set_nodeattr("outputDataType", adt.name)
        self.set_nodeattr("accDataType", adt.name)

        return DataType[cast("str", self.get_nodeattr("accDataType"))]

    def minimize_weight_bit_width(self, model: ModelWrapper) -> BaseDataType:
        """Minimize the bit width based on the values of the weights."""
        if not (self.runtime_writeable_weights or self.mem_mode == "external"):
            weights = model.get_initializer(self.onnx_node.input[1])
            if not isinstance(weights, np.ndarray):
                raise FINNInternalError(f"{self.onnx_node.name}: weight initializer is missing")
            w_min = weights.min()
            w_max = weights.max()
            if w_min < 0:
                if abs(w_min) > w_max:
                    wdt = DataType.get_smallest_possible(w_min)
                else:
                    wdt = DataType.get_smallest_possible(-w_max - 1)
            else:
                wdt = DataType.get_smallest_possible(w_max)
            self.set_nodeattr("weightDataType", wdt.name)

        return DataType[cast("str", self.get_nodeattr("weightDataType"))]

    def get_hw_compatible_threshold_tensor(self, orig_thres_matrix: np.ndarray) -> np.ndarray:
        """Convert the original numpy threshold matrix into hlslib-compatible form.

        The steps performed are:
        * ensure MH % PE == 0
        * for bipolar weights&inputs, ensure thresholds are positive
        * interleave rows between PEs
        * reshape into (PE, TMEM, n_thres_steps) and return
        """
        ch = self.channels
        pe = self.pe
        tmem = self.calc_tmem()
        if ch % pe != 0:
            raise FINNInternalError("Requirement Channels divisable by PE is violated.")
        if orig_thres_matrix.ndim != 2:
            raise FINNInternalError("Threshold matrix dimension is not as expected (2).")
        n_thres_steps = orig_thres_matrix.shape[1]
        inp_is_bipolar = self.get_input_datatype(0) == DataType["BIPOLAR"]
        wt_is_bipolar = self.get_input_datatype(1) == DataType["BIPOLAR"]
        # reinterpret inp/wt as bipolar if bin_xnor_mode is set
        inp_is_binary = self.get_input_datatype(0) == DataType["BINARY"]
        wt_is_binary = self.get_input_datatype(1) == DataType["BINARY"]
        bin_xnor_mode = self.binary_xnor_mode == 1
        inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
        wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)
        if inp_is_bipolar and wt_is_bipolar:
            # ensure all thresholds are nonnegative
            if not (orig_thres_matrix >= 0).all():
                raise FINNInternalError("Bipolar thresholds must be nonnegative")
            # ensure all thresholds are integer
            if not (orig_thres_matrix.astype(np.int32) == orig_thres_matrix).all():
                raise FINNInternalError("Bipolar thresholds must be integer")
        ret = orig_thres_matrix
        # ensure channels = mh , duplicating if necessary
        if ret.shape[0] == 1:
            ret = np.tile(ret, (ch, 1))
        if ret.shape[0] != ch:
            raise FINNInternalError("Channels of threshold matrix are not as expected (ch)")
        # distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        if ret.shape[0] != pe:
            raise FINNInternalError(
                "First dimension after distribution of the rows between PEs "
                "is not as expected (pe)"
            )
        if ret.shape[1] != tmem:
            raise FINNInternalError(
                "Second dimension after distribution of the rows between PEs "
                "is not as expected (tmem)"
            )
        if ret.shape[2] != n_thres_steps:
            raise FINNInternalError(
                "Third dimension after distribution of the rows between PEs "
                "is not as expected (n_thres_steps)"
            )
        return ret.reshape(1, pe, tmem, n_thres_steps)

    def get_hw_compatible_weight_tensor(self, orig_weight_matrix: np.ndarray) -> np.ndarray:
        """Convert weight matrix to hardware-compatible format."""
        pe = self.pe
        simd = self.simd
        ch = self.channels
        k_h, k_w = self.kernel
        wmem = self.calc_wmem()
        if orig_weight_matrix.shape != (ch, 1, k_h, k_w):
            raise FINNInternalError(
                "Weights matrix doesn't have expected shape "
                "(channels, 1, kernel_size, kernel_size)"
            )
        ret = orig_weight_matrix
        if self.get_input_datatype(1) == DataType["BIPOLAR"]:
            # convert bipolar to binary
            ret = (ret + 1) / 2
        ret = ret.reshape(ch, k_h * k_w)
        # distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        return ret.reshape(1, pe, wmem, simd)

    def make_weight_file(
        self, weights: np.ndarray, weight_file_mode: str, weight_file_name: str
    ) -> None:
        """Produce a file containing the given weights in the appropriate format.

        This file can be used for either synthesis or run-time reconfig of
        weights. ``weight_file_mode`` is one of ``hls_header``,
        ``decoupled_npy``, ``decoupled_verilog_dat`` or ``decoupled_runtime``.
        """
        # convert weights into hlslib-compatible format
        weight_tensor = self.get_hw_compatible_weight_tensor(weights)
        export_wdt = self.get_input_datatype(1)
        # we have converted bipolar weights to binary for export,
        # so use it as such for weight generation
        if self.get_input_datatype(1) == DataType["BIPOLAR"]:
            export_wdt = DataType["BINARY"]
        if weight_file_mode == "hls_header":
            weight_hls_code = numpy_to_hls_code(weight_tensor, export_wdt, "weights", True, True)
            # write weights into C++ header file as dictated by finn-hlslib
            if export_wdt.bitwidth() != 1:
                header = (
                    f"const FixedPointWeights<{self.simd},"
                    f"{export_wdt.get_hls_datatype_str()},{self.pe},{self.calc_wmem()}> weights = "
                )
            else:
                header = f"const BinaryWeights<{self.simd},{self.pe},{self.calc_wmem()}> weights = "
            Path(weight_file_name).write_text(header + weight_hls_code)
        elif "decoupled" in weight_file_mode:
            # create a weight stream for various flavors of internal_decoupled mode:
            # transpose weight tensor from (1, PE, WMEM, SIMD) to (1, WMEM, PE, SIMD)
            weight_tensor_unflipped = np.transpose(weight_tensor, (0, 2, 1, 3))
            # reverse SIMD flip for saving weights in .npy
            weight_tensor_simd_flipped = np.flip(weight_tensor_unflipped, axis=-1)
            # PE flip for saving weights in .dat
            weight_tensor_pe_flipped = np.flip(weight_tensor_unflipped, axis=-2)
            # SIMD & PE flip
            weight_tensor_pe_simd_flipped = np.flip(weight_tensor_pe_flipped, axis=-1)
            # reshape weight tensor (simd_flipped and pe_flipped) to desired shape
            pe = self.pe
            simd = self.simd
            # simd_flipped
            weight_tensor_simd_flipped = weight_tensor_simd_flipped.reshape(1, -1, pe * simd)
            weight_tensor_simd_flipped = weight_tensor_simd_flipped.copy()
            # flipped
            weight_tensor_pe_flipped = weight_tensor_pe_flipped.reshape(1, -1, pe * simd)
            weight_tensor_pe_flipped = weight_tensor_pe_flipped.copy()
            # SIMD & PE flipped
            weight_tensor_pe_simd_flipped = weight_tensor_pe_simd_flipped.reshape(1, -1, pe * simd)
            weight_tensor_pe_simd_flipped = weight_tensor_pe_simd_flipped.copy()
            if weight_file_mode == "decoupled_npy":
                # save weight stream into npy for cppsim
                if self.onnx_node.op_type == "VVAU_rtl":
                    weight_tensor_unflipped = weight_tensor_unflipped.reshape(1, -1, pe * simd)
                    weight_tensor_unflipped = weight_tensor_unflipped.copy()
                    np.save(weight_file_name, weight_tensor_unflipped)
                else:
                    np.save(weight_file_name, weight_tensor_simd_flipped)
            elif weight_file_mode == "decoupled_verilog_dat":
                # convert weight values into hexstring
                weight_width = self.get_instream_width(1)
                # pad to nearest 4 bits to get hex strings
                weight_width_padded = roundup_to_integer_multiple(weight_width, 4)
                if self.onnx_node.op_type == "VVAU_rtl":
                    weight_arr = pack_innermost_dim_as_hex_string(
                        weight_tensor_pe_simd_flipped, export_wdt, weight_width_padded, prefix=""
                    )
                else:
                    weight_arr = pack_innermost_dim_as_hex_string(
                        weight_tensor_pe_flipped, export_wdt, weight_width_padded, prefix=""
                    )
                # add zeroes to pad out file to 1024 entries
                weight_stream = weight_arr.flatten()
                weight_stream = weight_stream.copy()
                Path(weight_file_name).write_text("".join(f"{val}\n" for val in weight_stream))
            elif weight_file_mode == "decoupled_runtime":
                # memstream axi-lite interface will map each mem line to
                # one or multiple 32-bit words
                weight_width = self.get_instream_width(1)
                words_per_memwidth = 2 ** math.ceil(math.log2(weight_width / 32))
                words_per_memwidth = max(words_per_memwidth, 1)
                weight_width_padded = words_per_memwidth * 32
                # first, pack and ensure padding to 32 bits
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    weight_tensor_pe_flipped, export_wdt, weight_width_padded, prefix=""
                )
                weight_stream = weight_tensor_pe_flipped.flatten()
                weight_stream = weight_stream.copy()
                lines: list[str] = []
                for val in weight_stream:
                    # split into groups of 8 hex digits (= 32 bits)
                    words_32b = textwrap.wrap(val, 8)
                    words_32b.reverse()
                    lines.extend(f"{word_32b}\n" for word_32b in words_32b)
                Path(weight_file_name).write_text("".join(lines))
            else:
                raise FINNInternalError("Unknown weight_file_mode")

        else:
            raise FINNInternalError("Unknown weight_file_mode")

    def generate_params(self, model: ModelWrapper, path: str | Path) -> None:
        """Generate parameter files for hardware implementation."""
        mem_mode = self.mem_mode
        code_gen_dir = Path(path)
        # weights, if not external
        weights = model.get_initializer(self.onnx_node.input[1])
        if not isinstance(weights, np.ndarray):
            raise FINNInternalError(f"{self.onnx_node.name}: weight initializer is missing")
        if mem_mode == "internal_embedded":
            # save hlslib-compatible weights in params.h
            self.make_weight_file(weights, "hls_header", str(code_gen_dir / "params.h"))
        elif mem_mode in ("internal_decoupled", "external"):
            # save internal_decoupled weights for cppsim
            self.make_weight_file(weights, "decoupled_npy", str(code_gen_dir / "weights.npy"))
            if mem_mode == "internal_decoupled":
                # also save weights as Verilog .dat file
                # This file will be ignored when synthesizing UltraScale memory.
                self.make_weight_file(
                    weights, "decoupled_verilog_dat", str(code_gen_dir / "memblock.dat")
                )
        else:
            raise FINNInternalError(
                'Please set mem_mode to "internal_embedded", "internal_decoupled", or "external", '
                "currently no other parameter value is supported!"
            )

        # save thresholds in thresh.h
        if len(self.onnx_node.input) > 2:
            thresholds = model.get_initializer(self.onnx_node.input[2])
            if isinstance(thresholds, np.ndarray):
                threshold_tensor = self.get_hw_compatible_threshold_tensor(thresholds)
                # get computed threshold datatype from tensor
                tdt = model.get_tensor_datatype(self.onnx_node.input[2])

                if not np.vectorize(tdt.allowed)(threshold_tensor).all():
                    raise FINNUserError(
                        f"Thresholds in {self.onnx_node.name} can't be expressed with type {tdt!s}"
                    )
                thresholds_hls_code = numpy_to_hls_code(
                    threshold_tensor, tdt, "thresholds", False, True
                )
                # write thresholds into thresh.h
                tdt_hls = tdt.get_hls_datatype_str()
                # use binary to export bipolar activations
                export_odt = self.get_output_datatype()
                if self.get_output_datatype() == DataType["BIPOLAR"]:
                    export_odt = DataType["BINARY"]
                odt_hls = export_odt.get_hls_datatype_str()
                thresh_header = (
                    f"static ThresholdsActivation<{self.calc_tmem()},{self.pe},"
                    f"{threshold_tensor.shape[-1]},{tdt_hls},{odt_hls},{self.act_val},"
                    f"comp::less_equal<{tdt_hls}, {tdt_hls}>> threshs " + " " * 20 + "= "
                )
                (code_gen_dir / "thresh.h").write_text(thresh_header + thresholds_hls_code)

    def get_op_and_param_counts(self) -> dict[str, int]:
        """Get operation and parameter counts for this layer."""
        k_h, k_w = self.kernel
        fm = self.channels
        dim_h, dim_w = self.dim
        weight_bits = self.get_input_datatype(1).bitwidth()
        inp_bits = self.get_input_datatype(0).bitwidth()
        num_repetitions = int(dim_h * dim_w)
        mac_count = k_h * k_w * fm * num_repetitions
        # cannonicalize op type: highest bitwidth operand first s.t.
        # e.g. mac_8bx4b and mac_4bx8b don't appear as two different op types
        bw1 = min(inp_bits, weight_bits)
        bw2 = max(inp_bits, weight_bits)
        mac_op_type = f"op_mac_{bw1}bx{bw2}b"
        weight_param_type = f"param_weight_{weight_bits}b"
        weight_count = k_h * k_w * fm
        ret_dict = {mac_op_type: mac_count, weight_param_type: weight_count}
        if self.no_activation == 0:
            tdt = DataType[cast("str", self.get_nodeattr("accDataType"))]
            thres_bits = tdt.bitwidth()
            thres_param_type = f"param_threshold_{thres_bits}b"
            ret_dict[thres_param_type] = fm
        return ret_dict

    def get_verilog_top_module_intf_names(self) -> dict[str, list[tuple[str, int]] | list[str]]:
        """Get Verilog top module interface names."""
        intf_names = super().get_verilog_top_module_intf_names()
        mem_mode = self.mem_mode
        if mem_mode == "external":
            cast("list[tuple[str, int]]", intf_names["s_axis"]).append(
                ("in1_V", self.get_instream_width_padded(1))
            )
        # only expose axilite interface if the runtime-writeable attribute is set
        if mem_mode == "internal_decoupled" and self.runtime_writeable_weights == 1:
            intf_names["axilite"] = ["s_axilite"]
        return intf_names

    def code_generation_ipi(self) -> list[str]:
        """Generate IP integrator (IPI) commands for hardware synthesis."""
        source_target = f"./ip/verilog/rtl_ops/{self.onnx_node.name}"
        cmd = [f"file mkdir {source_target}"]
        # add streamer if needed
        mem_mode = self.mem_mode
        if mem_mode == "internal_decoupled":
            runtime_writeable = self.runtime_writeable_weights
            node_name = self.onnx_node.name
            # create a hierarchy for this layer, with the same port names
            intf_names = self.get_verilog_top_module_intf_names()
            clk_name = intf_names["clk"][0]
            rst_name = intf_names["rst"][0]
            dout_name = cast("list[tuple[str, int]]", intf_names["m_axis"])[0][0]
            din_name = cast("list[tuple[str, int]]", intf_names["s_axis"])[0][0]
            cmd.append(f"create_bd_cell -type hier {node_name}")
            cmd.append(f"create_bd_pin -dir I -type clk /{node_name}/{clk_name}")
            cmd.append(f"create_bd_pin -dir I -type rst /{node_name}/{rst_name}")
            cmd.append(
                "create_bd_intf_pin -mode Master "
                f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/{dout_name}"
            )
            cmd.append(
                "create_bd_intf_pin -mode Slave "
                f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/{din_name}"
            )
            # Instantiate either the HLS or RTL IP depending on operator
            self.instantiate_ip(cmd)

            # Instantiate a streamer and connect it to the IP
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
            axi_dir = str(Path(get_settings().finn_rtllib) / "axi/hdl") + "/"
            ms_rtllib_dir = str(Path(get_settings().finn_rtllib) / "memstream/hdl") + "/"
            file_suffix = "_memstream_wrapper.v"
            # automatically find memstream verilog component in code generation directory
            strm_tmpl = next(
                (f.name for f in Path(code_gen_dir).iterdir() if f.name.endswith(file_suffix)),
                None,
            )
            if strm_tmpl is None:
                raise FINNInternalError(
                    f"{node_name}: could not find a *{file_suffix} file in {code_gen_dir}"
                )
            strm_tmpl_name = strm_tmpl[:-2]
            sourcefiles = [
                str(Path(code_gen_dir) / strm_tmpl),
                axi_dir + "axilite.sv",
                ms_rtllib_dir + "memstream_axi.sv",
                ms_rtllib_dir + "memstream.sv",
            ]
            for f in sourcefiles:
                cmd += [f"add_files -copy_to {source_target} -norecurse {f}"]
            strm_inst = node_name + "_wstrm"
            cmd.append(
                f"create_bd_cell -type hier -reference {strm_tmpl_name} /{node_name}/{strm_inst}"
            )
            cmd.append(
                f"connect_bd_intf_net [get_bd_intf_pins {node_name}/{strm_inst}/m_axis_0] "
                f"[get_bd_intf_pins {node_name}/{node_name}/in1_V]"
            )
            cmd.append(
                f"connect_bd_net [get_bd_pins {node_name}/{rst_name}] "
                f"[get_bd_pins {node_name}/{strm_inst}/ap_rst_n]"
            )
            cmd.append(
                f"connect_bd_net [get_bd_pins {node_name}/{clk_name}] "
                f"[get_bd_pins {node_name}/{strm_inst}/ap_clk]"
            )
            # 2x clock is not used for decoupled VVAU weights
            # simply connect input to the 1x clock for now
            cmd.append(
                f"connect_bd_net [get_bd_pins {node_name}/{clk_name}] "
                f"[get_bd_pins {node_name}/{strm_inst}/ap_clk2x]"
            )
            cmd.append(
                f"connect_bd_net [get_bd_pins {node_name}/{rst_name}] "
                f"[get_bd_pins {node_name}/{node_name}/{rst_name}]"
            )
            cmd.append(
                f"connect_bd_net [get_bd_pins {node_name}/{clk_name}] "
                f"[get_bd_pins {node_name}/{node_name}/{clk_name}]"
            )
            cmd.append(
                f"connect_bd_intf_net [get_bd_intf_pins {node_name}/{din_name}] "
                f"[get_bd_intf_pins {node_name}/{node_name}/{din_name}]"
            )
            cmd.append(
                f"connect_bd_intf_net [get_bd_intf_pins {node_name}/{dout_name}] "
                f"[get_bd_intf_pins {node_name}/{node_name}/{dout_name}]"
            )
            if runtime_writeable:
                # expose axi lite interface for writeable weights
                axilite_name = self.get_verilog_top_module_intf_names()["axilite"][0]
                cmd.append(
                    "create_bd_intf_pin -mode Slave "
                    f"-vlnv xilinx.com:interface:aximm_rtl:1.0 /{node_name}/{axilite_name}"
                )
                cmd.append(
                    f"connect_bd_intf_net [get_bd_intf_pins {node_name}/{axilite_name}] "
                    f"[get_bd_intf_pins {node_name}/{strm_inst}/{axilite_name}]"
                )
                # TODO calculate and pass in segment size here
                cmd.append("assign_bd_address")
            cmd.append("save_bd_design")
        elif mem_mode in ("internal_embedded", "external"):
            # base class impl sufficient for internal_embedded/external modes
            self.instantiate_ip(cmd)
        else:
            raise FINNInternalError("Unrecognized mem_mode for VectorVectorActivation")
        return cmd

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

"""Matrix-Vector-Activation Unit (MVAU) hardware implementation.

This module implements the MVAU operation for FPGA deployment, which performs
matrix-vector multiplication optionally followed by activation/thresholding.
Supports various memory modes, parallelization strategies, and quantized datatypes.
"""
import math
import numpy as np
import qonnx.custom_op.general.xnorpopcount as xp
import textwrap
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

from finn.custom_op.fpgadataflow import register_custom_op
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.data_packing import numpy_to_hls_code, pack_innermost_dim_as_hex_string
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from onnx import GraphProto, NodeProto

# Type of the dictionary returned by get_nodeattr_types: maps attribute names to
# their (dtype, required, default[, allowed_values]) specification tuples
NodeAttrTypes = dict[
    str,
    tuple[str, bool, int | float | str | bool | np.ndarray | list]
    | tuple[str, bool, int | float | str | bool | np.ndarray | list, set | None],
]

# ONNX i/o tensor shape assumptions for MatrixVectorActivation:
# input 0 is the input tensor, shape (.., i_size) = (..., MW)
# input 1 is the weight tensor, shape (i_size, o_size) = (MW, MH)
# (optional) input 2 is the thresholds tensor, shape (o_size, n_thres)
# output 0 is the output tensor, shape (.., o_size) = (..., MH)
# the ... here can be any shape (representing groups of vectors)


@register_custom_op
class MVAU(HWCustomOp):
    """Abstraction layer for HW implementation of MatrixVectorActivation layers."""

    def __init__(self, onnx_node: "NodeProto", **kwargs: int) -> None:
        """Initialize the MVAU custom operation.

        Parameters
        ----------
        onnx_node : NodeProto
            ONNX node to wrap
        **kwargs : dict
            Additional arguments passed to parent class
        """
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Get dictionary of attribute names and their types for this node.

        Returns
        -------
        dict
            Dictionary mapping attribute names to type specifications
        """
        my_attrs: NodeAttrTypes = {
            "PE": ("i", True, 0),
            "SIMD": ("i", True, 0),
            "MW": ("i", True, 0),
            "MH": ("i", True, 0),
            "resType": ("s", False, "auto", {"auto", "lut", "dsp"}),
            "ActVal": ("i", False, 0),
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # FINN DataType for accumulator -- auto-computed and updated
            "accDataType": ("s", False, "INT32"),
            # use xnor-popcount for binary weights/inputs, thus treating them
            # as bipolar
            "binaryXnorMode": ("i", False, 0, {0, 1}),
            # no-activation mode (produce accumulators)
            "noActivation": ("i", False, 0, {0, 1}),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
            # memory mode for the FC weights
            # internal_embedded -- embedded weights, long compile/synth times
            # internal_decoupled -- default, streaming weights with streamer packaged inside IP
            # external -- streaming weights with external streamer
            "mem_mode": (
                "s",
                False,
                "internal_decoupled",
                {"internal_embedded", "internal_decoupled", "external"},
            ),
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
            # FPGA resource type for threshold memories (if noActivation is False)
            # auto -- let Vivado decide
            # block -- use BRAM
            # distributed -- use LUTRAM
            "ram_style_thresholds": (
                "s",
                False,
                "auto",
                {"auto", "block", "distributed"},
            ),
            # (mem_mode = internal_decoupled only) whether weights will be
            # writeable through an AXI-lite interface during runtime
            # 1 for enabled, 0 for disabled.
            # see finn-rtllib/memstream/doc/README for more about the memory
            # address map used for writable weights
            # IMPORTANT: After using AXI lite to either read or write the weights,
            # always "flush" the accelerator by first passing a dummy input
            # vector through the accelerator. This will get rid of any old
            # weight data from the weight FIFOs.
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
            "pumpedMemory": ("i", False, 0, {0, 1}),
            # dynamic input
            "dynamic_input": ("i", False, 0, {0, 1}),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    @property
    def pe(self) -> int:
        """Get the PE (output-channel) parallelism."""
        return cast("int", self.get_nodeattr("PE"))

    @property
    def simd(self) -> int:
        """Get the SIMD (input-channel) parallelism."""
        return cast("int", self.get_nodeattr("SIMD"))

    @property
    def mw(self) -> int:
        """Get the matrix width (number of input features)."""
        return cast("int", self.get_nodeattr("MW"))

    @property
    def mh(self) -> int:
        """Get the matrix height (number of output features)."""
        return cast("int", self.get_nodeattr("MH"))

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
    def ram_style_thresholds(self) -> str:
        """Get the FPGA resource type for the threshold memory."""
        return cast("str", self.get_nodeattr("ram_style_thresholds"))

    @property
    def binary_xnor_mode(self) -> int:
        """Get whether xnor-popcount (bipolar) mode is enabled (0/1)."""
        return cast("int", self.get_nodeattr("binaryXnorMode"))

    @property
    def no_activation(self) -> int:
        """Get whether the node runs without thresholding (0/1)."""
        return cast("int", self.get_nodeattr("noActivation"))

    @property
    def runtime_writeable_weights(self) -> int:
        """Get whether weights are writeable at runtime via AXI-lite (0/1)."""
        return cast("int", self.get_nodeattr("runtime_writeable_weights"))

    @property
    def pumped_memory(self) -> int:
        """Get whether the weight memory runs at 2x clock (0/1)."""
        return cast("int", self.get_nodeattr("pumpedMemory"))

    @property
    def dynamic_input(self) -> int:
        """Get whether the weights arrive as a dynamic streaming input (0/1)."""
        return cast("int", self.get_nodeattr("dynamic_input"))

    @property
    def mlo_max_iter(self) -> int:
        """Get the multi-layer-offload maximum iteration count (0 if disabled)."""
        return cast("int", self.get_nodeattr("mlo_max_iter"))

    @property
    def num_input_vectors(self) -> list[int]:
        """Get the number of input vectors along the non-feature axes."""
        return list(cast("list[int]", self.get_nodeattr("numInputVectors")))

    def instantiate_ip(self, cmd: list[str]) -> None:
        """Append the backend-specific IP instantiation TCL to ``cmd``."""
        raise NotImplementedError

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute this MVAU node.

        Performs matrix-vector multiplication and optional activation/thresholding.

        Parameters
        ----------
        context : dict
            Dictionary mapping tensor names to numpy arrays
        graph : GraphProto
            ONNX graph containing this node
        """
        node = self.onnx_node
        in_act = context[node.input[0]]
        # ensure that shape is compatible
        in_act = in_act.reshape(self.get_normal_input_shape())
        mvau_w = context[node.input[1]]
        # Matrix multiplication
        if self.binary_xnor_mode:
            # Note: activation/weights are expected to be binary
            # (by design coming from the transformation inferring this operation mode)
            result = xp.xnorpopcountmatmul(in_act, mvau_w)
        elif (
            self.get_nodeattr("inputDataType") == "BIPOLAR"
            and self.get_nodeattr("weightDataType") == "BIPOLAR"
        ):
            # Convert to binary and use xnorpopcountmatmul function
            result = xp.xnorpopcountmatmul((in_act + 1) / 2, (mvau_w + 1) / 2)
        else:
            # Regular matrix multiplication
            result = np.matmul(in_act, mvau_w)
        if self.no_activation == 0:
            mvau_thr = context[node.input[2]]
            odt_is_bipolar = self.get_nodeattr("outputDataType") == "BIPOLAR"
            out_scale = 2 if odt_is_bipolar else 1
            out_bias = -1 if odt_is_bipolar else self.act_val
            if result.ndim == 4:
                # NHWC to NCHW for multithreshold node
                result = result.transpose((0, 3, 1, 2))
            result = multithreshold(result, mvau_thr, out_scale, out_bias)
            if result.ndim == 4:
                # NCHW to NHWC
                result = result.transpose((0, 2, 3, 1))
        oshape = context[node.output[0]].shape
        context[node.output[0]] = result.reshape(oshape)

    def verify_node(self) -> list[str]:
        """Verify that this node has valid attributes and configuration.

        Returns
        -------
        list of str
            List of verification messages/warnings
        """
        info_messages = []
        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify that all necessary attributes exist
        # TODO collect automatically from get_nodeattr_types
        try:
            self.get_nodeattr("code_gen_dir_cppsim")
            self.get_nodeattr("executable_path")
            self.get_nodeattr("resType")
            self.get_nodeattr("MW")
            self.get_nodeattr("MH")
            self.get_nodeattr("SIMD")
            self.get_nodeattr("PE")
            self.get_nodeattr("inputDataType")
            self.get_nodeattr("weightDataType")
            self.get_nodeattr("outputDataType")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append("""The required MatrixVectorActivation attributes do not exist.""")

        # verify the number of inputs depending on noActivation value
        # check noActivation value to determine the number of inputs
        no_act = self.no_activation

        if no_act == 1:
            if len(self.onnx_node.input) == 2:
                info_messages.append("The number of inputs is correct")
            else:
                info_messages.append(
                    """MatrixVectorActivation needs in no
                            activation mode 2 inputs (data input and weights)"""
                )
        elif no_act == 0:
            if len(self.onnx_node.input) == 3:
                info_messages.append("The number of inputs is correct")
            else:
                info_messages.append(
                    """MatrixVectorActivation needs 3 inputs
                            (data input and weights and threshold values)"""
                )
        else:
            info_messages.append(
                f"""noActivation attribute contains {no_act} should
                be 0 or 1"""
            )
        return info_messages

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer and set output datatype based on input datatype and node attributes.

        Parameters
        ----------
        model : ModelWrapper
            FINN ModelWrapper containing this node
        """
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype(0):
            log.warning(
                f"inputDataType changing for {node.name}: "
                f"{self.get_input_datatype(0)!s} -> {idt!s}"
            )
        self.set_nodeattr("inputDataType", idt.name)
        # set output datatype from property
        model.set_tensor_datatype(node.output[0], self.get_output_datatype())

    def _require_mh_pe(self) -> None:
        """Raise if MH is not divisible by PE."""
        if self.mh % self.pe != 0:
            raise FINNUserError(
                f"{self.onnx_node.name}: MH ({self.mh}) must be divisible by PE ({self.pe})"
            )

    def _require_mw_simd(self) -> None:
        """Raise if MW is not divisible by SIMD."""
        if self.mw % self.simd != 0:
            raise FINNUserError(
                f"{self.onnx_node.name}: MW ({self.mw}) must be divisible by SIMD ({self.simd})"
            )

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:
        """Return FINN DataType of input."""
        # when performing FIFO insertion on an FC layer with ext weights, the ind
        # parameter can be > 0 (referring to the weights) so handle that here
        if ind == 0:
            return DataType[cast("str", self.get_nodeattr("inputDataType"))]
        if ind == 1:
            return DataType[cast("str", self.get_nodeattr("weightDataType"))]
        raise FINNInternalError(f"{self.onnx_node.name}: undefined input ind {ind} for MVAU")

    def get_accumulator_datatype(self) -> BaseDataType:
        """Return FINN DataType of accumulator."""
        return DataType[cast("str", self.get_nodeattr("accDataType"))]

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of output."""
        return DataType[cast("str", self.get_nodeattr("outputDataType"))]

    def get_instream_width(self, ind: int = 0) -> int:
        """Get width of input stream in bits.

        Parameters
        ----------
        ind : int
            Input stream index (0=activations, 1=weights, 2=thresholds)

        Returns
        -------
        int
            Bit width of the specified input stream
        """
        if ind == 0:
            i_bits = self.get_input_datatype(0).bitwidth()
            width = i_bits * self.simd
        elif ind == 1:
            if self.dynamic_input:
                width = (
                    self.get_folded_input_shape(ind)[-1] * self.get_input_datatype(ind).bitwidth()
                )
            elif self.mem_mode in ("internal_decoupled", "external") or self.mlo_max_iter:
                wp = self.get_input_datatype(1).bitwidth()
                width = self.pe * self.simd * wp
            else:
                width = 0
        elif ind == 2:
            # check if integrated thresholding and return 0
            # because threshold values are always embedded
            # or raise expection if there shouldn't be
            # a third input to the node
            act = not self.no_activation
            if act:
                width = 0
            else:
                raise FINNInternalError(f"{self.onnx_node.name}: input ind 2 out of range")
        else:
            raise FINNInternalError(f"{self.onnx_node.name}: input ind {ind} out of range")
        return width

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Get width of output stream in bits.

        Parameters
        ----------
        ind : int
            Output stream index

        Returns
        -------
        int
            Bit width of the output stream
        """
        return self.get_output_datatype().bitwidth() * self.pe

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Get shape of folded (parallelized) input tensor.

        Parameters
        ----------
        ind : int
            Input index (0=activations, 1=weights)

        Returns
        -------
        tuple of int
            Shape of folded input tensor
        """
        mw = self.mw
        mh = self.mh
        simd = self.simd
        pe = self.pe
        sf = mw // simd
        nf = mh // pe
        vecs = self.num_input_vectors

        if ind == 0:
            # calculate shape of input 0
            return (*vecs, sf, simd)
        if ind == 1:
            if self.dynamic_input:
                # calculate shape of input 1 (weights dynamic)
                return (*vecs[:2], mw, nf, pe)
            if self.mem_mode == "external" or self.mlo_max_iter:
                # calculate shape of input 1 (weights)
                return (*vecs, sf * nf, simd * pe)
            raise FINNInternalError(
                f"{self.onnx_node.name}: undefined folded input shape for input ind 1"
            )
        raise FINNInternalError(f"{self.onnx_node.name}: undefined input ind {ind} for MVAU")

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Get shape of folded (parallelized) output tensor.

        Parameters
        ----------
        ind : int
            Output index

        Returns
        -------
        tuple of int
            Shape of folded output tensor
        """
        nf = self.mh // self.pe
        return (*self.num_input_vectors, nf, self.pe)

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Get normal (non-folded) input shape.

        Parameters
        ----------
        ind : int
            Input index (0=activations, 1=weights)

        Returns
        -------
        tuple of int
            Normal input shape
        """
        if ind == 0:
            return (*self.num_input_vectors, self.mw)
        if ind == 1:
            return (self.mw, self.mh)
        raise FINNInternalError(f"{self.onnx_node.name}: undefined input shape for input ind {ind}")

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Get normal (non-folded) output shape.

        Parameters
        ----------
        ind : int
            Output index

        Returns
        -------
        tuple of int
            Normal output shape
        """
        return (*self.num_input_vectors, self.mh)

    def calc_wmem(self) -> int:
        """Calculate and return WMEM."""
        self._require_mh_pe()
        self._require_mw_simd()
        return self.mw * self.mh // (self.pe * self.simd)

    def calc_tmem(self) -> int:
        """Calculate and return TMEM."""
        if self.no_activation == 1:
            return 0
        return self.mh // self.pe

    def uram_estimation(self) -> int:
        """Estimate UltraRAM (URAM) resource usage.

        Returns
        -------
        int
            Estimated number of URAMs needed
        """
        p = self.pe
        q = self.simd
        w = self.get_input_datatype(1).bitwidth()
        d_in = self.mw
        d_out = self.mh
        omega = (d_in * d_out) / (q * p)
        mem_width = q * w * p
        mmode = self.mem_mode
        mstyle = self.ram_style
        if (
            (mmode == "internal_decoupled" and mstyle != "ultra")
            or (mmode == "internal_embedded" and self.calc_wmem() <= 128)
            or (mmode == "external")
            or self.mlo_max_iter
        ):
            return 0
        width_multiplier = math.ceil(mem_width / 72)
        depth_multiplier = math.ceil(omega / 4096)
        return width_multiplier * depth_multiplier

    def bram_estimation(self) -> int:
        """Calculate resource estimation for BRAM based on:
        - FINN-R: An End-to-End Deep-Learning Framework for Fast
        Exploration of Quantized Neural Networks
        - M. Blott, T. B. Preusser, N. J. Fraser, G. Gambardella, K. O'Brien,
        Y. Umuroglu, M. Leeser and K. Vissers
        - 12. Sep 2018.
        """
        # TODO add in/out FIFO contributions
        p = self.pe
        q = self.simd
        w = self.get_input_datatype(1).bitwidth()
        d_in = self.mw
        d_out = self.mh
        omega = (d_in * d_out) / (q * p)
        mem_width = q * w * p
        mmode = self.mem_mode
        mstyle = self.ram_style
        if (
            (mmode == "internal_decoupled" and mstyle in ["distributed", "ultra"])
            or (mmode == "internal_embedded" and self.calc_wmem() <= 128)
            or (mmode == "external")
            or self.mlo_max_iter
        ):
            return 0
        # assuming SDP mode RAMB18s (see UG573 Table 1-10)
        # assuming internal_decoupled (RTL) memory,
        # which is more efficient than internal_embedded (HLS)
        if mem_width == 1:
            return math.ceil(omega / 16384)
        if mem_width == 2:
            return math.ceil(omega / 8192)
        if mem_width <= 4:
            return (math.ceil(omega / 4096)) * (math.ceil(mem_width / 4))
        if mem_width <= 9:
            return (math.ceil(omega / 2048)) * (math.ceil(mem_width / 9))
        if mem_width <= 18 or omega > 512:
            return (math.ceil(omega / 1024)) * (math.ceil(mem_width / 18))
        return (math.ceil(omega / 512)) * (math.ceil(mem_width / 36))

    def bram_efficiency_estimation(self) -> float:
        """Estimate BRAM utilization efficiency.

        Returns
        -------
        float
            Efficiency ratio (actual bits used / total BRAM capacity allocated)
        """
        w = self.get_input_datatype(1).bitwidth()
        d_in = self.mw
        d_out = self.mh
        bram16_est = self.bram_estimation()
        if bram16_est == 0:
            return 1.0
        wbits = w * d_in * d_out
        bram16_est_capacity = bram16_est * 36 * 512
        return wbits / bram16_est_capacity

    def uram_efficiency_estimation(self) -> float:
        """Estimate URAM efficiency (parameter storage needed / allocated URAM storage)."""
        w = self.get_input_datatype(1).bitwidth()
        d_in = self.mw
        d_out = self.mh
        uram_est = self.uram_estimation()
        if uram_est == 0:
            return 1.0
        wbits = w * d_in * d_out
        uram_est_capacity = uram_est * 72 * 4096
        return wbits / uram_est_capacity

    def get_exp_cycles(self) -> int:
        """Get expected number of clock cycles for one inference.

        Returns
        -------
        int
            Number of clock cycles
        """
        pe = self.pe
        simd = self.simd
        num_inp_vec = self.num_input_vectors
        mh = self.mh
        mw = self.mw
        # since mmv != 1 is not supported yet, we set mmv for now to 1
        mmv = 1
        exp_cycles = (mh / pe) * (mw / simd) * np.prod(num_inp_vec) / mmv
        return int(exp_cycles)

    def minimize_accumulator_width(self, model: ModelWrapper) -> BaseDataType:
        """Minimize the accumulator bit width according to the weight values,
        input data types, and size of dot product.
        """
        idt = self.get_input_datatype(0)

        # if runtime-writeable weights or mem_mode=external, then the values of the weights can
        # change and we need to use the worst-case values from the datatypes
        if (
            self.runtime_writeable_weights
            or self.mem_mode == "external"
            or self.mlo_max_iter
            or self.dynamic_input
        ):
            mw = self.mw
            mh = self.mh
            wdt = self.get_input_datatype(1)
            lower_worst = wdt.min() * np.ones((mw, mh))
            lower_range = calculate_matvec_accumulator_range(lower_worst, idt)
            upper_worst = wdt.max() * np.ones((mw, mh))
            upper_range = calculate_matvec_accumulator_range(upper_worst, idt)
            acc_min = min(min(lower_range), min(upper_range))
            acc_max = max(max(lower_range), max(upper_range))
        else:
            weights = model.get_initializer(self.onnx_node.input[1])
            if not isinstance(weights, np.ndarray):
                raise FINNInternalError(
                    f"{self.onnx_node.name}: expected constant weights to "
                    f"minimize accumulator width"
                )
            # since in the calculation the values of the weight matrix are used,
            # for the bipolar case they need to be converted to bipolar
            if self.binary_xnor_mode:
                weights = 2 * weights - 1
            (acc_min, acc_max) = calculate_matvec_accumulator_range(weights, idt)

        # if the acc_range is always greater than 0, then acc_max <= 2^P - 1
        if acc_min >= 0:
            acc_bit_width = math.ceil(np.log2(acc_max + 1))
            adt = DataType[f"UINT{acc_bit_width}"]
        # if the acc_range is signed, then acc_min >= -2^{P-1} and acc_max <=
        # 2^{P - 1} - 1, which means 2^{P - 1} >= max(-acc_min, 1 + acc_max)
        else:
            _acc_max = max(-acc_min, 1 + acc_max)
            acc_bit_width = math.ceil(np.log2(_acc_max) + 1)
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
        if not (
            self.runtime_writeable_weights
            or self.mem_mode == "external"
            or self.mlo_max_iter
            or self.dynamic_input
        ):
            weights = model.get_initializer(self.onnx_node.input[1])
            if not isinstance(weights, np.ndarray):
                raise FINNInternalError(
                    f"{self.onnx_node.name}: expected constant weights to minimize weight bit width"
                )
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
        """Reshape a threshold matrix into the form expected by the hlslib call.

        Ensures ``MH % PE == 0``, forces positive integer thresholds for
        bipolar weights&inputs, interleaves rows between PEs, and reshapes into
        ``(1, PE, TMEM, n_thres_steps)``.
        """
        mh = self.mh
        pe = self.pe
        tmem = mh // pe
        self._require_mh_pe()
        if orig_thres_matrix.ndim != 2:
            raise FINNInternalError(
                f"{self.onnx_node.name}: threshold matrix must be 2D, got {orig_thres_matrix.ndim}D"
            )
        n_thres_steps = orig_thres_matrix.shape[1]
        inp_is_bipolar = self.get_input_datatype(0) == DataType["BIPOLAR"]
        wt_is_bipolar = self.get_input_datatype(1) == DataType["BIPOLAR"]
        # reinterpret inp/wt as bipolar if bin_xnor_mode is iset
        inp_is_binary = self.get_input_datatype(0) == DataType["BINARY"]
        wt_is_binary = self.get_input_datatype(1) == DataType["BINARY"]
        bin_xnor_mode = self.binary_xnor_mode == 1
        inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
        wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)
        if inp_is_bipolar and wt_is_bipolar:
            # ensure all thresholds are nonnegative
            if not (orig_thres_matrix >= 0).all():
                raise FINNUserError(
                    f"{self.onnx_node.name}: bipolar thresholds must all be nonnegative"
                )
            # ensure all thresholds are integer
            if not (orig_thres_matrix.astype(np.int32) == orig_thres_matrix).all():
                raise FINNUserError(
                    f"{self.onnx_node.name}: bipolar thresholds must all be integer-valued"
                )
        ret = orig_thres_matrix
        # ensure channels = mh , duplicating if necessary
        if ret.shape[0] == 1:
            ret = np.tile(ret, (mh, 1))
        if ret.shape[0] != mh:
            raise FINNInternalError(
                f"{self.onnx_node.name}: threshold matrix has {ret.shape[0]} channels, "
                f"expected {mh}"
            )
        # distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        if ret.shape[0] != pe or ret.shape[1] != tmem or ret.shape[2] != n_thres_steps:
            raise FINNInternalError(
                f"{self.onnx_node.name}: threshold matrix shape after PE distribution is "
                f"{ret.shape}, expected ({pe}, {tmem}, {n_thres_steps})"
            )
        return ret.reshape(1, pe, tmem, n_thres_steps)

    def get_hw_compatible_weight_tensor(self, orig_weight_matrix: np.ndarray) -> np.ndarray:
        """Reshape a weight matrix into the form expected by the hlslib call.

        Ensures ``MH % PE == 0`` and ``MW % SIMD == 0``, converts bipolar
        ``{-1, +1}`` weights to binary ``{0, 1}``, interleaves rows between PEs,
        and reshapes into ``(1, PE, WMEM, SIMD)``.
        """
        mw = self.mw
        mh = self.mh
        pe = self.pe
        simd = self.simd
        wmem = self.calc_wmem()
        if orig_weight_matrix.shape != (mw, mh):
            raise FINNInternalError(
                f"{self.onnx_node.name}: weight matrix shape {orig_weight_matrix.shape}, "
                f"expected ({mw}, {mh})"
            )
        self._require_mw_simd()
        self._require_mh_pe()
        # start by transposing the original weight matrix, since ONNX and
        # finn-hlslib use different assumptions
        # ONNX uses (in_features, out_features) and matmul(x, W)
        # finn-hlslib uses (out_features, in_features) and matmul(W, x)
        ret = orig_weight_matrix.T
        if self.get_input_datatype(1) == DataType["BIPOLAR"]:
            # convert bipolar to binary
            ret = (ret + 1) / 2
        # interleave rows between PEs and reshape
        # distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        # create SIMD as innermost dimension and add a dummy outer dim
        ret = ret.reshape(1, pe, wmem, simd)
        # reverse the SIMD dimension
        ret = np.flip(ret, axis=-1)
        return ret

    def make_weight_file(
        self, weights: np.ndarray, weight_file_mode: str, weight_file_name: str
    ) -> None:
        """Write ``weights`` to ``weight_file_name`` in the requested format.

        The file can be used for synthesis or run-time weight reconfiguration.
        ``weight_file_mode`` is one of ``hls_header``, ``decoupled_npy``,
        ``decoupled_verilog_dat`` or ``decoupled_runtime``.
        """
        # convert weights into hlslib/rtllib-compatible format
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
                    f"const FixedPointWeights<{self.simd},{export_wdt.get_hls_datatype_str()},"
                    f"{self.pe},{self.calc_wmem()}> weights = "
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
            # reshape weight tensor (simd_flipped and pe_flipped) to desired shape
            pe = self.pe
            simd = self.simd
            # simd_flipped
            weight_tensor_simd_flipped = weight_tensor_simd_flipped.reshape(1, -1, pe * simd)
            weight_tensor_simd_flipped = weight_tensor_simd_flipped.copy()
            # flipped
            weight_tensor_pe_flipped = weight_tensor_pe_flipped.reshape(1, -1, pe * simd)
            weight_tensor_pe_flipped = weight_tensor_pe_flipped.copy()
            if weight_file_mode == "decoupled_npy":
                # save weight stream into npy for cppsim
                np.save(weight_file_name, weight_tensor_simd_flipped)
            elif weight_file_mode == "decoupled_verilog_dat":
                # convert weight values into hexstring
                weight_width = self.get_instream_width(1)
                if self.dynamic_input:
                    weight_width = weight_width * simd
                # pad to nearest 4 bits to get hex strings
                weight_width_padded = roundup_to_integer_multiple(weight_width, 4)
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    weight_tensor_pe_flipped, export_wdt, weight_width_padded, prefix=""
                )
                # add zeroes to pad out file to 1024 entries
                weight_stream = weight_tensor_pe_flipped.flatten()
                weight_stream = weight_stream.copy()
                if self.pumped_memory:
                    # if pe = simd = 1, known bug, ask user to increase parallelism
                    if pe == simd == 1:
                        raise FINNUserError(
                            f"{self.onnx_node.name}: pumped memory with PE=SIMD=1 is not "
                            f"supported. Please increase parallelism."
                        )
                    split_w_stream = np.zeros([weight_stream.shape[0] * 2], dtype=object)
                    k = 0
                    for i in range(len(weight_stream)):
                        weight = weight_stream[i]
                        split_w_stream[k] = weight[len(weight) // 2 :]
                        split_w_stream[k + 1] = weight[: len(weight) // 2]
                        k += 2
                    weight_stream = split_w_stream
                with Path(weight_file_name).open("w") as f:
                    for val in weight_stream:
                        f.write(val + "\n")
            elif weight_file_mode == "decoupled_runtime":
                # memstream axi-lite interface will map each mem line to
                # one or multiple 32-bit words
                weight_width = self.get_instream_width(1)
                if self.dynamic_input:
                    weight_width = weight_width * simd
                words_per_memwidth = 2 ** math.ceil(math.log2(weight_width / 32))
                if words_per_memwidth < 1:
                    words_per_memwidth = 1
                weight_width_padded = words_per_memwidth * 32
                # first, pack and ensure padding to 32 bits
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    weight_tensor_pe_flipped, export_wdt, weight_width_padded, prefix=""
                )
                weight_stream = weight_tensor_pe_flipped.flatten()
                weight_stream = weight_stream.copy()
                with Path(weight_file_name).open("w") as f:
                    for val in weight_stream:
                        # split into groups of 8 hex digits (= 32 bits)
                        words_32b = textwrap.wrap(val, 8)
                        words_32b.reverse()
                        for word_32b in words_32b:
                            f.write(word_32b + "\n")
            else:
                raise FINNInternalError(
                    f"{self.onnx_node.name}: unknown weight_file_mode {weight_file_mode}"
                )

        else:
            raise FINNInternalError(
                f"{self.onnx_node.name}: unknown weight_file_mode {weight_file_mode}"
            )

    def generate_params(self, model: ModelWrapper, path: str | Path) -> None:
        """Generate parameter files (weights and thresholds) for hardware generation.

        Parameters
        ----------
        model : ModelWrapper
            FINN ModelWrapper containing this node
        path : str
            Output directory path for generated files
        """
        mem_mode = self.mem_mode
        code_gen_dir = Path(path)
        # weights, if not external
        weights = model.get_initializer(self.onnx_node.input[1])
        if isinstance(weights, np.ndarray):
            if mem_mode == "internal_embedded":
                # save hlslib-compatible weights in params.h
                self.make_weight_file(weights, "hls_header", str(code_gen_dir / "params.h"))
            elif mem_mode in ("internal_decoupled", "external"):
                # save internal_decoupled weights for cppsim
                self.make_weight_file(weights, "decoupled_npy", str(code_gen_dir / "input_1.npy"))
                if mem_mode == "internal_decoupled":
                    # also save weights as Verilog .dat file
                    # This file will be ignored when synthesizing UltraScale memory.
                    self.make_weight_file(
                        weights, "decoupled_verilog_dat", str(code_gen_dir / "memblock.dat")
                    )
        elif not (mem_mode == "external" or self.mlo_max_iter or self.dynamic_input):
            raise FINNInternalError(
                f"{self.onnx_node.name}: weight values not initialized, but neither "
                f'"external" case nor MLO'
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
                        f"{self.onnx_node.name}: thresholds cannot be expressed with type {tdt}"
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
        """Get dictionary of operations and parameter counts for this layer.

        Returns
        -------
        dict
            Dictionary with operation types and counts as key-value pairs
        """
        in_features = self.mw
        out_features = self.mh
        weight_bits = self.get_input_datatype(1).bitwidth()
        inp_bits = self.get_input_datatype(0).bitwidth()
        num_inp_vec = self.num_input_vectors
        num_repetitions = int(np.prod(num_inp_vec))
        mac_count = in_features * out_features * num_repetitions
        # cannonicalize op type: highest bitwidth operand first s.t.
        # e.g. mac_8bx4b and mac_4bx8b don't appear as two different op types
        bw1 = min(inp_bits, weight_bits)
        bw2 = max(inp_bits, weight_bits)
        mac_op_type = f"op_mac_{bw1}bx{bw2}b"
        weight_param_type = f"param_weight_{weight_bits}b"
        weight_count = in_features * out_features
        ret_dict = {mac_op_type: mac_count, weight_param_type: weight_count}
        if self.no_activation == 0:
            tdt = DataType[cast("str", self.get_nodeattr("accDataType"))]
            thres_bits = tdt.bitwidth()
            thres_param_type = f"param_threshold_{thres_bits}b"
            ret_dict[thres_param_type] = out_features
        return ret_dict

    def get_verilog_top_module_intf_names(self) -> dict[str, list[tuple[str, int]] | list[str]]:
        """Get Verilog top module interface names for this node.

        Returns
        -------
        dict
            Dictionary mapping interface types to port names
        """
        s_axis: list[tuple[str, int]] = [("in0_V", self.get_instream_width_padded(0))]
        m_axis: list[tuple[str, int]] = [("out0_V", self.get_outstream_width_padded(0))]
        aximm: list[tuple[str, int]] = []
        axilite: list[str] = []
        intf_names: dict[str, list[tuple[str, int]] | list[str]] = {
            "clk": ["ap_clk"],
            "rst": ["ap_rst_n"],
            "s_axis": s_axis,
            "m_axis": m_axis,
            "aximm": aximm,
            "axilite": axilite,
            "ap_none": [],
        }

        try:
            pumped_compute = self.get_nodeattr("pumpedCompute")
        except AttributeError:
            pumped_compute = 0

        if pumped_compute or self.pumped_memory:
            intf_names["clk2x"] = ["ap_clk2x"]

        if self.mlo_max_iter:
            aximm.append(("axi_mm", 64))
            s_axis.append(("in_idx0_V", 32))
        else:
            if self.dynamic_input:
                weight_width = self.get_instream_width(1) * self.simd
                s_axis.append(("in1_V", roundup_to_integer_multiple(weight_width, 8)))
            elif self.mem_mode == "external":
                s_axis.append(("in1_V", self.get_instream_width_padded(1)))
            elif self.mem_mode == "internal_decoupled":
                # only expose axilite interface if attribute is set
                if self.runtime_writeable_weights:
                    intf_names["axilite"] = ["s_axilite"]
        return intf_names

    def code_generation_ipi(self) -> list[str]:
        """Generate TCL commands for IP integrator (IPI) block design.

        Returns
        -------
        list of str
            List of TCL commands for Vivado IP integrator
        """
        source_target = f"./ip/verilog/rtl_ops/{self.onnx_node.name}"
        cmd = [f"file mkdir {source_target}"]
        dyn_input = self.dynamic_input
        mem_mode = self.mem_mode
        # check if additional components are needed
        if mem_mode == "internal_decoupled" or self.mlo_max_iter or dyn_input:
            runtime_writeable = self.runtime_writeable_weights
            node_name = self.onnx_node.name
            # create a hierarchy for this layer, with the same port names
            clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
            rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
            dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0][0]
            din_name = self.get_verilog_top_module_intf_names()["s_axis"][0][0]
            cmd.append(f"create_bd_cell -type hier {node_name}")
            # clock and reset
            cmd.append(f"create_bd_pin -dir I -type clk /{node_name}/{clk_name}")
            cmd.append(f"create_bd_pin -dir I -type rst /{node_name}/{rst_name}")
            # if we need a 2x clock for either compute or memory, instantiate the 2x clk port
            try:
                pumped_compute = self.get_nodeattr("pumpedCompute")
            except AttributeError:
                pumped_compute = 0
            if pumped_compute or self.pumped_memory:
                clk2x_name = self.get_verilog_top_module_intf_names()["clk2x"][0]
                cmd.append(f"create_bd_pin -dir I -type clk /{node_name}/{clk2x_name}")
            else:
                clk2x_name = None
            # streams
            cmd.append(
                "create_bd_intf_pin -mode Master "
                f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/{dout_name}"
            )
            cmd.append(
                "create_bd_intf_pin -mode Slave "
                f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/{din_name}"
            )

            if self.mlo_max_iter:
                cmd.append(
                    "create_bd_intf_pin -mode Slave "
                    f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/in_idx0_V"
                )
                cmd.append(
                    "create_bd_intf_pin -mode Master "
                    f"-vlnv xilinx.com:interface:aximm_rtl:1.0 /{node_name}/axi_mm"
                )

            # Instantiate either the HLS or RTL IP depending on operator
            self.instantiate_ip(cmd)
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
            rtllib = get_settings().finn_rtllib
            win_name = ""
            if dyn_input:
                # additional dynamic input
                win_name = self.get_verilog_top_module_intf_names()["s_axis"][1][0]
                cmd.append(
                    "create_bd_intf_pin -mode Slave "
                    f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/{win_name}"
                )
                # dynamic loader
                ram_rtllib_dir = f"{rtllib}/ram/"
                dyn_rtllib_dir = f"{rtllib}/dynload/hdl/"
                file_suffix = "_dynamic_load_wrapper.v"
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
                    ram_rtllib_dir + "ram_p_c.sv",
                    dyn_rtllib_dir + "dynamic_load.sv",
                ]
                for f in sourcefiles:
                    cmd += [f"add_files -copy_to {source_target} -norecurse {f}"]
                strm_inst = node_name + "_wdynld"
                strm_out_name = "m_axis_0"
            elif self.mlo_max_iter:
                # instantiate a fetch weights component and connect it to the IP
                mlo_rtllib_dir = f"{rtllib}/mlo/"
                reg_rtllib_dir = f"{rtllib}/skid/"
                ram_rtllib_dir = f"{rtllib}/ram/"
                dwc_rtllib_dir = f"{rtllib}/dwc/hdl/"
                dma_rtllib_dir = f"{rtllib}/cdma/"
                file_suffix = "_fetch_weights_wrapper.v"
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
                    reg_rtllib_dir + "skid.sv",
                    ram_rtllib_dir + "ram_p_c.sv",
                    dwc_rtllib_dir + "axis_adapter.v",
                    dwc_rtllib_dir + "axis_fifo_adapter.sv",
                    dwc_rtllib_dir + "axis_fifo.v",
                    mlo_rtllib_dir + "fetch_weights.sv",
                    mlo_rtllib_dir + "local_weight_buffer.sv",
                ]
                # add files from cdma dir
                for subdir in ("", "cdma_a/", "cdma_u/", "cdma_x/"):
                    cdma_path = Path(dma_rtllib_dir + subdir)
                    for file in sorted(p.name for p in cdma_path.iterdir()):
                        if file.endswith((".sv", ".svh")):
                            sourcefiles.append(str(cdma_path / file))

                for f in sourcefiles:
                    cmd += [f"add_files -copy_to {source_target} -norecurse {f}"]
                strm_inst = node_name + "_fetch_weights"
                strm_out_name = "out0_V"

            elif mem_mode == "internal_decoupled":
                # instantiate a streamer and connect it to the IP
                axi_dir = f"{rtllib}/axi/hdl/"
                ms_rtllib_dir = f"{rtllib}/memstream/hdl/"
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
                strm_out_name = "m_axis_0"
            else:
                raise FINNInternalError(
                    f"{node_name}: unreachable branch in code_generation_ipi weight streamer setup"
                )

            cmd.append(
                f"create_bd_cell -type hier -reference {strm_tmpl_name} /{node_name}/{strm_inst}"
            )

            if self.mlo_max_iter:
                cmd.append(
                    f"connect_bd_intf_net [get_bd_intf_pins {node_name}/in_idx0_V] "
                    f"[get_bd_intf_pins {node_name}/{strm_inst}/in_idx0_V]"
                )

                cmd.append(
                    f"connect_bd_intf_net [get_bd_intf_pins {node_name}/axi_mm] "
                    f"[get_bd_intf_pins {node_name}/{strm_inst}/axi_mm]"
                )

            if dyn_input:
                cmd.append(
                    f"connect_bd_intf_net [get_bd_intf_pins {node_name}/{win_name}] "
                    f"[get_bd_intf_pins {node_name}/{strm_inst}/s_axis_0]"
                )
            cmd.append(
                f"connect_bd_intf_net [get_bd_intf_pins {node_name}/{strm_inst}/{strm_out_name}] "
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
            # if using 2x pumped memory, connect the memstreamer's 2x clk input
            # to the 2x clock port. otherwise connect it to the regular clock port.
            if mem_mode == "internal_decoupled" and not (self.mlo_max_iter or dyn_input):
                if self.pumped_memory:
                    cmd.append(
                        f"connect_bd_net [get_bd_pins {node_name}/{clk2x_name}] "
                        f"[get_bd_pins {node_name}/{strm_inst}/ap_clk2x]"
                    )
                else:
                    cmd.append(
                        f"connect_bd_net [get_bd_pins {node_name}/{clk_name}] "
                        f"[get_bd_pins {node_name}/{strm_inst}/ap_clk2x]"
                    )
                # runtime writeable weights
                if runtime_writeable:
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

            # save bd
            cmd.append("save_bd_design")
        elif (mem_mode in ("internal_embedded", "external")) and not self.mlo_max_iter:
            # base class impl sufficient for internal_embedded/external modes
            self.instantiate_ip(cmd)
        else:
            raise FINNInternalError(
                f"{self.onnx_node.name}: unrecognized mem_mode for MatrixVectorActivation"
            )
        return cmd

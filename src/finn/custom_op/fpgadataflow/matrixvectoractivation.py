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

"""Matrix-Vector-Activation Unit (MVAU) hardware custom op base class.

This module implements the MVAU operation for FPGA deployment, which performs
matrix-vector multiplication optionally followed by activation/thresholding.
Supports various weight memory modes (embedded, decoupled, external streams,
external memory and dynamic weights), tiling, parallelization strategies and
quantized datatypes.
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
    get_by_name,
    interleave_matrix_outer_dim_from_partitions,
    roundup_to_integer_multiple,
)
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow import register_custom_op
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.basic import fifo_rtl_files, is_versal
from finn.util.data_packing import numpy_to_hls_code, pack_innermost_dim_as_hex_string
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from onnx import GraphProto, NodeProto

    from finn.transformation.fpgadataflow.loop_rolling import LoopBodyInputType

# Type of the dictionary returned by get_nodeattr_types: maps attribute names to
# their (dtype, required, default[, allowed_values]) specification tuples
NodeAttrTypes = dict[
    str,
    tuple[str, bool, int | float | str | bool | np.ndarray | list]
    | tuple[str, bool, int | float | str | bool | np.ndarray | list, set | None],
]

# AXI bus width (DATA_BITS in fetch_weights.sv) used for the external memory weight image
_FETCH_WEIGHTS_DATA_BITS = 256

# ONNX i/o tensor shape assumptions for MatrixVectorActivation:
# input 0 is the input tensor, shape (.., i_size) = (..., MW)
# input 1 is the weight tensor, shape (i_size, o_size) = (MW, MH)
# (optional) input 2 is the thresholds tensor, shape (o_size, n_thres)
# output 0 is the output tensor, shape (.., o_size) = (..., MH)
# the ... here can be any shape (representing groups of vectors)


def _find_codegen_file(code_gen_dir: str, file_suffix: str, node_name: str) -> str:
    """Find the generated file ending in ``file_suffix`` in ``code_gen_dir``.

    Args:
        code_gen_dir: Directory to search.
        file_suffix: Filename suffix to look for.
        node_name: Name of the node (for the error message).

    Returns:
        The name (without directory) of the matching file.
    """
    strm_tmpl = next(
        (f.name for f in Path(code_gen_dir).iterdir() if f.name.endswith(file_suffix)),
        None,
    )
    if strm_tmpl is None:
        raise FINNInternalError(
            f"{node_name}: could not find a *{file_suffix} file in {code_gen_dir}"
        )
    return strm_tmpl


@register_custom_op
class MVAU(HWCustomOp):
    """Abstraction layer for HW implementation of MatrixVectorActivation layers."""

    def __init__(self, onnx_node: "NodeProto", **kwargs: int) -> None:
        """Initialize the MVAU custom operation.

        Args:
            onnx_node: ONNX node to wrap.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return node attribute types for folding, datatypes, memory and weight infrastructure.

        Returns:
            Dictionary mapping attribute names to type specifications.
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
                {"internal_embedded", "internal_decoupled", "external", "external_mem", "dynamic"},
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
            # tiling height; only relevant for the RTL backend (MVAU_rtl), the
            # HLS backend supports the untiled case (TH=1) only.
            "TH": ("i", False, 1),
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
    def th(self) -> int:
        """Get the tiling height (TH > 1 only supported by the RTL backend)."""
        return cast("int", self.get_nodeattr("TH"))

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
        """Get the weight memory mode.

        One of internal_embedded, internal_decoupled, external, external_mem or dynamic.
        """
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
    def mlo_max_iter(self) -> int:
        """Get the multi-layer-offload maximum iteration count (0 if disabled)."""
        return cast("int", self.get_nodeattr("mlo_max_iter"))

    @property
    def num_input_vectors(self) -> list[int]:
        """Get the number of input vectors along the non-feature axes."""
        return list(cast("list[int]", self.get_nodeattr("numInputVectors")))

    def instantiate_ip(self, cmd: list[str]) -> None:
        """Append the backend-specific IP instantiation TCL to ``cmd``.

        Args:
            cmd: List of TCL commands to append to.
        """
        raise NotImplementedError

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute the node in Python via (xnorpopcount) matmul and optional multithresholding.

        Args:
            context: Dictionary mapping tensor names to numpy arrays.
            graph: ONNX graph containing this node.
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
        """Verify the node attributes, inputs and outputs.

        Returns:
            List of info messages.
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
        """Infer and set the input, weight and output datatypes from the model tensors.

        Args:
            model: FINN ModelWrapper containing this node.
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
        """Return FINN DataType of input ``ind`` (0: activations, 1: weights)."""
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
        """Return the width of input stream ``ind`` in bits.

        Args:
            ind: Input stream index (0=activations, 1=weights, 2=thresholds).

        Returns:
            Bit width of the specified input stream.
        """
        if ind == 0:
            i_bits = self.get_input_datatype(0).bitwidth()
            width = i_bits * self.simd
        elif ind == 1:
            wp = self.get_input_datatype(1).bitwidth()
            match self.mem_mode:
                case "dynamic":
                    width = self.pe * wp
                case "external" | "external_mem" | "internal_decoupled":
                    width = ((self.pe * self.simd) * wp) // self.th
                case _:
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
        """Return the width of the output stream in bits."""
        return self.get_output_datatype().bitwidth() * self.pe

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return the folded shape of input ``ind``.

        Args:
            ind: Input index (0=activations, 1=weights).

        Returns:
            Shape of the folded input tensor.
        """
        mw = self.mw
        mh = self.mh
        simd = self.simd
        pe = self.pe
        sf = mw // simd
        nf = mh // pe
        vecs = self.num_input_vectors
        n_vecs = int(np.prod(vecs))

        if ind == 0:
            # calculate shape of input 0
            return (*vecs, sf, simd)
        if ind == 1:
            match self.mem_mode:
                case "dynamic":
                    return (*vecs[:2], mw, nf, pe)
                case "external" | "external_mem" | "internal_decoupled":
                    return (n_vecs, sf * nf, (simd * pe) // self.th)
                case _:
                    raise FINNInternalError(
                        f"{self.onnx_node.name}: undefined folded input shape for input ind 1 "
                        f"with mem_mode {self.mem_mode}"
                    )
        raise FINNInternalError(f"{self.onnx_node.name}: undefined input ind {ind} for MVAU")

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the folded output shape."""
        nf = self.mh // self.pe
        return (*self.num_input_vectors, nf, self.pe)

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return the unfolded shape of input ``ind`` (0=activations, 1=weights)."""
        if ind == 0:
            return (*self.num_input_vectors, self.mw)
        if ind == 1:
            return (self.mw, self.mh)
        raise FINNInternalError(f"{self.onnx_node.name}: undefined input shape for input ind {ind}")

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the unfolded output shape."""
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

    def uram_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Estimate the number of URAMs used for weight storage."""
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
            or (mmode == "external_mem")
        ):
            return 0
        width_multiplier = math.ceil(mem_width / 72)
        depth_multiplier = math.ceil(omega / 4096)
        return width_multiplier * depth_multiplier

    def bram_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
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
            or (mmode == "external_mem")
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

    def bram_efficiency_estimation(self, fpgapart: str) -> float:
        """Return the fraction of estimated BRAM capacity actually used by weights."""
        w = self.get_input_datatype(1).bitwidth()
        d_in = self.mw
        d_out = self.mh
        bram16_est = self.bram_estimation(fpgapart)
        if bram16_est == 0:
            return 1.0
        wbits = w * d_in * d_out
        bram16_est_capacity = bram16_est * 36 * 512
        return wbits / bram16_est_capacity

    def uram_efficiency_estimation(self, fpgapart: str) -> float:
        """Estimate URAM efficiency (parameter storage needed / allocated URAM storage)."""
        # TODO: Versal URAM supports flexible bit widths (9/18/36/72) unlike
        # UltraScale+ which only supports 72-bit. This could improve efficiency
        # for narrow data types on Versal devices.
        w = self.get_input_datatype(1).bitwidth()
        d_in = self.mw
        d_out = self.mh
        uram_est = self.uram_estimation(fpgapart)
        if uram_est == 0:
            return 1.0
        wbits = w * d_in * d_out
        uram_est_capacity = uram_est * 72 * 4096
        return wbits / uram_est_capacity

    def adapt_for_loop_body(self, input_types: "list[LoopBodyInputType]") -> None:  # noqa: ARG002
        """Adapt the MVAU for loop body execution.

        When the weight tensor (input[1]) is indexed per iteration (PARAMETER
        type), the weights must be streamed from external memory over the
        AXI-MM interface. That path only exists in the RTL backend
        (MVAU_rtl overrides this method to set mem_mode "external_mem").
        The abstract HW op and the HLS backend cannot stream loop weights, so
        reaching this base implementation with a streamed weight input is an
        error: the node must be specialized to MVAU_rtl before loop rolling.

        LoopRolling flags nodes with a per-iteration indexed weight by setting
        mlo_max_iter (on the consumer of each PARAMETER loop input), so key off
        that per-node signal rather than the positional loop signature.

        NOTE: LoopRolling swallows KeyError/AttributeError from this hook, so
        this deliberately raises a FINNUserError (not one of those) to surface loudly.

        Args:
            input_types: Types of the loop body inputs (unused).
        """
        if self.mlo_max_iter > 0:
            raise FINNUserError(
                "MLO weight streaming requires the RTL MVAU backend. "
                "Specialize this MVAU to MVAU_rtl before loop rolling; "
                "the HLS/abstract MVAU cannot stream per-iteration weights "
                "over AXI-MM."
            )

    def get_exp_cycles(self) -> int:
        """Return the expected cycle count given the folding configuration and tiling factor."""
        pe = self.pe
        simd = self.simd
        th = self.th
        num_inp_vec = self.num_input_vectors
        mh = self.mh
        mw = self.mw
        # since mmv != 1 is not supported yet, we set mmv for now to 1
        mmv = 1
        # Tiling/systolic reduces throughput
        # TH>1 (tiling) reduces throughput by factor TH (tinner = PE*SIMD/TH)
        exp_cycles = (mh / pe) * (mw / simd) * np.prod(num_inp_vec) * th / mmv
        return int(exp_cycles)

    def minimize_accumulator_width(
        self, model: ModelWrapper, datatype_only: bool = False
    ) -> BaseDataType:
        """Minimize the accumulator bit width according to the weight values,
        input data types, and size of dot product.

        Args:
            model: FINN ModelWrapper containing this node.
            datatype_only: If True, use worst-case datatype bounds instead of actual
                weight values.

        Returns:
            The (possibly updated) accumulator datatype.
        """
        weights = model.get_initializer(self.onnx_node.input[1])
        idt = self.get_input_datatype(0)

        # if datatype_only, runtime-writeable weights, mem_mode=external, or
        # weights are absent (MLO), then we use worst-case values from datatypes
        if (
            datatype_only
            or self.runtime_writeable_weights
            or self.mem_mode in ["external", "external_mem", "dynamic"]
            or not isinstance(weights, np.ndarray)
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

    def minimize_weight_bit_width(
        self, model: ModelWrapper, datatype_only: bool = False
    ) -> BaseDataType:
        """Minimize the bit width based on the values of the weights.

        Args:
            model: FINN ModelWrapper containing this node.
            datatype_only: If True, skip value-based minimization.

        Returns:
            The (possibly updated) weight datatype.
        """
        if datatype_only:
            return DataType[cast("str", self.get_nodeattr("weightDataType"))]

        if not (
            self.runtime_writeable_weights
            or self.mem_mode in ["external", "external_mem", "dynamic"]
        ):
            weights = model.get_initializer(self.onnx_node.input[1])
            if not isinstance(weights, np.ndarray):
                return DataType[cast("str", self.get_nodeattr("weightDataType"))]
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

        Args:
            orig_thres_matrix: Threshold matrix of shape (MH or 1, n_thres_steps).

        Returns:
            The reshaped threshold tensor.
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

        Args:
            orig_weight_matrix: Weight matrix of shape (MW, MH).

        Returns:
            The reshaped weight tensor.
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

        Args:
            weights: Numpy array with the weights to be put into the file.
            weight_file_mode: One of ``hls_header``, ``decoupled_npy``,
                ``decoupled_verilog_dat`` or ``decoupled_runtime``.
            weight_file_name: Filename for the weight file to be generated.
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
            # tiling
            th = self.th
            tinner = (pe * simd) // th
            weight_tensor_simd_flipped = weight_tensor_simd_flipped.reshape(1, -1, tinner)
            # The .dat weights are PE-flipped (np.flip axis=-2), which reverses the
            # PE dimension - the same dimension that TH tiles into sub-tiles. This
            # reverses the order of the TH sub-tiles within each PE group. Undo that
            # by flipping the TH tile order back (no-op for th=1).
            weight_tensor_pe_flipped = weight_tensor_pe_flipped.reshape(1, -1, th, tinner)
            weight_tensor_pe_flipped = np.flip(weight_tensor_pe_flipped, axis=-2)
            weight_tensor_pe_flipped = weight_tensor_pe_flipped.reshape(1, -1, tinner)
            if weight_file_mode == "decoupled_npy":
                # save weight stream into npy for cppsim
                np.save(weight_file_name, weight_tensor_simd_flipped)
            elif weight_file_mode == "decoupled_verilog_dat" and (
                self.mlo_max_iter or self.mem_mode == "external_mem"
            ):
                # AXI-MM / fetch_weights path (MLO and external_mem): the external
                # memory image is DMA-word-aligned to match the wrapper VPC, which
                # reads DMA_PE = DATA_BITS // WEIGHT_WIDTH whole elements from the low
                # bits of each DATA_BITS-wide bus word (element 0 at the LSB) and
                # discards the top DATA_BITS % WEIGHT_WIDTH padding bits. Storing one
                # bus word per line (DATA_BITS bits) guarantees no element straddles a
                # bus-word boundary; when WEIGHT_WIDTH divides DATA_BITS this reduces to
                # tight packing (byte-identical to the previous per-group image).
                #
                # IWSIMD is (PE*SIMD)/TH for TH>1, SIMD otherwise. The element order
                # within each IWSIMD group and the per-group ordering depend on how
                # fetch_weights delivers the stream to the MVU:
                #   - TH>1 (tiled MVAU): the stream passes straight through to the tiled
                #     MVU, which expects the PE-flipped, TH-sub-tile-undone ordering.
                #     weight_tensor_pe_flipped is already in IWSIMD-sized chunks
                #     (tinner == (PE*SIMD)/TH == IWSIMD).
                #   - TH=1 (standard MVAU): the stream goes through local_weight_buffer,
                #     which distributes consecutive SIMD groups across PE lanes 0..PE-1
                #     (pe-minor) and pairs weight SIMD lane s with activation SIMD lane s.
                #     That reconstruction needs the natural (unflipped) PE/SIMD order.
                # Within a group the VPC's little-endian (element 0 = LSB) consumption
                # reverses the IWSIMD elements, so reverse each group here to build the
                # flat element stream the VPC sees, then pack DMA_PE elements/bus word.
                data_bits = _FETCH_WEIGHTS_DATA_BITS
                iwsimd = (pe * simd) // th if th > 1 else simd
                dma_pe = data_bits // export_wdt.bitwidth()  # whole elements per bus word
                if th > 1:
                    weight_tensor_iwsimd = weight_tensor_pe_flipped
                else:
                    weight_tensor_iwsimd = weight_tensor_unflipped.reshape(1, -1, pe * simd).copy()
                # reverse within each IWSIMD group -> flat VPC little-endian element order
                weight_tensor_iwsimd_groups = weight_tensor_iwsimd.reshape(1, -1, iwsimd)
                flat_elems = weight_tensor_iwsimd_groups[:, :, ::-1].reshape(1, -1)
                # pad the tail up to a whole number of bus words (unused VPC lanes)
                n_elems = flat_elems.shape[1]
                n_words = (n_elems + dma_pe - 1) // dma_pe
                pad_elems = n_words * dma_pe - n_elems
                if pad_elems:
                    flat_elems = np.concatenate(
                        [flat_elems, np.zeros((1, pad_elems), dtype=flat_elems.dtype)], axis=1
                    )
                # one bus word per line: DMA_PE elements packed element-0-at-LSB
                # (reverse_inner=True), top bits zero-padded to DATA_BITS.
                word_groups = flat_elems.reshape(1, n_words, dma_pe)
                word_groups = pack_innermost_dim_as_hex_string(
                    word_groups, export_wdt, data_bits, reverse_inner=True, prefix=""
                )
                weight_stream = word_groups.flatten().copy()
                with Path(weight_file_name).open("w") as f:
                    for val in weight_stream:
                        f.write(val + "\n")
            elif weight_file_mode == "decoupled_verilog_dat":
                # convert weight values into hexstring
                weight_width = self.get_instream_width(1)
                if self.mem_mode == "dynamic":
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
                if self.mem_mode == "dynamic":
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

    def generate_params(
        self, model: ModelWrapper, path: str | Path, fpgapart: str | None = None  # noqa: ARG002
    ) -> None:
        """Write weight and threshold parameter files for the selected mem_mode into ``path``.

        Args:
            model: FINN ModelWrapper containing this node.
            path: Output directory for the generated files.
            fpgapart: Target FPGA part (unused, the MVAU parameter layout is device-agnostic).
        """
        mem_mode = self.mem_mode
        code_gen_dir = Path(path)
        # weights, if not external
        weights = model.get_initializer(self.onnx_node.input[1])
        if isinstance(weights, np.ndarray):
            match mem_mode:
                case "internal_embedded":
                    # save hlslib-compatible weights in params.h
                    self.make_weight_file(weights, "hls_header", str(code_gen_dir / "params.h"))
                case "internal_decoupled" | "external" | "external_mem":
                    # save internal_decoupled weights for cppsim
                    self.make_weight_file(
                        weights, "decoupled_npy", str(code_gen_dir / "input_1.npy")
                    )
                    # also save weights as Verilog .dat file
                    # This file will be ignored when synthesizing UltraScale memory.
                    self.make_weight_file(
                        weights, "decoupled_verilog_dat", str(code_gen_dir / "memblock.dat")
                    )
        elif mem_mode not in ["external", "dynamic", "external_mem"]:
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
        """Return a dictionary with MAC operation, weight and threshold parameter counts."""
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
        """Return the top-level interface names grouped by protocol for the selected mem_mode.

        Returns:
            Dictionary mapping interface types to port names.
        """
        s_axis: list[tuple[str, int]] = [("in0_V", self.get_instream_width_padded(0))]
        m_axis: list[tuple[str, int]] = [("out0_V", self.get_outstream_width_padded(0))]
        aximm: list[tuple[str, int]] = []
        ap_none: list[str] = []
        intf_names: dict[str, list[tuple[str, int]] | list[str]] = {
            "clk": ["ap_clk"],
            "rst": ["ap_rst_n"],
            "s_axis": s_axis,
            "m_axis": m_axis,
            "aximm": aximm,
            "axilite": [],
            "ap_none": ap_none,
        }

        try:
            pumped_compute = self.get_nodeattr("pumpedCompute")
        except AttributeError:
            pumped_compute = 0

        if pumped_compute or self.pumped_memory:
            intf_names["clk2x"] = ["ap_clk2x"]

        match self.mem_mode:
            case "external_mem":
                aximm.append(("axi_mm", 64))
                if self.mlo_max_iter > 0:
                    s_axis.append(("in_idx0_V", 32))
                    if get_by_name(self.onnx_node.attribute, "address_offset") is not None:
                        ap_none.append("base_address")
            case "dynamic" | "external":
                s_axis.append(("in1_V", self.get_instream_width_padded(1)))
            case "internal_decoupled":
                # only expose axilite interface if attribute is set
                if self.runtime_writeable_weights:
                    intf_names["axilite"] = ["s_axilite"]

        return intf_names

    def generate_infra_hdl(self, fpgapart: str) -> None:
        """Generate the weight-infrastructure HDL for the streamed weight mem_modes.

        This is the dynamic load (dynamic), external-memory fetch (external_mem) or
        on-chip memstream (internal_decoupled) component. Shared by the HLS and RTL
        MVAU backends; the compute-core HDL is generated by each backend separately
        (RTLBackend.generate_hdl / HLSBackend codegen).

        Args:
            fpgapart: Target FPGA part.
        """
        match self.mem_mode:
            case "dynamic":
                self.generate_hdl_dynload()
            case "external_mem":
                self.generate_hdl_fetch_weights()
            case "internal_decoupled":
                if (
                    self.ram_style == "ultra"
                    and not is_versal(fpgapart)
                    and self.runtime_writeable_weights != 1
                ):
                    raise FINNUserError(
                        f"{self.onnx_node.name}: layer with URAM weights must have "
                        f"runtime_writeable_weights=1 if an Ultrascale device is targeted."
                    )
                self.generate_hdl_memstream(fpgapart, pumped_memory=self.pumped_memory)

    def code_generation_ipi(self) -> list[str]:
        """Construct the TCL for node instantiation in Vivado IPI, including weight infra.

        Returns:
            List of TCL commands for the Vivado IP integrator.
        """
        cmd: list[str] = []

        # check if additional components are needed
        mem_mode = self.mem_mode
        if mem_mode in ["internal_decoupled", "dynamic", "external_mem"]:
            # Base
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

            # Instantiate either the HLS or RTL IP depending on operator
            self.instantiate_ip(cmd)
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
            rtllib = Path(get_settings().finn_rtllib)
            win_name = ""

            match mem_mode:
                # Dynamic loader instantiation
                case "dynamic":
                    # additional dynamic input
                    win_name = self.get_verilog_top_module_intf_names()["s_axis"][1][0]
                    cmd.append(
                        "create_bd_intf_pin -mode Slave "
                        f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/{win_name}"
                    )

                    # dynamic loader
                    # automatically find dynamic load verilog component in code generation dir
                    strm_tmpl = _find_codegen_file(
                        code_gen_dir, "_dynamic_load_wrapper.v", node_name
                    )
                    sourcefiles = [
                        str(Path(code_gen_dir) / strm_tmpl),
                        str(rtllib / "dynload" / "hdl" / "dynamic_load.sv"),
                    ]
                    strm_inst = node_name + "_wdynld"
                    strm_out_name = "m_axis_0"

                # Fetch weights instantiation (MLO and tiling)
                case "external_mem":
                    # additional inputs
                    cmd.append(
                        "create_bd_intf_pin -mode Master "
                        f"-vlnv xilinx.com:interface:aximm_rtl:1.0 /{node_name}/axi_mm"
                    )
                    if self.mlo_max_iter > 0:
                        cmd.append(
                            "create_bd_intf_pin -mode Slave "
                            f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/in_idx0_V"
                        )

                        if get_by_name(self.onnx_node.attribute, "address_offset") is not None:
                            ba_name = self.get_verilog_top_module_intf_names()["ap_none"][0]
                            cmd.append(
                                f"create_bd_pin -dir I -from 63 -to 0 /{node_name}/{ba_name}"
                            )

                    # instantiate a fetch weights component and connect it to the IP
                    # automatically find fetch weights verilog component in code generation dir
                    strm_tmpl = _find_codegen_file(
                        code_gen_dir, "_fetch_weights_wrapper.v", node_name
                    )
                    fwg_rtllib_dir = rtllib / "fetch_weights"
                    dwc_rtllib_dir = rtllib / "dwc" / "hdl"
                    sourcefiles = [
                        str(Path(code_gen_dir) / strm_tmpl),
                        str(rtllib / "skid" / "skid.sv"),
                        str(fwg_rtllib_dir / "fetch_weights.sv"),
                        str(fwg_rtllib_dir / "local_weight_buffer.sv"),
                        str(dwc_rtllib_dir / "vpc.sv"),
                        str(dwc_rtllib_dir / "dwc_axi.sv"),
                        *fifo_rtl_files(),
                    ]
                    # Create Vivado axis_dwidth_converter IP
                    wdt = self.get_input_datatype(1)
                    iwsimd = (self.pe * self.simd) // self.th if self.th > 1 else self.simd
                    ds_bits_ba = ((iwsimd * wdt.bitwidth() + 7) // 8) * 8
                    dwc_ip_name = node_name + "_dwc"
                    s_bytes = _FETCH_WEIGHTS_DATA_BITS // 8
                    m_bytes = ds_bits_ba // 8
                    cmd += [
                        (
                            "create_ip -name axis_dwidth_converter -vendor xilinx.com "
                            f"-library ip -version 1.1 -module_name {dwc_ip_name}"
                        ),
                        (
                            "set_property -dict [list "
                            f"CONFIG.S_TDATA_NUM_BYTES {{{s_bytes}}} "
                            f"CONFIG.M_TDATA_NUM_BYTES {{{m_bytes}}} "
                            "CONFIG.HAS_TLAST {1} "
                            "CONFIG.HAS_TKEEP {1} "
                            f"] [get_ips {dwc_ip_name}]"
                        ),
                        f"generate_target all [get_ips {dwc_ip_name}]",
                    ]

                    # add files from cdma dir
                    for subdir in ("", "cdma_a", "cdma_u", "cdma_x"):
                        cdma_path = rtllib / "cdma" / subdir
                        for file in sorted(p.name for p in cdma_path.iterdir()):
                            if file.endswith((".sv", ".svh")):
                                sourcefiles.append(str(cdma_path / file))
                    strm_inst = node_name + "_fetch_weights"
                    strm_out_name = "out0_V"

                # Memstream instantiation
                case "internal_decoupled":
                    # instantiate a streamer and connect it to the IP
                    # automatically find memstream verilog component in code generation dir
                    strm_tmpl = _find_codegen_file(code_gen_dir, "_memstream_wrapper.v", node_name)
                    ms_rtllib_dir = rtllib / "memstream" / "hdl"
                    sourcefiles = [
                        str(Path(code_gen_dir) / strm_tmpl),
                        str(rtllib / "axi" / "hdl" / "axilite.sv"),
                        str(ms_rtllib_dir / "memstream_axi.sv"),
                        str(ms_rtllib_dir / "memstream.sv"),
                    ]
                    strm_inst = node_name + "_wstrm"
                    strm_out_name = "m_axis_0"

                case _:
                    raise FINNInternalError(
                        f"{node_name}: unreachable branch in code_generation_ipi weight "
                        f"streamer setup"
                    )

            cmd += [f"add_files -norecurse {f}" for f in sourcefiles]
            strm_tmpl_name = strm_tmpl[:-2]
            cmd.append(
                f"create_bd_cell -type hier -reference {strm_tmpl_name} /{node_name}/{strm_inst}"
            )

            # Connect
            match mem_mode:
                case "dynamic":
                    cmd.append(
                        f"connect_bd_intf_net [get_bd_intf_pins {node_name}/{win_name}] "
                        f"[get_bd_intf_pins {node_name}/{strm_inst}/s_axis_0]"
                    )

                case "external_mem":
                    cmd.append(
                        f"connect_bd_intf_net [get_bd_intf_pins {node_name}/axi_mm] "
                        f"[get_bd_intf_pins {node_name}/{strm_inst}/axi_mm]"
                    )
                    if self.mlo_max_iter > 0:
                        cmd.append(
                            f"connect_bd_intf_net [get_bd_intf_pins {node_name}/in_idx0_V] "
                            f"[get_bd_intf_pins {node_name}/{strm_inst}/in_idx0_V]"
                        )

                        if get_by_name(self.onnx_node.attribute, "address_offset") is not None:
                            cmd.append(
                                f"connect_bd_net [get_bd_pins {node_name}/base_address] "
                                f"[get_bd_pins {node_name}/{strm_inst}/base_address]"
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
            if mem_mode == "internal_decoupled":
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
                # runtime writeable weights (skip for MLO nodes)
                if runtime_writeable and not self.mlo_max_iter:
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

        elif mem_mode in ["internal_embedded", "external"]:
            # base class impl sufficient for internal_embedded/external modes
            self.instantiate_ip(cmd)

        else:
            raise FINNInternalError(
                f"{self.onnx_node.name}: unrecognized mem_mode {mem_mode} "
                "for MatrixVectorActivation"
            )

        return cmd

    def get_weight_mem_bytes(self) -> tuple[int, int]:
        """Return (size, offs) in bytes for one layer's weight matrix.

        The DDR image is DMA-word-aligned: each DATA_BITS-wide bus word holds
        DMA_PE = DATA_BITS // WEIGHT_WIDTH whole elements (top bits zero-padded),
        so the per-layer size is ceil(MH*MW / DMA_PE) bus words. That is already
        bus-aligned, so offs == size == LAYER_OFFS in fetch_weights.sv.

        Returns:
            Tuple of the weight matrix size and the offset to the next layer in bytes.
        """
        data_bits = _FETCH_WEIGHTS_DATA_BITS
        weight_width = self.get_input_datatype(1).bitwidth()
        dma_pe = data_bits // weight_width  # whole elements per bus word
        n_elems = self.mh * self.mw
        n_words = roundup_to_integer_multiple(n_elems, dma_pe) // dma_pe
        size = n_words * (data_bits // 8)
        offs = size  # already AXI bus-width aligned
        return size, offs

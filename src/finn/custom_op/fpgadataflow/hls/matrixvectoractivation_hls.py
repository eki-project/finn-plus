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

"""HLS backend implementation of the MatrixVectorActivation (MVAU) layer."""

import math
import numpy as np
from onnx import GraphProto, NodeProto
from pathlib import Path
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import cast

from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.hwcustomop import NodeAttrTypes
from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU
from finn.util.basic import MAX_ALLOWED_AP_INT_W
from finn.util.data_packing import hls_datatype_str, npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.exception import FINNInternalError, FINNUserError

# ONNX i/o tensor shape assumptions for MatrixVectorActivation_hls:
# input 0 is the input tensor, shape (.., i_size) = (..., MW)
# input 1 is the weight tensor, shape (i_size, o_size) = (MW, MH)
# (optional) input 2 is the thresholds tensor, shape (o_size, n_thres)
# output 0 is the output tensor, shape (.., o_size) = (..., MH)
# the ... here can be any shape (representing groups of vectors)

# weight memory modes in which the weights arrive through the in1_V stream
_WEIGHT_STREAM_MEM_MODES = ("internal_decoupled", "external", "external_mem", "dynamic")
# all weight memory modes supported by the HLS backend
_MEM_MODES = ("internal_embedded", *_WEIGHT_STREAM_MEM_MODES)


@register_custom_op
class MVAU_hls(MVAU, HLSBackend):
    """Corresponds to finn-hlslib MatrixVectorActivation_Batch function."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return node attribute types, combining MVAU and HLSBackend attributes."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(MVAU.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        # for HLS MVAU default resType to lut
        my_attrs["resType"] = ("s", False, "lut", {"auto", "lut", "dsp"})
        return my_attrs

    def _uses_weight_stream(self) -> bool:
        """Return whether weights arrive as a stream (decoupled/external/external_mem/dynamic)."""
        return self.mem_mode in _WEIGHT_STREAM_MEM_MODES

    def _weight_stream_width(self) -> int:
        """Return the width of the in1_V weight stream in bits.

        For dynamic weights, the stream carries a whole PE x SIMD weight block per cycle.
        """
        width = self.get_instream_width(1)
        if self.mem_mode == "dynamic":
            width = width * self.simd
        return width

    def lut_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Calculate resource estimations for LUTs based on:
        - FINN-R: An End-to-End Deep-Learning Framework for Fast
        Exploration of Quantized Neural Networks
        - M. Blott, T. B. Preusser, N. J. Fraser, G. Gambardella, K. O'Brien,
        Y. Umuroglu, M. Leeser and K. Vissers
        - 12. Sep 2018.
        """
        # TODO add in/out FIFO contributions
        p = self.pe
        q = self.simd
        mw = self.mw
        w = self.get_input_datatype(1).bitwidth()
        # determine tdt with input and weight data types
        idt = self.get_input_datatype(0)
        a = idt.bitwidth()
        # parameters from experiments in paper mentioned above
        c0 = 300
        c1 = 1.1
        c2 = 0
        mmode = self.mem_mode
        mstyle = self.ram_style
        if (mmode == "internal_decoupled" and mstyle == "distributed") or (
            mmode == "internal_embedded" and self.calc_wmem() <= 128
        ):
            c2 = (p * q * w) * math.ceil(self.calc_wmem() / 64)

        # multiplication
        mult_luts = 0 if self.res_type == "dsp" else q * (2 * math.ceil((w + a) / 6) - 1) * (w + a)
        # adder tree
        addertree_luts = (w + a) * (2 * q - 1)
        # accumulator
        acc_datatype = self.get_accumulator_datatype()
        # if accDataType is not set, then it will default to INT32, which would
        # be a large overestimate in most (if not all) cases. In this scenario,
        # we would use the minimum accumulator as determined by the data types
        # bound, derived in https://arxiv.org/abs/2301.13376
        alpha = math.log(mw, 2) + w + a - 1 - int(idt.signed())
        acc_bits = min(
            acc_datatype.bitwidth(),
            np.ceil(alpha + math.log(1 + pow(2, -alpha), 2) + 1),
        )
        acc_luts = acc_bits
        # thresholds and threshold comparators
        thr_luts = 0
        comp_luts = 0
        if (self.no_activation == 0) and (self.ram_style_thresholds == "distributed"):
            b = self.get_output_datatype().bitwidth()
            thr_luts = (2**b - 1) * acc_bits * math.ceil(self.calc_tmem() / 64)
            comp_luts = (2**b - 1) * acc_bits

        return int(
            c0 + c1 * (p * (mult_luts + addertree_luts + acc_luts + thr_luts + comp_luts)) + c2
        )

    def dsp_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Estimate the number of DSP slices used for the multiplications (0 if resType is lut)."""
        p = self.pe
        q = self.simd
        w = self.get_input_datatype(1).bitwidth()
        a = self.get_input_datatype(0).bitwidth()
        # TODO: more accurate modelling
        mult_dsp = p * q * np.ceil((w + a) / 48) if self.res_type == "dsp" else 0
        return int(mult_dsp)

    def code_generation_ipgen(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:
        """Generate c++ code and tcl script for ip generation."""
        super().code_generation_ipgen(model, fpgapart, clk)
        # generate the weight-infrastructure HDL (shared MVAU helper)
        self.generate_infra_hdl(fpgapart)

    def global_includes(self) -> None:
        """Populate the C++ includes for the generated HLS code."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "activations.hpp"', '#include "mvau.hpp"']

        if self.mem_mode not in _MEM_MODES:
            raise FINNInternalError(
                f"{self.onnx_node.name}: mem_mode must be one of {_MEM_MODES}, got {self.mem_mode}"
            )
        if self.mem_mode == "internal_embedded":
            self.code_gen_dict["$GLOBALS$"] += ['#include "params.h"']
        if self.calc_tmem() != 0:
            # TODO find a better way of checking for no pregenerated thresholds
            self.code_gen_dict["$GLOBALS$"] += ['#include "thresh.h"']

    def defines(self, var: str) -> None:
        """Emit the C++ constants for matrix dimensions and folding."""
        # Tiling (TH>1) is supported by the RTL backend. The HLS
        # backend implements the untiled MVAU, so TH must be 1 here.
        if self.th != 1:
            raise FINNUserError(
                f"{self.onnx_node.name}: tiled MVAU (TH>1) is only supported by the "
                "RTL backend (MVAU_rtl). Specialize this node to MVAU_rtl or set TH=1."
            )
        # Only ipgen mode: Make sure that SIMD parameter satisfies minimum requirements.
        if var == "ipgen":
            simd = self.simd
            mw = self.mw
            if simd < (mw / 1024):
                raise FINNUserError(
                    f"HLS synthesis of MatrixVectorActivation requires: SIMD >= MW / 1024. "
                    f"This is not fulfilled with: SIMD={simd} and MW={mw} for node: "
                    f"{self.onnx_node.name}."
                )
        self.code_gen_dict["$DEFINES$"] = [
            f"""static constexpr unsigned MW1 = {self.mw};
            static constexpr unsigned MH1 = {self.mh};
            static constexpr unsigned SIMD1 = {self.simd};
            static constexpr unsigned PE1 = {self.pe};"""
        ]

    def timeout_value(self) -> None:
        """Set the number of consecutive idle calls after which cppsim stops.

        The free-running top consumes one SIMD-wide input token per call and emits an
        output only after MW/SIMD of them, so the default of 1000 idle calls is too
        short for wider or less folded layers (e.g. MW 1152 at SIMD 1): the loop then
        ends early and the final output read blocks forever.
        """
        self.code_gen_dict["$TIMEOUT_VALUE$"] = [str(self.mw // self.simd + 1000)]

    def read_npy_data(self) -> None:
        """Emit C++ code that reads the input (and streamed weight) numpy data for cppsim."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        npy_type = "float"
        npy_in = f"{code_gen_dir}/input_0.npy"
        # stream type deduced from in0_V declaration
        self.code_gen_dict["$READNPYDATA$"] = [f'npy2tensorstream<{npy_type}>("{npy_in}", in0_V);']

        if self._uses_weight_stream():
            w_npy_in = f"{code_gen_dir}/input_1.npy"
            num_reps = int(np.prod(self.num_input_vectors))
            # stream type deduced from in1_V declaration
            self.code_gen_dict["$READNPYDATA$"].append(
                f"for(unsigned rep = 0; rep < {num_reps}; rep++) "
                f'npy2tensorstream<{npy_type}>("{w_npy_in}", in1_V);'
            )

    def strm_decl(self) -> None:
        """Emit C++ stream declarations for the input, weight and output streams."""
        simd = self.simd
        pe = self.pe
        i_elem = hls_datatype_str(self.get_input_datatype(0))
        o_elem = hls_datatype_str(self.get_output_datatype())
        self.code_gen_dict["$STREAMDECLARATIONS$"] = [
            # 2-D tensor: tensor<TI, MMV=1, SIMD>
            f'hls::stream<tensor<{i_elem},1,{simd}>> in0_V ("in0_V");',
            # 2-D tensor: tensor<TO, MMV=1, PE>
            f'hls::stream<tensor<{o_elem},1,{pe}>> out0_V ("out0_V");',
            f'hls::stream<tensor<{o_elem},1,{pe}>> strm0 ("strm0");',
        ]

        if self._uses_weight_stream():
            w_elem = hls_datatype_str(self.get_input_datatype(1))
            # 2-D tensor: tensor<TW, PE, SIMD>
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                f'hls::stream<tensor<{w_elem},{pe},{simd}>> in1_V ("in1_V");'
            )

    def docompute(self) -> None:
        """Emit the C++ call to the finn-hlslib MVAU kernel for the selected mem_mode."""
        map_to_hls_mult_style = {
            "auto": "ap_resource_dflt()",
            "lut": "ap_resource_lut()",
            "dsp": "ap_resource_dsp()",
        }
        if self.calc_tmem() == 0:
            odtype_hls_str = hls_datatype_str(self.get_output_datatype())
            threshs = f"PassThroughActivation<{odtype_hls_str}>()"
        else:
            threshs = "threshs"
        mult_style = map_to_hls_mult_style[self.res_type]
        if self.mem_mode == "internal_embedded":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                (
                    "Matrix_Vector_Activate_Batch<MW1, MH1, SIMD1, PE1>\n"
                    f"                (in0_V, out0_V, weights, {threshs}, {mult_style});"
                )
            ]
        elif self._uses_weight_stream():
            self.code_gen_dict["$DOCOMPUTE$"] = [
                (
                    "Matrix_Vector_Activate_Stream_Batch<MW1, MH1, SIMD1, PE1>\n"
                    f"                (in0_V, out0_V, in1_V, {threshs}, {mult_style});"
                )
            ]
        else:
            raise FINNInternalError(
                f"{self.onnx_node.name}: mem_mode must be one of {_MEM_MODES}, got {self.mem_mode}"
            )

    def dataoutstrm(self) -> None:
        """Emit C++ code that writes the output stream to a numpy file for cppsim."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        npy_type = "float"
        npy_out = f"{code_gen_dir}/output_0.npy"
        shape_cpp_str = str(self.get_folded_output_shape()).replace("(", "{").replace(")", "}")

        # stream type deduced from strm0 declaration
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            f'tensorstream2npy<{npy_type}>(strm0, {shape_cpp_str}, "{npy_out}");'
        ]

    def save_as_npy(self) -> None:
        """Emit C++ code for saving outputs as numpy (unused, handled by dataoutstrm)."""
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def blackboxfunction(self) -> None:
        """Emit the top-level HLS function signature for the selected mem_mode."""
        simd = self.simd
        pe = self.pe
        i_elem = hls_datatype_str(self.get_input_datatype(0))
        o_elem = hls_datatype_str(self.get_output_datatype())
        if self.mem_mode == "internal_embedded":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                (
                    f"void {self.onnx_node.name}(hls::stream<tensor<{i_elem},1,{simd}>> &in0_V, "
                    f"hls::stream<tensor<{o_elem},1,{pe}>> &out0_V)"
                )
            ]
        elif self._uses_weight_stream():
            w_elem = hls_datatype_str(self.get_input_datatype(1))
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                (
                    f"void {self.onnx_node.name}(hls::stream<tensor<{i_elem},1,{simd}>> &in0_V, "
                    f"hls::stream<tensor<{w_elem},{pe},{simd}>> &in1_V, "
                    f"hls::stream<tensor<{o_elem},1,{pe}>> &out0_V)"
                )
            ]
        else:
            raise FINNInternalError(
                f"{self.onnx_node.name}: mem_mode must be one of {_MEM_MODES}, got {self.mem_mode}"
            )

    def pragmas(self) -> None:
        """Emit HLS interface, resource and array partition pragmas."""
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0_V",
            "#pragma HLS INTERFACE axis port=out0_V",
            "#pragma HLS INTERFACE ap_ctrl_none port=return",
            "#pragma HLS DATAFLOW disable_start_propagation",
        ]

        if self.mem_mode == "internal_embedded":
            # weights is TW weights[WMEM][PE][SIMD]: one ROM per PE (dim 2),
            # each WMEM deep with SIMD elements packed per word (dim 3)
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS ARRAY_RESHAPE variable=weights complete dim=3"
            )
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS ARRAY_PARTITION variable=weights complete dim=2"
            )
        elif self._uses_weight_stream():
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=in1_V")
        else:
            raise FINNInternalError(
                f"{self.onnx_node.name}: mem_mode must be one of {_MEM_MODES}, got {self.mem_mode}"
            )

        # the threshold tensor is acc_type [PE][TMEM][N_THRES]
        # partition for parallel access along PE and N_THRES
        # dimensions (dims 1 and 3)
        if self.calc_tmem() != 0:
            # TODO find a better way of checking for no pregenerated thresholds
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=1"
            )
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=3"
            )
            # add resource pragma for thresholds if set
            ram_style_thresholds = self.ram_style_thresholds
            if ram_style_thresholds == "distributed":
                self.code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS RESOURCE variable=threshs.m_thresholds core=ROM_2P_LUTRAM"
                )
            elif ram_style_thresholds == "block":
                self.code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS RESOURCE variable=threshs.m_thresholds core=ROM_2P_BRAM"
                )
            elif ram_style_thresholds == "auto":
                # no pragma needed
                pass
            else:
                raise FINNInternalError(
                    f"{self.onnx_node.name}: unrecognized ram_style_thresholds value "
                    f"{ram_style_thresholds}"
                )

    def get_ap_int_max_w(self) -> int:
        """Return the maximum ap_int width needed, accounting for the
        weight stream and single-PE entries.
        """
        # base class impl (max of inp/out stream widths)
        max_of_io = super().get_ap_int_max_w()
        # internal_decoupled mode weight stream
        weightstream = self._weight_stream_width()
        # single PE weight entry
        weight_bits = self.get_input_datatype(1).bitwidth()
        single_pe_w = self.simd * weight_bits
        final = max([weightstream, max_of_io, single_pe_w])
        if final > MAX_ALLOWED_AP_INT_W:
            raise FINNInternalError(
                f"The HLS top module of node {self.onnx_node.name} requires AP_INT_MAX_W to be "
                f"set to {final}, but the maximum allowed is {MAX_ALLOWED_AP_INT_W}."
            )
        return final

    def execute_node(
        self, context: dict[str, np.ndarray], graph: GraphProto  # noqa: ARG002
    ) -> None:
        """Execute the node via cppsim or rtlsim, handling weight inputs per mem_mode.

        Args:
            context: Dictionary mapping tensor names to numpy arrays.
            graph: ONNX graph containing this node.
        """
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node

        # TODO ensure codegen dir exists
        if mode == "cppsim":
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_cppsim"))
        elif mode == "rtlsim":
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
        else:
            raise FINNInternalError(
                f"{self.onnx_node.name}: invalid exec_mode {mode}, "
                f'must be one of ("cppsim", "rtlsim")'
            )

        export_idt = self.get_input_datatype(0)
        # create a npy file fore each input of the node (in_ind is input index)
        for in_ind, inputs in enumerate(node.input):
            # it is assumed that the first input of the node is the data input
            # the second input are the weights
            if str(context[inputs].dtype) != "float32":
                raise FINNInternalError(
                    f"{self.onnx_node.name}: input datatype is not float32 as expected"
                )

            if in_ind == 0:
                expected_inp_shape = self.get_folded_input_shape(in_ind)
                reshaped_input = context[inputs].reshape(expected_inp_shape)
                export_idt = self.get_input_datatype(0)
                # make copy before saving the array
                reshaped_input = reshaped_input.copy()
                np.save(str(Path(code_gen_dir) / "input_0.npy"), reshaped_input)

            if in_ind == 1 and self._uses_weight_stream():
                reshaped_input = context[inputs].reshape(-1, context[inputs].shape[-1])
                self.make_weight_file(
                    reshaped_input, "decoupled_npy", f"{code_gen_dir}/input_1.npy"
                )

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            # reinterpret binary output as bipolar where needed
            if self.get_output_datatype() == DataType["BIPOLAR"]:
                context[node.output[0]] = 2 * context[node.output[0]] - 1
            if context[node.output[0]].shape != self.get_normal_output_shape():
                raise FINNInternalError(
                    f"{self.onnx_node.name}: cppsim did not produce expected output shape"
                )
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width(0)
            inp = npy_to_rtlsim_input(f"{code_gen_dir}/input_0.npy", export_idt, nbits)
            self.reset_rtlsim(sim)

            if self._uses_weight_stream():
                wnbits = self._weight_stream_width()
                export_wdt = self.get_input_datatype(1)
                # make_weight_file already converts bipolar weights to binary encoding,
                # so use BINARY as the export dtype for weight packing
                if export_wdt == DataType["BIPOLAR"]:
                    export_wdt = DataType["BINARY"]

                wei = npy_to_rtlsim_input(f"{code_gen_dir}/input_1.npy", export_wdt, wnbits)
                num_w_reps = np.prod(self.num_input_vectors)

                io_dict = {
                    "inputs": {"in0": inp, "in1": wei * num_w_reps},
                    "outputs": {"out0": []},
                }
            else:
                io_dict = {
                    "inputs": {"in0": inp},
                    "outputs": {"out0": []},
                }

            self.rtlsim_multi_io(sim, io_dict)
            super().close_rtlsim(sim)
            output = io_dict["outputs"]["out0"]
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = f"{code_gen_dir}/output_0.npy"
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(output, out_npy_path, odt, out_shape, packed_bits, target_bits)

            # load and reshape output
            output = np.load(out_npy_path)
            oshape = self.get_normal_output_shape()
            context[node.output[0]] = np.asarray([output], dtype=np.float32).reshape(*oshape)

    def minimize_weight_bit_width(
        self, model: ModelWrapper, datatype_only: bool = False
    ) -> BaseDataType:
        """Minimize weight and threshold datatypes, with HLS-specific adjustments.

        The HLS implementation uses the threshold datatype for comparisons.
        When the threshold datatype is narrower than the accumulator datatype,
        accumulator values get truncated, which can cause incorrect results.
        To prevent this, ensure threshold datatype is at least as wide as
        accumulator datatype.

        Args:
            model: FINN ModelWrapper containing this node.
            datatype_only: If True, skip value-based minimization. See base class.

        Returns:
            The (possibly updated) weight datatype.
        """
        # First, call the base class implementation to minimize weight datatype
        wdt = super().minimize_weight_bit_width(model, datatype_only=datatype_only)

        # Minimize threshold datatype if node has thresholds (noActivation=0)
        if self.no_activation == 0 and len(self.onnx_node.input) > 2:
            thresholds = model.get_initializer(self.onnx_node.input[2])
            if not isinstance(thresholds, np.ndarray):
                return wdt
            acc_dt = self.get_accumulator_datatype()
            current_tdt = model.get_tensor_datatype(self.onnx_node.input[2])

            # Only minimize if accumulator and thresholds are integer
            if acc_dt.is_integer() and current_tdt.is_integer():
                if datatype_only:
                    tdt = current_tdt
                else:
                    # Use double precision for intermediate calculations to prevent overflow
                    min_threshold = float(thresholds.min())
                    max_threshold = float(thresholds.max())
                    # Check if accumulator datatype is signed
                    acc_is_signed = acc_dt.signed()
                    if min_threshold < 0:
                        if abs(min_threshold) > max_threshold:
                            tdt = DataType.get_smallest_possible(min_threshold)
                        else:
                            tdt = DataType.get_smallest_possible(-max_threshold - 1)
                    elif acc_is_signed:
                        # If accumulator is signed, use signed threshold datatype
                        # even if thresholds are positive
                        tdt = DataType.get_smallest_possible(-max_threshold - 1)
                    else:
                        tdt = DataType.get_smallest_possible(max_threshold)

                # HLS-specific: ensure threshold datatype is at least as wide as
                # accumulator datatype to prevent truncation during comparison
                if tdt.bitwidth() < acc_dt.bitwidth():
                    tdt = acc_dt

                # Verify thresholds can be expressed with the chosen type
                threshold_tensor = self.get_hw_compatible_threshold_tensor(thresholds)
                if not np.vectorize(tdt.allowed)(threshold_tensor).all():
                    raise FINNUserError(
                        f"{self.onnx_node.name}: thresholds cannot be expressed with type {tdt}"
                    )

                # Update threshold datatype
                model.set_tensor_datatype(self.onnx_node.input[2], tdt)

        return wdt

    def instantiate_ip(self, cmd: list[str]) -> None:
        """Append Vivado IPI commands that instantiate the HLS IP for this node."""
        vlnv = self.get_nodeattr("ip_vlnv")
        node_name = self.onnx_node.name
        if self.mem_mode in ["internal_decoupled", "external_mem", "dynamic"]:
            cmd.append(f"create_bd_cell -type ip -vlnv {vlnv} /{node_name}/{node_name}")
        else:
            cmd.append(f"create_bd_cell -type ip -vlnv {vlnv} {node_name}")

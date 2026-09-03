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

"""HLS backend implementation of the Matrix-Vector-Activation Unit (MVAU)."""

import math
import numpy as np
from onnx import GraphProto, NodeProto
from pathlib import Path
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import cast

from finn.custom_op.fpgadataflow.base.matrixvectoractivation import MVAU, NodeAttrTypes
from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.basic import MAX_ALLOWED_AP_INT_W, is_versal
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.exception import FINNInternalError, FINNUserError

# ONNX i/o tensor shape assumptions for MatrixVectorActivation_hls:
# input 0 is the input tensor, shape (.., i_size) = (..., MW)
# input 1 is the weight tensor, shape (i_size, o_size) = (MW, MH)
# (optional) input 2 is the thresholds tensor, shape (o_size, n_thres)
# output 0 is the output tensor, shape (.., o_size) = (..., MH)
# the ... here can be any shape (representing groups of vectors)

_MEM_MODES = ("internal_embedded", "internal_decoupled", "external")


@register_custom_op
class MVAU_hls(MVAU, HLSBackend):
    """Corresponds to finn-hlslib MatrixVectorActivation_Batch function."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(MVAU.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        # for HLS MVAU default resType to lut
        my_attrs["resType"] = ("s", False, "lut", {"auto", "lut", "dsp"})
        return my_attrs

    def _uses_weight_stream(self) -> bool:
        """Return whether weights arrive as a stream (decoupled/external/MLO)."""
        return self.mem_mode in ("internal_decoupled", "external") or bool(self.mlo_max_iter)

    def lut_estimation(self) -> int:
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
        """Return dsp estimation."""
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
        if self.dynamic_input:
            self.generate_hdl_dynload()
        if self.mem_mode == "internal_decoupled" and not self.mlo_max_iter:
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
        elif self.mlo_max_iter:
            self.generate_hdl_fetch_weights(fpgapart)

    def get_template_param_values(self) -> dict[str, str]:
        """Return the template parameter values according to input, output and weight
        data types.
        """
        ret: dict[str, str] = {}
        inp_hls_str = self.get_input_datatype(0).get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()
        inp_is_binary = self.get_input_datatype(0) == DataType["BINARY"]
        wt_is_binary = self.get_input_datatype(1) == DataType["BINARY"]
        bin_xnor_mode = self.binary_xnor_mode == 1
        if (inp_is_binary or wt_is_binary) and (not bin_xnor_mode):
            raise FINNUserError(
                f"{self.onnx_node.name}: true binary (non-bipolar) inputs not yet supported"
            )
        inp_is_bipolar = self.get_input_datatype(0) == DataType["BIPOLAR"]
        wt_is_bipolar = self.get_input_datatype(1) == DataType["BIPOLAR"]
        # reinterpret inp/wt as bipolar if bin_xnor_mode is iset
        inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
        wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)
        # fill in TSrcI and TWeightI
        # TODO check these with Giulio
        # TODO handle non-bipolar binary inputs
        if inp_is_bipolar and wt_is_bipolar:
            ret["TSrcI"] = "Recast<XnorMul>"
            ret["TWeightI"] = "Identity"
        elif (not inp_is_bipolar) and wt_is_bipolar:
            ret["TSrcI"] = f"Slice<{inp_hls_str}>"
            ret["TWeightI"] = "Recast<Binary>"
        elif inp_is_bipolar and (not wt_is_bipolar):
            ret["TSrcI"] = "Recast<Binary>"
            ret["TWeightI"] = "Identity"
        elif (not inp_is_bipolar) and (not wt_is_bipolar):
            ret["TSrcI"] = f"Slice<{inp_hls_str}>"
            ret["TWeightI"] = "Identity"

        # fill in TDstI
        ret["TDstI"] = f"Slice<{out_hls_str}>"

        return ret

    def global_includes(self) -> None:
        """Return global includes."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "weights.hpp"']
        self.code_gen_dict["$GLOBALS$"] += ['#include "activations.hpp"']

        if self.mem_mode not in _MEM_MODES:
            raise FINNInternalError(
                f"{self.onnx_node.name}: mem_mode must be one of {_MEM_MODES}, got {self.mem_mode}"
            )
        self.code_gen_dict["$GLOBALS$"] += ['#include "mvau.hpp"']
        if self.calc_tmem() != 0:
            # TODO find a better way of checking for no pregenerated thresholds
            self.code_gen_dict["$GLOBALS$"] += ['#include "thresh.h"']

    def defines(self, var: str) -> None:
        """Return defines."""
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
        num_reps = np.prod(self.num_input_vectors)
        self.code_gen_dict["$DEFINES$"] = [
            f"""#define MW1 {self.mw}\n #define MH1 {self.mh}\n
            #define SIMD1 {self.simd}\n #define PE1 {self.pe}\n #define WMEM1 {self.calc_wmem()}\n
            #define TMEM1 {self.calc_tmem()}\n #define numReps {num_reps}"""
        ]
        if self._uses_weight_stream():
            wdt = self.get_input_datatype(1)
            self.code_gen_dict["$DEFINES$"].append(f"#define WP1 {wdt.bitwidth()}\n")

    def read_npy_data(self) -> None:
        """Return read npy data."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype(0)
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
        elem_bits = dtype.bitwidth()
        packed_hls_type = f"ap_uint<{self.get_instream_width(0)}>"
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_in = f"{code_gen_dir}/input_0.npy"
        # note: the innermost dim is reversed for the input
        self.code_gen_dict["$READNPYDATA$"] = [
            f"npy2apintstream<{packed_hls_type}, {elem_hls_type}, {elem_bits}, float>"
            f'("{npy_in}", in0_V, false);'
        ]

        if self._uses_weight_stream():
            wdt = self.get_input_datatype(1)
            w_elem_bits = wdt.bitwidth()
            packed_bits = self.get_instream_width(1)
            if self.dynamic_input:
                packed_bits = packed_bits * self.simd
            w_packed_hls_type = f"ap_uint<{packed_bits}>"
            w_elem_hls_type = wdt.get_hls_datatype_str()
            w_npy_in = f"{code_gen_dir}/input_1.npy"
            self.code_gen_dict["$READNPYDATA$"].append(
                f"npy2apintstream<{w_packed_hls_type}, {w_elem_hls_type}, {w_elem_bits}, float>"
                f'("{w_npy_in}", in1_V, false, numReps);'
            )

    def strm_decl(self) -> None:
        """Return strm decl."""
        self.code_gen_dict["$STREAMDECLARATIONS$"] = [
            f'hls::stream<ap_uint<{self.get_instream_width(0)}>> in0_V ("in0_V");',
            f'hls::stream<ap_uint<{self.get_outstream_width()}>> out0_V ("out0_V");',
        ]

        if self._uses_weight_stream():
            iwidth = self.get_instream_width(1)
            if self.dynamic_input:
                iwidth = iwidth * self.simd
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                f'hls::stream<ap_uint<{iwidth}>> in1_V ("in1_V");'
            )

    def docompute(self) -> None:
        """Return docompute."""
        map_to_hls_mult_style = {
            "auto": "ap_resource_dflt()",
            "lut": "ap_resource_lut()",
            "dsp": "ap_resource_dsp()",
        }
        tmpl_args = self.get_template_param_values()
        tsrci, tdsti, tweighti = tmpl_args["TSrcI"], tmpl_args["TDstI"], tmpl_args["TWeightI"]
        if self.calc_tmem() == 0:
            odtype_hls_str = self.get_output_datatype().get_hls_datatype_str()
            threshs = f"PassThroughActivation<{odtype_hls_str}>()"
        else:
            threshs = "threshs"
        mult_style = map_to_hls_mult_style[self.res_type]
        if self.mem_mode == "internal_embedded":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                "Matrix_Vector_Activate_Batch<MW1, MH1, SIMD1, PE1, 1, "
                f"{tsrci}, {tdsti}, {tweighti}>\n"
                f"                (in0_V, out0_V, weights, {threshs}, numReps, {mult_style});"
            ]
        elif self._uses_weight_stream():
            wdt = self.get_input_datatype(1)
            export_wdt = DataType["BINARY"] if wdt == DataType["BIPOLAR"] else wdt
            wdtype_hls_str = export_wdt.get_hls_datatype_str()
            self.code_gen_dict["$DOCOMPUTE$"] = [
                "Matrix_Vector_Activate_Stream_Batch<MW1, MH1, SIMD1, PE1, "
                f"{tsrci}, {tdsti}, {tweighti}, {wdtype_hls_str} >\n"
                f"                (in0_V, out0_V, in1_V, {threshs}, numReps, {mult_style});"
            ]
        else:
            raise FINNInternalError(
                f"{self.onnx_node.name}: mem_mode must be one of {_MEM_MODES}, got {self.mem_mode}"
            )

    def dataoutstrm(self) -> None:
        """Return dataoutstrm."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
        elem_bits = dtype.bitwidth()
        packed_hls_type = f"ap_uint<{self.get_outstream_width()}>"
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_out = f"{code_gen_dir}/output_0.npy"
        shape_cpp_str = str(self.get_folded_output_shape()).replace("(", "{").replace(")", "}")

        # note: the innermost dim is not reversed for the output
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            f"apintstream2npy<{packed_hls_type}, {elem_hls_type}, {elem_bits}, float>"
            f'(out0_V, {shape_cpp_str}, "{npy_out}", false);'
        ]

    def save_as_npy(self) -> None:
        """Save as npy."""
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def blackboxfunction(self) -> None:
        """Return blackboxfunction."""
        if self.mem_mode == "internal_embedded":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                f"""void {self.onnx_node.name}(
                    hls::stream<ap_uint<{self.get_instream_width(0)}>> &in0_V,
                    hls::stream<ap_uint<{self.get_outstream_width()}>> &out0_V
                    )"""
            ]
        elif self._uses_weight_stream():
            wwidth = self.get_instream_width(1)
            if self.dynamic_input:
                wwidth = wwidth * self.simd
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                f"""void {self.onnx_node.name}(
                    hls::stream<ap_uint<{self.get_instream_width(0)}>> &in0_V,
                    hls::stream<ap_uint<{wwidth}>> &in1_V,
                    hls::stream<ap_uint<{self.get_outstream_width()}>> &out0_V
                    )"""
            ]
        else:
            raise FINNInternalError(
                f"{self.onnx_node.name}: mem_mode must be one of {_MEM_MODES}, got {self.mem_mode}"
            )

    def pragmas(self) -> None:
        """Return pragmas."""
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0_V",
            "#pragma HLS INTERFACE axis port=out0_V",
            "#pragma HLS INTERFACE ap_ctrl_none port=return",
        ]

        if self.mem_mode == "internal_embedded":
            self.code_gen_dict["$PRAGMAS$"].append('#include "params.h"')
            # the weight tensor is ap_uint<simd*prec> [PE][WMEM]
            # partition for parallel access along the PE dimension (dim 1)
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS ARRAY_PARTITION variable=weights.m_weights complete dim=1"
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
        """Return ap int max w."""
        # base class impl (max of inp/out stream widths)
        max_of_io = super().get_ap_int_max_w()
        # internal_decoupled mode weight stream
        weightstream = self.get_instream_width(1)
        if self.dynamic_input:
            weightstream = weightstream * self.simd
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
        """Execute node."""
        mode = self.get_nodeattr("exec_mode")
        dynamic_input = self.dynamic_input
        mem_mode = self.mem_mode
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
                if self.get_input_datatype(0) == DataType["BIPOLAR"]:
                    # store bipolar activations as binary
                    reshaped_input = (reshaped_input + 1) / 2
                    export_idt = DataType["BINARY"]
                else:
                    export_idt = self.get_input_datatype(0)
                # make copy before saving the array
                reshaped_input = reshaped_input.copy()
                np.save(str(Path(code_gen_dir) / "input_0.npy"), reshaped_input)

            if in_ind == 1 and dynamic_input:
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

            if dynamic_input or mem_mode in ["external", "internal_decoupled"] or self.mlo_max_iter:
                wnbits = self.get_instream_width(1)
                if self.dynamic_input:
                    wnbits = wnbits * self.simd
                export_wdt = self.get_input_datatype(1)

                # we have converted bipolar weights to binary for export,
                # so use it as such for weight generation
                if self.get_input_datatype(1) == DataType["BIPOLAR"]:
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
        else:
            raise FINNInternalError(
                f"{self.onnx_node.name}: invalid exec_mode {mode}, "
                f'must be one of ("cppsim", "rtlsim")'
            )

    def minimize_weight_bit_width(self, model: ModelWrapper) -> BaseDataType:
        """Minimize weight and threshold datatypes, with HLS-specific adjustments.

        The HLS implementation uses the threshold datatype for comparisons.
        When the threshold datatype is narrower than the accumulator datatype,
        accumulator values get truncated, which can cause incorrect results.
        To prevent this, ensure threshold datatype is at least as wide as
        accumulator datatype.
        """
        # First, call the base class implementation to minimize weight datatype
        wdt = super().minimize_weight_bit_width(model)

        # Minimize threshold datatype if node has thresholds (noActivation=0)
        if self.no_activation == 0 and len(self.onnx_node.input) > 2:
            thresholds = model.get_initializer(self.onnx_node.input[2])
            if not isinstance(thresholds, np.ndarray):
                return wdt
            acc_dt = self.get_accumulator_datatype()

            # Only minimize if accumulator and thresholds are integer
            if (
                acc_dt.is_integer()
                and model.get_tensor_datatype(self.onnx_node.input[2]).is_integer()
            ):
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
        """Append the HLS IP instantiation TCL to ``cmd``."""
        vlnv = self.get_nodeattr("ip_vlnv")
        node_name = self.onnx_node.name
        if self.mem_mode == "internal_decoupled" or self.mlo_max_iter:
            cmd.append(f"create_bd_cell -type ip -vlnv {vlnv} /{node_name}/{node_name}")
        else:
            cmd.append(f"create_bd_cell -type ip -vlnv {vlnv} {node_name}")

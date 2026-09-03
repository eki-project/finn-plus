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

"""HLS backend implementation of the Vector-Vector Activation Unit (VVAU)."""

import math
import numpy as np
from onnx import GraphProto, NodeProto
from pathlib import Path
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import cast

from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.vectorvectoractivation import VVAU, NodeAttrTypes
from finn.util.basic import is_versal
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.exception import FINNInternalError, FINNUserError

_MEM_MODES = ("internal_embedded", "internal_decoupled", "external")


@register_custom_op
class VVAU_hls(VVAU, HLSBackend):
    """Corresponds to finn-hlslib Vector_Vector_Activate_Batch function."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(VVAU.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def _uses_weight_stream(self) -> bool:
        """Return whether the weights arrive over a stream (in1_V)."""
        return self.mem_mode in ("internal_decoupled", "external")

    def lut_estimation(self) -> int:
        """Calculate resource estimations for LUTs.

        Based on: FINN-R: An End-to-End Deep-Learning Framework for Fast
        Exploration of Quantized Neural Networks - M. Blott, T. B. Preusser,
        N. J. Fraser, G. Gambardella, K. O'Brien, Y. Umuroglu, M. Leeser and
        K. Vissers - 12. Sep 2018.
        """
        # TODO add in/out FIFO contributions
        p = self.pe
        q = self.simd
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
        res_type = self.res_type
        mult_luts = 0 if res_type == "dsp" else q * (2 * math.ceil((w + a) / 6) - 1) * (w + a)
        # adder tree
        addertree_luts = (w + a) * (2 * q - 1)
        # accumulator
        acc_datatype = self.get_accumulator_datatype()
        k_h, k_w = self.kernel
        # if accDataType is not set, then it will default to INT32, which would
        # be a large overestimate in most (if not all) cases. In this scenario,
        # we would use the minimum accumulator as determined by the data types
        # bound, derived in https://arxiv.org/abs/2301.13376
        alpha = math.log(k_h * k_w, 2) + w + a - 1 - int(idt.signed())
        acc_bits = min(
            acc_datatype.bitwidth(),
            np.ceil(alpha + math.log(1 + pow(2, -alpha), 2) + 1),
        )
        acc_luts = acc_bits
        # thresholds and threshold comparators
        thr_luts = 0
        comp_luts = 0
        noact = self.no_activation
        # TODO - add 'ram_style_threshold' node attribute
        if noact == 0:
            odt = self.get_output_datatype()
            b = odt.bitwidth()
            thr_luts = (2**b - 1) * acc_bits * self.calc_tmem() / 64
            comp_luts = (2**b - 1) * acc_bits

        return int(
            c0 + c1 * (p * (mult_luts + addertree_luts + acc_luts + thr_luts + comp_luts)) + c2
        )

    def dsp_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Return dsp estimation."""
        # multiplication
        p = self.pe
        res_type = self.res_type
        w = self.get_input_datatype(1).bitwidth()
        a = self.get_input_datatype(0).bitwidth()
        # TODO: more accurate modelling
        mult_dsp = p * np.ceil((w + a) / 48) if res_type == "dsp" else 0
        return int(mult_dsp)

    def execute_node(
        self, context: dict[str, np.ndarray], graph: GraphProto  # noqa: ARG002
    ) -> None:
        """Execute node."""
        mode = self.get_nodeattr("exec_mode")
        mem_mode = self.mem_mode
        node = self.onnx_node

        # TODO ensure codegen dir exists
        if mode == "cppsim":
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_cppsim"))
        elif mode == "rtlsim":
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
        else:
            raise FINNInternalError(
                f"Invalid value for attribute exec_mode! Is currently set to: {mode} "
                'has to be set to one of the following value ("cppsim", "rtlsim")'
            )

        export_idt = self.get_input_datatype(0)
        # create a npy file for each input of the node (in_ind is input index)
        for in_ind, inputs in enumerate(node.input):
            # it is assumed that the first input of the node is the data input
            # the second input are the weights
            # the third input are the thresholds
            if in_ind == 0:
                if str(context[inputs].dtype) != "float32":
                    raise FINNInternalError("Input datatype is not float32 as expected.")
                expected_inp_shape = self.get_folded_input_shape()
                reshaped_input = context[inputs].reshape(expected_inp_shape)
                if self.get_input_datatype(0) == DataType["BIPOLAR"]:
                    # store bipolar activations as binary
                    reshaped_input = (reshaped_input + 1) / 2
                    export_idt = DataType["BINARY"]
                else:
                    export_idt = self.get_input_datatype(0)
                # make copy before saving the array
                reshaped_input = reshaped_input.copy()
                np.save(Path(code_gen_dir) / f"input_{in_ind}.npy", reshaped_input)
            elif in_ind > 2:
                raise FINNInternalError("Unexpected input found for VectorVectorActivation")

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            # reinterpret binary output as bipolar where needed
            if self.get_output_datatype() == DataType["BIPOLAR"]:
                out = context[node.output[0]]
                out = 2 * out - 1
                context[node.output[0]] = out
            if context[node.output[0]].shape != self.get_normal_output_shape():
                raise FINNInternalError("cppsim did not produce expected output shape")
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width(0)
            inp = npy_to_rtlsim_input(f"{code_gen_dir}/input_0.npy", export_idt, nbits)
            super().reset_rtlsim(sim)

            if mem_mode in ("external", "internal_decoupled"):
                wnbits = self.get_instream_width(1)
                export_wdt = self.get_input_datatype(1)
                # we have converted bipolar weights to binary for export,
                # so use it as such for weight generation
                if self.get_input_datatype(1) == DataType["BIPOLAR"]:
                    export_wdt = DataType["BINARY"]
                wei = npy_to_rtlsim_input(f"{code_gen_dir}/weights.npy", export_wdt, wnbits)
                dim_h, dim_w = self.dim
                num_w_reps = dim_h * dim_w

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
            output = np.asarray([output], dtype=np.float32).reshape(*oshape)
            context[node.output[0]] = output
        else:
            raise FINNInternalError(
                f"Invalid value for attribute exec_mode! Is currently set to: {mode} "
                'has to be set to one of the following value ("cppsim", "rtlsim")'
            )

    def code_generation_ipgen(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:
        """Generate C++ code and tcl script for IP generation."""
        super().code_generation_ipgen(model, fpgapart, clk)
        mem_mode = self.mem_mode
        if mem_mode == "internal_decoupled":
            if (
                self.ram_style == "ultra"
                and not is_versal(fpgapart)
                and self.runtime_writeable_weights != 1
            ):
                raise FINNUserError(
                    "Layer with URAM weights must have runtime_writeable_weights=1 "
                    "if an Ultrascale device is targeted."
                )
            self.generate_hdl_memstream(fpgapart)

    def get_template_param_values(self) -> dict[str, str]:
        """Return the template parameter values according to input, output and weight data types."""
        ret: dict[str, str] = {}
        inp_hls_str = self.get_input_datatype(0).get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()
        inp_is_binary = self.get_input_datatype(0) == DataType["BINARY"]
        wt_is_binary = self.get_input_datatype(1) == DataType["BINARY"]
        bin_xnor_mode = self.binary_xnor_mode == 1
        if (inp_is_binary or wt_is_binary) and (not bin_xnor_mode):
            raise FINNUserError("True binary (non-bipolar) inputs not yet supported")
        inp_is_bipolar = self.get_input_datatype(0) == DataType["BIPOLAR"]
        wt_is_bipolar = self.get_input_datatype(1) == DataType["BIPOLAR"]
        # reinterpret inp/wt as bipolar if bin_xnor_mode is set
        inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
        wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)
        # fill in TSrcI and TWeightI
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
                'Please set mem_mode to "internal_embedded", "internal_decoupled", or "external", '
                "currently no other parameter value is supported!"
            )
        if self.calc_tmem() != 0:
            self.code_gen_dict["$GLOBALS$"] += ['#include "thresh.h"']

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Return defines."""
        dim_h, dim_w = self.dim
        num_reps = 1 * dim_h * dim_w
        k_h, k_w = self.kernel
        inner_prod_dim = k_h * k_w
        mem_mode = self.mem_mode

        self.code_gen_dict["$DEFINES$"] = [
            f"#define Channels1 {self.channels}\n"
            f" #define InnerProdDim {inner_prod_dim}\n"
            "\n"
            f"            #define SIMD1 {self.simd}\n"
            f" #define PE1 {self.pe}\n"
            f" #define numReps {num_reps}"
        ]
        if mem_mode in ("internal_decoupled", "external"):
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
        packed_bits = self.get_instream_width(0)
        packed_hls_type = f"ap_uint<{packed_bits}>"
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_in = f"{code_gen_dir}/input_0.npy"
        self.code_gen_dict["$READNPYDATA$"] = []
        # note: the innermost dim is reversed for the input
        self.code_gen_dict["$READNPYDATA$"].append(
            f"npy2apintstream<{packed_hls_type}, {elem_hls_type}, {elem_bits}, {npy_type}>("
            f'"{npy_in}", in0_V, false);'
        )

        mem_mode = self.mem_mode
        if mem_mode in ("internal_decoupled", "external"):
            wdt = self.get_input_datatype(1)
            elem_bits = wdt.bitwidth()
            packed_bits = self.get_instream_width(1)
            packed_hls_type = f"ap_uint<{packed_bits}>"
            elem_hls_type = wdt.get_hls_datatype_str()
            npy_type = "float"
            npy_in = f"{code_gen_dir}/weights.npy"

            self.code_gen_dict["$READNPYDATA$"].append(
                f"npy2apintstream<{packed_hls_type}, {elem_hls_type}, {elem_bits}, {npy_type}>("
                f'"{npy_in}", in1_V, false, numReps);'
            )

    def strm_decl(self) -> None:
        """Return strm decl."""
        mem_mode = self.mem_mode
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            f'hls::stream<ap_uint<{self.get_instream_width(0)}>> in0_V ("in0_V");'
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            f'hls::stream<ap_uint<{self.get_outstream_width()}>> out0_V ("out0_V");'
        )
        if mem_mode in ("internal_decoupled", "external"):
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                f'hls::stream<ap_uint<{self.get_instream_width(1)}>> in1_V ("in1_V");'
            )

    def docompute(self) -> None:
        """Return docompute."""
        mem_mode = self.mem_mode
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

        if mem_mode == "internal_embedded":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                f"Vector_Vector_Activate_Batch<Channels1, InnerProdDim, SIMD1, PE1, 1, "
                f"{tsrci}, {tdsti}, {tweighti}>\n"
                f"                (in0_V, out0_V, weights, {threshs}, numReps, {mult_style});"
            ]
        elif mem_mode in ("internal_decoupled", "external"):
            wdt = self.get_input_datatype(1)
            export_wdt = DataType["BINARY"] if wdt == DataType["BIPOLAR"] else wdt
            wdtype_hls_str = export_wdt.get_hls_datatype_str()
            self.code_gen_dict["$DOCOMPUTE$"] = [
                f"Vector_Vector_Activate_Stream_Batch<Channels1, InnerProdDim, SIMD1, PE1, 1, "
                f"{tsrci}, {tdsti}, {tweighti}, {wdtype_hls_str}>\n"
                f"                (in0_V, out0_V, in1_V, {threshs}, numReps, {mult_style});"
            ]
        else:
            raise FINNInternalError(
                'Please set mem_mode to "internal_embedded", "internal_decoupled", or "external", '
                "currently no other parameter value is supported!"
            )

    def dataoutstrm(self) -> None:
        """Return dataoutstrm."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_outstream_width()
        packed_hls_type = f"ap_uint<{packed_bits}>"
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_out = f"{code_gen_dir}/output_0.npy"
        shape = self.get_folded_output_shape()
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        # note: the innermost dim is not reversed for the output
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            f"apintstream2npy<{packed_hls_type}, {elem_hls_type}, {elem_bits}, {npy_type}>("
            f'out0_V, {shape_cpp_str}, "{npy_out}", false);'
        ]

    def save_as_npy(self) -> None:
        """Save as npy."""
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def blackboxfunction(self) -> None:
        """Return blackboxfunction."""
        mem_mode = self.mem_mode
        if mem_mode == "internal_embedded":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                f"""void {self.onnx_node.name}(
                hls::stream<ap_uint<{self.get_instream_width(0)}>> &in0_V,
                hls::stream<ap_uint<{self.get_outstream_width()}>> &out0_V
                )"""
            ]
        elif mem_mode in ("internal_decoupled", "external"):
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                f"""void {self.onnx_node.name}(
                    hls::stream<ap_uint<{self.get_instream_width(0)}>> &in0_V,
                    hls::stream<ap_uint<{self.get_instream_width(1)}>> &in1_V,
                    hls::stream<ap_uint<{self.get_outstream_width()}>> &out0_V
                    )"""
            ]
        else:
            raise FINNInternalError(
                'Please set mem_mode to "internal_embedded" or "internal_decoupled", '
                "currently no other parameter value is supported!"
            )

    def pragmas(self) -> None:
        """Return pragmas."""
        mem_mode = self.mem_mode
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0_V"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

        if mem_mode == "internal_embedded":
            self.code_gen_dict["$PRAGMAS$"].append('#include "params.h"')
            # the weight tensor is ap_uint<ch*prec> [PE][WMEM]
            # partition for parallel access along the PE dimension (dim 1)
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS ARRAY_PARTITION variable=weights.m_weights complete dim=1"
            )
        elif mem_mode in ("internal_decoupled", "external"):
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=in1_V")
        else:
            raise FINNInternalError(
                'Please set mem_mode to "internal_embedded", "internal_decoupled", or external, '
                "currently no other parameter value is supported!"
            )

        if self.calc_tmem() != 0:
            # TODO find a better way of checking for no pregenerated thresholds
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=1"
            )
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=3"
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
                    # If accumulator is signed, use a signed threshold datatype
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
                    raise FINNInternalError(f"Thresholds can't be expressed with type {tdt!s}")

                # Update threshold datatype
                model.set_tensor_datatype(self.onnx_node.input[2], tdt)

        return wdt

    def instantiate_ip(self, cmd: list[str]) -> None:
        """Return instantiate ip."""
        # instantiate the HLS IP
        vlnv = self.get_nodeattr("ip_vlnv")
        node_name = self.onnx_node.name
        if self.mem_mode == "internal_decoupled":
            cmd.append(f"create_bd_cell -type ip -vlnv {vlnv} /{node_name}/{node_name}")
        else:
            cmd.append(f"create_bd_cell -type ip -vlnv {vlnv} {node_name}")

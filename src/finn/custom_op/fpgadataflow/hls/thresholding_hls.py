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

"""HLS backend implementation for neural network thresholding operations.

This module provides the Thresholding_hls class which implements hardware-accelerated
thresholding/activation functions using High-Level Synthesis (HLS) for FPGA deployment.
Supports multiple memory modes, runtime weight loading, and various data types.
"""

import numpy as np
import textwrap
from math import ceil, log2
from onnx import NodeProto
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import roundup_to_integer_multiple
from typing import Any

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.thresholding import Thresholding
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    numpy_to_hls_code,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)
from finn.util.settings import get_settings

# ONNX i/o tensor shape assumptions for Thresholding:
# input 0 is the input tensor, shape (..., NumChannels)
# input 1 is the threshold tensor, shape (NumChannels, n_thres)
# output 0 is the output tensor, shape (..., NumChannels) - same as input
# the ... here can be any shape (representing groups of vectors)


class Thresholding_hls(Thresholding, HLSBackend):
    """Class that corresponds to finn-hls Thresholding_Batch function."""

    def __init__(self, onnx_node: NodeProto, **kwargs: Any) -> None:
        """Initialize the Thresholding_hls layer.

        Parameters
        ----------
        onnx_node : NodeProto
            ONNX node representing this operation
        **kwargs : dict
            Additional arguments passed to parent classes
        """
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(
        self,
    ) -> dict[str, tuple[str, bool, str, set[str]] | tuple[str, bool, int, set[int]]]:
        """Get the types and default values for node attributes.

        Returns
        -------
        dict
            Dictionary mapping attribute names to their type specifications
        """
        my_attrs = {
            # memory mode for the thresholds
            # internal_embedded -- embedded thresholds
            # internal_decoupled -- default, streaming thresholds with  streamer packaged inside IP
            "mem_mode": (
                "s",
                False,
                "internal_decoupled",
                {"internal_embedded", "internal_decoupled"},
            ),
            # string defining memory type
            "ram_style": ("s", False, "distributed", {"distributed", "block"}),
            # (mem_mode = internal_decoupled only) whether weights (thresholds) will be
            # writable through an AXI-lite interface during runtime
            # 1 for enabled, 0 for disabled.
            # see finn-rtllib/memstream/doc/README for more about the memory
            # address map used for writable weights
            # IMPORTANT: After using AXI lite to either read or write the weights,
            # always "flush" the accelerator by first passing a dummy input
            # vector through the accelerator. This will get rid of any old
            # weight data from the weight FIFOs.
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
        }
        my_attrs.update(Thresholding.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def bram_estimation(self) -> int:
        """Calculate BRAM cost if resource set to BRAM.

        Returns
        -------
        int
            Number of BRAM blocks required.
        """
        style = self.get_nodeattr("ram_style")
        pe = self.get_nodeattr("PE")
        idt = self.get_input_datatype(0)
        bitwidth = idt.bitwidth()
        tmem = self.calc_tmem()

        if style == "block" and tmem > 1:
            return ceil(bitwidth * pe / 16) * ceil(tmem / 1024)
        return 0

    def lut_estimation(self) -> int:
        """Calculate LUT cost, taking memory resource type into account.

        Returns
        -------
        int
            Number of LUTs required for comparators and optional LUTRAM.
        """
        # TODO add in/out FIFO contributions
        style = self.get_nodeattr("ram_style")
        p = self.get_nodeattr("PE")
        idt = self.get_input_datatype(0)
        a = idt.bitwidth()
        tmem = self.calc_tmem()
        # cost of comparators
        comparator_cost = a * p
        # cost of LUTRAM
        lutram_cost = p * a * ceil(tmem / 64) if style == "distributed" and tmem > 1 else 0
        # total cost
        return comparator_cost + lutram_cost

    def get_ap_int_max_w(self) -> int:
        """Get the maximum ap_int width used in this layer.

        Returns
        -------
        int
            Maximum bitwidth of any ap_int used in the operator.
        """
        ap_int_max_w = HLSBackend.get_ap_int_max_w(self)
        if self.get_nodeattr("mem_mode") == "internal_decoupled":
            weightstream = self.get_instream_width(1)
            ap_int_max_w = max([weightstream, ap_int_max_w])
        return ap_int_max_w

    def code_generation_ipgen(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:
        """Generate C++ code and tcl script for IP generation.

        Parameters
        ----------
        model : ModelWrapper
            The ONNX model wrapper.
        fpgapart : str
            Target FPGA part name.
        clk : float
            Clock period in nanoseconds.
        """
        super().code_generation_ipgen(model, fpgapart, clk)
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_decoupled":
            self.generate_hdl_memstream(fpgapart)

    def get_template_param_values(self) -> dict[str, str]:
        """Return the template parameter values according to input, output and weight
        data types.

        Returns
        -------
        dict[str, str]
            Dictionary of template parameter names to their values.
        """
        ret = {}
        inp_hls_str = self.get_input_datatype(0).get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()
        # fill in TSrcI
        ret["TSrcI"] = f"Slice<{inp_hls_str}>"
        # fill in TDstI
        ret["TDstI"] = f"Slice<{out_hls_str}>"

        return ret

    def make_weight_file(
        self, weights: np.ndarray, weight_file_mode: str, weight_file_name: str
    ) -> None:
        """Produce a file containing given weights (thresholds) in appropriate
        format for this layer. This file can be used for either synthesis or
        run-time reconfig of weights.

        Parameters
        ----------
        weights : np.ndarray
            Numpy array with weights to be put into the file.
        weight_file_mode : str
            One of {hls_header, decoupled_verilog_dat, decoupled_runtime, decoupled_npy}.
        weight_file_name : str
            Filename for the weight file to be generated.
        """
        threshold_tensor = self.get_hw_compatible_threshold_tensor(weights)
        tdt = self.get_input_datatype(1)
        assert np.vectorize(tdt.allowed)(
            threshold_tensor
        ).all(), f"Thresholds can't be expressed with type {tdt!s}"
        if weight_file_mode == "hls_header":
            # save thresholds in thresh.h
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
            with Path(weight_file_name).open("w") as f_thresh:
                f_thresh.write(
                    f"static ThresholdsActivation<"
                    f"{self.calc_tmem()},"
                    f"{self.get_nodeattr('PE')},"
                    f"{threshold_tensor.shape[-1]},"
                    f"{tdt_hls},"
                    f"{odt_hls},"
                    f"{self.get_nodeattr('ActVal')},"
                    f"comp::less_equal<{tdt_hls}, {tdt_hls}>> threshs = "
                )
                f_thresh.write(thresholds_hls_code)
        elif "decoupled" in weight_file_mode:
            # streaming thresholds need to be organized differently
            # (1, pe, tmem, n_thres_steps) -> (1, tmem, pe, n_thres_steps)
            decoupled_thres = np.transpose(threshold_tensor, (0, 2, 1, 3))
            # TODO add flips/reversals as needed here
            # (1, tmem, pe, n_thres_steps) -(1, tmem, pe * n_thres_steps)
            pe = self.get_nodeattr("PE")
            n_thres_steps = self.get_nodeattr("numSteps")
            decoupled_thres_pe_flipped = np.flip(decoupled_thres, axis=-2)
            decoupled_thres = decoupled_thres.reshape(1, -1, pe * n_thres_steps)
            decoupled_thres = decoupled_thres.copy()
            decoupled_thres_pe_flipped = decoupled_thres_pe_flipped.reshape(
                1, -1, pe * n_thres_steps
            )
            decoupled_thres_pe_flipped = decoupled_thres_pe_flipped.copy()

            if weight_file_mode == "decoupled_npy":
                # save weight stream into npy for cppsim
                np.save(weight_file_name, decoupled_thres)
            elif weight_file_mode == "decoupled_verilog_dat":
                # convert weight values into hexstring
                weight_width = self.get_instream_width(1)
                # pad to nearest 4 bits to get hex strings
                weight_width_padded = roundup_to_integer_multiple(weight_width, 4)
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    decoupled_thres_pe_flipped, tdt, weight_width_padded, prefix=""
                )
                weight_stream = weight_tensor_pe_flipped.flatten()
                weight_stream = weight_stream.copy()
                with Path(weight_file_name).open("w") as f:
                    for val in weight_stream:
                        f.write(val + "\n")
            elif weight_file_mode == "decoupled_runtime":
                # memstream axi-lite interface will map each mem line to
                # one or multiple 32-bit words
                weight_width = self.get_instream_width(1)
                words_per_memwidth = 2 ** ceil(log2(weight_width / 32))
                if words_per_memwidth < 1:
                    words_per_memwidth = 1
                weight_width_padded = words_per_memwidth * 32
                # first, pack and ensure padding to 32 bits
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    decoupled_thres_pe_flipped, tdt, weight_width_padded, prefix=""
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
                raise Exception("Decoupled weight export not yet implemented")
        else:
            raise Exception("Unknown weight_file_mode")

    def generate_params(self, model: ModelWrapper, path: str) -> None:
        """Generate parameter files for thresholds.

        Parameters
        ----------
        model : ModelWrapper
            The ONNX model wrapper containing initializers.
        path : str
            Code generation directory path.
        """
        code_gen_dir = path

        # Check input and threshold datatypes
        idt = self.get_input_datatype(0)
        tdt = self.get_input_datatype(1)
        if idt.is_integer() and not tdt.is_integer():
            raise ValueError(
                "Thresholds must be converted to integers for integer inputs "
                "using RoundAndClipThresholds transform before code generation."
            )
        if not idt.is_integer() and tdt.is_integer():
            raise ValueError("Floating-point inputs and integer thresholds are not supported.")

        thresholds = model.get_initializer(self.onnx_node.input[1])
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_embedded":
            # save thresholds in thresh.h
            weight_filename = f"{code_gen_dir}/thresh.h"
            self.make_weight_file(thresholds, "hls_header", weight_filename)
        elif mem_mode == "internal_decoupled":
            # save internal_decoupled weights for cppsim
            weight_filename_sim = f"{code_gen_dir}/thresholds.npy"
            self.make_weight_file(thresholds, "decoupled_npy", weight_filename_sim)
            # also save weights as Verilog .dat file
            weight_filename_rtl = f"{code_gen_dir}/memblock.dat"
            self.make_weight_file(thresholds, "decoupled_verilog_dat", weight_filename_rtl)
        else:
            raise Exception("Unrecognized mem_mode")

    def execute_node(self, context: dict, graph: object) -> None:  # noqa: ARG002
        """Execute this node in the given context.

        Parameters
        ----------
        context : dict
            Execution context containing input/output tensors.
        graph : onnx.GraphProto
            The ONNX graph (unused but required by interface).
        """
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node

        # TODO ensure codegen dir exists
        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(
                f"Invalid value for attribute exec_mode! Is currently set to: {mode} "
                'has to be set to one of the following value ("cppsim", "rtlsim")'
            )

        # create a npy file fore each input of the node (in_ind is input index)
        for in_ind, inputs in enumerate(node.input):
            # it is assumed that the first input of the node is the data input
            # the second input are the weights
            # the third input are the thresholds
            if in_ind == 0:
                assert str(context[inputs].dtype) in [
                    "float32",
                    "float16",
                ], """Input datatype is
                not float32 or float16 as expected."""
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
                np.save(
                    Path(code_gen_dir) / f"input_{in_ind}.npy",
                    reshaped_input,
                )
            elif in_ind > 2:
                raise Exception("Unexpected input found for Thresholding_Batch")

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
            oshape = self.get_normal_output_shape()
            assert context[node.output[0]].shape == oshape, """Output shape is not as expected"""
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width(0)
            inp = npy_to_rtlsim_input(f"{code_gen_dir}/input_0.npy", export_idt, nbits)
            super().reset_rtlsim(sim)
            if self.get_nodeattr("mem_mode") == "internal_decoupled":
                wnbits = self.get_instream_width(1)
                export_wdt = self.get_input_datatype(1)
                wei = npy_to_rtlsim_input(f"{code_gen_dir}/thresholds.npy", export_wdt, wnbits)
                num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))
                io_dict = {
                    "inputs": {"in0": inp, "in1": wei * num_w_reps},
                    "outputs": {"out0": []},
                }
            elif self.get_nodeattr("mem_mode") == "internal_embedded":
                io_dict = {
                    "inputs": {"in0": inp},
                    "outputs": {"out0": []},
                }
            else:
                raise Exception("Unrecognized mem_mode")
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
            raise Exception(
                f"Invalid value for attribute exec_mode! Is currently set to: {mode} "
                'has to be set to one of the following value ("cppsim", "rtlsim")'
            )

    def global_includes(self) -> None:
        """Generate list of global C++ includes."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "activations.hpp"']
        if self.get_nodeattr("mem_mode") == "internal_embedded":
            self.code_gen_dict["$GLOBALS$"] += ['#include "thresh.h"']

    def defines(self, var: object) -> None:  # noqa: ARG002
        """Generate C++ defines for template parameters.

        Parameters
        ----------
        var : object
            Unused parameter for compatibility with base class.
        """
        num_reps = 1
        num_input_vectors = list(self.get_nodeattr("numInputVectors"))
        total_spatial_size = int(np.prod(num_input_vectors))

        self.code_gen_dict["$DEFINES$"] = [
            f"#define NumChannels1 {self.get_nodeattr('NumChannels')}\n"
            f" #define PE1 {self.get_nodeattr('PE')}\n"
            f" #define numReps {num_reps}\n"
            f" #define ImgDim1 {total_spatial_size}"
        ]
        if self.get_nodeattr("mem_mode") == "internal_decoupled":
            self.code_gen_dict["$DEFINES$"].append(f"#define ActVal1 {self.get_nodeattr('ActVal')}")
            self.code_gen_dict["$DEFINES$"].append(
                f"#define ThresType1 {self.get_input_datatype(1).get_hls_datatype_str()}"
            )
            self.code_gen_dict["$DEFINES$"].append(
                f"#define NumSteps1 {self.get_nodeattr('numSteps')}"
            )

    def read_npy_data(self) -> None:
        """Generate C++ code for reading NPY data for simulation."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype(0)
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width(0)
        packed_hls_type = f"ap_uint<{packed_bits}>"
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "half" if elem_hls_type == "half" else "float"
        npy_in = "%s/input_0.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"] = []
        # note: the innermost dim is reversed for the input
        self.code_gen_dict["$READNPYDATA$"].append(
            f'npy2apintstream<{packed_hls_type}, {elem_hls_type}, {elem_bits}, {npy_type}>("'
            f'{npy_in}", in0_V, false);'
        )
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_decoupled":
            tdt = self.get_input_datatype(1)
            elem_bits = tdt.bitwidth()
            packed_bits = self.get_instream_width(1)
            packed_hls_type = f"ap_uint<{packed_bits}>"
            elem_hls_type = tdt.get_hls_datatype_str()
            npy_type = "half" if elem_hls_type == "half" else "float"
            npy_in = "%s/thresholds.npy" % code_gen_dir

            self.code_gen_dict["$READNPYDATA$"].append(
                f'npy2apintstream<{packed_hls_type}, {elem_hls_type}, {elem_bits}, {npy_type}>("'
                f'{npy_in}", in1_V, false, ImgDim1);'
            )

    def strm_decl(self) -> None:
        """Generate C++ stream declarations."""
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            f'hls::stream<ap_uint<{self.get_instream_width(0)}>> in0_V ("in0_V");'
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            f'hls::stream<ap_uint<{self.get_outstream_width()}>> out0_V ("out0_V");'
        )
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_decoupled":
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                f'hls::stream<ap_uint<{self.get_instream_width(1)}>> in1_V ("in1_V");'
            )

    def docompute(self) -> None:
        """Generate C++ code for the main computation."""
        tmpl_args = self.get_template_param_values()
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_embedded":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                f"Thresholding_Batch<ImgDim1, NumChannels1, PE1, "
                f"{tmpl_args['TSrcI']}, {tmpl_args['TDstI']}>"
                "(in0_V, out0_V, threshs, numReps);"
            ]
        elif mem_mode == "internal_decoupled":
            # note that numReps is set to 1 in the invocation below, since
            # - for cppsim the repetition comes from the threshold stream reader+input
            # - for synth the unit runs continuously anyway (ap_ctrl_none)
            self.code_gen_dict["$DOCOMPUTE$"] = [
                f"Thresholding_Stream_Batch<ImgDim1, NumChannels1, PE1, {tmpl_args['TSrcI']}, "
                f"{tmpl_args['TDstI']}, ActVal1, ThresType1, NumSteps1>"
                "(in0_V, out0_V, in1_V, numReps);"
            ]
        else:
            raise Exception("Unrecognized mem_mode")

    def dataoutstrm(self) -> None:
        """Generate C++ code for writing output data stream."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_outstream_width()
        packed_hls_type = f"ap_uint<{packed_bits}>"
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "half" if elem_hls_type == "half" else "float"
        npy_out = "%s/output_0.npy" % code_gen_dir
        shape = self.get_folded_output_shape()
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        # note: the innermost dim is not reversed for the output
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            f"apintstream2npy<{packed_hls_type}, {elem_hls_type}, {elem_bits}, {npy_type}>("
            f'out0_V, {shape_cpp_str}, "{npy_out}", false);'
        ]

    def blackboxfunction(self) -> None:
        """Generate C++ black box function signature."""
        if self.get_nodeattr("mem_mode") == "internal_embedded":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                f"void {self.onnx_node.name}("
                f"hls::stream<ap_uint<{self.get_instream_width(0)}>> &in0_V, "
                f"hls::stream<ap_uint<{self.get_outstream_width()}>> &out0_V)"
            ]
        elif self.get_nodeattr("mem_mode") == "internal_decoupled":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                f"void {self.onnx_node.name}("
                f"hls::stream<ap_uint<{self.get_instream_width(0)}>> &in0_V, "
                f"hls::stream<ap_uint<{self.get_instream_width(1)}>> &in1_V, "
                f"hls::stream<ap_uint<{self.get_outstream_width()}>> &out0_V)"
            ]
        else:
            raise Exception("Unrecognized mem_mode")

    def pragmas(self) -> None:
        """Generate HLS pragmas for synthesis."""
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0_V"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

        if self.get_nodeattr("mem_mode") == "internal_embedded":
            # the threshold tensor is acc_type [PE][TMEM][N_THRES]
            # partition for parallel access along PE and N_THRES
            # dimensions (dims 1 and 3)
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=1"
            )
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=3"
            )
            # set resource type
            ram_style = self.get_nodeattr("ram_style")
            pe = self.get_nodeattr("PE")
            ich = self.get_nodeattr("NumChannels")
            # if PE less than NumChannels, assign cores according to ram_style;
            # otherwise if PE == NumChannels, Vivado HLS will unroll to FFs
            if pe < ich:
                if ram_style == "distributed":
                    self.code_gen_dict["$PRAGMAS$"].append(
                        "#pragma HLS RESOURCE variable=threshs.m_thresholds core=ROM_2P_LUTRAM"
                    )
                elif ram_style == "block":
                    self.code_gen_dict["$PRAGMAS$"].append(
                        "#pragma HLS RESOURCE variable=threshs.m_thresholds core=ROM_2P_BRAM"
                    )
                else:
                    raise Exception(
                        f"Invalid value for attribute ram_style! Is currently set to: {ram_style} "
                        'has to be set to one of ("block", "distributed")'
                    )
        elif self.get_nodeattr("mem_mode") == "internal_decoupled":
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=in1_V")

    def code_generation_ipi(self) -> list[str]:
        """Generate Vivado IPI tcl commands for block design integration.

        Returns
        -------
        list[str]
            List of tcl commands for IPI block design generation.
        """
        source_target = f"./ip/verilog/rtl_ops/{self.onnx_node.name}"
        cmd = [f"file mkdir {source_target}"]
        # add streamer if needed
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_decoupled":
            node_name = self.onnx_node.name
            runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1
            # create a hierarchy for this layer, with the same port names
            clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
            rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
            dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0][0]
            din_name = self.get_verilog_top_module_intf_names()["s_axis"][0][0]
            cmd.append(f"create_bd_cell -type hier {node_name}")
            cmd.append(f"create_bd_pin -dir I -type clk /{node_name}/{clk_name}")
            cmd.append(f"create_bd_pin -dir I -type rst /{node_name}/{rst_name}")
            cmd.append(
                f"create_bd_intf_pin -mode Master "
                f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/{dout_name}"
            )
            cmd.append(
                f"create_bd_intf_pin -mode Slave "
                f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/{din_name}"
            )
            # instantiate the hls ip
            cmd.append(
                f"create_bd_cell -type ip -vlnv "
                f"{self.get_nodeattr('ip_vlnv')} /{node_name}/{node_name}"
            )
            # instantiate a streamer and connect it to the IP
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            from pathlib import Path

            finn_rtllib = Path(get_settings().finn_rtllib)
            axi_dir = finn_rtllib / "axi/hdl/"
            ms_rtllib_dir = finn_rtllib / "memstream/hdl/"
            file_suffix = "_memstream_wrapper.v"
            # automatically find memstream verilog component in code generation directory
            code_gen_path = Path(code_gen_dir)
            for fname in code_gen_path.iterdir():
                if fname.name.endswith(file_suffix):
                    strm_tmpl = fname.name
            strm_tmpl_name = strm_tmpl[:-2]
            sourcefiles = [
                str(code_gen_path / strm_tmpl),
                str(axi_dir / "axilite.sv"),
                str(ms_rtllib_dir / "memstream_axi.sv"),
                str(ms_rtllib_dir / "memstream.sv"),
            ]
            for f in sourcefiles:
                cmd += [f"add_files -copy_to {source_target} -norecurse {f}"]
            strm_inst = f"{node_name}_wstrm"
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
            # 2x clock is not used for decoupled thresholds
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
            if runtime_writable:
                # expose axi lite interface for writeable weights
                axilite_name = self.get_verilog_top_module_intf_names()["axilite"][0]
                cmd.append(
                    f"create_bd_intf_pin -mode Slave "
                    f"-vlnv xilinx.com:interface:aximm_rtl:1.0 /{node_name}/{axilite_name}"
                )
                cmd.append(
                    f"connect_bd_intf_net [get_bd_intf_pins {node_name}/{axilite_name}] "
                    f"[get_bd_intf_pins {node_name}/{strm_inst}/{axilite_name}]"
                )
                # TODO calculate and pass in segment size here
                cmd.append("assign_bd_address")
            cmd.append("save_bd_design")
        elif mem_mode == "internal_embedded":
            # base class impl sufficient for internal_embedded mode
            return super().code_generation_ipi()
        else:
            raise Exception("Unrecognized mem_mode for Thresholding_Batch")
        return cmd

    def get_op_and_param_counts(self) -> dict[str, int]:
        """Get operation and parameter counts for this layer.

        Returns
        -------
        dict[str, int]
            Dictionary mapping parameter type to count.
        """
        ret_dict = {}
        weight_bits = self.get_input_datatype(1).bitwidth()
        out_features = self.get_nodeattr("NumChannels")
        num_steps = self.get_nodeattr("numSteps")
        # thresholds are called weights in this layer
        thres_param_type = f"param_threshold_{weight_bits}b"
        thres_count = out_features * num_steps
        ret_dict[thres_param_type] = thres_count
        return ret_dict

    def ipgen_extra_directives(self) -> list[str]:
        """Return a list of extra tcl directives for HLS synthesis.

        Returns
        -------
        list[str]
            List of tcl directives for HLS IP generation.
        """
        return ["config_compile -pipeline_style frp"]

    def derive_characteristic_fxns(
        self, period: int, override_rtlsim_dict: dict | None = None  # noqa: ARG002
    ) -> None:
        """Derive characteristic functions for performance estimation.

        Parameters
        ----------
        period : int
            Clock period in nanoseconds
        override_rtlsim_dict : dict | None
            Optional dictionary to override RTL simulation parameters.

        Returns
        -------
        None
        """
        n_inps = np.prod(self.get_folded_input_shape()[:-1])
        io_dict = {
            "inputs": {
                "in0": [0 for i in range(n_inps)],
            },
            "outputs": {"out0": []},
        }
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode in ["internal_decoupled", "external"]:
            n_weight_inps = self.calc_tmem()
            num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))
            io_dict["inputs"]["in1"] = [0 for i in range(num_w_reps * n_weight_inps)]
        super().derive_characteristic_fxns(period, override_rtlsim_dict=io_dict)

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

"""HLS implementation of the Thresholding (multi-threshold activation) layer.

Provides the Thresholding_hls class, which maps the multi-threshold activation onto
the finn-hlslib ``Thresholding_Batch`` / ``Thresholding_Stream_Batch`` kernels, with
embedded or decoupled (streamed) threshold memories and optional runtime-writeable
thresholds.
"""

import numpy as np
import textwrap
from math import ceil, log2
from onnx import NodeProto
from pathlib import Path
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import roundup_to_integer_multiple
from typing import TYPE_CHECKING, Any, cast

from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.thresholding import NodeAttrTypes, Thresholding
from finn.util.basic import get_vivado_version, is_versal
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    numpy_to_hls_code,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from onnx import GraphProto

# ONNX i/o tensor shape assumptions for Thresholding:
# input 0 is the input tensor, shape (..., NumChannels)
# input 1 is the threshold tensor, shape (NumChannels, n_thres)
# output 0 is the output tensor, shape (..., NumChannels) - same as input
# the ... here can be any shape (representing groups of vectors)


@register_custom_op
class Thresholding_hls(Thresholding, HLSBackend):
    """Class that corresponds to finn-hls Thresholding_Batch function."""

    def __init__(self, onnx_node: NodeProto, **kwargs: Any) -> None:
        """Initialize the Thresholding_hls layer.

        Args:
            onnx_node: ONNX node representing this operation.
            **kwargs: Additional arguments passed to parent classes.
        """
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return node attribute types, adding mem_mode, ram_style and runtime_writeable_weights.

        Returns:
            Dictionary mapping attribute names to their type specifications.
        """
        my_attrs: NodeAttrTypes = {
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
            "ram_style": ("s", False, "distributed", {"distributed", "block", "ultra"}),
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

    def _threshold_mem_width(self) -> int:
        """Return the bit width of one threshold memory word (PE * threshold bits * numSteps)."""
        weight_bits = self.get_input_datatype(1).bitwidth()
        return self.pe * weight_bits * self.num_steps

    def bram_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Calculate BRAM cost if resource set to BRAM.

        Args:
            fpgapart: Target FPGA part (unused).

        Returns:
            Number of BRAM blocks required.
        """
        style = self.get_nodeattr("ram_style")
        mem_width = self._threshold_mem_width()
        tmem = self.calc_tmem()

        if style == "block" and tmem > 1:
            return ceil(mem_width / 16) * ceil(tmem / 1024)
        return 0

    def uram_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Calculate URAM cost if resource set to URAM.

        Args:
            fpgapart: Target FPGA part (unused).

        Returns:
            Number of URAM blocks required.
        """
        style = self.get_nodeattr("ram_style")
        tmem = self.calc_tmem()
        if style == "ultra" and tmem > 1:
            return ceil(self._threshold_mem_width() / 72) * ceil(tmem / 4096)
        return 0

    def bram_efficiency_estimation(self, fpgapart: str) -> float:
        """Return the fraction of estimated BRAM capacity actually used by thresholds.

        Args:
            fpgapart: Target FPGA part, forwarded to the BRAM estimation.

        Returns:
            Ratio of threshold bits to allocated BRAM bits (1 if no BRAM is used).
        """
        bram16_est = self.bram_estimation(fpgapart)
        if bram16_est == 0:
            return 1
        wbits = self._threshold_mem_width() * self.calc_tmem()
        bram16_est_capacity = bram16_est * 18 * 1024
        return wbits / bram16_est_capacity

    def uram_efficiency_estimation(self, fpgapart: str) -> float:
        """Return the fraction of estimated URAM capacity actually used by thresholds.

        Args:
            fpgapart: Target FPGA part, forwarded to the URAM estimation.

        Returns:
            Ratio of threshold bits to allocated URAM bits (1 if no URAM is used).
        """
        # TODO: Versal URAM supports flexible bit widths (9/18/36/72) unlike
        # UltraScale+ which only supports 72-bit. This could improve efficiency
        # for narrow data types on Versal devices.
        uram_est = self.uram_estimation(fpgapart)
        if uram_est == 0:
            return 1
        wbits = self._threshold_mem_width() * self.calc_tmem()
        uram_est_capacity = uram_est * 72 * 4096
        return wbits / uram_est_capacity

    def lut_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Calculate LUT cost, taking memory resource type into account.

        Args:
            fpgapart: Target FPGA part (unused).

        Returns:
            Number of LUTs required for comparators and optional LUTRAM.
        """
        # TODO add in/out FIFO contributions
        style = self.get_nodeattr("ram_style")
        p = self.pe
        idt = self.get_input_datatype(0)
        a = idt.bitwidth()
        tmem = self.calc_tmem()
        # cost of comparators
        comparator_cost = a * p
        # cost of LUTRAM
        if style == "distributed" and tmem > 1:
            lutram_cost = self._threshold_mem_width() * ceil(tmem / 64)
        else:
            lutram_cost = 0
        # total cost
        return comparator_cost + lutram_cost

    def get_ap_int_max_w(self) -> int:
        """Return the maximum ap_int width, including the threshold stream in decoupled mode.

        Returns:
            Maximum bitwidth of any ap_int used in the operator.
        """
        ap_int_max_w = HLSBackend.get_ap_int_max_w(self)
        if self.get_nodeattr("mem_mode") == "internal_decoupled":
            weightstream = self.get_instream_width(1)
            ap_int_max_w = max([weightstream, ap_int_max_w])
        return ap_int_max_w

    def code_generation_ipgen(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:
        """Generate C++ code and tcl script for IP generation.

        Args:
            model: The ONNX model wrapper.
            fpgapart: Target FPGA part name.
            clk: Clock period in nanoseconds.

        Raises:
            FINNUserError: If URAM thresholds are requested in a configuration the
                target device or tool version cannot support.
        """
        super().code_generation_ipgen(model, fpgapart, clk)
        mem_mode = self.get_nodeattr("mem_mode")
        if self.get_nodeattr("ram_style") == "ultra":
            if mem_mode == "internal_embedded":
                if not is_versal(fpgapart):
                    raise FINNUserError(
                        "URAM thresholds with internal_embedded mode are not supported "
                        "on non-Versal devices, as URAM cannot be initialized from bitfile."
                    )
                vivado_version = get_vivado_version()
                if vivado_version is not None and vivado_version < (2024, 2):
                    raise FINNUserError(
                        "URAM thresholds with internal_embedded mode require Vitis HLS 2024.2 "
                        "or newer because older versions cannot initialize URAM-backed arrays."
                    )
            elif mem_mode == "internal_decoupled" and not is_versal(fpgapart):
                runtime_writeable = self.get_nodeattr("runtime_writeable_weights")
                if runtime_writeable != 1:
                    raise FINNUserError(
                        "Layer with URAM thresholds must have runtime_writeable_weights=1 "
                        "if Ultrascale device is targeted."
                    )
        if mem_mode == "internal_decoupled":
            self.generate_hdl_memstream(fpgapart)

    def get_template_param_values(self) -> dict[str, str]:
        """Return the template parameter values according to input and output data types.

        Returns:
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
        """Produce a file containing given weights (thresholds) in appropriate format.

        This file can be used for either synthesis or run-time reconfig of weights.

        Args:
            weights: Numpy array with weights to be put into the file.
            weight_file_mode: One of {hls_header, decoupled_verilog_dat,
                decoupled_runtime, decoupled_npy}.
            weight_file_name: Filename for the weight file to be generated.
        """
        threshold_tensor = self.get_hw_compatible_threshold_tensor(weights)
        tdt = self.get_input_datatype(1)
        if not np.vectorize(tdt.allowed)(threshold_tensor).all():
            raise FINNInternalError(f"Thresholds can't be expressed with type {tdt!s}")
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
                # the run of spaces before "=" is part of the generated header
                f_thresh.write(
                    f"static ThresholdsActivation<"
                    f"{self.calc_tmem()},"
                    f"{self.get_nodeattr('PE')},"
                    f"{threshold_tensor.shape[-1]},"
                    f"{tdt_hls},"
                    f"{odt_hls},"
                    f"{self.get_nodeattr('ActVal')},"
                    f"comp::less_equal<{tdt_hls}, {tdt_hls}>> threshs                 = "
                )
                f_thresh.write(thresholds_hls_code)
        elif "decoupled" in weight_file_mode:
            # streaming thresholds need to be organized differently
            # (1, pe, tmem, n_thres_steps) -> (1, tmem, pe, n_thres_steps)
            decoupled_thres = np.transpose(threshold_tensor, (0, 2, 1, 3))
            # TODO add flips/reversals as needed here
            # (1, tmem, pe, n_thres_steps) -(1, tmem, pe * n_thres_steps)
            pe = self.pe
            n_thres_steps = self.num_steps
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
                words_per_memwidth = max(words_per_memwidth, 1)
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
                raise FINNInternalError("Decoupled weight export not yet implemented")
        else:
            raise FINNInternalError("Unknown weight_file_mode")

    def generate_params(
        self,
        model: ModelWrapper,
        path: str | Path,
        fpgapart: str | None = None,  # noqa: ARG002
    ) -> None:
        """Write threshold parameter files (header or decoupled npy/dat) into the given path.

        Args:
            model: The ONNX model wrapper containing initializers.
            path: Code generation directory path.
            fpgapart: Target FPGA part (unused).
        """
        code_gen_dir = path

        # Check input and threshold datatypes
        idt = self.get_input_datatype(0)
        tdt = self.get_input_datatype(1)
        if idt.is_integer() and not tdt.is_integer():
            raise FINNUserError(
                "Thresholds must be converted to integers for integer inputs "
                "using RoundAndClipThresholds transform before code generation."
            )
        if not idt.is_integer() and tdt.is_integer():
            raise FINNUserError("Floating-point inputs and integer thresholds are not supported.")

        thresholds = model.get_initializer(self.onnx_node.input[1])
        if not isinstance(thresholds, np.ndarray):
            raise FINNInternalError(f"{self.onnx_node.name}: threshold initializer is missing")
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
            raise FINNInternalError("Unrecognized mem_mode")

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute the node via cppsim or rtlsim, including streamed thresholds if decoupled.

        Args:
            context: Execution context containing input/output tensors.
            graph: The ONNX graph (unused but required by interface).
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
                f"Invalid value for attribute exec_mode! Is currently set to: {mode} "
                'has to be set to one of the following value ("cppsim", "rtlsim")'
            )

        export_idt = self.get_input_datatype(0)
        # create a npy file fore each input of the node (in_ind is input index)
        for in_ind, inputs in enumerate(node.input):
            # it is assumed that the first input of the node is the data input
            # the second input are the weights
            # the third input are the thresholds
            if in_ind == 0:
                if str(context[inputs].dtype) not in ("float32", "float16"):
                    raise FINNInternalError("Input datatype is not float32 or float16 as expected.")
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
                raise FINNInternalError("Unexpected input found for Thresholding_Batch")

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
            if context[node.output[0]].shape != oshape:
                raise FINNInternalError("Output shape is not as expected")
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width(0)
            inp = npy_to_rtlsim_input(f"{code_gen_dir}/input_0.npy", export_idt, nbits)
            super().reset_rtlsim(sim)
            if self.get_nodeattr("mem_mode") == "internal_decoupled":
                wnbits = self.get_instream_width(1)
                export_wdt = self.get_input_datatype(1)
                wei = npy_to_rtlsim_input(f"{code_gen_dir}/thresholds.npy", export_wdt, wnbits)
                num_w_reps = int(np.prod(cast("list[int]", self.get_nodeattr("numInputVectors"))))
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
                raise FINNInternalError("Unrecognized mem_mode")
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

    def global_includes(self) -> None:
        """Populate the C++ includes for the generated HLS code."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "activations.hpp"']
        if self.get_nodeattr("mem_mode") == "internal_embedded":
            self.code_gen_dict["$GLOBALS$"] += ['#include "thresh.h"']

    # TODO check and add whatever missing
    def defines(self, var: str) -> None:  # noqa: ARG002
        """Emit C++ macro definitions for channel count, folding, steps and datatypes.

        Args:
            var: Unused parameter for compatibility with base class.
        """
        num_reps = 1
        num_input_vectors = list(cast("list[int]", self.get_nodeattr("numInputVectors")))
        total_spatial_size = int(np.prod(num_input_vectors))

        # the blank line and indentation before ImgDim1 are part of the generated code
        self.code_gen_dict["$DEFINES$"] = [
            (
                f"#define NumChannels1 {self.get_nodeattr('NumChannels')}\n"
                f" #define PE1 {self.get_nodeattr('PE')}\n"
                f" #define numReps {num_reps}\n"
                "\n"
                f"               #define ImgDim1 {total_spatial_size}"
            )
        ]
        if self.get_nodeattr("mem_mode") == "internal_decoupled":
            self.code_gen_dict["$DEFINES$"].append(f"#define ActVal1 {self.act_val:d}")
            self.code_gen_dict["$DEFINES$"].append(
                f"#define ThresType1 {self.get_input_datatype(1).get_hls_datatype_str()}"
            )
            self.code_gen_dict["$DEFINES$"].append(f"#define NumSteps1 {self.num_steps:d}")

    def read_npy_data(self) -> None:
        """Emit C++ code that reads the input (and streamed threshold) numpy data for cppsim."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype(0)
        packed_bits = self.get_instream_width(0)
        packed_hls_type = f"ap_uint<{packed_bits}>"
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "half" if elem_hls_type == "half" else "float"
        npy_in = f"{code_gen_dir}/input_0.npy"
        self.code_gen_dict["$READNPYDATA$"] = []
        # note: the innermost dim is reversed for the input
        self.code_gen_dict["$READNPYDATA$"].append(
            f'npy2apintstream<{packed_hls_type}, {elem_hls_type}, {npy_type}>("'
            f'{npy_in}", in0_V, false);'
        )
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_decoupled":
            tdt = self.get_input_datatype(1)
            packed_bits = self.get_instream_width(1)
            packed_hls_type = f"ap_uint<{packed_bits}>"
            elem_hls_type = tdt.get_hls_datatype_str()
            npy_type = "half" if elem_hls_type == "half" else "float"
            npy_in = f"{code_gen_dir}/thresholds.npy"

            self.code_gen_dict["$READNPYDATA$"].append(
                f'npy2apintstream<{packed_hls_type}, {elem_hls_type}, {npy_type}>("'
                f'{npy_in}", in1_V, false, ImgDim1);'
            )

    def strm_decl(self) -> None:
        """Emit C++ stream declarations for the input, threshold and output streams."""
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
        """Emit the C++ call to the finn-hlslib Thresholding kernel for the selected mem_mode."""
        tmpl_args = self.get_template_param_values()
        mem_mode = self.get_nodeattr("mem_mode")
        # the line break and indentation inside the call are part of the generated code
        if mem_mode == "internal_embedded":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                (
                    f"Thresholding_Batch<ImgDim1, NumChannels1, PE1, "
                    f"{tmpl_args['TSrcI']}, {tmpl_args['TDstI']}>\n"
                    "                (in0_V, out0_V, threshs, numReps);"
                )
            ]
        elif mem_mode == "internal_decoupled":
            # note that numReps is set to 1 in the invocation below, since
            # - for cppsim the repetition comes from the threshold stream reader+input
            # - for synth the unit runs continuously anyway (ap_ctrl_none)
            self.code_gen_dict["$DOCOMPUTE$"] = [
                (
                    f"Thresholding_Stream_Batch<ImgDim1, NumChannels1, PE1, {tmpl_args['TSrcI']}, "
                    f"{tmpl_args['TDstI']}, ActVal1, ThresType1, NumSteps1>\n"
                    "                (in0_V, out0_V, in1_V, numReps);"
                )
            ]
        else:
            raise FINNInternalError("Unrecognized mem_mode")

    def dataoutstrm(self) -> None:
        """Emit C++ code that writes the output stream to a numpy file for cppsim."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
        packed_bits = self.get_outstream_width()
        packed_hls_type = f"ap_uint<{packed_bits}>"
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "half" if elem_hls_type == "half" else "float"
        npy_out = f"{code_gen_dir}/output_0.npy"
        shape = self.get_folded_output_shape()
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        # note: the innermost dim is not reversed for the output
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            (
                f"apintstream2npy<{packed_hls_type}, {elem_hls_type}, {npy_type}>("
                f'out0_V, {shape_cpp_str}, "{npy_out}", false);'
            )
        ]

    def blackboxfunction(self) -> None:
        """Emit the top-level HLS function signature for the selected mem_mode."""
        # the line breaks and indentation in the signature are part of the generated code
        indent = " " * 20
        if self.get_nodeattr("mem_mode") == "internal_embedded":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                (
                    f"void {self.onnx_node.name}("
                    f"hls::stream<ap_uint<{self.get_instream_width(0)}>> &in0_V,\n"
                    f"{indent}hls::stream<ap_uint<{self.get_outstream_width()}>> &out0_V\n"
                    f"{indent})"
                )
            ]
        elif self.get_nodeattr("mem_mode") == "internal_decoupled":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                (
                    f"void {self.onnx_node.name}("
                    f"hls::stream<ap_uint<{self.get_instream_width(0)}>> &in0_V,\n"
                    f"{indent}hls::stream<ap_uint<{self.get_instream_width(1)}>> &in1_V,\n"
                    f"{indent}hls::stream<ap_uint<{self.get_outstream_width()}>> &out0_V\n"
                    f"{indent})"
                )
            ]
        else:
            raise FINNInternalError("Unrecognized mem_mode")

    def pragmas(self) -> None:
        """Emit HLS interface and resource pragmas for the selected mem_mode."""
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
            pe = self.pe
            ich = self.num_channels
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
                elif ram_style == "ultra":
                    self.code_gen_dict["$PRAGMAS$"].append(
                        "#pragma HLS BIND_STORAGE variable=threshs.m_thresholds "
                        "type=RAM_S2P impl=URAM"
                    )
                else:
                    raise FINNUserError(
                        f"Invalid value for attribute ram_style! Is currently set to: {ram_style} "
                        'has to be set to one of ("block", "distributed", "ultra")'
                    )
        elif self.get_nodeattr("mem_mode") == "internal_decoupled":
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=in1_V")

    def code_generation_ipi(self) -> list[str]:
        """Construct the TCL for node instantiation in Vivado IPI, adding a memstream if decoupled.

        Returns:
            List of tcl commands for IPI block design generation.
        """
        cmd = []
        # add streamer if needed
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_decoupled":
            node_name = self.onnx_node.name
            runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1
            # create a hierarchy for this layer, with the same port names
            intf_names = self.get_verilog_top_module_intf_names()
            clk_name = intf_names["clk"][0]
            rst_name = intf_names["rst"][0]
            dout_name = intf_names["m_axis"][0][0]
            din_name = intf_names["s_axis"][0][0]
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
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
            finn_rtllib = Path(get_settings().finn_rtllib)
            axi_dir = finn_rtllib / "axi/hdl"
            ms_rtllib_dir = finn_rtllib / "memstream/hdl"
            file_suffix = "_memstream_wrapper.v"
            # automatically find memstream verilog component in code generation directory
            code_gen_path = Path(code_gen_dir)
            strm_tmpl = next(
                (f.name for f in code_gen_path.iterdir() if f.name.endswith(file_suffix)), None
            )
            if strm_tmpl is None:
                raise FINNInternalError(
                    f"{self.onnx_node.name}: no '*{file_suffix}' file found in {code_gen_dir}"
                )
            strm_tmpl_name = strm_tmpl[:-2]
            sourcefiles = [
                code_gen_path / strm_tmpl,
                axi_dir / "axilite.sv",
                ms_rtllib_dir / "memstream_axi.sv",
                ms_rtllib_dir / "memstream.sv",
            ]
            cmd += [f"add_files -norecurse {f}" for f in sourcefiles]
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
                axilite_name = intf_names["axilite"][0]
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
            raise FINNInternalError("Unrecognized mem_mode for Thresholding_Batch")
        return cmd

    def get_op_and_param_counts(self) -> dict[str, int]:
        """Return the number of threshold parameters keyed by their bit width.

        Returns:
            Dictionary mapping parameter type to count.
        """
        weight_bits = self.get_input_datatype(1).bitwidth()
        # thresholds are called weights in this layer
        thres_param_type = f"param_threshold_{weight_bits}b"
        thres_count = self.num_channels * self.num_steps
        return {thres_param_type: thres_count}

    def ipgen_extra_directives(self) -> list[str]:
        """Return a list of extra tcl directives for HLS synthesis.

        Returns:
            List of tcl directives for HLS IP generation.
        """
        return ["config_compile -pipeline_style frp"]

    def minimize_weight_bit_width(
        self, model: ModelWrapper, datatype_only: bool = False
    ) -> BaseDataType:
        """Minimize threshold datatype, with HLS-specific adjustments.

        The HLS implementation uses the threshold datatype for comparisons.
        When the threshold datatype is narrower than the input datatype,
        input values get truncated, which can cause incorrect results.
        To prevent this, ensure threshold datatype is at least as wide as
        input datatype.

        Args:
            model: The model containing this node and its threshold initializer.
            datatype_only: If True, skip value-based minimization. See base class.

        Returns:
            The (possibly widened) threshold datatype.
        """
        tdt = super().minimize_weight_bit_width(model, datatype_only=datatype_only)

        idt = self.get_input_datatype(0)
        if not idt.is_integer() or not tdt.is_integer():
            return tdt

        # Widen threshold to match input width to prevent truncation
        if tdt.bitwidth() < idt.bitwidth():
            # Use input datatype to ensure no truncation
            new_tdt = idt
            thresholds = model.get_initializer(self.onnx_node.input[1])
            if not isinstance(thresholds, np.ndarray):
                raise FINNInternalError(f"{self.onnx_node.name}: threshold initializer is missing")
            threshold_tensor = self.get_hw_compatible_threshold_tensor(thresholds)
            if not np.vectorize(new_tdt.allowed)(threshold_tensor).all():
                raise FINNInternalError(f"Thresholds can't be expressed with type {new_tdt!s}")
            self.set_nodeattr("weightDataType", new_tdt.name)
            model.set_tensor_datatype(self.onnx_node.input[1], new_tdt)
            return new_tdt

        return tdt

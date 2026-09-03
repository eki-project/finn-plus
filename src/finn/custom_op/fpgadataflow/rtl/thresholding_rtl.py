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

"""RTL implementation of thresholding activation.

This module provides an RTL-based implementation of thresholding activations
for quantization and activation functions in FPGA dataflow architectures.
"""

import math
import numpy as np
import shutil
from onnx import NodeProto
from pathlib import Path
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import roundup_to_integer_multiple
from typing import TYPE_CHECKING, Any, cast

from finn.custom_op.fpgadataflow.base.thresholding import NodeAttrTypes, Thresholding
from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.memutil import get_memutil_alternatives, mem_primitives_versal
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from onnx import GraphProto


@register_custom_op
class Thresholding_rtl(Thresholding, RTLBackend):
    """Class that corresponds to finn-rtllib 'thresholding' function."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize the RTL thresholding activation node."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return the dictionary of attribute names and their types for this node."""
        my_attrs: NodeAttrTypes = {
            # memory depth triggers for threshold storage
            "depth_trigger_uram": ("i", False, 0),
            "depth_trigger_bram": ("i", False, 0),
            # enable uniform thres optimization
            # doesn't actually do anything yet, only
            # for resource estimations
            "uniform_thres": ("i", False, 0, {0, 1}),
            # enable deep pipelining for easier timing closure
            # setting to 0 may save some FFs but otherwise leave on
            "deep_pipeline": ("i", False, 1, {0, 1}),
        }
        my_attrs.update(Thresholding.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def get_pe_mem_geometries(self) -> list[tuple[int, float]]:
        """Return a list of (bitwidth, depth) for PE memory configurations.

        Used in resource estimation: for each bitwidth, the depth is the number
        of thresholds that can be stored in a single memory block.
        """
        pe = self.pe
        wdt_bits = self.get_input_datatype(1).bitwidth()
        odt_bits = self.get_output_datatype().bitwidth()
        cf = self.num_channels / pe
        is_uniform = self.get_nodeattr("uniform_thres")
        if is_uniform:
            return [(odt_bits - x, cf * (2**x)) for x in range(1, odt_bits)]
        return [(wdt_bits, cf * 2**x) for x in range(odt_bits)]

    def get_memory_estimate(self) -> dict[str, int]:
        """Return the memory estimate for this node."""
        res_dict: dict[str, int] = {}
        depth_trigger_bram = cast("int", self.get_nodeattr("depth_trigger_bram"))
        depth_trigger_uram = cast("int", self.get_nodeattr("depth_trigger_uram"))
        pe = self.pe
        for mem_cfg in self.get_pe_mem_geometries():
            (_width, depth) = mem_cfg
            primitives = mem_primitives_versal
            if depth_trigger_bram != 0 or depth_trigger_uram != 0:
                if depth >= depth_trigger_bram and depth < depth_trigger_uram:
                    primitives = {k: v for (k, v) in mem_primitives_versal.items() if "BRAM" in k}
                elif depth >= depth_trigger_uram:
                    primitives = {k: v for (k, v) in mem_primitives_versal.items() if "URAM" in k}
            alts = get_memutil_alternatives(cast("tuple[int, int]", mem_cfg), primitives)
            primary_alt = alts[0]
            res_type = primary_alt[0].split("_")[0]
            res_count, _eff, _waste = primary_alt[1]
            res_dict[res_type] = res_dict.get(res_type, 0) + pe * res_count
        return res_dict

    def bram_estimation(self) -> int:
        """Return the number of BRAMs required for this node."""
        return self.get_memory_estimate().get("BRAM", 0)

    def uram_estimation(self) -> int:
        """Return the number of URAMs required for this node."""
        return self.get_memory_estimate().get("URAM", 0)

    def lut_estimation(self) -> int:
        """Return the number of LUTs required for this node."""
        return self.get_memory_estimate().get("LUTRAM", 0)

    def get_all_meminit_filenames(self, abspath: bool = False) -> list[str]:
        """Return a list of all .dat memory initializer files used for this node."""
        dat_files: list[str] = []
        t_path = cast("str", self.get_nodeattr("code_gen_dir_ipgen")) if abspath else "."
        pe = self.pe
        o_bitwidth = self.get_output_datatype().bitwidth()
        for stage in range(o_bitwidth):
            for pe_value in range(pe):
                dat_files.append(f"{t_path}/{self.onnx_node.name}_threshs_{pe_value}_{stage}.dat")
        return dat_files

    def prepare_codegen_rtl_values(self, model: ModelWrapper) -> dict[str, list[str]]:
        """Produce dictionary values to replace their key value(s) in the RTL templates."""
        code_gen_dict: dict[str, list[str]] = {}

        t_path = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))

        if not self.get_nodeattr("mlo_max_iter"):
            self.generate_params(model, t_path)

        bias = self.act_val  # activation bias value
        pe = self.pe
        num_channels = self.num_channels  # number of channels
        n_thres_steps = self.num_steps
        idt = self.get_input_datatype(0)
        wdt = self.get_input_datatype(1)

        if idt.is_integer() and not wdt.is_integer():
            raise FINNUserError(
                "Thresholds must be converted to integers for integer inputs "
                "using RoundAndClipThresholds transform before code generation."
            )
        if not idt.is_integer() and wdt.is_integer():
            raise FINNUserError("Non-integer inputs and integer thresholds are not supported.")
        if idt.is_fixed_point() and not wdt.is_fixed_point():
            raise FINNUserError(
                "Fixed-point inputs and floating-point thresholds are not supported."
            )
        if wdt.is_fixed_point() and not idt.is_fixed_point():
            raise FINNUserError(
                "Floating-point inputs and fixed-point thresholds are not supported."
            )
        if wdt.is_fixed_point() and idt.is_fixed_point():
            # scale_factor() is only defined on fixed-point DataTypes, guarded above
            wdt_sf = cast("Any", wdt).scale_factor()
            idt_sf = cast("Any", idt).scale_factor()
            if wdt_sf < idt_sf:
                raise FINNUserError(
                    "Fixed-point thresholds have more fractional bits than input. "
                    "Run RoundAndClipThresholds to reduce threshold fractional bits."
                )
            if wdt_sf > idt_sf:
                raise FINNUserError(
                    "Fixed-point inputs and with more fractional bits "
                    "than thresholds are not supported."
                )

        # If a single threshold value is found, set num_channels to PE
        thresholds_shape = model.get_tensor_shape(self.onnx_node.input[1])
        if thresholds_shape is not None and thresholds_shape[0] == 1:
            num_channels = pe

        code_gen_dict["$THRESHOLDS_PATH$"] = [f'"./{self.onnx_node.name}_"']

        # Identify the module name
        code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"] = [self.get_verilog_top_module_name()]
        # Set the top module name - AXI wrapper
        code_gen_dict["$TOP_MODULE$"] = code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"]

        # Identify the module variables
        i_bitwidth = idt.bitwidth()

        code_gen_dict["$N$"] = [str(n_thres_steps)]  # number of needed thresholds
        code_gen_dict["$WT$"] = [str(wdt.bitwidth())]  # threshold precision
        code_gen_dict["$WI$"] = [str(i_bitwidth)]  # input precision
        code_gen_dict["$C$"] = [str(num_channels)]  # number of channels
        code_gen_dict["$BIAS$"] = [str(bias)]  # activation bias value
        code_gen_dict["$PE$"] = [str(pe)]  # requires C = M*PE
        mlo_max_iter = self.get_nodeattr("mlo_max_iter")
        code_gen_dict["$SETS$"] = [str(mlo_max_iter)] if mlo_max_iter else [str(1)]

        # Is the input datatype signed or unsigned?
        # The thresholding core needs to know this when comparing weights to inputs
        code_gen_dict["$SIGNED$"] = [str(1)] if self.get_input_datatype(0).signed() else [str(0)]

        # Is the input datatype floating-point?
        if self.get_input_datatype(0) in ["FLOAT32", "FLOAT16"]:
            code_gen_dict["$FPARG$"] = [str(1)]
        else:
            code_gen_dict["$FPARG$"] = [str(0)]

        if bias >= 0:
            o_bits = math.ceil(math.log2(n_thres_steps + bias + 1))
        else:
            o_bits = 1 + math.ceil(
                math.log2(-bias if -bias >= (n_thres_steps + 1) / 2 else n_thres_steps + bias + 1)
            )
        code_gen_dict["$O_BITS$"] = [str(int(o_bits))]

        rt_weights = self.get_nodeattr("runtime_writeable_weights")
        code_gen_dict["$USE_AXILITE$"] = [str(rt_weights)]

        depth_trigger_uram = self.get_nodeattr("depth_trigger_uram")
        depth_trigger_bram = self.get_nodeattr("depth_trigger_bram")
        deep_pipeline = self.get_nodeattr("deep_pipeline")
        code_gen_dict["$DEPTH_TRIGGER_URAM$"] = [str(depth_trigger_uram)]
        code_gen_dict["$DEPTH_TRIGGER_BRAM$"] = [str(depth_trigger_bram)]
        code_gen_dict["$DEEP_PIPELINE$"] = [str(deep_pipeline)]
        return code_gen_dict

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        """Thresholding binary search RTL file list."""
        if abspath:
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen")) + "/"
            rtllib_dir = str(Path(get_settings().finn_rtllib) / "thresholding/hdl") + "/"
            axi_dir = str(Path(get_settings().finn_rtllib) / "axi/hdl") + "/"
        else:
            code_gen_dir = ""
            rtllib_dir = ""
            axi_dir = ""

        return [
            axi_dir + "axilite.sv",
            rtllib_dir + "thresholding.sv",
            rtllib_dir + "thresholding_axi.sv",
            code_gen_dir + cast("str", self.get_nodeattr("gen_top_module")) + ".v",
        ]

    def generate_hdl(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:  # noqa: ARG002
        """Prepare HDL files from templates for synthesis."""
        # Generate a dictionary of values to put in RTL template
        code_gen_dict = self.prepare_codegen_rtl_values(model)

        # Retrieve the destination directory for the final RTL files
        code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))

        # Set the 'gen_top_module' attribute for use later by xsi and IPI generation
        self.set_nodeattr("gen_top_module", code_gen_dict["$TOP_MODULE$"][0])
        axi_dir = Path(get_settings().finn_rtllib) / "axi/hdl/"
        rtlsrc = Path(get_settings().finn_rtllib) / "thresholding/hdl"
        template_wrapper = (rtlsrc / "thresholding_template_wrapper.v").read_text()
        for key, lines in code_gen_dict.items():
            # transform list into long string separated by '\n'
            template_wrapper = template_wrapper.replace(key, "\n".join(lines))
        (code_gen_dir / (cast("str", self.get_nodeattr("gen_top_module")) + ".v")).write_text(
            template_wrapper
        )

        for sv_file in ("thresholding.sv", "thresholding_axi.sv"):
            shutil.copy(str(rtlsrc / sv_file), str(code_gen_dir))
        shutil.copy(str(axi_dir / "axilite.sv"), str(code_gen_dir))

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        # i.e. during the HLSSynthIP() transformation
        self.set_nodeattr("ipgen_path", str(code_gen_dir))
        self.set_nodeattr("ip_path", str(code_gen_dir))

    def execute_node(self, context: dict[str, np.ndarray], graph: "GraphProto") -> None:
        """Execute this thresholding node via C++ or RTL simulation."""
        mode = self.get_nodeattr("exec_mode")
        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
        if mode == "cppsim":
            Thresholding.execute_node(self, context, graph)
        elif mode == "rtlsim":
            node = self.onnx_node
            export_idt = self.get_input_datatype(0)
            # create a npy file for each input of the node (in_ind is input index)
            for in_ind, inputs in enumerate(node.input):
                # it is assumed that the first input of the node is the data input
                # the second input are the thresholds
                if in_ind == 0:
                    if str(context[inputs].dtype) not in ("float32", "float16"):
                        raise FINNInternalError(
                            "Input datatype is not float32 or float16 as expected."
                        )
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
                    raise FINNInternalError("Unexpected input found for Thresholding_rtl")

            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(f"{code_gen_dir}/input_0.npy", export_idt, nbits)
            io_dict = {
                "inputs": {"in0": rtlsim_inp},
                "outputs": {"out0": []},
            }
            super().reset_rtlsim(sim)
            self.rtlsim_multi_io(sim, io_dict)
            super().close_rtlsim(sim)
            rtlsim_output = io_dict["outputs"]["out0"]

            # Manage output data
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = f"{code_gen_dir}/output.npy"
            out_shape = self.get_folded_output_shape()

            rtlsim_output_to_npy(
                rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )

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

    def code_generation_ipi(self) -> list[str]:
        """Construct the TCL commands for node instantiation as an RTL block."""
        rtl_file_list = self.get_rtl_file_list()
        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
        source_target = f"./ip/verilog/rtl_ops/{self.onnx_node.name}"
        cmd = [f"file mkdir {source_target}"]

        for rtl_file in rtl_file_list:
            full_path = Path(code_gen_dir) / rtl_file
            cmd.append(f"add_files -copy_to {source_target} -norecurse {full_path}")

        # Create an RTL block, not an IP core (-type ip)
        cmd.append(
            f"create_bd_cell -type module -reference "
            f"{self.get_nodeattr('gen_top_module')} {self.onnx_node.name}"
        )

        return cmd

    def get_verilog_top_module_intf_names(self) -> dict[str, list[tuple[str, int]] | list[str]]:
        """Get Verilog top module interface names for this node."""
        intf_names = super().get_verilog_top_module_intf_names()
        if self.get_nodeattr("runtime_writeable_weights") == 1:
            intf_names["axilite"] = ["s_axilite"]

        return intf_names

    def generate_params(self, model: ModelWrapper, path: str | Path) -> None:
        """Generate threshold parameter files for RTL implementation."""
        thresholds = model.get_initializer(self.onnx_node.input[1])
        if not isinstance(thresholds, np.ndarray):
            raise FINNInternalError(f"{self.onnx_node.name}: threshold initializer is missing")
        rt_weights = self.get_nodeattr("runtime_writeable_weights")
        file_name = f"{path}/memblock.dat"
        if rt_weights:
            self.make_weight_file(thresholds, "decoupled_runtime", file_name)
        self.make_weight_file(thresholds, "internal_embedded", file_name)

    def make_weight_file(
        self, weights: np.ndarray, weight_file_mode: str, weight_file_name: str
    ) -> None:
        """Produce a file containing the given thresholds in the appropriate format.

        This file can be used for either synthesis or run-time reconfig of weights.
        """
        path = Path(weight_file_name).parent
        if str(path) in ("", "."):
            path = Path.cwd()
        thresholds = weights
        pe = self.pe
        num_channels = self.num_channels  # number of channels
        o_bitwidth = self.get_output_datatype().bitwidth()
        expected_thresholds = 2**o_bitwidth - 1
        n_thres_steps = self.num_steps
        wdt = self.get_input_datatype(1)
        if expected_thresholds > n_thres_steps:
            thresholds = np.pad(
                thresholds,
                ((0, 0), (0, expected_thresholds - n_thres_steps)),
                mode="constant",
                constant_values=(0, 0),
            )

        if weight_file_mode == "decoupled_runtime":
            # If a single threshold value is found, broadcast the value
            if thresholds.shape[0] == 1:
                thresholds = np.broadcast_to(thresholds, (pe, expected_thresholds))
                num_channels = pe
            width_padded = roundup_to_integer_multiple(thresholds.shape[1], 2**o_bitwidth)
            thresh_padded = np.zeros((thresholds.shape[0], width_padded))
            thresh_padded[: thresholds.shape[0], :expected_thresholds] = thresholds
            thresh_stream: list[str] = []
            bw_hexdigit = roundup_to_integer_multiple(wdt.bitwidth(), 32)
            padding = np.zeros(width_padded, dtype=np.int32)

            chan_ind = 0
            cf = num_channels // pe
            for _fold in range(cf):
                for c in range(2 ** (pe - 1).bit_length()):
                    if (c == 0 or c % pe != 0) and c < pe:
                        for t in thresh_padded[chan_ind]:
                            t_packed = pack_innermost_dim_as_hex_string(
                                [t], wdt, bw_hexdigit, prefix=""
                            ).item()
                            thresh_stream.append(t_packed)
                        chan_ind += 1
                    else:
                        for _z in padding:
                            t_packed = pack_innermost_dim_as_hex_string(
                                [_z], wdt, bw_hexdigit, prefix=""
                            ).item()
                            thresh_stream.append(t_packed)
            Path(weight_file_name).write_text("".join(f"{val}\n" for val in thresh_stream))
        elif weight_file_mode == "internal_embedded":
            # add dummy dimension as final dimension (that's what gets packed with next call)
            t_expand = np.expand_dims(thresholds, axis=-1)
            bw_hexdigit = roundup_to_integer_multiple(wdt.bitwidth(), 4)
            t_packed = pack_innermost_dim_as_hex_string(t_expand, wdt, bw_hexdigit, prefix="")
            # If a single threshold value is found, broadcast the value
            if t_packed.shape[0] == 1:
                t_packed = np.broadcast_to(t_packed, (pe, expected_thresholds))
                num_channels = pe
            channel_fold = int(num_channels / pe)

            for stage in range(o_bitwidth):
                sn = o_bitwidth - stage - 1
                for pe_value in range(pe):
                    thresh_file = path / f"{self.onnx_node.name}_threshs_{pe_value}_{stage}.dat"
                    threshs = np.zeros([channel_fold * (2**stage)], dtype="object")
                    for ch in range(channel_fold):
                        for i in range(2**stage):
                            threshs[(ch << stage) + i] = t_packed[ch * pe + pe_value][
                                (i << (o_bitwidth - stage)) + 2**sn - 1
                            ]
                    thresh_file.write_text("".join(f"{val}\n" for val in threshs))

    def minimize_weight_bit_width(self, model: ModelWrapper) -> BaseDataType:
        """Minimize threshold datatype, with RTL-specific adjustments.

        The RTL implementation saturates inputs to the threshold datatype range
        when the threshold datatype is narrower than the input datatype. To ensure
        correct comparisons at saturation boundaries, the threshold datatype must
        be able to represent [min_threshold - 1 : max_threshold].
        """
        # First, call the base class implementation
        tdt = super().minimize_weight_bit_width(model)

        # Check if we need RTL-specific adjustments
        idt = self.get_input_datatype(0)
        if not idt.is_integer() or not tdt.is_integer():
            return tdt

        # If threshold datatype is smaller than input datatype, we need to ensure
        # it can represent min_threshold - 1 to handle RTL saturation correctly
        if tdt.bitwidth() < idt.bitwidth():
            thresholds = model.get_initializer(self.onnx_node.input[1])
            if not isinstance(thresholds, np.ndarray):
                raise FINNInternalError(f"{self.onnx_node.name}: threshold initializer is missing")
            min_threshold = float(thresholds.min())
            max_threshold = float(thresholds.max())
            min_required = min_threshold - 1
            max_required = max_threshold

            # Compute the new datatype that can represent the extended range
            if min_required < 0:
                if abs(min_required) > max_required:
                    new_tdt = DataType.get_smallest_possible(min_required)
                else:
                    new_tdt = DataType.get_smallest_possible(-max_required - 1)
            elif idt.signed():
                new_tdt = DataType.get_smallest_possible(-max_required - 1)
            else:
                new_tdt = DataType.get_smallest_possible(max_required)

            # Only update if the new datatype is wider
            if new_tdt.bitwidth() > tdt.bitwidth():
                threshold_tensor = self.get_hw_compatible_threshold_tensor(thresholds)
                if not np.vectorize(new_tdt.allowed)(threshold_tensor).all():
                    raise FINNInternalError(f"Thresholds can't be expressed with type {new_tdt!s}")
                self.set_nodeattr("weightDataType", new_tdt.name)
                model.set_tensor_datatype(self.onnx_node.input[1], new_tdt)
                return new_tdt

        return tdt

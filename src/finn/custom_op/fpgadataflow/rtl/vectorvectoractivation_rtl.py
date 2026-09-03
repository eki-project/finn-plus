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

"""RTL implementation of Vector-Vector Activation Unit (VVAU).

This module provides an RTL-based implementation of the Vector-Vector Activation
Unit for DSP-based computation of quantized neural network activations in FPGA
dataflow architectures.
"""

import numpy as np
from onnx import GraphProto, NodeProto
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import Literal, cast

from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.custom_op.fpgadataflow.vectorvectoractivation import VVAU, NodeAttrTypes
from finn.util.basic import is_versal
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.settings import get_settings

# finn-rtllib sources copied verbatim into the generated IP directory
_RTL_SOURCES = [
    "mvu_pkg.sv",
    "mvu_vvu_axi.sv",
    "replay_buffer.sv",
    "mvu.sv",
    "mvu_vvu_8sx9_dsp58.sv",
    "add_multi.sv",
]


class VVAU_rtl(VVAU, RTLBackend):
    """RTL implementation of Vector-Vector Activation Unit.

    Implements DSP-based activation functions using vector-vector
    multiply-accumulate operations for efficient FPGA execution.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize the RTL Vector-Vector Activation Unit node."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return the dictionary of attribute names and their types for this node."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(VVAU.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def execute_node(self, context: dict[str, np.ndarray], graph: GraphProto) -> None:
        """Execute this VVAU node via C++ or RTL simulation."""
        mode = self.get_nodeattr("exec_mode")
        mem_mode = self.mem_mode
        node = self.onnx_node

        if mode == "cppsim":
            VVAU.execute_node(self, context, graph)
        elif mode == "rtlsim":
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
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

            # NOTE: this block previously sat inside the input loop above (a stray
            # indentation bug), running rtlsim once per graph input. rtlsim is
            # deterministic, so only the number of (redundant) invocations changed;
            # the produced output tensor is identical.
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
            out_npy_path = f"{code_gen_dir}/output.npy"
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

    def lut_estimation(self) -> Literal[0]:
        """Estimate LUT utilization (always 0 for VVAU as it uses DSPs)."""
        return 0

    def dsp_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Estimate DSP utilization for this VVAU node (PE * ceil(SIMD / 3))."""
        return int(self.pe * np.ceil(self.simd / 3))

    def instantiate_ip(self, cmd: list[str]) -> None:
        """Add RTL IP instantiation commands to the Vivado script."""
        # instantiate the RTL IP
        node_name = self.onnx_node.name
        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
        rtllib_dir = str(Path(get_settings().finn_rtllib) / "mvu") + "/"
        gen_top_module = cast("str", self.get_nodeattr("gen_top_module"))
        sourcefiles = [str(Path(code_gen_dir) / f"{gen_top_module}_wrapper.v")] + [
            rtllib_dir + f for f in _RTL_SOURCES
        ]

        for f in sourcefiles:
            cmd.append(f"add_files -norecurse {f}")

        if self.mem_mode == "internal_decoupled":
            cmd.append(
                f"create_bd_cell -type hier -reference {gen_top_module} /{node_name}/{node_name}"
            )
        else:
            cmd.append(f"create_bd_cell -type hier -reference {gen_top_module} {node_name}")
        # Connect 2x clk to regular clk port
        clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
        cmd.append(
            f"connect_bd_net [get_bd_pins {node_name}/{clk_name}] "
            f"[get_bd_pins {node_name}/{node_name}/ap_clk2x]"
        )

    def generate_hdl(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:
        """Generate HDL code for this VVAU node."""
        # Generate params as part of IP preparation
        code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))
        self.generate_params(model, code_gen_dir)

        template_path, code_gen_dict = self.prepare_codegen_default(fpgapart, clk)
        # determine if weights are narrow range and add parameter to code gen dict
        weights = model.get_initializer(self.onnx_node.input[1])
        if not isinstance(weights, np.ndarray):
            raise FINNInternalError(f"{self.onnx_node.name}: weight initializer is missing")
        wdt = self.get_input_datatype(1)
        narrow_weights = 0 if np.min(weights) == wdt.min() else 1
        code_gen_dict["$NARROW_WEIGHTS$"] = [str(narrow_weights)]
        # add general parameters to dictionary
        code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"] = [self.get_verilog_top_module_name()]
        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        # apply code generation to template
        template_wrapper = Path(template_path).read_text()
        for key, value in code_gen_dict.items():
            # transform list into long string separated by '\n'
            template_wrapper = template_wrapper.replace(key, "\n".join(value))
        (code_gen_dir / f"{self.get_nodeattr('gen_top_module')}_wrapper.v").write_text(
            template_wrapper
        )

        if self.mem_mode == "internal_decoupled":
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

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", str(code_gen_dir))
        self.set_nodeattr("ip_path", str(code_gen_dir))

    def _resolve_segment_len(self, clk: float) -> float:
        """Resolve the DSP chain segment length based on the clock target."""
        # Insert pipeline registers in the DSP58 chain to meet target clock frequency
        # ~0.741 ns seems the worst-case delay through first DSP
        # ~0.605 ns seems to be (on average) delay for all subsequent DSPs
        # clk >= (critical_path_dsps - 1) * 0.605 + 0.741
        if clk <= 0.741:
            raise FINNUserError(
                f"Infeasible clk target of {clk} ns has been set, "
                "consider lowering the targeted clock frequency!"
            )
        critical_path_dsps = np.floor((clk - 0.741) / 0.605 + 1)
        max_chain_len = np.ceil(self.simd / 3)
        return critical_path_dsps if critical_path_dsps < max_chain_len else max_chain_len

    def _resolve_dsp_version(self, fpgapart: str) -> Literal[3]:
        """Resolve the DSP version based on the target FPGA part (3 for Versal DSP58)."""
        # Based on target device and activation/weight-width, choose the
        # supported RTL compute core
        if self.res_type == "lut":
            raise FINNUserError(
                "LUT-based RTL-VVU implementation currently not supported! Please change "
                f"resType for {self.onnx_node.name} to 'dsp' or consider switching to "
                "HLS-based VVAU!"
            )
        if not is_versal(fpgapart):
            raise FINNUserError(
                "DSP-based (RTL) VVU currently only supported on Versal (DSP58) devices"
            )

        return 3

    def prepare_codegen_default(
        self, fpgapart: str, clk: float
    ) -> tuple[str, dict[str, list[str]]]:
        """Prepare the default code generation dictionary for HDL templates."""
        template_path = str(Path(get_settings().finn_rtllib) / "mvu/mvu_vvu_axi_wrapper.v")

        code_gen_dict: dict[str, list[str]] = {}
        code_gen_dict["$IS_MVU$"] = [str(0)]
        code_gen_dict["$VERSION$"] = [str(self._resolve_dsp_version(fpgapart))]
        code_gen_dict["$PUMPED_COMPUTE$"] = [str(0)]
        mw = int(np.prod(self.kernel))
        code_gen_dict["$MW$"] = [str(mw)]
        code_gen_dict["$MH$"] = [str(self.channels)]
        code_gen_dict["$PE$"] = [str(self.pe)]
        code_gen_dict["$SIMD$"] = [str(self.simd)]
        code_gen_dict["$ACTIVATION_WIDTH$"] = [str(self.get_input_datatype(0).bitwidth())]
        code_gen_dict["$WEIGHT_WIDTH$"] = [str(self.get_input_datatype(1).bitwidth())]
        code_gen_dict["$ACCU_WIDTH$"] = [str(self.get_output_datatype().bitwidth())]
        code_gen_dict["$SIGNED_ACTIVATIONS$"] = (
            [str(1)] if (self.get_input_datatype(0).min() < 0) else [str(0)]
        )
        code_gen_dict["$SEGMENTLEN$"] = [str(self._resolve_segment_len(clk))]

        return template_path, code_gen_dict

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        """Get the list of RTL files needed for this VVAU node."""
        if abspath:
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen")) + "/"
            rtllib_dir = str(Path(get_settings().finn_rtllib) / "mvu") + "/"
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        return [
            code_gen_dir + cast("str", self.get_nodeattr("gen_top_module")) + "_wrapper.v",
            *[rtllib_dir + f for f in _RTL_SOURCES],
        ]

    def get_verilog_paths(self) -> list[str]:
        """Get the list of Verilog paths required for this node."""
        verilog_paths = super().get_verilog_paths()
        verilog_paths.append(str(Path(get_settings().finn_rtllib) / "mvu"))
        return verilog_paths

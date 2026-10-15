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

"""RTL implementation of the Matrix Vector Activation Unit (MVAU).

This module provides an RTL-based implementation of the Matrix Vector Activation
Unit for FPGA acceleration, supporting features like double-pumped DSPs, tiled
compute (TH > 1) and various weight memory modes.
"""

import numpy as np
from onnx import GraphProto, NodeProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING, Literal, cast

from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU, NodeAttrTypes
from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.basic import get_dsp_block, get_dsp_datapath_limits
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from finn.transformation.fpgadataflow.loop_rolling import LoopBodyInputType

# ONNX i/o tensor shape assumptions for MatrixVectorActivation_rtl:
# input 0 is the input tensor, shape (.., i_size) = (..., MW)
# input 1 is the weight tensor, shape (i_size, o_size) = (MW, MH)
# output 0 is the output tensor, shape (.., o_size) = (..., MH)
# the ... here can be any shape (representing groups of vectors)

# finn-rtllib/mvu sources of the untiled (TH=1) MVU core
_RTL_SOURCES = [
    "mvu_pkg.sv",
    "mvu_vvu_axi.sv",
    "replay_buffer.sv",
    "mvu.sv",
    "mvu_vvu_8sx9_dsp58.sv",
    "add_multi.sv",
]
# finn-rtllib/mvu_tiled sources of the tiled (TH>1) MVU core, in the order they are
# added to the Vivado IPI project
_RTL_SOURCES_TILED_IPI = [
    "../skid/skid.sv",
    "../mvu/mvu_pkg.sv",
    "../mvu/add_multi.sv",
    "mvu_tiled_axi.sv",
    "cu_mvau_tiled.sv",
    "acc_stage.sv",
    "input_gen.sv",
    "weights_buff_tile.sv",
]
# the same tiled MVU sources, in the order they are compiled for rtlsim
_RTL_SOURCES_TILED = [
    "../skid/skid.sv",
    "../mvu/mvu_pkg.sv",
    "../mvu/add_multi.sv",
    "acc_stage.sv",
    "input_gen.sv",
    "weights_buff_tile.sv",
    "cu_mvau_tiled.sv",
    "mvu_tiled_axi.sv",
]


@register_custom_op
class MVAU_rtl(MVAU, RTLBackend):
    """Class that corresponds to finn-rtl Matrix Vector Unit."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize the RTL Matrix Vector Activation Unit.

        Args:
            onnx_node: ONNX node to wrap.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return node attribute types, adding pumpedCompute to MVAU and RTLBackend attributes."""
        my_attrs: NodeAttrTypes = {
            # Double-pumped DSPs enabled
            "pumpedCompute": ("i", False, 0, {0, 1}),
        }
        my_attrs.update(MVAU.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    @property
    def pumped_compute(self) -> int:
        """Get whether the compute core runs at 2x clock (0/1)."""
        return cast("int", self.get_nodeattr("pumpedCompute"))

    def _rtllib_dir(self) -> Path:
        """Return the finn-rtllib directory of the MVU core (mvu or mvu_tiled)."""
        core = "mvu_tiled" if self.th > 1 else "mvu"
        return Path(get_settings().finn_rtllib) / core

    def adapt_for_loop_body(self, input_types: "list[LoopBodyInputType]") -> None:  # noqa: ARG002
        """Adapt MVAU_rtl for loop body execution.

        A per-iteration indexed weight must be streamed from external memory over
        the AXI-MM interface, which only mem_mode "external_mem" instantiates
        (along with the per-iteration index input). LoopRolling flags such nodes
        by setting mlo_max_iter on the consumer of each indexed (PARAMETER) loop
        input, so gate the switch on that per-node signal rather than the
        positional loop signature. Dynamic matmuls (streamed activation weights,
        e.g. a merged branch) are not indexed per iteration, do not receive
        mlo_max_iter, and thus correctly keep their mem_mode ("dynamic").

        Args:
            input_types: Types of the loop body inputs (unused).
        """
        if self.mlo_max_iter > 0:
            self.set_nodeattr("mem_mode", "external_mem")

    def execute_node(self, context: dict[str, np.ndarray], graph: GraphProto) -> None:
        """Execute the node via Python (cppsim) or rtlsim, packing inputs and weights per mem_mode.

        Args:
            context: Dictionary mapping tensor names to numpy arrays.
            graph: ONNX graph containing this node.
        """
        mode = self.get_nodeattr("exec_mode")
        mem_mode = self.mem_mode
        node = self.onnx_node

        if mode == "cppsim":
            MVAU.execute_node(self, context, graph)
        elif mode == "rtlsim":
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
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
                    export_idt = self.get_input_datatype(in_ind)
                    # make copy before saving the array
                    reshaped_input = reshaped_input.copy()
                    np.save(str(Path(code_gen_dir) / "input_0.npy"), reshaped_input)

                if in_ind == 1 and (
                    mem_mode in ["dynamic", "external"] or self.get_nodeattr("bodies")
                ):
                    reshaped_input = context[inputs].reshape(-1, context[inputs].shape[-1])
                    self.make_weight_file(
                        reshaped_input, "decoupled_npy", f"{code_gen_dir}/input_1.npy"
                    )

            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            inp = npy_to_rtlsim_input(f"{code_gen_dir}/input_0.npy", export_idt, nbits)
            super().reset_rtlsim(sim)
            if mem_mode in [
                "external",
                "dynamic",
                "internal_decoupled",
                "external_mem",
            ] or self.get_nodeattr("bodies"):
                wnbits = self.get_instream_width(1)
                if mem_mode == "dynamic":
                    wnbits = wnbits * self.simd
                export_wdt = self.get_input_datatype(1)

                wei = npy_to_rtlsim_input(f"{code_gen_dir}/input_1.npy", export_wdt, wnbits)
                num_w_reps = np.prod(self.num_input_vectors) // self.th
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
            context[node.output[0]] = np.asarray([output], dtype=np.float32).reshape(*oshape)
        else:
            raise FINNInternalError(
                f"{self.onnx_node.name}: invalid exec_mode {mode}, "
                f'must be one of ("cppsim", "rtlsim")'
            )

    def lut_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Return the LUT estimate (0, RTL MVU is DSP-based)."""
        return 0

    def dsp_estimation(self, fpgapart: str) -> int:
        """Estimate the number of DSPs used for the multiplications based on the DSP block type.

        Args:
            fpgapart: Target FPGA part number.

        Returns:
            Estimated number of DSP blocks.
        """
        # multiplication
        p = self.pe
        q = self.simd
        dsp_block = get_dsp_block(fpgapart)
        mult_dsp = p * np.ceil(q / 3) if dsp_block == "DSP58" else np.ceil(p / 4) * q
        return int(mult_dsp)

    def instantiate_ip(self, cmd: list[str]) -> None:
        """Append Vivado IPI commands that add the RTL sources and instantiate the MVU core.

        Args:
            cmd: List of TCL commands to which the instantiation commands are appended.
        """
        # instantiate the RTL IP
        node_name = self.onnx_node.name
        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
        gen_top_module = cast("str", self.get_nodeattr("gen_top_module"))
        rtllib_dir = f"{self._rtllib_dir()}/"
        core_sources = _RTL_SOURCES_TILED_IPI if self.th > 1 else _RTL_SOURCES
        sourcefiles = [str(Path(code_gen_dir) / f"{gen_top_module}_wrapper.v")] + [
            rtllib_dir + f for f in core_sources
        ]

        for f in sourcefiles:
            cmd.append(f"add_files -norecurse {f}")
        if self.mem_mode in ["internal_decoupled", "dynamic", "external_mem"] or self.mlo_max_iter:
            cmd.append(
                f"create_bd_cell -type hier -reference {gen_top_module} /{node_name}/{node_name}"
            )
            # if using 2x pumped compute, connect the MVU's 2x clk input
            # to the 2x clock port. Otherwise connect 2x clk to regular clk port
            clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
            if self.pumped_compute or self.pumped_memory:
                clk2x_name = self.get_verilog_top_module_intf_names()["clk2x"][0]
                cmd.append(
                    f"connect_bd_net [get_bd_pins {node_name}/{clk2x_name}] "
                    f"[get_bd_pins {node_name}/{node_name}/{clk2x_name}]"
                )
            else:
                cmd.append(
                    f"connect_bd_net [get_bd_pins {node_name}/{clk_name}] "
                    f"[get_bd_pins {node_name}/{node_name}/ap_clk2x]"
                )
        # external
        else:
            cmd.append(f"create_bd_cell -type hier -reference {gen_top_module} {node_name}")
            # For 2x pumped compute the MVU's 2x clk input is driven by the external
            # 2x clock port, which CreateStitchedIP exposes and connects for this
            # (flat) cell - nothing to connect here. Otherwise drive the unused 2x
            # clk input from the regular clock.
            clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
            if not self.pumped_compute:
                cmd.append(
                    f"connect_bd_net [get_bd_pins {node_name}/{clk_name}] "
                    f"[get_bd_pins {node_name}/ap_clk2x]"
                )

    def _resolve_segment_len(self, clk: float) -> float:
        """Return the DSP chain segment length that meets the target clock period.

        Args:
            clk: Target clock period in nanoseconds.

        Returns:
            Maximum DSP chain length for the target frequency.
        """
        # Insert pipeline registers in the DSP58 chain to meet target clock frequency
        # ~0.741 ns seems the worst-case delay through first DSP
        # ~0.605 ns seems to be (on average) delay for all subsequent DSPs
        # clk >= (critical_path_dsps - 1) * 0.605 + 0.741
        if self.pumped_compute:
            ref_clk = clk / 2
            simd_factor = 6
        else:
            ref_clk = clk
            simd_factor = 3

        if ref_clk <= 0.741:
            raise FINNUserError(
                f"{self.onnx_node.name}: infeasible clk target of {ref_clk} ns, "
                f"consider lowering the targeted clock frequency."
            )
        critical_path_dsps = np.floor((ref_clk - 0.741) / 0.605 + 1)
        max_chain_len = np.ceil(self.simd / simd_factor)
        dsp_chain_len = critical_path_dsps if critical_path_dsps < max_chain_len else max_chain_len
        return float(dsp_chain_len)

    def _resolve_dsp_version(self, dsp_block: str) -> Literal[3, 2, 1]:
        """Return the RTL compute core version for the given DSP block type.

        Args:
            dsp_block: DSP block type (e.g. 'DSP58', 'DSP48E2').

        Returns:
            DSP version number (1, 2, or 3).
        """
        # Based on target device and activation/weight-width, choose the
        # supported RTL compute core
        if self.res_type == "lut":
            raise FINNUserError(
                "LUT-based RTL-MVU implementation currently not supported! "
                f"Please change resType for {self.onnx_node.name} to 'dsp' "
                f"or consider switching to HLS-based MVAU!"
            )

        match dsp_block:
            case "DSP58":
                return 3
            case "DSP48E2":
                return 2
            case _:
                return 1

    def generate_hdl(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:
        """Generate parameters, render the MVU wrapper template and set the codegen attributes.

        Args:
            model: ONNX model wrapper.
            fpgapart: Target FPGA part number.
            clk: Target clock period in ns.
        """
        # Generate params as part of IP preparation
        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
        if not self.mlo_max_iter:
            self.generate_params(model, code_gen_dir)

        template_path, code_gen_dict = self.prepare_codegen_default(fpgapart, clk)
        # determine if weights are narrow range and add parameter to code gen dict
        weights = model.get_initializer(self.onnx_node.input[1])
        wdt = self.get_input_datatype(1)
        # Weights are only a constant initializer for the non-streamed cases; dynamic-weight
        # and external-memory (MLO) layers load them at runtime instead, so there is nothing
        # to inspect here for those (check the mem_mode before touching the weights).
        narrow_weights = (
            0
            if self.mem_mode in ["dynamic", "external_mem"]
            or not isinstance(weights, np.ndarray)
            or np.min(weights) == wdt.min()
            else 1
        )
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
            code_gen_line = "\n".join(value)
            template_wrapper = template_wrapper.replace(key, code_gen_line)
        gen_top_module = cast("str", self.get_nodeattr("gen_top_module"))
        (Path(code_gen_dir) / f"{gen_top_module}_wrapper.v").write_text(template_wrapper)

        # generate the weight-infrastructure HDL (shared MVAU helper)
        self.generate_infra_hdl(fpgapart)

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def _check_tiling_constraints(self) -> None:
        """Validate the tiled-MVU (TH>1) constraints, raise if they are violated."""
        theight = self.th
        tile = self.pe * self.simd
        if tile % theight != 0:
            raise FINNUserError(
                f"{self.onnx_node.name}: TH={theight} does not divide the tile "
                f"PE*SIMD={tile}. The tiled MVU requires (PE*SIMD) % TH == 0."
            )

        # The tiled wrapper conservatively caps both activation and weight widths at 8 bits.
        act_width = self.get_input_datatype(0).bitwidth()
        if act_width > 8:
            raise FINNUserError(
                f"{self.onnx_node.name}: activation width of {act_width} bits exceeds the "
                f"tiled MVU maximum of 8 bits."
            )
        weight_width = self.get_input_datatype(1).bitwidth()
        if weight_width > 8:
            raise FINNUserError(
                f"{self.onnx_node.name}: weight width of {weight_width} bits exceeds the "
                f"tiled MVU maximum of 8 bits."
            )

        # The tile size must divide the number of input vectors evenly; otherwise the
        # final partial tile can cause output to be dropped and eventually stall.
        num_inp_vec = int(np.prod(self.num_input_vectors))
        if num_inp_vec % theight != 0:
            valid_th = [t for t in range(1, num_inp_vec + 1) if num_inp_vec % t == 0]
            raise FINNUserError(
                f"{self.onnx_node.name}: TH={theight} does not divide "
                f"numInputVectors={num_inp_vec}. The tiled MVU requires TH | numInputVectors; "
                f"choose a divisor (valid: {valid_th})."
            )

    def _check_dsp_datapath_limits(self, dsp_block: str) -> None:
        """Raise if the operand or accumulator widths exceed the target DSP datapath.

        Args:
            dsp_block: DSP block type of the target device.
        """
        # the operand and accumulator widths must fit the target DSP datapath
        # (activation->B, weight->A, accumulator->P); wider values would be
        # silently truncated or fail synthesis on the DSP-based RTL MVU
        max_act, max_weight, max_acc = get_dsp_datapath_limits(dsp_block)
        idt = self.get_input_datatype(0)
        signed_act = 1 if idt.signed() else 0
        act_width = idt.bitwidth()
        weight_width = self.get_input_datatype(1).bitwidth()
        acc_width = self.get_output_datatype().bitwidth()
        # activations sit in the signed B datapath; unsigned activations cost one
        # extra bit, so the effective width must stay below the B datapath width
        if act_width - signed_act >= max_act:
            signedness = "signed" if signed_act else "unsigned"
            raise FINNUserError(
                f"{self.onnx_node.name}: {signedness} activation width of {act_width} bits "
                f"exceeds the {dsp_block} activation datapath limit of "
                f"{max_act - 1 + signed_act} bits. Use the HLS MVAU for wider activations."
            )
        if weight_width > max_weight:
            raise FINNUserError(
                f"{self.onnx_node.name}: weight width of {weight_width} bits exceeds the "
                f"{dsp_block} weight datapath limit of {max_weight} bits. "
                f"Use the HLS MVAU for wider weights."
            )
        if acc_width > max_acc:
            raise FINNUserError(
                f"{self.onnx_node.name}: accumulator width of {acc_width} bits exceeds the "
                f"{dsp_block} accumulator datapath limit of {max_acc} bits. "
                f"Use the HLS MVAU for wider accumulators."
            )

    def prepare_codegen_default(
        self, fpgapart: str, clk: float
    ) -> tuple[str, dict[str, list[str]]]:
        """Return the wrapper template path and the code generation dictionary for this node.

        Args:
            fpgapart: Target FPGA part number.
            clk: Target clock period in ns.

        Returns:
            Template file path and code generation dictionary.
        """
        if self.th > 1:
            template_path = str(self._rtllib_dir() / "mvu_tiled_axi_wrapper.v")
        else:
            template_path = str(self._rtllib_dir() / "mvu_vvu_axi_wrapper.v")

        # check if settings are valid
        if self.pumped_compute and self.simd == 1:
            raise FINNUserError(
                f"{self.onnx_node.name}: clock pumping an input of SIMD=1 is not meaningful. "
                f"Please increase SIMD."
            )

        if self.th > 1:
            self._check_tiling_constraints()

        dsp_block = get_dsp_block(fpgapart)
        self._check_dsp_datapath_limits(dsp_block)

        signed_acts = "1" if (self.get_input_datatype(0).min() < 0) else "0"
        code_gen_dict: dict[str, list[str]] = {
            "$IS_MVU$": [str(1)],
            "$VERSION$": [str(self._resolve_dsp_version(dsp_block))],
            "$PUMPED_COMPUTE$": [str(self.pumped_compute)],
            "$MW$": [str(self.mw)],
            "$MH$": [str(self.mh)],
            "$PE$": [str(self.pe)],
            "$SIMD$": [str(self.simd)],
            "$TH$": [str(self.th)],
            "$ACTIVATION_WIDTH$": [str(self.get_input_datatype(0).bitwidth())],
            "$WEIGHT_WIDTH$": [str(self.get_input_datatype(1).bitwidth())],
            "$ACCU_WIDTH$": [str(self.get_output_datatype().bitwidth())],
            "$SIGNED_ACTIVATIONS$": [signed_acts],
            "$SEGMENTLEN$": [str(self._resolve_segment_len(clk))],
        }

        return template_path, code_gen_dict

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        """Return the list of RTL files (wrapper and MVU core sources) for this node.

        Args:
            abspath: If True, return absolute file paths; otherwise return file names
                relative to the code generation / finn-rtllib directories.

        Returns:
            List of RTL file paths.
        """
        if abspath:
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen")) + "/"
            rtllib_dir = f"{self._rtllib_dir()}/"
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        core_sources = _RTL_SOURCES_TILED if self.th > 1 else _RTL_SOURCES
        gen_top_module = cast("str", self.get_nodeattr("gen_top_module"))
        return [f"{code_gen_dir}{gen_top_module}_wrapper.v"] + [
            rtllib_dir + f for f in core_sources
        ]

    def get_verilog_paths(self) -> list[str]:
        """Return the Verilog include paths, adding the mvu or mvu_tiled rtllib directory."""
        verilog_paths = super().get_verilog_paths()
        verilog_paths.append(str(self._rtllib_dir()))
        return verilog_paths

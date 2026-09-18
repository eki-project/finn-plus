# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RTL backend implementation of the uniform-affine requantization operator."""

import numpy as np
from onnx import GraphProto, NodeProto
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from finn.custom_op.fpgadataflow.base.requant import NodeAttrTypes, Requant
from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.basic import get_dsp_block, make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.exception import FINNInternalError
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

_RTL_SOURCES = ["queue.sv", "requant.sv", "requant_axi.sv"]


@register_custom_op
class Requant_rtl(Requant, RTLBackend):
    """RTL backend for Requant operation using finn-rtllib/requant."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(Requant.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def _resolve_dsp_version(self, fpgapart: str) -> Literal[3, 2, 1]:
        """Determine DSP version based on FPGA part."""
        match get_dsp_block(fpgapart):
            case "DSP58":
                return 3
            case "DSP48E2":
                return 2
            case _:
                return 1

    def generate_hdl(
        self, model: "ModelWrapper", fpgapart: str, clk: float  # noqa: ARG002
    ) -> None:
        """Generate RTL code for the requant operation."""
        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
        if code_gen_dir == "":
            code_gen_dir = str(make_build_dir("requant_rtl_ipgen_"))
            self.set_nodeattr("code_gen_dir_ipgen", code_gen_dir)
        code_gen_path = Path(code_gen_dir)

        pe = self.pe
        num_channels = self.num_channels
        cf = num_channels // pe  # Channel fold

        idt = self.get_input_datatype(0)
        odt = self.get_output_datatype()
        k = idt.bitwidth()  # Input precision
        n = odt.bitwidth()  # Output precision

        version = self._resolve_dsp_version(fpgapart)

        # Get scale and bias from model
        scale = self.get_scale(model)
        bias = self.get_bias(model)

        # Broadcast scalar scale/bias to all channels if needed
        if scale.size == 1:
            scale = np.full(num_channels, scale.item(), dtype=np.float32)
        if bias.size == 1:
            bias = np.full(num_channels, bias.item(), dtype=np.float32)

        # Reshape for PE interleaving: the RTL expects [PE][CF] layout
        scale_reshaped = scale.reshape(cf, pe).T  # [PE][CF]
        bias_reshaped = bias.reshape(cf, pe).T  # [PE][CF]

        def format_sv_array(arr: np.ndarray) -> str:
            """Format a 2D numpy array as a SystemVerilog array literal."""
            lines = []
            for pe_idx in range(arr.shape[0]):
                # Fixed-point notation with 6 decimal places (shortreal is 32-bit float)
                row = ", ".join(f"{float(v):.6f}" for v in arr[pe_idx])
                lines.append("'{" + row + "}")
            return "'{" + ", ".join(lines) + "}"

        scales_sv = format_sv_array(scale_reshaped)
        biases_sv = format_sv_array(bias_reshaped)

        # Calculate stream widths (byte-aligned)
        in_stream_width = ((pe * k + 7) // 8) * 8
        out_stream_width = ((pe * n + 7) // 8) * 8

        top_module_name = self.get_verilog_top_module_name()
        rtllib_dir = Path(get_settings().finn_rtllib) / "requant" / "hdl"

        # Generate SystemVerilog implementation module (with _impl suffix)
        sv_code = (rtllib_dir / "requant_wrapper_template.sv").read_text()
        for placeholder, value in {
            "$TOP_MODULE_NAME$": top_module_name,
            "$VERSION$": str(version),
            "$K$": str(k),
            "$N$": str(n),
            "$C$": str(num_channels),
            "$PE$": str(pe),
            "$SCALES$": scales_sv,
            "$BIASES$": biases_sv,
            "$IN_STREAM_WIDTH$": str(in_stream_width),
            "$OUT_STREAM_WIDTH$": str(out_stream_width),
        }.items():
            sv_code = sv_code.replace(placeholder, value)
        (code_gen_path / f"{top_module_name}_impl.sv").write_text(sv_code)

        # Generate Verilog stub wrapper (for IP packaging - must be .v)
        v_code = (rtllib_dir / "requant_wrapper_template.v").read_text()
        for placeholder, value in {
            "$TOP_MODULE_NAME$": top_module_name,
            "$IN_STREAM_WIDTH$": str(in_stream_width),
            "$OUT_STREAM_WIDTH$": str(out_stream_width),
        }.items():
            v_code = v_code.replace(placeholder, value)
        (code_gen_path / f"{top_module_name}.v").write_text(v_code)

        self.set_nodeattr("gen_top_module", top_module_name)

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stitch ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        """Return list of RTL files needed for this node."""
        rtllib_dir = str(Path(get_settings().finn_rtllib) / "requant" / "hdl") + "/"
        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))

        rtl_files = [rtllib_dir + f for f in _RTL_SOURCES]

        # Add generated wrappers (Verilog stub + SystemVerilog impl)
        top_module = cast("str", self.get_nodeattr("gen_top_module"))
        if top_module == "":
            top_module = self.get_verilog_top_module_name()
        rtl_files.append(str(Path(code_gen_dir) / f"{top_module}_impl.sv"))
        rtl_files.append(str(Path(code_gen_dir) / f"{top_module}.v"))

        if abspath:
            return rtl_files
        return [Path(f).name for f in rtl_files]

    def code_generation_ipi(self) -> list[str]:
        """Construct and return the TCL for node instantiation in Vivado IPI."""
        sourcefiles = self.get_rtl_file_list(abspath=True)
        top_module = cast("str", self.get_nodeattr("gen_top_module"))

        cmd = [f"add_files -norecurse {f}" for f in sourcefiles]
        cmd += [f"create_bd_cell -type module -reference {top_module} {self.onnx_node.name}"]
        return cmd

    def execute_node(self, context: dict[str, np.ndarray], graph: GraphProto) -> None:
        """Execute the node, using RTL simulation if exec_mode is rtlsim."""
        mode = self.get_nodeattr("exec_mode")
        if mode != "rtlsim":
            # Use base class Python execution
            Requant.execute_node(self, context, graph)
            return

        # Custom RTL sim that only passes input 0 (data), not scale/bias
        # which are embedded as parameters in the generated HDL
        node = self.onnx_node
        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))

        # Only process input 0 (data tensor)
        inp = node.input[0]
        exp_ishape = tuple(self.get_normal_input_shape(0))
        folded_ishape = self.get_folded_input_shape(0)
        inp_val = context[inp]
        if str(inp_val.dtype) != "float32":
            raise FINNInternalError(f"{node.name}: input datatype is not float32")
        if inp_val.shape != exp_ishape:
            raise FINNInternalError(f"{node.name}: input shape {inp_val.shape} != {exp_ishape}")
        export_idt = self.get_input_datatype(0)

        reshaped_input = inp_val.reshape(folded_ishape)
        np.save(str(Path(code_gen_dir) / "input_0.npy"), reshaped_input)
        nbits = self.get_instream_width(0)
        rtlsim_inp = npy_to_rtlsim_input(f"{code_gen_dir}/input_0.npy", export_idt, nbits)

        io_dict = {
            "inputs": {"in0": rtlsim_inp},
            "outputs": {"out0": []},
        }

        sim = self.get_rtlsim()
        self.reset_rtlsim(sim)
        self.rtlsim_multi_io(sim, io_dict)
        self.close_rtlsim(sim)

        # Process output
        rtlsim_output = io_dict["outputs"]["out0"]
        odt = self.get_output_datatype(0)
        target_bits = odt.bitwidth()
        packed_bits = self.get_outstream_width(0)
        out_npy_path = f"{code_gen_dir}/output.npy"
        out_shape = self.get_folded_output_shape(0)
        rtlsim_output_to_npy(rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits)

        # Load and reshape output
        exp_oshape = tuple(self.get_normal_output_shape(0))
        output = np.load(out_npy_path)
        output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
        context[node.output[0]] = output

        if context[node.output[0]].shape != exp_oshape:
            raise FINNInternalError(
                f"{node.name}: output shape {context[node.output[0]].shape} != {exp_oshape}"
            )

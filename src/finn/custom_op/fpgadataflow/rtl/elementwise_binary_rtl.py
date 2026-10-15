############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################
"""RTL backend implementations of elementwise binary operations (Add, Sub, Mul)."""

import numpy as np
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import roundup_to_integer_multiple
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow import elementwise_binary
from finn.custom_op.fpgadataflow.elementwise_binary import ElementwiseBinaryOperation, NodeAttrTypes
from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.transformation.fpgadataflow.loop_rolling import LoopBodyInputType
from finn.util.basic import fifo_rtl_files
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from onnx import GraphProto, NodeProto


class ElementwiseBinary_rtl(ElementwiseBinaryOperation, RTLBackend):
    """Base CustomOp wrapper for the finn-rtllib eltwise component.

    Supports float/float, int/float, float/int, and int/int data paths.
    """

    def __init__(self, onnx_node: "NodeProto", **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return node attribute types, adding mem_mode and runtime_writeable_weights."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(ElementwiseBinaryOperation.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        my_attrs.update(
            {
                # memory mode for the const value
                # has to be internal_decoupled for rtl elementwise ops
                "mem_mode": (
                    "s",
                    False,
                    "internal_decoupled",
                ),
                "runtime_writeable_weights": ("i", False, 0, {0, 1}),
            }
        )
        return my_attrs

    def adapt_for_loop_body(self, input_types: list) -> None:
        """Adapt elementwise binary operator for loop body execution.

        When an elementwise operator is placed inside a loop, parameters that
        are indexed per iteration (PARAMETER type) need to be received as
        streaming inputs rather than embedded constants. This method changes
        the lhs_style/rhs_style attributes from "const" to "input" as needed.
        """
        # If rhs (input[1]) is a PARAMETER (streamed per iteration),
        # change its style to "input"
        if (
            len(input_types) > 1
            and input_types[1] == LoopBodyInputType.PARAMETER
            and self.rhs_style == "const"
        ):
            self.set_nodeattr("rhs_style", "input")

        # Similarly for lhs if needed
        if (
            len(input_types) > 0
            and input_types[0] == LoopBodyInputType.PARAMETER
            and self.lhs_style == "const"
        ):
            self.set_nodeattr("lhs_style", "input")

    def generate_hdl(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:  # noqa: ARG002
        """Render the eltwise wrapper template and generate the memstream if needed.

        The static RTL sources are referenced in place from finn-rtllib (see
        ``get_rtl_file_list``), so nothing is copied.

        Args:
            model: Model containing this node, used to look up constant initializers.
            fpgapart: Target FPGA part string.
            clk: Target clock period (unused).

        Raises:
            FINNUserError: If the memory mode or the data types are not supported
                by the RTL eltwise core.
        """
        lhs_style = self.lhs_style
        rhs_style = self.rhs_style
        mlo = cast("int", self.get_nodeattr("mlo_max_iter"))
        # When both inputs are dynamic (input/input), there are no weights to embed.
        # MLO mode also allows rhs to be "input" for parameter streaming.
        has_const = lhs_style == "const" or rhs_style == "const"

        if has_const and self.get_nodeattr("mem_mode") != "internal_decoupled":
            raise FINNUserError(
                f"{self.onnx_node.name}: only internal_decoupled mode is supported for "
                "rtl elementwise ops with const inputs"
            )

        # Resolve data types
        lhs_dtype = self.get_input_datatype(ind=0)
        rhs_dtype = self.get_input_datatype(ind=1)
        out_dtype = self.get_output_datatype(ind=0)

        lhs_float = lhs_dtype == DataType["FLOAT32"]
        rhs_float = rhs_dtype == DataType["FLOAT32"]
        both_int = not lhs_float and not rhs_float
        op_name = self._get_rtl_op_name()  # e.g. '"ADD"'

        # Validate type constraints mirroring RTL parameter validation
        if both_int:
            if lhs_dtype.bitwidth() != rhs_dtype.bitwidth():
                raise FINNUserError(
                    f"{self.onnx_node.name}: Int/int path requires matching widths: "
                    f"A_WIDTH={lhs_dtype.bitwidth()}, B_WIDTH={rhs_dtype.bitwidth()}"
                )
            if lhs_dtype.signed() != rhs_dtype.signed():
                raise FINNUserError(
                    f"{self.onnx_node.name}: Int/int path requires matching signedness"
                )
            if op_name == '"MUL"':
                max_w = 24 if lhs_dtype.signed() else 23
                if lhs_dtype.bitwidth() > max_w:
                    raise FINNUserError(
                        f"{self.onnx_node.name}: Int MUL width {lhs_dtype.bitwidth()} "
                        f"exceeds DSP58 capacity ({max_w})"
                    )

        # Validate output width matches RTL O_WIDTH derivation
        if both_int:
            if op_name == '"MUL"':
                expected_owidth = 2 * lhs_dtype.bitwidth()
            else:
                expected_owidth = lhs_dtype.bitwidth() + 1
            if out_dtype.bitwidth() != expected_owidth:
                raise FINNUserError(
                    f"{self.onnx_node.name}: Int/int {op_name} requires out_dtype "
                    f"bitwidth={expected_owidth}, got {out_dtype.bitwidth()}"
                )
        elif out_dtype.bitwidth() != 32:
            raise FINNUserError(
                f"{self.onnx_node.name}: Float path requires 32-bit output, "
                f"got {out_dtype.bitwidth()}"
            )

        code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))

        # Only generate weight params when there is a const input
        if has_const:
            self.generate_params(model, code_gen_dir)

        template_path = Path(get_settings().finn_rtllib) / "eltwise" / "eltwise_template.v"
        pe = self.pe

        a_width = lhs_dtype.bitwidth()
        b_width = rhs_dtype.bitwidth()

        code_gen_dict: dict[str, str | int | float] = {
            "TOP_MODULE_NAME": self.get_verilog_top_module_name(),
            "PE": pe,
            "OP": op_name,
            "B_SCALE": 1.0,
            "A_FLOAT": 1 if lhs_float else 0,
            "B_FLOAT": 1 if rhs_float else 0,
            "A_WIDTH": a_width,
            "A_SIGNED": 1 if (not lhs_float and lhs_dtype.signed()) else 0,
            "B_WIDTH": b_width,
            "B_SIGNED": 1 if (not rhs_float and rhs_dtype.signed()) else 0,
            "A_STREAM_BITS": pe * a_width,
            "B_STREAM_BITS": pe * b_width,
            "O_STREAM_BITS": pe * out_dtype.bitwidth(),
        }

        template = template_path.read_text()
        for key_name, value in code_gen_dict.items():
            template = template.replace(f"${key_name}$", str(value))

        (code_gen_dir / f"{self.get_verilog_top_module_name()}.v").write_text(template)

        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        # Only generate memstream wrapper when there is a const input
        if has_const or mlo:
            self.generate_hdl_memstream(fpgapart)

        self.set_nodeattr("ipgen_path", str(code_gen_dir))
        self.set_nodeattr("ip_path", str(code_gen_dir))

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        """Return the list of RTL files (eltwise core, wrapper and FIFO) for this node.

        Args:
            abspath: If True, return absolute paths; otherwise bare file names.

        Returns:
            List of RTL source file paths.
        """
        if abspath:
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen")) + "/"
            rtllib_dir = str(Path(get_settings().finn_rtllib) / "eltwise") + "/"
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        top_module = cast("str", self.get_nodeattr("gen_top_module"))
        return [
            f"{rtllib_dir}eltwise.sv",
            f"{rtllib_dir}binopf.sv",
            f"{rtllib_dir}binopi.sv",
            f"{rtllib_dir}int_to_fp32.sv",
            f"{code_gen_dir}{top_module}.v",
            *fifo_rtl_files(abspath),
        ]

    def get_verilog_top_module_intf_names(self) -> dict[str, list[tuple[str, int]] | list[str]]:
        """Return the interface names for the Verilog top module.

        For RTL elementwise operations, this includes handling for MLO mode
        where the rhs parameter may be streamed as an input.
        """
        # Start collecting interface names in a dictionary starting with clock and reset
        intf_names: dict[str, list[tuple[str, int]] | list[str]] = {
            "clk": ["ap_clk"],
            "rst": ["ap_rst_n"],
        }

        # AXI stream input interfaces
        s_axis: list[tuple[str, int]] = []

        # If the left-hand-side is provided as runtime input interface names need to be inserted
        if self.lhs_style == "input":
            s_axis.append(("in0_V", self.get_instream_width_padded(ind=0)))

        # If the right-hand-side is provided as runtime input interface names need to be inserted
        # (input/input mode or MLO mode where adapt_for_loop_body changes rhs_style to "input")
        if self.rhs_style == "input":
            s_axis.append(("in1_V", self.get_instream_width_padded(ind=1)))

        intf_names["s_axis"] = s_axis
        # AXI stream output interfaces
        intf_names["m_axis"] = [("out0_V", self.get_outstream_width_padded(ind=0))]

        # No AXI-MM, AXI-Lite or protocol-less interfaces
        intf_names["aximm"] = []
        intf_names["axilite"] = []
        intf_names["ap_none"] = []

        # Return the interface name dictionary
        return intf_names

    def code_generation_ipi(self) -> list[str]:
        """Construct and return the TCL for node instantiation in Vivado IPI."""
        cmd: list[str] = []
        node_name = self.onnx_node.name
        intf = self.get_verilog_top_module_intf_names()
        clk_name = cast("list[str]", intf["clk"])[0]
        rst_name = cast("list[str]", intf["rst"])[0]
        dout_name = cast("list[tuple[str, int]]", intf["m_axis"])[0][0]
        din_name = cast("list[tuple[str, int]]", intf["s_axis"])[0][0]
        mlo = cast("int", self.get_nodeattr("mlo_max_iter"))
        lhs_style = self.lhs_style
        rhs_style = self.rhs_style
        both_input = lhs_style == "input" and rhs_style == "input" and not mlo

        cmd.append(f"create_bd_cell -type hier {node_name}")
        cmd.append(f"create_bd_pin -dir I -type clk /{node_name}/{clk_name}")
        cmd.append(f"create_bd_pin -dir I -type rst /{node_name}/{rst_name}")
        cmd.append(
            "create_bd_intf_pin -mode Master "
            f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/{dout_name}"
        )
        cmd.append(
            "create_bd_intf_pin -mode Slave "
            f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/{din_name}"
        )

        # When both inputs are dynamic or MLO streams rhs, expose in1_V
        if both_input or (mlo and rhs_style == "input"):
            cmd.append(
                "create_bd_intf_pin -mode Slave "
                f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{node_name}/in1_V"
            )

        # instantiate the RTL block
        self.instantiate_ip(cmd)

        # connect elementwise core
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

        if both_input:
            # input/input mode: wire in1_V directly to eltwise core, no memstream
            cmd.append(
                f"connect_bd_intf_net [get_bd_intf_pins {node_name}/in1_V] "
                f"[get_bd_intf_pins {node_name}/{node_name}/in1_V]"
            )
        else:
            # const or MLO mode: use memstream wrapper
            code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))
            runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1

            axi_dir = Path(get_settings().finn_rtllib) / "axi/hdl"
            ms_rtllib_dir = Path(get_settings().finn_rtllib) / "memstream/hdl"
            file_suffix = "_memstream_wrapper.v"

            strm_tmpl = None
            for fname in code_gen_dir.iterdir():
                if fname.name.endswith(file_suffix):
                    strm_tmpl = fname.name

            if strm_tmpl is None:
                raise FINNInternalError(f"No memstream wrapper found in {code_gen_dir}")

            strm_tmpl_name = strm_tmpl[:-2]
            sourcefiles = [
                str(code_gen_dir / strm_tmpl),
                str(axi_dir / "axilite.sv"),
                str(ms_rtllib_dir / "memstream_axi.sv"),
                str(ms_rtllib_dir / "memstream.sv"),
            ]
            cmd.extend(f"add_files -norecurse {f}" for f in sourcefiles)
            strm_inst = node_name + "_wstrm"
            cmd.append(
                f"create_bd_cell -type hier -reference {strm_tmpl_name} /{node_name}/{strm_inst}"
            )
            cmd.append(
                f"connect_bd_net [get_bd_pins {node_name}/{clk_name}] "
                f"[get_bd_pins {node_name}/{strm_inst}/ap_clk]"
            )
            cmd.append(
                f"connect_bd_net [get_bd_pins {node_name}/{rst_name}] "
                f"[get_bd_pins {node_name}/{strm_inst}/ap_rst_n]"
            )
            cmd.append(
                f"connect_bd_net [get_bd_pins {node_name}/{clk_name}] "
                f"[get_bd_pins {node_name}/{strm_inst}/ap_clk2x]"
            )
            # MLO mode: connect external in1_V to memstream input
            if mlo and rhs_style == "input":
                cmd.append(
                    f"connect_bd_intf_net [get_bd_intf_pins {node_name}/in1_V] "
                    f"[get_bd_intf_pins {node_name}/{strm_inst}/s_axis_0]"
                )
            # Connect memstream output to core input
            cmd.append(
                f"connect_bd_intf_net [get_bd_intf_pins {node_name}/{strm_inst}/m_axis_0] "
                f"[get_bd_intf_pins {node_name}/{node_name}/in1_V]"
            )
            if runtime_writable:
                axilite_name = cast("list[str]", intf["axilite"])[0]
                cmd.append(
                    "create_bd_intf_pin -mode Slave "
                    f"-vlnv xilinx.com:interface:aximm_rtl:1.0 /{node_name}/{axilite_name}"
                )
                cmd.append(
                    f"connect_bd_intf_net [get_bd_intf_pins {node_name}/{axilite_name}] "
                    f"[get_bd_intf_pins {node_name}/{strm_inst}/{axilite_name}]"
                )
                cmd.append("assign_bd_address")
        cmd.append("save_bd_design")
        return cmd

    def instantiate_ip(self, cmd: list[str]) -> None:
        """Append Vivado IPI commands that add the RTL sources and instantiate the core.

        Args:
            cmd: TCL command list, extended in place.
        """
        node_name = self.onnx_node.name
        top_module = cast("str", self.get_nodeattr("gen_top_module"))

        cmd.extend(f"add_files -norecurse {f}" for f in self.get_rtl_file_list(abspath=True))

        # Always create the core inside the hierarchical wrapper
        cmd.append(f"create_bd_cell -type hier -reference {top_module} /{node_name}/{node_name}")

    def execute_node(self, context: dict[str, np.ndarray], graph: "GraphProto") -> None:
        """Execute the node via rtlsim (packing lhs/rhs streams) or fall back to Python.

        Args:
            context: Execution context mapping tensor names to values.
            graph: The ONNX graph containing this node.
        """
        mode = self.get_nodeattr("exec_mode")
        if mode != "rtlsim":
            ElementwiseBinaryOperation.execute_node(self, context, graph)
            return

        node = self.onnx_node
        code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))
        lhs = context[node.input[0]]
        rhs = context[node.input[1]]

        if list(lhs.shape) != self.get_normal_input_shape(ind=0):
            raise FINNInternalError(f"Input shape mismatch for {node.input[0]}")
        if self.rhs_style != "const" and list(rhs.shape) != self.get_normal_input_shape(ind=1):
            raise FINNInternalError(f"Input shape mismatch for {node.input[1]}")

        out_shape = self.get_normal_output_shape(ind=0)

        if self.lhs_style == "const":
            lhs = np.broadcast_to(lhs, out_shape)
            lhs = lhs.reshape(self.get_folded_output_shape(ind=0))
        else:
            lhs = lhs.reshape(self.get_folded_input_shape(ind=0))

        if self.rhs_style == "const":
            rhs = np.broadcast_to(rhs, out_shape)
            rhs = rhs.reshape(self.get_folded_output_shape(ind=0))
        else:
            rhs = rhs.reshape(self.get_folded_input_shape(ind=1))

        lhs_filename = code_gen_dir / "input_0.npy"
        rhs_filename = code_gen_dir / "input_1.npy"
        np.save(lhs_filename, lhs)
        np.save(rhs_filename, rhs)

        io_dict: dict[str, dict] = {"inputs": {}, "outputs": {"out0": []}}
        lhs_dtype = self.get_input_datatype(ind=0)
        lhs_width = self.get_instream_width(ind=0)
        rhs_dtype = self.get_input_datatype(ind=1)
        rhs_width = self.get_instream_width(ind=1)

        mem_mode = self.get_nodeattr("mem_mode")
        mlo = self.get_nodeattr("mlo_max_iter")
        lhs_decoupled = self.lhs_style == "const" and mem_mode == "internal_decoupled"
        # MLO mode: stream RHS when it's marked as input (from adapt_for_loop_body)
        rhs_decoupled = (self.rhs_style == "const" and mem_mode == "internal_decoupled") or (
            self.rhs_style == "input" and mlo
        )

        if self.lhs_style == "input" or lhs_decoupled:
            io_dict["inputs"]["in0"] = npy_to_rtlsim_input(str(lhs_filename), lhs_dtype, lhs_width)
        if self.rhs_style == "input" or rhs_decoupled:
            io_dict["inputs"]["in1"] = npy_to_rtlsim_input(str(rhs_filename), rhs_dtype, rhs_width)

        sim = self.get_rtlsim()
        self.reset_rtlsim(sim)
        self.rtlsim_multi_io(sim, io_dict)
        self.close_rtlsim(sim)

        out = io_dict["outputs"]["out0"]
        dtype = self.get_output_datatype(ind=0)
        width = self.get_outstream_width(ind=0)
        shape = self.get_folded_output_shape(ind=0)
        filename = code_gen_dir / "output_0.npy"
        rtlsim_output_to_npy(out, str(filename), dtype, shape, width, dtype.bitwidth())
        out = np.load(filename)
        out = out.reshape(self.get_normal_output_shape(ind=0))
        if dtype.is_integer():
            context[node.output[0]] = out
        else:
            context[node.output[0]] = out.astype(np.float32)

    def generate_params(
        self,
        model: ModelWrapper,
        path: str | Path,
        fpgapart: str | None = None,  # noqa: ARG002
    ) -> None:
        """Write the constant operand as decoupled npy and Verilog dat files.

        Args:
            model: Model containing this node, used to look up the initializer.
            path: Code generation directory to write the files into.
            fpgapart: Target FPGA part (unused, the parameter layout is device
                independent).
        """
        weights = model.get_initializer(self.onnx_node.input[1])
        if isinstance(weights, np.ndarray):
            self.make_weight_file(weights, "decoupled_npy", f"{path}/input_1.npy")
            self.make_weight_file(weights, "decoupled_verilog_dat", f"{path}/memblock.dat")

    def make_weight_file(
        self, weights: np.ndarray, weight_file_mode: str, weight_file_name: str
    ) -> None:
        """Write the folded constant operand in the requested weight file format.

        Args:
            weights: Constant operand tensor.
            weight_file_mode: Either ``"decoupled_npy"`` or ``"decoupled_verilog_dat"``.
            weight_file_name: Output file path.
        """
        folded_weight_shape = self.get_folded_input_shape(1)
        weight_tensor = weights.reshape(folded_weight_shape).copy()

        # When broadcasting the last axis (rhs_shape[-1]==1), replicate the
        # scalar value across PE lanes so memstream provides PE values per cycle
        if self.broadcast_last_axis and weight_tensor.shape[-1] == 1:
            weight_tensor = np.tile(
                weight_tensor, (1,) * (len(weight_tensor.shape) - 1) + (self.pe,)
            )

        # Reshape to compact 3D form for decoupled modes
        weight_tensor = weight_tensor.reshape(1, -1, weight_tensor.shape[-1]).copy()

        if weight_file_mode == "decoupled_npy":
            np.save(weight_file_name, weight_tensor)
            return

        # decoupled_verilog_dat
        export_wdt = self.get_input_datatype(1)
        weight_width = self.get_instream_width(1)
        weight_width_padded = roundup_to_integer_multiple(weight_width, 4)

        weight_tensor_hex = pack_innermost_dim_as_hex_string(
            weight_tensor,
            export_wdt,
            weight_width_padded,
            reverse_inner=True,
            prefix="",
        )

        weight_stream = weight_tensor_hex.flatten()
        with Path(weight_file_name).open("w") as f:
            for val in weight_stream:
                f.write(val + "\n")

    def calc_wmem(self) -> int:
        """Return the number of constant memory words."""
        return int(super().calc_wmem())

    def calc_wmem_reps(self) -> int:
        """Return how many times the compact parameter stream is replayed.

        Elementwise constants often broadcast across non-channel axes, for
        example a [C] scale applied to [N, T, C]. The memstream only needs to
        store the compact [C] stream; regular single-set memstreams cycle on
        their own, and FINNLoop multi-set memstreams use this value as the
        stream-tap repetition count for each loop iteration.

        Returns:
            Number of replays of the compact parameter stream per output stream.

        Raises:
            FINNInternalError: If the const stream length is invalid or does not
                divide the output stream length.
        """
        base_wmem = int(super().calc_wmem())
        num_w_reps = int(np.prod(self.calc_num_input_vectors()))
        if base_wmem <= 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: invalid const stream length {base_wmem}"
            )
        if num_w_reps % base_wmem != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: const stream length {base_wmem} "
                f"does not divide output stream length {num_w_reps}"
            )
        return num_w_reps // base_wmem

    def calc_num_input_vectors(self) -> list[int]:
        """Return the outer (non-channel) dimensions of the folded lhs input shape."""
        folded_lhs = self.get_folded_input_shape(0)
        if len(folded_lhs) >= 2:
            return list(folded_lhs[:-1])
        return [1]

    def _get_rtl_op_name(self) -> str:
        """Return the RTL operation name; override in subclasses."""
        raise NotImplementedError("Subclasses must implement _get_rtl_op_name")


@register_custom_op
class ElementwiseAdd_rtl(ElementwiseBinary_rtl, elementwise_binary.ElementwiseAdd):
    """RTL implementation of elementwise addition."""

    _operation = "Add", np.add, "({0} + {1})", '"ADD"'

    def _get_rtl_op_name(self) -> str:
        """Return the RTL operation name literal for addition."""
        return '"ADD"'


@register_custom_op
class ElementwiseSub_rtl(ElementwiseBinary_rtl, elementwise_binary.ElementwiseSub):
    """RTL implementation of elementwise subtraction."""

    _operation = "Sub", np.subtract, "({0} - {1})", '"SUB"'

    def _get_rtl_op_name(self) -> str:
        """Return the RTL operation name literal for subtraction."""
        return '"SUB"'


@register_custom_op
class ElementwiseMul_rtl(ElementwiseBinary_rtl, elementwise_binary.ElementwiseMul):
    """RTL implementation of elementwise multiplication."""

    _operation = "Mul", np.multiply, "({0} * {1})", '"MUL"'

    def _get_rtl_op_name(self) -> str:
        """Return the RTL operation name literal for multiplication."""
        return '"MUL"'

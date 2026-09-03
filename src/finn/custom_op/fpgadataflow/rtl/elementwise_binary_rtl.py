############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################
"""RTL backend implementation for elementwise binary operations."""

import numpy as np
import shutil
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import roundup_to_integer_multiple
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow import elementwise_binary
from finn.custom_op.fpgadataflow.elementwise_binary import ElementwiseBinaryOperation, NodeAttrTypes
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
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
    """Base CustomOp wrapper for the finn-rtllib eltwisef component."""

    def __init__(self, onnx_node: "NodeProto", **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
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
        from finn.transformation.fpgadataflow.loop_rolling import LoopBodyInputType

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
        """Generate hdl."""
        rhs_style = self.rhs_style
        mlo = self.get_nodeattr("mlo_max_iter")

        # MLO mode allows rhs to be "input" for parameter streaming
        if not mlo and rhs_style != "const":
            raise FINNUserError(
                f"{self.onnx_node.name}: rhs is not const input and MLO is not enabled. "
                "Try setting the preferred_impl_style to hls or enabling MLO"
            )

        if self.get_nodeattr("mem_mode") != "internal_decoupled":
            raise FINNUserError(
                f"{self.onnx_node.name}: only internal_decoupled mode is supported for "
                "rtl elementwise ops"
            )

        # eltwisef core operates on floats (all dtypes must be FLOAT32)
        for attr in ("lhs_dtype", "rhs_dtype", "out_dtype"):
            val = self.get_nodeattr(attr)
            if val != "FLOAT32":
                raise FINNUserError(
                    f"{self.onnx_node.name}: RTL elementwise requires FLOAT32 dtypes, "
                    f"got {attr}={val}"
                )

        code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))
        self.generate_params(model, code_gen_dir)

        rtlsrc = Path(get_settings().finn_rtllib) / "eltwisef"
        template_path = rtlsrc / "eltwisef_template.v"
        pe = self.pe

        code_gen_dict: dict[str, str | int | float] = {
            "TOP_MODULE_NAME": self.get_verilog_top_module_name(),
            "PE": pe,
            "OP": self._get_rtl_op_name(),
            "B_SCALE": 1.0,
            "FORCE_BEHAVIORAL": 0,
            "STREAM_BITS": pe * 32,
        }

        template = template_path.read_text()
        for key_name, value in code_gen_dict.items():
            template = template.replace(f"${key_name}$", str(value))

        (code_gen_dir / f"{self.get_verilog_top_module_name()}.v").write_text(template)

        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        self.generate_hdl_memstream(fpgapart)

        for sv_file in ["eltwisef.sv", "binopf.sv", "queue.sv"]:
            shutil.copy(rtlsrc / sv_file, code_gen_dir)
        self.set_nodeattr("ipgen_path", str(code_gen_dir))
        self.set_nodeattr("ip_path", str(code_gen_dir))

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        """Return rtl file list."""
        if abspath:
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen")) + "/"
            rtllib_dir = str(Path(get_settings().finn_rtllib) / "eltwisef") + "/"
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        top_module = cast("str", self.get_nodeattr("gen_top_module"))
        return [
            f"{rtllib_dir}eltwisef.sv",
            f"{rtllib_dir}binopf.sv",
            f"{rtllib_dir}queue.sv",
            f"{code_gen_dir}{top_module}.v",
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
        # (This includes MLO mode where adapt_for_loop_body changes rhs_style to "input")
        if self.rhs_style == "input":
            if cast("int", self.get_nodeattr("mlo_max_iter")) <= 0:
                raise FINNUserError(
                    f"{self.onnx_node.name}: rhs_style is 'input' but MLO is not enabled. "
                    "RTL elementwise ops require MLO to be enabled for input rhs_style."
                )
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
        node_name = self.onnx_node.name
        source_target = f"./ip/verilog/rtl_ops/{node_name}"
        cmd = [f"file mkdir {source_target}"]

        clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
        rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
        dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0][0]
        din_name = self.get_verilog_top_module_intf_names()["s_axis"][0][0]
        mlo = self.get_nodeattr("mlo_max_iter")

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

        # MLO mode: Create additional input interface for streamed parameters
        if mlo and self.rhs_style == "input":
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
        for f in sourcefiles:
            cmd += [f"add_files -copy_to {source_target} -norecurse {f}"]
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
        if mlo and self.rhs_style == "input":
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
            axilite_name = self.get_verilog_top_module_intf_names()["axilite"][0]
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
        """Append TCL commands instantiating the RTL core to ``cmd``."""
        node_name = self.onnx_node.name
        top_module = cast("str", self.get_nodeattr("gen_top_module"))
        source_target = f"./ip/verilog/rtl_ops/{node_name}"

        sourcefiles = self.get_rtl_file_list(abspath=True)

        for f in sourcefiles:
            cmd.append(f"add_files -copy_to {source_target} -norecurse {f}")

        # Always create the core inside the hierarchical wrapper
        cmd.append(f"create_bd_cell -type hier -reference {top_module} /{node_name}/{node_name}")

    def execute_node(self, context: dict[str, np.ndarray], graph: "GraphProto") -> None:
        """Execute node."""
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
        context[node.output[0]] = out.reshape(self.get_normal_output_shape(ind=0)).astype(
            np.float32
        )

    def generate_params(self, model: ModelWrapper, path: str | Path) -> None:
        """Generate params."""
        weights = model.get_initializer(self.onnx_node.input[1])
        if isinstance(weights, np.ndarray):
            self.make_weight_file(weights, "decoupled_npy", f"{path}/input_1.npy")
            self.make_weight_file(weights, "decoupled_verilog_dat", f"{path}/memblock.dat")

    def make_weight_file(
        self, weights: np.ndarray, weight_file_mode: str, weight_file_name: str
    ) -> None:
        """Create weight file."""
        folded_weight_shape = self.get_folded_input_shape(1)
        weight_tensor = weights.reshape(folded_weight_shape).copy()

        # When broadcasting the last axis (rhs_shape[-1]==1), replicate the
        # scalar value across PE lanes so memstream provides PE values per cycle
        if self.broadcast_last_axis and weight_tensor.shape[-1] == 1:
            weight_tensor = np.tile(
                weight_tensor, (1,) * (len(weight_tensor.shape) - 1) + (self.pe,)
            )

        if weight_file_mode == "decoupled_verilog_dat":
            num_w_reps = np.prod(self.calc_num_input_vectors())
            base_wmem = super().calc_wmem()
            mlo = self.get_nodeattr("mlo_max_iter")
            # In MLO mode, tile only enough to match per-iteration consumption
            # (num_w_reps entries). base_wmem entries already exist, so tile by
            # num_w_reps / base_wmem.
            tile_factor = int(num_w_reps // base_wmem) if mlo and base_wmem > 1 else int(num_w_reps)
            weight_tensor = np.tile(
                weight_tensor, (tile_factor,) + (1,) * (len(weight_tensor.shape) - 1)
            )

        export_wdt = self.get_input_datatype(1)
        weight_width = self.get_instream_width(1)
        weight_width_padded = roundup_to_integer_multiple(weight_width, 4)

        if weight_file_mode == "decoupled_verilog_dat":
            shape = weight_tensor.shape
            weight_tensor_hex = pack_innermost_dim_as_hex_string(
                weight_tensor.reshape(1, -1, shape[-1]),
                export_wdt,
                weight_width_padded,
                reverse_inner=True,
                prefix="",
            )
        else:
            weight_tensor_hex = pack_innermost_dim_as_hex_string(
                weight_tensor.reshape(1, -1, weight_tensor.shape[-1]),
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
        """Compute wmem."""
        base_wmem = super().calc_wmem()
        num_w_reps = np.prod(self.calc_num_input_vectors())
        mlo = self.get_nodeattr("mlo_max_iter")
        if mlo:
            return int(num_w_reps)
        return int(base_wmem * num_w_reps)

    def calc_num_input_vectors(self) -> list[int]:
        """Compute the number of input vectors from the folded lhs shape."""
        folded_lhs = self.get_folded_input_shape(0)
        if len(folded_lhs) >= 2:
            return list(folded_lhs[:-1])
        return [1]

    def minimize_weight_bit_width(self, model: ModelWrapper) -> None:
        """Return minimize weight bit width."""
        super().minimize_weight_bit_width(model)

    def _get_rtl_op_name(self) -> str:
        """Return the RTL operation name; override in subclasses."""
        raise NotImplementedError("Subclasses must implement _get_rtl_op_name")


class ElementwiseAdd_rtl(ElementwiseBinary_rtl, elementwise_binary.ElementwiseAdd):
    """RTL implementation of elementwise addition for FLOAT32."""

    _operation = "Add", np.add, "({0} + {1})", '"ADD"'

    def _get_rtl_op_name(self) -> str:
        """Return rtl op name."""
        return '"ADD"'


class ElementwiseSub_rtl(ElementwiseBinary_rtl, elementwise_binary.ElementwiseSub):
    """RTL implementation of elementwise subtraction for FLOAT32."""

    _operation = "Sub", np.subtract, "({0} - {1})", '"SUB"'

    def _get_rtl_op_name(self) -> str:
        """Return rtl op name."""
        return '"SUB"'


class ElementwiseMul_rtl(ElementwiseBinary_rtl, elementwise_binary.ElementwiseMul):
    """RTL implementation of elementwise multiplication for FLOAT32."""

    _operation = "Mul", np.multiply, "({0} * {1})", '"MUL"'

    def _get_rtl_op_name(self) -> str:
        """Return rtl op name."""
        return '"MUL"'

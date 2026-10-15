# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""RTL backend implementation of the elementwise Where layer."""

import numpy as np
from onnx import NodeProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow.hwwhere import HWWhere, NodeAttrTypes
from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from onnx import GraphProto


def _rtlsrc_dir() -> Path:
    """Return the directory of the finn-rtllib where HDL sources."""
    return Path(get_settings().finn_rtllib) / "where" / "hdl"


@register_custom_op
class HWWhere_rtl(HWWhere, RTLBackend):
    """RTL implementation of the ONNX Where operator with multidirectional broadcasting."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return node attribute types, combining HWWhere and RTLBackend attributes."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(HWWhere.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def _shape_literal(self, shape: tuple[int, ...]) -> str:
        """Return the shape as a SystemVerilog unpacked-array literal string."""
        rtl_shape = self._rtl_shape(shape)
        return "'{ " + ", ".join(str(int(x)) for x in rtl_shape) + " }"

    def _pad_shape_to_rank(self, shape: tuple[int, ...], rank: int) -> tuple[int, ...]:
        """Return the shape left-padded with ones to the given rank."""
        rtl_shape = self._rtl_shape(shape)
        if len(rtl_shape) > rank:
            raise FINNUserError("Input rank must not exceed output rank")
        return (1,) * (rank - len(rtl_shape)) + rtl_shape

    def generate_hdl(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:  # noqa: ARG002
        """Render the where wrapper and core templates and copy the
        RTL sources into the codegen dir.
        """
        pe = self._output_stream_pe()
        out_shape = self.get_normal_output_shape()
        cond_shape = self.get_normal_input_shape(0)
        x_shape = self.get_normal_input_shape(1)
        y_shape = self.get_normal_input_shape(2)
        out_rtl_shape = self._rtl_shape(out_shape)
        ndims = len(out_rtl_shape)
        cond_rtl_shape = self._pad_shape_to_rank(cond_shape, ndims)
        x_rtl_shape = self._pad_shape_to_rank(x_shape, ndims)
        y_rtl_shape = self._pad_shape_to_rank(y_shape, ndims)
        if out_rtl_shape[-1] % pe != 0:
            raise FINNUserError("PE must divide the output innermost dimension")

        rtlsrc = _rtlsrc_dir()
        template = (rtlsrc / "where_template.v").read_text()
        # Shape parameters use SystemVerilog unpacked-array literals. Keep the
        # external AXI wrapper in Verilog and render a separate SystemVerilog
        # core wrapper to pass those literals into the where module.
        core_template = (rtlsrc / "where_core_template.sv").read_text()

        topname = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", topname)

        elem_width = self.get_input_datatype(1).bitwidth()
        cond_width = self.get_instream_width(0)
        x_width = self.get_instream_width(1)
        y_width = self.get_instream_width(2)
        out_width = self.get_outstream_width(0)
        code_gen_dict = {
            "TOP_MODULE_NAME": topname,
            "PE": pe,
            "DATA_WIDTH": elem_width,
            "NDIMS": ndims,
            "OUT_SHAPE": self._shape_literal(out_rtl_shape),
            "COND_SHAPE": self._shape_literal(cond_rtl_shape),
            "X_SHAPE": self._shape_literal(x_rtl_shape),
            "Y_SHAPE": self._shape_literal(y_rtl_shape),
            "COND_WIDTH": cond_width,
            "X_WIDTH": x_width,
            "Y_WIDTH": y_width,
            "OUT_WIDTH": out_width,
            "RAM_STYLE": f'"{self.ram_style}"',
        }

        for key, value in code_gen_dict.items():
            template = template.replace(f"${key}$", str(value))
            core_template = core_template.replace(f"${key}$", str(value))

        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
        (Path(code_gen_dir) / f"{topname}.v").write_text(template)
        (Path(code_gen_dir) / f"{topname}_core.sv").write_text(core_template)

        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        """Return the list of RTL files (core sources and generated wrappers) for this node."""
        if abspath:
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen")) + "/"
            rtllib_dir = str(_rtlsrc_dir()) + "/"
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        gen_top_module = cast("str", self.get_nodeattr("gen_top_module"))
        return [
            rtllib_dir + "input_gen.sv",
            rtllib_dir + "where.sv",
            code_gen_dir + gen_top_module + "_core.sv",
            code_gen_dir + gen_top_module + ".v",
        ]

    def code_generation_ipi(self) -> list[str]:
        """Construct and return the TCL for node instantiation in Vivado IPI."""
        cmd = [f"add_files -norecurse {f}" for f in self.get_rtl_file_list(abspath=True)]
        cmd.append(
            f"create_bd_cell -type module -reference {self.get_nodeattr('gen_top_module')} "
            f"{self.onnx_node.name}"
        )
        return cmd

    def execute_node(self, context: dict[str, np.ndarray], graph: "GraphProto") -> None:
        """Execute the node via Python (cppsim) or rtlsim depending on exec_mode."""
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            HWWhere.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)
        else:
            raise FINNInternalError(
                f"Invalid value for attribute exec_mode! Is currently set to: {mode} "
                'has to be set to one of the following values ("cppsim", "rtlsim")'
            )

############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################
"""RTL backend implementation of the parallel 2D transpose (inner shuffle)."""

import numpy as np
from onnx import NodeProto
from pathlib import Path
from qonnx.core.datatype import BaseDataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow.hwcustomop import NodeAttrTypes
from finn.custom_op.fpgadataflow.inner_shuffle import InnerShuffle
from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.basic import fifo_rtl_files
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from onnx import GraphProto


@register_custom_op
class InnerShuffle_rtl(InnerShuffle, RTLBackend):
    """CustomOp wrapper for the finn-rtllib inner_shuffle component."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(InnerShuffle.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def get_template_values(
        self, idims: list[int], simd: int, dt: BaseDataType
    ) -> dict[str, str | int]:
        """Return template values."""
        return {
            "TOP_MODULE_NAME": self.get_verilog_top_module_name(),
            "I": idims[0],
            "J": idims[1],
            "SIMD": simd,
            "WIDTH": dt.bitwidth(),
            "STREAM_BITS": simd * dt.bitwidth(),
        }

    def generate_hdl(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:  # noqa: ARG002
        """Generate hdl."""
        rtlsrc = Path(get_settings().finn_rtllib) / "inner_shuffle"
        template_path = rtlsrc / "inner_shuffle_template.v"
        code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))
        code_gen_dict = self.get_template_values(
            [self.transpose_in_shape[-2], self.transpose_in_shape[-1]], self.simd, self.dtype
        )
        template = template_path.read_text()
        for key_name, value in code_gen_dict.items():
            template = template.replace(f"${key_name}$", str(value))

        (code_gen_dir / f"{self.get_verilog_top_module_name()}.v").write_text(template)

        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        self.set_nodeattr("ipgen_path", str(code_gen_dir))
        self.set_nodeattr("ip_path", str(code_gen_dir))

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        """Return rtl file list."""
        if abspath:
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen")) + "/"
            rtllib_dir = str(Path(get_settings().finn_rtllib) / "inner_shuffle") + "/"
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        top_module = cast("str", self.get_nodeattr("gen_top_module"))
        return [
            f"{rtllib_dir}inner_shuffle.sv",
            f"{rtllib_dir}skid.sv",
            f"{rtllib_dir}elasticmem.sv",
            f"{code_gen_dir}{top_module}.v",
            *fifo_rtl_files(abspath),
        ]

    def code_generation_ipi(self) -> list[str]:
        """Construct and return the TCL for node instantiation in Vivado IPI."""
        cmd = [f"add_files -norecurse {f}" for f in self.get_rtl_file_list(abspath=True)]
        cmd.append(
            "create_bd_cell -type module -reference "
            f"{self.get_nodeattr('gen_top_module')} {self.onnx_node.name}"
        )
        return cmd

    def execute_node(self, context: dict[str, np.ndarray], graph: "GraphProto") -> None:
        """Execute node."""
        mode = self.get_nodeattr("exec_mode")
        if mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)
        else:
            InnerShuffle.execute_node(self, context, graph)

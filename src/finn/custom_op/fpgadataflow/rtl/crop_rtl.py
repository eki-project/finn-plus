# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""RTL backend implementation of the Crop layer."""

import numpy as np
from onnx import NodeProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow.crop import Crop, NodeAttrTypes
from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from onnx import GraphProto


def _rtlsrc_dir() -> Path:
    """Return the directory of the finn-rtllib crop HDL sources."""
    return Path(get_settings().finn_rtllib) / "crop" / "hdl"


@register_custom_op
class Crop_rtl(Crop, RTLBackend):
    """RTL implementation of Crop using the finn-rtllib crop core."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return node attribute types, combining Crop and RTLBackend attributes."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(Crop.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def _get_template_param_dict(self) -> dict[str, int | str]:
        """Return the crop-core template parameters for this node. Subclasses
        that reuse the crop core (e.g. SelectToken_rtl) override this to map
        their own geometry onto the crop parameters.
        """
        height, width = self.img_dim
        if height == 0:
            height = 1
        channels = self.num_channels
        simd = self.simd
        if channels % simd != 0:
            raise FINNUserError("SIMD must divide NumChannels")
        return {
            "H": height,
            "W": width,
            "CF": channels // simd,
            "FOLD_WIDTH": self.get_input_datatype().bitwidth() * simd,
            "CROP_N": self.crop_north,
            "CROP_E": self.crop_east,
            "CROP_S": self.crop_south,
            "CROP_W": self.crop_west,
        }

    def generate_hdl(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:  # noqa: ARG002
        """Render the crop wrapper template with this node's parameters into the codegen dir."""
        template = (_rtlsrc_dir() / "crop_template.v").read_text()

        topname = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", topname)
        code_gen_dict = self._get_template_param_dict()
        code_gen_dict["TOP_MODULE_NAME"] = topname
        for key, value in code_gen_dict.items():
            template = template.replace(f"${key}$", str(value))

        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
        (Path(code_gen_dir) / f"{topname}.v").write_text(template)
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        """Return the list of RTL files (crop core and generated wrapper) for this node."""
        gen_top_module = cast("str", self.get_nodeattr("gen_top_module"))
        if abspath:
            code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))
            return [
                str(_rtlsrc_dir() / "crop.sv"),
                str(code_gen_dir / f"{gen_top_module}.v"),
            ]
        return ["crop.sv", gen_top_module + ".v"]

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
            Crop.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)
        else:
            raise FINNInternalError('exec_mode must be either "cppsim" or "rtlsim"')

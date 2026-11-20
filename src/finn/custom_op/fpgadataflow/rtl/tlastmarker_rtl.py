# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""RTL implementation of the TLastMarker, which inserts or strips AXI-Stream TLAST."""

import numpy as np
import numpy.typing as npt
from collections.abc import Sequence
from pathlib import Path
from qonnx.core.datatype import BaseDataType, DataType
from typing import TYPE_CHECKING, Literal, cast

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp, NodeAttrTypes
from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.exception import FINNInternalError
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from onnx import GraphProto, NodeProto
    from qonnx.core.modelwrapper import ModelWrapper


def _rtlsrc_dir() -> Path:
    """Return the directory of the finn-rtllib lastmarker HDL sources."""
    return Path(get_settings().finn_rtllib) / "lastmarker" / "hdl"


@register_custom_op
class TLastMarker_rtl(RTLBackend):
    """RTL implementation of TLastMarker using a finn-rtllib counter core.

    Direction="out" inserts TLAST every NumIters beats.
    Direction="in" strips incoming TLAST (pure wire-through).
    """

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return node attribute types of the marker on top of the HWCustomOp/RTLBackend ones."""
        my_attrs: NodeAttrTypes = {
            # normal shape of input/output (optional, only needed for shape queries)
            "normal_shape": ("ints", False, []),
            # FINN DataType of input/output (optional, only needed for datatype queries)
            "dataType": ("s", False, ""),
            # number of beats per frame; TLAST=1 is generated on the last one
            "NumIters": ("i", True, 0),
            # direction: whether to insert ("out") or strip ("in") TLAST
            "Direction": ("s", False, "out", {"out", "in"}),
            # width of input-output data streams, in bits
            "StreamWidth": ("i", True, 0),
            # width of individual element in stream, in bits
            "ElemWidth": ("i", True, 0),
        }
        my_attrs.update(HWCustomOp.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Forward the input unchanged.

        TLAST is not part of the tensor data, so the marker is transparent in
        node-by-node execution regardless of exec_mode. Its effect is only
        visible in stitched-IP rtlsim.
        """
        i_name = self.onnx_node.input[0]
        o_name = self.onnx_node.output[0]
        context[o_name] = context[i_name]

    def generate_hdl(
        self, model: "ModelWrapper", fpgapart: str, clk: float  # noqa: ARG002
    ) -> None:
        """Render the lastmarker wrapper template into the code generation directory."""
        template = (_rtlsrc_dir() / "lastmarker_template.v").read_text()

        topname = self.get_verilog_top_module_name()
        self.set_nodeattr("gen_top_module", topname)

        direction = self.get_nodeattr("Direction")
        params = {
            "TOP_MODULE_NAME": topname,
            "DATA_WIDTH": str(self.get_nodeattr("StreamWidth")),
            "DIRECTION_OUT": "1" if direction == "out" else "0",
            "NUM_ITERS": str(self.get_nodeattr("NumIters")),
        }
        for key, value in params.items():
            template = template.replace(f"${key}$", value)

        code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))
        (Path(code_gen_dir) / f"{topname}.v").write_text(template)
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        """Return the list of RTL files (lastmarker core and generated wrapper) for this node."""
        gen_top_module = cast("str", self.get_nodeattr("gen_top_module"))
        if abspath:
            code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))
            return [
                str(_rtlsrc_dir() / "lastmarker_axi.sv"),
                str(code_gen_dir / f"{gen_top_module}.v"),
            ]
        return ["lastmarker_axi.sv", gen_top_module + ".v"]

    def code_generation_ipi(self) -> list[str]:
        """Construct and return the TCL for node instantiation in Vivado IPI."""
        cmd = [f"add_files -norecurse {f}" for f in self.get_rtl_file_list(abspath=True)]
        cmd.append(
            f"create_bd_cell -type module -reference {self.get_nodeattr('gen_top_module')} "
            f"{self.onnx_node.name}"
        )
        return cmd

    def make_shape_compatible_op(self, model: "ModelWrapper") -> "NodeProto":
        """Create a shape-compatible op (the marker does not change the shape)."""
        return super().make_shape_compatible_op(model)

    def infer_node_datatype(self, model: "ModelWrapper") -> None:
        """Do nothing: the marker does not change the datatype of the stream."""

    def get_number_output_values(self) -> int:
        """Return the number of output beats per frame."""
        return cast("int", self.get_nodeattr("NumIters"))

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return the FINN DataType of the stream, from the optional dataType attribute."""
        dtype = self.get_nodeattr("dataType")
        if not isinstance(dtype, str) or dtype == "":
            raise FINNInternalError(
                f"dataType attribute not set in {self.onnx_node.name}, cannot get the datatype"
            )
        return DataType[dtype]

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return the FINN DataType of the stream, from the optional dataType attribute."""
        return self.get_input_datatype()

    def get_normal_input_shape(
        self, ind: int = 0  # noqa: ARG002
    ) -> Sequence[int] | npt.NDArray[np.int_]:
        """Return the normal (unfolded) shape, from the optional normal_shape attribute."""
        normal_shape = cast("list[int]", self.get_nodeattr("normal_shape"))
        if len(normal_shape) == 0:
            raise FINNInternalError(
                f"normal_shape attribute not set in {self.onnx_node.name}, "
                "cannot get the normal shape"
            )
        return [int(d) for d in normal_shape]

    def get_normal_output_shape(
        self, ind: int = 0  # noqa: ARG002
    ) -> Sequence[int] | npt.NDArray[np.int_]:
        """Return the normal (unfolded) shape, from the optional normal_shape attribute."""
        return self.get_normal_input_shape()

    def get_folded_input_shape(self, ind: int = 0) -> tuple[Literal[1], int, int]:  # noqa: ARG002
        """Return the folded shape: NumIters beats of StreamWidth / ElemWidth elements."""
        stream_width = cast("int", self.get_nodeattr("StreamWidth"))
        elem_width = cast("int", self.get_nodeattr("ElemWidth"))
        n_packed_elems = stream_width // elem_width
        n_iters = cast("int", self.get_nodeattr("NumIters"))
        return (1, n_iters, n_packed_elems)

    def get_folded_output_shape(self, ind: int = 0) -> tuple[Literal[1], int, int]:  # noqa: ARG002
        """Return the folded shape: NumIters beats of StreamWidth / ElemWidth elements."""
        return self.get_folded_input_shape()

    def get_instream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return the input stream width in bits."""
        return cast("int", self.get_nodeattr("StreamWidth"))

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return the output stream width in bits."""
        return cast("int", self.get_nodeattr("StreamWidth"))

    def get_verilog_top_module_intf_names(self) -> dict[str, list[tuple[str, int]] | list[str]]:
        """Return the interface names of the generated top module."""
        intf_names = super().get_verilog_top_module_intf_names()
        stream_width = cast("int", self.get_nodeattr("StreamWidth"))
        intf_names["s_axis"] = [("in0_V", stream_width)]
        intf_names["m_axis"] = [("out0_V", stream_width)]
        return intf_names

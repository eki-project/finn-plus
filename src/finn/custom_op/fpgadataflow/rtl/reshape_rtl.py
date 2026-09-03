"""RTL backend implementation of the Reshape operator."""

import numpy as np
import shutil
from onnx import GraphProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from typing import cast

from finn.custom_op.fpgadataflow.reshape import NodeAttrTypes, Reshape
from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.exception import FINNInternalError
from finn.util.settings import get_settings


@register_custom_op
class Reshape_rtl(Reshape, RTLBackend):
    """RTL backend implementation of the Reshape operator.

    Implements an AXI pass-through via the data-width converter, which reduces
    to a no-op for identical input and output stream widths.
    """

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return custom node attributes with their types and default values."""
        attrs: NodeAttrTypes = {}
        attrs.update(Reshape.get_nodeattr_types(self))
        attrs.update(RTLBackend.get_nodeattr_types(self))
        return attrs

    def execute_node(self, context: dict[str, np.ndarray], graph: GraphProto) -> None:
        """Execute reshape operation (RTL simulation or Python fallback)."""
        if self.get_nodeattr("exec_mode") != "rtlsim":
            Reshape.execute_node(self, context, graph)
        else:
            RTLBackend.execute_node(self, context, graph)

    def generate_hdl(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:  # noqa: ARG002
        """Generate HDL code by filling in the verilog template."""
        # Path to RTL sources implementing the AXI pass-through operator via the
        # data-width converter (a no-op for identical input and output width).
        rtlsrc = Path(get_settings().finn_rtllib) / "dwc" / "hdl"
        template_path = rtlsrc / "dwc_template.v"

        top = self.get_verilog_top_module_name()
        code_gen_dict = {
            # Name of the top-level module to instantiate
            "TOP_MODULE_NAME": top,
            # Bitwidth of the input and output stream (same as this is passthrough)
            "IBITS": self.get_instream_width(),
            "OBITS": self.get_outstream_width(),
        }

        # Save top module name so we can refer to it after this node has been
        # renamed (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", top)

        code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))

        template = template_path.read_text()
        for placeholder, value in code_gen_dict.items():
            template = template.replace(f"${placeholder}$", str(value))
        (code_gen_dir / f"{top}.v").write_text(template)

        # Copy implementation files from the library into the instance code-gen directory
        shutil.copy(rtlsrc / "dwc.sv", code_gen_dir)
        shutil.copy(rtlsrc / "dwc_axi.sv", code_gen_dir)

        # Set ipgen_path and ip_path so that HLS-Synth transformation and
        # stitched_ip transformation do not complain
        self.set_nodeattr("ipgen_path", str(code_gen_dir))
        self.set_nodeattr("ip_path", str(code_gen_dir))

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        """Return list of RTL files required for this custom operation.

        Args:
            abspath: Whether to return absolute paths (default: False).

        Returns:
            List of paths pointing to required RTL files.

        Raises:
            FINNInternalError: If ``code_gen_dir_ipgen`` or ``gen_top_module``
             attributes are invalid.

        """
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") if abspath else ""
        top_name = self.get_nodeattr("gen_top_module")
        if not isinstance(code_gen_dir, str):
            raise FINNInternalError(
                f"code_gen_dir_ipgen attribute not set in {self.onnx_node.name}, "
                f"cannot get RTL file list"
            )
        if not isinstance(top_name, str) or top_name == "":
            raise FINNInternalError(
                f"gen_top_module attribute not set in {self.onnx_node.name}, "
                f"cannot get RTL file list"
            )

        return [
            str(Path(code_gen_dir) / "dwc.sv"),
            str(Path(code_gen_dir) / "dwc_axi.sv"),
            str(Path(code_gen_dir) / f"{top_name}.v"),
        ]

    def code_generation_ipi(self) -> list[str]:
        """Construct and return the TCL for node instantiation in Vivado IPI."""
        sourcefiles = self.get_rtl_file_list(abspath=True)

        cmd = [f"add_files -norecurse {f}" for f in sourcefiles]
        cmd += [
            "create_bd_cell -type module -reference "
            f"{self.get_nodeattr('gen_top_module')} {self.onnx_node.name}"
        ]
        return cmd

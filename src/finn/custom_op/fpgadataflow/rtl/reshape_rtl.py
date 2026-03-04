"""RTLBackend specialization of the Reshape operator."""

# Handling paths
import os

# Copying files and directories
import shutil

# The generic HW custom operator version of the operator as a base class
from finn.custom_op.fpgadataflow.reshape import Reshape

# Utility for registering RTLBackend HWCustomOp implementations into the module
# scope
from finn.custom_op.fpgadataflow.rtl import register_custom_op

# Base class for specializing HW operators as implemented via RTL
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend

# Logging and error handling in FINN
from finn.util.exception import FINNInternalError
from finn.util.settings import get_settings


@register_custom_op
class Reshape_rtl(Reshape, RTLBackend):
    """RTLBackend specialization of the Reshape operator"""

    def get_nodeattr_types(self):
        """Custom node attributes with their types and default values."""
        # Start from parent operator class attributes
        attrs = Reshape.get_nodeattr_types(self)
        # Add the HLSBackend default attributes on top
        attrs.update(RTLBackend.get_nodeattr_types(self))
        # Add/Specialize implementation specific attributes here...
        # Return the updated attributes dictionary
        return attrs

    def execute_node(self, context, graph):
        """Execute reshape operation (RTL simulation or Python fallback)."""
        if self.get_nodeattr("exec_mode") != "rtlsim":
            Reshape.execute_node(self, context, graph)
        else:
            RTLBackend.execute_node(self, context, graph)

    def generate_hdl(self, model, fpgapart, clk):
        """Generate HLD code by filling in the verilog template."""

        # Path to RTL sources implementing the AXI pass-through operator
        # Note: Implements AXI pass-through via the data width converter, which,
        # for identical input and output width, reduces to a no-op.
        rtlsrc = os.path.join(get_settings().finn_rtllib, "dwc", "hdl")
        # Path to the verilog template of the top-level module
        template = os.path.join(rtlsrc, "dwc_template.v")

        # Template arguments: Mapping template parameters to the instance values
        code_gen_dict = {
            # Name of the top-level module to instantiate
            "TOP_MODULE_NAME": (top := self.get_verilog_top_module_name()),
            # Bitwidth of the input and output stream (same as this is
            # passthrough)
            "IBITS": self.get_instream_width(),
            "OBITS": self.get_outstream_width(),
        }

        # Save top module name so we can refer to it after this node has been
        # renamed (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", top)

        # Directory for code generation outputs
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        # Load the code template and fill in the parameter values from the dict
        with open(template, "r") as f:
            template = f.read()
            for placeholder, value in code_gen_dict.items():
                template = template.replace(f"${placeholder}$", str(value))
            # Open the code generation file to save the filled-in template
            with open(os.path.join(code_gen_dir, f"{top}.v"), "w") as out:
                out.write(template)

        # Copy implementation files from the library into the instance code
        # generation dictionary
        # shutil.copy(os.path.join(rtlsrc, "passthru_axi.sv"), code_gen_dir)
        shutil.copy(os.path.join(rtlsrc, "dwc.sv"), code_gen_dir)
        shutil.copy(os.path.join(rtlsrc, "dwc_axi.sv"), code_gen_dir)

        # Set ipgen_path and ip_path so that HLS-Synth transformation and
        # stitched_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath: bool = False):
        """Return list of RTL files required for this custom operation.

        Args:
            abspath: Whether to return absolute paths (default: False).

        Returns:
            List of paths pointing to required RTL files.

        Raises:
            FINNInternalError: If code_gen_dir_ipgen or gen_top_module
             attributes are invalid.

        """
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") if abspath else ""

        top_name = self.get_nodeattr("gen_top_module")
        if type(code_gen_dir) is not str:
            raise FINNInternalError(
                f"code_gen_dir_ipgen attribute not set in"
                f" {self.onnx_node.name}, cannot get RTL file list"
            )
        if type(top_name) is not str or top_name == "":
            raise FINNInternalError(
                f"gen_top_module attribute not set in {self.onnx_node.name},"
                f" cannot get RTL file list"
            )

        return [
            # os.path.join(code_gen_dir, "passthru_axi.sv"),
            os.path.join(code_gen_dir, "dwc.sv"),
            os.path.join(code_gen_dir, "dwc_axi.sv"),
            os.path.join(code_gen_dir, f"{top_name}.v"),
        ]

    def code_generation_ipi(self):
        """Code generation for IP integration."""
        sourcefiles = self.get_rtl_file_list(abspath=True)

        cmd = []
        for f in sourcefiles:
            cmd += [f"add_files -norecurse {f}"]
        cmd += [
            "create_bd_cell -type module -reference "
            f"{self.get_nodeattr('gen_top_module')} {self.onnx_node.name}"
        ]
        return cmd

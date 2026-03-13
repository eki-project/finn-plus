"""Support for memory stream operations in FPGA dataflow."""

import os
from pathlib import Path
from typing import cast

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.basic import is_versal


class MemStreamSupport(HWCustomOp):
    """Custom Op for memory stream operations in FPGA dataflow."""

    def calc_tmem(self) -> int:
        """Abstract method to calculate threshold memory size.
        The default implementation raises NotImplementedError because
        some subclasses dont implement calc_tmem."""
        raise NotImplementedError()

    def calc_wmem(self) -> int:
        """Abstract method to calculate weight memory size.
        The default implementation raises NotImplementedError because
        some subclasses dont implement calc_wmem."""
        raise NotImplementedError()

    def generate_hdl_memstream(self, fpgapart: str, pumped_memory: int = 0) -> None:
        """Generate verilog code for memstream component.

        Currently utilized by MVAU, VVAU and HLS Thresholding layer.

        Args:
            fpgapart: Target FPGA part string.
            pumped_memory: Whether to use pumped memory (default: 0).

        """
        ops = ["MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl", "Thresholding_hls"]
        if self.onnx_node.op_type in ops or self.onnx_node.op_type.startswith("Elementwise"):
            template_path = (
                Path(os.environ["FINN_RTLLIB"]) / "memstream/hdl/memstream_wrapper_template.v"
            )
            mname = self.onnx_node.name
            if self.onnx_node.op_type.startswith("Thresholding"):
                depth = self.calc_tmem()
            else:
                depth = self.calc_wmem()
            padded_width = self.get_instream_width_padded(1)
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen"))

            ram_style = cast("str", self.get_nodeattr("ram_style"))
            init_file = str(Path(code_gen_dir) / "memblock.dat")
            if ram_style == "ultra" and not is_versal(fpgapart):
                init_file = ""
            code_gen_dict = {
                "$MODULE_NAME$": [mname],
                "$SETS$": ["1"],
                "$DEPTH$": [str(depth)],
                "$WIDTH$": [str(padded_width)],
                "$INIT_FILE$": [init_file],
                "$RAM_STYLE$": [ram_style],
                "$PUMPED_MEMORY$": [str(pumped_memory)],
            }
            # apply code generation to template
            with template_path.open() as f:
                template_wrapper = f.read()
            for key in code_gen_dict:
                # transform list into long string separated by '\n'
                code_gen_line = "\n".join(code_gen_dict[key])
                template_wrapper = template_wrapper.replace(key, code_gen_line)
            output_path = Path(code_gen_dir) / f"{mname}_memstream_wrapper.v"
            with output_path.open("w") as f:
                f.write(template_wrapper)

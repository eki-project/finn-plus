# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Transformation for out-of-context Vivado synthesis on stitched IP designs."""

from onnx import NodeProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from shutil import copy2

from finn.util.basic import make_build_dir
from finn.util.exception import FINNInternalError
from finn.util.fpgadataflow import is_hls_node
from finn.util.vivado import out_of_context_synth


def is_hls_float_op(node: NodeProto, model: ModelWrapper) -> bool:
    """Check if a node is an HLS operator with floating-point inputs."""
    if is_hls_node(node):
        for inp in node.input:
            if model.get_tensor_datatype(inp).name.startswith("FLOAT"):
                return True
    return False


class SynthOutOfContext(Transformation):
    """Run out-of-context Vivado synthesis on a stitched IP design."""

    def __init__(self, part: str, clk_period_ns: float, clk_name: str = "ap_clk") -> None:
        """Initialize the SynthOutOfContext transformation.

        Args:
            part: Target FPGA part for synthesis
            clk_period_ns: Clock period in nanoseconds
            clk_name: Clock signal name (default: "ap_clk")
        """
        super().__init__()
        self.part = part
        self.clk_period_ns = clk_period_ns
        self.clk_name = clk_name

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply out-of-context synthesis transformation to the model."""

        def file_to_basename(x: str | Path) -> str:
            """Extract basename from a file path."""
            return Path(x).resolve().name

        vivado_stitch_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
        top_module_name = model.get_metadata_prop("wrapper_filename")
        if vivado_stitch_proj_dir is None or top_module_name is None:
            raise FINNInternalError("Need stitched IP and wrapper filename metadata to be set.")
        top_module_name = file_to_basename(top_module_name).strip(".v")
        build_dir = make_build_dir("synth_out_of_context_")
        verilog_extensions = [".v", ".sv", ".vh"]
        with Path(vivado_stitch_proj_dir + "/all_verilog_srcs.txt").open() as f:
            all_verilog_srcs = f.read().split()
        for file in all_verilog_srcs:
            if any(file.endswith(x) for x in verilog_extensions):
                copy2(file, build_dir)
        # extract additional tcl commands to set up floating-point ips correctly
        float_ip_tcl = []
        for node in model.graph.node:
            if is_hls_float_op(node, model):
                code_gen_dir = getCustomOp(node).get_nodeattr("code_gen_dir_ipgen")
                verilog_path = Path(f"{code_gen_dir}/project_{node.name}/sol1/impl/verilog/")
                file_suffix = ".tcl"
                for fname in verilog_path.iterdir():
                    if fname.name.endswith(file_suffix):
                        float_ip_tcl.append(str(fname))
        ret = out_of_context_synth(
            build_dir, top_module_name, float_ip_tcl, self.part, self.clk_name, self.clk_period_ns
        )
        model.set_metadata_prop("res_total_ooc_synth", str(ret))
        return (model, False)

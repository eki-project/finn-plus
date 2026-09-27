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

"""Transformation for out-of-context place & route of stitched IP designs."""

from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from finn.util.exception import FINNInternalError
from finn.util.vivado import parse_ooc_synth_results, run_ooc_pnr


class SynthOutOfContext(Transformation):
    """Run out-of-context synthesis and place & route on the stitched IP design.

    Operates directly on the Vivado project created by ``CreateStitchedIP``
    (metadata ``vivado_stitch_proj``): synthesizes the design out-of-context if
    not done already, runs opt/place/route and writes utilization, timing and
    power reports which are parsed into the ``res_total_ooc_synth`` metadata
    property (see :func:`finn.util.vivado.parse_ooc_synth_results`).
    """

    def __init__(self, part: str, clk_period_ns: float, clk_name: str = "ap_clk") -> None:
        """Initialize the SynthOutOfContext transformation.

        Args:
            part: Target FPGA part (informational, the project already targets it)
            clk_period_ns: Clock period in nanoseconds
            clk_name: Clock port name of the stitched design (default: "ap_clk")
        """
        super().__init__()
        self.part = part
        self.clk_period_ns = clk_period_ns
        self.clk_name = clk_name

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply out-of-context synthesis and P&R to the stitched IP project."""
        vivado_stitch_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
        block_vlnv = model.get_metadata_prop("vivado_stitch_vlnv")
        if vivado_stitch_proj_dir is None or block_vlnv is None:
            raise FINNInternalError("Need stitched IP project and VLNV metadata to be set.")
        block_name = block_vlnv.split(":")[2]
        xpr_files = list(Path(vivado_stitch_proj_dir).glob("*.xpr"))
        if len(xpr_files) != 1:
            raise FINNInternalError(
                f"Expected exactly one Vivado project in {vivado_stitch_proj_dir}, "
                f"found {len(xpr_files)}."
            )
        prjname = xpr_files[0].stem
        report_dir = run_ooc_pnr(
            vivado_stitch_proj_dir, prjname, block_name, self.clk_period_ns, self.clk_name
        )
        ret = parse_ooc_synth_results(report_dir)
        if ret is None:
            raise FINNInternalError(
                f"Out-of-context P&R did not produce report files in {report_dir}."
            )
        ret["vivado_proj_folder"] = str(report_dir)
        ret["routed_dcp"] = str(Path(report_dir) / f"{block_name}_routed.dcp")
        model.set_metadata_prop("res_total_ooc_synth", str(ret))
        return (model, False)

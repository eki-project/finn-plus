# Copyright (c) 2021 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

"""Utilities for Vitis HLS synthesis and IP generation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from subprocess import CalledProcessError

from finn.util.basic import launch_process_helper, which
from finn.util.exception import FINNInternalError, FINNUserError


class CallHLS:
    """Call vitis_hls to run HLS build tcl scripts."""

    def __init__(
        self, tcl_script: str | Path, code_gen_dir: str | Path, ipgen_path: str | Path
    ) -> None:
        """Create a new HLS builder.

        Args:
            tcl_script: Path to the Tcl script that will be executed to build.
            code_gen_dir: Directory that contains the Tcl script (and sources).
            ipgen_path: Path to the IP Generation project.
        """
        self.tcl_script = Path(tcl_script)
        self.ipgen_path = Path(ipgen_path)
        self.code_gen_dir = Path(code_gen_dir)
        self.ipgen_script: Path | None = None
        if self.tcl_script not in self.code_gen_dir.iterdir():
            raise FINNInternalError(
                f"code_gen_dir {code_gen_dir} does not"
                f"seem to contain the Tcl script {tcl_script}"
            )

    def build(self) -> None:
        """Create and run a shell script to execute the Tcl script in code_gen_dir."""
        vivado_path = os.environ.get("XILINX_VIVADO")
        if vivado_path is None:
            raise FINNUserError("XILINX_VIVADO was not set but is required.")
        # xsi kernel lib name depends on Vivado version (renamed in 2024.2)
        match = re.search(r"\b(20\d{2})\.(1|2)\b", vivado_path)
        if match is None:
            raise FINNUserError(f"Could not find a version number in XILINX_VIVADO: {vivado_path}")
        year, minor = int(match.group(1)), int(match.group(2))
        if (year, minor) > (2024, 2):
            if which("vitis-run") is None:
                raise FINNUserError("vitis-run not found in PATH")
            vitis_cmd = f"vitis-run --mode hls --tcl {self.tcl_script}\n"
        else:
            if which("vitis_hls") is None:
                raise FINNUserError("vitis_hls was not found in PATH!")
            vitis_cmd = f"vitis_hls {self.tcl_script}\n"
        self.ipgen_script = self.code_gen_dir / "ipgen.sh"
        working_dir = Path.cwd()
        with self.ipgen_script.open("w") as f:
            f.write("#!/bin/bash \n")
            f.write("set -e\n")
            f.write(f"cd {self.code_gen_dir}\n")
            f.write(vitis_cmd)
            f.write(f"cd {working_dir}\n")
        try:
            launch_process_helper(["bash", str(self.ipgen_script)], print_stdout=False)
        except CalledProcessError as e:
            raise FINNUserError(
                f"HLS IP Generation failed. Logs and sources are in {self.code_gen_dir}"
            ) from e

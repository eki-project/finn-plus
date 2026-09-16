# Copyright (c) 2020, Xilinx, Inc.
# Copyright (C) 2024, Advanced Micro Devices, Inc.
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

"""Module for removing generated files produced by fpgadataflow nodes."""
import qonnx.custom_op.registry as registry
import shutil
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from typing import cast

from finn.util.exception import FINNUserError
from finn.util.fpgadataflow import is_fpgadataflow_node


class CleanUp(Transformation):
    """Remove any generated files for fpgadataflow nodes."""

    def __init__(self) -> None:
        """Initialize instance."""
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Remove generated PYNQ/IP-stitching projects and per-node codegen dirs."""
        # delete PYNQ project, if any
        vivado_pynq_proj_dir = model.get_metadata_prop("vivado_pynq_proj")
        if vivado_pynq_proj_dir is not None and Path(vivado_pynq_proj_dir).is_dir():
            shutil.rmtree(vivado_pynq_proj_dir)
        model.set_metadata_prop("vivado_pynq_proj", "")
        # delete IP stitching project, if any
        ipstitch_path = model.get_metadata_prop("vivado_stitch_proj")
        if ipstitch_path is not None and Path(ipstitch_path).is_dir():
            shutil.rmtree(ipstitch_path)
        model.set_metadata_prop("vivado_stitch_proj", "")
        for node in model.graph.node:
            op_type = node.op_type
            if is_fpgadataflow_node(node):
                try:
                    # lookup op_type in registry of CustomOps
                    inst = registry.getCustomOp(node)
                    # delete code_gen_dir from cppsim
                    code_gen_dir = cast("str", inst.get_nodeattr("code_gen_dir_cppsim"))
                    if Path(code_gen_dir).is_dir():
                        shutil.rmtree(code_gen_dir)
                    inst.set_nodeattr("code_gen_dir_cppsim", "")
                    inst.set_nodeattr("executable_path", "")
                    # delete code_gen_dir from ipgen and project folder
                    code_gen_dir = cast("str", inst.get_nodeattr("code_gen_dir_ipgen"))
                    ipgen_path = cast("str", inst.get_nodeattr("ipgen_path"))
                    if Path(code_gen_dir).is_dir():
                        shutil.rmtree(code_gen_dir)
                    if Path(ipgen_path).is_dir():
                        shutil.rmtree(ipgen_path)
                    inst.set_nodeattr("code_gen_dir_ipgen", "")
                    inst.set_nodeattr("ipgen_path", "")
                    # delete Java HotSpot Performance data log
                    for d_path in Path("/tmp/").iterdir():
                        if "hsperfdata" in d_path.name:
                            shutil.rmtree(d_path)

                except KeyError as e:
                    # exception if op_type is not supported
                    raise FINNUserError(
                        f"Custom op_type {op_type} is currently not supported."
                    ) from e
        return (model, False)

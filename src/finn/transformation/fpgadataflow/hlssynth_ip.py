"""HLSSynthIP transformation."""
# Copyright (C) 2020, Xilinx, Inc.
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

import qonnx.custom_op.registry as registry
from onnx import NodeProto
from pathlib import Path
from qonnx.transformation.base import NodeLocalTransformation
from typing import TYPE_CHECKING, Literal, cast

from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.fpgadataflow import is_hls_node
from finn.util.logging import log

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend


class HLSSynthIP(NodeLocalTransformation):
    """For each HLS node: generate IP block from code in folder
    that is referenced in node attribute "code_gen_dir_ipgen"
    and save path of generated project in node attribute "ipgen_path".
    All nodes in the graph must have the fpgadataflow backend attribute.
    Any nodes that already have a ipgen_path attribute pointing to a valid path
    will be skipped.

    This transformation calls Vitis HLS for synthesis, so it will run for
    some time (minutes to hours depending on configuration).

    * num_workers (int or None) number of parallel workers, see documentation in
      NodeLocalTransformation for more details.
    """

    def __init__(self, fpgapart: str | None = None, num_workers: int | None = None) -> None:
        """Initialize the transformation with the given FPGA part and number of workers."""
        self.fpgapart = fpgapart
        super().__init__(num_workers=num_workers)

    def applyNodeLocal(self, node: "NodeProto") -> tuple[NodeProto, Literal[False]]:  # noqa: N802
        """Apply the transformation to a single node.
        See documentation in NodeLocalTransformation for more details."""
        op_type = node.op_type
        if is_hls_node(node) or node.op_type == "FINNLoop":
            try:
                # lookup op_type in registry of CustomOps
                inst = cast("HLSBackend", registry.getCustomOp(node))
                # ensure that code is generated
                if inst.get_nodeattr("code_gen_dir_ipgen") == "":
                    raise FINNUserError(
                        "Node attribute 'code_gen_dir_ipgen' is empty. "
                        "Please run transformation PrepareIP first."
                    )
                ip_path = cast("str", inst.get_nodeattr("ipgen_path"))
                ip_path_p = Path(ip_path)
                if (
                    not (ip_path_p.is_dir() or ip_path_p.is_file())
                    or cast("str", inst.get_nodeattr("code_gen_dir_ipgen")) not in ip_path
                ):
                    # call the compilation function for this node
                    inst.ipgen_singlenode_code(self.fpgapart)
                else:
                    log.debug(f"Using cached IP for {node.name}")
                # ensure that executable path is now set
                if inst.get_nodeattr("ipgen_path") == "":
                    raise FINNInternalError(
                        "Transformation HLSSynthIP was not successful. "
                        "Node attribute 'ipgen_path' is empty."
                    )
            except KeyError:
                # exception if op_type is not supported
                raise FINNUserError(
                    f"Custom op_type {op_type} is currently not supported."
                ) from None
        return (node, False)

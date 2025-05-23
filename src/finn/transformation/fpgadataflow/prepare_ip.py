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

import os
import qonnx.custom_op.registry as registry
from qonnx.transformation.base import Transformation

from finn.util.basic import make_build_dir
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.util.logging import log


def _codegen_single_node(node, model, fpgapart, clk):
    """Calls C++ code generation for one node. Resulting code can be used
    to generate a Vivado IP block for the node."""

    op_type = node.op_type
    try:
        # lookup op_type in registry of CustomOps
        inst = registry.getCustomOp(node)
        # get the path of the code generation directory
        code_gen_dir = inst.get_nodeattr("code_gen_dir_ipgen")
        # ensure that there is a directory
        if code_gen_dir == "" or not os.path.isdir(code_gen_dir):
            code_gen_dir = make_build_dir(prefix="code_gen_ipgen_" + str(node.name) + "_")
            inst.set_nodeattr("code_gen_dir_ipgen", code_gen_dir)
            # ensure that there is generated code inside the dir
            inst.code_generation_ipgen(model, fpgapart, clk)
        else:
            log.info(f"Using pre-existing code for {node.name}")
    except KeyError:
        # exception if op_type is not supported
        raise Exception(f"Custom op_type {op_type} is currently not supported.")


class PrepareIP(Transformation):
    """Call custom implementation to generate code for single custom node
    and create folder that contains all the generated files.
    All nodes in the graph must have the fpgadataflow backend attribute and
    transformation gets additional arguments:

    * fpgapart (string)

    * clk in ns (int)

    Any nodes that already have a code_gen_dir_ipgen attribute pointing to a valid path
    will be skipped.

    Outcome if succesful: Node attribute "code_gen_dir_ipgen" contains path to folder
    that contains:

    * For HLS layers: generated C++ code that can be used to generate a Vivado IP block.
      The necessary subsequent transformation is HLSSynthIP.

    * For RTL layers: filled template verilog files that can be used to instantiate as
      module during IP stitching.

    """

    def __init__(self, fpgapart, clk):
        super().__init__()
        self.fpgapart = fpgapart
        self.clk = clk

    def apply(self, model):
        for node in model.graph.node:
            if is_hls_node(node) or is_rtl_node(node):
                _codegen_single_node(node, model, self.fpgapart, self.clk)
        return (model, False)

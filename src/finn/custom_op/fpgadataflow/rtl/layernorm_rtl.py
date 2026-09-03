############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright for portions of this file is held by AMD and Microsoft under
# MIT license as part of project Brainsmith.
# All other copyright is held by AMD and is provided under BSD-3-Clause license.
#
############################################################################

"""RTL backend implementation of the layer-normalization operator."""

import math
import numpy as np
import shutil
from onnx import NodeProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow.layernorm import LayerNorm, NodeAttrTypes
from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.exception import FINNInternalError
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from onnx import GraphProto

# finn-rtllib sources copied verbatim into the generated IP directory
_RTL_SOURCES = ["layernorm.sv", "queue.sv", "accuf.sv", "binopf.sv", "rsqrtf.sv"]


@register_custom_op
class LayerNorm_rtl(LayerNorm, RTLBackend):
    """RTL backend implementation of LayerNorm.

    Generates a thin Verilog wrapper around the finn-rtllib ``layernorm``
    component for hardware synthesis.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        my_attrs.update(LayerNorm.get_nodeattr_types(self))
        return my_attrs

    def _check_simd_divides_n(self, n: int) -> None:
        """Raise if SIMD does not divide the innermost input dimension ``n``."""
        if n % self.simd != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: SIMD ({self.simd}) must divide the innermost "
                f"input dimension ({n})"
            )

    def generate_hdl(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:  # noqa: ARG002
        """Generate hdl."""
        rtllib_dir = Path(get_settings().finn_rtllib) / "layernorm"
        template_path = rtllib_dir / "layernorm_wrapper_template.v"
        n = self.get_normal_input_shape()[-1]
        self._check_simd_divides_n(n)
        topname = self.get_verilog_top_module_name()
        code_gen_dict = {
            "$N$": int(n),
            "$SIMD$": self.simd,
            "$TOP_MODULE_NAME$": topname,
        }

        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", topname)

        code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))
        template = template_path.read_text()
        for key, value in code_gen_dict.items():
            template = template.replace(key, str(value))
        (code_gen_dir / f"{topname}.v").write_text(template)

        for sv_file in _RTL_SOURCES:
            shutil.copy(rtllib_dir / sv_file, code_gen_dir)
        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", str(code_gen_dir))
        self.set_nodeattr("ip_path", str(code_gen_dir))

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        """Return rtl file list."""
        if abspath:
            code_gen_dir = cast("str", self.get_nodeattr("code_gen_dir_ipgen")) + "/"
            rtllib_dir = str(Path(get_settings().finn_rtllib) / "layernorm") + "/"
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        top_module = cast("str", self.get_nodeattr("gen_top_module"))
        return [f"{rtllib_dir}{f}" for f in _RTL_SOURCES] + [f"{code_gen_dir}{top_module}.v"]

    def code_generation_ipi(self) -> list[str]:
        """Construct and return the TCL for node instantiation in Vivado IPI."""
        code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))
        top_module = cast("str", self.get_nodeattr("gen_top_module"))
        sourcefiles = [str(code_gen_dir / f) for f in [*_RTL_SOURCES, f"{top_module}.v"]]

        cmd = [f"add_files -norecurse {f}" for f in sourcefiles]
        cmd += [f"create_bd_cell -type module -reference {top_module} {self.onnx_node.name}"]
        return cmd

    def execute_node(self, context: dict[str, np.ndarray], graph: "GraphProto") -> None:
        """Execute node."""
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            LayerNorm.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)

    def get_exp_cycles(self) -> int:
        """Return exp cycles."""
        n = self.get_normal_input_shape()[-1]
        self._check_simd_divides_n(n)
        val_queue_len_0 = n // self.simd + math.ceil(math.log2(self.simd)) * 2 + 7
        val_queue_len_1 = n // self.simd + math.ceil(math.log2(self.simd)) * 2 + 24
        streaming = int(np.prod(self.get_normal_input_shape())) // self.simd
        return val_queue_len_0 + val_queue_len_1 + streaming + 5

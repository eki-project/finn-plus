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
"""Streaming FIFO hardware custom operator (inter-layer buffering)."""

import numpy as np
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow import register_custom_op
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.basic import is_versal
from finn.util.exception import FINNInternalError
from finn.util.logging import log
from finn.util.resource_models import _fifo_cost, _resolve

if TYPE_CHECKING:
    from onnx import GraphProto

    from finn.util.resource_models import FifoCost

# Type of the dictionary returned by get_nodeattr_types: maps attribute names to
# their (dtype, required, default[, allowed_values]) specification tuples
NodeAttrTypes = dict[
    str,
    tuple[str, bool, int | float | str | bool | np.ndarray | list]
    | tuple[str, bool, int | float | str | bool | np.ndarray | list, set | None],
]


@register_custom_op
class StreamingFIFO(HWCustomOp):
    """Hardware abstraction layer for a streaming FIFO buffering data between layers."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return node attribute types for depth, shapes, datatype and storage style."""
        my_attrs: NodeAttrTypes = dict(super().get_nodeattr_types())
        my_attrs.update(
            {
                # FIFO depth
                "depth": ("i", True, 0),
                # folded shape of input/output
                "folded_shape": ("ints", True, []),
                # normal shape of input/output
                "normal_shape": ("ints", True, []),
                # FINN DataTypes for inputs/outputs
                "dataType": ("s", True, ""),
                # requested FPGA resource for the storage, passed to fifo.sv (with srl
                # mapped to the RTL's "shift" token at the codegen boundary):
                # auto (its RAM_STYLE_EFF ladder decides), srl (SRL shift register),
                # block (BRAM), distributed (LUTRAM) or ultra (URAM, on UltraScale+/Versal)
                "ram_style": (
                    "s",
                    False,
                    "auto",
                    {"auto", "srl", "block", "distributed", "ultra"},
                ),
                # whether the maxcount occupancy output is exposed on the wrapper
                "depth_monitor": ("i", False, 0),
                # the FIFO does not need its own FIFOs
                "inFIFODepths": ("ints", False, [0]),
                "outFIFODepths": ("ints", False, [0]),
                "debug_log_path": ("s", False, ""),
            }
        )
        return my_attrs

    @property
    def depth(self) -> int:
        """Get the configured FIFO depth."""
        return cast("int", self.get_nodeattr("depth"))

    @property
    def folded_shape(self) -> list[int]:
        """Get the folded input/output shape."""
        return list(cast("list[int]", self.get_nodeattr("folded_shape")))

    @property
    def normal_shape(self) -> list[int]:
        """Get the normal (unfolded) input/output shape."""
        return list(cast("list[int]", self.get_nodeattr("normal_shape")))

    @property
    def dtype(self) -> BaseDataType:
        """Get the element data type."""
        return DataType[cast("str", self.get_nodeattr("dataType"))]

    @property
    def ram_style(self) -> str:
        """Get the requested storage style (see get_ram_style() for the built one)."""
        return cast("str", self.get_nodeattr("ram_style"))

    @property
    def depth_monitor(self) -> int:
        """Get whether the maxcount occupancy output is exposed (0/1)."""
        return cast("int", self.get_nodeattr("depth_monitor"))

    def resolve_ram_style(self) -> str:
        """Predict which of srl/distributed/block/ultra fifo.sv will elaborate.

        generate_hdl() forwards ram_style to the RTL untouched, so this decides
        nothing: it reproduces the RAM_STYLE_EFF ladder so that resource estimation,
        the build report and the folding config describe what actually gets built.
        The ladder is a function of depth, width and the request alone, so unlike the
        estimators this needs no fpgapart.

        Returns:
            str: The storage style fifo.sv will build (srl, distributed, block or ultra).
        """
        requested = self.ram_style
        depth = self.depth
        width = self.get_instream_width_padded()
        style = cast("str", _resolve(depth, width, requested))
        # fifo.sv checks depth<=33 before it checks RAM_STYLE!=auto (fifo.sv:85-86), so
        # at this depth any explicit memory request is forced to a shift register
        # regardless. Warn for every explicit style (block/ultra/distributed alike) so
        # the override is not silent
        if depth <= 33 and requested not in ("auto", "srl"):
            log.warning(
                f"{self.onnx_node.name}: ram_style={requested} requested but depth {depth} "
                "<= 33 is built as a shift register regardless"
            )
        # srl is never auto-selected past 257, so a deeper one is an explicit
        # ram_style=srl request, whose name does not suggest the LUT cost it carries.
        if style == "srl" and depth > 257:
            log.warning(
                f"{self.onnx_node.name}: ram_style=srl at depth {depth} costs roughly "
                f"{_fifo_cost(depth, width, 'srl').lut} LUTs of shift register; consider "
                "distributed/block/ultra instead"
            )
        return style

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer and set the datatype from the input tensor and propagate it to the output."""
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            log.warning(
                f"inputDataType changing for {node.name}: {self.get_input_datatype()!s} -> {idt!s}"
            )
        self.set_nodeattr("dataType", idt.name)
        # data type stays the same
        model.set_tensor_datatype(node.output[0], idt)

    def get_normal_input_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return the unfolded input shape."""
        if self.depth < 1:
            raise FINNInternalError(f"{self.onnx_node.name}: FIFO depth ({self.depth}) is too low")
        return self.normal_shape

    def get_normal_output_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return the unfolded output shape (same as input)."""
        return self.get_normal_input_shape()

    def get_folded_input_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return the folded input shape."""
        return self.folded_shape

    def get_folded_output_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return the folded output shape (same as input)."""
        return self.folded_shape

    def get_instream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return the width of the input stream in bits."""
        return self.folded_shape[-1] * self.dtype.bitwidth()

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return the width of the output stream in bits."""
        return self.folded_shape[-1] * self.dtype.bitwidth()

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return the FINN DataType of the input."""
        return self.dtype

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return the FINN DataType of the output."""
        return self.dtype

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute the node in Python by passing the input through unchanged."""
        node = self.onnx_node
        context[node.output[0]] = context[node.input[0]]

    def get_ram_style(self) -> str:
        """Return the storage style this FIFO is built with.

        Derived on demand rather than stored: the style depends only on attributes this
        node already carries, so recomputing cannot go stale the way a recorded value
        would when set_fifo_depths changes the depth.
        """
        return self.resolve_ram_style()

    def get_fifo_cost(self, fpgapart: str) -> "FifoCost":
        """Return the FifoCost of this node as fifo.sv implements it.

        Needs the fpgapart because BRAM/URAM aspects differ on Versal.

        Args:
            fpgapart: The target FPGA part.

        Returns:
            FifoCost: The estimated (bram, uram, lut) cost.
        """
        return _fifo_cost(
            self.depth,
            self.get_instream_width_padded(),
            self.get_ram_style(),
            is_versal(fpgapart),
        )

    def bram_estimation(self, fpgapart: str) -> int:
        """Calculate resource estimation for BRAM."""
        return self.get_fifo_cost(fpgapart).bram

    def uram_estimation(self, fpgapart: str) -> int:
        """Calculate resource estimation for URAM."""
        return self.get_fifo_cost(fpgapart).uram

    def lut_estimation(self, fpgapart: str) -> int:
        """Calculate resource estimation for LUTs."""
        return self.get_fifo_cost(fpgapart).lut

    def bram_efficiency_estimation(self, fpgapart: str) -> float:
        """Return the fraction of estimated BRAM capacity actually used by the FIFO."""
        bram_est = self.bram_estimation(fpgapart)
        if bram_est == 0:
            return 1
        wbits = self.get_instream_width_padded() * self.depth
        return wbits / (bram_est * 18 * 1024)

    def uram_efficiency_estimation(self, fpgapart: str) -> float:
        """Return the fraction of estimated URAM capacity actually used by the FIFO."""
        # every URAM288 aspect holds 288 Kib, so this capacity is correct on the
        # Versal ladder too; narrow words show up as a smaller uram_estimation()
        uram_est = self.uram_estimation(fpgapart)
        if uram_est == 0:
            return 1
        wbits = self.get_instream_width_padded() * self.depth
        return wbits / (uram_est * 72 * 4096)

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

import math
import numpy as np
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow import register_custom_op
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.exception import FINNInternalError
from finn.util.logging import log

if TYPE_CHECKING:
    from onnx import GraphProto

# Type of the dictionary returned by get_nodeattr_types: maps attribute names to
# their (dtype, required, default[, allowed_values]) specification tuples
NodeAttrTypes = dict[
    str,
    tuple[str, bool, int | float | str | bool | np.ndarray | list]
    | tuple[str, bool, int | float | str | bool | np.ndarray | list, set | None],
]

_NOT_SPECIALIZED_MSG = (
    "is still in hw abstraction format, please run SpecializeLayers() before proceeding"
)


@register_custom_op
class StreamingFIFO(HWCustomOp):
    """Abstraction layer for HW implementation of a streaming FIFO."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
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
                # FPGA resource type for FIFOs when impl_style is vivado
                # auto -- let Vivado decide
                # block -- use BRAM
                # distributed -- use LUTRAM
                # ultra -- use URAM (on UltraScale+)
                "ram_style": (
                    "s",
                    False,
                    "auto",
                    {"auto", "block", "distributed", "ultra"},
                ),
                # whether depth monitoring is enabled (impl_style=rtl only)
                "depth_monitor": ("i", False, 0),
                # the FIFO does not need its own FIFOs
                "inFIFODepths": ("ints", False, [0]),
                "outFIFODepths": ("ints", False, [0]),
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
        """Get the FPGA resource type used for the FIFO memory."""
        return cast("str", self.get_nodeattr("ram_style"))

    @property
    def depth_monitor(self) -> int:
        """Get whether depth monitoring is enabled (0/1)."""
        return cast("int", self.get_nodeattr("depth_monitor"))

    def get_adjusted_depth(self) -> int:
        """Return the FIFO depth adjusted for the backend (backend-specific)."""
        raise NotImplementedError

    def _resolve_depth(self) -> int:
        """Return the adjusted depth if the backend provides one, else the raw depth."""
        try:
            return self.get_adjusted_depth()
        except (AttributeError, NotImplementedError):
            return self.depth

    def _require_specialized(self) -> str:
        """Return impl_style, or raise if this node has not been specialized yet."""
        try:
            return cast("str", self.get_nodeattr("impl_style"))
        except AttributeError as exc:
            raise FINNInternalError(f"{self.onnx_node.name} {_NOT_SPECIALIZED_MSG}") from exc

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer node datatype."""
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            log.warning(
                f"inputDataType changing for {node.name}: {self.get_input_datatype()!s} -> {idt!s}"
            )
        self.set_nodeattr("dataType", idt.name)
        # data type stays the same
        model.set_tensor_datatype(node.output[0], idt)

    def get_verilog_top_module_intf_names(self) -> dict[str, list[tuple[str, int]] | list[str]]:
        """Return verilog top module intf names."""
        ret = super().get_verilog_top_module_intf_names()
        is_rtl = self._require_specialized() == "rtl"
        if is_rtl and self.depth_monitor == 1:
            ret["ap_none"] = ["maxcount"]
        return ret

    def get_normal_input_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return normal input shape."""
        depth = self._resolve_depth()
        if depth < 1:
            raise FINNInternalError(f"{self.onnx_node.name}: FIFO depth ({depth}) is too low")
        try:
            is_rtl = self.get_nodeattr("impl_style") == "rtl"
        except AttributeError:
            is_rtl = False
        if depth > 256 and is_rtl:
            log.warning("Depth is high, set between 2 and 256 for efficient SRL implementation")
        return self.normal_shape

    def get_normal_output_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return normal output shape."""
        return self.get_normal_input_shape()

    def get_folded_input_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return folded input shape."""
        return self.folded_shape

    def get_folded_output_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return folded output shape."""
        return self.folded_shape

    def get_instream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return instream width."""
        return self.folded_shape[-1] * self.dtype.bitwidth()

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return outstream width."""
        return self.folded_shape[-1] * self.dtype.bitwidth()

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return input datatype."""
        return self.dtype

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return output datatype."""
        return self.dtype

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute node."""
        node = self.onnx_node
        context[node.output[0]] = context[node.input[0]]

    def bram_estimation(self) -> int:
        """Calculate resource estimation for BRAM."""
        # NOTE: pre-existing behaviour - ``is_rtl`` is a bool, so the string
        # comparisons in the guard below are always False and the early
        # ``return 0`` is effectively dead. Preserved to keep the estimate
        # unchanged; only the unspecialized-node check is load-bearing.
        is_rtl = self._require_specialized() == "rtl"
        ram_type = self.ram_style
        depth = self._resolve_depth()
        w = self.get_instream_width()

        if is_rtl == "rtl" or (is_rtl == "vivado" and ram_type != "block"):
            # Non-BRAM based implementation
            return 0

        if w == 1:
            return math.ceil(depth / 16384)
        if w == 2:
            return math.ceil(depth / 8192)
        if w <= 4:
            return math.ceil(depth / 4096) * math.ceil(w / 4)
        if w <= 9:
            return math.ceil(depth / 2048) * math.ceil(w / 9)
        if w <= 18 or depth > 512:
            return math.ceil(depth / 1024) * math.ceil(w / 18)
        return math.ceil(depth / 512) * math.ceil(w / 36)

    def uram_estimation(self) -> int:
        """Calculate resource estimation for URAM."""
        # NOTE: see bram_estimation - the guard is pre-existing dead code.
        is_rtl = self._require_specialized() == "rtl"
        ram_type = self.ram_style
        depth = self._resolve_depth()
        w = self.get_instream_width()

        if is_rtl == "rtl" or (is_rtl == "vivado" and ram_type != "ultra"):
            # Non-URAM based implementation
            return 0
        return math.ceil(depth / 4096) * math.ceil(w / 72)

    def bram_efficiency_estimation(self) -> float:
        """Return bram efficiency estimation."""
        depth = self._resolve_depth()
        w = self.get_instream_width()
        bram16_est = self.bram_estimation()
        if bram16_est == 0:
            return 1.0
        wbits = w * depth
        bram16_est_capacity = bram16_est * 36 * 512
        return wbits / bram16_est_capacity

    def lut_estimation(self) -> int:
        """Calculate resource estimation for LUTs."""
        # NOTE: see bram_estimation - the ram_luts branch below is pre-existing
        # dead code, so ``ram_luts`` is always 0.
        is_rtl = self._require_specialized() == "rtl"
        ram_type = self.ram_style
        depth = self._resolve_depth()
        w = self.get_instream_width()

        address_luts = 2 * math.ceil(math.log(depth, 2))

        if is_rtl == "rtl" or (is_rtl == "vivado" and ram_type == "distributed"):
            ram_luts = math.ceil(depth / 32) * math.ceil(w / 2)
        else:
            ram_luts = 0

        return int(address_luts + ram_luts)

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Transformation assigning DDR address offsets to MLO loops and their weights."""
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.basic import roundup_to_integer_multiple
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.rtl.finn_loop import FINNLoop
    from finn.custom_op.fpgadataflow.rtl.matrixvectoractivation_rtl import MVAU_rtl


class AssignMemoryOffset(Transformation):
    """Assign non-overlapping DDR address offsets to FINNLoop frames and MVAU_rtl weights."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Walk the model and assign address offsets; the graph is never marked as modified."""
        self._offset = 0
        self._walk(model)
        return model, False

    def _walk(self, model: ModelWrapper) -> None:
        """Recursively assign 32-byte aligned offsets to DDR loops and MLO MVAU_rtl nodes."""
        for node in model.graph.node:
            if node.op_type == "FINNLoop":
                loop_inst = cast("FINNLoop", getCustomOp(node))
                if loop_inst.get_nodeattr("mem_type") != "DDR":
                    continue
                body_model = cast("ModelWrapper", loop_inst.get_nodeattr("body"))
                self._walk(body_model)
                loop_inst.set_nodeattr("body", body_model.graph)
                loop_inst.set_nodeattr("address_offset", self._offset)
                self._offset += loop_inst.intermediate_frame_bytes()
                self._offset = roundup_to_integer_multiple(self._offset, 32)
            else:
                inst = getCustomOp(node)
                mlo_max_iter = cast("int", inst.get_nodeattr("mlo_max_iter"))
                if not mlo_max_iter or node.op_type != "MVAU_rtl":
                    continue
                mvau_inst = cast("MVAU_rtl", inst)
                mvau_inst.set_nodeattr("address_offset", self._offset)
                self._offset += mlo_max_iter * mvau_inst.get_weight_mem_bytes()[1]
                self._offset = roundup_to_integer_multiple(self._offset, 32)

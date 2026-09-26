# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Transformation assigning DDR address offsets to MLO loops and their weights."""
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.basic import roundup_to_integer_multiple


class AssignMemoryOffset(Transformation):
    """Assign non-overlapping DDR address offsets to FINNLoop frames and MVAU_rtl weights."""

    def apply(self, model):
        """Walk the model and assign address offsets; the graph is never marked as modified."""
        self._offset = 0
        self._walk(model)
        return model, False

    def _walk(self, model):
        """Recursively assign 32-byte aligned offsets to DDR loops and MLO MVAU_rtl nodes."""
        for node in model.graph.node:
            if node.op_type == "FINNLoop":
                loop_inst = getCustomOp(node)
                if loop_inst.get_nodeattr("mem_type") != "DDR":
                    continue
                body_model = loop_inst.get_nodeattr("body")
                self._walk(body_model)
                loop_inst.set_nodeattr("body", body_model.graph)
                loop_inst.set_nodeattr("address_offset", self._offset)
                self._offset += loop_inst.intermediate_frame_bytes()
                self._offset = roundup_to_integer_multiple(self._offset, 32)
            else:
                inst = getCustomOp(node)
                mlo_max_iter = inst.get_nodeattr("mlo_max_iter")
                if not mlo_max_iter or not node.op_type == "MVAU_rtl":
                    continue
                inst.set_nodeattr("address_offset", self._offset)
                self._offset += mlo_max_iter * inst.get_weight_mem_bytes()[1]
                self._offset = roundup_to_integer_multiple(self._offset, 32)

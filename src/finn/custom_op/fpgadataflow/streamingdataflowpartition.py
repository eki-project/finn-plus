# Copyright (c) 2020 Xilinx, Inc.
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

"""Container node grouping a partitioned FINN-ONNX dataflow sub-model."""

import numpy as np
from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.base import CustomOp
from typing import TYPE_CHECKING, cast

from finn.core.onnx_exec import execute_onnx
from finn.custom_op.fpgadataflow import register_custom_op
from finn.util.exception import FINNInternalError

if TYPE_CHECKING:
    from onnx import GraphProto

# Type of the dictionary returned by get_nodeattr_types: maps attribute names to
# their (dtype, required, default[, allowed_values]) specification tuples
NodeAttrTypes = dict[
    str,
    tuple[str, bool, int | float | str | bool | np.ndarray | list]
    | tuple[str, bool, int | float | str | bool | np.ndarray | list, set | None],
]


@register_custom_op
class StreamingDataflowPartition(CustomOp):
    """Meta/container node for a group of fpgadataflow nodes.

    The grouped nodes have been separated out into a FINN-ONNX model of their
    own. This node is a placeholder only - it does not produce any HLS or
    bitfile by itself.
    """

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        return {
            "model": ("s", True, ""),
            "res_estimate": ("s", False, ""),
            "res_hls": ("s", False, ""),
            "res_synth": ("s", False, ""),
            "slr": ("i", False, -1),
            "partition_id": ("i", False, 0),
            "device_id": ("i", False, 0),
            "mem_port": ("s", False, ""),
            "instance_name": ("s", False, ""),
            "return_full_exec_context": ("i", False, 0),
            "network_connections": ("strings", False, []),
        }

    def make_shape_compatible_op(self, model: ModelWrapper) -> NodeProto:  # noqa: ARG002
        """Not supported - StreamingDataflowPartition is a container node."""
        raise FINNInternalError(
            f"{self.onnx_node.name}: shape inference is not defined for "
            f"StreamingDataflowPartition container nodes"
        )

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Not supported - StreamingDataflowPartition is a container node."""

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute node by running the referenced partition sub-model."""
        model = ModelWrapper(cast("str", self.get_nodeattr("model")))
        return_full_exec_context = self.get_nodeattr("return_full_exec_context") == 1
        node = self.onnx_node
        inp_ctx = {k: v for k, v in context.items() if k in node.input}
        # inputs may have been renamed in partition
        for i, old_iname in enumerate(node.input):
            new_iname = model.graph.input[i].name
            if old_iname != new_iname:
                inp_ctx[new_iname] = inp_ctx.pop(old_iname)
        ret = execute_onnx(model, inp_ctx, return_full_exec_context)
        # outputs may have been renamed in partition
        for i, node_oname in enumerate(node.output):
            model_oname = model.graph.output[i].name
            context[node_oname] = ret[model_oname]
        # prefix and insert exec context entries
        if return_full_exec_context:
            model_onames = {x.name for x in model.graph.output}
            for tname in ret:
                if tname not in model_onames:
                    context[f"{node.name}_{tname}"] = ret[tname]

    def verify_node(self) -> list[str]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Verify node.

        Note: the qonnx ``CustomOp.verify_node`` base is annotated ``-> None`` but
        the FINN verification passes expect a list of messages.
        """
        info_messages = []

        # verify number of attributes
        num_of_attr = 1
        if len(self.onnx_node.attribute) == num_of_attr:
            info_messages.append("The number of attributes is correct")
        else:
            info_messages.append(
                f"""The number of attributes is incorrect,
            {self.onnx_node.op_type} should have {num_of_attr} attributes"""
            )
        # verify that all necessary attributes exist
        try:
            self.get_nodeattr("model")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                """The necessary attributes do not exist.
                StreamingDataflowPartition needs the following attribute(s):
                model"""
            )

        # verify the number of inputs
        if len(self.onnx_node.input) >= 1:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("StreamingDataflowPartition needs 1 data input")

        return info_messages

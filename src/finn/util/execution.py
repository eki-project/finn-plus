# Copyright (c) 2020, Xilinx
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

"""Utility functions for executing ONNX models in FINN.

This module contains functions for executing parent models containing
StreamingDataflowPartition nodes and other execution-related utilities.
"""

import numpy as np
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.core.onnx_exec import execute_onnx


def load_model_checkpoint(filename: str | Path) -> ModelWrapper:
    """Load given .onnx file and return ModelWrapper.

    Args:
        filename (str|Path): Path to the ONNX model file

    Returns:
        ModelWrapper: Loaded model

    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    if Path(filename).is_file():
        model = ModelWrapper(str(filename))
        return model
    raise FileNotFoundError(f"Model file {filename} not found")


def execute_parent(
    parent_path: str | Path,
    child_path: str | Path,
    input_tensor_npy: np.ndarray,
    return_full_ctx: bool = False,
    batch_child: bool = False,
) -> np.ndarray | dict[str, np.ndarray] | list[dict[str, np.ndarray]]:
    """Execute parent model containing a single StreamingDataflowPartition by
    replacing it with the model at child_path and return result.

    With ``batch_child=True`` and more than one input frame, the parent nodes keep
    their compiled batch-one shapes and run once per frame, while the partition child
    receives all preprocessed frames in a single execution.

    Args:
        parent_path (str|Path): Path to the parent ONNX model file
        child_path (str|Path): Path to the child ONNX model file to replace the partition
        input_tensor_npy (numpy.ndarray): Input tensor data
        return_full_ctx (bool): If True, return full execution context,
                               otherwise return only output tensor
        batch_child (bool): If True, execute the partition child once for all frames

    Returns:
        numpy.ndarray, dict or list of dict: Output tensor or full execution context. A
        batched execution with return_full_ctx returns one parent context per frame.
    """
    parent_model = load_model_checkpoint(str(parent_path))
    iname = parent_model.get_first_global_in()
    oname = parent_model.get_first_global_out()
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_inst = getCustomOp(sdp_node)
    sdp_inst.set_nodeattr("model", str(child_path))
    sdp_inst.set_nodeattr("return_full_exec_context", 1 if return_full_ctx else 0)

    batch_size = input_tensor_npy.shape[0]
    if not batch_child or batch_size == 1:
        ret = execute_onnx(parent_model, {iname: input_tensor_npy}, True)
        if return_full_ctx:
            return ret
        return ret[oname]

    # parent nodes ahead of the partition, one frame at a time
    sdp_index = parent_model.get_node_index(sdp_node)
    frame_contexts = []
    for batch_index in range(batch_size):
        frame_input = input_tensor_npy[batch_index : batch_index + 1]
        if sdp_index == 0:
            frame_context = parent_model.make_empty_exec_context()
            frame_context[iname] = frame_input
        else:
            frame_context = execute_onnx(
                parent_model,
                {iname: frame_input},
                True,
                end_node=parent_model.graph.node[sdp_index - 1],
            )
        frame_contexts.append(frame_context)

    # The partition only reads its declared inputs, so it does not need the full
    # parent execution contexts.
    partition_context = {
        name: np.concatenate([context[name] for context in frame_contexts], axis=0)
        for name in sdp_node.input
    }
    sdp_inst.execute_node(partition_context, parent_model.graph)

    for output_name in sdp_node.output:
        batched_output = partition_context[output_name]
        for batch_index, frame_context in enumerate(frame_contexts):
            frame_context[output_name] = batched_output[batch_index : batch_index + 1]

    # parent nodes behind the partition, again one frame at a time
    if sdp_index + 1 < len(parent_model.graph.node):
        suffix_start = parent_model.graph.node[sdp_index + 1]
        for batch_index, frame_context in enumerate(frame_contexts):
            resume_context = {
                name: value for name, value in frame_context.items() if value is not None
            }
            frame_contexts[batch_index] = execute_onnx(
                parent_model,
                resume_context,
                True,
                start_node=suffix_start,
            )

    if return_full_ctx:
        return frame_contexts
    return np.concatenate([context[oname] for context in frame_contexts], axis=0)

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
# Note: This transform is inspired by a transformation from Thomas Keller (ExpandNorms)
# and ExtractQuantScaleZeroPt from qonnx.
#
############################################################################

"""Module for extracting norm scale bias."""
import numpy as np
import numpy.typing as npt
from onnx import TensorProto
from onnx import helper as oh
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueParameterTensors, SortGraph
from qonnx.transformation.remove import RemoveIdentityOps
from typing import Any, cast

from finn.util.exception import FINNInternalError


class ExtractNormScaleBias(Transformation):
    """Extract LayerNormalization scale and bias into separate nodes
    and set initializers to 1 or 0 respectively."""

    def __init__(self) -> None:
        """Initialize instance."""
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply transformation."""
        graph = model.graph
        for node in graph.node:
            if node.op_type == "LayerNormalization":
                ln_node = node
                input_ln = node.input[0]
                scale_tensor = node.input[1]
                # bias input is optional input
                bias_tensor = node.input[2] if len(node.input) > 2 else None
                bias = model.get_initializer(bias_tensor) if bias_tensor is not None else None
                scale = model.get_initializer(scale_tensor)
                extract_scale = False
                extract_bias = False
                if scale is not None and (cast("npt.NDArray[Any]", scale) != 1).any():
                    extract_scale = True
                if bias is not None and np.any(cast("npt.NDArray[Any]", bias)):
                    extract_bias = True
                if (not extract_scale) and (not extract_bias):
                    continue
                act_shape = model.get_tensor_shape(input_ln)
                if act_shape is None:
                    raise FINNInternalError(f"Could not determine shape of tensor {input_ln}")
                last_node = ln_node
                final_output = ln_node.output[0]
                if extract_scale:
                    # create new Mul node that applies the scale
                    # Create new tensor
                    scale_act_in_name = model.make_new_valueinfo_name()
                    scale_act_in = oh.make_tensor_value_info(
                        scale_act_in_name, TensorProto.FLOAT, act_shape
                    )
                    last_node.output[0] = scale_act_in_name
                    graph.value_info.append(scale_act_in)
                    scale_node = oh.make_node(
                        "Mul", [scale_act_in_name, scale_tensor], [final_output]
                    )
                    graph.node.append(scale_node)
                    # important: when tracking a pointer to newly added nodes,
                    # ensure the item from the container is used, and not the
                    # make_node result -- those are different objects
                    # e.g. if we use last_node = scale_node below,
                    # this will point to the wrong object and cause bugs later
                    last_node = graph.node[-1]
                    # remove scale from LayerNorm node
                    new_scale_name = model.make_new_valueinfo_name()
                    model.set_initializer(new_scale_name, np.ones(act_shape[-1], dtype=np.float32))
                    ln_node.input[1] = new_scale_name
                if extract_bias:
                    if bias_tensor is None:
                        raise FINNInternalError(
                            "extract_bias set without a bias tensor on the LayerNormalization node"
                        )
                    # create new Add node that applies bias
                    # create new tensor
                    bias_act_in_name = model.make_new_valueinfo_name()
                    bias_act_in = oh.make_tensor_value_info(
                        bias_act_in_name, TensorProto.FLOAT, act_shape
                    )
                    graph.value_info.append(bias_act_in)
                    bias_node = oh.make_node("Add", [bias_act_in_name, bias_tensor], [final_output])
                    last_node.output[0] = bias_act_in_name
                    graph.node.append(bias_node)
                    # remove bias from LayerNorm node
                    new_bias_name = model.make_new_valueinfo_name()
                    model.set_initializer(new_bias_name, np.zeros(act_shape[-1], dtype=np.float32))
                    ln_node.input[2] = new_bias_name

                if extract_scale or extract_bias:
                    # since we used append() for new nodes, need to call
                    # SortGraph to ensure correct (topological) order
                    model = model.transform(SortGraph())
                    # Remove potential unity multiplications from alpha and beta attributes
                    model = model.transform(RemoveIdentityOps())
                    # Ensure unique parameter tensors
                    model = model.transform(GiveUniqueParameterTensors())
                    return model, True

        return model, False

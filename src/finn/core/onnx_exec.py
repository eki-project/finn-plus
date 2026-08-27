# Copyright (c) 2022, Xilinx, Inc.
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

"""ONNX model execution utilities for FINN.

This module provides enhanced ONNX model execution capabilities that extend
the base QONNX execution functionality with FINN-specific features, including
RTL simulation support and model debugging utilities.

Key Functions:
- execute_onnx: Enhanced ONNX execution with RTL simulation support
- execute_onnx_and_make_model: Create debug models with intermediate activations
- compare_execution: Compare outputs between two models
"""

import copy
import numpy as np
import qonnx.analysis.topology as ta
from collections.abc import Callable
from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx as execute_onnx_base
from typing import cast

from finn.core.rtlsim_exec import rtlsim_exec
from finn.util.exception import FINNInternalError

from finn.xsi import SimEngine


def execute_onnx(
    model: "ModelWrapper",
    input_dict: dict[str, np.ndarray],
    return_full_exec_context: bool = False,
    start_node: NodeProto | None = None,
    end_node: NodeProto | None = None,
    pre_hook: Callable[[SimEngine], None] | None = None,
    post_hook: Callable[[SimEngine], None] | None = None,
) -> dict[str, np.ndarray]:
    """Execute given ONNX ModelWrapper with given named inputs.
    If return_full_exec_context is False, a dict of named outputs is returned
    as indicated by the model.graph.output.
    If return return_full_exec_context is True, the full set of tensors used by
    the execution (including inputs, weights, activations and final outputs)
    will be returned as a dict.
    When start_node and end_node are set to None, the whole graph is executed.
    If they are set to particular ONNX nodes, only the subgraph between (and
    including) those nodes is executed.
    """
    # check if model has an execution mode set
    # if None, execute model node using the QONNX-provided execute_onnx impl
    # if set to "rtlsim" execute model using xsi
    model_exec_mode = model.get_metadata_prop("exec_mode")
    if (model_exec_mode is None) or (model_exec_mode == ""):
        return execute_onnx_base(model, input_dict, return_full_exec_context, start_node, end_node)
    if model_exec_mode == "rtlsim":
        # check sanity of model and then use stitched IP for rtlsim
        if not model.check_all_tensor_shapes_specified():
            raise Exception("Found unspecified tensor shapes, try infer_shapes")
        ret = model.analysis(ta.nodes_topologically_sorted)
        assert (
            ret["nodes_topologically_sorted"] is True
        ), """Nodes must be
        topologically sorted."""

        graph = model.graph
        # first, we need to make sure that every variable required by the graph has
        # some buffer associated with it. this includes graph inputs (which includes
        # the input data as well as the trained parameters) and the graph ValueInfo
        # (intermediate tensors between layers)
        # this is provided by the execution_context, which is a dict of np.ndarray
        execution_context = model.make_empty_exec_context()
        # fill in any inputs provided to this function
        for inp_name in input_dict.keys():
            if inp_name in execution_context:
                ex = execution_context[inp_name]
                if ex is None:
                    raise FINNInternalError(
                        f"Shape of input {inp_name} is None in the execution "
                        f"context, but an input value was provided."
                    )
                if ex.shape == input_dict[inp_name].shape:
                    execution_context[inp_name] = input_dict[inp_name]
                else:
                    raise FINNInternalError(
                        f"Shape mismatch for provided input {inp_name}: "
                        f"found {ex.shape!s} expected {input_dict[inp_name].shape!s} "
                    )

        # use stitched IP for rtlsim
        rtlsim_exec(model, cast("dict[str, np.ndarray]", execution_context), pre_hook, post_hook)
    else:
        raise FINNInternalError(
            """Metadata property "exec_mode" is set to an unknown value. Can be left
            unset or has to be set to "rtlsim" for execution using xsi!"""
        )

    if return_full_exec_context:
        if "" in execution_context:
            del execution_context[""]  # remove empty string entry if it exists
        return cast("dict[str, np.ndarray]", execution_context)
    # provide outputs as dict
    output_dict = {}
    for out_tensor in graph.output:
        out_name = out_tensor.name
        output_dict[out_name] = execution_context[out_name]
    return output_dict


def execute_onnx_and_make_model(
    model: "ModelWrapper", input_dict: dict[str, np.ndarray]
) -> ModelWrapper:
    """Execute given ONNX ModelWrapper with given named inputs and return a new
    ModelWrapper where an initializer is provided for each tensor as taken from
    the execution. This new model is useful for debugging, since it contains
    all the intermediate activation values."""
    # retrieve the full execution context
    execution_context = execute_onnx(model, input_dict, True)
    new_model = copy.deepcopy(model)
    # create value_info entries and initializers for everything
    for i in execution_context.keys():
        if i != "" and execution_context[i] is not None:
            new_model.set_initializer(i, execution_context[i])
    for vi in new_model.graph.value_info:
        new_model.graph.output.append(vi)
    return new_model


def compare_execution(
    model_a: "ModelWrapper",
    model_b: "ModelWrapper",
    input_dict: dict[str, np.ndarray],
    compare_fxn: Callable[[list | np.ndarray, list | np.ndarray], bool | np.bool_] = lambda x, y: (
        np.isclose(x, y, atol=1e-3).all()
    ),
) -> bool | np.bool_:
    """Execute two ONNX models and compare their outputs using given function.

    compare_fxn should take in two tensors and return a Boolean"""
    # compare values from first output tensors produced
    res_a = next(iter(execute_onnx(model_a, input_dict).items()))[1]
    res_b = next(iter(execute_onnx(model_b, input_dict).items()))[1]
    return compare_fxn(res_a, res_b)

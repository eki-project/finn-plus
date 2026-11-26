# Copyright (C) 2026, Paderborn University
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the custom build steps in finn.builder.custom_step_library."""

import pytest

import numpy as np
import qonnx.core.data_layout as dl
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.builder.custom_step_library.general import flatten_global_input


def make_mlp_input_model(first_op):
    """[1, 1, 4, 4] image -> (Reshape | Flatten | nothing) -> MatMul."""
    in_shape = [1, 1, 4, 4] if first_op else [1, 16]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 3])
    nodes = []
    if first_op == "Reshape":
        nodes.append(helper.make_node("Reshape", ["inp", "shape"], ["flat"]))
    elif first_op == "Flatten":
        nodes.append(helper.make_node("Flatten", ["inp"], ["flat"], axis=1))
    nodes.append(helper.make_node("MatMul", ["flat" if first_op else "inp", "W"], ["outp"]))
    graph = helper.make_graph(nodes, "mlp_input", [inp], [outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="mlp-input"))
    if first_op == "Reshape":
        model.set_initializer("shape", np.asarray([1, -1], dtype=np.int64))
    model.set_initializer("W", gen_finn_dt_tensor(DataType["INT4"], (16, 3)))
    model.set_tensor_datatype("inp", DataType["UINT8"])
    return model.transform(InferShapes())


@pytest.mark.util
@pytest.mark.parametrize("first_op", ["Reshape", "Flatten"])
def test_flatten_global_input(first_op):
    model = make_mlp_input_model(first_op)
    x = gen_finn_dt_tensor(DataType["UINT8"], (1, 1, 4, 4))
    y_ref = oxe.execute_onnx(model, {"inp": x})["outp"]

    model = flatten_global_input(model, None)
    assert [n.op_type for n in model.graph.node] == ["MatMul"]
    assert model.get_tensor_shape("inp") == [1, 16]
    assert model.get_tensor_datatype("inp") == DataType["UINT8"]
    assert model.get_tensor_layout("inp") == dl.NC
    assert model.get_initializer("shape") is None
    # flattening the image in row-major order is what the removed node did
    y = oxe.execute_onnx(model, {"inp": x.reshape(1, 16)})["outp"]
    assert np.array_equal(y, y_ref)


@pytest.mark.util
def test_flatten_global_input_without_reshape():
    model = make_mlp_input_model(None)
    before = model.model.SerializeToString()
    model = flatten_global_input(model, None)
    assert model.model.SerializeToString() == before

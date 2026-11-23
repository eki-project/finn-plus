"""MoveMulPastJoinAdd compares per-channel constants element-wise."""

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.reorder import MoveMulPastJoinAdd


def make_model(scale_a: np.ndarray, scale_b: np.ndarray) -> ModelWrapper:
    """Two Mul producers with constants ``scale_a`` and ``scale_b`` feeding an Add."""
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4, 8, 8])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 8, 8])
    nodes = [
        helper.make_node("Mul", ["inp", "scale_a"], ["a"]),
        helper.make_node("Mul", ["inp", "scale_b"], ["b"]),
        helper.make_node("Add", ["a", "b"], ["out"]),
    ]
    model = ModelWrapper(qonnx_make_model(helper.make_graph(nodes, "join", [inp], [out])))
    model.set_initializer("scale_a", scale_a.astype(np.float32))
    model.set_initializer("scale_b", scale_b.astype(np.float32))
    return model.transform(InferShapes())


@pytest.mark.streamline
@pytest.mark.parametrize("identical", [True, False])
def test_move_mul_past_join_add_array_constants(identical: bool) -> None:
    scale_a = np.arange(1, 5, dtype=np.float32).reshape(1, 4, 1, 1)
    scale_b = scale_a if identical else scale_a[:, ::-1]
    model = make_model(scale_a, scale_b)
    x = np.random.rand(1, 4, 8, 8).astype(np.float32)
    expected = oxe.execute_onnx(model, {"inp": x})["out"]
    moved = model.transform(MoveMulPastJoinAdd())
    op_types = [n.op_type for n in moved.graph.node]
    assert op_types == (["Add", "Mul"] if identical else ["Mul", "Mul", "Add"])
    assert np.allclose(oxe.execute_onnx(moved, {"inp": x})["out"], expected, atol=1e-5)

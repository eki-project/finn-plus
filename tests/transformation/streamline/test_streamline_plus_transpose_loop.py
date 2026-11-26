"""StreamlinePlus terminates on Transpose -> scalar Mul (eki-project/finn-plus#321)."""

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.streamline.reorder import (
    MoveTransposePastEltwise,
    MoveTransposePastJoinAdd,
)
from finn.transformation.streamline.streamline_plus import StreamlinePlus


def make_model(scalar: bool, consumer: str = "Sigmoid") -> ModelWrapper:
    """Transpose -> Mul(constant) -> consumer, with a scalar or a per-channel constant.

    A ``Sigmoid`` consumer can neither absorb the constant nor let it pass, which left
    ``MoveTransposePastEltwise`` and ``MoveScalarLinearPastInvariants`` moving the Transpose
    and the scalar past each other forever.
    """
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4, 8, 8])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 8, 8, 4])
    nodes = [
        helper.make_node("Transpose", ["inp"], ["t"], perm=[0, 2, 3, 1]),
        helper.make_node("Mul", ["t", "scale"], ["m"]),
        helper.make_node(consumer, ["m"], ["out"]),
    ]
    model = ModelWrapper(qonnx_make_model(helper.make_graph(nodes, "loop", [inp], [out])))
    scale = (
        np.array(0.5, dtype=np.float32) if scalar else np.random.rand(1, 1, 1, 4).astype(np.float32)
    )
    model.set_initializer("scale", scale)
    return model.transform(InferShapes())


def run(model: ModelWrapper, x: np.ndarray) -> np.ndarray:
    """Execute the model on ``x`` (tensors may have been renamed by the cleanup)."""
    inp, out = model.graph.input[0].name, model.graph.output[0].name
    return oxe.execute_onnx(model, {inp: x})[out]


@pytest.mark.streamline
@pytest.mark.parametrize("scalar", [True, False])
@pytest.mark.timeout(120)
def test_streamline_plus_terminates(scalar: bool) -> None:
    model = make_model(scalar)
    x = np.random.rand(1, 4, 8, 8).astype(np.float32)
    expected = run(model, x)
    streamlined = model.transform(StreamlinePlus())
    assert np.allclose(run(streamlined, x), expected, atol=1e-5)
    # a scalar constant stays behind the Transpose, a per-channel one is moved in front of it
    expected_ops = ["Transpose", "Mul", "Sigmoid"] if scalar else ["Mul", "Transpose", "Sigmoid"]
    assert [n.op_type for n in streamlined.graph.node] == expected_ops
    assert [d.dim_value for d in streamlined.graph.input[0].type.tensor_type.shape.dim] == [
        1,
        4,
        8,
        8,
    ]


@pytest.mark.streamline
def test_transpose_is_not_moved_past_scalar_constant() -> None:
    """MoveScalarLinearPastInvariants owns scalar constants; moving the Transpose past them
    would undo its work and the two transformations would alternate forever."""
    model = make_model(scalar=True)
    moved = model.transform(MoveTransposePastEltwise())
    assert [n.op_type for n in moved.graph.node] == ["Transpose", "Mul", "Sigmoid"]
    moved = make_model(scalar=False).transform(MoveTransposePastEltwise())
    assert [n.op_type for n in moved.graph.node] == ["Mul", "Transpose", "Sigmoid"]
    assert [d.dim_value for d in moved.graph.input[0].type.tensor_type.shape.dim] == [1, 4, 8, 8]


@pytest.mark.streamline
def test_transpose_moves_past_self_join() -> None:
    inp = helper.make_tensor_value_info("global_in", TensorProto.FLOAT, [1, 4, 8, 8])
    out = helper.make_tensor_value_info("global_out", TensorProto.FLOAT, [1, 8, 8, 4])
    nodes = [
        helper.make_node("Transpose", ["global_in"], ["t"], perm=[0, 2, 3, 1]),
        helper.make_node("Add", ["t", "t"], ["global_out"]),
    ]
    model = ModelWrapper(qonnx_make_model(helper.make_graph(nodes, "self_join", [inp], [out])))
    model = model.transform(InferShapes())
    x = np.random.rand(1, 4, 8, 8).astype(np.float32)
    moved = model.transform(MoveTransposePastJoinAdd())
    assert [n.op_type for n in moved.graph.node] == ["Add", "Transpose"]
    add = moved.graph.node[0]
    assert list(add.input) == ["global_in", "global_in"]
    assert np.allclose(run(moved, x), run(model, x))

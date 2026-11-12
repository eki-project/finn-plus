"""Tests of the ONNX Passes frontend build step (step_passes_frontend).

Operators of the ONNX Passes domain are unknown to QONNX and FINN: the step has to
either export them to their QONNX counterparts or fail with a clear error.
"""

import pytest

import numpy as np
import onnx_ir as ir
from onnx_passes.ops import DOMAIN as PASSES_DOMAIN
from onnx_passes.ops.qonnx import DOMAIN as QONNX_DOMAIN
from qonnx.core.modelwrapper import ModelWrapper

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.builder.passes import assert_exported, multithreshold_export_blocker, step_passes_frontend
from finn.util.exception import FINNFrontendExportError


def make_multithreshold_model(
    thresholds: np.ndarray, weights: np.ndarray | None, channels: int
) -> ir.Model:
    """Build a model with a single ONNX Passes MultiThreshold on a (1, channels) input.

    Thresholds are a constant, the step weights are a constant if given and a
    second graph input otherwise.
    """
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[1, channels])
    t = ir.val("thresholds", const_value=ir.tensor(thresholds.astype(np.float32)))

    inputs, initializers = [x], [t]

    if weights is not None:
        w = ir.val("weights", const_value=ir.tensor(weights.astype(np.float32)))
        initializers.append(w)
    else:
        w = ir.val("weights", dtype=ir.DataType.FLOAT, shape=[channels, 1])
        inputs.append(w)

    node = ir.Node(PASSES_DOMAIN, "MultiThreshold", [x, t, w], name="MultiThreshold_0")
    (y,) = node.outputs
    y.name, y.shape, y.type = "y", ir.Shape([1, channels]), ir.TensorType(ir.DataType.FLOAT)

    graph = ir.Graph(
        inputs,
        [y],
        nodes=[node],
        initializers=initializers,
        opset_imports={"": 20, PASSES_DOMAIN: 1},
        name="multithreshold",
    )

    return ir.Model(graph, ir_version=10)


def make_build_config(tmp_path) -> DataflowBuildConfig:
    """Build a minimal configuration for running the frontend step in isolation."""
    return DataflowBuildConfig(
        output_dir=str(tmp_path),
        synth_clk_period_ns=10.0,
        generate_outputs=[],
        save_intermediate_models=True,
    )


PER_CHANNEL = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])


@pytest.mark.transform
@pytest.mark.parametrize(
    "thresholds, weights, reason",
    [
        (PER_CHANNEL, np.ones((3, 1)), None),
        (PER_CHANNEL[np.newaxis], np.ones((3, 1)), None),
        (np.array([0.5]), np.ones((3, 1)), None),
        (np.broadcast_to(PER_CHANNEL, (2, 3, 2)), np.ones((3, 1)), "not per-channel"),
        (PER_CHANNEL, np.array([[1.0], [0.0], [1.0]]), "not all 1"),
        (PER_CHANNEL, np.array([[2.0], [2.0], [2.0]]), "not all 1"),
        (PER_CHANNEL, None, "not constant"),
    ],
)
def test_multithreshold_export_blocker(thresholds, weights, reason):
    """The export rule condition explains per-element thresholds and non-unit weights."""
    model = make_multithreshold_model(thresholds, weights, channels=3)
    (node,) = model.graph
    blocker = multithreshold_export_blocker(node.inputs[1], node.inputs[2])

    if reason is None:
        assert blocker is None
    else:
        assert blocker is not None and reason in blocker


@pytest.mark.transform
def test_assert_exported_reports_remaining_passes_ops(tmp_path):
    """Every remaining ONNX Passes operator is named together with the reason."""
    model = make_multithreshold_model(PER_CHANNEL, np.array([[1.0], [0.0], [1.0]]), channels=3)
    (node,) = model.graph
    extra = ir.Node(PASSES_DOMAIN, "Any", [node.outputs[0]], name="Any_0")
    model.graph.append(extra)

    save_to = tmp_path / "failed.onnx"

    with pytest.raises(FINNFrontendExportError) as excinfo:
        assert_exported(model, save_to)

    message = str(excinfo.value)
    assert "2 operator(s)" in message
    assert "MultiThreshold 'MultiThreshold_0'" in message and "not all 1" in message
    assert "Any 'Any_0'" in message and "no export rule" in message
    assert str(save_to) in message and save_to.exists()

    # Nodes in other domains are fine, including the QONNX export targets
    node.domain, extra.domain = QONNX_DOMAIN, ""
    assert_exported(model, save_to)


@pytest.mark.transform
def test_step_passes_frontend_rejects_unexported_multithreshold(tmp_path):
    """A MultiThreshold with non-unit step weights fails the step with a clear error."""
    weights = np.array([[1.0], [0.0], [1.0]])
    model = ModelWrapper(ir.to_proto(make_multithreshold_model(PER_CHANNEL, weights, channels=3)))

    with pytest.raises(FINNFrontendExportError, match="not all 1"):
        step_passes_frontend(model, make_build_config(tmp_path))

    assert (tmp_path / "intermediate_models" / "step_passes_frontend_failed.onnx").exists()


@pytest.mark.transform
def test_step_passes_frontend_exports_multithreshold(tmp_path):
    """A per-channel MultiThreshold with unit step weights is exported to QONNX."""
    model = ModelWrapper(ir.to_proto(make_multithreshold_model(PER_CHANNEL, np.ones((3, 1)), 3)))

    model = step_passes_frontend(model, make_build_config(tmp_path))

    nodes = [node for node in model.graph.node if node.op_type == "MultiThreshold"]
    assert len(nodes) == 1 and nodes[0].domain == QONNX_DOMAIN
    assert not any(node.domain == PASSES_DOMAIN for node in model.graph.node)
    assert model.get_tensor_datatype(nodes[0].output[0]).bitwidth() == 2

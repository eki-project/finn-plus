"""Tests for the ``unmapped_layer_report`` analysis pass and the typed error that
``step_create_dataflow_partition`` raises for unmapped layers in between hardware layers.
"""

import pytest

import logging
import numpy as np
import re
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.analysis.fpgadataflow.unsupported_layers import unmapped_layer_report, unsupported_layers
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.builder.build_dataflow_steps import step_create_dataflow_partition
from finn.util.basic import make_build_dir, robust_rmtree
from finn.util.exception import FINNUnsupportedOpError, FINNUserError

HW_DOMAIN = "finn.custom_op.fpgadataflow"
SW_DOMAIN = "somethingelse"


def make_model(nodes: list, inputs: list[str], outputs: list[str]) -> ModelWrapper:
    """Build a model from (name, input tensors, output tensors, is hardware layer) tuples."""
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "HwOp" if hw else "SwOp",
                node_inputs,
                node_outputs,
                domain=(HW_DOMAIN + ".hls") if hw else SW_DOMAIN,
                name=name,
            )
            for name, node_inputs, node_outputs, hw in nodes
        ],
        name="graph",
        inputs=[helper.make_tensor_value_info(t, TensorProto.FLOAT, None) for t in inputs],
        outputs=[helper.make_tensor_value_info(t, TensorProto.FLOAT, None) for t in outputs],
    )
    return ModelWrapper(qonnx_make_model(graph))


def positions(model: ModelWrapper) -> dict[str, str]:
    """Return {node name: position} of the report."""
    return {entry["name"]: entry["position"] for entry in unmapped_layer_report(model)}


def make_linear_model() -> ModelWrapper:
    """Return a linear graph with unmapped nodes at the start, in between and at the end."""
    return make_model(
        [
            ("a", ["in"], ["t0"], False),
            ("b", ["t0"], ["t1"], True),
            ("c", ["t1"], ["t2"], False),
            ("d", ["t2"], ["t3"], False),
            ("e", ["t3"], ["t4"], True),
            ("f", ["t4"], ["out"], False),
        ],
        ["in"],
        ["out"],
    )


@pytest.mark.analysis
def test_unmapped_layer_report_linear() -> None:
    """Positions in a linear graph."""
    model = make_linear_model()
    assert model.analysis(unmapped_layer_report) == [
        {"name": "a", "op_type": "SwOp", "domain": SW_DOMAIN, "position": "start"},
        {"name": "c", "op_type": "SwOp", "domain": SW_DOMAIN, "position": "interior"},
        {"name": "d", "op_type": "SwOp", "domain": SW_DOMAIN, "position": "interior"},
        {"name": "f", "op_type": "SwOp", "domain": SW_DOMAIN, "position": "end"},
    ]
    assert unsupported_layers(model)[0] is False


@pytest.mark.analysis
def test_unmapped_layer_report_branching() -> None:
    """Positions consider all paths through a node."""
    model = make_model(
        [
            ("pre", ["in0"], ["t0"], False),
            ("hw0", ["t0"], ["t1", "t2"], True),
            # between hw0 and hw1 on one path, although it also feeds a graph output directly
            ("fork", ["t1"], ["t3", "out0"], False),
            # joins the accelerator from a graph input: nothing before it is hardware
            ("side", ["in1"], ["t4"], False),
            ("hw1", ["t3", "t4"], ["t5"], True),
            ("post", ["t5"], ["out1"], False),
            # leaves the accelerator and never returns
            ("tail", ["t2"], ["t6"], False),
            ("tail2", ["t6"], ["out2"], False),
            # bypasses the accelerator completely
            ("bypass", ["in2"], ["out3"], False),
        ],
        ["in0", "in1", "in2"],
        ["out0", "out1", "out2", "out3"],
    )
    assert positions(model) == {
        "pre": "start",
        "fork": "interior",
        "side": "start",
        "post": "end",
        "tail": "end",
        "tail2": "end",
        "bypass": "start",
    }
    assert unsupported_layers(model)[0] is False


@pytest.mark.analysis
def test_unmapped_layer_report_parallel_branches() -> None:
    """An unmapped branch in parallel to the hardware layers is interior."""
    # the unmapped branch runs in parallel to a mapped branch between the same hardware layers
    model = make_model(
        [
            ("hw0", ["in"], ["t0", "t1"], True),
            ("hw1", ["t0"], ["t2"], True),
            ("sw", ["t1"], ["t3"], False),
            ("hw2", ["t2", "t3"], ["out"], True),
        ],
        ["in"],
        ["out"],
    )
    assert positions(model) == {"sw": "interior"}
    assert unsupported_layers(model)[0] is False


@pytest.mark.analysis
def test_unmapped_layer_report_without_interior_nodes() -> None:
    """Models without interior nodes, without unmapped nodes and without hardware layers."""
    model = make_model(
        [
            ("a", ["in"], ["t0"], False),
            ("b", ["t0"], ["t1"], True),
            ("c", ["t1"], ["t2"], True),
            ("d", ["t2"], ["out"], False),
        ],
        ["in"],
        ["out"],
    )
    assert positions(model) == {"a": "start", "d": "end"}
    assert unsupported_layers(model)[0] is True
    # fully mapped and completely unmapped models
    model = make_model([("b", ["in"], ["t"], True), ("c", ["t"], ["out"], True)], ["in"], ["out"])
    assert model.analysis(unmapped_layer_report) == []
    model = make_model([("b", ["in"], ["t"], False), ("c", ["t"], ["out"], False)], ["in"], ["out"])
    assert positions(model) == {"b": "start", "c": "start"}


@pytest.mark.analysis
def test_unmapped_layer_report_matches_unsupported_layers(finn_test_seed: int) -> None:
    """The interior nodes are exactly what makes unsupported_layers reject a model."""
    # "interior" nodes are exactly what makes unsupported_layers reject a model
    rng = np.random.default_rng(finn_test_seed)
    verdicts = set()
    for _ in range(200):
        num_nodes = int(rng.integers(2, 12))
        hw_probability = rng.choice([0.3, 0.6, 0.9])
        nodes = []
        for idx in range(num_nodes):
            if idx == 0:
                node_inputs = ["in"]
            else:
                fan_in = int(rng.integers(1, min(idx, 2) + 1))
                producers = rng.choice(idx, size=fan_in, replace=False)
                node_inputs = [f"t{int(producer)}" for producer in producers]
            nodes.append((f"n{idx}", node_inputs, [f"t{idx}"], bool(rng.random() < hw_probability)))
        consumed = {tensor for _, node_inputs, _, _ in nodes for tensor in node_inputs}
        outputs = [f"t{idx}" for idx in range(num_nodes) if f"t{idx}" not in consumed]
        model = make_model(nodes, ["in"], outputs)

        report = positions(model)
        supported, first_node = unsupported_layers(model)
        assert supported == ("interior" not in report.values())
        if not supported:
            assert report[first_node.name] == "interior"
        verdicts.add(supported)
    assert verdicts == {True, False}


@pytest.mark.analysis
def test_step_create_dataflow_partition_unmapped_error() -> None:
    """The partitioning step raises a typed error that keeps the message format."""
    model = make_linear_model()
    with pytest.raises(FINNUnsupportedOpError) as excinfo:
        step_create_dataflow_partition(model, None)
    err = excinfo.value
    # stays a FINNUserError and keeps the message format
    assert isinstance(err, FINNUserError)
    assert re.search(r"starting with node (\S+)\.", str(err)).group(1) == err.node_name
    listed = re.search(r"Complete list of unmapped nodes: (.*)$", str(err)).group(1)
    assert listed == "a, c, d, f"
    # the reported node is the last unmapped one before the hardware layers continue
    assert (err.node_name, err.op_type) == ("d", "SwOp")
    assert err.unmapped_nodes == model.analysis(unmapped_layer_report)
    interior = [node["name"] for node in err.unmapped_nodes if node["position"] == "interior"]
    assert interior == ["c", "d"]


@pytest.mark.analysis
def test_step_create_dataflow_partition_boundary_nodes_only_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unmapped nodes at the start and end of the graph only cause a warning."""
    # Mul -> MVAU -> Mul: the unmapped nodes at the start and end are only reported
    scale = np.asarray([2.0], dtype=np.float32)
    nodes = [
        helper.make_node("Mul", ["inp", "scale"], ["t0"], name="Mul_0"),
        helper.make_node(
            "MVAU",
            ["t0", "weights"],
            ["t1"],
            domain=HW_DOMAIN,
            backend="fpgadataflow",
            name="MVAU_0",
            MW=4,
            MH=2,
            SIMD=1,
            PE=1,
            inputDataType="UINT4",
            weightDataType="INT4",
            outputDataType="INT16",
            ActVal=0,
            binaryXnorMode=0,
            noActivation=1,
            numInputVectors=[1],
        ),
        helper.make_node("Mul", ["t1", "scale"], ["outp"], name="Mul_1"),
    ]
    graph = helper.make_graph(
        nodes,
        "graph",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 2])],
        value_info=[
            helper.make_tensor_value_info("t0", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("t1", TensorProto.FLOAT, [1, 2]),
        ],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("scale", scale)
    model.set_initializer("weights", np.ones((4, 2), dtype=np.float32))
    model.set_tensor_datatype("weights", DataType["INT4"])
    model.set_tensor_datatype("t0", DataType["UINT4"])
    model.set_tensor_datatype("t1", DataType["INT16"])
    assert positions(model) == {"Mul_0": "start", "Mul_1": "end"}

    build_dir = make_build_dir("test_unmapped_layer_report_")
    try:
        cfg = DataflowBuildConfig(
            output_dir=build_dir,
            synth_clk_period_ns=10.0,
            fpga_part="xc7z020clg400-1",
            generate_outputs=[],
        )
        with caplog.at_level(logging.WARNING):
            dataflow_model = step_create_dataflow_partition(model, cfg)
    finally:
        robust_rmtree(build_dir)
    assert [node.op_type for node in dataflow_model.graph.node] == ["MVAU"]
    warnings = [record.getMessage() for record in caplog.records]
    assert any("start/end of the graph" in msg and "Mul_0, Mul_1" in msg for msg in warnings)

"""Tests for the node-wise verification helpers in finn.util.verification."""

import pytest

import numpy as np
from onnx import TensorProto, helper
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.exception import FINNUserError
from finn.util.verification import (
    compare_execution_contexts,
    despecialize_model,
    first_deviation,
    nodewise_verification,
    write_nodewise_report,
)


def make_two_layer_mvau_model(
    rng: np.random.Generator, mw: int = 8, mh: int = 4, pe: int = 2, simd: int = 4
) -> tuple[ModelWrapper, DataType]:
    """Two MVAU layers (with thresholding activation) as a hardware abstraction layer graph."""
    idt = DataType["UINT4"]
    wdt = DataType["INT4"]
    odt = DataType["UINT4"]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])
    mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, mh])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])
    nodes = []
    inits = {}
    for i, (in_name, out_name, w_shape) in enumerate(
        [("inp", "mid", (mw, mh)), ("mid", "outp", (mh, mh))]
    ):
        nodes.append(
            helper.make_node(
                "MVAU",
                [in_name, f"w{i}", f"t{i}"],
                [out_name],
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                MW=w_shape[0],
                MH=w_shape[1],
                SIMD=simd if i == 0 else pe,
                PE=pe,
                inputDataType=idt.name if i == 0 else odt.name,
                weightDataType=wdt.name,
                outputDataType=odt.name,
                ActVal=odt.min(),
                binaryXnorMode=0,
                noActivation=0,
            )
        )
        inits[f"w{i}"] = gen_finn_dt_tensor(wdt, w_shape)
        thresholds = np.sort(
            rng.integers(-40, 40, size=(w_shape[1], odt.get_num_possible_values() - 1)), axis=1
        )
        inits[f"t{i}"] = thresholds.astype(np.float32)
    graph = helper.make_graph(nodes, "mvau_graph", inputs=[inp], outputs=[outp], value_info=[mid])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="test"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("mid", odt)
    model.set_tensor_datatype("outp", odt)
    for w_name, t_name in [("w0", "t0"), ("w1", "t1")]:
        model.set_tensor_datatype(w_name, wdt)
        model.set_tensor_datatype(t_name, DataType["INT8"])
        model.set_initializer(w_name, inits[w_name])
        model.set_initializer(t_name, inits[t_name])
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model, idt


@pytest.mark.parametrize("impl_style", ["hls", "rtl"])
def test_despecialize_model_restores_python_execution(impl_style: str) -> None:
    """Despecialized copies execute in Python and match the abstraction layer model."""
    hw_model, idt = make_two_layer_mvau_model(np.random.default_rng(42))
    inp = gen_finn_dt_tensor(idt, (1, 8))
    ref_ctx = execute_onnx(hw_model, {"global_in": inp}, return_full_exec_context=True)

    spec_model = hw_model.transform(SpecializeLayers("xc7z020clg400-1"))
    for node in spec_model.graph.node:
        node.op_type = "MVAU_" + impl_style
        node.domain = "finn.custom_op.fpgadataflow." + impl_style
    spec_model.set_metadata_prop("exec_mode", "rtlsim")
    despec_model = despecialize_model(spec_model)

    # the specialized model is untouched, the copy is back on the abstraction layer
    assert all(n.op_type == "MVAU_" + impl_style for n in spec_model.graph.node)
    assert all(n.op_type == "MVAU" for n in despec_model.graph.node)
    assert all(n.domain == "finn.custom_op.fpgadataflow" for n in despec_model.graph.node)
    assert [n.name for n in despec_model.graph.node] == [n.name for n in spec_model.graph.node]
    assert despec_model.get_metadata_prop("exec_mode") in (None, "")

    despec_ctx = execute_onnx(despec_model, {"global_in": inp}, return_full_exec_context=True)
    for tensor in ["MVAU_0_out0", "global_out"]:
        assert (despec_ctx[tensor] == ref_ctx[tensor]).all()


def test_despecialize_model_resets_exec_mode() -> None:
    """Ops that dispatch on exec_mode must execute in Python after de-specialization.

    ReplicateStream keeps the simulation mode otherwise and fails on the missing cppsim
    code generation directory (seen on the transformer benchmark model).
    """
    node = helper.make_node(
        "ReplicateStream_hls",
        ["inp"],
        ["out0", "out1"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        num=2,
        dtype="UINT4",
        num_elems=8,
        PE=1,
        num_inputs=[1],
        exec_mode="cppsim",
    )
    shape = [1, 8]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outs = [helper.make_tensor_value_info(o, TensorProto.FLOAT, shape) for o in ("out0", "out1")]
    graph = helper.make_graph([node], "replicate", inputs=[inp], outputs=outs)
    model = ModelWrapper(qonnx_make_model(graph, producer_name="test"))
    for tensor in ("inp", "out0", "out1"):
        model.set_tensor_datatype(tensor, DataType["UINT4"])
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())

    despec = despecialize_model(model)
    assert despec.graph.node[0].op_type == "ReplicateStream"
    assert getCustomOp(despec.graph.node[0]).get_nodeattr("exec_mode") == "python"
    x = gen_finn_dt_tensor(DataType["UINT4"], (1, 8))
    ctx = execute_onnx(despec, {despec.graph.input[0].name: x}, return_full_exec_context=True)
    for out in despec.graph.output:
        assert (ctx[out.name] == x).all()

    # ops without a Python mode (MVAU) get the attribute cleared instead
    hw_model, _ = make_two_layer_mvau_model(np.random.default_rng(0))
    spec = hw_model.transform(SpecializeLayers("xc7z020clg400-1"))
    for n in spec.graph.node:
        getCustomOp(n).set_nodeattr("exec_mode", "cppsim")
    despec = despecialize_model(spec)
    assert all(getCustomOp(n).get_nodeattr("exec_mode") == "" for n in despec.graph.node)


def test_despecialize_model_rejects_unknown_ops() -> None:
    """Specialized ops without an abstraction layer counterpart are rejected."""
    hw_model, _ = make_two_layer_mvau_model(np.random.default_rng(0))
    hw_model.graph.node[0].op_type = "NoSuchOp_hls"
    hw_model.graph.node[0].domain = "finn.custom_op.fpgadataflow.hls"
    with pytest.raises(FINNUserError):
        despecialize_model(hw_model)


def test_compare_execution_contexts_and_report(tmp_path: Path) -> None:
    """Per-tensor comparison handles prefixes, folded shapes, deviations and missing tensors."""
    hw_model, _ = make_two_layer_mvau_model(np.random.default_rng(0))
    ref_ctx = {
        "MVAU_0_out0": np.arange(4, dtype=np.float32).reshape(1, 4),
        "global_out": np.ones((1, 4), dtype=np.float32),
    }
    sim_ctx = {
        # same values, but delivered in folded shape and with the partition prefix
        "SDP_0_MVAU_0_out0": np.arange(4, dtype=np.float32).reshape(1, 2, 2),
        # one element off, stored under the parent's output name
        "parent_out": np.array([[1, 1, 1, 3]], dtype=np.float32),
    }
    rows = compare_execution_contexts(
        hw_model, sim_ctx, ref_ctx, sim_prefix="SDP_0_", sim_name_map={"global_out": "parent_out"}
    )
    assert [r["tensor"] for r in rows] == ["MVAU_0_out0", "global_out"]
    assert rows[0]["status"] == "ok"
    assert rows[0]["num_mismatch"] == 0
    assert rows[1]["status"] == "deviates"
    assert rows[1]["num_mismatch"] == 1
    assert rows[1]["max_abs_err"] == 2.0
    assert rows[1]["mean_abs_err"] == 0.5
    assert first_deviation(rows)["node"] == "MVAU_1"

    # missing tensors and shape mismatches are reported but never raise
    rows = compare_execution_contexts(hw_model, {}, ref_ctx)
    assert [r["status"] for r in rows] == ["missing", "missing"]
    assert first_deviation(rows) is None
    rows = compare_execution_contexts(
        hw_model, {"global_out": np.zeros((1, 5))}, ref_ctx, sim_prefix=""
    )
    assert rows[1]["status"] == "shape"

    report = tmp_path / "nodewise.txt"
    write_nodewise_report(report, rows, header="header line")
    text = report.read_text()
    assert text.startswith("header line")
    assert "First deviating node:         MVAU_1" in text
    assert "Deviating tensors:            1/2" in text


def test_nodewise_verification_end_to_end(tmp_path: Path) -> None:
    """The report locates the first deviating node and tolerates missing Python references."""
    hw_model, idt = make_two_layer_mvau_model(np.random.default_rng(7))
    inp = gen_finn_dt_tensor(idt, (1, 8))
    spec_model = hw_model.transform(SpecializeLayers("xc7z020clg400-1"))
    ref_ctx = execute_onnx(hw_model, {"global_in": inp}, return_full_exec_context=True)
    # emulate a "simulation" context as the parent model would deliver it, with a corrupted
    # output of the second layer
    sim_ctx = {"SDP_" + k: np.array(v) for k, v in ref_ctx.items()}
    sim_ctx["parent_out"] = sim_ctx.pop("SDP_global_out")
    sim_ctx["parent_out"][0, 0] += 1
    report = tmp_path / "nodewise.txt"
    name_map = {"global_out": "parent_out"}
    rows = nodewise_verification(
        spec_model,
        {"global_in": inp},
        sim_ctx,
        "SDP_",
        report,
        atol=1e-3,
        rtol=1e-5,
        sim_name_map=name_map,
    )
    assert rows is not None
    assert first_deviation(rows)["tensor"] == "global_out"
    assert rows[0]["status"] == "ok"
    assert report.is_file()

    # a graph without Python reference does not break the caller
    spec_model.graph.node[0].op_type = "NoSuchOp_hls"
    assert (
        nodewise_verification(
            spec_model, {"global_in": inp}, sim_ctx, "SDP_", report, atol=1e-3, rtol=1e-5
        )
        is None
    )

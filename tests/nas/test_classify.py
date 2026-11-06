"""Unit tests for the failure classification, without running FINN."""

import pytest

import numpy as np
import onnx.helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.nas.classify import (
    boundary_unmapped_nodes,
    classify_failure,
    classify_interior_float,
    classify_unmapped,
    node_from_traceback,
    unhandled_folding_ops,
)
from finn.nas.result import FailureCategory
from finn.util.exception import FINNConfigurationError, FINNInternalError, FINNUserError


def _model_with(op_types: list[tuple[str, str, str]]) -> ModelWrapper:
    """Build a model with nodes ``(name, op_type, domain)`` chained in a row."""
    nodes = []
    prev = "inp"
    for i, (name, op_type, domain) in enumerate(op_types):
        out = f"t{i}"
        nodes.append(oh.make_node(op_type, [prev], [out], name=name, domain=domain))
        prev = out
    graph = oh.make_graph(
        nodes,
        "g",
        [oh.make_tensor_value_info("inp", 1, [1, 4])],
        [oh.make_tensor_value_info(prev, 1, [1, 4])],
    )
    return ModelWrapper(oh.make_model(graph))


HW = "finn.custom_op.fpgadataflow"
PARTITION = "step_create_dataflow_partition"


def _chain(specs: list[tuple], in_dtype: str = "FLOAT32") -> ModelWrapper:
    """Build a chain of nodes ``(name, op_type, domain, out_dtype[, const_dtype])``.

    ``const_dtype`` adds a constant second input (weights, scale, ...) of that datatype.
    """
    nodes = []
    prev = "inp"
    consts = {}
    dtypes = {"inp": in_dtype}
    for i, (name, op_type, domain, out_dtype, *const) in enumerate(specs):
        out = f"t{i}"
        inputs = [prev]
        if const:
            consts[f"{name}_param"] = const[0]
            inputs.append(f"{name}_param")
        nodes.append(oh.make_node(op_type, inputs, [out], name=name, domain=domain))
        dtypes[out] = out_dtype
        prev = out
    graph = oh.make_graph(
        nodes,
        "g",
        [oh.make_tensor_value_info("inp", 1, [1, 4])],
        [oh.make_tensor_value_info(prev, 1, [1, 4])],
    )
    model = ModelWrapper(oh.make_model(graph))
    for name, dtype in consts.items():
        model.set_initializer(name, np.ones((1, 4), dtype=np.float32))
        dtypes[name] = dtype
    for name, dtype in dtypes.items():
        model.set_tensor_datatype(name, DataType[dtype])
    return model


def _partition_error(model: ModelWrapper) -> FINNUserError:
    """The error FINN's partitioning step raises for the unmapped nodes of ``model``."""
    unmapped = [n.name for n in model.graph.node if not n.domain.startswith(HW)]
    return FINNUserError(
        "Unsupported/unmapped layer(s) found in between FINN operators, starting with node "
        f"{unmapped[-1]}. Complete list of unmapped nodes: " + ", ".join(unmapped)
    )


@pytest.mark.util
class TestClassifyUnmapped:
    """Root-cause selection among the unmapped nodes FINN lists."""

    def test_boundary_node_is_not_blamed_and_float_producer_is(self) -> None:
        # A float ReLU (converted to a float ElementwiseMax) starves the next conv of
        # integers. The input Transpose is unmapped too, but harmless.
        model = _chain(
            [
                ("Transpose_0", "Transpose", "", "FLOAT32"),
                ("Thresholding_0", "Thresholding", HW, "INT2"),
                ("ElementwiseMax_0", "ElementwiseMax", HW, "FLOAT32"),
                ("Shuffle_0", "Shuffle", HW, "FLOAT32"),
                ("Im2Col_0", "Im2Col", "qonnx.custom_op.general", "FLOAT32"),
                ("MatMul_0", "MatMul", "", "FLOAT32", "INT2"),
                ("Thresholding_1", "Thresholding", HW, "UINT2"),
            ]
        )
        f = classify_failure(PARTITION, _partition_error(model), model)
        assert f.category == FailureCategory.DATATYPE_UNSUPPORTED
        assert (f.node, f.op_type) == ("ElementwiseMax_0", "ElementwiseMax")
        assert "Im2Col_0" in f.message

    def test_unabsorbed_scale_is_streamline_incomplete(self) -> None:
        model = _chain(
            [
                ("MVAU_0", "MVAU", HW, "INT6"),
                ("ElementwiseMul_0", "ElementwiseMul", HW, "FLOAT32", "FLOAT32"),
                ("MatMul_0", "MatMul", "", "FLOAT32", "INT2"),
                ("Thresholding_0", "Thresholding", HW, "UINT2"),
            ]
        )
        f = classify_failure(PARTITION, _partition_error(model), model)
        assert f.category == FailureCategory.STREAMLINE_INCOMPLETE
        assert (f.node, f.op_type) == ("ElementwiseMul_0", "ElementwiseMul")

    def test_float_weights(self) -> None:
        model = _chain(
            [
                ("MVAU_0", "MVAU", HW, "UINT2"),
                ("MatMul_0", "MatMul", "", "FLOAT32", "FLOAT32"),
                ("Thresholding_0", "Thresholding", HW, "UINT2"),
            ]
        )
        f = classify_failure(PARTITION, _partition_error(model), model)
        assert f.category == FailureCategory.DATATYPE_UNSUPPORTED
        assert (f.node, f.op_type) == ("MatMul_0", "MatMul")
        assert "weights" in f.message

    def test_unconverted_quantizer(self) -> None:
        model = _chain(
            [
                ("MVAU_0", "MVAU", HW, "INT6"),
                ("Quant_0", "Quant", "qonnx.custom_op.general", "FLOAT32"),
                ("MatMul_0", "MatMul", "", "FLOAT32", "INT2"),
                ("MVAU_1", "MVAU", HW, "INT6"),
            ]
        )
        f = classify_failure(PARTITION, _partition_error(model), model)
        assert f.category == FailureCategory.DATATYPE_UNSUPPORTED
        assert (f.node, f.op_type) == ("Quant_0", "Quant")

    def test_convertible_with_integer_operands(self) -> None:
        model = _chain(
            [
                ("MVAU_0", "MVAU", HW, "UINT2"),
                ("Resize_0", "Resize", "", "UINT2"),
                ("MVAU_1", "MVAU", HW, "INT6"),
            ]
        )
        f = classify_failure(PARTITION, _partition_error(model), model)
        assert f.category == FailureCategory.UNMAPPED_LAYER
        assert (f.node, f.op_type) == ("Resize_0", "Resize")

    def test_host_side_layer_without_input_quantizer(self) -> None:
        model = _chain(
            [
                ("Transpose_0", "Transpose", "", "FLOAT32"),
                ("MatMul_0", "MatMul", "", "FLOAT32", "INT2"),
                ("Thresholding_0", "Thresholding", HW, "UINT2"),
            ]
        )
        f = classify_unmapped(PARTITION, ["MatMul_0"], "host", model, interior=False)
        assert f.category == FailureCategory.DATATYPE_UNSUPPORTED
        assert f.node == "MatMul_0"
        assert "graph input" in f.message

    def test_interior_float(self) -> None:
        model = _chain(
            [
                ("Thresholding_0", "Thresholding", HW, "INT2"),
                ("ElementwiseMul_0", "ElementwiseMul", HW, "FLOAT32", "FLOAT32"),
                ("ElementwiseMax_0", "ElementwiseMax", HW, "FLOAT32"),
                ("Thresholding_1", "Thresholding", HW, "UINT2"),
            ]
        )
        floats = [["ElementwiseMul_0", "ElementwiseMul"], ["ElementwiseMax_0", "ElementwiseMax"]]
        f = classify_interior_float(PARTITION, floats, model)
        assert f.category == FailureCategory.STREAMLINE_INCOMPLETE
        assert f.node == "ElementwiseMul_0"
        f = classify_interior_float(PARTITION, floats[1:], model)
        assert f.category == FailureCategory.DATATYPE_UNSUPPORTED
        assert f.node == "ElementwiseMax_0"


@pytest.mark.util
class TestClassify:
    """Failure classification cases."""

    def test_partition_error_streamline_incomplete(self) -> None:
        model = _model_with(
            [("MVAU_0", "MVAU", "finn.custom_op.fpgadataflow"), ("Mul_3", "Mul", "")]
        )
        exc = FINNUserError(
            "Unsupported/unmapped layer(s) found in between FINN operators, starting with node "
            "Mul_3. Complete list of unmapped nodes: Mul_3"
        )
        f = classify_failure("step_create_dataflow_partition", exc, model)
        assert f.category == FailureCategory.STREAMLINE_INCOMPLETE
        assert f.node == "Mul_3"
        assert f.op_type == "Mul"

    def test_partition_error_unsupported_op_root_cause(self) -> None:
        # FINN names the MatMul that stayed unconverted *because of* the Sigmoid; the
        # classifier must blame the Sigmoid.
        model = _model_with([("Sigmoid_0", "Sigmoid", ""), ("MatMul_0", "MatMul", "")])
        exc = FINNUserError(
            "Unsupported/unmapped layer(s) found in between FINN operators, starting with node "
            "MatMul_0. Complete list of unmapped nodes: Sigmoid_0, MatMul_0"
        )
        f = classify_failure("step_create_dataflow_partition", exc, model)
        assert f.category == FailureCategory.UNSUPPORTED_OP
        assert f.node == "Sigmoid_0"
        assert f.op_type == "Sigmoid"

    def test_partition_error_only_convertible_left(self) -> None:
        model = _model_with([("MatMul_0", "MatMul", "")])
        model.set_tensor_datatype("inp", DataType["INT2"])
        exc = FINNUserError(
            "... starting with node MatMul_0. Complete list of unmapped nodes: MatMul_0"
        )
        f = classify_failure("step_create_dataflow_partition", exc, model)
        assert f.category == FailureCategory.UNMAPPED_LAYER
        assert f.op_type == "MatMul"

    def test_partition_error_unknown_node(self) -> None:
        exc = FINNUserError("... starting with node Foo_9. Complete list of unmapped nodes: Foo_9")
        f = classify_failure("step_create_dataflow_partition", exc, None)
        assert f.category == FailureCategory.UNMAPPED_LAYER
        assert f.node == "Foo_9"

    def test_no_hw_variant(self) -> None:
        exc = Exception(
            "Node Upsample_0 with optype UpsampleNearestNeighbour "
            "has no hw implementation variant)"
        )
        f = classify_failure("step_specialize_layers", exc, None)
        assert f.category == FailureCategory.UNSUPPORTED_OP
        assert f.node == "Upsample_0"
        assert f.op_type == "UpsampleNearestNeighbour"

    def test_bad_impl_style_is_config_error(self) -> None:
        exc = Exception("Invalid value for attribute preferred_impl_style! Is currently set to: x")
        f = classify_failure("step_specialize_layers", exc, None)
        assert f.category == FailureCategory.CONFIG_ERROR

    def test_folding_exception(self) -> None:
        exc = Exception("Expected SWU on DW op input, found Thresholding_rtl")
        f = classify_failure("folding", exc, None)
        assert f.category == FailureCategory.FOLDING_INFEASIBLE

    def test_frontend_and_config_errors(self) -> None:
        assert (
            classify_failure("step_streamline", KeyError("x"), None).category
            == FailureCategory.FRONTEND_FAILED
        )
        assert (
            classify_failure("step_tidy_up", FINNConfigurationError("no part"), None).category
            == FailureCategory.CONFIG_ERROR
        )
        assert (
            classify_failure("estimate", FINNInternalError("boom"), None).category
            == FailureCategory.INTERNAL_ERROR
        )

    def test_codegen_assert(self) -> None:
        assert (
            classify_failure("step_hw_codegen", AssertionError("MH divisable by PE"), None).category
            == FailureCategory.CODEGEN_ASSERT
        )
        # FINN+ raises FINNUserError instead of asserting in some custom ops.
        exc = FINNUserError("MVAU_hls_0: MW (16) must be divisable by SIMD (currently 5).")
        assert classify_failure("step_hw_codegen", exc, None).category == (
            FailureCategory.CODEGEN_ASSERT
        )
        assert classify_failure("step_hw_codegen", KeyError("x"), None).category == (
            FailureCategory.INTERNAL_ERROR
        )

    def test_folding_constraint_after_folding(self) -> None:
        # Illegal PE/SIMD values only surface when a later stage asks for the folded shapes.
        exc = AssertionError("Requirement MH divisable by PE is violated.")
        f = classify_failure("step_minimize_bit_width", exc, None)
        assert f.category == FailureCategory.FOLDING_INFEASIBLE
        exc = AssertionError("SIMD must divide IFMChannels")
        assert classify_failure("estimate", exc, None).category == (
            FailureCategory.FOLDING_INFEASIBLE
        )
        exc = AssertionError("weights out of range")
        assert classify_failure("step_minimize_bit_width", exc, None).category == (
            FailureCategory.DATATYPE_UNSUPPORTED
        )

    def test_node_named_in_message(self) -> None:
        # The traceback only knows the MatMul being converted; the message names the
        # activation that is at fault.
        model = _model_with(
            [("MatMul_0", "MatMul", ""), ("MultiThreshold_1", "MultiThreshold", "")]
        )
        exc = FINNUserError("MultiThreshold_1: Signed output requires actval < 0")
        f = classify_failure("step_convert_to_hw", exc, model)
        assert f.category == FailureCategory.DATATYPE_UNSUPPORTED
        assert (f.node, f.op_type) == ("MultiThreshold_1", "MultiThreshold")

    def test_node_from_traceback(self) -> None:
        model = _model_with([("N0", "Relu", "")])
        node = model.graph.node[0]

        def inner(node: object) -> None:  # noqa: ARG001
            raise AssertionError("bad")

        try:
            inner(node)
        except AssertionError as exc:
            assert node_from_traceback(exc) == ("N0", "Relu")
            f = classify_failure("step_hw_codegen", exc, None)
            assert f.node == "N0"

    def test_warning_parsers(self) -> None:
        model = _model_with([("Mul_0", "Mul", ""), ("TopK_1", "TopK", "")])
        msgs = [
            "SetFolding doesn't know how to handle op_type Foo_hls",
            "The following nodes at the start/end of the graph will not be mapped to the "
            "accelerator, so they will need to be implemented manually (e.g., in software): "
            "Mul_0, TopK_1",
        ]
        assert unhandled_folding_ops(msgs) == ["Foo_hls"]
        assert boundary_unmapped_nodes(msgs, model) == [["Mul_0", "Mul"], ["TopK_1", "TopK"]]

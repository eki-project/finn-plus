"""Deliberately broken candidates: does the oracle report the root cause a search can learn from?

Every network in ``broken_models.py`` has exactly one defect. The expectations below were
established empirically against the FINN+ flow; a changed expectation means FINN+ (or the
oracle's classification) changed what it accepts and should be looked at, not just updated.
No Vivado needed.
"""

import pytest

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finn.nas import FailureCategory, FinnOracle, FoldingSpec, OracleConfig

pytest.importorskip("brevitas")
torch = pytest.importorskip("torch")
bm = pytest.importorskip("tests.nas.broken_models")

PARTITION = "step_create_dataflow_partition"
STRICT_BOUNDARY = {"boundary_unmapped_is_failure": True}


def _config(tmp_path: Path, **kwargs: Any) -> OracleConfig:
    base = {
        "fpga_part": "xc7z020clg400-1",
        "synth_clk_period_ns": 10.0,
        "folding": FoldingSpec(target_fps=1000),
        "isolation": "inprocess",
        "work_dir": str(tmp_path / "oracle"),
    }
    base.update(kwargs)
    return OracleConfig(**base)


def _evaluate(tmp_path: Path, model: str, config: dict[str, Any], model_kwargs: dict[str, Any]):
    torch.manual_seed(0)
    module = getattr(bm, model)(**model_kwargs)
    return FinnOracle(_config(tmp_path, **config)).evaluate(module, module.input_shape)


@dataclass
class Broken:
    """A broken candidate and what the oracle is expected to report for it."""

    model: str
    category: FailureCategory
    stage: str
    #: Op type of the node the oracle blames (``None``: no node can be known).
    op_type: str | None
    #: Text that the failure message must contain.
    message: str = ""
    config: dict[str, Any] = field(default_factory=dict)


BROKEN = [
    # --- operators FINN has no hardware conversion for -------------------------------------
    Broken("SinMLP", FailureCategory.UNSUPPORTED_OP, PARTITION, "Sin"),
    Broken("LeakyReluCNN", FailureCategory.UNSUPPORTED_OP, PARTITION, "LeakyRelu"),
    # Only Brevitas' TruncAvgPool2d (QuantAvgPool2d) is convertible, not a float AveragePool.
    Broken("FloatAvgPoolCNN", FailureCategory.UNSUPPORTED_OP, PARTITION, "AveragePool"),
    # --- convertible op that FINN nevertheless left unmapped -------------------------------
    # InferUpsample handles Resize, but step_convert_to_hw only runs it if an "Upsample" node
    # exists. No finer reason is available to the oracle, hence the coarse category.
    Broken("UpsampleCNN", FailureCategory.UNMAPPED_LAYER, PARTITION, "Resize"),
    # --- float scale/shift that streamlining cannot absorb ---------------------------------
    # FINN converts leftover Mul/Add to float elementwise layers; the conv behind stays
    # unmapped. The oracle blames the elementwise layer, i.e. the BatchNorm.
    Broken(
        "UnabsorbedBatchNormCNN",
        FailureCategory.STREAMLINE_INCOMPLETE,
        PARTITION,
        "ElementwiseMul",
        "Im2Col_0",
    ),
    # Different scales at a residual add leave a float Mul on one branch. All nodes are HW
    # layers here, so this is found by the interior float check, not by FINN.
    Broken(
        "MismatchedResidualCNN", FailureCategory.STREAMLINE_INCOMPLETE, PARTITION, "ElementwiseMul"
    ),
    # Equal but non-unit scales: the default streamlining cannot move the scale past the
    # fork (ResNets need the custom steps of finn.builder.custom_step_library.resnet).
    Broken("ScaledResidualCNN", FailureCategory.STREAMLINE_INCOMPLETE, PARTITION, "ElementwiseMul"),
    # --- non-integer data or parameters ----------------------------------------------------
    # The float ReLU becomes a float ElementwiseMax, which starves the next conv of integers.
    Broken(
        "FloatReluCNN", FailureCategory.DATATYPE_UNSUPPORTED, PARTITION, "ElementwiseMax", "Im2Col"
    ),
    Broken("FloatLinearMLP", FailureCategory.DATATYPE_UNSUPPORTED, PARTITION, "MatMul", "weights"),
    # The float conv is lowered to Im2Col + MatMul (1x1: MatMul only) before it is rejected.
    Broken("FloatConvCNN", FailureCategory.DATATYPE_UNSUPPORTED, PARTITION, "MatMul", "weights"),
    # 16 bit > max_multithreshold_bit_width (8): the Quant node is not converted.
    Broken("WideActMLP", FailureCategory.DATATYPE_UNSUPPORTED, PARTITION, "Quant", "16 bit"),
    # FINN's MVAU inference raises for the 1-bit unsigned activation; the traceback points at
    # the MatMul, the message at the activation.
    Broken(
        "OneBitReluCNN",
        FailureCategory.DATATYPE_UNSUPPORTED,
        "step_convert_to_hw",
        "MultiThreshold",
    ),
    # --- crashes before any hardware mapping -----------------------------------------------
    Broken("KernelTooLargeCNN", FailureCategory.EXPORT_FAILED, "export", None, "Kernel size"),
    # LowerConvsToMatMul cannot handle groups other than 1 or the channel count and crashes.
    Broken("GroupedConvCNN", FailureCategory.FRONTEND_FAILED, "step_streamline", "Conv"),
    # --- computation left to the host at the graph boundary --------------------------------
    # Failures only because boundary_unmapped_is_failure is on (the default), see
    # test_host_side_computation for the opt-out.
    Broken(
        "NoInputQuantMLP",
        FailureCategory.DATATYPE_UNSUPPORTED,
        PARTITION,
        "MatMul",
        "graph input",
        STRICT_BOUNDARY,
    ),
    Broken(
        "NoInputQuantCNN",
        FailureCategory.DATATYPE_UNSUPPORTED,
        PARTITION,
        "Im2Col",
        "graph input",
        STRICT_BOUNDARY,
    ),
    Broken("SinOutputMLP", FailureCategory.UNSUPPORTED_OP, PARTITION, "Sin", "", STRICT_BOUNDARY),
]

FEASIBLE = [
    ("GoodMLP", {}, {}),
    ("GoodCNN", {}, {}),
    ("IntegerResidualCNN", {}, {}),
    # Surprisingly legal: no activation between two convs, and overlapping max pooling.
    ("NoActivationCNN", {}, {}),
    ("OverlappingMaxPoolCNN", {}, {}),
    # A freshly initialized BatchNorm is a uniform scale that streamlining moves like a
    # scalar: the very topology that fails once BatchNorm has per-channel statistics passes.
    ("UnabsorbedBatchNormCNN", {}, {"trained_bn": False}),
    # The NCHW -> NHWC Transpose in front of a CNN stays on the host; it must not count as
    # host-side computation.
    ("GoodCNN", STRICT_BOUNDARY, {}),
    ("IntegerResidualCNN", STRICT_BOUNDARY, {}),
]


@pytest.mark.fpgadataflow
class TestBrokenCandidates:
    """One defect per network, each with the category a NAS loop should learn from."""

    @pytest.mark.parametrize(
        "case", BROKEN, ids=[c.model + ("-strict" if c.config else "") for c in BROKEN]
    )
    def test_broken(self, tmp_path: Path, case: Broken) -> None:
        res = _evaluate(tmp_path, case.model, case.config, {})
        report = (res.failure_category, res.failure_stage, res.failure_node, res.failure_message)
        assert not res.feasible, report
        assert res.failure_category == case.category, report
        assert res.failure_stage == case.stage, report
        assert res.failure_op_type == case.op_type, report
        if case.op_type is None:
            assert res.failure_node is None, report
        else:
            # FINN names nodes "<op type>_<index>".
            assert res.failure_node is not None, report
            assert res.failure_node.startswith(case.op_type + "_"), report
        assert case.message in (res.failure_message or ""), report
        assert not res.resources_total

    @pytest.mark.parametrize(
        ("model", "config", "model_kwargs"),
        FEASIBLE,
        ids=[m + ("-strict" if c else "") + ("-untrained" if k else "") for m, c, k in FEASIBLE],
    )
    def test_feasible(
        self, tmp_path: Path, model: str, config: dict[str, Any], model_kwargs: dict[str, Any]
    ) -> None:
        res = _evaluate(tmp_path, model, config, model_kwargs)
        assert res.feasible, (res.failure_category, res.failure_node, res.failure_message)
        assert res.resources_total["LUT"] > 0
        assert res.est_fps is not None and res.est_fps > 0

    @pytest.mark.parametrize(
        ("model", "host_ops"),
        [
            ("NoInputQuantMLP", {"MatMul"}),
            ("NoInputQuantCNN", {"Transpose", "Im2Col", "MatMul"}),
            ("SinOutputMLP", {"Sin"}),
        ],
    )
    def test_host_side_computation(self, tmp_path: Path, model: str, host_ops: set[str]) -> None:
        # With boundary_unmapped_is_failure=False the oracle accepts these, but the estimates
        # then cover only the part that FINN maps: for the missing input quantizer the whole
        # first layer runs on the host. The host nodes are reported so that a caller who
        # intends host-side pre/post-processing can still see them.
        res = _evaluate(tmp_path, model, {"boundary_unmapped_is_failure": False}, {})
        assert res.feasible
        assert {op for _, op in res.unmapped_boundary_nodes} == host_ops

    def test_missing_input_quantizer_hides_the_first_layer(self, tmp_path: Path) -> None:
        lenient = {"boundary_unmapped_is_failure": False}
        good = _evaluate(tmp_path, "GoodMLP", lenient, {})
        bad = _evaluate(tmp_path, "NoInputQuantMLP", lenient, {})
        assert bad.feasible
        assert len(bad.resources_per_node) < len(good.resources_per_node)
        assert bad.resources_total["LUT"] < good.resources_total["LUT"]

    def test_resource_budget_exceeded(self, tmp_path: Path) -> None:
        # Unfolded towards an unreachable throughput, 192 channels need more LUTs than 70 %
        # of a Pynq-Z1. The same network is fine structurally, so estimates are returned.
        config = {"fpga_part": None, "board": "Pynq-Z1", "folding": FoldingSpec(target_fps=10**6)}
        res = _evaluate(tmp_path, "WideCNN", config, {})
        assert not res.feasible
        assert res.failure_category == FailureCategory.RESOURCE_BUDGET_EXCEEDED
        assert res.failure_stage == "estimate"
        assert res.device_fit["LUT"] > 1.0
        assert "LUT" in res.failure_message
        assert res.resources_total["LUT"] > 0
        small = _evaluate(tmp_path, "GoodCNN", config, {})
        assert small.feasible, (small.failure_category, small.failure_message)
        assert 0 < small.device_fit["LUT"] < 1.0

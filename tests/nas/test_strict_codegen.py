"""Strict mode of the oracle: ``step_hw_codegen`` on top of the estimate flow.

Code generation only writes HLS/RTL source files: it starts no Vivado/Vitis HLS process and
does not read finn-hlslib (its path is merely written into the generated Tcl script), so
these tests need neither a tool licence nor the external dependencies.
"""

import pytest

import json
import subprocess
from pathlib import Path
from typing import Any

from finn.nas import FailureCategory, FinnOracle, FoldingSpec, OracleConfig

pytest.importorskip("brevitas")
torch = pytest.importorskip("torch")
models = pytest.importorskip("tests.nas.models")


def _evaluate(tmp_path: Path, model: str, **kwargs: Any):
    base = {
        "fpga_part": "xc7z020clg400-1",
        "synth_clk_period_ns": 10.0,
        "folding": FoldingSpec(target_fps=1000),
        "isolation": "inprocess",
        "work_dir": str(tmp_path / "oracle"),
    }
    base.update(kwargs)
    torch.manual_seed(0)
    cls = getattr(models, model)
    return FinnOracle(OracleConfig(**base)).evaluate(cls(), cls.input_shape)


def _fixed_folding(tmp_path: Path, config: dict[str, dict[str, int]]) -> FoldingSpec:
    path = tmp_path / "folding.json"
    path.write_text(json.dumps(config))
    return FoldingSpec(kind="fixed", target_fps=None, folding_config_file=str(path))


def _first_mvau(tmp_path: Path, model: str) -> str:
    """Name of the first matrix-vector unit, as a folding config has to address it."""
    plain = _evaluate(tmp_path, model)
    assert plain.feasible, (plain.failure_category, plain.failure_message)
    return next(name for name in plain.folding if name.startswith("MVAU"))


@pytest.mark.fpgadataflow
class TestStrictCodegen:
    """Feasible candidates and failures that only code generation reveals."""

    @pytest.mark.parametrize("model", ["TinyMLP", "TinyCNN"])
    def test_feasible(self, tmp_path: Path, model: str, monkeypatch: pytest.MonkeyPatch) -> None:
        plain = _evaluate(tmp_path, model)

        def forbidden(self: subprocess.Popen, *args: Any, **kwargs: Any) -> None:  # noqa: ARG001
            raise AssertionError(f"strict mode started a subprocess: {args}")

        # No external tool may be started: strict mode has to stay usable in a search loop.
        monkeypatch.setattr(subprocess.Popen, "__init__", forbidden)
        strict = _evaluate(tmp_path, model, strict_codegen=True, keep_artifacts="always")
        assert strict.feasible, (strict.failure_category, strict.failure_message)
        assert "step_hw_codegen" in strict.timings_s
        assert "step_hw_codegen" not in plain.timings_s
        assert strict.resources_total == plain.resources_total
        assert strict.cycles_per_node == plain.cycles_per_node
        # HLS and RTL sources were written for the generated layers.
        generated = {p.name for p in (tmp_path / "oracle").rglob("code_gen_ipgen_*/*")}
        assert any(name.startswith("top_MVAU_hls") and name.endswith(".cpp") for name in generated)
        assert any(
            name.startswith("Thresholding_rtl") and name.endswith(".v") for name in generated
        )

    def test_illegal_simd_is_only_found_by_codegen(self, tmp_path: Path) -> None:
        # 5 does not divide the 16 inputs of the first layer. The estimate flow never asks
        # the MVAU for a consistency check and happily returns numbers for it.
        mvau = _first_mvau(tmp_path, "TinyMLP")
        folding = _fixed_folding(tmp_path, {mvau: {"SIMD": 5}})
        plain = _evaluate(tmp_path, "TinyMLP", folding=folding)
        assert plain.feasible
        assert plain.folding[mvau]["SIMD"] == 5

        strict = _evaluate(tmp_path, "TinyMLP", folding=folding, strict_codegen=True)
        assert not strict.feasible
        assert strict.failure_category == FailureCategory.CODEGEN_ASSERT
        assert strict.failure_stage == "step_hw_codegen"
        assert strict.failure_node == mvau
        assert strict.failure_op_type == "MVAU_hls"
        assert "SIMD" in strict.failure_message

    def test_illegal_pe_is_found_without_codegen(self, tmp_path: Path) -> None:
        # 3 does not divide the 8 outputs. Unlike SIMD, PE enters the folded output shape
        # that bit-width minimization queries, so every mode rejects it.
        mvau = _first_mvau(tmp_path, "TinyMLP")
        res = _evaluate(tmp_path, "TinyMLP", folding=_fixed_folding(tmp_path, {mvau: {"PE": 3}}))
        assert not res.feasible
        assert res.failure_category == FailureCategory.FOLDING_INFEASIBLE
        assert res.failure_stage == "step_minimize_bit_width"
        assert res.failure_node == mvau

    def test_setfolding_result_rejected_by_codegen(self, tmp_path: Path) -> None:
        # Not only hand-written foldings: SetFolding itself leaves the 2048-input layer at
        # SIMD=1, which MVAU_hls refuses (SIMD >= MW / 1024).
        plain = _evaluate(tmp_path, "WideInputMLP")
        assert plain.feasible, (plain.failure_category, plain.failure_message)
        strict = _evaluate(tmp_path, "WideInputMLP", strict_codegen=True)
        assert not strict.feasible
        assert strict.failure_category == FailureCategory.CODEGEN_ASSERT
        assert strict.failure_stage == "step_hw_codegen"
        assert strict.failure_op_type == "MVAU_hls"
        assert "MW / 1024" in strict.failure_message
        # The estimates of the (illegal) folding are still reported.
        assert strict.resources_total["LUT"] > 0

"""Tests for the ``finn nas eval`` command line front end."""

import pytest

import json
from click.testing import CliRunner
from pathlib import Path
from rich.console import Console

from finn.interface.run_finn import nas_eval
from finn.nas import FoldingSpec, OracleConfig
from finn.nas.cli import EXIT_ERROR, EXIT_FEASIBLE, EXIT_INFEASIBLE, evaluate_model, exit_code_for
from finn.nas.export import export_candidate
from finn.nas.result import EvalResult, FailureCategory

pytest.importorskip("brevitas")
models = pytest.importorskip("tests.nas.models")


def _config(tmp_path: Path) -> OracleConfig:
    return OracleConfig(
        fpga_part="xc7z020clg400-1",
        synth_clk_period_ns=10.0,
        folding=FoldingSpec(target_fps=1000),
        isolation="inprocess",
        work_dir=str(tmp_path / "oracle"),
    )


@pytest.mark.util
class TestNasEvalArguments:
    """Argument handling that happens before FINN is prepared."""

    def test_help(self) -> None:
        result = CliRunner().invoke(nas_eval, ["--help"])
        assert result.exit_code == 0
        assert "--target-fps" in result.output

    def test_target_is_required(self, tmp_path: Path) -> None:
        model = tmp_path / "m.onnx"
        model.write_bytes(b"")
        result = CliRunner().invoke(nas_eval, [str(model), "--target-fps", "100"])
        assert result.exit_code == 2
        assert "--fpga-part or --board" in result.output

    def test_exactly_one_folding_source(self, tmp_path: Path) -> None:
        model = tmp_path / "m.onnx"
        model.write_bytes(b"")
        result = CliRunner().invoke(nas_eval, [str(model), "--board", "Pynq-Z1"])
        assert result.exit_code == 2
        assert "--target-fps and --folding-config" in result.output

    def test_exit_codes(self) -> None:
        assert exit_code_for(EvalResult(feasible=True)) == EXIT_FEASIBLE
        infeasible = EvalResult.failed("s", FailureCategory.UNSUPPORTED_OP)
        assert exit_code_for(infeasible) == EXIT_INFEASIBLE
        for category in (
            FailureCategory.TIMEOUT,
            FailureCategory.CONFIG_ERROR,
            FailureCategory.INTERNAL_ERROR,
        ):
            assert exit_code_for(EvalResult.failed("s", category)) == EXIT_ERROR


@pytest.mark.fpgadataflow
class TestNasEvalModel:
    """The evaluation and reporting behind ``finn nas eval``."""

    def test_feasible_model(self, tmp_path: Path) -> None:
        onnx_path = export_candidate(
            models.TinyMLP(), models.TinyMLP.input_shape, tmp_path / "tiny_mlp.onnx"
        )
        out = tmp_path / "out" / "result.json"
        console = Console(record=True, width=120)
        code = evaluate_model(onnx_path, _config(tmp_path), out, console)
        assert code == EXIT_FEASIBLE
        data = json.loads(out.read_text())
        assert data["feasible"] is True
        assert data["resources_total"]["LUT"] > 0
        text = console.export_text()
        assert "FINN-compatible" in text
        assert "LUT" in text

    def test_infeasible_model(self, tmp_path: Path) -> None:
        onnx_path = export_candidate(
            models.BadMLP(), models.BadMLP.input_shape, tmp_path / "bad_mlp.onnx"
        )
        console = Console(record=True, width=120)
        code = evaluate_model(onnx_path, _config(tmp_path), None, console)
        assert code == EXIT_INFEASIBLE
        text = console.export_text()
        assert "Not FINN-compatible" in text
        assert "node:" in text

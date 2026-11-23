"""Tests for the export of run candidates for full builds (``finn nas export``)."""

import pytest

import json
import onnx
import yaml
from click.testing import CliRunner
from pathlib import Path

from finn.interface.run_finn import nas_export
from finn.nas import FoldingSpec, OracleConfig
from finn.nas.evaluate import CandidateEvaluation, EvaluatorConfig
from finn.nas.experiments.export import (
    BENCH_CONFIG_FILE,
    DUT,
    EXPORT_SUMMARY_FILE,
    build_steps,
    export_run,
    select_records,
)
from finn.nas.experiments.sampling import CONFIG_FILE, RECORDS_FILE, SUMMARY_FILE, append_records
from finn.nas.pipeline import STREAMLINE_PLUS_STEP
from finn.nas.result import EvalResult, FailureCategory
from finn.nas.space import SpaceConfig, count_params, genotype_hash, infer_shapes
from finn.nas.space.seeds import seed_genotypes

pytest.importorskip("brevitas")


def _record(genotype: dict, feasible: bool = True) -> CandidateEvaluation:
    if feasible:
        result = EvalResult(
            feasible=True, est_fps=100.0, max_cycles=1000, resources_total={"LUT": 10.0}
        )
    else:
        result = EvalResult.failed("step_streamline", FailureCategory.FRONTEND_FAILED, "nope")
    return CandidateEvaluation(
        genotype=genotype,
        hash=genotype_hash(genotype),
        result=result,
        params=count_params(genotype),
        nodes=len(infer_shapes(genotype).nodes),
    )


def _write_run(
    tmp_path: Path, records: list[CandidateEvaluation], front: list[str] | None = None
) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    oracle = OracleConfig(
        board="Pynq-Z1", synth_clk_period_ns=10.0, folding=FoldingSpec(target_fps=1000)
    )
    evaluator = EvaluatorConfig(oracle=oracle, calibration_steps=2, calibration_batch_size=8)
    config = {"space": SpaceConfig().to_dict(), "evaluator": evaluator.to_dict()}
    (run / CONFIG_FILE).write_text(json.dumps(config))
    append_records(run / RECORDS_FILE, records)
    if front is not None:
        entries = [{"trial": i, "hash": h} for i, h in enumerate(front)]
        (run / SUMMARY_FILE).write_text(json.dumps({"pareto_front": entries}))
    return run


@pytest.mark.util
class TestSelection:
    """Which records are exported."""

    def test_feasible_dedupes_and_limits(self) -> None:
        seeds = seed_genotypes()
        records = [_record(seeds["mlp_like"]), _record(seeds["cnv_like"], feasible=False)]
        records.append(_record(seeds["mlp_like"]))  # duplicate
        records.append(_record(seeds["residual"]))
        chosen = select_records(records, {}, "auto")
        assert [r.hash for r in chosen] == [records[0].hash, records[3].hash]
        assert len(select_records(records, {}, "feasible", limit=1)) == 1

    def test_front_follows_the_summary(self) -> None:
        seeds = seed_genotypes()
        records = [_record(seeds["mlp_like"]), _record(seeds["cnv_like"])]
        summary = {"pareto_front": [{"trial": 0, "hash": records[1].hash}]}
        assert [r.hash for r in select_records(records, summary, "auto")] == [records[1].hash]
        assert [r.hash for r in select_records(records, summary, "feasible")] == [
            r.hash for r in records
        ]
        with pytest.raises(ValueError, match="Pareto front"):
            select_records(records, {}, "front")

    def test_build_steps_follow_the_oracle_flow(self) -> None:
        plus = build_steps(OracleConfig(fpga_part="x", streamline="plus"))
        default = build_steps(OracleConfig(fpga_part="x", streamline="default"))
        assert STREAMLINE_PLUS_STEP in plus
        assert "step_streamline" in default
        assert plus[: plus.index(STREAMLINE_PLUS_STEP)] == ["step_qonnx_to_finn", "step_tidy_up"]
        assert plus[-1] == "step_deployment_package"
        assert "step_target_fps_parallelization" in plus
        assert plus.index("step_minimize_bit_width") > plus.index("step_apply_folding_config")

    def test_cli_help(self) -> None:
        result = CliRunner().invoke(nas_export, ["--help"])
        assert result.exit_code == 0
        assert "--select" in result.output
        assert "--model-path-prefix" in result.output


@pytest.mark.fpgadataflow
class TestExportRun:
    """The exported files of a small sampling run."""

    def test_models_records_and_bench_config(self, tmp_path: Path) -> None:
        seeds = seed_genotypes()
        records = [
            _record(seeds["mlp_like"]),
            _record(seeds["cnv_like"], feasible=False),
            _record(seeds["residual"]),
        ]
        run = _write_run(tmp_path, records)
        out = tmp_path / "export"
        summary = export_run(run, out, model_path_prefix="/share/nas/export", seed=1)

        assert summary["count"] == 2
        models = sorted(out.glob("*.onnx"))
        assert [m.name for m in models] == sorted(
            f"{r.hash[:16]}.onnx" for r in (records[0], records[2])
        )
        for model in models:
            onnx.checker.check_model(onnx.load(str(model)))
        exported = [json.loads(line) for line in (out / RECORDS_FILE).read_text().splitlines()]
        assert [e["hash"] for e in exported] == [records[0].hash, records[2].hash]
        assert json.loads((out / EXPORT_SUMMARY_FILE).read_text())["count"] == 2

        bench = yaml.safe_load((out / BENCH_CONFIG_FILE).read_text())
        assert len(bench) == 2
        entry = bench[0]
        assert entry["dut"] == [DUT]
        assert entry["model_path"] == [f"/share/nas/export/{records[0].hash[:16]}.onnx"]
        assert entry["board"] == ["Pynq-Z1"]
        assert entry["target_fps"] == [1000]
        assert entry["synth_clk_period_ns"] == [10.0]
        assert entry["generate_outputs"] == [["estimate_reports", "bitfile"]]
        assert STREAMLINE_PLUS_STEP in entry["steps"][0]
        # every value is a one-element list (the grid notation of ci/cfg)
        assert all(isinstance(v, list) and len(v) == 1 for v in entry.values())

    def test_front_of_a_search_run(self, tmp_path: Path) -> None:
        seeds = seed_genotypes()
        records = [_record(seeds["mlp_like"]), _record(seeds["cnv_like"])]
        run = _write_run(tmp_path, records, front=[records[1].hash])
        summary = export_run(run, tmp_path / "export")
        assert [c["hash"] for c in summary["candidates"]] == [records[1].hash]
        assert summary["model_path_prefix"] == str((tmp_path / "export").absolute())

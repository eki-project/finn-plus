"""End-to-end smoke tests of the oracle on tiny Brevitas networks. No Vivado needed."""

import pytest

from pathlib import Path

from finn.nas import FailureCategory, FinnOracle, FoldingSpec, OracleConfig

pytest.importorskip("brevitas")
models = pytest.importorskip("tests.nas.models")
BadMLP, TinyCNN, TinyMLP = models.BadMLP, models.TinyCNN, models.TinyMLP

FPGA_PART = "xc7z020clg400-1"


def _config(tmp_path: Path, **kwargs) -> OracleConfig:
    base = {
        "fpga_part": FPGA_PART,
        "synth_clk_period_ns": 10.0,
        "folding": FoldingSpec(target_fps=1000),
        "isolation": "inprocess",
        "work_dir": str(tmp_path / "oracle"),
    }
    base.update(kwargs)
    return OracleConfig(**base)


@pytest.mark.fpgadataflow
class TestOracleSmoke:
    """Feasible and infeasible candidates through the in-process pipeline."""

    def test_tiny_mlp_feasible(self, tmp_path: Path) -> None:
        oracle = FinnOracle(_config(tmp_path))
        res = oracle.evaluate(TinyMLP(), TinyMLP.input_shape)
        assert res.feasible, (res.failure_category, res.failure_message, res.warnings)
        assert res.resources_total["LUT"] > 0
        assert res.est_fps is not None and res.est_fps > 0
        assert res.critical_path_cycles is not None and res.critical_path_cycles > 0
        assert set(res.timings_s) >= {"step_streamline", "folding", "estimate", "total"}
        assert any("PE" in attrs for attrs in res.folding.values())
        assert res.candidate_hash
        assert not res.from_cache

    def test_tiny_cnn_feasible(self, tmp_path: Path) -> None:
        oracle = FinnOracle(_config(tmp_path))
        res = oracle.evaluate(TinyCNN(), TinyCNN.input_shape)
        assert res.feasible, (res.failure_category, res.failure_message, res.warnings)
        assert res.resources_total["LUT"] > 0
        assert any(n.startswith("ConvolutionInputGenerator") for n in res.resources_per_node)

    def test_bad_mlp_infeasible(self, tmp_path: Path) -> None:
        oracle = FinnOracle(_config(tmp_path))
        res = oracle.evaluate(BadMLP(), BadMLP.input_shape)
        assert not res.feasible
        assert res.failure_category in {
            FailureCategory.UNSUPPORTED_OP,
            FailureCategory.STREAMLINE_INCOMPLETE,
            FailureCategory.UNMAPPED_LAYER,
            FailureCategory.DATATYPE_UNSUPPORTED,
        }
        assert res.failure_stage == "step_create_dataflow_partition"
        assert res.failure_node is not None

    def test_cache_hit(self, tmp_path: Path) -> None:
        oracle = FinnOracle(_config(tmp_path, cache_path=str(tmp_path / "cache.sqlite")))
        first = oracle.evaluate(TinyMLP(), TinyMLP.input_shape)
        second = oracle.evaluate(TinyMLP(), TinyMLP.input_shape)
        assert first.feasible and second.feasible
        assert second.from_cache
        assert first.candidate_hash == second.candidate_hash
        assert first.resources_total == second.resources_total
        # A different architecture must not hit the cache.
        third = oracle.evaluate(TinyMLP(hidden=12), TinyMLP.input_shape)
        assert not third.from_cache

    def test_target_missed_is_reported(self, tmp_path: Path) -> None:
        spec = FoldingSpec(target_fps=10**9, require_target_met=True)
        oracle = FinnOracle(_config(tmp_path, folding=spec))
        res = oracle.evaluate(TinyMLP(), TinyMLP.input_shape)
        assert res.throughput_target_missed
        assert res.failure_category == FailureCategory.THROUGHPUT_TARGET_MISSED
        # Estimates are still available for the best achievable folding.
        assert res.resources_total["LUT"] > 0

    def test_missing_part_is_config_error(self, tmp_path: Path) -> None:
        oracle = FinnOracle(_config(tmp_path, fpga_part=None))
        res = oracle.evaluate(TinyMLP(), TinyMLP.input_shape)
        assert res.failure_category == FailureCategory.CONFIG_ERROR


@pytest.mark.fpgadataflow
class TestOracleSubprocess:
    """The spawn-isolated worker."""

    def test_process_matches_inprocess(self, tmp_path: Path) -> None:
        model = TinyMLP()
        inproc = FinnOracle(_config(tmp_path)).evaluate(model, TinyMLP.input_shape)
        proc = FinnOracle(_config(tmp_path, isolation="process", timeout_s=600)).evaluate(
            model, TinyMLP.input_shape
        )
        assert proc.feasible, (proc.failure_category, proc.failure_message, proc.warnings)
        assert proc.resources_total == inproc.resources_total
        assert proc.cycles_per_node == inproc.cycles_per_node

    def test_timeout(self, tmp_path: Path) -> None:
        oracle = FinnOracle(_config(tmp_path, isolation="process", timeout_s=0.5))
        res = oracle.evaluate(TinyMLP(), TinyMLP.input_shape)
        assert res.failure_category == FailureCategory.TIMEOUT

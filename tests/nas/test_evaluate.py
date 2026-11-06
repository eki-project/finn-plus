"""Tests of the parallel genotype evaluator and the sampling experiment driver."""

import pytest

import json
import random
import torch
from pathlib import Path
from torch import nn

from finn.nas.config import FoldingSpec, OracleConfig
from finn.nas.evaluate import EvaluatorConfig, GenotypeEvaluator, randomize_batchnorm
from finn.nas.experiments.sampling import (
    RECORDS_FILE,
    SUMMARY_FILE,
    read_records,
    run_sampling,
    summarize,
)
from finn.nas.space import SearchSpace, SpaceConfig, seed_genotypes


def _evaluator_config(tmp_path: Path, **kwargs) -> EvaluatorConfig:
    oracle = OracleConfig(
        fpga_part="xc7z020clg400-1",
        synth_clk_period_ns=10.0,
        folding=FoldingSpec(target_fps=100),
        isolation="inprocess",
        work_dir=str(tmp_path / "oracle"),
    )
    return EvaluatorConfig(oracle=oracle, compute_proxies=False, **kwargs)


@pytest.mark.util
def test_randomize_batchnorm_is_deterministic_and_non_identity() -> None:
    def make() -> nn.Module:
        return nn.Sequential(nn.BatchNorm2d(4), nn.BatchNorm1d(3))

    a, b = randomize_batchnorm(make(), seed=7), randomize_batchnorm(make(), seed=7)
    for pa, pb in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.equal(pa, pb)
    assert not torch.allclose(a[0].weight, torch.ones(4))
    assert not torch.allclose(a[0].running_mean, torch.zeros(4))


@pytest.mark.fpgadataflow
class TestEvaluator:
    """In-process and worker-pool evaluation give the same verdicts."""

    def test_in_process_and_workers_agree(self, tmp_path: Path) -> None:
        seeds = seed_genotypes()
        genotypes = [seeds["mlp_like"], seeds["cnv_like"]]
        config = _evaluator_config(tmp_path)
        with GenotypeEvaluator(config, tmp_path / "w0", n_workers=0) as evaluator:
            local = evaluator.evaluate(genotypes)
        with GenotypeEvaluator(config, tmp_path / "w2", n_workers=2) as evaluator:
            pooled = evaluator.evaluate(genotypes)
        for a, b in zip(local, pooled, strict=True):
            assert a.hash == b.hash and a.result.feasible and b.result.feasible
            assert a.result.resources_total == b.result.resources_total
            assert a.params > 0 and a.nodes > 0
            assert a.result.candidate_hash == b.result.candidate_hash
        # JSON round trip of the records
        again = type(local[0]).from_dict(json.loads(json.dumps(local[0].to_dict())))
        assert again.result.feasible and again.genotype == local[0].genotype

    def test_timeout_is_reported(self, tmp_path: Path) -> None:
        config = _evaluator_config(tmp_path, timeout_s=0.001)
        with GenotypeEvaluator(config, tmp_path / "w", n_workers=0) as evaluator:
            (evaluation,) = evaluator.evaluate([seed_genotypes()["cnv_like"]])
        assert evaluation.result.failure_category.value == "timeout"

    def test_run_sampling_resumes(self, tmp_path: Path) -> None:
        space = SpaceConfig(max_nodes=8, max_params=100_000)
        config = _evaluator_config(tmp_path)
        out = tmp_path / "run"
        first = run_sampling(space, config, 3, 1, out, n_workers=0)
        assert first["count"] == 3 and (out / RECORDS_FILE).exists()
        assert json.loads((out / SUMMARY_FILE).read_text())["count"] == 3
        # extending the run evaluates only the new genotypes
        second = run_sampling(space, config, 5, 1, out, n_workers=0)
        records = read_records(out / RECORDS_FILE)
        assert second["count"] == len(records) == 5
        assert sorted(r.meta["index"] for r in records) == [0, 1, 2, 3, 4]
        # the summary is consistent with the records
        assert summarize(records)["feasible"] == second["feasible"]
        # the same genotypes come out of the same seed
        rng = random.Random(1)
        expected = [SearchSpace(space).sample(rng) for _ in range(5)]
        assert [r.genotype for r in sorted(records, key=lambda r: r.meta["index"])] == expected

"""A long run must survive duplicate draws and broken candidates (``finn.nas.evaluate``)."""

import pytest

import random
from pathlib import Path

from finn.nas.config import FoldingSpec, OracleConfig
from finn.nas.evaluate import EvaluatorConfig, candidate_dir, evaluate_genotype
from finn.nas.experiments.sampling import draw_genotypes
from finn.nas.result import FailureCategory
from finn.nas.space import SearchSpace, SpaceConfig, genotype_hash, seed_genotypes

pytest.importorskip("brevitas")


@pytest.mark.util
class TestDrawGenotypes:
    """Every genotype of a sampling run is evaluated exactly once."""

    def test_duplicate_draws_are_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seeds = seed_genotypes()
        draws = [seeds["mlp_like"], seeds["cnv_like"], seeds["mlp_like"], seeds["residual"]]
        monkeypatch.setattr(SearchSpace, "sample", lambda self, rng: draws.pop(0))  # noqa: ARG005
        done = {genotype_hash(seeds["residual"])}
        pending, skipped = draw_genotypes(SearchSpace(SpaceConfig()), 4, random.Random(0), done)
        assert [index for index, _ in pending] == [0, 1]
        assert skipped == 2
        assert len(done) == 3


@pytest.mark.util
class TestCandidateDir:
    """Concurrent evaluations of one genotype never share a directory."""

    def test_unique_per_evaluation(self, tmp_path: Path) -> None:
        first = candidate_dir(tmp_path / "work", "abcdef0123456789abcdef")
        second = candidate_dir(tmp_path / "work", "abcdef0123456789abcdef")
        assert first != second
        assert first.is_dir() and second.is_dir()
        assert first.name.startswith("abcdef0123456789-")
        assert second.name.startswith("abcdef0123456789-")


@pytest.mark.util
class TestBrokenCandidate:
    """An unexpected exception becomes an INTERNAL_ERROR record instead of ending the run."""

    def test_internal_error_is_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import finn.nas.space.build as build

        def explode(genotype: dict) -> None:  # noqa: ARG001
            raise RuntimeError("boom")

        monkeypatch.setattr(build, "to_module", explode)
        config = EvaluatorConfig(
            oracle=OracleConfig(fpga_part="xc7z020clg400-1", folding=FoldingSpec(target_fps=1000)),
            compute_proxies=False,
            timeout_s=0,
        )
        evaluation = evaluate_genotype(seed_genotypes()["mlp_like"], config, tmp_path / "work")
        assert evaluation.result.failure_category == FailureCategory.INTERNAL_ERROR
        assert "RuntimeError: boom" in (evaluation.result.failure_message or "")
        assert evaluation.result.failure_stage == "worker"

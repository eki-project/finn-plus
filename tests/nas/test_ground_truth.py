"""Tests of the proxy ground-truth driver on synthetic data."""

import pytest

import json
import random
import torch
from pathlib import Path

from finn.nas.data import ImageDataset
from finn.nas.evaluate import CandidateEvaluation
from finn.nas.experiments.ground_truth import (
    GroundTruthConfig,
    rank_correlations,
    run_ground_truth,
    select_candidates,
)
from finn.nas.experiments.sampling import append_records
from finn.nas.result import EvalResult
from finn.nas.space import SearchSpace, SpaceConfig, count_params, genotype_hash, infer_shapes
from finn.nas.train import TrainConfig


def _records(n: int, feasible_every: int = 1) -> list[CandidateEvaluation]:
    space = SearchSpace(SpaceConfig(input_shape=(3, 8, 8), num_classes=2, max_nodes=6))
    rng = random.Random(0)
    records = []
    for i in range(n):
        g = space.sample(rng)
        feasible = i % feasible_every == 0
        result = EvalResult(feasible=feasible, resources_total={"LUT": 100.0 + i})
        records.append(
            CandidateEvaluation(
                genotype=g,
                hash=genotype_hash(g),
                result=result,
                proxies={"scores": {"nwot": float(i), "synflow": float(-i)}, "errors": {}},
                params=count_params(g),
                nodes=len(infer_shapes(g).nodes),
            )
        )
    return records


@pytest.mark.util
class TestSelection:
    def test_stratified_spans_sizes_and_skips_infeasible(self) -> None:
        records = _records(12, feasible_every=2)
        chosen = select_candidates(records, GroundTruthConfig(count=3))
        assert len(chosen) == 3 and all(r.result.feasible for r in chosen)
        params = [r.params for r in chosen]
        assert params == sorted(params)

    def test_first_and_dedup(self) -> None:
        records = _records(4)
        records.append(records[0])
        chosen = select_candidates(records, GroundTruthConfig(count=10, selection="first"))
        assert [r.hash for r in chosen] == [r.hash for r in records[:4]]

    def test_rank_correlations(self) -> None:
        rows = [
            {
                "train": {"best_val_acc": acc},
                "proxies": {"good": acc * 2, "bad": -acc, "missing": float("nan")},
                "params": 10,
                "estimates": {"LUT": 1.0},
            }
            for acc in (0.1, 0.4, 0.2, 0.9, 0.5)
        ]
        summary = rank_correlations(rows)
        assert summary["proxies"]["good"]["spearman"] == pytest.approx(1.0)
        assert summary["proxies"]["bad"]["kendall"] == pytest.approx(-1.0)
        assert summary["proxies"]["missing"] == {"n": 0}
        assert summary["proxies"]["params"]["n"] == 5


@pytest.mark.util
def test_run_ground_truth_trains_and_resumes(tmp_path: Path) -> None:
    records = _records(3)
    records_path = tmp_path / "samples.jsonl"
    append_records(records_path, records)
    generator = torch.Generator().manual_seed(0)
    labels = torch.arange(64) % 2
    images = torch.where(labels == 0, 40, 200)[:, None, None, None] + torch.randint(
        -20, 21, (64, 3, 8, 8), generator=generator
    )
    images = images.clamp(0, 255).to(torch.uint8)
    dataset = ImageDataset(images, labels, images[:16], labels[:16], images[:16], labels[:16], 2)
    config = GroundTruthConfig(count=2, train=TrainConfig(epochs=1, batch_size=16))
    summary = run_ground_truth(records_path, tmp_path / "gt", config, dataset, "cpu")
    assert summary["trained"] == 2 and summary["usable"] == 2
    rows = [json.loads(line) for line in (tmp_path / "gt" / "training.jsonl").open()]
    assert all(row["train"]["epochs_run"] == 1 for row in rows)
    # resuming trains nothing new
    again = run_ground_truth(records_path, tmp_path / "gt", config, dataset, "cpu")
    assert again["trained"] == 2

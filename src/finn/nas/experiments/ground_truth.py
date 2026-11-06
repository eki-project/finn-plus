"""Ground truth for the zero-cost proxies: train sampled candidates and correlate.

Published proxies are validated on float networks. Whether they rank low-bit Brevitas networks
of this search space after quantization-aware training is an open question, so a subset of
the feasible candidates of a sampling run is trained with the shared short recipe and every
proxy is rank-correlated with the resulting validation accuracy.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from finn.nas.experiments.sampling import read_records
from finn.nas.train import TrainConfig, train

if TYPE_CHECKING:
    import torch
    from collections.abc import Callable

    from finn.nas.data import ImageDataset
    from finn.nas.evaluate import CandidateEvaluation

TRAINING_FILE = "training.jsonl"
SUMMARY_FILE = "summary.json"
CONFIG_FILE = "config.json"


@dataclass
class GroundTruthConfig:
    """Which candidates to train and how."""

    count: int = 50
    #: "stratified": evenly spaced over the candidates sorted by parameter count, so that
    #: the trained set spans the size range; "first": the first feasible records.
    selection: str = "stratified"
    seed: int = 0
    train: TrainConfig = field(default_factory=lambda: TrainConfig(epochs=10))
    #: Record only candidates whose proxies were computed (needed for the correlation).
    require_proxies: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundTruthConfig:
        """Reconstruct from :meth:`to_dict` output."""
        data = dict(data)
        data["train"] = TrainConfig(**data["train"])
        return cls(**data)


def select_candidates(
    records: list[CandidateEvaluation], config: GroundTruthConfig
) -> list[CandidateEvaluation]:
    """Choose the feasible candidates to train, deduplicated by hash."""
    seen: set[str] = set()
    feasible = []
    for record in records:
        if not record.result.feasible or record.hash in seen:
            continue
        if config.require_proxies and not record.proxies:
            continue
        seen.add(record.hash)
        feasible.append(record)
    if config.count >= len(feasible) or config.selection == "first":
        return feasible[: config.count]
    ordered = sorted(feasible, key=lambda r: (r.params, r.hash))
    step = len(ordered) / config.count
    return [ordered[int(i * step)] for i in range(config.count)]


def rank_correlations(rows: list[dict[str, Any]], target: str = "best_val_acc") -> dict[str, Any]:
    """Spearman and Kendall correlation of every proxy (and size statistic) with ``target``.

    Rows are the training records; proxies live in ``row["proxies"]``. Pairs with a
    non-finite proxy value are dropped per proxy, and the number used is reported.
    """
    from scipy import stats

    y_all = [row["train"][target] for row in rows]
    keys: set[str] = set()
    for row in rows:
        keys.update(row.get("proxies", {}).keys())
    extra = {
        "params": [row["params"] for row in rows],
        "estimated_lut": [row["estimates"].get("LUT", float("nan")) for row in rows],
    }
    result: dict[str, Any] = {"count": len(rows), "target": target, "proxies": {}}
    series = {k: [row.get("proxies", {}).get(k, float("nan")) for row in rows] for k in keys}
    series.update(extra)
    for name, values in sorted(series.items()):
        pairs = [
            (v, y)
            for v, y in zip(values, y_all, strict=True)
            if isinstance(v, (int, float)) and math.isfinite(v) and math.isfinite(y)
        ]
        if len(pairs) < 3:
            result["proxies"][name] = {"n": len(pairs)}
            continue
        xs, ys = zip(*pairs, strict=True)
        spearman = stats.spearmanr(xs, ys)
        kendall = stats.kendalltau(xs, ys)
        result["proxies"][name] = {
            "n": len(pairs),
            "spearman": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "kendall": float(kendall.statistic),
            "kendall_p": float(kendall.pvalue),
        }
    return result


def read_training(path: Path) -> list[dict[str, Any]]:
    """Read the training records (missing file: empty list)."""
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def run_ground_truth(
    records_path: str | Path,
    out_dir: str | Path,
    config: GroundTruthConfig,
    dataset: ImageDataset,
    device: torch.device | str,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train the selected candidates of a sampling run and correlate proxies with accuracy.

    Resumable: candidates with a training record in ``out_dir`` are skipped.
    """
    import torch

    from finn.nas.space.build import to_module

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / CONFIG_FILE).write_text(
        json.dumps({"records": str(records_path), "config": config.to_dict()}, indent=2)
    )
    say = progress or (lambda _msg: None)
    records = read_records(Path(records_path))
    selected = select_candidates(records, config)
    training_path = out_dir / TRAINING_FILE
    done = {row["hash"] for row in read_training(training_path)}
    todo = [r for r in selected if r.hash not in done]
    say(f"{len(selected)} candidates selected, {len(todo)} to train")
    t_start = time.perf_counter()
    for i, record in enumerate(todo):
        torch.manual_seed(config.seed)
        model = to_module(record.genotype)
        result = train(model, dataset, config.train, device=device)
        row = {
            "hash": record.hash,
            "genotype": record.genotype,
            "params": record.params,
            "nodes": record.nodes,
            "proxies": (record.proxies or {}).get("scores", {}),
            "estimates": record.result.resources_total,
            "est_fps": record.result.est_fps,
            "train": result.to_dict(),
        }
        with training_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        elapsed = time.perf_counter() - t_start
        say(
            f"{i + 1}/{len(todo)} trained: best val acc {result.best_val_acc:.3f} "
            f"({result.epochs_run} epochs, {result.seconds:.0f} s, error={result.error}), "
            f"{elapsed / (i + 1):.0f} s per candidate"
        )
    rows = [row for row in read_training(training_path) if row["train"]["error"] is None]
    summary = rank_correlations(rows) if rows else {"count": 0}
    summary["trained"] = len(read_training(training_path))
    summary["usable"] = len(rows)
    (out_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    return summary

"""Random sampling of the search space through the oracle.

The feasibility rate and the failure histogram of uniform random samples are the baseline
that every search has to beat, and the records (genotype, verdict, estimates, proxies) are
the raw material of the proxy ground-truth study. Results are appended to a JSON-lines file,
so a run can be resumed or extended.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from finn.nas.evaluate import CandidateEvaluation, GenotypeEvaluator
from finn.nas.space import SearchSpace, genotype_hash

if TYPE_CHECKING:
    import torch
    from collections.abc import Callable, Iterable

    from finn.nas.evaluate import EvaluatorConfig
    from finn.nas.space import SpaceConfig

RECORDS_FILE = "samples.jsonl"
SUMMARY_FILE = "summary.json"
CONFIG_FILE = "config.json"


def read_records(path: Path) -> list[CandidateEvaluation]:
    """Read all evaluations from a JSON-lines records file (missing file: empty list)."""
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(CandidateEvaluation.from_dict(json.loads(line)))
    return records


def append_records(path: Path, evaluations: Iterable[CandidateEvaluation]) -> None:
    """Append evaluations to a JSON-lines records file."""
    with path.open("a") as f:
        for evaluation in evaluations:
            f.write(json.dumps(evaluation.to_dict()) + "\n")


def summarize(evaluations: list[CandidateEvaluation]) -> dict[str, Any]:
    """Feasibility rate, failure histograms and timing of a set of evaluations."""
    n = len(evaluations)
    feasible = [e for e in evaluations if e.result.feasible]
    infeasible = [e for e in evaluations if not e.result.feasible]
    summary: dict[str, Any] = {
        "count": n,
        "feasible": len(feasible),
        "feasibility_rate": len(feasible) / n if n else float("nan"),
        "failure_category": dict(Counter(e.result.failure_category.value for e in infeasible)),
        "failure_stage": dict(Counter(str(e.result.failure_stage) for e in infeasible)),
        "failure_op_type": dict(
            Counter(str(e.result.failure_op_type) for e in infeasible).most_common(15)
        ),
        "mean_seconds": sum(e.seconds for e in evaluations) / n if n else float("nan"),
        "mean_nodes": sum(e.nodes for e in evaluations) / n if n else float("nan"),
        "mean_nodes_feasible": (
            sum(e.nodes for e in feasible) / len(feasible) if feasible else float("nan")
        ),
    }
    if feasible:
        luts = sorted(e.result.resources_total.get("LUT", 0.0) for e in feasible)
        fps = sorted(e.result.est_fps or 0.0 for e in feasible)
        summary["feasible_lut"] = {"min": luts[0], "median": luts[len(luts) // 2], "max": luts[-1]}
        summary["feasible_fps"] = {"min": fps[0], "median": fps[len(fps) // 2], "max": fps[-1]}
    with_proxies = [e for e in evaluations if e.proxies]
    if with_proxies:
        failed: Counter[str] = Counter()
        for e in with_proxies:
            failed.update(e.proxies["errors"].keys())  # type: ignore[index]
        summary["proxy_failures"] = dict(failed)
    return summary


def run_sampling(
    space_config: SpaceConfig,
    evaluator_config: EvaluatorConfig,
    count: int,
    seed: int,
    out_dir: str | Path,
    n_workers: int = 1,
    proxy_batch: tuple[torch.Tensor, torch.Tensor] | None = None,
    chunk_size: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Draw ``count`` random genotypes, evaluate them and write records and summary.

    Genotypes already present in the records file (same hash) are not evaluated again, so
    re-running with a larger ``count`` extends an earlier run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / RECORDS_FILE
    (out_dir / CONFIG_FILE).write_text(
        json.dumps(
            {
                "space": space_config.to_dict(),
                "evaluator": evaluator_config.to_dict(),
                "count": count,
                "seed": seed,
                "n_workers": n_workers,
            },
            indent=2,
        )
    )
    existing = read_records(records_path)
    done = {e.hash for e in existing}
    space = SearchSpace(space_config)
    rng = random.Random(seed)
    pending: list[tuple[int, dict[str, Any]]] = []
    for index in range(count):
        genotype = space.sample(rng)
        if genotype_hash(genotype) not in done:
            pending.append((index, genotype))
    say = progress or (lambda _msg: None)
    say(f"{len(existing)} records exist, {len(pending)} of {count} genotypes to evaluate")
    chunk = chunk_size or max(1, 4 * max(1, n_workers))
    t_start = time.perf_counter()
    evaluated = 0
    with GenotypeEvaluator(
        evaluator_config, out_dir / "work", n_workers=n_workers, proxy_batch=proxy_batch
    ) as evaluator:
        for start in range(0, len(pending), chunk):
            batch = pending[start : start + chunk]
            evaluations = evaluator.evaluate(
                [g for _, g in batch], seeds=[seed * 100003 + i for i, _ in batch]
            )
            for (index, _), evaluation in zip(batch, evaluations, strict=True):
                evaluation.meta = {"index": index, "seed": seed}
            append_records(records_path, evaluations)
            evaluated += len(batch)
            feasible = sum(e.result.feasible for e in evaluations)
            elapsed = time.perf_counter() - t_start
            say(
                f"{evaluated}/{len(pending)} evaluated, {feasible}/{len(batch)} feasible in "
                f"this chunk, {elapsed / evaluated:.2f} s per candidate"
            )
    summary = summarize(read_records(records_path))
    (out_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    return summary

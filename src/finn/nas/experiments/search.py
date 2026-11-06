"""The search experiment: NSGA-II (or random search) over the space through the oracle.

``run_search`` drives an Optuna study in ask/evaluate/tell batches: the sampler proposes
genotypes, the :class:`~finn.nas.evaluate.GenotypeEvaluator` scores a batch in worker
processes, and every trial is told its objective values plus its feasibility, in the form
selected by :class:`~finn.nas.search.SearchConfig`. The study lives in a journal file so a
run can be resumed; every evaluation is additionally appended to a JSON-lines file that the
summary and the later analysis read (the records are self-contained, the journal is not).
"""

from __future__ import annotations

import copy
import json
import optuna
import time
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState
from pathlib import Path
from typing import TYPE_CHECKING, Any

from finn.nas.evaluate import CandidateEvaluation, GenotypeEvaluator
from finn.nas.experiments.sampling import append_records, read_records, summarize
from finn.nas.search import (
    SearchConfig,
    hypervolume_2d,
    hypervolume_full,
    infeasible_values,
    make_sampler,
    nondominated,
    objective_values,
    proxy_score,
    reference_point,
    to_losses,
    violation_of,
)
from finn.nas.space import SearchSpace, genotype_hash

if TYPE_CHECKING:
    import torch
    from collections.abc import Callable
    from optuna.trial import Trial

    from finn.nas.evaluate import EvaluatorConfig
    from finn.nas.search import GenotypeSampler
    from finn.nas.space import SpaceConfig

RECORDS_FILE = "records.jsonl"
SUMMARY_FILE = "summary.json"
CONFIG_FILE = "config.json"
JOURNAL_FILE = "study.journal"
STUDY_NAME = "nas_search"


def create_study(
    search_config: SearchConfig, sampler: GenotypeSampler, out_dir: Path
) -> optuna.Study:
    """Create or reopen the study of ``out_dir`` (journal file storage, resumable)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = JournalStorage(JournalFileBackend(str(out_dir / JOURNAL_FILE)))
    study = optuna.create_study(
        directions=search_config.directions,
        sampler=sampler,
        storage=storage,
        study_name=STUDY_NAME,
        load_if_exists=True,
    )
    # Trials left running by an interrupted run were never evaluated: close them so they do
    # not linger (they carry no record and therefore do not count towards the budget).
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(trial.number, state=TrialState.FAIL)
    return study


def _ask_batch(
    study: optuna.Study, sampler: GenotypeSampler, max_size: int
) -> tuple[int, list[tuple[Trial, dict[str, Any]]]]:
    """Ask up to ``max_size`` trials of one generation and propose their genotypes."""
    first = study.ask()
    generation = sampler.generation_of(study, first)
    size = min(max_size, sampler.generation_size_left(study, first))
    trials = [first] + [study.ask() for _ in range(size - 1)]
    batch = []
    for trial in trials:
        genotype = sampler.propose(study, trial)
        trial.set_user_attr("genotype", genotype)
        trial.set_user_attr("hash", genotype_hash(genotype))
        trial.set_user_attr("generation", generation)
        batch.append((trial, genotype))
    return generation, batch


def _tell(
    study: optuna.Study,
    search_config: SearchConfig,
    trial: Trial,
    evaluation: CandidateEvaluation,
    generation: int,
) -> None:
    """Store the evaluation on the trial, finish it and fill ``evaluation.meta``."""
    result = evaluation.result
    values = objective_values(search_config, evaluation)
    violation = violation_of(search_config, result)
    trial.set_user_attr("feasible", result.feasible)
    trial.set_user_attr("failure_category", result.failure_category.value)
    trial.set_user_attr("failure_stage", result.failure_stage)
    trial.set_user_attr("resources", result.resources_total)
    trial.set_user_attr("est_fps", result.est_fps)
    trial.set_user_attr("proxies", (evaluation.proxies or {}).get("scores", {}))
    trial.set_user_attr("violation", violation)
    trial.set_user_attr("params", evaluation.params)
    trial.set_user_attr("nodes", evaluation.nodes)
    told: list[float] | None
    if values is not None:
        if search_config.feasibility == "constraint":
            trial.set_constraint("violation", 0.0)
        told = values
        study.tell(trial, values)
    elif search_config.feasibility == "constraint":
        trial.set_constraint("violation", violation)
        told = infeasible_values(search_config)
        study.tell(trial, told)
    else:
        told = None
        study.tell(trial, state=TrialState.FAIL)
    evaluation.meta.update(
        {
            "trial": trial.number,
            "generation": generation,
            "violation": violation,
            "values": told,
            "feasible": result.feasible,
        }
    )


def _duplicate_of(original: CandidateEvaluation) -> CandidateEvaluation:
    """Return the record of a genotype proposed again: the earlier verdict, not re-run."""
    return CandidateEvaluation(
        genotype=copy.deepcopy(original.genotype),
        hash=original.hash,
        result=copy.deepcopy(original.result),
        proxies=copy.deepcopy(original.proxies),
        params=original.params,
        nodes=original.nodes,
        seconds=0.0,
        meta={"duplicate_of": original.meta.get("trial")},
    )


def run_search(
    space_config: SpaceConfig,
    evaluator_config: EvaluatorConfig,
    search_config: SearchConfig,
    out_dir: str | Path,
    n_workers: int = 1,
    proxy_batch: tuple[torch.Tensor, torch.Tensor] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run (or resume) a search until ``search_config.budget`` trials exist; return the summary.

    Every trial, duplicate and failed ones included, counts towards the budget: a converged
    population that keeps proposing known genotypes must not run forever. Duplicates are not
    re-evaluated but copy the earlier verdict (``meta["duplicate_of"]``).
    """
    if "proxy" in search_config.objectives and not evaluator_config.compute_proxies:
        raise ValueError('objective "proxy" needs compute_proxies=True on the evaluator')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / RECORDS_FILE
    (out_dir / CONFIG_FILE).write_text(
        json.dumps(
            {
                "space": space_config.to_dict(),
                "evaluator": evaluator_config.to_dict(),
                "search": search_config.to_dict(),
                "n_workers": n_workers,
            },
            indent=2,
        )
    )
    say = progress or (lambda _msg: None)
    sampler = make_sampler(SearchSpace(space_config), search_config)
    study = create_study(search_config, sampler, out_dir)
    existing = read_records(records_path)
    known: dict[str, CandidateEvaluation] = {}
    for evaluation in existing:
        known.setdefault(evaluation.hash, evaluation)
    done = len(existing)
    say(f"{done} of {search_config.budget} trials exist, {len(known)} distinct genotypes")
    batch_size = max(1, n_workers)
    t_start = time.perf_counter()
    evaluated = 0
    with GenotypeEvaluator(
        evaluator_config, out_dir / "work", n_workers=n_workers, proxy_batch=proxy_batch
    ) as evaluator:
        while done < search_config.budget:
            generation, batch = _ask_batch(
                study, sampler, min(batch_size, search_config.budget - done)
            )
            fresh = [(t, g) for t, g in batch if t.user_attrs["hash"] not in known]
            # The same genotype twice within one batch is evaluated once as well.
            unique: dict[str, tuple[Trial, dict[str, Any]]] = {}
            for trial, genotype in fresh:
                unique.setdefault(trial.user_attrs["hash"], (trial, genotype))
            evaluations = evaluator.evaluate(
                [g for _, g in unique.values()],
                seeds=[search_config.seed * 100003 + t.number for t, _ in unique.values()],
            )
            for (trial, _), evaluation in zip(unique.values(), evaluations, strict=True):
                _tell(study, search_config, trial, evaluation, generation)
                known[evaluation.hash] = evaluation
            records = list(evaluations)
            evaluated_trials = {t.number for t, _ in unique.values()}
            for trial, _ in batch:
                if trial.number in evaluated_trials:
                    continue
                duplicate = _duplicate_of(known[trial.user_attrs["hash"]])
                _tell(study, search_config, trial, duplicate, generation)
                records.append(duplicate)
            records.sort(key=lambda e: e.meta["trial"])
            append_records(records_path, records)
            done += len(records)
            evaluated += len(evaluations)
            feasible = sum(e.result.feasible for e in records)
            elapsed = time.perf_counter() - t_start
            say(
                f"generation {generation}: {done}/{search_config.budget} trials, "
                f"{feasible}/{len(records)} feasible in this batch "
                f"({len(records) - len(evaluations)} duplicates), "
                f"{elapsed / max(1, evaluated):.2f} s per evaluation"
            )
    summary = summarize_search(search_config, read_records(records_path))
    (out_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    return summary


def pareto_front(
    search_config: SearchConfig, records: list[CandidateEvaluation]
) -> list[dict[str, Any]]:
    """Return the non-dominated feasible records (one per genotype) with their essentials."""
    feasible: dict[str, CandidateEvaluation] = {}
    for record in records:
        if record.result.feasible and record.meta.get("values") is not None:
            feasible.setdefault(record.hash, record)
    entries = list(feasible.values())
    losses = [to_losses(e.meta["values"], search_config.directions) for e in entries]
    front = []
    for index in nondominated(losses):
        record = entries[index]
        front.append(
            {
                "trial": record.meta["trial"],
                "hash": record.hash,
                "values": record.meta["values"],
                "resources": record.result.resources_total,
                "est_fps": record.result.est_fps,
                "proxy": proxy_score(record, search_config.proxy),
                "params": record.params,
                "nodes": record.nodes,
            }
        )
    return sorted(front, key=lambda e: e["trial"])


def hypervolume_history(
    search_config: SearchConfig, records: list[CandidateEvaluation]
) -> dict[str, Any]:
    """Hypervolume of the feasible trials up to each generation (cumulative).

    The 2-D projection is proxy vs. LUT when both are objectives, otherwise the first two
    objectives; its hypervolume is exact. The full-dimensional one uses Optuna's private
    hypervolume code and is ``None`` when that is not usable. The reference point is fixed
    from all feasible trials of the run (:func:`finn.nas.search.reference_point`) so the
    history is monotone and comparable across generations of one run.
    """
    objectives = search_config.objectives
    pair = ["proxy", "LUT"] if {"proxy", "LUT"} <= set(objectives) else objectives[:2]
    pair_index = [objectives.index(name) for name in pair]
    feasible = [r for r in records if r.result.feasible and r.meta.get("values") is not None]
    losses = [to_losses(r.meta["values"], search_config.directions) for r in feasible]
    generations = [r.meta["generation"] for r in feasible]
    reference = reference_point(losses) if losses else [1.0] * len(objectives)
    reference_2d = [reference[i] for i in pair_index]
    history_2d: list[float] = []
    history_full: list[float | None] = []
    last_generation = max((r.meta["generation"] for r in records), default=-1)
    for generation in range(last_generation + 1):
        so_far = [p for p, g in zip(losses, generations, strict=True) if g <= generation]
        history_2d.append(
            hypervolume_2d([[p[i] for i in pair_index] for p in so_far], reference_2d)
        )
        history_full.append(
            hypervolume_full(so_far, reference) if len(objectives) > 2 else history_2d[-1]
        )
    return {
        "objectives_2d": pair,
        "reference_point": dict(zip(objectives, reference, strict=True)),
        "per_generation_2d": history_2d,
        "per_generation_full": history_full,
    }


def summarize_search(
    search_config: SearchConfig, records: list[CandidateEvaluation]
) -> dict[str, Any]:
    """Per-generation feasibility, the overall rates, the Pareto front and the hypervolumes."""
    by_generation: dict[int, list[CandidateEvaluation]] = {}
    for record in records:
        by_generation.setdefault(record.meta["generation"], []).append(record)
    seen_feasible: set[str] = set()
    generations = []
    for generation in sorted(by_generation):
        group = by_generation[generation]
        feasible = [r for r in group if r.result.feasible]
        new_hashes = {r.hash for r in feasible} - seen_feasible
        seen_feasible |= new_hashes
        generations.append(
            {
                "generation": generation,
                "trials": len(group),
                "feasible": len(feasible),
                "feasibility_rate": len(feasible) / len(group),
                "mean_violation": sum(r.meta["violation"] for r in group) / len(group),
                "new_feasible_hashes": len(new_hashes),
                "duplicates": sum("duplicate_of" in r.meta for r in group),
            }
        )
    front = pareto_front(search_config, records)
    return {
        "search": search_config.to_dict(),
        "count": len(records),
        "feasible": sum(r.result.feasible for r in records),
        "feasibility_rate": (
            sum(r.result.feasible for r in records) / len(records) if records else float("nan")
        ),
        "distinct_feasible": len(seen_feasible),
        "duplicates": sum("duplicate_of" in r.meta for r in records),
        "generations": generations,
        "pareto_front_size": len(front),
        "pareto_front": front,
        "hypervolume": hypervolume_history(search_config, records),
        "evaluations": summarize([r for r in records if "duplicate_of" not in r.meta]),
    }

"""Tests of the NSGA-II search over genotypes and its experiment driver."""

from __future__ import annotations

import pytest

import json
import optuna
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from finn.nas.config import FoldingSpec, OracleConfig
from finn.nas.evaluate import CandidateEvaluation, EvaluatorConfig
from finn.nas.experiments import search as driver
from finn.nas.experiments.search import run_search
from finn.nas.result import EvalResult, FailureCategory
from finn.nas.search import (
    INFEASIBLE_VALUE,
    MISSING_PROXY_VALUE,
    VIOLATION_LEVELS,
    GenotypeNSGAIISampler,
    SearchConfig,
    constrained_dominates,
    hypervolume_2d,
    hypervolume_full,
    nondominated,
    objective_values,
    reference_point,
    violation_of,
)
from finn.nas.space import SearchSpace, SpaceConfig, count_params, genotype_hash, infer_shapes

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from finn.nas.space.genotype import Genotype

SMALL_SPACE = SpaceConfig(max_nodes=8, max_params=100_000)


def _evaluator_config(tmp_path: Path, **kwargs: Any) -> EvaluatorConfig:
    """Oracle settings of the real runs (cheap folding target, in-process pipeline)."""
    oracle = OracleConfig(
        fpga_part="xc7z020clg400-1",
        synth_clk_period_ns=10.0,
        folding=FoldingSpec(target_fps=100),
        isolation="inprocess",
        work_dir=str(tmp_path / "oracle"),
    )
    kwargs.setdefault("compute_proxies", False)
    return EvaluatorConfig(oracle=oracle, **kwargs)


@pytest.mark.util
class TestSearchConfig:
    """Configuration round trip and validation."""

    def test_round_trip(self) -> None:
        config = SearchConfig(
            algorithm="random", objectives=["LUT", "cycles"], feasibility="discard", seed=3
        )
        again = SearchConfig.from_dict(json.loads(json.dumps(config.to_dict())))
        assert again == config
        assert again.directions == ["minimize", "minimize"]
        assert SearchConfig().directions == ["maximize", "minimize", "minimize", "minimize"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"objectives": ["LUT", "FF"]},
            {"objectives": []},
            {"objectives": ["LUT", "LUT"]},
            {"algorithm": "tpe"},
            {"feasibility": "ignore"},
            {"violation": "linear"},
            {"population_size": 1},
            {"mutation_prob": 1.5},
        ],
    )
    def test_rejects_bad_settings(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            SearchConfig(**kwargs)


@pytest.mark.util
class TestViolation:
    """Graded violations follow the pipeline order; binary ones are all 1."""

    def test_graded_order(self) -> None:
        graded = SearchConfig(violation="graded")
        stages = [
            ("export", FailureCategory.EXPORT_FAILED),
            ("step_streamline", FailureCategory.FRONTEND_FAILED),
            ("step_create_dataflow_partition", FailureCategory.UNSUPPORTED_OP),
            ("folding", FailureCategory.FOLDING_INFEASIBLE),
            ("step_minimize_bit_width", FailureCategory.DATATYPE_UNSUPPORTED),
            ("estimate", FailureCategory.RESOURCE_BUDGET_EXCEEDED),
        ]
        violations = [violation_of(graded, EvalResult.failed(s, c)) for s, c in stages]
        assert violations == sorted(violations, reverse=True)
        assert len(set(violations)) == VIOLATION_LEVELS
        assert violations[0] == 1.0 and violations[-1] == pytest.approx(1 / VIOLATION_LEVELS)
        # the same stage is ranked equally regardless of category
        assert violation_of(
            graded, EvalResult.failed("step_tidy_up", FailureCategory.FRONTEND_FAILED)
        ) == violation_of(
            graded, EvalResult.failed("step_qonnx_to_finn", FailureCategory.FRONTEND_FAILED)
        )
        # transient failures and unknown stages are the worst
        assert violation_of(graded, EvalResult.failed("worker", FailureCategory.TIMEOUT)) == 1.0
        assert violation_of(graded, EvalResult.failed("???", FailureCategory.INTERNAL_ERROR)) == 1.0
        assert violation_of(graded, EvalResult(feasible=True)) == 0.0

    def test_binary(self) -> None:
        binary = SearchConfig(violation="binary")
        for stage, category in [
            ("export", FailureCategory.EXPORT_FAILED),
            ("estimate", FailureCategory.RESOURCE_BUDGET_EXCEEDED),
        ]:
            assert violation_of(binary, EvalResult.failed(stage, category)) == 1.0
        assert violation_of(binary, EvalResult(feasible=True)) == 0.0

    def test_objective_values_and_sentinels(self) -> None:
        config = SearchConfig(objectives=["proxy", "LUT", "BRAM", "cycles"])
        result = EvalResult(
            feasible=True, resources_total={"LUT": 10.0, "BRAM_18K": 2.0}, max_cycles=400
        )
        evaluation = CandidateEvaluation(
            genotype={}, hash="x", result=result, proxies={"scores": {"nwot": 5.5}, "errors": {}}
        )
        assert objective_values(config, evaluation) == [5.5, 10.0, 2.0, 400.0]
        evaluation.proxies = {"scores": {"nwot": float("nan")}, "errors": {"nwot": "boom"}}
        assert objective_values(config, evaluation) == [MISSING_PROXY_VALUE, 10.0, 2.0, 400.0]
        evaluation.result = EvalResult.failed("export", FailureCategory.EXPORT_FAILED)
        assert objective_values(config, evaluation) is None
        assert INFEASIBLE_VALUE["maximize"] < MISSING_PROXY_VALUE < 0 < INFEASIBLE_VALUE["minimize"]


@pytest.mark.util
class TestFrontAndHypervolume:
    """Non-dominated filter and hypervolume on hand-made points."""

    def test_nondominated(self) -> None:
        points = [[1, 5], [2, 2], [5, 1], [3, 3], [1, 5], [2, 2.5]]
        assert nondominated(points) == [0, 1, 2]
        assert nondominated([]) == []
        assert nondominated([[0, 0]]) == [0]

    def test_hypervolume_2d(self) -> None:
        reference = [10.0, 10.0]
        assert hypervolume_2d([[2, 2]], reference) == 64.0
        # a staircase: (2,8) (4,4) (8,2) plus dominated and out-of-box points
        points = [[2, 8], [4, 4], [8, 2], [5, 5], [12, 1], [1, 12]]
        expected = (10 - 2) * (10 - 8) + (10 - 4) * (8 - 4) + (10 - 8) * (4 - 2)
        assert hypervolume_2d(points, reference) == expected
        assert hypervolume_2d([], reference) == 0.0
        assert hypervolume_2d([[10, 1]], reference) == 0.0
        full = hypervolume_full(points, reference)
        assert full is None or full == pytest.approx(expected)

    def test_hypervolume_full_3d(self) -> None:
        volume = hypervolume_full([[1, 1, 1], [0, 2, 2]], [2.0, 2.0, 2.0])
        assert volume is None or volume == pytest.approx(1.0 + 0.0)  # second point: zero width

    def test_reference_point(self) -> None:
        assert reference_point([[-3.0, 100.0], [-1.0, 200.0]]) == pytest.approx([0.0, 220.0])
        assert reference_point([[-3.0, 100.0], [-MISSING_PROXY_VALUE, 50.0]]) == pytest.approx(
            [0.0, 110.0]
        )
        assert reference_point([]) == [1.0, 1.0]


def _stub_evaluation(genotype: Genotype) -> CandidateEvaluation:
    """Deterministic stand-in for the oracle: cost from the genotype, feasibility by a rule."""
    info = infer_shapes(genotype)
    params = count_params(genotype)
    nodes = len(info.nodes)
    kinds = [n["kind"] for _, n in _walk(genotype["body"])]
    if "relu_float" in kinds:
        result = EvalResult.failed(
            "step_create_dataflow_partition", FailureCategory.DATATYPE_UNSUPPORTED
        )
    elif params > 60_000:
        result = EvalResult.failed("estimate", FailureCategory.RESOURCE_BUDGET_EXCEEDED)
    else:
        result = EvalResult(
            feasible=True,
            resources_total={"LUT": float(params) / 10 + nodes, "BRAM_18K": 0.0},
            max_cycles=1000 // max(1, nodes),
            est_fps=1.0,
        )
    return CandidateEvaluation(
        genotype=genotype,
        hash=genotype_hash(genotype),
        result=result,
        proxies={"scores": {"nwot": float(nodes)}, "errors": {}},
        params=params,
        nodes=nodes,
    )


def _walk(
    node: dict[str, Any], path: tuple[int, ...] = ()
) -> Iterable[tuple[tuple[int, ...], dict[str, Any]]]:
    """Pre-order traversal of a body tree (local copy to keep the stub self-contained)."""
    yield path, node
    for index, child in enumerate(node.get("children", [])):
        yield from _walk(child, (*path, index))


class _StubEvaluator:
    """Replaces ``GenotypeEvaluator`` in the driver: no workers, no FINN, no torch."""

    calls: list[int] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Accept and ignore the evaluator arguments."""

    def __enter__(self) -> _StubEvaluator:
        """No pool to start."""
        return self

    def __exit__(self, *exc: object) -> None:
        """No pool to stop."""

    def evaluate(
        self, genotypes: Iterable[Genotype], seeds: Sequence[int] | None = None
    ) -> list[CandidateEvaluation]:
        """Evaluate with the stub rule and remember the batch sizes."""
        genotypes = list(genotypes)
        assert seeds is None or len(seeds) == len(genotypes)
        self.calls.append(len(genotypes))
        return [_stub_evaluation(g) for g in genotypes]


@pytest.mark.util
class TestSearchLoopWithStub:
    """The whole loop on a real Optuna study with a stub oracle."""

    @pytest.fixture(autouse=True)
    def _stub(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _StubEvaluator.calls = []
        monkeypatch.setattr(driver, "GenotypeEvaluator", _StubEvaluator)

    def _run(self, tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
        config = SearchConfig(
            population_size=6,
            budget=kwargs.pop("budget", 24),
            objectives=["proxy", "LUT", "cycles"],
            mutation_prob=kwargs.pop("mutation_prob", 0.3),
            seed=kwargs.pop("seed", 1),
            **kwargs,
        )
        return run_search(
            SMALL_SPACE,
            _evaluator_config(tmp_path, compute_proxies=True),
            config,
            tmp_path / "run",
            n_workers=3,
        )

    def test_generations_children_and_front(self, tmp_path: Path) -> None:
        summary = self._run(tmp_path)
        assert summary["count"] == 24
        assert [g["trials"] for g in summary["generations"]] == [6, 6, 6, 6]
        assert summary["pareto_front_size"] >= 1
        assert len(summary["hypervolume"]["per_generation_2d"]) == 4
        assert summary["hypervolume"]["objectives_2d"] == ["proxy", "LUT"]
        history = summary["hypervolume"]["per_generation_2d"]
        assert history == sorted(history) and history[-1] > 0
        # batches never straddle a generation: 6 trials = 3 + 3 workers
        assert all(c <= 3 for c in _StubEvaluator.calls)
        study = optuna.load_study(
            study_name=driver.STUDY_NAME,
            storage=optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(str(tmp_path / "run" / "study.journal"))
            ),
        )
        trials = study.get_trials(deepcopy=False)
        assert len(trials) == 24
        for trial in trials:
            assert trial.state == optuna.trial.TrialState.COMPLETE
            assert {"genotype", "hash", "generation", "feasible", "violation"} <= set(
                trial.user_attrs
            )
            assert trial.constraints == {"violation": trial.user_attrs["violation"]}
            assert (trial.user_attrs["violation"] == 0.0) == trial.user_attrs["feasible"]
        # later generations are bred from the parent population, not sampled afresh
        space = SearchSpace(SMALL_SPACE)
        rng = random.Random(1)
        fresh = {genotype_hash(space.sample(rng)) for _ in range(24)}
        later = [t for t in trials if t.user_attrs["generation"] > 0]
        assert later and sum(t.user_attrs["hash"] in fresh for t in later) < len(later)
        sampler = study.sampler
        assert isinstance(sampler, optuna.samplers.BaseSampler)

    def test_propose_breeds_from_parents(self, tmp_path: Path) -> None:
        space = SearchSpace(SMALL_SPACE)
        config = SearchConfig(population_size=4, objectives=["LUT", "cycles"], seed=5)
        sampler = GenotypeNSGAIISampler(space, config)
        study = optuna.create_study(directions=config.directions, sampler=sampler)
        rng = random.Random(0)
        parents = []
        for _ in range(4):
            trial = study.ask()
            genotype = space.sample(rng)
            assert sampler.generation_of(study, trial) == 0
            assert sampler.propose(study, trial)  # a fresh sample in generation 0
            trial.set_user_attr("genotype", genotype)
            trial.set_constraint("violation", 0.0)
            study.tell(trial, [float(count_params(genotype)), 1.0])
            parents.append(genotype)
        trial = study.ask()
        assert sampler.generation_of(study, trial) == 1
        assert len(sampler.get_parent_population(study, 1)) == 4
        children = [sampler.propose(study, trial) for _ in range(10)]
        parent_hashes = {genotype_hash(p) for p in parents}
        assert all(genotype_hash(c) not in parent_hashes for c in children)
        for child in children:
            space.validate(child)

    def test_constrained_dominance(self, tmp_path: Path) -> None:
        study = optuna.create_study(directions=["maximize", "minimize"])

        def make(values: list[float], violation: float) -> optuna.trial.FrozenTrial:
            trial = study.ask()
            trial.set_constraint("violation", violation)
            study.tell(trial, values)
            return study.get_trials(deepcopy=False)[-1]

        feasible_good = make([5.0, 1.0], 0.0)
        feasible_bad = make([4.0, 2.0], 0.0)
        feasible_other = make([6.0, 3.0], 0.0)
        nearly = make([-1e9, 1e9], 0.2)
        far = make([-1e9, 1e9], 1.0)
        d = study.directions
        assert constrained_dominates(feasible_good, feasible_bad, d)
        assert not constrained_dominates(feasible_bad, feasible_good, d)
        assert not constrained_dominates(feasible_good, feasible_other, d)
        assert constrained_dominates(feasible_bad, nearly, d)
        assert constrained_dominates(nearly, far, d)
        assert not constrained_dominates(far, nearly, d)
        assert not constrained_dominates(far, far, d)

    def test_discard_fails_infeasible_trials(self, tmp_path: Path) -> None:
        summary = self._run(tmp_path, feasibility="discard", budget=18)
        assert summary["count"] == 18
        records = driver.read_records(tmp_path / "run" / driver.RECORDS_FILE)
        infeasible = [r for r in records if not r.result.feasible]
        assert infeasible and all(r.meta["values"] is None for r in infeasible)
        study = optuna.load_study(
            study_name=driver.STUDY_NAME,
            storage=optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(str(tmp_path / "run" / "study.journal"))
            ),
        )
        states = {t.number: t.state for t in study.get_trials(deepcopy=False)}
        for record in records:
            expected = (
                optuna.trial.TrialState.COMPLETE
                if record.result.feasible
                else optuna.trial.TrialState.FAIL
            )
            assert states[record.meta["trial"]] == expected
        # generations only count completed trials, so they are longer than the population
        assert sum(g["trials"] for g in summary["generations"]) == 18

    def test_random_baseline(self, tmp_path: Path) -> None:
        summary = self._run(tmp_path, algorithm="random", budget=12)
        assert summary["count"] == 12
        assert [g["trials"] for g in summary["generations"]] == [6, 6]
        assert summary["pareto_front_size"] >= 1

    def test_resume_and_duplicates(self, tmp_path: Path) -> None:
        first = self._run(tmp_path, budget=12, mutation_prob=0.0)
        assert first["count"] == 12
        second = self._run(tmp_path, budget=24, mutation_prob=0.0)
        assert second["count"] == 24
        records = driver.read_records(tmp_path / "run" / driver.RECORDS_FILE)
        assert [r.meta["trial"] for r in records] == list(range(24))
        assert [g["trials"] for g in second["generations"]] == [6, 6, 6, 6]
        # a repeated genotype copies the earlier verdict instead of being evaluated again
        hashes = [r.hash for r in records]
        duplicates = [r for r in records if "duplicate_of" in r.meta]
        assert len(duplicates) == len(hashes) - len(set(hashes))
        for record in duplicates:
            original = next(r for r in records if r.meta["trial"] == record.meta["duplicate_of"])
            assert original.hash == record.hash
            assert original.result.feasible == record.result.feasible
            assert record.meta["trial"] > original.meta["trial"]
        assert second["duplicates"] == len(duplicates)
        assert sum(_StubEvaluator.calls) == len(set(hashes))


@pytest.mark.fpgadataflow
@pytest.mark.slow
class TestSearchWithOracle:
    """Small real runs through the oracle for each feedback variant."""

    @pytest.mark.parametrize(
        ("algorithm", "feasibility"),
        [("nsga2", "constraint"), ("nsga2", "discard"), ("random", "constraint")],
    )
    def test_small_run(self, tmp_path: Path, algorithm: str, feasibility: str) -> None:
        config = SearchConfig(
            algorithm=algorithm,  # type: ignore[arg-type]
            population_size=8,
            budget=24,
            objectives=["LUT", "cycles"],
            feasibility=feasibility,  # type: ignore[arg-type]
            violation="graded",
            seed=0,
        )
        out = tmp_path / "run"
        summary = run_search(SMALL_SPACE, _evaluator_config(tmp_path), config, out, n_workers=2)
        records = driver.read_records(out / driver.RECORDS_FILE)
        assert len(records) == 24 == summary["count"]
        if feasibility == "constraint":
            assert [g["trials"] for g in summary["generations"]] == [8, 8, 8]
        else:
            # only feasible trials fill a generation, so generations are longer
            assert sum(g["trials"] for g in summary["generations"]) == 24
            assert summary["generations"][0]["trials"] >= 8
        assert summary["feasible"] >= 1 and summary["pareto_front_size"] >= 1
        assert (out / driver.SUMMARY_FILE).exists() and (out / driver.CONFIG_FILE).exists()
        for entry in summary["pareto_front"]:
            assert entry["resources"]["LUT"] > 0 and entry["values"][0] == entry["resources"]["LUT"]

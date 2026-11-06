"""Multi-objective search over genotypes: NSGA-II with Optuna doing the bookkeeping.

Optuna's :class:`~optuna.samplers.NSGAIISampler` keeps generations, the parent population
(elite selection with constrained non-dominated sorting and crowding distance) and the study
storage. Its own variation operators work on flat parameter dictionaries, which cannot
represent a tree-shaped genotype, so :class:`GenotypeNSGAIISampler` adds :meth:`propose`: the
driver asks Optuna for a trial and then obtains the genotype from the grammar's own
``sample``/``mutate``/``crossover``. Nothing is ever suggested through ``trial.suggest_*``.

Feasibility is the central signal of the project: an infeasible candidate has no objective
values, and the search variants differ only in how that is fed back (see
:class:`SearchConfig`). The functions at the bottom (dominance, non-dominated filter,
hypervolume) are the small amount of multi-objective arithmetic the summary needs and are
kept independent of Optuna trial objects so they can be tested on hand-made points.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass, field
from optuna.samplers import NSGAIISampler, RandomSampler
from optuna.study import StudyDirection
from optuna.trial import TrialState
from typing import TYPE_CHECKING, Any, Literal

from finn.nas.result import FailureCategory
from finn.nas.space import genotype_hash

if TYPE_CHECKING:
    from collections.abc import Sequence
    from optuna import Study
    from optuna.trial import FrozenTrial, Trial

    from finn.nas.evaluate import CandidateEvaluation
    from finn.nas.result import EvalResult
    from finn.nas.space import SearchSpace
    from finn.nas.space.genotype import Genotype

#: Objective name -> Optuna direction. ``proxy`` is a zero-cost proxy score (higher is
#: better); the others are resource estimates and the initiation interval (``max_cycles``).
OBJECTIVE_DIRECTIONS: dict[str, str] = {
    "proxy": "maximize",
    "LUT": "minimize",
    "BRAM": "minimize",
    "URAM": "minimize",
    "DSP": "minimize",
    "cycles": "minimize",
}
#: Objective name -> key in ``EvalResult.resources_total``.
RESOURCE_KEYS: dict[str, str] = {"LUT": "LUT", "BRAM": "BRAM_18K", "URAM": "URAM", "DSP": "DSP"}

#: Objective values told to Optuna for an infeasible trial under ``feasibility="constraint"``:
#: the worst finite value per direction (Optuna rejects NaN). They never influence the search
#: because constrained domination ranks infeasible trials by violation only.
INFEASIBLE_VALUE: dict[str, float] = {"maximize": -1e9, "minimize": 1e9}
#: Proxy value of a *feasible* candidate whose proxy score is missing or NaN. It stays
#: feasible (the hardware verdict is what matters) but ranks last on the proxy axis.
MISSING_PROXY_VALUE = -1e8
#: Values with this magnitude or more are sentinels, not measurements (see the two above).
SENTINEL_MAGNITUDE = 1e8

#: How far through the oracle pipeline an infeasible candidate got, by failure stage.
#: Later groups are "almost feasible" and get a smaller graded violation. The order follows
#: ``finn.nas.pipeline.run_pipeline``: export, the three front-end steps, the conversion /
#: partitioning / specialization steps, folding, bit-width minimization and estimation.
#: Failures after the estimates (resource budget, throughput target, strict code generation)
#: are matched by category below and form the last group.
STAGE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"export", "config", "worker"}),
    frozenset({"step_qonnx_to_finn", "step_tidy_up", "step_streamline"}),
    frozenset(
        {
            "step_convert_to_hw",
            "step_minimize_bit_width_initial",
            "step_create_dataflow_partition",
            "step_specialize_layers",
        }
    ),
    frozenset({"folding"}),
    frozenset({"step_minimize_bit_width", "estimate"}),
)
#: Categories that are only raised after the complete estimate flow succeeded.
POST_ESTIMATE_CATEGORIES = frozenset(
    {
        FailureCategory.RESOURCE_BUDGET_EXCEEDED,
        FailureCategory.THROUGHPUT_TARGET_MISSED,
        FailureCategory.CODEGEN_ASSERT,
    }
)
#: Number of violation levels: the stage groups plus the post-estimate group.
VIOLATION_LEVELS = len(STAGE_GROUPS) + 1


@dataclass
class SearchConfig:
    """Settings of one search run (JSON round trip via ``to_dict``/``from_dict``).

    ``feasibility`` selects how an infeasible oracle verdict reaches the search:

    - ``"constraint"``: the trial completes with sentinel values and a constraint violation
      > 0. NSGA-II's constrained domination ranks every feasible trial above every infeasible
      one and infeasible ones among themselves by violation, so the population keeps
      "almost feasible" material and the search can learn where the feasible region is.
    - ``"discard"``: the trial is reported to Optuna as failed; it never enters a population
      and the search gets no feedback about it at all.

    ``violation`` is only used with ``"constraint"``: ``"binary"`` gives every infeasible
    candidate the violation 1.0, ``"graded"`` lowers it the further the candidate got through
    the oracle pipeline (:func:`violation_of`).
    """

    algorithm: Literal["nsga2", "random"] = "nsga2"
    population_size: int = 50
    #: Total number of trials (evaluations, failed and duplicate ones included).
    budget: int = 1000
    objectives: list[str] = field(default_factory=lambda: ["proxy", "LUT", "BRAM", "cycles"])
    #: Zero-cost proxy measure behind the ``proxy`` objective.
    proxy: str = "nwot"
    feasibility: Literal["constraint", "discard"] = "constraint"
    violation: Literal["binary", "graded"] = "graded"
    #: Probability of mutating a child after crossover (a child identical to a parent is
    #: always mutated).
    mutation_prob: float = 0.3
    crossover_prob: float = 0.9
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject settings the search loop could not act on."""
        unknown = [o for o in self.objectives if o not in OBJECTIVE_DIRECTIONS]
        if unknown or not self.objectives:
            raise ValueError(
                f"objectives must be a non-empty subset of {sorted(OBJECTIVE_DIRECTIONS)}, "
                f"got {self.objectives}"
            )
        if len(set(self.objectives)) != len(self.objectives):
            raise ValueError(f"duplicate objective in {self.objectives}")
        if self.algorithm not in ("nsga2", "random"):
            raise ValueError(f"unknown algorithm {self.algorithm!r}")
        if self.feasibility not in ("constraint", "discard"):
            raise ValueError(f"unknown feasibility handling {self.feasibility!r}")
        if self.violation not in ("binary", "graded"):
            raise ValueError(f"unknown violation kind {self.violation!r}")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        for name in ("mutation_prob", "crossover_prob"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")

    @property
    def directions(self) -> list[str]:
        """Optuna directions of the objectives, in order."""
        return [OBJECTIVE_DIRECTIONS[o] for o in self.objectives]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchConfig:
        """Reconstruct from :meth:`to_dict` output."""
        return cls(**data)


def violation_level(result: EvalResult) -> int:
    """Return how far an infeasible ``result`` got: 0 (export) .. ``VIOLATION_LEVELS - 1``."""
    if result.failure_category in POST_ESTIMATE_CATEGORIES:
        return VIOLATION_LEVELS - 1
    for level, stages in enumerate(STAGE_GROUPS):
        if result.failure_stage in stages:
            return level
    return 0


def violation_of(config: SearchConfig, result: EvalResult) -> float:
    """Constraint violation of ``result``: 0 if feasible, else 1.0 (binary) or graded.

    The graded violation is ``1 - level / VIOLATION_LEVELS`` and therefore in
    ``(0, 1]``: a candidate that failed at export gets 1.0, one that only exceeded the
    resource budget ``1 / VIOLATION_LEVELS``. Transient failures (timeout, internal error)
    carry no information about the candidate and count as level 0.
    """
    if result.feasible:
        return 0.0
    if config.violation == "binary":
        return 1.0
    return 1.0 - violation_level(result) / VIOLATION_LEVELS


def proxy_score(evaluation: CandidateEvaluation, measure: str) -> float | None:
    """Return the score of proxy ``measure`` or ``None`` if it is missing or not finite."""
    if not evaluation.proxies:
        return None
    score = evaluation.proxies.get("scores", {}).get(measure)
    if score is None or not math.isfinite(score):
        return None
    return float(score)


def objective_values(config: SearchConfig, evaluation: CandidateEvaluation) -> list[float] | None:
    """Return the objective vector of a feasible evaluation, ``None`` if it is infeasible."""
    result = evaluation.result
    if not result.feasible:
        return None
    values: list[float] = []
    for name in config.objectives:
        if name == "proxy":
            score = proxy_score(evaluation, config.proxy)
            values.append(MISSING_PROXY_VALUE if score is None else score)
        elif name == "cycles":
            values.append(float(result.max_cycles if result.max_cycles is not None else 0))
        else:
            values.append(float(result.resources_total.get(RESOURCE_KEYS[name], 0.0)))
    return values


def infeasible_values(config: SearchConfig) -> list[float]:
    """Return the sentinel objective vector told for an infeasible trial."""
    return [INFEASIBLE_VALUE[d] for d in config.directions]


def is_sentinel(value: float) -> bool:
    """Return whether ``value`` is one of the sentinels rather than a measurement."""
    return abs(value) >= SENTINEL_MAGNITUDE


def constrained_dominates(
    trial0: FrozenTrial, trial1: FrozenTrial, directions: Sequence[StudyDirection]
) -> bool:
    """NSGA-II's constrained domination of complete trials (Deb et al., 2002).

    ``trial0`` dominates ``trial1`` if it is feasible and ``trial1`` is not, if both are
    infeasible and ``trial0`` has the smaller violation, or if both are feasible and
    ``trial0`` Pareto-dominates ``trial1``. A trial without constraints counts as feasible,
    which is what Optuna's elite selection does as well.
    """
    violation0 = sum(v for v in trial0.constraints.values() if v > 0)
    violation1 = sum(v for v in trial1.constraints.values() if v > 0)
    if violation0 > 0 or violation1 > 0:
        return violation0 < violation1
    losses0 = to_losses(trial0.values, directions)
    losses1 = to_losses(trial1.values, directions)
    return dominates(losses0, losses1)


def to_losses(values: Sequence[float], directions: Sequence[StudyDirection | str]) -> list[float]:
    """Flip maximized objectives so that every component is to be minimized."""
    return [
        -v if d in ("maximize", StudyDirection.MAXIMIZE) else v
        for v, d in zip(values, directions, strict=True)
    ]


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Return whether loss vector ``a`` Pareto-dominates ``b`` (<= everywhere, < somewhere)."""
    return all(x <= y for x, y in zip(a, b, strict=True)) and any(
        x < y for x, y in zip(a, b, strict=True)
    )


def nondominated(losses: Sequence[Sequence[float]]) -> list[int]:
    """Return the indices of the non-dominated loss vectors (first of equal ones wins)."""
    front: list[int] = []
    for i, candidate in enumerate(losses):
        if any(
            dominates(losses[j], candidate) or (j < i and list(losses[j]) == list(candidate))
            for j in range(len(losses))
            if j != i
        ):
            continue
        front.append(i)
    return front


def hypervolume_2d(losses: Sequence[Sequence[float]], reference: Sequence[float]) -> float:
    """Exact hypervolume of 2-D loss vectors w.r.t. ``reference`` (points beyond it count 0).

    Sort the non-dominated points by the first coordinate and add up the rectangles between
    consecutive points and the reference, which is exact in two dimensions.
    """
    inside = [p for p in losses if p[0] < reference[0] and p[1] < reference[1]]
    front = sorted(tuple(inside[k]) for k in nondominated(inside))
    area = 0.0
    previous_y = reference[1]
    for x, y in front:
        area += (reference[0] - x) * (previous_y - y)
        previous_y = y
    return area


def hypervolume_full(losses: Sequence[Sequence[float]], reference: Sequence[float]) -> float | None:
    """Hypervolume in all dimensions via Optuna's private ``compute_hypervolume``, if usable."""
    inside = [list(p) for p in losses if all(x < r for x, r in zip(p, reference, strict=True))]
    if not inside:
        return 0.0
    try:
        import numpy as np
        from optuna._hypervolume import compute_hypervolume

        return float(
            compute_hypervolume(np.asarray(inside, dtype=float), np.asarray(reference, dtype=float))
        )
    except Exception:  # private API: may change or be removed
        return None


def reference_point(losses: Sequence[Sequence[float]]) -> list[float]:
    """Return the hypervolume reference point: ``max(0, 1.1 * max loss)`` per dimension.

    For a minimized resource this is 110 % of the largest value seen, for the (negated)
    maximized proxy it is 0 as long as every proxy score is positive. Loss vectors that
    contain a sentinel are ignored so that one missing proxy does not stretch the box.
    """
    clean = [p for p in losses if not any(is_sentinel(v) for v in p)]
    if not clean:
        return [1.0] * (len(losses[0]) if losses else 2)
    return [max(0.0, 1.1 * max(p[d] for p in clean)) for d in range(len(clean[0]))]


class GenotypeNSGAIISampler(NSGAIISampler):
    """NSGA-II whose children come from the grammar instead of parameter crossover.

    Generation tracking, the parent population cache and the elite selection are inherited
    from Optuna (``get_trial_generation``, ``get_parent_population``, ``select_parent``).
    Parents are chosen by binary tournament with constrained domination from the parent
    population Optuna selected for the trial's generation; their genotypes are read from
    ``trial.user_attrs["genotype"]`` which the driver stores for every trial.
    """

    def __init__(self, space: SearchSpace, config: SearchConfig) -> None:
        """Create the sampler; ``config.seed`` seeds both Optuna and the grammar's RNG."""
        super().__init__(
            population_size=config.population_size,
            crossover_prob=config.crossover_prob,
            seed=config.seed,
        )
        self.space = space
        self.config = config
        self.rng = random.Random(config.seed)

    def generation_of(self, study: Study, trial: Trial) -> int:
        """Return the generation of ``trial`` (assigned on first call, as Optuna does).

        ``get_trial_generation`` needs the frozen trial (``Trial`` exposes no system
        attributes), which is looked up among the running trials by its number.
        """
        frozen = next(
            t
            for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
            if t.number == trial.number
        )
        return self.get_trial_generation(study, frozen)

    def generation_size_left(self, study: Study, trial: Trial) -> int:
        """How many trials of ``trial``'s generation, ``trial`` included, may still be asked.

        Only completed trials fill a generation, so under ``feasibility="discard"`` a
        generation lasts until ``population_size`` feasible candidates were found.
        """
        generation = self.generation_of(study, trial)
        return max(1, self.population_size - len(self.get_population(study, generation)))

    def propose(self, study: Study, trial: Trial) -> Genotype:
        """Return the genotype for ``trial``: a random sample in generation 0, else a child."""
        generation = self.generation_of(study, trial)
        parents = self.get_parent_population(study, generation)
        if not parents:
            return self.space.sample(self.rng)
        return self.breed(parents, study.directions)

    def tournament(
        self, parents: Sequence[FrozenTrial], directions: Sequence[StudyDirection]
    ) -> FrozenTrial:
        """Binary tournament: the constrained-dominating one of two random parents."""
        a = parents[self.rng.randrange(len(parents))]
        b = parents[self.rng.randrange(len(parents))]
        return a if constrained_dominates(a, b, directions) else b

    def breed(
        self, parents: Sequence[FrozenTrial], directions: Sequence[StudyDirection]
    ) -> Genotype:
        """Crossover (with ``crossover_prob``) of two tournament winners, then mutation.

        Mutation is applied with ``mutation_prob`` and always when the child is identical to
        one of its parents, so that no evaluation is spent on a copy.
        """
        first = self.tournament(parents, directions)
        parent_genotypes = [first.user_attrs["genotype"]]
        if self.rng.random() < self.config.crossover_prob and len(parents) > 1:
            others = [p for p in parents if p.number != first.number] or list(parents)
            second = self.tournament(others, directions)
            parent_genotypes.append(second.user_attrs["genotype"])
            child, _ = self.space.crossover(parent_genotypes[0], parent_genotypes[1], self.rng)
        else:
            child = copy.deepcopy(parent_genotypes[0])
        parent_hashes = {genotype_hash(g) for g in parent_genotypes}
        if self.rng.random() < self.config.mutation_prob or genotype_hash(child) in parent_hashes:
            child = self.space.mutate(child, self.rng)
        return child


class GenotypeRandomSampler(RandomSampler):
    """The random-search baseline with the same ``propose`` interface.

    Generations are a plain bookkeeping notion here (``population_size`` consecutive
    trials), so that the per-generation summaries of the variants are comparable.
    """

    def __init__(self, space: SearchSpace, config: SearchConfig) -> None:
        """Create the sampler; ``config.seed`` seeds the grammar's RNG."""
        super().__init__(seed=config.seed)
        self.space = space
        self.config = config
        self.rng = random.Random(config.seed)

    def generation_of(self, study: Study, trial: Trial) -> int:  # noqa: ARG002
        """Return the generation of ``trial``: its number divided by the population size."""
        return trial.number // self.config.population_size

    def generation_size_left(self, study: Study, trial: Trial) -> int:  # noqa: ARG002
        """Trials left in ``trial``'s generation, ``trial`` included."""
        return self.config.population_size - trial.number % self.config.population_size

    def propose(self, study: Study, trial: Trial) -> Genotype:  # noqa: ARG002
        """Return a fresh random sample of the space."""
        return self.space.sample(self.rng)


GenotypeSampler = GenotypeNSGAIISampler | GenotypeRandomSampler


def make_sampler(space: SearchSpace, config: SearchConfig) -> GenotypeSampler:
    """Return the sampler for ``config.algorithm``."""
    if config.algorithm == "random":
        return GenotypeRandomSampler(space, config)
    return GenotypeNSGAIISampler(space, config)

# NAS experiments: random sampling and the feasibility-aware search

Two drivers live here, both started through the FINN+ CLI so that the environment and the
dependencies are set up:

- `sampling.py` (`finn nas sample`): uniform random samples of the search space through the
  oracle and the zero-cost proxies. Its feasibility rate and failure histogram are the
  baseline, its records the ground truth for the proxy study.
- `search.py` (`finn nas search`): the multi-objective search. NSGA-II over the grammar's
  genotypes with Optuna doing the bookkeeping (`finn.nas.search`), or random search with the
  same output format.

The central question of the search experiments is whether feasibility feedback lets the
search learn FINN's constraints: the variants below differ only in how an infeasible oracle
verdict reaches the algorithm.

## Commands

All runs share the oracle options of `finn nas eval` (`--fpga-part`/`--board`,
`--target-fps`, `--clk-period-ns`, ...) and write into `OUTPUT_DIR`; re-running with a larger
`--count`/`--budget` extends resp. resumes a run. `--workers` evaluation processes run in
parallel, each candidate has `--eval-timeout` seconds.

```bash
DATA=../datasets/cifar10
ORACLE="--board Pynq-Z1 --target-fps 100"

# baseline: 1000 uniform random samples
finn nas sample runs/sampling -n 1000 $ORACLE --data-root $DATA --workers 8

# search variants, 1000 trials each, 50 per generation
finn nas search runs/nsga2_graded $ORACLE --data-root $DATA --workers 8 \
    --budget 1000 --population 50 --algorithm nsga2 --feasibility constraint --violation graded
finn nas search runs/nsga2_binary $ORACLE --data-root $DATA --workers 8 \
    --budget 1000 --population 50 --algorithm nsga2 --feasibility constraint --violation binary
finn nas search runs/nsga2_discard $ORACLE --data-root $DATA --workers 8 \
    --budget 1000 --population 50 --algorithm nsga2 --feasibility discard
finn nas search runs/random $ORACLE --data-root $DATA --workers 8 \
    --budget 1000 --population 50 --algorithm random
```

Further options: `--objectives proxy,LUT,BRAM,cycles` (any subset of `proxy`, `LUT`, `BRAM`,
`URAM`, `DSP`, `cycles`; `proxy` is maximized, the rest minimized), `--proxy nwot` (the
zero-cost measure behind the `proxy` objective; `--proxies` and `--data-root` are required
when it is an objective, `--no-proxies` otherwise skips the proxy computation),
`--mutation-prob 0.3`, `--crossover-prob 0.9`, `--seed`, `--space-config space.json`
(`finn.nas.space.SpaceConfig` as JSON). Use the same seed for all variants: generation 0 of
every NSGA-II variant and the first generation of the random run then draw the same
genotypes, so differences only come from the feedback.

## The variants

| Variant | Algorithm | Infeasible candidates |
|---|---|---|
| `nsga2 / constraint / graded` | NSGA-II | stay in the population as constraint-violating trials; violation decreases the further the candidate got through the oracle pipeline |
| `nsga2 / constraint / binary` | NSGA-II | as above with violation 1.0 for every infeasible candidate (feasible/infeasible is the only signal) |
| `nsga2 / discard` | NSGA-II | told to Optuna as failed trials: they never enter a population, the search gets no feedback about them |
| `random` | random search | evaluated and recorded like the others, no feedback by construction |

NSGA-II (Optuna's `NSGAIISampler`) selects the parent population of a generation by
constrained non-dominated sorting and crowding distance over the previous generation and its
parents: every feasible trial ranks above every infeasible one, infeasible ones rank among
themselves by violation, feasible ones by Pareto dominance. Parents are chosen by binary
tournament with the same constrained dominance; children are made by the grammar's
`crossover` (probability `crossover_prob`, first of the two children) followed by `mutate`
(probability `mutation_prob`, and always when the child equals one of its parents). Optuna's
own parameter-level crossover/mutation are not used because the genotypes are trees.

With `--feasibility discard` only completed (feasible) trials fill a generation, so a
generation lasts until `population_size` feasible candidates were found; the per-generation
rows of such a run are therefore longer than the population size (and the run may end inside
generation 0 if the feasibility rate is low). The budget counts every trial, failed and
duplicate ones included, so a run always stops.

### Violation

`feasible` candidates have violation 0. `binary`: 1.0 for every infeasible candidate.
`graded`: `1 - level / 6` with the level given by the oracle stage that failed (see
`STAGE_GROUPS` in `finn.nas.search`):

| level | failure | violation |
|---|---|---|
| 0 | export, timeout, internal/config error | 1.000 |
| 1 | `step_qonnx_to_finn`, `step_tidy_up`, `step_streamline` | 0.833 |
| 2 | `step_convert_to_hw`, `step_minimize_bit_width_initial`, `step_create_dataflow_partition`, `step_specialize_layers` | 0.667 |
| 3 | folding | 0.500 |
| 4 | `step_minimize_bit_width`, estimate | 0.333 |
| 5 | after the estimates: `RESOURCE_BUDGET_EXCEEDED`, `THROUGHPUT_TARGET_MISSED`, `CODEGEN_ASSERT` | 0.167 |

### Sentinels

Optuna needs a value per objective for every completed trial, so under `constraint` an
infeasible trial is told `proxy = -1e9` and `LUT/BRAM/URAM/DSP/cycles = 1e9`
(`INFEASIBLE_VALUE`). They never influence the ranking (constrained dominance looks at the
violation only) and are excluded from the hypervolumes. A *feasible* candidate whose proxy
score is missing or not finite (e.g. `nwot` is `-inf` for some networks) stays feasible and
gets `proxy = -1e8` (`MISSING_PROXY_VALUE`): it ranks last on the proxy axis but can still be
on the Pareto front through its resources. Values with magnitude >= 1e8 are sentinels, not
measurements.

### Duplicates

A genotype that was already evaluated in the run (same `genotype_hash`, also within one
batch) is not evaluated again: the trial is told the earlier values and feasibility and the
record carries `meta.duplicate_of` (the earlier trial number). Duplicates count towards the
budget and appear in the population like any other trial.

### Resuming

The Optuna study is stored in `study.journal` (`JournalStorage` with a file backend); a run
with a larger `--budget` in the same directory continues the study and the generation it
stopped in. Trials left running by an interrupted run are closed as failed and are not
counted.

## Files and formats

### Records (`samples.jsonl` for sampling, `records.jsonl` for search)

One `finn.nas.evaluate.CandidateEvaluation.to_dict()` per line: `genotype`, `hash`,
`result` (the complete `EvalResult`: `feasible`, `failure_category`, `failure_stage`,
`resources_total` with `LUT`/`BRAM_18K`/`URAM`/`DSP`, `max_cycles`, `est_fps`, `device_fit`,
...), `proxies` (`{"scores": {measure: value}, "errors": {measure: message}}` or null),
`params`, `nodes`, `seconds`, `meta`.

`meta` of a sampling record: `{"index", "seed"}`. `meta` of a search record:

| key | meaning |
|---|---|
| `trial` | Optuna trial number (0-based, in order of proposal) |
| `generation` | generation the trial was proposed in (for `random`: `trial // population_size`) |
| `violation` | constraint violation (0 for feasible) |
| `values` | objective values told to Optuna in `objectives` order (sentinels for infeasible trials under `constraint`, null under `discard`) |
| `feasible` | the oracle verdict |
| `duplicate_of` | only on duplicates: trial number of the evaluation that was copied |

The same information is on the Optuna trials as user attributes (`genotype`, `hash`,
`generation`, `feasible`, `failure_category`, `failure_stage`, `resources`, `est_fps`,
`proxies`, `violation`, `params`, `nodes`) and the violation as the trial constraint
`violation` (`FrozenTrial.constraints`), so the journal can be analysed with Optuna's tools
(`optuna.load_study(study_name="nas_search", storage=JournalStorage(JournalFileBackend(...)))`).

### `config.json`

Sampling: `space`, `evaluator`, `count`, `seed`, `n_workers`. Search: `space`, `evaluator`,
`search` (`finn.nas.search.SearchConfig.to_dict()`), `n_workers`.

### `summary.json`

Sampling: `finn.nas.experiments.sampling.summarize` (count, feasibility rate, failure
histograms, timing, LUT/fps ranges of the feasible samples, proxy failures).

Search (`finn.nas.experiments.search.summarize_search`, computed from the records):

| key | content |
|---|---|
| `search` | the `SearchConfig` |
| `count`, `feasible`, `feasibility_rate`, `distinct_feasible`, `duplicates` | overall |
| `generations` | per generation: `trials`, `feasible`, `feasibility_rate`, `mean_violation`, `new_feasible_hashes` (feasible genotypes first seen in this generation), `duplicates` |
| `pareto_front`, `pareto_front_size` | non-dominated feasible trials (one entry per genotype): `trial`, `hash`, `values`, `resources`, `est_fps`, `proxy`, `params`, `nodes` |
| `hypervolume` | `objectives_2d` (proxy vs. LUT when both are objectives, else the first two), `reference_point` per objective, `per_generation_2d` and `per_generation_full`: hypervolume of the feasible trials up to and including each generation |
| `evaluations` | `summarize` of the non-duplicate records (failure histograms etc.) |

The reference point is `max(0, 1.1 * max loss)` per objective over all feasible trials of
the run (losses are the values with maximized objectives negated): 110 % of the largest
resource/cycle count seen and 0 for the proxy as long as the scores are positive; trials with
sentinel values are ignored when it is computed and contribute nothing to the hypervolume.
The 2-D hypervolume is computed exactly by the staircase formula. The full-dimensional one
uses `optuna._hypervolume.compute_hypervolume`, a private Optuna API: it is `null` when that
import or call fails (and equal to the 2-D value when there are only two objectives).

## Tests

`tests/nas/test_search.py`: configuration, violation ordering, the non-dominated filter and
hypervolumes on hand-made points, the whole loop on a real Optuna study with a stub oracle
(`util`), and small real runs of the three feedback variants through the oracle
(`fpgadataflow`, `slow`).

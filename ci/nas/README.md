# NAS experiments on the CI

The NAS experiments (`src/finn/nas`, see its [README](../../src/finn/nas/README.md) and the
[experiments README](../../src/finn/nas/experiments/README.md)) run on the PC² cluster through
the GitLab mirror as a child pipeline (`ci/.gitlab-nas.yml`) of the regular pipeline. One
pipeline is one study: the random-sampling baseline, the four search variants, the export of
the Pareto fronts for full builds, the proxy ground truth and the report.

## Prerequisites (once, by a maintainer)

Two project CI/CD variables point to directories on the cluster that are visible from the
Slurm nodes (nothing here is tracked with DVC):

| Variable | Content |
|---|---|
| `OTUS_NAS_CIFAR10_DIR` | CIFAR-10 as torchvision lays it out: the directory containing `cifar-10-batches-py/` (what `torchvision.datasets.CIFAR10(root)` downloads; copy it there once, the jobs do not rely on the download working on the nodes) |
| `OTUS_NAS_RESULT_DIR_STORE` | Persistent result directory; every pipeline writes a subdirectory `<date>_<sha>_<parent pipeline id>_<config>` with all records, exports, figures and tables (the GitLab artifacts expire, this copy does not) |

The jobs install the `nas` extra of the wheel (optuna, torchvision) on top of the regular CI
venv (`.setup_nas_2024_2` in `ci/.gitlab-setup.yml`). Ground-truth training needs a GPU node;
the partition and GRES are variables (`NAS_GPU_PARTITION`, `NAS_GPU_GRES`).

## Launching a study

Trigger a pipeline on the branch with `NAS_CFG` set to the name of a search space config in this
directory (`default.json` is the built-in space, `finn.nas.space.SpaceConfig` as JSON; add a
file per study variant). Setting `NAS_CFG` replaces the test suite and the benchmark child
pipelines with the NAS child pipeline, so nothing else needs to be deselected:

```bash
# dry run first (prints the ref, the plan and the variables), then --yes
python3 scripts/gitlab_ci.py trigger --ref feature/nas_oracle --name "nas baseline" \
    --var NAS_CFG=default
python3 scripts/gitlab_ci.py trigger --ref feature/nas_oracle --name "nas smoke" \
    --var NAS_CFG=default --var NAS_SAMPLE_COUNT=50 --var NAS_BUDGET=100 --var NAS_GT_COUNT=5
```

Knobs (all pipeline variables with defaults in `.gitlab-ci.yml`; sizing of the Slurm jobs in
`ci/.gitlab-nas.yml`):

| Variable | Default | Meaning |
|---|---|---|
| `NAS_SAMPLE_COUNT` | 1000 | random samples through the oracle |
| `NAS_BUDGET`, `NAS_POPULATION` | 1000, 50 | trials per search variant, trials per generation |
| `NAS_PROXY`, `NAS_OBJECTIVES` | synflow, proxy,LUT,BRAM,cycles | search objectives |
| `NAS_GT_COUNT`, `NAS_GT_EPOCHS` | 50, 10 | candidates trained for the proxy ground truth, epochs each |
| `NAS_BOARD`, `NAS_CLK_PERIOD_NS`, `NAS_TARGET_FPS` | Pynq-Z1, 10, 1000 | oracle target |
| `NAS_EVAL_TIMEOUT` | 300 | seconds per candidate evaluation |
| `NAS_SEED` | 0 | seed of sampling, search and training (same seed for all variants) |
| `NAS_WORKERS`, `NAS_CPU_CORES`, `NAS_MEM` | 48, 64, 160G | oracle worker processes per sampling/search job and the node resources (about 2 GB per worker) |
| `NAS_EXPORT_SAMPLES` | 30 | feasible random samples exported for full builds |
| `NAS_OUTCOME_VARIANT` | nsga2_graded | search variant whose Pareto front is trained for the outcome comparison |

## Jobs of the child pipeline

| Job | Node | Command | Output (under `nas_artifacts/`) |
|---|---|---|---|
| `NAS Sample` | CPU, `NAS_CPU_CORES` cores | `finn nas sample` | `sampling/` (`samples.jsonl`, `summary.json`, `config.json`) |
| `NAS Search: [<variant>]` | CPU, one job per variant | `finn nas search` | `search_<variant>/` (`records.jsonl`, `study.journal`, `summary.json`) |
| `NAS Export` | CPU, small | `finn nas export` | `export/<variant>/` (Pareto front) and `export/sampling/` (feasible samples): `<hash>.onnx` (+ `.onnx.data`), `samples.jsonl`, `bench_config.yml`, `export_summary.json` |
| `NAS Ground Truth` | GPU | `finn nas groundtruth` on the sampling run | `groundtruth/` |
| `NAS Ground Truth (Front)` | GPU | the same on the exported front of `NAS_OUTCOME_VARIANT` | `groundtruth_front/` |
| `NAS Report` | CPU, small | `finn nas report`, then copies everything to the result store | `report/` (figures, `summary.md`) |

Export and report run even when earlier jobs failed (partial results are still reported). The
per-candidate build directories (`work/`) of a run are not kept. Sampling and search are
independent, so they run in parallel; a 1000-sample baseline and four 1000-trial searches take
well under an hour each on one node.

## Full builds of exported candidates

`NAS Export` writes one `bench_config.yml` per exported set whose `model_path` entries point into
the result store, so the synthesis of a Pareto front (estimate-versus-synthesis fidelity of the
oracle) is a manual bench run of that file with `dut: nas_candidate`
(`src/finn/benchmarking/dut/nas_candidate.yml`: the oracle's flow, StreamlinePlus included,
through to the bitfile). `MANUAL_CFG_PATH` is joined onto `LOCAL_CFG_DIR`, so an absolute path
works; set `PARALLEL_JOBS` to the number of candidates in the file:

```bash
python3 scripts/gitlab_ci.py trigger --ref feature/nas_oracle --name "nas front synthesis" \
    --var MANUAL_CFG_PATH=<OTUS_NAS_RESULT_DIR_STORE>/<result dir>/export/nsga2_graded/bench_config.yml \
    --var PARALLEL_JOBS=<number of candidates> --var TEST_SUITE=none
```

## Adding a search space config

Copy `default.json`, change the field values (`finn.nas.space.SpaceConfig`; unknown keys are
rejected) and trigger with `NAS_CFG=<file name without .json>`. `default.json` is checked
against the code's defaults by `tests/nas/test_ci_config.py`; regenerate it after changing a
default (`json.dumps(SpaceConfig().to_dict(), indent=2)`).

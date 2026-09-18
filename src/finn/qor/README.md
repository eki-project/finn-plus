# Empirical QoR estimation

Learned (empirical) estimation of post-synthesis resources and measured power for FINN
dataflow graphs. Regression models are fitted on the CI microbenchmark database and applied
per node during `step_generate_estimate_reports`, next to the analytical FINN and HLS
estimates.

## Flow

```
CI microbenchmarks  ──collect.py──▶  microbenchmark DB  ──ci/qor/fit_estimators.py──▶  model dir
                                     ($FINN_MICROBENCHMARK_DATABASE)                 ($FINN_QOR_MODEL_DIR)
                                                                                           │
finn build ──step_generate_estimate_reports──▶  estimate_layer_resources_empirical.json  ◀──┘
                                               estimate_power_empirical.json
                                                     │
                            ci/qor/end2end_report.py ┴─▶ figures/tables vs. post-synth results
```

| Environment variable          | Used by                                   | Meaning                                                      |
|-------------------------------|-------------------------------------------|--------------------------------------------------------------|
| `FINN_MICROBENCHMARK_DATABASE` | `fit_estimators.py`, `finn.qor.database`  | Directory with one subfolder per operator holding the JSON files written by `ci/collect/collect.py` |
| `FINN_QOR_MODEL_DIR`          | `fit_estimators.py`, `step_generate_estimate_reports` | Directory with fitted models (`<operator>__<target>.pkl` + `.json` sidecar). If unset or empty, no empirical reports are generated. |

## Package layout

- `features.py` – per-operator feature specification (`OperatorFeatureSpec`), the single
  source of truth for the feature columns used both when fitting on the database and when
  mapping a node of a dataflow graph to features. Pure helpers (`parse_datatype`,
  `datatype_features`, `split_ints`, `broadcast_kind`) derive columns on both sides.
- `database.py` – loading, filtering and deduplication of the microbenchmark database.
- `estimator.py` – `QoREstimator` (sklearn pipeline, fit/predict/save/load), the regressor
  grid, cross-validated model selection and learning-curve evaluation.
- `evaluation.py` – plots and tables (regressor comparison, estimator accuracy, end2end
  resource breakdowns).
- `finn.analysis.fpgadataflow.empirical_qor_estimation` – node-to-feature extractors
  (registered per operator with `@register_node_features`) and the analysis passes
  `empirical_res_estimation` / `empirical_power_estimation` (these are the only parts that
  import FINN/QONNX).
- `finn.benchmarking.dut.microbench_base` – base class of the single-operator microbenchmark
  DUTs (`finn.benchmarking.dut.<operator>`), which generate the training data.

Everything in `finn.qor` only depends on pandas, numpy, scikit-learn and matplotlib, so the
CI scripts run outside the FINN environment (`ci/qor/*.py` add `src/` to `sys.path`).

## Supported operators and targets

| Operator (`dut`) | Node types | Microbenchmark DUT | Notes |
|---|---|---|---|
| `mvau` | `MVAU_hls`, `MVAU_rtl` | `dut/mvau.py` | folding via `sf`/`nf` (-1 = max), weight sparsity options |
| `thresholding` | `Thresholding_hls`, `Thresholding_rtl` | `dut/thresholding.py` | `mem_mode`/`ram_style` hls only, depth triggers rtl only |
| `swg` | `ConvolutionInputGenerator_rtl` | `dut/swg.py` | default and parallel-window implementation, 1D and 2D |
| `vvau` | `VVAU_hls`, `VVAU_rtl` | `dut/vvau.py` | RTL variant requires a Versal part (not measurable on the RFSoC CI boards) |
| `fifo` | `StreamingFIFO_rtl` | `dut/fifo.py` | `impl_style` rtl or vivado (no rtlsim for vivado) |
| `dwc` | `StreamingDataWidthConverter_hls`, `_rtl` | `dut/dwc.py` | rtl requires an integer width ratio |
| `pool` | `Pool_hls` | `dut/pool.py` | MaxPool and QuantAvgPool |
| `fmpadding` | `FMPadding_rtl` | `dut/fmpadding.py` | |
| `eltwise` | `ElementwiseAdd_hls/_rtl`, `ElementwiseMul_hls/_rtl` | `dut/eltwise.py` | constant (rhs) operand only; RTL variant requires Versal + FLOAT32 |

Targets: `metrics.synth.resources.LUT` and `power` (measured PL/PS rail power minus baseline,
in mW; old and new measurement report schemas are combined, see `POWER_COLS`).

Resource types and nodes without a fitted model fall back to the analytical
`node_res_estimation`; nodes without a power model are reported as 0. A node whose
categorical feature values (backend, memory mode, ...) never occurred in the training data
also falls back (`QoREstimator.covers`), so models are never silently extrapolated to e.g.
Versal-only backends or two-stream elementwise operations.

### Microbenchmark conventions

- Every DUT is a `MicrobenchDUT` subclass with a pure `validate(params)` (also used by the
  random sampler), a declarative `param_space()` and `make_model(params, part)`.
- New operators take the folding directly as `pe`/`simd` (the MVAU keeps `sf`/`nf` for
  compatibility with the existing database).
- Parameters that do not apply to the selected backend/variant must be `null` (strings) or
  `0` (numbers), e.g. `ram_style: null` for RTL thresholding. The database deduplicates on all
  `params.*` columns, so inapplicable values would create spurious duplicates.
- The DUT writes `report/dut_info.json` with derived facts about the generated node
  (`dut_node_name`, `dut_op_type`, `dut_backend` plus the operator-specific keys listed in
  `OperatorFeatureSpec.dut_info_keys`). `collect.py` uses `dut_node_name` to pick the DUT's
  hierarchy level from `post_synth_resources.json` and logs the resources of any other
  (unexpectedly inserted) nodes under `synth/resources_extra/`.
- Stream widths are limited to 1024 bits by the instrumentation shell.

## Growing the database: random sampling and the artifact exchange

**Sampling.** Instead of enumerating parameter grids, a config entry with `mode: sample` draws
`num_samples` configurations at random from the DUT's declarative parameter space
(`MicrobenchDUT.param_space()`, see `finn.benchmarking.param_space`), rejects invalid ones via
`validate()` and, with `skip_existing: true`, skips configurations that are already in the
database (`$FINN_MICROBENCHMARK_DATABASE`, key = the run's parameter set, shared with the
database deduplication in `finn.qor.params_key`). See `ci/cfg/microbenchmark_sample_*.yml`;
`space:` overrides individual dimensions, other keys fix parameters:

```yaml
- mode: sample
  dut: mvau
  num_samples: 200
  seed: 1234              # default: $SAMPLE_SEED, else the CI pipeline id
  skip_existing: true     # false | true (runs with status ok) | "all"
  space: {mw: {type: pow2, lo: 16, hi: 1024}, idt: {type: choice, values: [INT2, INT4]}}
  board: RFSoC2x2
```

The expansion is deterministic per seed; on the cluster the first SLURM array task publishes
its expansion in the exchange directory and the others follow it. `sampling_stats.json`
(build artifact) reports attempts and rejection reasons. Locally: `finn bench --sample mvau:20`.
In CI, launch the manual bench pipeline with `MANUAL_CFG_PATH=microbenchmark_sample_<dut>`;
`SAMPLE_COUNT` / `SAMPLE_SEED` override the config.

**Artifact exchange.** Per-run artifacts (reports, `deploy.zip` bitstream packages) are
exchanged between the build (cluster), measurement (board) and collection runners through a
directory on the cluster fileshare instead of GitLab artifacts (`finn.benchmarking.exchange`,
stdlib only). Each job exports `FINN_BENCH_EXCHANGE_DIR` from its runner-specific project
variable (`OTUS_EXCHANGE_DIR`, `BOARD_EXCHANGE_DIR`, `LOCAL_EXCHANGE_DIR`); the build job also
reads the database via `OTUS_BENCHMARK_DIR_STORE`. Layout:

```
<exchange>/CI_<pipeline id>/CREATED
    build_artifacts[_followup]/runs_output/run_<id>/{reports/, deploy.zip, DONE}, TASK_<k>_DONE
    measurement_artifacts[_followup]/runs_output/run_<id>/{reports/, DONE}
```

`DONE` markers signal completed runs; GitLab artifacts only carry configs, summaries and
`run_index.json`. If the variable is unset, everything falls back to `build_artifacts/` in the
working directory as before. The `Exchange Cleanup` job deletes the pipeline's bitstreams
(unless `KEEP_EXCHANGE_DEPLOY=1`) and pipeline directories older than
`EXCHANGE_RETENTION_DAYS` (default 14). Prerequisite on the share: a common group with the
setgid bit on the exchange root, and write access for root on the board (NFS root_squash).
`ci/exchange_cleanup.py --dry-run` previews what would be removed.

## Adding an operator

1. Add an `OperatorFeatureSpec` in `features.py` (feature columns, derived columns via the
   pure helpers, `irrelevant_param_cols` for loop bounds, `dut_info_keys`) and register it in
   `SPECS`.
2. Register the node feature extractor in the analysis module with
   `@register_node_features("<name>")`; it must produce exactly the spec's columns from the
   node attributes (use the same helpers as the spec).
3. Add a `MicrobenchDUT` subclass in `finn.benchmarking.dut.<name>` (`NAME` == spec name ==
   `dut` config parameter == database subfolder) and register it in `MICROBENCH_DUTS`.
4. Add a case to `tests/qor/test_node_features.py`; the parity test asserts that database
   and node features agree for the generated model.
5. Add a smoke configuration to `ci/cfg/microbenchmark_basic.yml`.

## Local usage

```bash
# Fit models (reduced grid for a quick check) and inspect the selection results
FINN_MICROBENCHMARK_DATABASE=/path/to/db FINN_QOR_MODEL_DIR=/path/to/models \
    python ci/qor/fit_estimators.py --quick --output-dir qor_fit_artifacts

# Use the models in a build
FINN_QOR_MODEL_DIR=/path/to/models finn build my_config.yaml my_model.onnx

# Evaluate a benchmark pipeline's artifacts
python ci/qor/end2end_report.py --artifacts-dir /path/with/build_artifacts --output-dir qor_e2e

# Run the (synthesis-free) tests
finn test --batch --variant custom --args "tests/qor"
```

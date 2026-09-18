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

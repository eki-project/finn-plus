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
finn build ──step_generate_estimate_reports──▶ estimate_layer_resources_empirical.json  ◀──┘
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
  mapping a node of a dataflow graph to features.
- `database.py` – loading, filtering and deduplication of the microbenchmark database.
- `estimator.py` – `QoREstimator` (sklearn pipeline, fit/predict/save/load), the regressor
  grid, cross-validated model selection and learning-curve evaluation.
- `evaluation.py` – plots and tables (regressor comparison, estimator accuracy, end2end
  resource breakdowns).
- `finn.analysis.fpgadataflow.empirical_qor_estimation` – node-to-feature mapping and the
  analysis passes `empirical_res_estimation` / `empirical_power_estimation` (these are the only
  parts that import FINN/QONNX).

Everything in `finn.qor` only depends on pandas, numpy, scikit-learn and matplotlib, so the
CI scripts run outside the FINN environment (`ci/qor/*.py` add `src/` to `sys.path`).

## Supported operators and targets

| Operator | Node types             | Targets                                        |
|----------|------------------------|------------------------------------------------|
| `mvau`   | `MVAU_hls`, `MVAU_rtl` | `metrics.synth.resources.LUT`, `power` (measured total power minus baseline, in the unit of the database, i.e. mW) |

Resource types and nodes without a fitted model fall back to the analytical
`node_res_estimation`; nodes without a power model are reported as 0.

## Adding an operator

1. Add an `OperatorFeatureSpec` in `features.py` (feature columns, derived columns, filters
   for known-broken runs) and register it in `SPECS`.
2. Add the matching node mapping in `node_features()` of the analysis module.
3. Make sure the microbenchmark DUT writes the required `dut_info.json` keys and that
   `collect.py` stores its results under the operator's name in the database.

## Local usage

```bash
# Fit models (reduced grid for a quick check) and inspect the selection results
FINN_MICROBENCHMARK_DATABASE=/path/to/db FINN_QOR_MODEL_DIR=/path/to/models \
    python ci/qor/fit_estimators.py --quick --output-dir qor_fit_artifacts

# Use the models in a build
FINN_QOR_MODEL_DIR=/path/to/models finn build my_config.yaml my_model.onnx

# Evaluate a benchmark pipeline's artifacts
python ci/qor/end2end_report.py --artifacts-dir /path/with/build_artifacts --output-dir qor_e2e
```

# Copyright (c) 2025, Paderborn University
# SPDX-License-Identifier: BSD-3-Clause

"""Plots of the regressor selection: must stay warning-free for a single-regressor result
(matplotlib calls float() on single-element pandas Series, a pandas FutureWarning that
flooded the manual QoR jobs' logs)."""

import pytest

import numpy as np
import pandas as pd
import warnings

from finn.qor.estimator import SelectionResult
from finn.qor.evaluation import plot_regressor_comparison

pytestmark = pytest.mark.qor


def _single_regressor_result() -> SelectionResult:
    rng = np.random.default_rng(0)
    y_true = rng.uniform(10, 1000, 40)
    y_pred = y_true * rng.uniform(0.8, 1.2, 40)
    errors = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "abs_error": np.abs(y_true - y_pred),
            "abs_pct_error": np.abs(y_true - y_pred) / y_true * 100,
        }
    )
    scores = pd.DataFrame(
        [
            {
                "regressor": "OnlyRegressor",
                "mean_score": -0.3,
                "std_score": 0.05,
                "min_fold_score": -0.4,
                "max_fold_score": -0.2,
                "worst_mean_score": -0.5,
                "worst_min_fold_score": -0.6,
                "worst_max_fold_score": -0.4,
                "fit_time_s": 0.1,
                "single_predict_ms": 50.0,
            }
        ]
    ).set_index("regressor")
    return SelectionResult(
        best_name="OnlyRegressor",
        best_pipeline=None,
        best_score=-0.3,
        scoring="neg_mean_absolute_percentage_error",
        scores=scores,
        sample_errors={"OnlyRegressor": errors},
        fold_scores={"OnlyRegressor": np.array([-0.4, -0.3, -0.2])},
    )


def test_regressor_comparison_plot_single_regressor_has_no_warnings(tmp_path):
    out = tmp_path / "regressors.png"
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        plot_regressor_comparison(_single_regressor_result(), str(out))
    assert out.stat().st_size > 0

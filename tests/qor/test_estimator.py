# Copyright (c) 2025, Paderborn University
# SPDX-License-Identifier: BSD-3-Clause

"""QoREstimator: scoring policy, zero-inflated targets, persistence (pure sklearn)."""

import pytest

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from finn.qor.database import POWER_TARGET_COL
from finn.qor.estimator import (
    DEFAULT_TARGETS,
    REGRESSOR_GRID_QUICK,
    RESOURCE_TARGETS,
    QoREstimator,
    _mape,
    _zero_hit_rate,
    default_scoring,
    has_enough_signal,
    resource_target,
    select_regressor,
    target_signal,
)
from finn.qor.features import SPECS

pytestmark = pytest.mark.qor


def _synthetic_mvau(n=240, seed=0):
    rng = np.random.default_rng(seed)
    backend = rng.choice(["hls", "rtl"], n)
    mw = rng.choice([16, 32, 64, 128], n)
    mh = rng.choice([16, 32, 64], n)
    simd = np.array([rng.choice([d for d in (1, 2, 4, 8) if w % d == 0]) for w in mw])
    pe = np.array([rng.choice([d for d in (1, 2, 4, 8) if h % d == 0]) for h in mh])
    df = pd.DataFrame(
        {
            "params.backend": backend,
            "params.mem_mode": rng.choice(["internal_embedded", "internal_decoupled"], n),
            "params.ram_style": rng.choice(["auto", "distributed", "block"], n),
            "params.ram_style_thr": ["distributed"] * n,
            "idt_bitwidth": rng.choice([2, 4, 8], n),
            "idt_sign": rng.choice([True, False], n),
            "wdt_bitwidth": rng.choice([2, 4, 8], n),
            "wdt_sign": [True] * n,
            "params.act": rng.choice(["INT4", None], n),
            "params.mw": mw,
            "params.mh": mh,
            "dut_info.simd": simd,
            "dut_info.pe": pe,
            "dut_info.zero_weights": rng.uniform(0, 0.3, n).round(2),
            "commit": ["abc"] * n,
            "pipeline_id": [1] * n,
        }
    )
    lut = 50 + 3.0 * simd * pe * df["wdt_bitwidth"] + rng.normal(0, 5, n)
    df[RESOURCE_TARGETS["LUT"]] = lut.round()
    # DSPs only for the RTL backend: zero-inflated target
    df[RESOURCE_TARGETS["DSP"]] = np.where(backend == "rtl", simd * pe, 0)
    df[RESOURCE_TARGETS["URAM"]] = 0
    df[POWER_TARGET_COL] = 0.5 * lut + rng.normal(0, 2, n)
    return df


def test_targets_and_scoring_policy():
    assert resource_target("BRAM_18K") == "metrics.synth.resources.BRAM_18K_equiv"
    assert resource_target("LUT") == "metrics.synth.resources.LUT"
    assert DEFAULT_TARGETS[-1] == POWER_TARGET_COL and len(DEFAULT_TARGETS) == 5
    assert default_scoring(POWER_TARGET_COL) == "r2"
    assert default_scoring(RESOURCE_TARGETS["LUT"]) == "neg_mean_absolute_percentage_error"
    for res in ("DSP", "BRAM_18K", "URAM"):
        assert default_scoring(RESOURCE_TARGETS[res]) == "neg_mean_absolute_error"


def test_mape_and_zero_hit_rate_ignore_or_count_zeros():
    y = np.array([0, 0, 10, 20])
    p = np.array([0.3, 2.0, 11, 18])
    assert _mape(y, p) == pytest.approx((10 + 10) / 2)
    assert np.isnan(_mape(np.zeros(3), np.ones(3)))
    assert _zero_hit_rate(y, p) == 0.5
    assert np.isnan(_zero_hit_rate(np.ones(2), np.ones(2)))


def test_target_signal_gating():
    assert target_signal(np.array([0, 0, 1, 2, np.nan])) == {
        "n": 4,
        "n_nonzero": 2,
        "n_distinct": 3,
        "zero_fraction": 0.5,
    }
    assert not has_enough_signal(np.zeros(100))
    assert not has_enough_signal(np.array([0] * 50 + [4] * 50))  # only 2 distinct values
    assert has_enough_signal(np.array([0] * 50 + [1, 2, 3, 4] * 10))


def test_fit_predict_save_load_and_covers(tmp_path):
    df = _synthetic_mvau()
    est = QoREstimator("mvau", RESOURCE_TARGETS["LUT"]).fit(df, KNeighborsRegressor(3))
    row = df.iloc[[0]]
    pred = est.predict(row)
    assert isinstance(pred, int) and pred > 0
    assert est.metadata["n_samples"] == len(df)
    assert est.metadata["target_signal"]["n_nonzero"] == len(df)
    assert set(est.metadata["categories"]) == set(est.metadata["categorical_cols"])
    assert "None" in est.metadata["categories"]["params.act"]
    assert est.covers(row)
    unseen = row.copy()
    unseen["params.ram_style"] = "ultra"
    assert not est.covers(unseen)

    est.save(str(tmp_path))
    loaded = QoREstimator.load(str(tmp_path), "mvau", RESOURCE_TARGETS["LUT"])
    assert loaded.predict(row) == pred
    assert QoREstimator.available_models(str(tmp_path)) == [("mvau", RESOURCE_TARGETS["LUT"])]
    assert loaded.covers(row) and not loaded.covers(unseen)

    # resource predictions are clipped at zero
    class Negative:
        def predict(self, X):
            return np.array([-3.4])

    est.pipeline = Negative()
    assert est.predict(row) == 0
    power = QoREstimator("mvau", POWER_TARGET_COL, pipeline=Negative())
    assert power.predict(row) == pytest.approx(-3.4)


def test_rows_without_target_are_ignored():
    df = _synthetic_mvau(n=60)
    df.loc[:9, POWER_TARGET_COL] = np.nan
    est = QoREstimator("mvau", POWER_TARGET_COL)
    assert len(est.rows_with_target(df)) == 50
    with pytest.raises(KeyError):
        est.rows_with_target(df.drop(columns=[POWER_TARGET_COL]))


def test_select_regressor_on_zero_inflated_target():
    df = _synthetic_mvau()
    est = QoREstimator("mvau", RESOURCE_TARGETS["DSP"])
    result = select_regressor(est, df, cv=3, n_jobs=1, grid=REGRESSOR_GRID_QUICK)
    assert result.scoring == "neg_mean_absolute_error"
    assert est.metadata["cv_scoring"] == result.scoring
    scores = result.scores.dropna(subset=["mean_score"])
    assert {"fold_mae", "zero_hit_rate", "sample_mae", "fold_mape"} <= set(scores.columns)
    assert scores["zero_hit_rate"].between(0, 1).all()
    assert result.best_name in ("KNeighborsRegressor", "RandomForestRegressor")
    # spec columns are the estimator's features
    assert est.feature_cols == SPECS["mvau"].feature_cols

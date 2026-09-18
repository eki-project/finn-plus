# Copyright (c) 2025, Paderborn University
# SPDX-License-Identifier: BSD-3-Clause

"""Loading of the microbenchmark database: derived resource/power targets (pure pandas)."""

import pytest

import json
import numpy as np
import pandas as pd

from finn.qor.database import (
    BRAM_TARGET_COL,
    POWER_TARGET_COL,
    derive_power_target,
    derive_resource_targets,
    load_microbenchmark_database,
)

pytestmark = pytest.mark.qor

_PARAMS = {
    "dut": "mvau",
    "backend": "hls",
    "idt": "INT4",
    "wdt": "INT4",
    "act": "INT4",
    "nhw": [1],
    "mw": 64,
    "mh": 64,
    "sf": 8,
    "nf": 8,
    "m": 1,
    "sparsity_type": "none",
    "sparsity_amount": 0,
    "mem_mode": "internal_embedded",
    "ram_style": "distributed",
    "ram_style_thr": "distributed",
}


def _run(mw, synth, hls=None, power=None, status="ok"):
    metrics = {"status": status, "synth": {"resources": synth}}
    if hls is not None:
        metrics["hls_estimate"] = {"resources": hls}
    if power is not None:
        metrics["measurement"] = {"power": power}
    return {
        "params": {**_PARAMS, "mw": mw},
        "dut_info": {"simd": 8, "pe": 8, "zero_weights": 0.05},
        "metrics": metrics,
    }


@pytest.fixture
def database(tmp_path):
    db = tmp_path / "db"
    (db / "mvau").mkdir(parents=True)
    runs = [
        _run(
            16,
            {"LUT": 100, "DSP": 0, "BRAM_18K": 2, "BRAM_36K": 1, "URAM": 0},
            hls={"LUT": 120, "DSP48E": 3},
            power={"avg_0V85_power": 1500.0},
        ),
        _run(
            32,
            {"LUT": 200, "DSP": 4, "BRAM_18K": 0, "BRAM_36K": 0, "URAM": 0},
            hls={"LUT": 220, "DSP": 5},
            power={"power_pl_ps_load": 1.7},
        ),
        _run(64, {"LUT": 300, "DSP": 0, "URAM": 2}, hls={"LUT": 330}),
    ]
    (db / "mvau" / "a.json").write_text(
        json.dumps(
            {
                "dut": "mvau",
                "date": "2025-01-01",
                "commit": "abc",
                "pipeline_id": "1",
                "pipeline_name": "p",
                "runs": runs,
            }
        )
    )
    return db


def test_derive_resource_targets_bram_and_dsp():
    df = pd.DataFrame(
        {
            "metrics.synth.resources.BRAM_18K": [2, 0, np.nan, np.nan],
            "metrics.synth.resources.BRAM_36K": [1, 0, 3, np.nan],
            "metrics.hls_estimate.resources.DSP48E": [3, np.nan, np.nan, np.nan],
            "metrics.hls_estimate.resources.DSP58E": [np.nan, 7, np.nan, np.nan],
        }
    )
    out = derive_resource_targets(df)
    assert out[BRAM_TARGET_COL].tolist()[:3] == [4, 0, 6]
    assert np.isnan(out[BRAM_TARGET_COL].iloc[3])
    dsp = out["metrics.hls_estimate.resources.DSP"]
    assert dsp.tolist()[:2] == [3, 7] and np.isnan(dsp.iloc[2])
    # frames without any of the columns are returned unchanged
    plain = pd.DataFrame({"x": [1]})
    assert derive_resource_targets(plain).columns.tolist() == ["x"]


def test_derive_power_target_combines_schemas(caplog):
    df = pd.DataFrame(
        {
            "metrics.measurement.power.avg_0V85_power": [1500.0, np.nan, np.nan],
            "metrics.measurement.power.power_pl_ps_load": [np.nan, 1.7, np.nan],
        }
    )
    out = derive_power_target(df)
    # baseline = min - 0.01 mW, so the smallest run keeps 0.01 mW
    assert out[POWER_TARGET_COL].round(2).tolist()[:2] == [0.01, 200.01]
    assert np.isnan(out[POWER_TARGET_COL].iloc[2])
    assert derive_power_target(pd.DataFrame({"x": [1]})).columns.tolist() == ["x"]


def test_load_database_derives_targets(database):
    df, stats = load_microbenchmark_database("mvau", str(database))
    assert stats.final == 3
    df = df.sort_values("params.mw").reset_index(drop=True)
    # the third run never logged BRAM keys: unknown, not zero
    assert df[BRAM_TARGET_COL].tolist()[:2] == [4, 0] and np.isnan(df[BRAM_TARGET_COL].iloc[2])
    assert df["metrics.hls_estimate.resources.DSP"].tolist()[:2] == [3, 5]
    assert df["metrics.synth.resources.URAM"].tolist() == [0, 0, 2]
    assert df[POWER_TARGET_COL].notna().tolist() == [True, True, False]
    # the operator's feature columns are all present
    from finn.qor.features import SPECS

    assert all(c in df.columns for c in SPECS["mvau"].feature_cols)


def test_load_database_unknown_operator(database):
    with pytest.raises(FileNotFoundError):
        load_microbenchmark_database("fifo", str(database))

# Copyright (c) 2025, Paderborn University
# SPDX-License-Identifier: BSD-3-Clause

"""The random sampler and the database loader must agree on the identity of a run."""

import pytest

import json

from finn.qor.database import load_microbenchmark_database
from finn.qor.params_key import (
    IRRELEVANT_PARAMS,
    existing_param_keys,
    iter_database_runs,
    make_hashable,
    params_key,
)

pytestmark = pytest.mark.qor


def _run(params, status="ok", lut=100):
    return {
        "params": params,
        "dut_info": {"simd": 8, "pe": 8, "zero_weights": 0.05},
        "metrics": {"status": status, "synth": {"resources": {"LUT": lut}}},
    }


_BASE = {
    "dut": "mvau",
    "backend": "hls",
    "idt": "INT4",
    "wdt": "INT4",
    "act": "INT4",
    "nhw": [1, 32, 32],
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
    "generate_outputs": ["bitfile"],
    "store_results_in_dvc_data": True,
}


@pytest.fixture
def database(tmp_path):
    db = tmp_path / "db"
    (db / "mvau").mkdir(parents=True)
    old = {
        "dut": "mvau",
        "date": "2025-01-01",
        "commit": "aaa",
        "pipeline_id": "1",
        "pipeline_name": "x",
        "runs": [
            _run(_BASE, lut=100),
            _run({**_BASE, "mw": 128}, status="failed"),
            _run({**_BASE, "mw": 256}, status="skipped"),
        ],
    }
    new = {
        "dut": "mvau",
        "date": "2025-02-01",
        "commit": "bbb",
        "pipeline_id": "2",
        "pipeline_name": "y",
        # same config as in the old file, but with a different nhw (irrelevant) and
        # different infrastructure switches: must deduplicate to the newer run
        "runs": [_run({**_BASE, "nhw": [1], "generate_outputs": ["stitched_ip"]}, lut=110)],
    }
    (db / "mvau" / "a.json").write_text(json.dumps(old))
    (db / "mvau" / "b.json").write_text(json.dumps(new))
    return db


def test_params_key_basics():
    assert make_hashable({"a": [1, [2, 3]]}) == (("a", (1, (2, 3))),)
    key = params_key({"b": [1, 2], "a": 1, "generate_outputs": ["x"]})
    assert key == (("a", 1), ("b", (1, 2)))
    # projection onto a key set: missing keys are None, order is canonical
    assert params_key({"a": 1}, keys=["b", "a"]) == (("a", 1), ("b", None))
    assert params_key({"a": 1, "n": 5}, ignore=("n",)) == (("a", 1),)
    assert all(name in IRRELEVANT_PARAMS for name in ("generate_outputs",))


def test_iter_and_existing_keys(database):
    runs = list(iter_database_runs(str(database), "mvau"))
    assert len(runs) == 4
    assert runs[0][0]["pipeline_id"] == "1"
    assert list(iter_database_runs(str(database), "unknown")) == []

    names = [k for k in _BASE if k != "nhw"]
    keys, stats = existing_param_keys(str(database), "mvau", names)
    assert stats == {"files": 2, "runs": 4, "keys": 1}
    assert params_key(_BASE, names) in keys
    assert params_key({**_BASE, "mw": 128}, names) not in keys
    keys_all, _ = existing_param_keys(str(database), "mvau", names, statuses=None)
    assert len(keys_all) == 3


def test_loader_dedup_matches_sampler_key(database):
    df, stats = load_microbenchmark_database("mvau", str(database))
    # 4 runs, minus failed and skipped, minus the duplicate of the base config
    assert stats.total == 4 and stats.failed == 1 and stats.skipped == 1
    assert stats.duplicates == 1 and stats.final == 1
    # the newer run wins
    assert df["metrics.synth.resources.LUT"].iloc[0] == 110
    assert "params.nhw" not in df.columns and "params.generate_outputs" not in df.columns
    # the loader's surviving parameter columns are exactly the sampler's key names
    names = sorted(c[len("params.") :] for c in df.columns if c.startswith("params."))
    keys, _ = existing_param_keys(str(database), "mvau", names)
    row = {n: df[f"params.{n}"].iloc[0] for n in names}
    assert params_key(row, names) in keys

# Copyright (c) 2025, Paderborn University
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the random sampling of microbenchmark configurations (pure python: fake DUTs)."""

import pytest

import json
import random
import threading

from finn.benchmarking.param_space import Choice, Conditional, Divisor, Fixed, Pow2Range
from finn.benchmarking.sampling import (
    build_space,
    dimension_from_config,
    expand_config,
    pop_sampling_info,
    publish_expanded,
    sample_entry,
)
from finn.qor.params_key import params_key

pytestmark = pytest.mark.qor


class FakeDUT:
    NAME = "fake"

    @staticmethod
    def validate(params):
        if params["mw"] % params["simd"] != 0:
            return "simd must divide mw"
        if params["backend"] == "rtl" and params["act"] is not None:
            return "rtl has no activation"
        return None

    @classmethod
    def param_space(cls):
        return {
            "backend": Choice(["hls", "rtl"]),
            "act": Conditional("backend", {"rtl": Fixed(None)}, Choice([None, "INT4"])),
            "mw": Pow2Range(16, 256),
            "simd": Divisor("mw"),
            "nhw": Fixed([1, 8, 8]),
        }

    @classmethod
    def sample_defaults(cls):
        return {"dut": cls.NAME, "store_results_in_dvc_data": True, "generate_outputs": ["x"]}


REGISTRY = {"fake": FakeDUT}


def _entry(**kwargs):
    return {"mode": "sample", "dut": "fake", "num_samples": 10, "seed": 1, **kwargs}


def test_dimension_from_config():
    assert dimension_from_config([1, 2]) == Choice([1, 2])
    assert dimension_from_config(5) == Fixed(5)
    assert dimension_from_config({"type": "pow2", "lo": 4, "hi": 64}).values() == [4, 8, 16, 32, 64]
    assert dimension_from_config({"type": "choice", "values": ["a"], "weights": [1]}).weights == [1]
    with pytest.raises(ValueError):
        dimension_from_config({"type": "nope"})


def test_build_space_overrides_and_fixed():
    space, fixed = build_space(
        FakeDUT.param_space(),
        _entry(space={"mw": {"type": "pow2", "lo": 32, "hi": 64}}, board="X", ram=["a", "b"]),
    )
    assert space["mw"].values() == [32, 64]
    assert fixed == {"board": "X"}
    assert space["ram"] == Choice(["a", "b"])
    assert "num_samples" not in space and "mode" not in space


def test_sample_entry_reproducible_and_valid():
    space, fixed = build_space(FakeDUT.param_space(), _entry())
    a, stats_a = sample_entry(_entry(), space, FakeDUT.validate, rng=random.Random(1), fixed=fixed)
    b, _ = sample_entry(_entry(), space, FakeDUT.validate, rng=random.Random(1), fixed=fixed)
    assert a == b and len(a) == 10
    assert all(FakeDUT.validate(p) is None for p in a)
    assert stats_a.produced == 10 and stats_a.attempts >= 10
    assert (
        stats_a.rejected_duplicate
        + stats_a.rejected_existing
        + sum(stats_a.rejected_invalid.values())
        == stats_a.attempts - 10
    )
    keys = {params_key(p) for p in a}
    assert len(keys) == 10


def test_sample_entry_skips_existing_and_stops_at_max_attempts():
    space, fixed = build_space(FakeDUT.param_space(), _entry())
    first, _ = sample_entry(_entry(), space, FakeDUT.validate, rng=random.Random(1), fixed=fixed)
    names = sorted(set(space) | set(fixed))
    existing = {params_key(p, names) for p in first}
    second, stats = sample_entry(
        _entry(), space, FakeDUT.validate, existing, random.Random(1), fixed
    )
    assert stats.rejected_existing >= 10
    assert not existing & {params_key(p, names) for p in second}
    # tiny attempt budget: fewer samples, no exception
    few, stats = sample_entry(
        _entry(max_attempts=3), space, FakeDUT.validate, rng=random.Random(1), fixed=fixed
    )
    assert stats.attempts == 3 and len(few) <= 3


def _write_db(tmp_path, runs):
    db = tmp_path / "db"
    (db / "fake").mkdir(parents=True)
    (db / "fake" / "a.json").write_text(
        json.dumps(
            {
                "dut": "fake",
                "date": "2025-01-01",
                "commit": "c",
                "pipeline_id": "1",
                "pipeline_name": "p",
                "runs": runs,
            }
        )
    )
    return db


def test_expand_config_mixed_entries_and_database(tmp_path, monkeypatch):
    monkeypatch.delenv("SAMPLE_COUNT", raising=False)
    monkeypatch.delenv("SAMPLE_SEED", raising=False)
    monkeypatch.delenv("CI_PIPELINE_ID", raising=False)
    cartesian = {"dut": ["fake"], "a": [1, 2], "b": ["x"]}
    expanded, stats = expand_config([cartesian, _entry()], REGISTRY, None)
    assert expanded[:2] == [{"dut": "fake", "a": 1, "b": "x"}, {"dut": "fake", "a": 2, "b": "x"}]
    assert len(expanded) == 12 and len(stats) == 1
    sampled = expanded[2:]
    # defaults from the DUT and bookkeeping are attached
    assert all(p["store_results_in_dvc_data"] is True and p["dut"] == "fake" for p in sampled)
    info = pop_sampling_info(sampled[0])
    assert info == {"entry": 1, "seed": 1} and "_sampling" not in sampled[0]

    # seed the database with the first two sampled runs (plus a failed one) and re-expand:
    # the ok runs are skipped, the failed one is retried unless skip_existing == "all"
    for p in sampled:
        pop_sampling_info(p)
    runs = [
        {"params": sampled[0], "metrics": {"status": "ok"}},
        {"params": sampled[1], "metrics": {"status": "ok"}},
        {"params": sampled[2], "metrics": {"status": "failed"}},
    ]
    db = _write_db(tmp_path, runs)
    names = sorted(sampled[0])
    again, stats = expand_config([_entry()], REGISTRY, str(db))
    again_keys = {params_key(p, names) for p in (pop_sampling_info(p) or p for p in again)}
    for p in again:
        pop_sampling_info(p)
    again_keys = {params_key(p, names) for p in again}
    assert params_key(sampled[0], names) not in again_keys
    assert params_key(sampled[1], names) not in again_keys
    assert params_key(sampled[2], names) in again_keys
    assert stats[0].rejected_existing >= 2 and stats[0].existing_keys == 2
    assert stats[0].database == {"files": 1, "runs": 3, "keys": 2}

    strict, stats = expand_config([_entry(skip_existing="all")], REGISTRY, str(db))
    for p in strict:
        pop_sampling_info(p)
    assert params_key(sampled[2], names) not in {params_key(p, names) for p in strict}
    assert stats[0].existing_keys == 3


def test_expand_config_env_overrides(monkeypatch):
    monkeypatch.setenv("SAMPLE_COUNT", "3")
    monkeypatch.setenv("SAMPLE_SEED", "77")
    monkeypatch.delenv("CI_PIPELINE_ID", raising=False)
    entry = {"mode": "sample", "dut": "fake", "num_samples": 10}
    expanded, stats = expand_config([entry, dict(entry)], REGISTRY, None)
    assert len(expanded) == 6
    assert [s.seed for s in stats] == [77, 78]
    assert pop_sampling_info(expanded[0]) == {"entry": 0, "seed": 77}
    with pytest.raises(ValueError):
        expand_config([{"mode": "sample", "dut": "unknown", "num_samples": 1}], REGISTRY, None)


def test_publish_expanded_first_writer_wins(tmp_path):
    results = {}

    def worker(name, config):
        results[name] = publish_expanded(tmp_path, config, wait_s=5)

    first = [{"dut": "fake", "a": 1}]
    second = [{"dut": "fake", "a": 2}]
    assert publish_expanded(tmp_path, first) == first
    t = threading.Thread(target=worker, args=("late", second))
    t.start()
    t.join()
    # the late caller gets the published expansion, not its own
    assert results["late"] == first
    assert json.loads((tmp_path / "bench_config_exp.json").read_text()) == first

# Copyright (c) 2025, Paderborn University
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the fileshare-based artifact exchange helper (stdlib only)."""

import pytest

import os
import time

from finn.benchmarking import exchange

pytestmark = pytest.mark.qor


@pytest.fixture
def no_exchange(monkeypatch):
    monkeypatch.delenv(exchange.EXCHANGE_ENV_VAR, raising=False)
    monkeypatch.delenv(exchange.PIPELINE_ENV_VAR, raising=False)
    monkeypatch.delenv("CI_PIPELINE_ID", raising=False)


@pytest.fixture
def with_exchange(monkeypatch, tmp_path):
    monkeypatch.setenv(exchange.EXCHANGE_ENV_VAR, str(tmp_path / "share"))
    monkeypatch.delenv(exchange.PIPELINE_ENV_VAR, raising=False)
    monkeypatch.setenv("CI_PIPELINE_ID", "4711")
    return tmp_path / "share"


def test_fallback_layout_without_exchange(no_exchange, tmp_path):
    assert exchange.exchange_root() is None
    assert exchange.pipeline_exchange_dir(create=True) is None
    assert (
        exchange.artifacts_dir("build")
        == tmp_path.joinpath("build_artifacts").relative_to(tmp_path)
        or exchange.artifacts_dir("build").name == "build_artifacts"
    )
    assert exchange.artifacts_dir("measurement", followup=True, base=tmp_path) == (
        tmp_path / "measurement_artifacts_followup"
    )
    assert (
        exchange.run_dir("build", 3, base=tmp_path)
        == tmp_path / "build_artifacts" / "runs_output" / "run_3"
    )
    assert exchange.list_run_ids("build", base=tmp_path) == []
    assert "not set" in exchange.describe()


def test_empty_variable_counts_as_unset(monkeypatch, tmp_path):
    monkeypatch.setenv(exchange.EXCHANGE_ENV_VAR, "   ")
    assert exchange.exchange_root() is None
    assert exchange.artifacts_dir("build", base=tmp_path) == tmp_path / "build_artifacts"


def test_pipeline_dir_and_markers(with_exchange):
    assert exchange.pipeline_dir_name() == "CI_4711"
    pipeline_dir = exchange.pipeline_exchange_dir(create=True)
    assert pipeline_dir == with_exchange / "CI_4711"
    assert (pipeline_dir / exchange.CREATED_MARKER).is_file()
    assert exchange.artifacts_dir("build", base="/ignored") == pipeline_dir / "build_artifacts"
    assert exchange.artifacts_dir("measurement", followup=True) == (
        pipeline_dir / "measurement_artifacts_followup"
    )
    assert str(pipeline_dir) in exchange.describe()


def test_pipeline_override(with_exchange, monkeypatch):
    monkeypatch.setenv(exchange.PIPELINE_ENV_VAR, "CI_old")
    assert exchange.pipeline_exchange_dir() == with_exchange / "CI_old"


def test_unknown_kind_rejected(with_exchange):
    with pytest.raises(ValueError):
        exchange.artifacts_dir("foo")


def test_done_markers_and_listing(with_exchange):
    for run_id in (0, 1, 2):
        exchange.ensure_dir(exchange.run_dir("build", run_id))
    (exchange.artifacts_dir("build") / "runs_output" / "not_a_run").mkdir()
    # no markers at all: old trees are consumed as they are
    assert exchange.list_run_ids("build") == [0, 1, 2]
    assert exchange.list_run_ids("build", require_done=True) == []
    marker = exchange.mark_done(exchange.run_dir("build", 1), "ok", task_id=7)
    assert marker.is_file() and not marker.with_suffix(".tmp").exists()
    assert exchange.read_done(exchange.run_dir("build", 1)) == {
        "status": "ok",
        "task_id": 7,
        "time": exchange.read_done(exchange.run_dir("build", 1))["time"],
    }
    # as soon as one marker exists, only marked runs are listed by default
    assert exchange.list_run_ids("build") == [1]
    assert exchange.list_run_ids("build", require_done=False) == [0, 1, 2]
    assert exchange.is_done(exchange.run_dir("build", 1))
    assert not exchange.is_done(exchange.run_dir("build", 0))
    assert exchange.read_done(exchange.run_dir("build", 0)) is None


def test_check_writable(with_exchange, tmp_path):
    target = exchange.ensure_dir(with_exchange / "x")
    exchange.check_writable(target)
    with pytest.raises(RuntimeError):
        exchange.check_writable(tmp_path / "does" / "not" / "exist")


def test_cleanup_pipeline_removes_only_deploy(with_exchange):
    pipeline_dir = exchange.pipeline_exchange_dir(create=True)
    run = exchange.ensure_dir(exchange.run_dir("build", 0))
    (run / "deploy.zip").write_bytes(b"x" * 100)
    (run / "reports").mkdir()
    (run / "reports" / "a.json").write_text("{}")
    run_f = exchange.ensure_dir(exchange.run_dir("build", 0, followup=True))
    (run_f / "deploy.zip").write_bytes(b"y" * 50)
    assert exchange.cleanup_pipeline(pipeline_dir, dry_run=True) == {"files": 2, "bytes": 150}
    assert (run / "deploy.zip").exists()
    assert exchange.cleanup_pipeline(pipeline_dir) == {"files": 2, "bytes": 150}
    assert not (run / "deploy.zip").exists()
    assert not (run_f / "deploy.zip").exists()
    assert (run / "reports" / "a.json").exists()


def test_cleanup_stale_only_old_marked_dirs(with_exchange):
    current = exchange.pipeline_exchange_dir(create=True)
    old = exchange.ensure_dir(with_exchange / "CI_1")
    (old / exchange.CREATED_MARKER).write_text("old\n")
    stale_time = time.time() - 30 * 86400
    os.utime(old / exchange.CREATED_MARKER, (stale_time, stale_time))
    old_kept = exchange.ensure_dir(with_exchange / "CI_2")
    (old_kept / exchange.CREATED_MARKER).write_text("old\n")
    os.utime(old_kept / exchange.CREATED_MARKER, (stale_time, stale_time))
    no_marker = exchange.ensure_dir(with_exchange / "CI_3")
    other = exchange.ensure_dir(with_exchange / "misc")
    # the current pipeline is old as well but must never be removed
    os.utime(current / exchange.CREATED_MARKER, (stale_time, stale_time))

    removed = exchange.cleanup_stale(with_exchange, 14, dry_run=True)
    assert removed == [old, old_kept]
    assert old.exists()
    removed = exchange.cleanup_stale(with_exchange, 14, keep=("CI_2",))
    assert removed == [old]
    assert not old.exists()
    assert old_kept.exists() and no_marker.exists() and other.exists() and current.exists()
    assert exchange.cleanup_stale(with_exchange / "missing", 14) == []

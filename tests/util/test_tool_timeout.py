############################################################################
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import pytest

import os
import subprocess
import time

from finn.benchmarking.dut.microbench_base import TOOL_TIMEOUT_S
from finn.benchmarking.dut.mvau import bench_mvau
from finn.util.basic import TOOL_TIMEOUT_ENV_VAR, default_tool_timeout, launch_process_helper
from finn.util.exception import FINNUserError

pytestmark = pytest.mark.util

# a sleep duration no other process on the machine is likely to use, so pgrep can find it
SLEEP_MARKER = "37.31"


def _sleeper_alive():
    """Whether a child of the time-limited call (the marked sleep) is still running."""
    result = subprocess.run(["pgrep", "-f", f"sleep {SLEEP_MARKER}"], capture_output=True)
    return result.returncode == 0


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, None),
        ("", None),
        ("  ", None),
        ("0", None),
        ("-5", None),
        ("abc", None),
        ("90", 90.0),
        ("1.5", 1.5),
    ],
)
def test_default_tool_timeout_parsing(monkeypatch, raw, expected):
    """FINN_TOOL_TIMEOUT_S yields a positive number of seconds or no limit."""
    if raw is None:
        monkeypatch.delenv(TOOL_TIMEOUT_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(TOOL_TIMEOUT_ENV_VAR, raw)
    assert default_tool_timeout() == expected


def test_env_timeout_terminates_process_group(monkeypatch):
    """A call over the env limit raises FINNUserError and leaves no tool child behind."""
    monkeypatch.setenv(TOOL_TIMEOUT_ENV_VAR, "1")
    start = time.monotonic()
    # bash stays the parent of the sleep: only a process-group kill reaches the sleep
    with pytest.raises(FINNUserError, match=TOOL_TIMEOUT_ENV_VAR):
        launch_process_helper(
            ["bash", "-c", f"sleep {SLEEP_MARKER}; echo never"], print_stdout=False
        )
    assert time.monotonic() - start < 10
    deadline = time.monotonic() + 5
    while _sleeper_alive() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _sleeper_alive()


def test_env_timeout_does_not_affect_short_calls(monkeypatch):
    """Calls that finish within the limit behave as before."""
    monkeypatch.setenv(TOOL_TIMEOUT_ENV_VAR, "30")
    out, _ = launch_process_helper(["echo", "quick"], print_stdout=False)
    assert out == "quick"


def test_explicit_timeout_still_raises_timeout_expired(monkeypatch):
    """Callers that pass their own timeout keep getting subprocess.TimeoutExpired."""
    monkeypatch.delenv(TOOL_TIMEOUT_ENV_VAR, raising=False)
    with pytest.raises(subprocess.TimeoutExpired):
        launch_process_helper(["sleep", "5"], timeout=0.2, print_stdout=False)


def _setup_dut(params):
    """Run the microbenchmark build setup on a bare DUT object with the given params."""
    dut = object.__new__(bench_mvau)
    dut._params = dict(params)
    return dut._step_build_setup()


def test_microbench_build_setup_arms_time_limit(monkeypatch):
    """Microbenchmark builds export the limit; tool_timeout_s overrides or disables it."""
    monkeypatch.delenv(TOOL_TIMEOUT_ENV_VAR, raising=False)
    cfg = _setup_dut({"dut": "mvau"})
    assert cfg.target_fps is None
    assert os.environ[TOOL_TIMEOUT_ENV_VAR] == str(TOOL_TIMEOUT_S)
    assert default_tool_timeout() == float(TOOL_TIMEOUT_S)
    _setup_dut({"dut": "mvau", "tool_timeout_s": 600})
    assert default_tool_timeout() == 600.0
    _setup_dut({"dut": "mvau", "tool_timeout_s": 0})
    assert TOOL_TIMEOUT_ENV_VAR not in os.environ
    assert default_tool_timeout() is None

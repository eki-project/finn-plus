# Copyright (c) 2025, Paderborn University
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the microbenchmark helpers of the CI result collector."""

import pytest

import importlib.util
import json
import os

pytestmark = pytest.mark.qor

_CI_COLLECT = os.path.join(os.path.dirname(__file__), "..", "..", "ci", "collect", "collect_fn.py")


def _load_collect_fn():
    spec = importlib.util.spec_from_file_location("collect_fn", _CI_COLLECT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REPORT = {
    "(top)": {"LUT": 1000, "DSP": 3},
    "top_instrumentation_wrap_0_0": {"LUT": 500, "DSP": 0},
    "Thresholding_rtl_0": {"LUT": 300, "DSP": 0},
    "TLastMarker_0": {"LUT": 20, "DSP": 0},
}


def test_find_dut_hierarchy_exact_prefixed_and_legacy():
    fn = _load_collect_fn()
    assert fn.find_dut_hierarchy(_REPORT, "Thresholding_rtl_0", None) == "Thresholding_rtl_0"
    # the Zynq flow prefixes the node name with the partition name
    prefixed = {
        "(top)": {},
        "StreamingDataflowPartition_0_Thresholding_rtl_0": {},
        "StreamingDataflowPartition_0_TLastMarker_hls_0": {},
    }
    assert (
        fn.find_dut_hierarchy(prefixed, "Thresholding_rtl_0", None)
        == "StreamingDataflowPartition_0_Thresholding_rtl_0"
    )
    # a second node of the same type does not confuse the anchored match (different index)
    two = {**prefixed, "StreamingDataflowPartition_0_Thresholding_rtl_1": {}}
    assert fn.find_dut_hierarchy(two, "Thresholding_rtl_1", None).endswith("_rtl_1")
    # ... but two partitions holding the same node name are ambiguous -> None, never a guess
    ambiguous = {**prefixed, "StreamingDataflowPartition_1_Thresholding_rtl_0": {}}
    assert fn.find_dut_hierarchy(ambiguous, "Thresholding_rtl_0", None) is None
    # a bare substring (e.g. a different index) is not enough
    assert fn.find_dut_hierarchy(_REPORT, "Thresholding_rtl", None) is None
    # legacy artifacts: op_type substring, unique
    assert fn.find_dut_hierarchy(_REPORT, None, "Thresholding") == "Thresholding_rtl_0"
    assert fn.find_dut_hierarchy(_REPORT, "MVAU_hls_0", "MVAU") is None
    # shell components are never returned
    assert fn.find_dut_hierarchy(_REPORT, None, "top") is None


def test_parse_wns(tmp_path):
    fn = _load_collect_fn()
    rpt = tmp_path / "post_route_timing.rpt"
    rpt.write_text(
        "Design Timing Summary\n| WNS(ns)  TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints"
        "  WHS(ns)\n| -------  -------  --------------------  -------------------  -------\n"
        "   -0.123    -1.5                    3                  4321    0.045\n"
    )
    assert fn.parse_wns(rpt) == -0.123
    rpt.write_text("WNS(ns)      TNS(ns)\n      0.517        0.000\n")
    assert fn.parse_wns(rpt) == 0.517
    assert fn.parse_wns(tmp_path / "missing.rpt") is None


def _write_power(exp_dir, name, samples):
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / f"{name}_run_1.json").write_text(json.dumps({"rails": samples}))


def test_power_report_per_experiment(tmp_path):
    fn = _load_collect_fn()
    rails = ["0V85_power", "3V3_power", "total_power"]
    sample = lambda a, b: {"0V85_power": a, "3V3_power": b, "total_power": a + b}  # noqa: E731
    # listing order on the filesystem must not matter: idle and load are kept apart
    _write_power(
        tmp_path / "instrumentation_experiment",
        "instrumentation_experiment",
        [sample(2000.0, 900.0), sample(2100.0, 900.0)],
    )
    _write_power(tmp_path / "idle_experiment", "idle_experiment", [sample(1800.0, 900.0)])
    report = fn.generate_power_report(rails, str(tmp_path))
    assert set(report) == {"idle_experiment", "instrumentation_experiment"}
    assert report["instrumentation_experiment"]["avg_0V85_power"] == 2050.0
    metrics = fn.power_metrics(report, rails)
    # legacy flat names carry the load experiment, dyn_* = load - idle
    assert metrics["avg_0V85_power"] == 2050.0 and metrics["avg_total_power"] == 2950.0
    assert metrics["dyn_0V85_power"] == 250.0 and metrics["dyn_3V3_power"] == 0.0
    assert metrics["idle_experiment/avg_0V85_power"] == 1800.0
    # without an idle experiment there is no dynamic power, the flat names still work
    only_load = {"instrumentation_experiment": report["instrumentation_experiment"]}
    assert "dyn_0V85_power" not in fn.power_metrics(only_load, rails)
    assert fn.power_metrics(only_load, rails)["avg_0V85_power"] == 2050.0
    # unknown experiment names (other flows): the first non-idle one is the load
    other = {
        "idle_experiment": report["idle_experiment"],
        "zzz": report["instrumentation_experiment"],
    }
    assert fn.power_metrics(other, rails)["dyn_0V85_power"] == 250.0
    assert fn.generate_power_report(rails, str(tmp_path / "nope")) is None


def test_sum_extra_hierarchies():
    fn = _load_collect_fn()
    extra = fn.sum_extra_hierarchies(_REPORT, "Thresholding_rtl_0", ["LUT", "DSP"])
    assert extra == {"LUT": 20, "DSP": 0, "num_nodes": 1}

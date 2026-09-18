# Copyright (c) 2025, Paderborn University
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of the microbenchmark helpers of the CI result collector."""

import pytest

import importlib.util
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


def test_find_dut_hierarchy_exact_and_substring():
    fn = _load_collect_fn()
    assert fn.find_dut_hierarchy(_REPORT, "Thresholding_rtl_0", None) == "Thresholding_rtl_0"
    # renamed node: substring of the node name still matches
    assert fn.find_dut_hierarchy(_REPORT, "Thresholding_rtl", None) == "Thresholding_rtl_0"
    # legacy artifacts: op_type substring
    assert fn.find_dut_hierarchy(_REPORT, None, "Thresholding") == "Thresholding_rtl_0"
    assert fn.find_dut_hierarchy(_REPORT, "MVAU_hls_0", "MVAU") is None
    # shell components are never returned
    assert fn.find_dut_hierarchy(_REPORT, None, "top") is None


def test_sum_extra_hierarchies():
    fn = _load_collect_fn()
    extra = fn.sum_extra_hierarchies(_REPORT, "Thresholding_rtl_0", ["LUT", "DSP"])
    assert extra == {"LUT": 20, "DSP": 0, "num_nodes": 1}

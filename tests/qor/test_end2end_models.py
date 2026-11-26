"""Tests of ci/qor/end2end_models.py: offline evaluation of model sets on end2end builds."""

import pytest

import importlib.util
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from finn.qor.database import POWER_TARGET_COL
from finn.qor.estimator import resource_target

SCRIPT = Path(__file__).resolve().parents[2] / "ci" / "qor" / "end2end_models.py"


@pytest.fixture(scope="module")
def e2e():
    """The script as a module (it adds ci/qor and src/ to sys.path itself)."""
    spec = importlib.util.spec_from_file_location("end2end_models", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["end2end_models"] = module
    spec.loader.exec_module(module)
    return module


class _Estimator:
    def __init__(self, value, covered=True):
        self.value = value
        self.covered = covered

    def covers(self, features):
        return self.covered

    def predict(self, features):
        return self.value


class _ModelSet:
    """Stub with the QoRModelSet interface: (op_type, target) -> estimator."""

    def __init__(self, models):
        self.models = models

    def get(self, op_type, target):
        return self.models.get((op_type, target))


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def artifacts(tmp_path):
    """One run with an MVAU (features), a FIFO (unsupported) and measurements."""
    reports = tmp_path / "build_artifacts" / "runs_output" / "run_0" / "reports"
    _write(reports / "metadata_bench.json", {"params": {"dut": "cnv"}})
    _write(
        reports / "qor_node_features.json",
        {
            "MVAU_hls_0": {
                "op_type": "MVAU_hls",
                "backend": "hls",
                "operator": "mvau",
                "features": {"params.backend": "hls", "dut_info.pe": 4},
            },
            "StreamingFIFO_rtl_0": {
                "op_type": "StreamingFIFO_rtl",
                "backend": "rtl",
                "operator": None,
            },
        },
    )
    _write(
        reports / "post_synth_resources.json",
        {
            # synthesis names the nodes by their instance path in the stitched design
            "StreamingDataflowPartition_1_MVAU_hls_0": {
                "LUT": 100,
                "BRAM_36K": 1,
                "DSP48E": 2,
                "URAM": 0,
            },
            "StreamingDataflowPartition_1_StreamingFIFO_rtl_0": {
                "LUT": 10,
                "BRAM_36K": 0,
                "DSP48E": 0,
                "URAM": 0,
            },
            "(top)": {"LUT": 130, "BRAM_36K": 1, "DSP48E": 2, "URAM": 0},
        },
    )
    _write(
        reports / "estimate_layer_resources.json",
        {
            "MVAU_hls_0": {"LUT": 80, "BRAM_18K": 2, "DSP": 1, "URAM": 0},
            "StreamingFIFO_rtl_0": {"LUT": 12, "BRAM_18K": 0, "DSP": 0, "URAM": 0},
            "total": {"LUT": 92, "BRAM_18K": 2, "DSP": 1, "URAM": 0},
        },
    )
    _write(
        reports / "estimate_power_empirical.json",
        {"MVAU_hls_0": {"power": 50}, "total": {"power": 50}},
    )
    measurement = tmp_path / "measurement_artifacts" / "runs_output" / "run_0" / "reports"
    for experiment, power in (("idle_experiment", 1800.0), ("instrumentation_experiment", 1850.0)):
        _write(
            measurement / experiment / f"{experiment}_run_1.json",
            {"rails": [{"0V85_power": power}, {"0V85_power": power}]},
        )
    return tmp_path


def test_load_runs_normalizes_resources(e2e, artifacts):
    nodes, runs = e2e.load_runs(artifacts)
    assert len(nodes) == 2 and list(runs) == [0]
    mvau = nodes[nodes["node"] == "MVAU_hls_0"].iloc[0]
    assert mvau["operator"] == "mvau" and mvau["features"] == {
        "params.backend": "hls",
        "dut_info.pe": 4,
    }
    assert mvau["measured.LUT"] == 100 and mvau["measured.BRAM"] == 2 and mvau["measured.DSP"] == 2
    assert mvau["finn.LUT"] == 80 and mvau["finn.BRAM"] == 2
    assert runs[0]["measured_power"]["dyn"] == pytest.approx(50.0)
    assert runs[0]["power_build"] == 50


def test_predict_nodes_and_totals_with_fallback(e2e, artifacts):
    nodes, runs = e2e.load_runs(artifacts)
    stub = _ModelSet(
        {
            ("MVAU_hls", resource_target("LUT")): _Estimator(90),
            ("MVAU_hls", resource_target("DSP")): _Estimator(7, covered=False),
            ("MVAU_hls", POWER_TARGET_COL): _Estimator(42.0),
        }
    )
    nodes = e2e.predict_nodes(nodes, {"stub": stub})
    mvau = nodes[nodes["node"] == "MVAU_hls_0"].iloc[0]
    fifo = nodes[nodes["node"] == "StreamingFIFO_rtl_0"].iloc[0]
    assert mvau["stub.LUT"] == 90 and np.isnan(mvau["stub.DSP"]) and mvau["stub.power"] == 42.0
    assert np.isnan(fifo["stub.LUT"])  # unsupported operator: no prediction

    totals = e2e.run_totals(nodes, runs, ["stub"])
    row = totals.iloc[0]
    # MVAU LUT predicted, everything else falls back to the FINN estimate of the node
    assert row["set_stub.MVAU.LUT"] == 90 and row["set_stub.Total.LUT"] == 90 + 12
    assert row["set_stub.MVAU.DSP"] == 1 and row["set_stub.fallback_nodes"] > 0
    assert (
        row["post_synth_resources.Total.LUT"] == 110 and row["post_synth_resources.Top.LUT"] == 130
    )
    assert row["set_stub.power"] == 42.0 and row["empirical_build.power"] == 50
    assert row["measured_power.dyn"] == pytest.approx(50.0)

    errors = e2e.node_error_tables(nodes, ["stub"])
    assert errors["LUT"].loc["stub", ("mvau", "MAPE")] == pytest.approx(10.0)
    assert errors["LUT"].loc["FINN", ("All", "n")] == 2
    power = e2e.power_table(totals, ["stub"])
    assert power.loc["cnv", "stub [mW]"] == 42.0

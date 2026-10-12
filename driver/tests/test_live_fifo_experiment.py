"""End-to-end test of the live FIFO sizing experiment on an emulated virtual FIFO ring.

The overlay's AXI-lite accesses (FIFO controller ring operations, instrumentation, reset) are
replaced by an emulation of the gauge behaviour on top of the synthetic monotone predicate of
``test_fifo_search``: no board or PYNQ required. This exercises the strategy dispatch, the
hardware oracle, the capability probe and the reports written by ``experiment_fifosizing``.

Run from the driver directory with: python -m pytest tests
"""

import pytest

import json

from finn_plus_driver.overlays.fifo_search import (
    STALL_IRDY,
    STALL_IVLD,
    STALL_STICKY_SHIFT,
    UNBOUNDED_DEPTH,
    TotalBitsCost,
)
from finn_plus_driver.overlays.live_fifo import (
    OP_COMP_PERIOD,
    OP_NOP,
    OP_READ_FILL,
    OP_READ_STALL,
    OP_RUN_BOUNDED,
    OP_RUN_DETACHED4,
    OP_RUN_PACED,
    OP_WRITE_FILL,
    FINNLiveFIFOOverlay,
)
from test_fifo_search import (
    DEGRADED_INTERVAL,
    DEGRADED_LATENCY,
    FEASIBLE_INTERVAL,
    FEASIBLE_LATENCY,
    MockOracle,
    brute_force_optimum,
)


class EmulatedLiveFIFOOverlay(FINNLiveFIFOOverlay):
    """Live FIFO overlay whose hardware accesses are emulated.

    ``paced_peaks`` are the occupancies reported after the paced run (the start configuration
    of the search); the feasibility of a bounded configuration and the flags/peaks it reports
    come from the ``MockOracle`` predicate.
    """

    def __init__(self, mock, widths, paced_peaks, extended_hw=True):
        """Bypass the PYNQ overlay constructor and set up the emulation state."""
        # no super().__init__(): the PYNQ Overlay would load a bitfile
        self.error = False
        self.fifo_widths = {str(i): w for i, w in enumerate(widths)}
        self.num_fifos = len(widths)
        self.folding_config = {"Defaults": {}}
        self.fifo_depth_offset = 1
        self.hw_capabilities = {"peak_occupancy": False, "sticky_stall_flags": False}
        self.fclk_mhz_actual = 100.0
        self.seed = 1
        self.mock = mock
        self.paced_peaks = list(paced_peaks)
        self.extended_hw = extended_hw
        self.depth_regs = [0] * self.num_fifos
        self.mode = "bounded"
        self.running = False
        self.num_runs = 0

    # --- emulated hardware -------------------------------------------------------------
    def reset_accelerator(self):
        """Reset: nothing to do in the emulation."""
        self.running = False

    def start_accelerator(self, throttle_interval=0, avg_window_size=64, mux_interval=0):
        """Start the emulated run."""
        self.running = True
        self.num_runs += 1

    def stop_accelerator(self):
        """Stop the emulated run."""
        self.running = False

    def ctrl_write(self, opcode=0x00, fifo_id=0x0000, value=0x00000000):
        """Only WRITE_FILL is written."""
        assert opcode == OP_WRITE_FILL
        self.depth_regs[fifo_id] = value

    def ctrl_read(self, opcode=0x00, fifo_id=0x0000, check_success=False):
        """Emulate the ring operations of the FIFO controller."""
        if opcode == OP_NOP:
            return OP_NOP
        if opcode in (OP_RUN_BOUNDED, OP_RUN_PACED, OP_RUN_DETACHED4):
            self.mode = {OP_RUN_BOUNDED: "bounded", OP_RUN_PACED: "paced"}.get(opcode, "detached")
            return opcode
        if opcode == OP_COMP_PERIOD:
            return 1000
        if opcode == OP_READ_FILL:
            if self.mode == "paced":
                return self.paced_peaks[fifo_id]
            if not self.extended_hw or not self.mock.feasible(self.depth_regs):
                return self.depth_regs[fifo_id]  # old gauge: the capacity register
            return self.mock.peak(self.depth_regs)[fifo_id]
        if opcode == OP_READ_STALL:
            if not self.extended_hw:
                return 0  # old controller: wrong packet alignment returns 0
            if self.mode == "paced":
                return STALL_IVLD << STALL_STICKY_SHIFT
            blocked = self.mock.evaluate(self.depth_regs, need_flags=True).producer_blocked
            sticky = STALL_IVLD | (STALL_IRDY if blocked[fifo_id] else 0)
            return sticky << STALL_STICKY_SHIFT
        raise AssertionError(f"unexpected opcode 0x{opcode:02x}")

    def observe_instrumentation(self, debug_print=True):
        """Report the performance of the current configuration."""
        if self.mode == "paced":
            feasible = True
        else:
            feasible = self.mock.feasible(self.depth_regs)
        if feasible:
            interval, latency = FEASIBLE_INTERVAL, FEASIBLE_LATENCY
        else:
            interval, latency = DEGRADED_INTERVAL, DEGRADED_LATENCY
        return (False, False, 1, 0, latency, latency, interval, latency, interval, 0, 0)


def make_overlay(extended_hw=True):
    """Instance with a 4-FIFO predicate whose optimum differs from the binary search result."""
    # FIFO 0 and 1 are coupled; shrinking the wide FIFO 1 pays off more than FIFO 0
    mock = MockOracle(4, [({0, 1}, 8), ({1}, 2), ({2}, 3), ({3}, 1)])
    widths = [8, 64, 16, 32]
    paced_peaks = [6, 6, 5, 4]
    return EmulatedLiveFIFOOverlay(mock, widths, paced_peaks, extended_hw=extended_hw), mock


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Skip the wall-clock waits of the phases."""
    import finn_plus_driver.overlays.live_fifo as live_fifo

    monkeypatch.setattr(live_fifo.time, "sleep", lambda s: None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"search_strategy": "binary_largest_first"},
        {"fifo_search_order": "forward"},
        {"search_strategy": "bottom_up_exact"},
        {"search_strategy": "top_down_bnb", "seed_strategy": "binary_largest_first"},
        {"search_strategy": "top_down_bnb", "seed_strategy": None, "node_order": "best_bound"},
        {"search_strategy": "bottom_up_exact", "pareto": True},
        {"search_strategy": "top_down_bnb", "max_evaluations": 5},
    ],
)
def test_experiment_fifosizing_writes_reports(tmp_path, kwargs):
    """Every strategy writes a complete, feasible report set into its own directory."""
    overlay, mock = make_overlay()
    with pytest.warns(DeprecationWarning) if "fifo_search_order" in kwargs else _nowarn():
        overlay.experiment_fifosizing(report_dir=str(tmp_path), stop_condition="both", **kwargs)

    strategy = kwargs.get("search_strategy") or "binary_" + kwargs["fifo_search_order"]
    report_dir = tmp_path / strategy / "both"
    for name in (
        "fifo_sizing_report.json",
        "fifo_config.json",
        "folding_config.json",
        "report_experiment_fifosizing.json",
        "fifo_sizing_graph.png",
    ):
        assert (report_dir / name).is_file(), name

    with open(report_dir / "fifo_sizing_report.json") as f:
        sizing = json.load(f)
    with open(report_dir / "fifo_config.json") as f:
        fifo_config = json.load(f)
    with open(report_dir / "report_experiment_fifosizing.json") as f:
        perf = json.load(f)

    # the exported configuration is feasible and consistent across the files
    regs = [sizing["search"]["fifo_depths"][i] for i in range(4)]
    assert mock.feasible(regs)
    assert [sizing["fifo_depths"][str(i)] for i in range(4)] == [r + 1 for r in regs]
    assert [fifo_config["fifo_depths"][f"StreamingFIFO_rtl_{i}"] for i in range(4)] == [
        r + 1 for r in regs
    ]
    assert sizing["fifo_size_total_kB"] == pytest.approx(TotalBitsCost([8, 64, 16, 32]).cost(regs))
    assert perf["interval_cycles"] == FEASIBLE_INTERVAL and not perf["error"]
    assert sizing["interval_degradation"] == 0 and sizing["latency_degradation"] == 0

    search = sizing["search"]
    assert search["search_strategy"] == strategy
    assert search["hw_capabilities"] == {"peak_occupancy": True, "sticky_stall_flags": True}
    assert search["all_iterations"]["0"]["fifo_depths"] == [6, 6, 5, 4]
    assert search["num_evaluations"] == len(search["all_iterations"]) == overlay.num_runs - 1
    for it in search["all_iterations"].values():
        assert "deadlock" in it and "fifo_depths" in it
    assert search["target_interval_cycles"] == FEASIBLE_INTERVAL

    optimum = brute_force_optimum(mock, TotalBitsCost([8, 64, 16, 32]), [6, 6, 5, 4])
    if strategy.startswith("binary_"):
        assert "binary_search" in sizing
        assert sizing["binary_search"]["search_order"] == strategy[len("binary_") :]
        assert set(sizing["binary_search"]) >= {
            "fifo_depths",
            "log_total_fifo_size",
            "all_iterations",
        }
        assert not search["optimal_proven"]
    elif "max_evaluations" in kwargs:
        assert search["budget_exhausted"] or search["optimal_proven"]
        assert search["lower_bound_total_kB"] <= optimum + 1e-9
        assert search["optimality_gap_kB"] >= 0
    else:
        assert "binary_search" not in sizing
        assert search["optimal_proven"], search["abort_reason"]
        assert search["lower_bounds"] == [1, 2, 3, 1]  # FIFO 0 is covered by FIFO 1 unbounded
        assert sizing["fifo_size_total_kB"] == pytest.approx(optimum)
        assert search["optimality_gap_kB"] == pytest.approx(0)
        if kwargs.get("pareto"):
            assert search["pareto"] and search["pareto"][-1]["feasible"]
        if strategy == "top_down_bnb":
            assert search["seed_total_kB"] >= optimum - 1e-9
            assert search["incumbent_confirmed"]


def test_exact_search_beats_binary_search_on_coupled_fifos(tmp_path):
    """The coupled instance is where the per-FIFO binary search is not globally optimal."""
    overlay, mock = make_overlay()
    overlay.experiment_fifosizing(report_dir=str(tmp_path), search_strategy="binary_forward")
    overlay.experiment_fifosizing(report_dir=str(tmp_path), search_strategy="top_down_bnb")
    with open(tmp_path / "binary_forward" / "both" / "fifo_sizing_report.json") as f:
        binary = json.load(f)["fifo_size_total_kB"]
    with open(tmp_path / "top_down_bnb" / "both" / "fifo_sizing_report.json") as f:
        exact = json.load(f)["fifo_size_total_kB"]
    assert exact < binary


def test_old_hardware_is_detected_and_bottom_up_refuses(tmp_path):
    """Without the gauge extensions the probe reports no capabilities and the seed is returned."""
    overlay, mock = make_overlay(extended_hw=False)
    overlay.experiment_fifosizing(
        report_dir=str(tmp_path), search_strategy="bottom_up_exact", stop_condition="interval"
    )
    with open(tmp_path / "bottom_up_exact" / "interval" / "fifo_sizing_report.json") as f:
        sizing = json.load(f)
    search = sizing["search"]
    assert search["hw_capabilities"] == {"peak_occupancy": False, "sticky_stall_flags": False}
    assert not search["optimal_proven"] and search["abort_reason"]
    assert search["fifo_depths"] == [6, 6, 5, 4]  # the seed (start configuration)
    assert mock.feasible(search["fifo_depths"])
    # the lower-bound probes use the unbounded register value
    assert any(UNBOUNDED_DEPTH in it["fifo_depths"] for it in search["all_iterations"].values())
    # the B&B still works without peaks (no tightening) and the binary search is unaffected
    overlay.experiment_fifosizing(
        report_dir=str(tmp_path), search_strategy="top_down_bnb", stop_condition="interval"
    )
    with open(tmp_path / "top_down_bnb" / "interval" / "fifo_sizing_report.json") as f:
        sizing = json.load(f)
    assert sizing["search"]["optimal_proven"]
    optimum = brute_force_optimum(mock, TotalBitsCost([8, 64, 16, 32]), [6, 6, 5, 4])
    assert sizing["fifo_size_total_kB"] == pytest.approx(optimum)


def test_relaxation_sweep_only_for_binary(tmp_path):
    """The sweep re-runs the binary search; the exact strategies skip it with a warning."""
    overlay, mock = make_overlay()
    overlay.experiment_fifosizing(
        report_dir=str(tmp_path), search_strategy="binary_reverse", relaxation_sweep=True
    )
    with open(tmp_path / "binary_reverse" / "both" / "fifo_sizing_report.json") as f:
        sizing = json.load(f)
    assert len(sizing["relaxation_sweep"]) == 11
    with pytest.warns(UserWarning, match="relaxation_sweep"):
        overlay.experiment_fifosizing(
            report_dir=str(tmp_path), search_strategy="top_down_bnb", relaxation_sweep=True
        )
    with open(tmp_path / "top_down_bnb" / "both" / "fifo_sizing_report.json") as f:
        assert "relaxation_sweep" not in json.load(f)


def test_size_iteratively_binary_search_compat(tmp_path):
    """The old entry point still returns the old dict layout."""
    overlay, mock = make_overlay()
    overlay.hw_capabilities = {"peak_occupancy": True, "sticky_stall_flags": True}
    log = overlay.size_iteratively_binary_search(
        start_depth=[6, 6, 5, 4], iteration_runtime=0.0, fifo_order_strategy="deepest_first"
    )
    assert set(log) >= {
        "duration",
        "fifo_depths",
        "log_total_fifo_size",
        "log_interval",
        "log_min_latency",
        "log_latency",
        "all_iterations",
    }
    assert mock.feasible(log["fifo_depths"])
    assert log["all_iterations"]["1"]["tested_fifo"] in (0, 1)  # deepest first: depth 6


class _nowarn:
    """Context manager that does nothing (counterpart of pytest.warns in the parametrisation)."""

    def __enter__(self):
        """Enter."""
        return self

    def __exit__(self, *exc):
        """Exit without suppressing anything."""
        return False

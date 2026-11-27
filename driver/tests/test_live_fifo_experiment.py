"""End-to-end test of the live FIFO sizing experiment on an emulated virtual FIFO ring.

The overlay's AXI-lite accesses (FIFO controller ring operations, instrumentation, reset) are
replaced by an emulation of the gauge behaviour on top of the synthetic monotone predicate of
``test_fifo_search``: no board or PYNQ required. This exercises the strategy dispatch, the
hardware oracle with its steady-state detection (frames completing over polls, deadlocking
and slowly degrading configurations), the capability probe and the reports written by
``experiment_fifosizing``.

Run from the driver directory with: python -m pytest tests
"""

import pytest

import json
from finn_plus_driver.overlays.fifo_search import (
    STALL_IRDY,
    STALL_IVLD,
    STALL_STICKY_SHIFT,
    UNBOUNDED_DEPTH,
    FeasibilityCheck,
    PrimitiveCost,
    TotalBitsCost,
)
from finn_plus_driver.overlays.live_fifo import (
    DEFAULT_WINDOW_FRAMES,
    OP_BARRIER_CLEAN,
    OP_COMP_PERIOD,
    OP_NOP,
    OP_READ_FILL,
    OP_READ_STALL,
    OP_RUN_BOUNDED,
    OP_RUN_DETACHED4,
    OP_RUN_PACED,
    OP_WRITE_FILL,
    FINNLiveFIFOOverlay,
    LiveFIFOOracle,
    evaluation_cap,
)
from test_fifo_search import (
    DEGRADED_INTERVAL,
    DEGRADED_LATENCY,
    FEASIBLE_INTERVAL,
    FEASIBLE_LATENCY,
    MockOracle,
    brute_force_optimum,
    synthetic_tables,
)


class EmulatedLiveFIFOOverlay(FINNLiveFIFOOverlay):
    """Live FIFO overlay whose hardware accesses are emulated.

    ``paced_peaks`` are the occupancies reported after the paced run (the start configuration
    of the search); the feasibility of a bounded configuration and the flags/peaks it reports
    come from the ``MockOracle`` predicate.

    The steady-state registers (``steady_state_hw``) are emulated per run with a virtual
    clock that advances by one poll per ``observe_steady_state`` call: a feasible configuration
    completes ``frames_per_poll`` frames per poll with a constant window (minimum = maximum);
    a degraded one (mock: ``DEGRADED_*``) runs steadily slow, its whole window growing by
    ``degrade_step`` per poll; a severely violated one (mock: deadlock) completes one frame and
    then idles with a growing pending-frame age. ``frozen_mode`` makes every bounded run look
    frozen before its first frame (no frames, no pending age, only the run cycle counter
    advancing); ``oscillate_mode`` makes it complete frames whose window minimum is within
    and whose window maximum is above the interval threshold, which no rule decides, to
    exercise the wall-clock cap. The run cycle counter advances by ``cycles_per_poll``.
    """

    def __init__(
        self,
        mock,
        widths,
        paced_peaks,
        extended_hw=True,
        steady_state_hw=True,
        frames_per_poll=3,
        degrade_step=10,
        fifo_cost_tables=None,
    ):
        """Bypass the PYNQ overlay constructor and set up the emulation state."""
        # no super().__init__(): the PYNQ Overlay would load a bitfile
        self.error = False
        self.fifo_widths = {str(i): w for i, w in enumerate(widths)}
        self.num_fifos = len(widths)
        self.fifo_cost_tables = fifo_cost_tables
        self.folding_config = {"Defaults": {}}
        self.fifo_depth_offset = 1
        # the mock predicate is defined on the requested depths: no capacity rounding unless a
        # test installs a CapacityModel
        self.capacity_model = None
        self.hw_capabilities = {
            "peak_occupancy": False,
            "sticky_stall_flags": False,
            "steady_state_registers": False,
        }
        self.fclk_mhz_actual = 100.0
        self.seed = 1
        self.mock = mock
        self.paced_peaks = list(paced_peaks)
        self.extended_hw = extended_hw
        self.steady_state_hw = steady_state_hw
        self.frames_per_poll = frames_per_poll
        self.degrade_step = degrade_step
        self.frozen_mode = False
        self.oscillate_mode = False
        self.cycles_per_poll = 1000
        self.depth_regs = [0] * self.num_fifos
        self.mode = "bounded"
        self.running = False
        self.num_runs = 0
        self.polls = 0
        self.num_polls = 0
        self.avg_window_sizes = []

    # --- emulated hardware -------------------------------------------------------------
    def reset_accelerator(self):
        """Reset: nothing to do in the emulation."""
        self.running = False

    def start_accelerator(self, throttle_interval=0, avg_window_size=64, mux_interval=0):
        """Start the emulated run."""
        self.running = True
        self.num_runs += 1
        self.polls = 0
        self.avg_window_sizes.append(avg_window_size)

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
        if opcode == OP_BARRIER_CLEAN:
            return OP_BARRIER_CLEAN
        if opcode == OP_COMP_PERIOD:
            return FEASIBLE_INTERVAL  # the pipeline keeps up with the pacing at this period
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

    def _bounded_class(self):
        """Classify the bounded configuration like ``MockOracle.evaluate`` does."""
        if self.mode == "paced" or self.mock.feasible(self.depth_regs):
            return "feasible"
        severe = any(sum(self.depth_regs[e] for e in T) * 2 < b for T, b in self.mock.constraints)
        return "deadlock" if severe else "degraded"

    def observe_instrumentation(self, debug_print=True):
        """Report the performance of the current configuration (single read, last sample)."""
        cls = self._bounded_class()
        if cls == "feasible":
            interval, latency = FEASIBLE_INTERVAL, FEASIBLE_LATENCY
        elif cls == "deadlock" and self.steady_state_hw:
            interval, latency = 0, 0
        else:
            interval, latency = DEGRADED_INTERVAL, DEGRADED_LATENCY
        return (False, False, 1, 0, latency, latency, interval, latency, interval, 0, 0)

    # --- emulated steady-state registers ------------------------------------------------
    def has_steady_state_registers(self):
        """Whether the emulated bitfile has the steady-state registers."""
        return self.steady_state_hw

    def instrumentation_read(self, name):
        """Only min_latency is read directly by the oracle."""
        assert name == "min_latency"
        return FEASIBLE_LATENCY

    def observe_steady_state(self):
        """One poll of the emulated steady-state registers of the current run."""
        assert self.steady_state_hw and self.running
        self.polls += 1
        self.num_polls += 1
        polls = self.polls
        cls = self._bounded_class()
        window = self.avg_window_sizes[-1]
        regs = {
            "overflow": False,
            "underflow": False,
            "run_cycles": self.cycles_per_poll * polls,
        }

        def win(frames, interval, latency):
            """Set the window registers of a run with a constant interval and latency."""
            regs.update(run_frames=frames, avg_fill=min(frames, window))
            regs.update(int_win_min=interval, int_win_max=interval)
            regs.update(lat_win_min=latency, lat_win_max=latency)

        if self.frozen_mode:
            win(0, 0, 0)
            regs.update(idle_cycles=0, oldest_pending_age=0)
        elif self.oscillate_mode:
            win(self.frames_per_poll * polls, FEASIBLE_INTERVAL, FEASIBLE_LATENCY)
            regs.update(int_win_max=FEASIBLE_INTERVAL + 20)
            regs.update(
                idle_cycles=FEASIBLE_INTERVAL // 2, oldest_pending_age=FEASIBLE_LATENCY // 2
            )
        elif cls == "feasible":
            win(self.frames_per_poll * polls, FEASIBLE_INTERVAL, FEASIBLE_LATENCY)
            regs.update(
                idle_cycles=FEASIBLE_INTERVAL // 2, oldest_pending_age=FEASIBLE_LATENCY // 2
            )
        elif cls == "degraded":
            win(
                self.frames_per_poll * polls,
                FEASIBLE_INTERVAL + self.degrade_step * polls,
                FEASIBLE_LATENCY + 4 * self.degrade_step * polls,
            )
            regs.update(
                idle_cycles=FEASIBLE_INTERVAL // 2, oldest_pending_age=FEASIBLE_LATENCY // 2
            )
        else:  # deadlock after the first frame
            win(1, 0, FEASIBLE_LATENCY)
            regs.update(idle_cycles=1000 * polls, oldest_pending_age=1000 * polls + 50)
        return regs


def make_overlay(extended_hw=True, cost_tables=False, **kwargs):
    """Instance with a 4-FIFO predicate whose optimum differs from the binary search result."""
    # FIFO 0 and 1 are coupled; shrinking the wide FIFO 1 pays off more than FIFO 0
    mock = MockOracle(4, [({0, 1}, 8), ({1}, 2), ({2}, 3), ({3}, 1)])
    widths = [8, 64, 16, 32]
    paced_peaks = [6, 6, 5, 4]
    tables = synthetic_tables(widths, max_depth=16) if cost_tables else None
    overlay = EmulatedLiveFIFOOverlay(
        mock, widths, paced_peaks, extended_hw=extended_hw, fifo_cost_tables=tables, **kwargs
    )
    return overlay, mock


class FakeClock:
    """Replacement of the ``time`` module: sleeping advances a virtual monotonic clock."""

    def __init__(self):
        """Start at zero."""
        self.now = 0.0
        self.num_sleeps = 0
        self.slept_s = 0.0

    def sleep(self, s):
        """Advance the clock instead of waiting."""
        self.now += s
        self.num_sleeps += 1
        self.slept_s += s

    def monotonic(self):
        """Virtual monotonic time."""
        return self.now

    def time(self):
        """Virtual wall-clock time."""
        return self.now


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    """Skip the wall-clock waits of the phases and make the evaluation cap deterministic."""
    import finn_plus_driver.overlays.live_fifo as live_fifo

    fake = FakeClock()
    monkeypatch.setattr(live_fifo, "time", fake)
    return fake


#: Window of the experiment tests: full after 2 polls of the emulation (3 frames per poll)
WINDOW = 4
#: Decision rules of the polling oracle that end an evaluation without the cap
DECIDED = {"window_clean", "window_slow", "latency_violation", "idle_timeout", "pending_timeout"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"search_strategy": "binary_largest_first"},
        {"fifo_search_order": "forward"},
        {"search_strategy": "bottom_up_exact"},
        {"search_strategy": "top_down_bnb", "seed_strategy": "binary_largest_first"},
        {
            "search_strategy": "top_down_bnb",
            "seed_strategy": None,
            "node_order": "best_bound",
            "report_name": "top_down_bnb_seed_occupancy",
        },
        {"search_strategy": "bottom_up_exact", "pareto": True},
        {"search_strategy": "top_down_bnb", "max_evaluations": 5},
    ],
)
def test_experiment_fifosizing_writes_reports(tmp_path, kwargs):
    """Every strategy writes a complete, feasible report set into its own directory."""
    overlay, mock = make_overlay()
    with pytest.warns(DeprecationWarning) if "fifo_search_order" in kwargs else _nowarn():
        overlay.experiment_fifosizing(
            report_dir=str(tmp_path), stop_condition="both", window_frames=WINDOW, **kwargs
        )

    strategy = kwargs.get("search_strategy") or "binary_" + kwargs["fifo_search_order"]
    report_dir = tmp_path / kwargs.get("report_name", strategy) / "both"
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
    assert search["hw_capabilities"] == {
        "peak_occupancy": True,
        "sticky_stall_flags": True,
        "steady_state_registers": True,
    }
    assert search["all_iterations"]["0"]["fifo_depths"] == [6, 6, 5, 4]
    assert search["num_evaluations"] == len(search["all_iterations"]) == overlay.num_runs - 1
    for it in search["all_iterations"].values():
        assert "deadlock" in it and "fifo_depths" in it
        assert it["decided_by"] in DECIDED and "undecided" not in it
        assert it["frames"] >= 1 and it["eval_time_s"] >= 0
    assert search["target_interval_cycles"] == FEASIBLE_INTERVAL
    assert search["target_latency_cycles"] == FEASIBLE_LATENCY
    # the target measurement waited for a clean window, every verdict was decided by a rule
    assert search["all_iterations"]["0"]["decided_by"] == "window_clean"
    assert search["all_iterations"]["0"]["frames"] >= WINDOW + 1
    assert search["steady_state_detection"] and search["window_frames"] == WINDOW
    assert (
        search["evaluation_cap_frames"]
        == evaluation_cap(FEASIBLE_LATENCY, FEASIBLE_INTERVAL, WINDOW, 100.0, 4.0, 0.002)[0]
    )
    assert set(overlay.avg_window_sizes[1:]) == {WINDOW}  # [0] is the paced run
    assert sum(search["decisions"].values()) == search["num_evaluations"]
    assert search["decisions"]["window_clean"] >= 1
    assert "cap_expired" not in search["decisions"]
    assert "fixed_runtime" not in search["decisions"]

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


def test_old_hardware_is_detected_and_bottom_up_refuses(tmp_path, clock):
    """Without the gauge extensions the probe reports no capabilities and the seed is returned."""
    overlay, mock = make_overlay(extended_hw=False, steady_state_hw=False)
    overlay.experiment_fifosizing(
        report_dir=str(tmp_path), search_strategy="bottom_up_exact", stop_condition="interval"
    )
    with open(tmp_path / "bottom_up_exact" / "interval" / "fifo_sizing_report.json") as f:
        sizing = json.load(f)
    search = sizing["search"]
    assert search["hw_capabilities"] == {
        "peak_occupancy": False,
        "sticky_stall_flags": False,
        "steady_state_registers": False,
    }
    # old bitfile: every evaluation slept for the fixed runtime and read once
    assert not search["steady_state_detection"] and search["window_frames"] is None
    assert search["decisions"] == {"fixed_runtime": search["num_evaluations"]}
    assert overlay.num_polls == 0
    assert clock.num_sleeps >= search["num_evaluations"] + 1  # evaluations + the paced run
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


def test_evaluation_cap_from_paced_measurements():
    """The cap covers cap_factor pipeline depths plus windows at the clock, plus the floor."""
    # mobilenetv1: latency 711946 / interval 394272 -> 2 + 1 frames of depth, window 4 -> 28
    frames, cap_s = evaluation_cap(711946, 394272, 4, 100.0, 4.0, 0.002)
    assert frames == 28
    assert cap_s == pytest.approx(28 * 394272 / 1e8 + 0.002)
    # cybsec: the floor dominates the 36 frames of 70 cycles
    frames, cap_s = evaluation_cap(213, 70, 4, 100.0, 4.0, 0.002)
    assert frames == 36 and cap_s == pytest.approx(36 * 70 / 1e8 + 0.002)
    assert evaluation_cap(1000, 0, 4, 100.0, 4.0, 0.0)[0] >= 4 * 1005  # degenerate interval


def test_default_window_and_fixed_runtime_disabled(tmp_path):
    """Without kwargs the default window and the cap from the paced run apply; the detection
    can be disabled."""
    overlay, mock = make_overlay()
    overlay.experiment_fifosizing(report_dir=str(tmp_path), search_strategy="binary_forward")
    with open(tmp_path / "binary_forward" / "both" / "fifo_sizing_report.json") as f:
        search = json.load(f)["search"]
    assert search["window_frames"] == DEFAULT_WINDOW_FRAMES == 4
    frames, cap_s = evaluation_cap(FEASIBLE_LATENCY, FEASIBLE_INTERVAL, 4, 100.0, 4.0, 0.002)
    assert search["evaluation_cap_frames"] == frames
    assert search["evaluation_cap_s"] == pytest.approx(cap_s)
    assert search["all_iterations"]["0"]["frames"] == 6  # 2 polls of 3 frames fill the window
    assert "cap_expired" not in search["decisions"]

    overlay, mock = make_overlay()
    overlay.experiment_fifosizing(
        report_dir=str(tmp_path),
        search_strategy="binary_forward",
        report_name="fixed",
        steady_state_detection=False,
    )
    with open(tmp_path / "fixed" / "both" / "fifo_sizing_report.json") as f:
        search = json.load(f)["search"]
    assert not search["hw_capabilities"]["steady_state_registers"]
    assert search["decisions"] == {"fixed_runtime": search["num_evaluations"]}
    assert overlay.num_polls == 0


def make_oracle(overlay, stop_condition, frames=WINDOW, cap_s=0.004, feasibility=True):
    """Polling oracle on the emulated overlay with the thresholds of the mock targets."""
    overlay.hw_capabilities = {
        "peak_occupancy": True,
        "sticky_stall_flags": True,
        "steady_state_registers": True,
    }
    oracle = LiveFIFOOracle(
        overlay,
        iteration_runtime=0.001,
        throttle_interval=FEASIBLE_INTERVAL,
        hw_capabilities=overlay.hw_capabilities,
        window_frames=frames,
        poll_interval_s=0.0003,
        cap_s=cap_s,
    )
    if feasibility:
        oracle.feasibility = FeasibilityCheck(
            stop_condition, 0.0, FEASIBLE_INTERVAL, FEASIBLE_LATENCY
        )
    assert oracle.polling
    return oracle


@pytest.mark.parametrize("stop_condition", ["interval", "latency", "both"])
def test_oracle_decision_rules(stop_condition, clock):
    """Each class of configuration ends by the expected rule, at the expected poll."""
    overlay, mock = make_overlay()
    oracle = make_oracle(overlay, stop_condition)
    fb = oracle.feasibility
    # degraded instance: constraint {2} >= 3 violated by 2 (not severe), {3} >= 1 deadlocks at 0
    feasible, degraded, deadlock = [6, 6, 5, 4], [6, 6, 2, 4], [6, 6, 5, 0]

    # feasible: the window (4 frames after the first) is full and clean at the 2nd poll
    r = oracle.evaluate(feasible, need_flags=True, need_peaks=True)
    assert r.decided_by == "window_clean" and r.frames == 6 and overlay.polls == 2
    assert (r.interval, r.latency, r.min_latency) == (
        FEASIBLE_INTERVAL,
        FEASIBLE_LATENCY,
        FEASIBLE_LATENCY,
    )
    assert fb.feasible(r) and not r.deadlock and not r.undecided and r.peak is not None

    # steadily slow: a latency above the target is definitive at once (1st poll); the interval
    # alone needs a full window whose minimum is above the threshold (2nd poll)
    r = oracle.evaluate(degraded)
    if stop_condition == "interval":
        assert r.decided_by == "window_slow" and overlay.polls == 2 and r.frames == 6
        assert r.interval == FEASIBLE_INTERVAL + 20 > fb.relaxed_interval_threshold
    else:
        assert r.decided_by == "latency_violation" and overlay.polls == 1 and r.frames == 3
        assert r.latency > fb.relaxed_latency_threshold
    assert not fb.feasible(r) and not r.deadlock

    # deadlock after the first frame: pending age (latency rules, 1st poll) or idle longer than
    # the stall bound of latency + interval (2nd poll)
    r = oracle.evaluate(deadlock, need_flags=True)
    if stop_condition == "interval":
        assert r.decided_by == "idle_timeout" and overlay.polls == 2
        assert r.interval == 2000 > fb.stall_bound == FEASIBLE_LATENCY + FEASIBLE_INTERVAL
    else:
        assert r.decided_by == "pending_timeout" and overlay.polls == 1
        assert r.latency == 1050 > fb.relaxed_latency_threshold
    assert r.frames == 1
    assert not fb.feasible(r) and not r.deadlock and r.producer_blocked is not None


def test_oracle_degradation_within_threshold_is_clean(clock):
    """A degradation below the relaxed threshold is accepted once the window is full."""
    overlay, mock = make_overlay(degrade_step=1)  # +1 cycle per poll
    oracle = make_oracle(overlay, "interval")
    oracle.feasibility = FeasibilityCheck("interval", 0.1, FEASIBLE_INTERVAL, FEASIBLE_LATENCY)
    r = oracle.evaluate([6, 6, 2, 4])
    assert r.decided_by == "window_clean" and r.frames == 6 and overlay.polls == 2
    assert r.interval == FEASIBLE_INTERVAL + 2  # the window maximum when the window filled
    assert oracle.feasibility.feasible(r)
    # a tighter relaxation turns the same configuration into a slow window at the same poll
    oracle.feasibility = FeasibilityCheck("interval", 0.01, FEASIBLE_INTERVAL, FEASIBLE_LATENCY)
    r = oracle.evaluate([6, 6, 2, 4])
    assert r.decided_by == "window_slow" and overlay.polls == 2
    assert not oracle.feasibility.feasible(r)


@pytest.mark.parametrize("stop_condition", ["interval", "both"])
def test_oracle_frozen_before_first_frame_is_idle(stop_condition, clock):
    """A run that never completes a frame is rejected by the idle rule on the run cycles.

    ``idle_cycles`` counts from the reset and is meaningless before the first frame (it
    includes the FIFO configuration phase), so the rule uses the cycles since the start of
    the run there instead of waiting for the cap.
    """
    overlay, mock = make_overlay()
    overlay.frozen_mode = True
    oracle = make_oracle(overlay, stop_condition)
    fb = oracle.feasibility
    r = oracle.evaluate([6, 6, 5, 4])
    assert r.decided_by == "idle_timeout" and r.frames == 0 and r.deadlock and not r.undecided
    assert overlay.polls == 2  # 2000 run cycles > stall bound of 1500 at the 2nd poll
    assert r.interval == 2000 > fb.stall_bound == FEASIBLE_LATENCY + FEASIBLE_INTERVAL
    assert not fb.feasible(r)


def test_oracle_cap_rejects(clock):
    """When no rule fires before the cap, the evaluation is undecided and never feasible."""
    overlay, mock = make_overlay()
    # the window minimum stays within and its maximum above the interval threshold: neither a
    # slow nor a clean window
    overlay.oscillate_mode = True
    oracle = make_oracle(overlay, "interval", cap_s=0.004)
    t0 = clock.now
    r = oracle.evaluate([6, 6, 5, 4])
    assert r.decided_by == "cap_expired" and r.frames > 0 and r.undecided and not r.deadlock
    assert clock.now - t0 == pytest.approx(oracle.cap_s, abs=oracle.poll_interval_s)
    assert not oracle.feasibility.feasible(r)
    # a slow start that never fills the window within the cap is rejected the same way, even
    # though its values are within the thresholds
    overlay.oscillate_mode = False
    overlay.frames_per_poll = 1
    oracle = make_oracle(overlay, "interval", cap_s=0.0006)  # 3 polls: 3 frames < window + 1
    r = oracle.evaluate([6, 6, 5, 4])
    assert r.decided_by == "cap_expired" and r.undecided and not r.deadlock
    assert r.interval == FEASIBLE_INTERVAL and not oracle.feasibility.feasible(r)


def test_oracle_target_measurement_without_thresholds(clock):
    """Before the thresholds exist (target measurement) only a constant window or the cap decide."""
    overlay, mock = make_overlay()
    oracle = make_oracle(overlay, "both", feasibility=False)
    r = oracle.evaluate([6, 6, 5, 0])  # would be a deadlock, but no threshold is known yet
    assert r.decided_by == "cap_expired" and r.frames == 1 and r.undecided
    r = oracle.evaluate([6, 6, 5, 4])
    assert r.decided_by == "window_clean" and r.frames == 6


def test_target_without_steady_state_aborts(tmp_path):
    """A start configuration that never shows a clean window aborts the experiment."""
    overlay, mock = make_overlay()
    overlay.frames_per_poll = 1
    with pytest.raises(RuntimeError, match="no steady state within the evaluation cap"):
        overlay.experiment_fifosizing(
            report_dir=str(tmp_path),
            search_strategy="binary_forward",
            cap_factor=0.0,
            cap_floor_s=0.0002,  # 3 polls at 100 us, 3 frames: the window never fills
        )


class _nowarn:
    """Context manager that does nothing (counterpart of pytest.warns in the parametrisation)."""

    def __enter__(self):
        """Enter."""
        return self

    def __exit__(self, *exc):
        """Exit without suppressing anything."""
        return False


def test_experiment_fifosizing_with_capacity_model(tmp_path):
    """With a CapacityModel the gauges get fifo.sv's capacities and the report realized bits."""
    from finn_plus_driver.overlays.fifo_capacity import CapacityModel

    # a predicate on the register values the hardware really gets: FIFO 1 needs 300 slots
    mock = MockOracle(4, [({0}, 3), ({1}, 299), ({2}, 1), ({3}, 1)])
    widths = [8, 64, 16, 32]
    overlay = EmulatedLiveFIFOOverlay(mock, widths, paced_peaks=[6, 400, 5, 4])
    overlay.capacity_model = CapacityModel([8, 64, 16, 32])
    written = []
    original = overlay.configure_fifos_bounded

    def record(depths):
        """Record the register values the hardware is configured with."""
        written.append(list(depths))
        return original(depths)

    overlay.configure_fifos_bounded = record
    overlay.experiment_fifosizing(
        report_dir=str(tmp_path), search_strategy="binary_forward", window_frames=WINDOW
    )
    with open(tmp_path / "binary_forward" / "both" / "fifo_sizing_report.json") as f:
        sizing = json.load(f)
    with open(tmp_path / "binary_forward" / "both" / "fifo_config.json") as f:
        fifo_config = json.load(f)

    # every register value written is a capacity of fifo.sv minus the offset: never below the
    # 5-slot shift-register floor, 515 slots for a block RAM request between 258 and 513
    assert written and all(all(r >= 4 for r in regs) for regs in written)
    assert all(
        regs[1] in (514, 516, 1026) or regs[1] >= 1026 or regs[1] == 514
        for regs in written
        if regs[1] > 256
    )
    # the search starts at the floor-lifted peaks and ends on plateau tops
    search = sizing["search"]
    requested = search["fifo_depths"]
    realized = search["realized_fifo_depths"]
    assert realized == [4, 512, 4, 4] and requested[1] <= 512 and requested[1] >= 257
    assert sizing["fifo_depths"] == {str(i): d + 1 for i, d in enumerate(realized)}
    assert sizing["fifo_capacities"] == {"0": 5, "1": 515, "2": 5, "3": 5}
    assert search["fifo_capacities"] == [5, 515, 5, 5]
    assert sizing["fifo_sizes"] == {"0": 40, "1": 515 * 64, "2": 80, "3": 160}
    assert sizing["fifo_size_total_kB"] == pytest.approx((40 + 515 * 64 + 80 + 160) / 8000)
    assert sizing["fifo_size_requested_kB"] == pytest.approx(
        sum((d + 1) * w for d, w in zip(requested, widths)) / 8000
    )
    assert sizing["fifo_size_requested_kB"] <= sizing["fifo_size_total_kB"]
    assert search["total_kB"] == pytest.approx(sizing["fifo_size_total_kB"])
    assert [fifo_config["fifo_depths"][f"StreamingFIFO_rtl_{i}"] for i in range(4)] == [
        5,
        513,
        5,
        5,
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"search_strategy": "binary_largest_first"},
        {"search_strategy": "bottom_up_exact"},
        {"search_strategy": "top_down_bnb"},
    ],
)
def test_experiment_fifosizing_with_primitive_cost(tmp_path, kwargs):
    """cost_model="primitives" searches the cost ladder and exports the realized depths."""
    overlay, mock = make_overlay(cost_tables=True)
    overlay.experiment_fifosizing(report_dir=str(tmp_path), cost_model="primitives", **kwargs)
    strategy = kwargs["search_strategy"]
    report_dir = tmp_path / strategy / "both"
    with open(report_dir / "fifo_sizing_report.json") as f:
        sizing = json.load(f)
    with open(report_dir / "fifo_config.json") as f:
        fifo_config = json.load(f)

    cost_model = PrimitiveCost(overlay.fifo_cost_tables, offset=1)
    search = sizing["search"]
    assert sizing["cost_model"] == "primitives" and sizing["cost_unit"] == "LUTeq"
    assert search["cost_model"] == "primitives" and search["cost_unit"] == "LUTeq"
    measured = [search["fifo_depths"][i] for i in range(4)]
    realized = [search["realized_fifo_depths"][i] for i in range(4)]
    assert mock.feasible(measured) and mock.feasible(realized)
    assert realized == list(cost_model.realize(measured))
    assert all(r >= m for r, m in zip(realized, measured))
    assert [sizing["fifo_depths"][str(i)] for i in range(4)] == [r + 1 for r in realized]
    assert [fifo_config["fifo_depths"][f"StreamingFIFO_rtl_{i}"] for i in range(4)] == [
        r + 1 for r in realized
    ]
    assert sizing["total_cost"] == pytest.approx(cost_model.cost(realized))
    assert sizing["total_cost"] == pytest.approx(search["total_kB"])
    assert sizing["fifo_size_total_kB"] == pytest.approx(
        TotalBitsCost([8, 64, 16, 32]).cost(realized)
    )
    prim = sizing["primitive_cost"]
    assert prim["unit"] == "LUTeq" and prim["weights"]["bram18"] == 30.0
    assert prim["total"] == cost_model.vector(realized)
    assert prim["scalar"] == pytest.approx(sizing["total_cost"])
    assert (
        prim["fifos"]["1"]["depth"] == realized[1] + 1 and prim["fifos"]["1"]["ram_style"] == "auto"
    )
    assert sum(prim["fifos"][str(i)]["lut"] for i in range(4)) == prim["total"]["lut"]
    # the search only evaluated candidate depths (or the unbounded probes / start depths)
    lattice = overlay.make_cost_model("primitives", (6, 6, 5, 4))[1]
    for it in search["all_iterations"].values():
        for i, d in enumerate(it["fifo_depths"]):
            assert d >= UNBOUNDED_DEPTH or d in lattice.cands(i) or d == (6, 6, 5, 4)[i]
    optimum = brute_force_optimum(mock, cost_model, [6, 6, 5, 4])
    if strategy.startswith("binary_"):
        assert sizing["total_cost"] >= optimum - 1e-9
    else:
        assert search["optimal_proven"], search["abort_reason"]
        assert sizing["total_cost"] == pytest.approx(optimum)


def test_bits_cost_model_still_reports_primitives(tmp_path):
    """With tables available the default bits search reports the primitive vector as well."""
    overlay, mock = make_overlay(cost_tables=True)
    overlay.experiment_fifosizing(report_dir=str(tmp_path), search_strategy="bottom_up_exact")
    with open(tmp_path / "bottom_up_exact" / "both" / "fifo_sizing_report.json") as f:
        sizing = json.load(f)
    assert sizing["cost_model"] == "bits" and sizing["cost_unit"] == "kB"
    assert sizing["total_cost"] == pytest.approx(sizing["fifo_size_total_kB"])
    regs = [sizing["search"]["fifo_depths"][i] for i in range(4)]
    assert sizing["search"]["realized_fifo_depths"] == regs
    assert sizing["primitive_cost"]["total"]["lut"] > 0
    optimum = brute_force_optimum(mock, TotalBitsCost([8, 64, 16, 32]), [6, 6, 5, 4])
    assert sizing["fifo_size_total_kB"] == pytest.approx(optimum)


def test_primitive_cost_model_needs_tables(tmp_path):
    """Without fifo_cost_tables in the settings the primitive cost model is refused."""
    overlay, mock = make_overlay()
    with pytest.raises(ValueError, match="fifo_cost_tables"):
        overlay.experiment_fifosizing(
            report_dir=str(tmp_path), search_strategy="top_down_bnb", cost_model="primitives"
        )
    with pytest.raises(ValueError, match="Unknown cost_model"):
        overlay.experiment_fifosizing(report_dir=str(tmp_path), cost_model="joules")
    # no tables: the report says so instead of failing
    overlay.experiment_fifosizing(report_dir=str(tmp_path), search_strategy="binary_forward")
    with open(tmp_path / "binary_forward" / "both" / "fifo_sizing_report.json") as f:
        assert json.load(f)["primitive_cost"] is None

"""Driver for live FIFO sizing experiments."""

import json
import math
import matplotlib.pyplot as plt
import random
import time
import warnings

from finn_plus_driver.overlays.fifo_capacity import CapacityModel
from finn_plus_driver.overlays.fifo_search import (
    STALL_IVLD,
    STALL_STICKY_SHIFT,
    UNBOUNDED_DEPTH,
    EvalResult,
    EvaluationCache,
    FeasibilityCheck,
    Lattice,
    PrimitiveCost,
    SearchBudget,
    SearchContext,
    TotalBitsCost,
    cache_to_iterations,
    decision_counts,
    producer_blocked_from_stall,
    resolve_strategy_name,
    run_search,
    search_result_to_dict,
)
from finn_plus_driver.overlays.instrumentation import FINNInstrumentationOverlay
from pathlib import Path
from typing import Any

# Ring opcodes of the virtual FIFO gauge (finn-rtllib/fifo_virtual/hdl/fifo_gauge_pkg.sv)
OP_NOP = 0x00
OP_RUN_BOUNDED = 0x04
OP_RUN_PACED = 0x05
OP_RUN_DETACHED4 = 0x07
OP_BARRIER_CLEAN = 0x08
OP_COMP_PERIOD = 0x0A
OP_READ_FILL = 0x0C
OP_READ_STALL = 0x0D
OP_WRITE_FILL = 0x0E

#: Kwargs of experiment_fifosizing that are passed on to the search strategy
STRATEGY_KWARGS = (
    "seed_strategy",
    "upper_bound_factor",
    "node_order",
    "pareto",
    "pareto_cost_limit",
    "allow_unreliable_flags",
)


#: Defaults of the steady-state detection: frames of a clean window that count as steady
#: state, the safety factor of the evaluation time cap on the frames a clean window can take,
#: and the fixed share of the cap covering the AXI-lite polling of short-frame designs.
DEFAULT_WINDOW_FRAMES = 4
DEFAULT_CAP_FACTOR = 4.0
DEFAULT_CAP_FLOOR_S = 0.002
#: Largest window the instrumentation wrapper is built with by default
#: (``instrumentation_avg_n``); the hardware clamps a larger request to its own size.
DEFAULT_HW_WINDOW_MAX = 64


def evaluation_cap(paced_latency, paced_interval, window_frames, fclk_mhz, cap_factor, cap_floor_s):
    """Return the frame budget and the wall-clock cap of one polled evaluation.

    A feasible configuration shows a clean window once its warm-up is over, and the warm-up is
    on the order of the pipeline depth in frames (the paced latency over the paced interval),
    so the cap is ``cap_factor`` times the frames of one pipeline depth plus the window,
    converted to seconds at the accelerator clock, plus ``cap_floor_s`` for the polling of
    designs whose frames are shorter than one AXI-lite access. Only pathological evaluations
    (neither clean nor violated) ever reach it; they are rejected.
    """
    interval = max(int(paced_interval), 1)
    depth_frames = math.ceil(int(paced_latency) / interval) + 1
    cap_frames = int(math.ceil(cap_factor * (depth_frames + int(window_frames))))
    cap_s = cap_frames * interval / (float(fclk_mhz) * 1e6) + float(cap_floor_s)
    return cap_frames, cap_s


class LiveFIFOOracle:
    """Hardware implementation of the ``Oracle`` protocol of ``fifo_search``.

    One evaluation is the inner loop of the original binary search: reset the accelerator,
    configure the bounded FIFO depths, start it with the throttled input and observe the
    instrumentation. Only when a strategy asks for them, the peak occupancies (``READ_FILL``)
    and the stall flags (``READ_STALL``) of every FIFO are read from the ring afterwards. Ring
    reads are non-intrusive; they cost a few ms per FIFO.

    With the steady-state registers of the instrumentation wrapper (``hw_capabilities
    ["steady_state_registers"]``) and a window of ``window_frames`` frames, the observation
    is a polling loop that ends by the first of these rules (``EvalResult.decided_by``):

    - ``instrumentation_error``: timestamp overflow/underflow (deadlock), negative;
    - ``idle_timeout``: no frame completed for longer than ``FeasibilityCheck.stall_bound``
      (the relaxed latency plus one relaxed interval), negative for every stop condition;
    - ``pending_timeout``: the oldest frame in flight is older than the relaxed latency
      threshold (stop conditions with latency), negative;
    - ``latency_violation``: the window maximum of the latency exceeds its threshold (stop
      conditions with latency; latency never decreases under the paced input), negative;
    - ``window_slow``: the window is full and even its minimum interval exceeds the relaxed
      interval threshold (stop conditions with interval), negative;
    - ``window_clean``: the window is full and within the thresholds of the stop condition
      (for the target measurement, which has no thresholds yet: the interval is constant over
      the window), positive;
    - ``cap_expired``: none of the above within the wall-clock cap ``cap_s``; the result is
      ``undecided`` and therefore never feasible.

    The window is full once it holds ``window_frames`` completed frames after the first frame
    of the run, whose interval the hardware excludes. A warm-up transient therefore leaves
    the window by itself; no frame count is assumed. Only ``window_slow`` and ``window_clean``
    depend on the window length: a periodic pattern or a late transient longer than the
    window is misjudged, in the positive case by at most one slow frame per window. The
    measured interval and latency of a decided result are the window maxima.

    Without the registers (or without a window) the observation is the fixed sleep of
    ``iteration_runtime`` seconds and one read (``fixed_runtime``), exactly as before.
    """

    def __init__(
        self,
        overlay,
        iteration_runtime,
        throttle_interval,
        hw_capabilities,
        window_frames=None,
        poll_interval_s=0.0003,
        cap_s=None,
        feasibility=None,
    ):
        """Bind to an overlay, the fixed-runtime fallback, the input throttle and the window.

        ``cap_s`` is the wall-clock cap of a polled evaluation (see ``evaluation_cap``); it
        defaults to ``DEFAULT_CAP_FACTOR`` times the fixed runtime. ``feasibility`` (a
        ``FeasibilityCheck``) provides the thresholds of the rules; it is ``None`` for the
        target measurement and set afterwards.
        """
        self.overlay = overlay
        self.num_fifos = overlay.num_fifos
        self.iteration_runtime = iteration_runtime
        self.throttle_interval = throttle_interval
        self.hw_capabilities = hw_capabilities
        self.window_frames = None if window_frames is None else int(window_frames)
        self.poll_interval_s = poll_interval_s
        self.cap_s = iteration_runtime * DEFAULT_CAP_FACTOR if cap_s is None else float(cap_s)
        self.feasibility = feasibility
        self.num_evaluations = 0

    @property
    def polling(self):
        """Whether evaluations poll the steady-state registers instead of sleeping."""
        return self.window_frames is not None and bool(
            self.hw_capabilities.get("steady_state_registers", False)
        )

    def _observe_fixed(self):
        """Old observation: sleep for the iteration runtime and read the last sample once."""
        time.sleep(self.iteration_runtime)
        return self._read_last_sample("fixed_runtime")

    def _read_last_sample(self, decided_by, frames=None):
        """Single read of the instrumentation, decided as in the original binary search."""
        (
            overflow_err,
            underflow_err,
            frame,
            checksum,
            min_latency,
            latency,
            interval,
            *_,
        ) = self.overlay.observe_instrumentation(False)
        deadlock = interval == 0 or bool(overflow_err) or bool(underflow_err)
        return dict(
            interval=interval,
            latency=latency,
            min_latency=min_latency,
            deadlock=deadlock,
            decided_by=decided_by,
            frames=frames,
        )

    def _observe_polling(self):
        """Poll the steady-state registers until a rule decides the evaluation."""
        ov = self.overlay
        fb = self.feasibility
        window = self.window_frames
        check_interval = fb is not None and fb.stop_condition in ("interval", "both")
        check_latency = fb is not None and fb.stop_condition in ("latency", "both")
        deadline = time.monotonic() + self.cap_s
        decided_by = None
        s = None
        while True:
            s = ov.observe_steady_state()
            frames = s["run_frames"]
            # the first frame of a run has no valid interval: the window is full once it holds
            # ``window`` frames after it
            window_full = frames >= window + 1 and s["avg_fill"] >= window
            if s["overflow"] or s["underflow"]:
                decided_by = "instrumentation_error"
            elif fb is not None and frames >= 1 and s["idle_cycles"] > fb.stall_bound:
                decided_by = "idle_timeout"
            elif check_latency and s["oldest_pending_age"] > fb.relaxed_latency_threshold:
                decided_by = "pending_timeout"
            elif check_latency and s["lat_win_max"] > fb.relaxed_latency_threshold:
                decided_by = "latency_violation"
            elif window_full:
                if check_interval and s["int_win_min"] > fb.relaxed_interval_threshold:
                    decided_by = "window_slow"
                elif fb is None:
                    if s["int_win_max"] == s["int_win_min"]:
                        decided_by = "window_clean"
                elif not check_interval or s["int_win_max"] <= fb.relaxed_interval_threshold:
                    decided_by = "window_clean"
            if decided_by is None and time.monotonic() >= deadline:
                decided_by = "cap_expired"
            if decided_by is not None:
                break
            time.sleep(self.poll_interval_s)

        min_latency = ov.instrumentation_read("min_latency")
        frames = s["run_frames"]
        # Window maxima as the measured values; a timeout reports the bound it has observed
        # (the next interval / the pending frame's latency is at least that long)
        interval = s["int_win_max"]
        latency = s["lat_win_max"]
        if decided_by in ("idle_timeout", "cap_expired"):
            interval = max(interval, s["idle_cycles"])
        if decided_by in ("pending_timeout", "cap_expired"):
            latency = max(latency, s["oldest_pending_age"])
        deadlock = frames == 0 or decided_by == "instrumentation_error"
        return dict(
            interval=interval,
            latency=latency,
            min_latency=min_latency,
            deadlock=deadlock,
            decided_by=decided_by,
            frames=frames,
            undecided=decided_by == "cap_expired",
        )

    def evaluate(self, depths, need_flags=False, need_peaks=False):
        """Measure one bounded configuration on the FPGA."""
        ov = self.overlay
        depths = [int(d) for d in depths]
        ov.reset_accelerator()
        # every virtual FIFO gets the capacity fifo.sv builds for the requested depth
        ov.configure_fifos_bounded(ov.capacity_registers(depths))
        t0 = time.monotonic()
        ov.start_accelerator(
            throttle_interval=self.throttle_interval,
            avg_window_size=self.window_frames if self.polling else DEFAULT_HW_WINDOW_MAX,
        )
        observed = self._observe_polling() if self.polling else self._observe_fixed()
        ov.stop_accelerator()
        eval_time_s = time.monotonic() - t0
        self.num_evaluations += 1

        producer_blocked = None
        stall_flags = None
        flags_reliable = True
        if need_flags:
            # Registers are only cleared by RUN_* commands, so reading them after the run is safe.
            stall_flags = tuple(ov.read_stall_flags())
            sticky = bool(self.hw_capabilities.get("sticky_stall_flags", False))
            producer_blocked = tuple(producer_blocked_from_stall(f, sticky) for f in stall_flags)
            flags_reliable = sticky
        peak = None
        if need_peaks and self.hw_capabilities.get("peak_occupancy", False):
            peak = tuple(ov.read_peak_occupancy())

        return EvalResult(
            depths=tuple(depths),
            producer_blocked=producer_blocked,
            stall_flags=stall_flags,
            peak=peak,
            flags_reliable=flags_reliable,
            eval_time_s=eval_time_s,
            **observed,
        )


class FINNLiveFIFOOverlay(FINNInstrumentationOverlay):
    """FINN overlay for live FIFO sizing."""

    def __init__(
        self,
        bitfile_name: str,
        platform: str = "zynq-iodma",
        fclk_mhz: float = 100.0,
        device: Any = None,
        download: bool = True,
        seed: int = 1,
        fifo_widths: dict | None = None,
        folding_config: dict | None = None,
        fifo_cost_tables: dict | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize live FIFO overlay.

        ``fifo_widths``, ``folding_config`` and ``fifo_cost_tables`` come from the
        ``driver_information`` of ``settings.json`` (written by ``MakePYNQDriver``); the cost
        tables carry the FPGA primitive cost ladder of every FIFO for ``cost_model="primitives"``
        and the primitive report of the final configuration.
        """
        super().__init__(
            bitfile_name,
            platform=platform,
            fclk_mhz=fclk_mhz,
            seed=seed,
            download=download,
            device=device,
        )

        self.error = False
        self.fifo_widths = {} if fifo_widths is None else fifo_widths
        self.num_fifos = len(self.fifo_widths)
        self.fifo_cost_tables = fifo_cost_tables

        # The settings preserve the folding configuration needed for a
        # follow-up build with FIFO depths measured from this exact design.
        self.folding_config = folding_config

        # Account for additional FIFO depth or implicit registers introduced by the virtual FIFO
        # implementation that are not present in real FIFOs.
        # This results in a minimum possible FIFO depth of 1 + 1 = 2.
        self.fifo_depth_offset = 1

        # The capacity fifo.sv builds for a requested depth, per FIFO (padded stream width and
        # ram_style from the cost tables; without tables the data width rounded up to bytes
        # and ram_style "auto"). The virtual FIFOs are configured with it, see
        # capacity_registers(), and the storage of a configuration is reported with it.
        self.capacity_model = self.make_capacity_model()

        # Capabilities of the virtual FIFO hardware, probed in experiment_fifosizing:
        # - peak_occupancy: READ_FILL reports the peak occupancy of a bounded run
        # - sticky_stall_flags: READ_STALL reports run-sticky stall flags in bits [7:4]
        # - steady_state_registers: the instrumentation wrapper has the sliding-window
        #   extremes, idle_cycles and oldest_pending_age registers
        self.hw_capabilities = {
            "peak_occupancy": False,
            "sticky_stall_flags": False,
            "steady_state_registers": False,
        }

        # Sanity check
        # We expect 4 AXI-Lite peripherals:
        # fifo_controller_0, instrumentation_wrap_0, axi_gpio_0 (for reset), zynq_ps
        # We expect no additional FINN SDPs with AXI-Lite, such as runtime-writable weights
        if len(self.ip_dict.keys()) != 4:
            print(
                f"Error: # of AXI-Lite interfaces ({len(self.ip_dict)}) "
                f"does not match expected number of 4."
            )
            self.error = True
        if "fifo_controller_0" not in self.ip_dict:
            print("Error: fifo_controller_0 AXI-Lite interface not found.")
            self.error = True

    def ctrl_read(
        self, opcode: int = 0x00, fifo_id: int = 0x0000, check_success: bool = False
    ) -> int:
        """Read a value from the FIFO controller via AXI-Lite."""
        address = (fifo_id << 8) | opcode
        # Shift by 2 because FIFO controller operates on word addresses
        response = self.fifo_controller_0.read(offset=(address << 2))
        if check_success and response != opcode:
            print(
                f"Error: FIFO controller returned 0x{response:02x} "
                f"instead of expected 0x{opcode:02x}."
            )
            self.error = True
        return response

    def ctrl_write(
        self, opcode: int = 0x00, fifo_id: int = 0x0000, value: int = 0x00000000
    ) -> None:
        """Write a value to the FIFO controller via AXI-Lite."""
        address = (fifo_id << 8) | opcode
        # Shift by 2 because FIFO controller operates on word addresses
        self.fifo_controller_0.write(offset=(address << 2), value=value)

    def ctrl_set_depth(self, fifo_id: int, depth: int = 2) -> None:
        """Set FIFO depth via WRITE_FILL instruction."""
        # Issue WRITE_FILL instruction (asynchronous, returns immediately)
        self.ctrl_write(opcode=OP_WRITE_FILL, fifo_id=fifo_id, value=depth)
        # Read to confirm controller has returned to idle state
        self.ctrl_read(check_success=True)

    def configure_fifos_bounded(self, depths: int | list[int]) -> None:
        """Configure all FIFOs with bounded depths.

        Caller can supply a list of depths or a single depth for all FIFOs.
        """
        if isinstance(depths, (list, tuple)):
            fifo_depths = list(depths)
        else:
            fifo_depths = [depths] * self.num_fifos

        # Set depth for each FIFO
        for i in range(self.num_fifos):
            self.ctrl_set_depth(i, fifo_depths[i])

        # Issue RUN_BOUNDED instruction once all depths have been set
        self.ctrl_read(opcode=OP_RUN_BOUNDED, check_success=True)

    def read_peak_occupancy(self):
        """Read the peak occupancy of every FIFO since the last RUN_* command (READ_FILL)."""
        return [self.ctrl_read(opcode=OP_READ_FILL, fifo_id=i) for i in range(self.num_fifos)]

    def read_stall_flags(self):
        """Read the READ_STALL status byte of every FIFO.

        Bits [3:0] are latched per frame, bits [7:4] accumulate since the last RUN_* command
        (only with the extended gauge, see ``hw_capabilities``).
        """
        return [
            self.ctrl_read(opcode=OP_READ_STALL, fifo_id=i) & 0xFF for i in range(self.num_fifos)
        ]

    def probe_hw_capabilities(self):
        """Detect the virtual FIFO extensions from the stall flags after a paced run.

        In paced mode every gauge is always ready and sees no valid input at least until the
        pipeline fills, so the run-sticky "producer idle" flag (bit 4) must be set for every
        FIFO if the extended gauge and the fixed controller are in the bitfile. An old controller
        returns 0 for READ_STALL, an old gauge leaves bits [7:4] clear. Must be called after
        ``run_paced`` and before the next RUN_* command.
        """
        flags = self.read_stall_flags()
        sticky = all((f >> STALL_STICKY_SHIFT) & STALL_IVLD for f in flags)
        self.hw_capabilities = {
            "peak_occupancy": sticky,
            "sticky_stall_flags": sticky,
            "steady_state_registers": self.has_steady_state_registers(),
        }
        print(
            "HW CAPABILITIES: %s (stall flags after paced run: %s)"
            % (self.hw_capabilities, ["0x%02x" % f for f in flags])
        )
        return self.hw_capabilities

    def run_detached(self) -> int:
        """Run FIFOs in detached mode to determine bottleneck period."""
        self.reset_accelerator()

        # Issue RUN_DETACHED4 instruction
        self.ctrl_read(opcode=OP_RUN_DETACHED4, check_success=True)
        print("DEBUG: RUN_DETACHED4 completed")

        # Wait on detached run to complete by issuing BARRIER_CLEAN. Internally, the
        # controller re-issues this instruction until every gauge reports clean, i.e. has
        # produced its detached output frames; the read blocks until then. The input FIFO's
        # detached consumption is not awaited, so the instrumentation source need not run.
        self.ctrl_read(opcode=OP_BARRIER_CLEAN, check_success=True)
        print("DEBUG: BARRIER_CLEAN completed")

        # Issue COMP_PERIOD instruction to collect global max period across all FIFOs
        max_period = self.ctrl_read(opcode=OP_COMP_PERIOD)
        print("DEBUG: COMP_PERIOD completed")
        return max_period

    def run_paced(
        self, throttle_interval: int = 0, runtime_s: float = 1
    ) -> tuple[list[int], int, int]:
        """Run FIFOs in paced mode to determine the unbounded peak occupancies.

        Returns the peak occupancy of every FIFO and the latency and interval of the last
        completed frame.
        """
        self.reset_accelerator()

        # Issue RUN_PACED instruction
        self.ctrl_read(opcode=OP_RUN_PACED, check_success=True)

        # Let accelerator run for specified wallclock time
        self.start_accelerator(throttle_interval=throttle_interval)
        time.sleep(runtime_s)
        (
            _overflow_err,
            _underflow_err,
            _frame,
            _checksum,
            _min_latency,
            latency,
            interval,
            *_,
        ) = self.observe_instrumentation(debug_print=True)
        self.stop_accelerator()

        # Collect maximum occupancy of all FIFOs by issuing READ_FILL instructions
        max_occupancy = self.read_peak_occupancy()

        return max_occupancy, latency, interval

    def make_capacity_model(self):
        """The ``CapacityModel`` of this design from ``fifo_cost_tables`` and ``fifo_widths``.

        ``None`` without FIFOs (e.g. an emulated overlay that sets its own model).
        """
        if self.num_fifos == 0:
            return None
        tables = (self.fifo_cost_tables or {}).get("fifos", {})
        padded, styles = [], []
        for i in range(self.num_fifos):
            entry = tables.get(str(i), {})
            width = self.fifo_widths[str(i)]
            padded.append(int(entry.get("width", (int(width) + 7) // 8 * 8)))
            styles.append(entry.get("ram_style", "auto"))
        return CapacityModel(padded, styles)

    def capacity_registers(self, depths):
        """Capacity register values of a configuration: what ``fifo.sv`` builds for the depths.

        Register values stay register values (capacity minus the depth offset); an unbounded
        FIFO stays unbounded. Without a capacity model the depths are passed through.
        """
        if self.capacity_model is None:
            return [int(d) for d in depths]
        off = self.fifo_depth_offset
        return [
            int(d) if d >= UNBOUNDED_DEPTH else self.capacity_model.capacity(i, int(d) + off) - off
            for i, d in enumerate(depths)
        ]

    def fifo_capacities(self, depths):
        """Items every FIFO of a configuration (register values) really holds."""
        return [r + self.fifo_depth_offset for r in self.capacity_registers(depths)]

    def total_fifo_size(self, depths: list[int], requested: bool = False) -> float:
        """Total FIFO storage of a configuration (register values) in kB.

        The realized storage, capacity times data width, unless ``requested`` asks for the
        storage of the requested depths.
        """
        # Assuming FIFO SDP/AXI-Lite interfaces are ordered consistently with FIFO IDs
        total_size_bits = 0
        items = (
            [d + self.fifo_depth_offset for d in depths]
            if requested
            else self.fifo_capacities(depths)
        )
        for i, n in enumerate(items):
            total_size_bits += n * self.fifo_widths[str(i)]
        return total_size_bits / 8.0 / 1000.0

    def fifo_width_list(self):
        """FIFO widths in bits as a list indexed by FIFO ID."""
        return [self.fifo_widths[str(i)] for i in range(self.num_fifos)]

    def primitive_cost_model(self):
        """The ``PrimitiveCost`` of this accelerator, or ``None`` without cost tables."""
        if self.fifo_cost_tables is None:
            return None
        cost_model = PrimitiveCost(self.fifo_cost_tables, offset=self.fifo_depth_offset)
        if cost_model.num_fifos != self.num_fifos:
            raise ValueError(
                "fifo_cost_tables in settings.json cover %d FIFOs, the design has %d"
                % (cost_model.num_fifos, self.num_fifos)
            )
        return cost_model

    def make_cost_model(self, cost_model, start_depths):
        """The cost model and lattice of a search: ``"bits"`` (kB of realized storage, the
        capacity ``fifo.sv`` builds times the data width, candidate depths at the tops of the
        capacity plateaus) or ``"primitives"`` (FPGA primitives from the exported tables,
        candidate depths at the steps of the cost ladder). The lattice spans ``[min, start]``
        per FIFO, ``min`` being the top of the shift-register floor (depth 5) with a capacity
        model and 1 otherwise, like the default of ``SearchContext``; the exact searches widen
        it to their box."""
        lower = [1] * self.num_fifos
        if cost_model == "bits":
            model = TotalBitsCost(
                self.fifo_width_list(),
                offset=self.fifo_depth_offset,
                capacities=self.capacity_model,
            )
            lower = [model.min_depth(i) for i in range(self.num_fifos)]
            # plateau tops up to four start depths: the box of the exact searches (default
            # upper_bound_factor 2); beyond, the lattice continues the last plateau width
            candidates = [
                model.breakpoints(i, max(4 * int(d), lower[i] + 1))
                for i, d in enumerate(start_depths)
            ]
            if all(c is None for c in candidates):
                candidates = None
            return model, Lattice(lower, start_depths, candidates=candidates)
        if cost_model == "primitives":
            model = self.primitive_cost_model()
            if model is None:
                raise ValueError(
                    "cost_model='primitives' needs fifo_cost_tables in settings.json "
                    "(written by builds with the current FINN+ version)"
                )
            candidates = [model.breakpoints(i) for i in range(self.num_fifos)]
            return model, Lattice(lower, start_depths, candidates=candidates)
        raise ValueError(f"Unknown cost_model: {cost_model} (available: bits, primitives)")

    def primitive_report(self, depths):
        """Primitive vector of a configuration (register values), ``None`` without tables."""
        model = self.primitive_cost_model()
        if model is None:
            return None
        realized = model.realize(depths)
        report = {
            **model.describe(),
            "total": model.vector(realized),
            "scalar": model.cost(realized),
            "fifos": {
                str(i): {
                    "depth": realized[i] + self.fifo_depth_offset,
                    "ram_style": model.ram_styles[i],
                    **dict(zip(model.resources, model.vector_fifo(i, realized[i]))),
                }
                for i in range(self.num_fifos)
            },
        }
        return report

    def run_fifo_search(
        self,
        search_strategy,
        start_depth,
        iteration_runtime,
        throttle_interval=0,
        stop_condition="both",
        relaxation=0.0,
        max_evaluations=None,
        max_runtime_s=None,
        window_frames=None,
        poll_interval_s=0.0003,
        evaluation_cap_s=None,
        evaluation_cap_frames=None,
        cost_model="bits",
        **strategy_kwargs,
    ):
        """Run a FIFO depth search strategy on the hardware oracle.

        Parameters
        ----------
        search_strategy : str
            Name from ``fifo_search.SEARCH_STRATEGIES``: ``binary_<order>`` for the per-FIFO
            binary search (orders: forward, reverse, largest_first, deepest_first, alternating,
            random), ``bottom_up_exact`` or ``top_down_bnb``.
        start_depth : int or list
            Initial depth(s) for FIFOs; the performance targets are measured here.
        iteration_runtime : float
            Runtime of each test iteration in seconds without steady-state detection (fixed
            sleep and one read)
        throttle_interval : int
            Throttle interval in cycles
        stop_condition : str
            Metric to use for determining if a FIFO depth is too small. Options:
            - "interval": Stop if interval degrades from target_interval
            - "latency": Stop if latency degrades from target_latency
            - "both": Stop if either interval or latency degrades
        relaxation : float
            Allowed degradation tolerance (0.0 to 1.0, where 1.0 = 100% degradation allowed).
            Default 0.0 means no degradation allowed.
        max_evaluations, max_runtime_s :
            Budget of the search (hardware evaluations / wall-clock seconds), None = unlimited.
        window_frames : int or None
            Frames of a clean sliding window that count as steady state (``LiveFIFOOracle``;
            written to the instrumentation as its window size, which the hardware clamps to
            the build's ``instrumentation_avg_n``); None keeps the fixed-runtime observation.
            Requires the steady-state registers in the bitfile.
        poll_interval_s, evaluation_cap_s, evaluation_cap_frames :
            Poll period in seconds and the wall-clock cap of one polled evaluation (see
            ``evaluation_cap``; the frame budget it was derived from is only reported). An
            evaluation that reaches the cap is rejected; the target measurement reaching it
            aborts the search with a ``RuntimeError``.
        cost_model : str
            ``"bits"`` minimises the realized storage in kB (the capacity ``fifo.sv`` builds
            for a depth times the data width) over the tops of the capacity plateaus;
            ``"primitives"`` minimises the FPGA primitives of the RTL FIFO (LUT equivalents of
            the device capacity, from the ``fifo_cost_tables`` of ``settings.json``) over the
            depths at which that cost changes. ``realized_fifo_depths`` are the (deeper or
            equal) depths that attain the cost of the found configuration. Every evaluated
            configuration is applied to the virtual FIFOs with the capacities ``fifo.sv``
            would build for it (``capacity_registers``).
        strategy_kwargs :
            Passed on to the strategy (seed_strategy, upper_bound_factor, node_order, pareto,
            pareto_cost_limit, allow_unreliable_flags).

        Returns a dict with the keys of the original binary search (duration, fifo_depths,
        log_total_fifo_size, log_interval, log_min_latency, log_latency, all_iterations) plus
        ``search`` (the strategy summary), ``result`` (the ``SearchResult``), ``cost_model``
        (its name and the model object) and ``realized_fifo_depths``.
        """
        if isinstance(start_depth, (list, tuple)):
            start_depths = tuple(int(d) for d in start_depth)
        else:
            start_depths = tuple([int(start_depth)] * self.num_fifos)

        widths = self.fifo_width_list()
        cost_model_name = cost_model
        cost_model, lattice = self.make_cost_model(cost_model_name, start_depths)
        # a start depth below the lattice (a peak occupancy under the shift-register floor)
        # builds the same FIFO as the floor: search from there
        start_depths = tuple(max(d, lo) for d, lo in zip(start_depths, lattice.lower))
        oracle = LiveFIFOOracle(
            self,
            iteration_runtime,
            throttle_interval,
            self.hw_capabilities,
            window_frames=window_frames,
            poll_interval_s=poll_interval_s,
            cap_s=evaluation_cap_s,
        )
        cache = EvaluationCache(oracle, cost_model)
        if window_frames is not None and not oracle.polling:
            print("STEADY STATE DETECTION: registers not in the bitfile, using the fixed runtime")
        elif oracle.polling:
            if window_frames > DEFAULT_HW_WINDOW_MAX:
                warnings.warn(
                    f"window_frames={window_frames} exceeds the default hardware window of "
                    f"{DEFAULT_HW_WINDOW_MAX} frames (build option instrumentation_avg_n); a "
                    "clamped window never fills and every evaluation is rejected at the cap"
                )
            print(
                "STEADY STATE DETECTION: window %d frames, poll %.0f us, cap %.3f s%s"
                % (
                    window_frames,
                    poll_interval_s * 1e6,
                    oracle.cap_s,
                    "" if evaluation_cap_frames is None else " (%d frames)" % evaluation_cap_frames,
                )
            )

        # Run once at the start configuration to determine the targets (iteration "0"); the
        # thresholds derived from it enable the early-exit rules of the oracle
        start_time = time.time()
        target = cache.evaluate(start_depths, role="target")
        if target.result.undecided or target.result.deadlock:
            raise RuntimeError(
                "the start configuration reached no steady state within the evaluation cap "
                f"({oracle.cap_s:.3f} s, {target.result.frames} frames, "
                f"{target.result.decided_by}): no target can be derived"
            )
        if oracle.polling and target.result.interval != throttle_interval:
            warnings.warn(
                f"the paced start configuration runs at {target.result.interval} cycles per "
                f"frame instead of the detached maximum period {throttle_interval} it is paced "
                "at; the measured value is the target"
            )
        feasibility = FeasibilityCheck.from_result(stop_condition, relaxation, target.result)
        oracle.feasibility = feasibility
        budget = SearchBudget(max_evaluations=max_evaluations, max_runtime_s=max_runtime_s)
        ctx = SearchContext(
            oracle,
            feasibility,
            cost_model,
            start_depths,
            widths,
            cache=cache,
            lattice=lattice,
            budget=budget,
            log=print,
            rng=random,
        )
        result = run_search(search_strategy, ctx, **strategy_kwargs)
        duration = int(time.time() - start_time)
        print(f"Done ({duration} seconds)")
        if result.abort_reason:
            print(f"SEARCH ABORTED: {result.abort_reason}")
        if result.budget_exhausted:
            print("SEARCH BUDGET EXHAUSTED: reporting the best configuration found so far")

        # Log of the successful configurations (target measurement + every PASS with a finite
        # cost), as plotted by the original binary search graph
        log_total_fifo_size, log_interval, log_min_latency, log_latency = [], [], [], []
        for entry in cache.log:
            if entry.index == 0 or (
                entry.info.get("status") == "PASS" and math.isfinite(entry.cost)
            ):
                log_total_fifo_size.append(entry.cost)
                log_interval.append(entry.result.interval)
                log_min_latency.append(entry.result.min_latency)
                log_latency.append(entry.result.latency)

        return {
            "duration": duration,
            "fifo_depths": list(result.depths),
            "realized_fifo_depths": list(cost_model.realize(result.depths)),
            "cost_model": cost_model_name,
            "cost_model_obj": cost_model,
            "log_total_fifo_size": log_total_fifo_size,
            "log_interval": log_interval,
            "log_min_latency": log_min_latency,
            "log_latency": log_latency,
            "all_iterations": cache_to_iterations(cache),
            "search": {
                **search_result_to_dict(result),
                "cost_model": cost_model_name,
                "stop_condition": stop_condition,
                "relaxation": relaxation,
                "iteration_runtime_s": iteration_runtime,
                "throttle_interval": throttle_interval,
                "target_interval_cycles": feasibility.target_interval,
                "target_latency_cycles": feasibility.target_latency,
                "relaxed_interval_threshold": feasibility.relaxed_interval_threshold,
                "relaxed_latency_threshold": feasibility.relaxed_latency_threshold,
                "hw_capabilities": dict(self.hw_capabilities),
                "num_cache_hits": cache.num_cache_hits,
                "steady_state_detection": oracle.polling,
                "window_frames": window_frames if oracle.polling else None,
                "poll_interval_s": poll_interval_s if oracle.polling else None,
                "evaluation_cap_s": oracle.cap_s if oracle.polling else None,
                "evaluation_cap_frames": evaluation_cap_frames if oracle.polling else None,
                "decisions": decision_counts(cache),
            },
            "result": result,
        }

    def size_iteratively_binary_search(
        self,
        start_depth: int | list[int],
        iteration_runtime: float,
        throttle_interval: int = 0,
        fifo_order_strategy: str = "largest_first",
        stop_condition: str = "both",
        relaxation: float = 0.0,
    ) -> dict:
        """Reduce FIFO depths iteratively via binary search to find the minimum for each FIFO.

        Kept for compatibility: this is ``run_fifo_search`` with the ``binary_<order>``
        strategy and returns the same dict as before (plus the ``search`` summary).
        """
        return self.run_fifo_search(
            f"binary_{fifo_order_strategy}",
            start_depth,
            iteration_runtime,
            throttle_interval=throttle_interval,
            stop_condition=stop_condition,
            relaxation=relaxation,
        )

    def generate_fifosizing_graph(
        self,
        log_total_fifo_size,
        log_min_latency,
        log_latency,
        log_interval,
        report_dir,
        stop_condition="interval",
        unit="kB",
    ):
        """Generate and save FIFO sizing visualization graph."""
        # Round total FIFO size to integer values
        log_total_fifo_size = [int(round(x)) for x in log_total_fifo_size]

        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel(f"Total FIFO cost [{unit}]", color=color)
        ax1.plot(range(len(log_total_fifo_size)), log_total_fifo_size, color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_xlim(left=0)
        ax1.set_ylim(0, max(log_total_fifo_size))

        if stop_condition == "interval":
            # Plot both latencies when optimizing for interval
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = "tab:blue"
            ax2.set_ylabel("Cycles", color=color)
            ax2.plot(
                range(len(log_total_fifo_size)),
                log_min_latency,
                color=color,
                label="First-frame latency",
            )
            ax2.plot(
                range(len(log_total_fifo_size)),
                log_latency,
                color="tab:green",
                label="Steady-state latency",
            )
            ax2.tick_params(axis="y", labelcolor=color)
            ax2.legend(loc="upper center")
        elif stop_condition == "latency":
            # Plot interval when optimizing for latency
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = "tab:orange"
            ax2.set_ylabel("Cycles", color=color)
            ax2.plot(
                range(len(log_total_fifo_size)),
                log_interval,
                color=color,
                label="Interval",
            )
            ax2.tick_params(axis="y", labelcolor=color)
            ax2.legend(loc="upper center")

        plt.tight_layout()
        plt.savefig(Path(report_dir) / "fifo_sizing_graph.png", dpi=300)
        plt.close(fig)

    def generate_search_graph(self, search_log, report_dir):
        """Plot the incumbent cost over the evaluations of an exact search and its lower bound."""
        result = search_log["result"]
        num_evaluations = max(1, result.num_evaluations)
        trace = list(result.incumbent_trace)
        if not trace or trace[0][0] > 0:
            trace.insert(0, (0, search_log["log_total_fifo_size"][0]))
        xs = [i for i, _ in trace] + [num_evaluations]
        ys = [c for _, c in trace] + [trace[-1][1]]

        fig, ax = plt.subplots()
        ax.step(xs, ys, where="post", color="tab:red", label="Incumbent (best feasible)")
        if result.lower_bounds_cost is not None and math.isfinite(result.lower_bounds_cost):
            ax.axhline(
                result.lower_bounds_cost,
                color="tab:gray",
                linestyle=":",
                label="Cost of per-FIFO lower bounds",
            )
        if result.lower_bound_cost is not None and math.isfinite(result.lower_bound_cost):
            ax.axhline(
                result.lower_bound_cost,
                color="tab:blue",
                linestyle="--",
                label="Proven lower bound",
            )
        ax.set_xlabel("Hardware evaluation")
        ax.set_ylabel(f"Total FIFO cost [{result.cost_unit}]")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_title(
            "%s: %s" % (result.strategy, "optimal proven" if result.optimal_proven else "gap open")
        )
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(Path(report_dir) / "fifo_sizing_graph.png", dpi=300)
        plt.close(fig)

    def experiment_fifosizing(self, *args, **kwargs):
        """Run live FIFO sizing experiment and save report.

        Kwargs: ``search_strategy`` (default ``binary_largest_first``; ``fifo_search_order``
        is accepted as a deprecated alias for ``binary_<order>``), ``stop_condition``,
        ``relaxation``, ``relaxation_sweep`` (binary strategies only), ``report_name`` (report
        sub-directory, default: the strategy name), ``max_evaluations``,
        ``max_runtime_s``, ``iteration_runtime_factor`` (safety factor on the per-evaluation
        runtime heuristic), ``hw_capabilities`` ("auto" or a bool overriding the probe of the
        virtual FIFO extensions), ``cost_model`` (``"bits"``, the default, or
        ``"primitives"``, see ``run_fifo_search``) and the strategy-specific
        ``seed_strategy``, ``upper_bound_factor``, ``node_order``, ``pareto``,
        ``pareto_cost_limit``, ``allow_unreliable_flags``.

        The exported ``fifo_config.json`` holds the realized depths (the tops of the capacity
        plateaus of ``fifo.sv`` for ``"bits"``, the depths attaining the cost of the found
        configuration for ``"primitives"``; never shallower). The report gives the storage
        of that configuration as realized bits (``fifo_size_total_kB``: the capacity every
        FIFO really gets, ``fifo_capacities``, times its data width), the storage of the
        depths the search found (``fifo_size_requested_kB``) and the primitive vector
        whenever cost tables are available.

        Steady-state detection (``LiveFIFOOracle``, needs the steady-state registers of the
        instrumentation wrapper; older bitfiles fall back to the fixed runtime):
        ``steady_state_detection`` ("auto", or False to force the fixed runtime),
        ``window_frames`` (frames of a clean sliding window that count as steady state,
        default ``DEFAULT_WINDOW_FRAMES``; at most the build's ``instrumentation_avg_n``),
        ``poll_interval_s`` (default 300 us), ``cap_factor`` and ``cap_floor_s`` (wall-clock
        cap of an evaluation, see ``evaluation_cap``; defaults 4 and 2 ms). The target
        interval and latency are the window maxima of the clean window of the paced start
        configuration, so they are free of the warm-up transient.
        """
        fifo_search_order = kwargs.get("fifo_search_order")
        if fifo_search_order is not None:
            warnings.warn(
                "fifo_search_order is deprecated, use search_strategy='binary_<order>'",
                DeprecationWarning,
            )
        search_strategy = resolve_strategy_name(kwargs.get("search_strategy"), fifo_search_order)
        stop_condition = kwargs.get("stop_condition", "both")
        relaxation = kwargs.get("relaxation", 0.0)
        relaxation_sweep = kwargs.get("relaxation_sweep", False)
        max_evaluations = kwargs.get("max_evaluations")
        max_runtime_s = kwargs.get("max_runtime_s")
        iteration_runtime_factor = kwargs.get("iteration_runtime_factor", 1.0)
        hw_capabilities = kwargs.get("hw_capabilities", "auto")
        steady_state_detection = kwargs.get("steady_state_detection", "auto")
        steady_state_kwargs = {k: kwargs[k] for k in ("poll_interval_s",) if k in kwargs}
        cost_model = kwargs.get("cost_model", "bits")
        strategy_kwargs = {k: kwargs[k] for k in STRATEGY_KWARGS if k in kwargs}
        is_binary = search_strategy.startswith("binary_")
        base_report_dir = kwargs.get("report_dir")
        # Create subdirectory for this search strategy + stop condition. report_name lets
        # several variants of one strategy (e.g. different seeds) keep separate reports.
        report_name = kwargs.get("report_name") or search_strategy
        report_dir = str(Path(base_report_dir) / report_name / stop_condition)
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        reportfile = Path(report_dir) / "report_experiment_fifosizing.json"

        print("---PHASE 1: RUN_DETACHED---")
        max_period = self.run_detached()
        print(f"MEASURED MAX PERIOD: {max_period} cycles")

        print("---PHASE 2: RUN_PACED---")
        # TODO: Use better heuristic for runtime?
        max_occupancy, paced_latency, paced_interval = self.run_paced(
            throttle_interval=max_period, runtime_s=1
        )
        print("MEASURED MAX FIFO OCCUPANCIES:")
        print("FIFO ID | MAX OCCUPANCY")
        for fifo_id, occupancy in enumerate(max_occupancy):
            print(f"{fifo_id:7} | {occupancy:13}")
        print(
            "TOTAL FIFO SIZE @ MAX OCCUPANCY (kB): %f realized (%f requested)"
            % (self.total_fifo_size(max_occupancy), self.total_fifo_size(max_occupancy, True))
        )
        # Detect the virtual FIFO extensions (peak occupancy, run-sticky stall flags)
        if hw_capabilities == "auto":
            self.probe_hw_capabilities()
        else:
            self.hw_capabilities = {
                "peak_occupancy": bool(hw_capabilities),
                "sticky_stall_flags": bool(hw_capabilities),
                "steady_state_registers": self.has_steady_state_registers(),
            }
            print("HW CAPABILITIES (from kwargs): %s" % self.hw_capabilities)
        if steady_state_detection != "auto" and not steady_state_detection:
            self.hw_capabilities["steady_state_registers"] = False
            print("STEADY STATE DETECTION: disabled by kwargs")

        print("---PHASE 3: ITERATIVE MINIMIZATION---")
        print("SEARCH STRATEGY: %s" % search_strategy)
        print("STOP CONDITION: %s" % stop_condition)
        print("RELAXATION: %.1f%%" % (relaxation * 100))
        print("RELAXATION SWEEP: %s" % ("Enabled" if relaxation_sweep else "Disabled"))
        print("COST MODEL: %s" % cost_model)
        if strategy_kwargs:
            print("STRATEGY KWARGS: %s" % strategy_kwargs)
        if max_evaluations is not None or max_runtime_s is not None:
            print("BUDGET: %s evaluations, %s s" % (max_evaluations, max_runtime_s))
        # Determine search iteration runtime via heuristic based on free-running latency
        iteration_runtime = max(0.001, (paced_latency * 4) * 10 / 1000 / 1000 / 1000)
        iteration_runtime *= iteration_runtime_factor
        # Steady-state detection: window length and the evaluation cap from the paced run
        window_frames = int(kwargs.get("window_frames", DEFAULT_WINDOW_FRAMES))
        cap_frames, cap_s = evaluation_cap(
            paced_latency,
            paced_interval,
            window_frames,
            self.fclk_mhz_actual,
            kwargs.get("cap_factor", DEFAULT_CAP_FACTOR),
            kwargs.get("cap_floor_s", DEFAULT_CAP_FLOOR_S),
        )
        steady_state_kwargs["window_frames"] = window_frames
        steady_state_kwargs["evaluation_cap_s"] = cap_s
        steady_state_kwargs["evaluation_cap_frames"] = cap_frames
        print(
            "STEADY STATE WINDOW: %d frames, evaluation cap %d frames / %.3f s (paced latency "
            "%d, interval %d cycles)"
            % (window_frames, cap_frames, cap_s, paced_latency, paced_interval)
        )

        search_log = self.run_fifo_search(
            search_strategy,
            start_depth=max_occupancy,
            iteration_runtime=iteration_runtime,
            throttle_interval=max_period,
            stop_condition=stop_condition,
            relaxation=relaxation,
            max_evaluations=max_evaluations,
            max_runtime_s=max_runtime_s,
            **steady_state_kwargs,
            cost_model=cost_model,
            **strategy_kwargs,
        )

        # The depths to build: the found configuration, lifted onto the tops of the capacity
        # plateaus (bits) or the cost envelope (primitives); equal or deeper, so still feasible
        fifo_depths = search_log["realized_fifo_depths"]
        requested_depths = list(search_log["fifo_depths"])
        cost_unit = search_log["cost_model_obj"].unit
        log_total_fifo_size = search_log["log_total_fifo_size"]
        log_interval = search_log["log_interval"]
        log_min_latency = search_log["log_min_latency"]
        log_latency = search_log["log_latency"]
        result = search_log["result"]

        # Generate visualization graph
        if is_binary:
            self.generate_fifosizing_graph(
                log_total_fifo_size,
                log_min_latency,
                log_latency,
                log_interval,
                report_dir,
                stop_condition,
                unit=cost_unit,
            )
        else:
            self.generate_search_graph(search_log, report_dir)

        # Final performance: the measurement of the final configuration if it was evaluated,
        # otherwise the last successful configuration of the log
        target_interval = log_interval[0]
        target_latency = log_latency[0]
        if result.result is not None:
            final_interval = result.result.interval
            final_latency = result.result.latency
            final_min_latency = result.result.min_latency
        else:
            final_interval = log_interval[-1]
            final_latency = log_latency[-1]
            final_min_latency = log_min_latency[-1]
        final_total_fifo_size = self.total_fifo_size(fifo_depths)
        final_capacities = self.fifo_capacities(fifo_depths)

        interval_degradation = (
            (final_interval - target_interval) / target_interval if target_interval != 0 else 0
        )
        latency_degradation = (
            (final_latency - target_latency) / target_latency if target_latency != 0 else 0
        )

        # Relaxation sweep: explore additional relaxation values if enabled
        relaxation_sweep_results = []
        if relaxation_sweep and not is_binary:
            warnings.warn(
                "relaxation_sweep only applies to the binary strategies; use pareto=True "
                "with bottom_up_exact to record the cost/performance trade-off"
            )
        elif relaxation_sweep:
            print("---RELAXATION SWEEP---")
            # Pre-defined sequence of relaxation values to explore
            relaxation_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
            # Filter out values <= current relaxation to avoid redundant searches
            relaxation_values = [r for r in relaxation_values if r > relaxation]

            for sweep_relaxation in relaxation_values:
                print(f"Testing relaxation: {sweep_relaxation:.2f} ({sweep_relaxation*100:.0f}%)")

                sweep_search_log = self.run_fifo_search(
                    search_strategy,
                    start_depth=max_occupancy,
                    iteration_runtime=iteration_runtime,
                    throttle_interval=max_period,
                    stop_condition=stop_condition,
                    relaxation=sweep_relaxation,
                    **steady_state_kwargs,
                    cost_model=cost_model,
                )

                # Extract only essential metrics
                sweep_log_total_fifo_size = sweep_search_log["log_total_fifo_size"]
                sweep_log_interval = sweep_search_log["log_interval"]
                sweep_log_latency = sweep_search_log["log_latency"]

                sweep_final_interval = sweep_log_interval[-1]
                sweep_final_latency = sweep_log_latency[-1]
                sweep_interval_degradation = (
                    (sweep_final_interval - target_interval) / target_interval
                    if target_interval != 0
                    else 0
                )
                sweep_latency_degradation = (
                    (sweep_final_latency - target_latency) / target_latency
                    if target_latency != 0
                    else 0
                )

                relaxation_sweep_results.append(
                    {
                        "relaxation": sweep_relaxation,
                        "fifo_size_total_kB": sweep_log_total_fifo_size[-1],
                        "interval_degradation": sweep_interval_degradation,
                        "latency_degradation": sweep_latency_degradation,
                        "final_interval_cycles": sweep_final_interval,
                        "final_latency_cycles": sweep_final_latency,
                    }
                )

                print(
                    f"  Result: FIFO size={sweep_log_total_fifo_size[-1]:.2f} kB, "
                    f"interval degradation={sweep_interval_degradation*100:.1f}%, "
                    f"latency degradation={sweep_latency_degradation*100:.1f}%"
                )

            print("RELAXATION SWEEP COMPLETE")

        # Generate fifo_sizing_report.json
        search_summary = search_log["search"]
        fifo_report = {
            "error": self.error,
            # realized storage: the capacity fifo.sv builds for every depth times the data width
            "fifo_size_total_kB": final_total_fifo_size,
            # storage of the depths the search found before lifting them onto the plateau tops
            "fifo_size_requested_kB": self.total_fifo_size(requested_depths, requested=True),
            "cost_model": cost_model,
            "cost_unit": cost_unit,
            "total_cost": search_log["cost_model_obj"].cost(fifo_depths),
            "primitive_cost": self.primitive_report(fifo_depths),
            "detached_max_period_cycles": max_period,
            "target_interval_cycles": target_interval,
            "final_interval_cycles": final_interval,
            "interval_degradation": interval_degradation,
            "target_latency_cycles": target_latency,
            "final_latency_cycles": final_latency,
            "latency_degradation": latency_degradation,
            "fifo_depths": {},
            "fifo_capacities": {},
            "fifo_sizes": {},
            "search": {
                **search_summary,
                "realized_fifo_depths": list(fifo_depths),
                "fifo_capacities": list(final_capacities),
                "all_iterations": search_log["all_iterations"],
            },
        }
        if is_binary:
            # Report block of the original binary search, unchanged for its consumers
            fifo_report["binary_search"] = {
                "search_order": search_strategy[len("binary_") :],
                "stop_condition": stop_condition,
                "relaxation": relaxation,
                "iteration_runtime_s": iteration_runtime,
                **{
                    k: v
                    for k, v in search_log.items()
                    if k not in ("search", "result", "cost_model_obj")
                },
            }

        # Add relaxation sweep results if available
        if relaxation_sweep_results:
            fifo_report["relaxation_sweep"] = relaxation_sweep_results
        for fifo, depth in enumerate(fifo_depths):
            fifo_report["fifo_depths"][fifo] = depth + self.fifo_depth_offset
            fifo_report["fifo_capacities"][fifo] = final_capacities[fifo]
            fifo_report["fifo_sizes"][fifo] = final_capacities[fifo] * self.fifo_widths[str(fifo)]
        with (Path(report_dir) / "fifo_sizing_report.json").open("w") as f:
            json.dump(fifo_report, f, indent=2)

        # Export measured FIFO settings for use by the follow-up FINN build.
        fifo_config = {
            "fifo_depths": {},
            "impl_style": {},
            "ram_style": {},
        }
        tables = (self.fifo_cost_tables or {}).get("fifos", {})
        for fifo, depth in enumerate(fifo_depths):
            fifo_name = f"StreamingFIFO_rtl_{fifo}"
            final_depth = depth + self.fifo_depth_offset
            fifo_config["fifo_depths"][fifo_name] = final_depth
            fifo_config["impl_style"][fifo_name] = "rtl"
            # the style the cost tables were computed for (the node's ram_style at build time)
            fifo_config["ram_style"][fifo_name] = tables.get(str(fifo), {}).get("ram_style", "auto")
        with (Path(report_dir) / "fifo_config.json").open("w") as f:
            json.dump(fifo_config, f, indent=2)

        if self.folding_config is None:
            raise ValueError("Live FIFO sizing requires folding_config in settings.json.")
        with (Path(report_dir) / "folding_config.json").open("w") as f:
            json.dump(self.folding_config, f, indent=2)

        # Generate the usual instrumentation performance report based on final state
        min_latency = final_min_latency
        latency = final_latency
        interval = final_interval
        report = {
            "error": self.error,
            "checksum": 0,
            "min_latency_cycles": min_latency,
            "latency_cycles": latency,
            "interval_cycles": interval,
            "frequency_mhz": round(self.fclk_mhz_actual),
            "min_latency_ms": round(min_latency * (1 / (self.fclk_mhz_actual * 1e6)) * 1e3, 6),
            "latency_ms": round(latency * (1 / (self.fclk_mhz_actual * 1e6)) * 1e3, 6),
            "throughput_fps": (
                round(1 / (interval * (1 / (self.fclk_mhz_actual * 1e6)))) if interval != 0 else 0
            ),
            "min_pipeline_depth": round(min_latency / interval, 2) if interval != 0 else 0,
            "pipeline_depth": round(latency / interval, 2) if interval != 0 else 0,
        }
        with reportfile.open("w") as f:
            json.dump(report, f, indent=2)

        print("Done.")

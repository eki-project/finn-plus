"""Driver for live FIFO sizing experiments."""

import json
import math
import matplotlib.pyplot as plt
import os
import random
import time
import warnings

from finn_plus_driver.overlays.fifo_search import (
    STALL_IVLD,
    STALL_STICKY_SHIFT,
    EvalResult,
    EvaluationCache,
    FeasibilityCheck,
    SearchBudget,
    SearchContext,
    TotalBitsCost,
    cache_to_iterations,
    producer_blocked_from_stall,
    resolve_strategy_name,
    run_search,
    search_result_to_dict,
)
from finn_plus_driver.overlays.instrumentation import FINNInstrumentationOverlay

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


class LiveFIFOOracle:
    """Hardware implementation of the ``Oracle`` protocol of ``fifo_search``.

    One evaluation is the inner loop of the original binary search: reset the accelerator,
    configure the bounded FIFO depths, run with the throttled input for ``iteration_runtime``
    seconds and read the instrumentation. Only when a strategy asks for them, the peak
    occupancies (``READ_FILL``) and the stall flags (``READ_STALL``) of every FIFO are read
    from the ring afterwards. Ring reads are non-intrusive; they cost a few ms per FIFO.
    """

    def __init__(self, overlay, iteration_runtime, throttle_interval, hw_capabilities):
        """Bind to an overlay, the per-evaluation runtime and the input throttle interval."""
        self.overlay = overlay
        self.num_fifos = overlay.num_fifos
        self.iteration_runtime = iteration_runtime
        self.throttle_interval = throttle_interval
        self.hw_capabilities = hw_capabilities
        self.num_evaluations = 0

    def evaluate(self, depths, need_flags=False, need_peaks=False):
        """Measure one bounded configuration on the FPGA."""
        ov = self.overlay
        depths = [int(d) for d in depths]
        ov.reset_accelerator()
        ov.configure_fifos_bounded(depths)
        ov.start_accelerator(throttle_interval=self.throttle_interval)
        time.sleep(self.iteration_runtime)
        (
            overflow_err,
            underflow_err,
            frame,
            checksum,
            min_latency,
            latency,
            interval,
            *_,
        ) = ov.observe_instrumentation(False)
        ov.stop_accelerator()
        self.num_evaluations += 1
        deadlock = interval == 0 or bool(overflow_err) or bool(underflow_err)

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
            interval=interval,
            latency=latency,
            min_latency=min_latency,
            deadlock=deadlock,
            producer_blocked=producer_blocked,
            stall_flags=stall_flags,
            peak=peak,
            flags_reliable=flags_reliable,
        )


class FINNLiveFIFOOverlay(FINNInstrumentationOverlay):
    """FINN overlay for live FIFO sizing."""

    def __init__(
        self,
        bitfile_name,
        platform="zynq-iodma",
        fclk_mhz=100.0,
        device=None,
        download=True,
        seed=1,
        fifo_widths=dict(),
        folding_config=None,
        **kwargs,
    ):
        """Initialize live FIFO overlay."""
        super().__init__(
            bitfile_name,
            platform=platform,
            fclk_mhz=fclk_mhz,
            seed=seed,
            download=download,
            device=device,
        )

        self.error = False
        self.fifo_widths = fifo_widths
        self.num_fifos = len(self.fifo_widths)

        # The settings preserve the folding configuration needed for a
        # follow-up build with FIFO depths measured from this exact design.
        self.folding_config = folding_config

        # Account for additional FIFO depth or implicit registers introduced by the virtual FIFO
        # implementation that are not present in real FIFOs.
        # This results in a minimum possible FIFO depth of 1 + 1 = 2.
        self.fifo_depth_offset = 1

        # Capabilities of the virtual FIFO hardware, probed in experiment_fifosizing:
        # - peak_occupancy: READ_FILL reports the peak occupancy of a bounded run
        # - sticky_stall_flags: READ_STALL reports run-sticky stall flags in bits [7:4]
        self.hw_capabilities = {"peak_occupancy": False, "sticky_stall_flags": False}

        # Sanity check
        # We expect 4 AXI-Lite peripherals:
        # fifo_controller_0, instrumentation_wrap_0, axi_gpio_0 (for reset), zynq_ps
        # We expect no additional FINN SDPs with AXI-Lite, such as runtime-writable weights
        if len(self.ip_dict.keys()) != 4:
            print(
                "Error: # of AXI-Lite interfaces (%d) does not match expected number of 4."
                % (len(self.ip_dict.keys()))
            )
            self.error = True
        if "fifo_controller_0" not in self.ip_dict.keys():
            print("Error: fifo_controller_0 AXI-Lite interface not found.")
            self.error = True

    def ctrl_read(self, opcode=0x00, fifo_id=0x0000, check_success=False):
        """Read a value from the FIFO controller via AXI-Lite."""
        address = (fifo_id << 8) | opcode
        # Shift by 2 because FIFO controller operates on word addresses
        response = self.fifo_controller_0.read(offset=(address << 2))
        if check_success and response != opcode:
            print(
                "Error: FIFO controller returned 0x%02x instead of expected 0x%02x."
                % (response, opcode)
            )
            self.error = True
        return response

    def ctrl_write(self, opcode=0x00, fifo_id=0x0000, value=0x00000000):
        """Write a value to the FIFO controller via AXI-Lite."""
        address = (fifo_id << 8) | opcode
        # Shift by 2 because FIFO controller operates on word addresses
        self.fifo_controller_0.write(offset=(address << 2), value=value)

    def ctrl_set_depth(self, fifo_id, depth=2):
        """Set FIFO depth via WRITE_FILL instruction."""
        # Issue WRITE_FILL instruction (asynchronous, returns immediately)
        self.ctrl_write(opcode=OP_WRITE_FILL, fifo_id=fifo_id, value=depth)
        # Read to confirm controller has returned to idle state
        self.ctrl_read(check_success=True)

    def configure_fifos_bounded(self, depths):
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
        self.hw_capabilities = {"peak_occupancy": sticky, "sticky_stall_flags": sticky}
        print(
            "HW CAPABILITIES: %s (stall flags after paced run: %s)"
            % (self.hw_capabilities, ["0x%02x" % f for f in flags])
        )
        return self.hw_capabilities

    def run_detached(self):
        """Run FIFOs in detached mode to determine bottleneck period."""
        self.reset_accelerator()

        # Issue RUN_DETACHED4 instruction
        self.ctrl_read(opcode=OP_RUN_DETACHED4, check_success=True)
        print("DEBUG: RUN_DETACHED4 completed")

        # Wait on detached run to complete by issuing BARRIER_CLEAN
        # Internally, the controller will re-issue this instruction until it succeeds
        # TODO: FIX BARRIER_CLEAN, simply sleep as a workaround
        time.sleep(5)
        # self.ctrl_read(opcode=OP_BARRIER_CLEAN, check_success=True)
        # print("DEBUG: BARRIER_CLEAN completed")

        # Issue COMP_PERIOD instruction to collect global max period across all FIFOs
        max_period = self.ctrl_read(opcode=OP_COMP_PERIOD)
        print("DEBUG: COMP_PERIOD completed")
        return max_period

    def run_paced(self, throttle_interval=0, runtime_s=1):
        """Run FIFOs in paced mode to determine bottleneck period."""
        self.reset_accelerator()

        # Issue RUN_PACED instruction
        self.ctrl_read(opcode=OP_RUN_PACED, check_success=True)

        # Let accelerator run for specified wallclock time
        self.start_accelerator(throttle_interval=throttle_interval)
        time.sleep(runtime_s)
        (
            overflow_err,
            underflow_err,
            frame,
            checksum,
            min_latency,
            latency,
            interval,
            *_,
        ) = self.observe_instrumentation(debug_print=True)
        self.stop_accelerator()

        # Collect maximum occupancy of all FIFOs by issuing READ_FILL instructions
        max_occupancy = self.read_peak_occupancy()

        return max_occupancy, latency

    def total_fifo_size(self, depths):
        """Calculate total FIFO size in kB."""
        # Assuming FIFO SDP/AXI-Lite interfaces are ordered consistently with FIFO IDs
        total_size_bits = 0
        for i, depth in enumerate(depths):
            total_size_bits += (depth + self.fifo_depth_offset) * self.fifo_widths[str(i)]
        total_size_kB = total_size_bits / 8.0 / 1000.0
        return total_size_kB

    def fifo_width_list(self):
        """FIFO widths in bits as a list indexed by FIFO ID."""
        return [self.fifo_widths[str(i)] for i in range(self.num_fifos)]

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
            Runtime for each test iteration in seconds
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
        strategy_kwargs :
            Passed on to the strategy (seed_strategy, upper_bound_factor, node_order, pareto,
            pareto_cost_limit, allow_unreliable_flags).

        Returns a dict with the keys of the original binary search (duration, fifo_depths,
        log_total_fifo_size, log_interval, log_min_latency, log_latency, all_iterations) plus
        ``search`` (the strategy summary) and ``result`` (the ``SearchResult``).
        """
        if isinstance(start_depth, (list, tuple)):
            start_depths = tuple(int(d) for d in start_depth)
        else:
            start_depths = tuple([int(start_depth)] * self.num_fifos)

        widths = self.fifo_width_list()
        cost_model = TotalBitsCost(widths, offset=self.fifo_depth_offset)
        oracle = LiveFIFOOracle(self, iteration_runtime, throttle_interval, self.hw_capabilities)
        cache = EvaluationCache(oracle, cost_model)

        # Run once at the start configuration to determine the targets (iteration "0")
        start_time = time.time()
        target = cache.evaluate(start_depths, role="target")
        feasibility = FeasibilityCheck.from_result(stop_condition, relaxation, target.result)
        budget = SearchBudget(max_evaluations=max_evaluations, max_runtime_s=max_runtime_s)
        ctx = SearchContext(
            oracle,
            feasibility,
            cost_model,
            start_depths,
            widths,
            cache=cache,
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
            "log_total_fifo_size": log_total_fifo_size,
            "log_interval": log_interval,
            "log_min_latency": log_min_latency,
            "log_latency": log_latency,
            "all_iterations": cache_to_iterations(cache),
            "search": {
                **search_result_to_dict(result),
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
            },
            "result": result,
        }

    def size_iteratively_binary_search(
        self,
        start_depth,
        iteration_runtime,
        throttle_interval=0,
        fifo_order_strategy="largest_first",
        stop_condition="both",
        relaxation=0.0,
    ):
        """Iteratively reduce FIFO depths using binary search to find minimum for each FIFO.

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
    ):
        """Generate and save FIFO sizing visualization graph."""
        # Round total FIFO size to integer kB values
        log_total_fifo_size = [int(round(x)) for x in log_total_fifo_size]

        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Total FIFO Size [kB]", color=color)
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
        plt.savefig(os.path.join(report_dir, "fifo_sizing_graph.png"), dpi=300)
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
        ax.set_ylabel("Total FIFO Size [kB]")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_title(
            "%s: %s" % (result.strategy, "optimal proven" if result.optimal_proven else "gap open")
        )
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "fifo_sizing_graph.png"), dpi=300)
        plt.close(fig)

    def experiment_fifosizing(self, *args, **kwargs):
        """Run live FIFO sizing experiment and save report.

        Kwargs: ``search_strategy`` (default ``binary_largest_first``; ``fifo_search_order``
        is accepted as a deprecated alias for ``binary_<order>``), ``stop_condition``,
        ``relaxation``, ``relaxation_sweep`` (binary strategies only), ``report_name`` (report
        sub-directory, default: the strategy name), ``max_evaluations``,
        ``max_runtime_s``, ``iteration_runtime_factor`` (safety factor on the per-evaluation
        runtime heuristic), ``hw_capabilities`` ("auto" or a bool overriding the probe) and the
        strategy-specific ``seed_strategy``, ``upper_bound_factor``, ``node_order``, ``pareto``,
        ``pareto_cost_limit``, ``allow_unreliable_flags``.
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
        strategy_kwargs = {k: kwargs[k] for k in STRATEGY_KWARGS if k in kwargs}
        is_binary = search_strategy.startswith("binary_")
        base_report_dir = kwargs.get("report_dir")
        # Create subdirectory for this search strategy + stop condition. report_name lets
        # several variants of one strategy (e.g. different seeds) keep separate reports.
        report_name = kwargs.get("report_name") or search_strategy
        report_dir = os.path.join(base_report_dir, report_name, stop_condition)
        os.makedirs(report_dir, exist_ok=True)
        reportfile = os.path.join(report_dir, "report_experiment_fifosizing.json")

        print("---PHASE 1: RUN_DETACHED---")
        max_period = self.run_detached()
        print("MEASURED MAX PERIOD: %d cycles" % max_period)

        print("---PHASE 2: RUN_PACED---")
        # TODO: Use better heuristic for runtime?
        max_occupancy, paced_latency = self.run_paced(throttle_interval=max_period, runtime_s=1)
        print("MEASURED MAX FIFO OCCUPANCIES:")
        print("FIFO ID | MAX OCCUPANCY")
        for fifo_id, occupancy in enumerate(max_occupancy):
            print(f"{fifo_id:7} | {occupancy:13}")
        print("TOTAL FIFO SIZE @ MAX OCCUPANCY (kB): %f" % self.total_fifo_size(max_occupancy))
        # Detect the virtual FIFO extensions (peak occupancy, run-sticky stall flags)
        if hw_capabilities == "auto":
            self.probe_hw_capabilities()
        else:
            self.hw_capabilities = {
                "peak_occupancy": bool(hw_capabilities),
                "sticky_stall_flags": bool(hw_capabilities),
            }
            print("HW CAPABILITIES (from kwargs): %s" % self.hw_capabilities)

        print("---PHASE 3: ITERATIVE MINIMIZATION---")
        print("SEARCH STRATEGY: %s" % search_strategy)
        print("STOP CONDITION: %s" % stop_condition)
        print("RELAXATION: %.1f%%" % (relaxation * 100))
        print("RELAXATION SWEEP: %s" % ("Enabled" if relaxation_sweep else "Disabled"))
        if strategy_kwargs:
            print("STRATEGY KWARGS: %s" % strategy_kwargs)
        if max_evaluations is not None or max_runtime_s is not None:
            print("BUDGET: %s evaluations, %s s" % (max_evaluations, max_runtime_s))
        # Determine search iteration runtime via heuristic based on free-running latency
        iteration_runtime = max(0.001, (paced_latency * 4) * 10 / 1000 / 1000 / 1000)
        iteration_runtime *= iteration_runtime_factor

        search_log = self.run_fifo_search(
            search_strategy,
            start_depth=max_occupancy,
            iteration_runtime=iteration_runtime,
            throttle_interval=max_period,
            stop_condition=stop_condition,
            relaxation=relaxation,
            max_evaluations=max_evaluations,
            max_runtime_s=max_runtime_s,
            **strategy_kwargs,
        )

        fifo_depths = search_log["fifo_depths"]
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
            "fifo_size_total_kB": final_total_fifo_size,
            "detached_max_period_cycles": max_period,
            "target_interval_cycles": target_interval,
            "final_interval_cycles": final_interval,
            "interval_degradation": interval_degradation,
            "target_latency_cycles": target_latency,
            "final_latency_cycles": final_latency,
            "latency_degradation": latency_degradation,
            "fifo_depths": {},
            "fifo_sizes": {},
            "search": {
                **search_summary,
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
                **{k: v for k, v in search_log.items() if k not in ("search", "result")},
            }

        # Add relaxation sweep results if available
        if relaxation_sweep_results:
            fifo_report["relaxation_sweep"] = relaxation_sweep_results
        for fifo, depth in enumerate(fifo_depths):
            size = (depth + self.fifo_depth_offset) * self.fifo_widths[str(fifo)]
            fifo_report["fifo_depths"][fifo] = depth + self.fifo_depth_offset
            fifo_report["fifo_sizes"][fifo] = size
        with open(os.path.join(report_dir, "fifo_sizing_report.json"), "w") as f:
            json.dump(fifo_report, f, indent=2)

        # Export measured FIFO settings for use by the follow-up FINN build.
        fifo_config = {
            "fifo_depths": {},
            "impl_style": {},
            "ram_style": {},
        }
        for fifo, depth in enumerate(fifo_depths):
            fifo_name = "StreamingFIFO_rtl_%d" % fifo
            final_depth = depth + self.fifo_depth_offset
            fifo_config["fifo_depths"][fifo_name] = final_depth
            fifo_config["impl_style"][fifo_name] = "rtl"
            fifo_config["ram_style"][fifo_name] = "auto"
        with open(os.path.join(report_dir, "fifo_config.json"), "w") as f:
            json.dump(fifo_config, f, indent=2)

        if self.folding_config is None:
            raise ValueError("Live FIFO sizing requires folding_config in settings.json.")
        with open(os.path.join(report_dir, "folding_config.json"), "w") as f:
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
        with open(reportfile, "w") as f:
            json.dump(report, f, indent=2)

        print("Done.")

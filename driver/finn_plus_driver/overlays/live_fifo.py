"""Driver for live FIFO sizing experiments."""

import json
import matplotlib.pyplot as plt
import os
import random
import time

from finn_plus_driver.overlays.instrumentation import FINNInstrumentationOverlay


class FINNLiveFIFOOverlay(FINNInstrumentationOverlay):
    """FINN overlay for live FIFO sizing."""

    def __init__(
        self,
        bitfile_name,
        platform="zynq",
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
        self.ctrl_write(opcode=0x0E, fifo_id=fifo_id, value=depth)
        # Read to confirm controller has returned to idle state
        self.ctrl_read(check_success=True)

    def configure_fifos_bounded(self, depths):
        """Configure all FIFOs with bounded depths.
        Caller can supply a list of depths or a single depth for all FIFOs.
        """
        if isinstance(depths, list):
            fifo_depths = depths
        else:
            fifo_depths = [depths] * self.num_fifos

        # Set depth for each FIFO
        for i in range(self.num_fifos):
            self.ctrl_set_depth(i, fifo_depths[i])

        # Issue RUN_BOUNDED instruction once all depths have been set
        self.ctrl_read(opcode=0x04, check_success=True)

    def run_detached(self):
        """Run FIFOs in detached mode to determine bottleneck period."""
        self.reset_accelerator()

        # Issue RUN_DETACHED4 instruction
        self.ctrl_read(opcode=0x07, check_success=True)
        print("DEBUG: RUN_DETACHED4 completed")

        # Wait on detached run to complete by issuing BARRIER_CLEAN
        # Internally, the controller will re-issue this instruction until it succeeds
        # TODO: FIX BARRIER_CLEAN, simply sleep as a workaround
        time.sleep(5)
        # self.ctrl_read(opcode=0x08, check_success=True)
        # print("DEBUG: BARRIER_CLEAN completed")

        # Issue COMP_PERIOD instruction to collect global max period across all FIFOs
        max_period = self.ctrl_read(opcode=0x0A)
        print("DEBUG: COMP_PERIOD completed")
        return max_period

    def run_paced(self, throttle_interval=0, runtime_s=1):
        """Run FIFOs in paced mode to determine bottleneck period."""
        self.reset_accelerator()

        # Issue RUN_PACED instruction
        self.ctrl_read(opcode=0x05, check_success=True)

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
        ) = self.observe_instrumentation(debug_print=True)
        self.stop_accelerator()

        # Collect maximum occupancy of all FIFOs by issuing READ_FILL instructions
        max_occupancy = []
        for i in range(self.num_fifos):
            max_occupancy.append(self.ctrl_read(opcode=0x0C, fifo_id=i))

        return max_occupancy, latency

    def total_fifo_size(self, depths):
        """Calculate total FIFO size in kB."""
        # Assuming FIFO SDP/AXI-Lite interfaces are ordered consistently with FIFO IDs
        total_size_bits = 0
        for i, depth in enumerate(depths):
            total_size_bits += (depth + self.fifo_depth_offset) * self.fifo_widths[str(i)]
        total_size_kB = total_size_bits / 8.0 / 1000.0
        return total_size_kB

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

        Parameters
        ----------
        start_depth : int or list
            Initial depth(s) for FIFOs
        iteration_runtime : float
            Runtime for each test iteration in seconds
        throttle_interval : int
            Throttle interval in cycles
        fifo_order_strategy : str
            Strategy for ordering FIFO optimization. Options:
            - "forward": Topological order (FIFO 0 to N-1)
            - "reverse": Reverse topological order (FIFO N-1 to 0)
            - "largest_first": Sort by initial size (depth * width)
            - "deepest_first": Sort by initial depth
            - "alternating": Ping-pong between first and last FIFOs
            - "random": Random shuffle order
        stop_condition : str
            Metric to use for determining if a FIFO depth is too small. Options:
            - "interval": Stop if interval degrades from target_interval
            - "latency": Stop if latency degrades from target_latency
            - "both": Stop if either interval or latency degrades
        relaxation : float
            Allowed degradation tolerance (0.0 to 1.0, where 1.0 = 100% degradation allowed).
            Default 0.0 means no degradation allowed.
        """
        fifo_minimum_reached = [False] * self.num_fifos

        if isinstance(start_depth, list):
            # Individual start depth for each FIFO has been supplied
            fifo_depths = start_depth.copy()
        else:
            # Initialize all depths to the same start depth
            fifo_depths = [start_depth] * self.num_fifos

        # Reset accelerator and configure FIFOs
        self.reset_accelerator()
        self.configure_fifos_bounded(fifo_depths)

        # Run once to determine target interval
        self.start_accelerator(throttle_interval=throttle_interval)
        time.sleep(iteration_runtime)
        (
            overflow_err,
            underflow_err,
            frame,
            checksum,
            min_latency,
            latency,
            interval,
        ) = self.observe_instrumentation(False)
        log_total_fifo_size = [self.total_fifo_size(fifo_depths)]
        log_interval = [interval]
        log_min_latency = [min_latency]
        log_latency = [latency]
        all_iterations = {
            "0": {
                "interval": interval,
                "min_latency": min_latency,
                "latency": latency,
                "total_fifo_size_kB": self.total_fifo_size(fifo_depths),
                "fifo_depths": fifo_depths.copy(),
            }
        }
        target_interval = interval
        target_latency = latency

        # Apply relaxation to thresholds
        # Always allow 0.05% latency degradation due to jitter observed for some models
        latency_default_relaxation = 0.0005
        relaxed_interval_threshold = target_interval * (1 + relaxation)
        relaxed_latency_threshold = target_latency * (1 + (latency_default_relaxation + relaxation))

        # Binary search for each FIFO to find minimum depth
        iteration = 0
        start_time = time.time()

        # Determine search order based on strategy
        if fifo_order_strategy == "forward":
            fifo_order = list(range(self.num_fifos))
        elif fifo_order_strategy == "reverse":
            fifo_order = list(range(self.num_fifos - 1, -1, -1))
        elif fifo_order_strategy == "largest_first":
            fifo_order = sorted(
                range(self.num_fifos), key=lambda i: -fifo_depths[i] * self.fifo_widths[str(i)]
            )
        elif fifo_order_strategy == "deepest_first":
            fifo_order = sorted(range(self.num_fifos), key=lambda i: -fifo_depths[i])
        elif fifo_order_strategy == "alternating":
            # Ping-pong between first and last
            fifo_order = []
            left, right = 0, self.num_fifos - 1
            while left <= right:
                fifo_order.append(left)
                if left != right:
                    fifo_order.append(right)
                left += 1
                right -= 1
        elif fifo_order_strategy == "random":
            fifo_order = list(range(self.num_fifos))
            random.shuffle(fifo_order)
        else:
            raise ValueError(f"Unknown fifo_order_strategy: {fifo_order_strategy}")

        for fifo_id in fifo_order:
            print(f"Binary searching for FIFO {fifo_id}...")

            # Binary search bounds
            low = 1
            high = fifo_depths[fifo_id]
            best_working_depth = high

            while low <= high:
                mid = (low + high) // 2

                # Test with this depth
                test_depths = fifo_depths.copy()
                test_depths[fifo_id] = mid

                # Reset accelerator
                self.reset_accelerator()

                # Configure all FIFOs
                self.configure_fifos_bounded(test_depths)

                # Start accelerator
                self.start_accelerator(throttle_interval=throttle_interval)

                # Let it run
                time.sleep(iteration_runtime)

                # Check if throughput dropped or deadlock occurred
                (
                    overflow_err,
                    underflow_err,
                    frame,
                    checksum,
                    min_latency,
                    latency,
                    interval,
                ) = self.observe_instrumentation(False)

                # Determine if this depth causes degradation based on stop_condition
                if stop_condition == "interval":
                    degraded = interval > relaxed_interval_threshold
                elif stop_condition == "latency":
                    degraded = latency > relaxed_latency_threshold
                elif stop_condition == "both":
                    degraded = (
                        interval > relaxed_interval_threshold or latency > relaxed_latency_threshold
                    )
                else:
                    raise ValueError(f"Unknown stop_condition: {stop_condition}")

                if degraded or interval == 0 or overflow_err or underflow_err:
                    # This depth is too small, search higher
                    low = mid + 1
                    result_status = "FAIL"
                else:
                    # This depth works, try smaller
                    best_working_depth = mid
                    high = mid - 1
                    result_status = "PASS"

                    # Log this successful configuration
                    log_total_fifo_size.append(self.total_fifo_size(test_depths))
                    log_interval.append(interval)
                    log_min_latency.append(min_latency)
                    log_latency.append(latency)

                iteration += 1

                # Log all iterations
                all_iterations[str(iteration)] = {
                    "tested_fifo": fifo_id,
                    "tested_depth": mid,
                    "status": result_status,
                    "search_bounds": [low, high],
                    "best_working_depth": best_working_depth,
                    "interval": interval,
                    "min_latency": min_latency,
                    "latency": latency,
                    "total_fifo_size_kB": self.total_fifo_size(test_depths),
                    "fifo_depths": test_depths.copy(),
                }

                # Report status
                result = result_status
                print(f"  Iteration {iteration}: Testing depth {mid} - {result}")
                print(f"    Binary search bounds: [{low}, {high}]")
                print(f"    Best working depth so far: {best_working_depth}")
                if stop_condition == "interval" or stop_condition == "both":
                    print(
                        f"    Interval: {interval}, "
                        f"Threshold: {relaxed_interval_threshold:.1f} "
                        f"(Target: {target_interval})"
                    )
                if stop_condition == "latency" or stop_condition == "both":
                    print(
                        f"    Latency: {latency}, "
                        f"Threshold: {relaxed_latency_threshold:.1f} "
                        f"(Target: {target_latency})"
                    )

            # Set the FIFO to its minimum working depth
            fifo_depths[fifo_id] = best_working_depth
            fifo_minimum_reached[fifo_id] = True

            print(f"  FIFO {fifo_id} minimized to depth {best_working_depth}")
            print(f"  Number of minimized FIFOs: {sum(fifo_minimum_reached)}/{self.num_fifos}")
            print(f"  Total FIFO Size (kB): {self.total_fifo_size(fifo_depths)}")

        end_time = time.time()
        duration = int(end_time - start_time)
        print(f"Done ({duration} seconds)")

        return {
            "duration": duration,
            "fifo_depths": fifo_depths,
            "log_total_fifo_size": log_total_fifo_size,
            "log_interval": log_interval,
            "log_min_latency": log_min_latency,
            "log_latency": log_latency,
            "all_iterations": all_iterations,
        }

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

    def experiment_fifosizing(self, *args, **kwargs):
        """Run live FIFO sizing experiment and save report."""
        fifo_search_order = kwargs.get("fifo_search_order", "largest_first")
        stop_condition = kwargs.get("stop_condition", "both")
        relaxation = kwargs.get("relaxation", 0.0)
        relaxation_sweep = kwargs.get("relaxation_sweep", False)
        base_report_dir = kwargs.get("report_dir")
        # Create subdirectory for this search order + stop condition
        report_dir = os.path.join(base_report_dir, fifo_search_order, stop_condition)
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

        print("---PHASE 3: ITERATIVE MINIMIZATION---")
        print("FIFO SEARCH ORDER: %s" % fifo_search_order)
        print("STOP CONDITION: %s" % stop_condition)
        print("RELAXATION: %.1f%%" % (relaxation * 100))
        print("RELAXATION SWEEP: %s" % ("Enabled" if relaxation_sweep else "Disabled"))
        # Determine search iteration runtime via heuristic based on free-running latency
        iteration_runtime = max(0.001, (paced_latency * 4) * 10 / 1000 / 1000 / 1000)

        search_log = self.size_iteratively_binary_search(
            start_depth=max_occupancy,
            iteration_runtime=iteration_runtime,
            throttle_interval=max_period,
            fifo_order_strategy=fifo_search_order,
            stop_condition=stop_condition,
            relaxation=relaxation,
        )

        fifo_depths = search_log["fifo_depths"]
        log_total_fifo_size = search_log["log_total_fifo_size"]
        log_interval = search_log["log_interval"]
        log_min_latency = search_log["log_min_latency"]
        log_latency = search_log["log_latency"]

        # Generate visualization graph
        self.generate_fifosizing_graph(
            log_total_fifo_size,
            log_min_latency,
            log_latency,
            log_interval,
            report_dir,
            stop_condition,
        )

        # Calculate relative degradation
        target_interval = log_interval[0]
        target_latency = log_latency[0]
        final_interval = log_interval[-1]
        final_latency = log_latency[-1]

        interval_degradation = (
            (final_interval - target_interval) / target_interval if target_interval != 0 else 0
        )
        latency_degradation = (
            (final_latency - target_latency) / target_latency if target_latency != 0 else 0
        )

        # Relaxation sweep: explore additional relaxation values if enabled
        relaxation_sweep_results = []
        if relaxation_sweep:
            print("---RELAXATION SWEEP---")
            # Pre-defined sequence of relaxation values to explore
            relaxation_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
            # Filter out values <= current relaxation to avoid redundant searches
            relaxation_values = [r for r in relaxation_values if r > relaxation]

            for sweep_relaxation in relaxation_values:
                print(f"Testing relaxation: {sweep_relaxation:.2f} ({sweep_relaxation*100:.0f}%)")

                sweep_search_log = self.size_iteratively_binary_search(
                    start_depth=max_occupancy,
                    iteration_runtime=iteration_runtime,
                    throttle_interval=max_period,
                    fifo_order_strategy=fifo_search_order,
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
        fifo_report = {
            "error": self.error,
            "fifo_size_total_kB": log_total_fifo_size[-1],
            "detached_max_period_cycles": max_period,
            "target_interval_cycles": target_interval,
            "final_interval_cycles": final_interval,
            "interval_degradation": interval_degradation,
            "target_latency_cycles": target_latency,
            "final_latency_cycles": final_latency,
            "latency_degradation": latency_degradation,
            "fifo_depths": {},
            "fifo_sizes": {},
            "binary_search": {
                "search_order": fifo_search_order,
                "stop_condition": stop_condition,
                "relaxation": relaxation,
                "iteration_runtime_s": iteration_runtime,
                **search_log,
            },
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
        min_latency = log_min_latency[-1]
        latency = log_latency[-1]
        interval = log_interval[-1]
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

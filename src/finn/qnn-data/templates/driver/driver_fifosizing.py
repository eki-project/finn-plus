import argparse
import json
import matplotlib.pyplot as plt
import os
import sys
import time
from driver_instrumentation import FINNInstrumentationOverlay
from pynq import PL
from pynq.pl_server.device import Device


class FINNLiveFIFOOverlay(FINNInstrumentationOverlay):
    def __init__(
        self,
        bitfile_name,
        platform="zynq",
        fclk_mhz=100.0,
        device=None,
        download=True,
        seed=1,
        fifo_widths=dict(),
    ):
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
        # Try to account for additional registers introduced by virtual FIFO HLS implementation
        self.fifo_depth_offset = 4

        # Sanity check
        # We expect 3 AXI-Lite peripherals next to the virtual FIFOs: instrumentation_wrap_0, axi_gpio_0 (for reset), zynq_ps
        # We don't expect any additional FINN SDPs with AXI-Lite interface, such as runtime-writable weights
        if (len(self.ip_dict.keys()) - 3) != self.num_fifos:
            print(
                "Error: Number of expected FIFOs (%d) doesn't match number of AXI-Lite interfaces (%d)"
                % (self.num_fifos, len(self.ip_dict.keys()) - 3)
            )
            self.error = True

    def configure_fifo(self, i, mode, depth=2):
        ### Virtual FIFO register map ###
        mode_offset = 0x10
        depth_offset = 0x18
        occupancy_offset = 0x20
        occupancy_ctrl_offset = 0x24
        max_occupancy_offset = 0x30
        max_occupancy_ctrl_offset = 0x34

        ip_name = "StreamingDataflowPartition_%d" % i
        getattr(self, ip_name).write(offset=mode_offset, value=mode)
        getattr(self, ip_name).write(offset=depth_offset, value=depth)

    def total_fifo_size(self, depths):
        # Assuming FIFO SDP/AXI-Lite interfaces are ordered consistently with FIFO IDs
        total_size_bits = 0
        for i, depth in enumerate(depths):
            total_size_bits += (depth + self.fifo_depth_offset) * self.fifo_widths[i]
        total_size_kB = total_size_bits / 8.0 / 1000.0
        return total_size_kB

    def size_iteratively(self, start_depth, iteration_runtime, reduction_factor=0.5):
        ### Iterative FIFO-sizing function ###
        fifo_minimum_reached = [False] * self.num_fifos

        if isinstance(start_depth, list):
            # Individual start depth for each FIFO has been supplied
            fifo_depths = start_depth
        else:
            # Initialize all depths to the same start depth
            fifo_depths = [start_depth] * self.num_fifos

        # Reset accelerator and configure FIFOs
        self.reset_accelerator()
        for i in range(0, self.num_fifos):
            self.configure_fifo(i, mode=1, depth=fifo_depths[i])

        # Run once to determine target interval
        self.start_accelerator()
        time.sleep(1)
        (
            overflow_err,
            underflow_err,
            frame,
            checksum,
            min_latency,
            latency,
            interval,
        ) = self.observe_instrumentation(False)
        log_total_fifo_size = [int(self.total_fifo_size(fifo_depths))]
        log_interval = [interval]
        log_min_latency = [min_latency]
        log_latency = [latency]
        target_interval = interval

        # Iteratively reduce FIFO depth until all FIFOs are minimized
        iteration = 0
        start_time = time.time()
        while not all(fifo_minimum_reached):
            for fifo_id in range(0, self.num_fifos):
                if not fifo_minimum_reached[fifo_id]:
                    fifo_depth_before = fifo_depths[fifo_id]
                    fifo_depths[fifo_id] = int(fifo_depths[fifo_id] * reduction_factor)

                    # Reset accelerator
                    self.reset_accelerator()

                    # Configure all FIFOs
                    for i in range(0, self.num_fifos):
                        self.configure_fifo(i, mode=1, depth=fifo_depths[i])

                    # Start accelerator
                    self.start_accelerator()

                    # Let it run
                    time.sleep(iteration_runtime)

                    # Check if throughput dropped or deadlock occured
                    (
                        overflow_err,
                        underflow_err,
                        frame,
                        checksum,
                        min_latency,
                        latency,
                        interval,
                    ) = self.observe_instrumentation(False)

                    if interval > target_interval or interval == 0 or overflow_err or underflow_err:
                        # Revert depth reduction and mark FIFO as minimized
                        fifo_depths[fifo_id] = fifo_depth_before
                        fifo_minimum_reached[fifo_id] = True
                    else:
                        log_total_fifo_size.append(int(self.total_fifo_size(fifo_depths)))
                        log_interval.append(interval)
                        log_min_latency.append(min_latency)
                        log_latency.append(latency)

                    if fifo_depths[fifo_id] == 1:
                        fifo_minimum_reached[fifo_id] = True

            # Report status
            print("Iteration: %d" % iteration)
            print("Numer of minimized FIFOs: %d/%d" % (sum(fifo_minimum_reached), self.num_fifos))
            print("Interval: %d" % log_interval[-1])
            print("Min. latency / latency: %d/%d" % (log_min_latency[-1], log_latency[-1]))
            print("Total FIFO Size (kB): %d" % log_total_fifo_size[-1])

            iteration += 1

        end_time = time.time()
        duration = int(end_time - start_time)
        print("Done (%d seconds)" % duration)

        return (
            fifo_depths,
            log_total_fifo_size,
            log_interval,
            log_min_latency,
            log_latency,
            duration,
        )

    def determine_start_depth(
        self,
    ):
        ### Attempt to determine start depth for all FIFOs automatically ###
        # If it doesn't find a working setting, start depth must be set manually, potentially on per-FIFO basis
        start_depth = 64
        last_interval = 0
        start_depth_found = False

        while not start_depth_found and not self.error:
            print("Testing start depth of %d" % start_depth)
            self.reset_accelerator()

            # Configure FIFOs
            for i in range(0, self.num_fifos):
                self.configure_fifo(i, mode=1, depth=start_depth)

            # Start accelerator and let it run for a long time
            self.start_accelerator()
            time.sleep(1)

            # Examine performance
            (
                overflow_err,
                underflow_err,
                frame,
                checksum,
                min_latency,
                latency,
                interval,
            ) = self.observe_instrumentation()
            if (
                interval > 0
                and interval == last_interval
                and not overflow_err
                and not underflow_err
            ):
                # Accelerator runs with stable interval, reset to previous start depth
                start_depth_found = True
                start_depth = last_start_depth
            else:
                # Start depth is still too small, increase for next try
                last_start_depth = start_depth
                start_depth = start_depth * 2

            last_interval = interval

            if start_depth > 1000000:
                print("Couldn't find a working start depth, please set manually")
                self.error = True

        # Determine runtime per iteration based on performance, so that stable-state is guaranteed
        # Use a simple overestimation for now to be safe
        iteration_runtime = max(0.01, (min_latency * 5) * 10 / 1000 / 1000 / 1000)

        print("Determined start depth for all FIFOs: %d" % start_depth)
        print("Determined iteration runtime based on performance: %f s" % iteration_runtime)
        return (start_depth, iteration_runtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile performance of FINN-generated accelerator using instrumentation wrapper"
    )
    parser.add_argument("--runtime", help="Runtime in seconds", type=int, default=10)
    parser.add_argument(
        "--frequency", help="FPGA clock frequency in MHz", type=float, default=100.0
    )
    parser.add_argument("--seed", help="LFSR seed for input data generation", type=int, default=1)
    parser.add_argument("--device", help="FPGA device to be used", type=int, default=0)
    parser.add_argument("--bitfile", help="Name of bitfile", default="finn-accel.bit")
    parser.add_argument(
        "--reportfile",
        help="Name of output .json report file",
        type=str,
        default="measured_performance.json",
    )
    parser.add_argument(
        "--settingsfile", help="Name of optional input .json settings file", type=str, default=""
    )
    # parse arguments
    args = parser.parse_args()
    runtime = args.runtime
    frequency = args.frequency
    seed = args.seed
    bitfile = args.bitfile
    reportfile = args.reportfile
    report_dir = os.path.dirname(reportfile)
    settingsfile = args.settingsfile
    devID = args.device
    device = Device.devices[devID]

    # overwrite frequency if specified in settings file
    if settingsfile != "":
        with open(settingsfile, "r") as f:
            settings = json.load(f)
            if "fclk_mhz" in settings:
                frequency = settings["fclk_mhz"]

            # For live FIFO-sizing, we also expect a fifo_widths.json file exported by FINN listing the width of each FIFO, e.g.,
            # {'fifo_widths': {0: 8, 1: 32, 2: 24}}
            fifo_widths = settings["fifo_widths"]

    print("Programming FPGA..")
    PL.reset()  # reset PYNQ cache
    accel = FINNLiveFIFOOverlay(
        bitfile_name=bitfile, device=device, fclk_mhz=frequency, seed=seed, fifo_widths=fifo_widths
    )
    if accel.error:
        print("Error: Accelerator initialization failed.")
        sys.exit(1)

    print("Determining start depth..")
    (start_depth, iteration_runtime) = accel.determine_start_depth()

    ### First pass
    print("Starting first pass..")
    pass1_result = accel.size_iteratively(start_depth, iteration_runtime)
    (
        fifo_depths,
        log_total_fifo_size,
        log_interval,
        log_min_latency,
        log_latency,
        duration,
    ) = pass1_result

    ### Visualize results and save as "fifo_sizing_graph.png"
    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Total FIFO Size [kB]", color=color)
    ax1.plot(range(len(log_total_fifo_size)), log_total_fifo_size, color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, max(log_total_fifo_size))

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel("Latency [cycles]", color=color)
    ax2.plot(range(len(log_total_fifo_size)), log_latency, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    # ax2.set_ylim(0, max(log_latency))

    ax2.axhline(log_min_latency[0], color="green", label="Minimum (1st frame) Latency")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "fifo_sizing_graph.png"), dpi=300)

    ### Second pass for fine-tuning
    print("Starting second pass..")
    pass2_result = accel.size_iteratively(fifo_depths, iteration_runtime, reduction_factor=0.95)
    (
        fifo_depths,
        log_total_fifo_size,
        log_interval,
        log_min_latency,
        log_latency,
        duration,
    ) = pass2_result

    ### Generate fifo_sizing_report.json
    fifo_report = {
        "error": accel.error,
        "fifo_size_total_kB": log_total_fifo_size[-1],
        "fifo_depths": {},
        "fifo_sizes": {},
        "pass_1": {
            "duration": pass1_result[5],
            "log_total_fifo_size": pass1_result[1],
            "log_interval": pass1_result[2],
            "log_min_latency": pass1_result[3],
            "log_latency": pass1_result[4],
        },
        "pass_2": {
            "duration": pass2_result[5],
            "log_total_fifo_size": pass2_result[1],
            "log_interval": pass2_result[2],
            "log_min_latency": pass2_result[3],
            "log_latency": pass2_result[4],
        },
    }
    for fifo, depth in enumerate(fifo_depths):
        size = (depth + accel.fifo_depth_offset) * accel.fifo_widths[fifo]
        fifo_report["fifo_depths"][fifo] = depth + accel.fifo_depth_offset
        fifo_report["fifo_sizes"][fifo] = size
    with open(os.path.join(report_dir, "fifo_sizing_report.json"), "w") as f:
        json.dump(fifo_report, f, indent=2)

    ### Generate fifo_depth_export.json to export FIFO depths for use in FINN
    fifo_depth_export = {}
    for fifo, depth in enumerate(fifo_depths):
        fifo_depth_export["StreamingFIFO_rtl_%d" % fifo] = {}
        fifo_depth_export["StreamingFIFO_rtl_%d" % fifo]["depth"] = depth + accel.fifo_depth_offset
    with open(os.path.join(report_dir, "fifo_depth_export.json"), "w") as f:
        json.dump(fifo_depth_export, f, indent=2)

    ### Generate the usual instrumentation performance report based on final state
    min_latency = log_min_latency[-1]
    latency = log_latency[-1]
    interval = log_interval[-1]
    report = {
        "error": accel.error,
        "checksum": 0,
        "min_latency_cycles": min_latency,
        "latency_cycles": latency,
        "interval_cycles": interval,
        "frequency_mhz": round(accel.fclk_mhz_actual),
        "min_latency_ms": round(min_latency * (1 / (accel.fclk_mhz_actual * 1e6)) * 1e3, 6),
        "latency_ms": round(latency * (1 / (accel.fclk_mhz_actual * 1e6)) * 1e3, 6),
        "throughput_fps": round(1 / (interval * (1 / (accel.fclk_mhz_actual * 1e6)))),
        "min_pipeline_depth": round(min_latency / interval, 2),
        "pipeline_depth": round(latency / interval, 2),
    }
    with open(reportfile, "w") as f:
        json.dump(report, f, indent=2)

    print("Done.")

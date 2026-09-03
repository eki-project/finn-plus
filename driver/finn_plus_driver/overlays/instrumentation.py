"""Driver for instrumentation-only accelerators without DMA."""

import json
import os
import time
from finn_plus_driver.hwh import get_clk_wiz_params_from_hwh
from pynq import Overlay
from pynq.ps import Clocks


class FINNInstrumentationOverlay(Overlay):
    """FINN overlay for instrumentation."""

    def __init__(
        self,
        bitfile_name,
        platform="zynq-iodma",
        fclk_mhz=100.0,
        device=None,
        download=True,
        seed=1,
        **kwargs,
    ):
        """Initialize instrumentation overlay."""
        super().__init__(bitfile_name, download=download, device=device)

        self.platform = platform
        self.fclk_mhz = fclk_mhz  # currently ignored (TODO: make clocking wizard configurable)
        self.seed = seed

        # configure clock (for ZYNQ platforms)
        if self.platform == "zynq-iodma":
            clk_wiz_params = get_clk_wiz_params_from_hwh(bitfile_name)
            Clocks.fclk0_mhz = 100.0  # Clocking Wizard is configured for fixed 100 MHz input clock
            self.fclk_mhz_actual = float(
                clk_wiz_params.get(
                    "CLKOUT1_OUT_FREQ",
                    clk_wiz_params.get("CLKOUT1_REQUESTED_OUT_FREQ", str(self.fclk_mhz)),
                )
            )

    def instrumentation_read(self, name):
        """Read instrumentation register."""
        return self.instrumentation_wrap_0.read(
            offset=self.ip_dict["instrumentation_wrap_0"]["registers"][name]["address_offset"]
        )

    def instrumentation_write(self, name, value):
        """Write instrumentation register."""
        return self.instrumentation_wrap_0.write(
            offset=self.ip_dict["instrumentation_wrap_0"]["registers"][name]["address_offset"],
            value=value,
        )

    def reset_accelerator(self):
        """Reset the accelerator."""
        self.axi_gpio_0.write(
            offset=self.ip_dict["axi_gpio_0"]["registers"]["GPIO_DATA"]["address_offset"], value=0
        )

    def start_accelerator(self, throttle_interval=0, avg_window_size=64, mux_interval=0):
        """Start the accelerator. Input is throttled to the specified interval (in cycles)
        by pausing after each FM transmission. A throttle_interval of 0 means no throttling.
        mux_interval controls tUSER round-robin scheduling: 0 = fixed tUSER=0,
        N = advance tUSER every N frames.
        """
        # Set seed
        lfsr_seed = (self.seed << 16) & 0xFFFF0000  # upper 16 bits
        self.instrumentation_write("seed", lfsr_seed)

        # Set average measurement window size (in frames),
        # maximum is configured in build config, default value = 64
        self.instrumentation_write("avg_n", avg_window_size)

        # Set tUSER multiplexing interval (frames per tUSER value, 0 = fixed)
        self.instrumentation_write("mux_interval", mux_interval)

        # Start operation
        self.instrumentation_write("cfg", (throttle_interval << 1) | 1)  # bit 0 = start

    def stop_accelerator(self):
        """Stop the accelerator."""
        self.instrumentation_write("cfg", 0)  # bit 0 = stop

    def observe_instrumentation(self, debug_print=True):
        """Read and report instrumentation metrics."""
        status_reg = self.instrumentation_read("status")
        chksum_reg = self.instrumentation_read("checksum")
        min_latency = self.instrumentation_read("min_latency")
        latency = self.instrumentation_read("latency")
        interval = self.instrumentation_read("interval")
        lat_sum_lo = self.instrumentation_read("lat_sum_lo")
        lat_sum_hi = self.instrumentation_read("lat_sum_hi")
        int_sum_lo = self.instrumentation_read("int_sum_lo")
        int_sum_hi = self.instrumentation_read("int_sum_hi")
        avg_fill = self.instrumentation_read("avg_fill")
        run_cycles_lo = self.instrumentation_read("run_cycles_lo")
        run_cycles_hi = self.instrumentation_read("run_cycles_hi")
        run_frames = self.instrumentation_read("run_frames")

        frame = (chksum_reg >> 24) & 0x000000FF
        checksum = chksum_reg & 0x00FFFFFF
        overflow_err = (status_reg & 0x00000001) != 0
        underflow_err = (status_reg & 0x00000002) != 0
        run_cycles = (run_cycles_hi << 32) | run_cycles_lo
        lat_sum = (lat_sum_hi << 32) | lat_sum_lo
        int_sum = (int_sum_hi << 32) | int_sum_lo
        avg_latency = lat_sum // avg_fill if avg_fill > 0 else 0
        avg_interval = int_sum // avg_fill if avg_fill > 0 else 0

        if debug_print:
            print("---INSTRUMENTATION_REPORT---")
            if overflow_err or underflow_err:
                print("Status ERROR")
                print("Overflow error: %s" % overflow_err)
                print("Underflow error: %s" % underflow_err)
            else:
                print("Status OK")
            print("Frame number (8-bit): %d" % frame)
            print("Checksum: 0x%06x" % checksum)
            print("Min Latency (cycles): %d" % min_latency)
            print("Latency (cycles): %d" % latency)
            print("Interval (cycles): %d" % interval)
            print("Average Latency (cycles): %d" % avg_latency)
            print("Average Interval (cycles): %d" % avg_interval)
            print("Run Cycles: %d" % run_cycles)
            print("Run Frames: %d" % run_frames)
            if run_frames > 0:
                print("Run Average Interval (cycles): %.1f" % (run_cycles / run_frames))
            print("----------------------------")

        return (
            overflow_err,
            underflow_err,
            frame,
            checksum,
            min_latency,
            latency,
            interval,
            avg_latency,
            avg_interval,
            run_cycles,
            run_frames,
        )

    def experiment_instrumentation(self, *args, **kwargs):
        """Run instrumentation experiment and save report."""
        runtime = kwargs.get("runtime")
        report_dir = kwargs.get("report_dir")
        mux_interval = kwargs.get("mux_interval", 0)

        # start accelerator
        print("Running accelerator for %d seconds.." % runtime)
        self.start_accelerator(mux_interval=mux_interval)

        # let it run for specified runtime
        time.sleep(runtime)

        # read measurement from instrumentation
        (
            overflow_err,
            underflow_err,
            frame,
            checksum,
            min_latency,
            latency,
            interval,
            avg_latency,
            avg_interval,
            run_cycles,
            run_frames,
        ) = self.observe_instrumentation()

        # write report to file
        fclk = self.fclk_mhz_actual * 1e6
        report = {
            "error": overflow_err or underflow_err or interval == 0,
            "checksum": checksum,
            "min_latency_cycles": min_latency,
            "latency_cycles": latency,
            "interval_cycles": interval,
            "avg_latency_cycles": avg_latency,
            "avg_interval_cycles": avg_interval,
            "run_cycles": run_cycles,
            "run_frames": run_frames,
            "frequency_mhz": round(self.fclk_mhz_actual),
            "min_latency_ms": round(min_latency * (1 / fclk) * 1e3, 6),
            "latency_ms": round(latency * (1 / fclk) * 1e3, 6),
            "avg_latency_ms": round(avg_latency * (1 / fclk) * 1e3, 6),
            "throughput_fps": round(fclk / interval) if interval != 0 else 0,
            "avg_throughput_fps": round(fclk / avg_interval) if avg_interval != 0 else 0,
            "run_avg_throughput_fps": round(run_frames / (run_cycles / fclk))
            if run_cycles > 0
            else 0,
            "min_pipeline_depth": round(min_latency / interval, 2) if interval != 0 else 0,
            "pipeline_depth": round(latency / interval, 2) if interval != 0 else 0,
        }
        reportfile = os.path.join(report_dir, "report_experiment_instrumentation.json")
        with open(reportfile, "w") as f:
            json.dump(report, f, indent=2)

        print("Done.")

    def idle(self, *args, **kwargs):
        """Run idle for specified time."""
        runtime = kwargs.get("time")
        print("Running idle for %d seconds.." % runtime)
        time.sleep(runtime)
        print("Done.")

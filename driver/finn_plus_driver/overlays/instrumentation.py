"""Driver for instrumentation-only accelerators without DMA."""

import json
import os
import time
from pynq import Overlay
from pynq.ps import Clocks


class FINNInstrumentationOverlay(Overlay):
    """FINN overlay for instrumentation."""

    def __init__(
        self,
        bitfile_name,
        platform="zynq",
        fclk_mhz=100.0,
        device=None,
        download=True,
        seed=1,
        **kwargs,
    ):
        """Initialize instrumentation overlay."""
        super().__init__(bitfile_name, download=download, device=device)

        self.platform = platform
        self.fclk_mhz = fclk_mhz
        self.seed = seed

        # configure clock (for ZYNQ platforms)
        if self.platform == "zynq":
            if self.fclk_mhz > 0:
                Clocks.fclk0_mhz = self.fclk_mhz
                self.fclk_mhz_actual = Clocks.fclk0_mhz

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

    def start_accelerator(self, throttle_interval=0, avg_window_size=64):
        """Start the accelerator. Input is throttled to the specified interval (in cycles)
        by pausing after each FM transmission. A throttle_interval of 0 means no throttling.
        """
        # Set seed
        lfsr_seed = (self.seed << 16) & 0xFFFF0000  # upper 16 bits
        self.instrumentation_write("seed", lfsr_seed)

        # Set average measurement window size (in frames),
        # maximum is configured in build config, default value = 64
        self.instrumentation_write("avg_n", avg_window_size)

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
        avg_latency = self.instrumentation_read("avg_latency")
        avg_interval = self.instrumentation_read("avg_interval")

        frame = (chksum_reg >> 24) & 0x000000FF
        checksum = chksum_reg & 0x00FFFFFF
        overflow_err = (status_reg & 0x00000001) != 0
        underflow_err = (status_reg & 0x00000002) != 0

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
        )

    def experiment_instrumentation(self, *args, **kwargs):
        """Run instrumentation experiment and save report."""
        runtime = kwargs.get("runtime")
        report_dir = kwargs.get("report_dir")

        # start accelerator
        print("Running accelerator for %d seconds.." % runtime)
        self.start_accelerator()

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
        ) = self.observe_instrumentation()

        # write report to file
        report = {
            "error": overflow_err or underflow_err or interval == 0,
            "checksum": checksum,
            "min_latency_cycles": min_latency,
            "latency_cycles": latency,
            "interval_cycles": interval,
            "avg_latency_cycles": avg_latency,
            "avg_interval_cycles": avg_interval,
            "frequency_mhz": round(self.fclk_mhz_actual),
            "min_latency_ms": round(min_latency * (1 / (self.fclk_mhz_actual * 1e6)) * 1e3, 6),
            "latency_ms": round(latency * (1 / (self.fclk_mhz_actual * 1e6)) * 1e3, 6),
            "avg_latency_ms": round(avg_latency * (1 / (self.fclk_mhz_actual * 1e6)) * 1e3, 6),
            "throughput_fps": (
                round(1 / (interval * (1 / (self.fclk_mhz_actual * 1e6)))) if interval != 0 else 0
            ),
            "avg_throughput_fps": (
                round(1 / (avg_interval * (1 / (self.fclk_mhz_actual * 1e6))))
                if avg_interval != 0
                else 0
            ),
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

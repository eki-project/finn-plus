import argparse
import json
import time
from pynq import Overlay
from pynq.pl_server.device import Device
from pynq.ps import Clocks

# Instrumentation wrapper register map #
# ap_uint<32>  cfg,   	// [0] - 0:hold, 1:lfsr; [31:16] - LFSR seed
# ap_uint<32> &status,	// [0] - timestamp overflow; [1] - timestamp underflow
# ap_uint<32> &latency,
# ap_uint<32> &interval,
# ap_uint<32> &checksum,
# ap_uint<32> &min_latency


class FINNInstrumentationOverlay(Overlay):
    def __init__(
        self,
        bitfile_name,
        platform="zynq",
        fclk_mhz=100.0,
        device=None,
        download=True,
        seed=1,
    ):
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
        return self.instrumentation_wrap_0.read(
            offset=self.ip_dict["instrumentation_wrap_0"]["registers"][name]["address_offset"]
        )

    def instrumentation_write(self, name, value):
        return self.instrumentation_wrap_0.write(
            offset=self.ip_dict["instrumentation_wrap_0"]["registers"][name]["address_offset"],
            value=value,
        )

    def reset_accelerator(self):
        self.axi_gpio_0.write(
            offset=self.ip_dict["axi_gpio_0"]["registers"]["GPIO_DATA"]["address_offset"], value=0
        )

    def start_accelerator(self):
        lfsr_seed = (self.seed << 16) & 0xFFFF0000  # upper 16 bits
        self.instrumentation_write("cfg", lfsr_seed + 1)  # start operation

    def observe_instrumentation(self, debug_print=True):
        status_reg = self.instrumentation_read("status")
        chksum_reg = self.instrumentation_read("checksum")
        min_latency = self.instrumentation_read("min_latency")
        latency = self.instrumentation_read("latency")
        interval = self.instrumentation_read("interval")

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
            print("----------------------------")

        return (overflow_err, underflow_err, frame, checksum, min_latency, latency, interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile FINN-generated accelerator using instrumentation wrapper"
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
    settingsfile = args.settingsfile
    devID = args.device
    device = Device.devices[devID]

    # overwrite frequency if specified in settings file
    if settingsfile != "":
        with open(settingsfile, "r") as f:
            settings = json.load(f)
            if "fclk_mhz" in settings:
                frequency = settings["fclk_mhz"]

    # instantiate FINN accelerator driver and pass batchsize and bitfile
    print("Programming FPGA..")
    accel = FINNInstrumentationOverlay(
        bitfile_name=bitfile, device=device, fclk_mhz=frequency, seed=seed
    )

    # start accelerator
    print("Running accelerator..")
    accel.start_accelerator()

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
    ) = accel.observe_instrumentation()

    # write report to file
    report = {
        "error": overflow_err or underflow_err or interval == 0,
        "checksum": checksum,
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

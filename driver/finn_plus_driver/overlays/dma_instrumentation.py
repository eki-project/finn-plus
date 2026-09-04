"""Driver combining DMA data movement with instrumentation."""

import json
import math
import numpy as np
import os
import re
import time
from finn_plus_driver.overlays.dma import FINNDMAOverlay
from finn_plus_driver.overlays.instrumentation import FINNInstrumentationOverlay
from pathlib import Path
from pynq import Bitstream, allocate
from typing import Any


class FINNDMAInstrumentationOverlay(FINNDMAOverlay, FINNInstrumentationOverlay):
    """FINN overlay for DMA and instrumentation (with Switch Block)."""

    class DFXController:
        """Manages the DFX Controller IP for partial reconfiguration."""

        # Note: sockets have to ordered correctly.
        def __init__(
            self,
            dfx_controller_inst: Any,
            sockets: list[str],
            bitstream_folder: str | None = None,
        ) -> None:
            """Initialize DFXController, load bitstreams and configure the hardware."""
            self.dfx_controller_inst = dfx_controller_inst
            self.bitstream_folder = bitstream_folder
            if not Path(self.bitstream_folder).is_dir():
                raise ValueError(f"Bitstream folder does not exist: {self.bitstream_folder}")

            self.socket_map = {socket: idx for idx, socket in enumerate(sockets)}

            # Expected pattern: partial_<socket_name>_<bs_id>_icap.bin
            self.socket_dict_paths = {}
            for socket in sockets:
                pattern = re.compile(rf"^partial_{re.escape(socket)}_(\d+)_icap\.bin$")
                socket_files = {}
                for entry in Path(self.bitstream_folder).iterdir():
                    m = pattern.match(entry.name)
                    if m:
                        bs_id = int(m.group(1))
                        socket_files[bs_id] = str(entry)
                self.socket_dict_paths[socket] = socket_files

            # Allocate Pynq Buffers for every bitstream
            self.socket_buffers = {}
            for socket, bs_dict in self.socket_dict_paths.items():
                self.socket_buffers[socket] = {}
                for bs_id, path in bs_dict.items():
                    raw = np.fromfile(path, dtype=np.uint32)
                    buf = allocate(shape=(len(raw),), dtype=np.uint32)
                    buf[:] = raw
                    buf.flush()
                    self.socket_buffers[socket][bs_id] = buf

            # Compute Address Layout
            self.max_num_rm = max(
                len(bs_dict.keys()) for bs_dict in self.socket_dict_paths.values()
            )
            self.num_sockets = len(self.socket_dict_paths)
            # Address encoding: [Virtual Socket Manager Select] [Bank Select] [Register Select] [00]
            self.reg_select_shift = 2
            bank0_bits = 1
            bank1_bits = math.ceil(math.log2(self.max_num_rm))  # We assume one trigger per RM
            bank2_bits = math.ceil(math.log2(self.max_num_rm)) + 1
            bank3_bits = math.ceil(math.log2(self.max_num_rm)) + 2
            self.reg_select_bits = max(bank0_bits, bank1_bits, bank2_bits, bank3_bits)
            self.bank_select_shift = self.reg_select_shift + self.reg_select_bits
            self.vsm_select_shift = self.bank_select_shift + 2

            # Initialize the DFX Controller
            for socket in self.socket_dict_paths.keys():
                self.shutdown(vsm=socket)

            for socket, rm in self.socket_buffers.items():
                for bs_id, buf in rm.items():
                    self.set_rm_bs_index(rm_id=bs_id, bs_index=bs_id, clear_bs_index=0, vsm=socket)
                    self.set_bs_address(bs_row=bs_id, address=buf.device_address, vsm=socket)
                    self.set_bs_size(bs_row=bs_id, size=len(buf) * 4, vsm=socket)

            for socket in self.socket_dict_paths.keys():
                self.restart_with_status(vsm=socket, is_full=True, rm_id=0)

        def _map_socket(self, socket_name: str | int) -> int:
            """Map a socket name or integer index to its numeric index."""
            if isinstance(socket_name, int):
                return socket_name
            return self.socket_map[socket_name]

        def _reg_addr(self, vsm: int, bank: int, reg_select: int) -> int:
            """Compute the register address from VSM, bank and register-select fields."""
            return (
                (vsm << self.vsm_select_shift)
                | (bank << self.bank_select_shift)
                | (reg_select << self.reg_select_shift)
            )

        def _extract_bits(self, value: int, high: int, low: int) -> int:
            """Extract a bit field from value between positions high and low (inclusive)."""
            mask = (1 << (high - low + 1)) - 1
            return (value >> low) & mask

        def get_status(self, vsm: str | int) -> dict:
            """Return a status dict for the given virtual socket manager."""
            vsm = self._map_socket(vsm)
            addr = self._reg_addr(vsm, bank=0, reg_select=0)
            raw = self.dfx_controller_inst.read(addr)
            shutdown = bool(self._extract_bits(raw, 7, 7))
            state_val = self._extract_bits(raw, 2, 0)
            err_code = self._extract_bits(raw, 6, 3)
            return {
                "raw": hex(raw),
                "rm_id": self._extract_bits(raw, 23, 8),
                "shutdown": shutdown,
                "error": err_code != 0,
                "error_code": err_code,
                "state": state_val,
            }

        def set_control(
            self, cmd: int, vsm: str | int, byte_field: int = 0, halfword_field: int = 0
        ) -> None:
            """Write a control word to the DFX controller for the given VSM."""
            vsm = self._map_socket(vsm)
            control_value = (
                ((halfword_field & 0xFFFF) << 16) | ((byte_field & 0xFF) << 8) | (cmd & 0xFF)
            )
            addr = self._reg_addr(vsm, bank=0, reg_select=0)
            self.dfx_controller_inst.write(addr, control_value)

        def shutdown(self, vsm: str | int) -> None:
            """Shutdown the given virtual socket manager."""
            self.set_control(0, vsm=vsm)

        def restart_with_status(
            self, vsm: str | int, is_full: bool = False, rm_id: int = 0
        ) -> None:
            """Restart the VSM, optionally with a full reconfiguration for a given RM."""
            byte_field = 1 if is_full else 0
            self.set_control(2, vsm=vsm, byte_field=byte_field, halfword_field=rm_id)

        def set_rm_bs_index(
            self, rm_id: int, bs_index: int, vsm: str | int, clear_bs_index: int = 0
        ) -> None:
            """Map a reconfigurable module ID to a bitstream index in the controller."""
            vsm = self._map_socket(vsm)
            reg_sel = (rm_id << 1) | 0
            addr = self._reg_addr(vsm, bank=2, reg_select=reg_sel)
            value = ((clear_bs_index & 0xFFFF) << 16) | (bs_index & 0xFFFF)
            self.dfx_controller_inst.write(addr, value)

        def set_rm_control(
            self,
            rm_id: int,
            vsm: str | int,
            shutdown_required: int = 0,
            startup_required: int = 0,
            reset_required: int = 0,
            reset_duration: int = 1,
        ) -> None:
            """Write control flags for a reconfigurable module to the controller."""
            vsm = self._map_socket(vsm)
            reg_sel = (rm_id << 1) | 1
            addr = self._reg_addr(vsm, bank=2, reg_select=reg_sel)
            value = (
                (((reset_duration - 1) & 0xFF) << 5)
                | ((reset_required & 0x3) << 3)
                | ((startup_required & 0x1) << 2)
                | (shutdown_required & 0x3)
            )
            self.dfx_controller_inst.write(addr, value)

        def set_bs_id(self, bs_row: int, bs_id: int, vsm: str | int) -> None:
            """Write the bitstream ID for the given row to the controller."""
            vsm = self._map_socket(vsm)
            reg_sel = (bs_row << 2) | 0
            addr = self._reg_addr(vsm, bank=3, reg_select=reg_sel)
            value = bs_id & 0x1
            self.dfx_controller_inst.write(addr, value)

        def set_bs_address(self, bs_row: int, address: int, vsm: str | int) -> None:
            """Write the bitstream memory address for the given row to the controller."""
            vsm = self._map_socket(vsm)
            reg_sel = (bs_row << 2) | 1
            addr = self._reg_addr(vsm, bank=3, reg_select=reg_sel)
            self.dfx_controller_inst.write(addr, address)

        def set_bs_size(self, bs_row: int, size: int, vsm: str | int) -> None:
            """Write the bitstream byte size for the given row to the controller."""
            vsm = self._map_socket(vsm)
            reg_sel = (bs_row << 2) | 2
            addr = self._reg_addr(vsm, bank=3, reg_select=reg_sel)
            self.dfx_controller_inst.write(addr, size)

        def print_status(self, vsm: str | int) -> None:
            """Print the current status of the given virtual socket manager."""
            s = self.get_status(vsm=vsm)
            print(f"VSM {vsm} Status: {s['raw']}")
            print(f"RM ID: {s['rm_id']}")
            print(f"Shutdown: {s['shutdown']}")
            print(f"Error: {s['error']}")
            print(f"State: {s['state']}")

    def get_config_reg(self) -> str:
        """Read and return the ZynqMP configuration register value."""
        os.system("echo 0xffca3008 > /sys/firmware/zynqmp/config_reg")
        result = os.popen("cat /sys/firmware/zynqmp/config_reg").read()
        return result.strip()

    def enable_icap(self) -> None:
        """Enable ICAP as the configuration source."""
        os.system("echo 0xffca3008 0xff 0x0 > /sys/firmware/zynqmp/config_reg")

    def enable_pcap(self) -> None:
        """Enable PCAP as the configuration source."""
        os.system("echo 0xffca3008 0xff 0x1 > /sys/firmware/zynqmp/config_reg")

    def __init__(
        self,
        bitfile_name: str,
        io_shape_dict: dict,
        platform: str = "zynq-iodma",
        fclk_mhz: float = 100.0,
        device: Any = None,
        download: bool = True,
        runtime_weight_dir: str = "runtime_weights/",
        validation_dataset: str | None = None,
        batch_size: int = 1,
        seed: int = 1,
        multidnn_mode: str | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize DMA instrumentation overlay."""
        super().__init__(
            bitfile_name,
            io_shape_dict=io_shape_dict,
            platform=platform,
            fclk_mhz=fclk_mhz,
            device=device,
            download=download,
            runtime_weight_dir=runtime_weight_dir,
            validation_dataset=validation_dataset,
            batch_size=batch_size,
            seed=seed,
        )
        self.multidnn_mode = multidnn_mode

    def set_current_mode(self, mode: str) -> None:
        """Set accelerator mode ('dma' or 'instr')."""
        if self.get_current_mode() != mode:
            self.reset_accelerator()
            val = 1 if mode == "instr" else 0
            self.axi_gpio_0.write(
                offset=self.ip_dict["axi_gpio_0"]["registers"]["GPIO2_DATA"]["address_offset"],
                value=val,
            )

    def get_current_mode(self) -> str:
        """Get accelerator mode."""
        val = self.axi_gpio_0.read(
            offset=self.ip_dict["axi_gpio_0"]["registers"]["GPIO2_DATA"]["address_offset"]
        )
        return "instr" if val == 1 else "dma"

    def throughput_test(self, **kwargs: Any) -> dict:
        """Run throughput test (DMA mode)."""
        self.set_current_mode("dma")
        return super().throughput_test(**kwargs)

    def execute(self, input_npy: np.ndarray | list[np.ndarray]) -> np.ndarray | list[np.ndarray]:
        """Execute (DMA mode)."""
        self.set_current_mode("dma")
        return super().execute(input_npy)

    def experiment_instrumentation(self, **kwargs: Any) -> None:
        """Run instrumentation experiment (instrumentation mode)."""
        self.set_current_mode("instr")
        if self.multidnn_mode == "SelectableWeights":
            selector = self.Selector(self.ip_dict["StreamingDataflowPartition_1_selector"])
            selector.set_schedule(schedule=[1, 1])
            selector.start()
        return super().experiment_instrumentation(**kwargs)

    def validate(self, *args: Any, **kwargs: Any) -> None:
        """Run validation in DMA mode."""
        self.set_current_mode("dma")
        return super().validate(*args, **kwargs)

    def experiment_ma(self, **kwargs: Any) -> int:
        """Run a multi-DNN reconfiguration experiment and save results."""
        report_dir = kwargs.get("report_dir")
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        report = {}
        pr_bitstream_folder = str(Path(self.bitfile_name).parent / "partial_bitstreams")
        socket_prefix = kwargs.get("pr_bitstream_prefix", "StreamingDataflowPartition")
        instr_runtime = kwargs.get("instr_runtime", 1)
        avg_window_size = kwargs.get("avg_window_size", 64)
        num_measurements = kwargs.get("num_measurements", 10)
        # Optional: also measure full vs. partial reconfiguration times.
        # Enable via the driver CLI with: -fk measure_reconfiguration_time=True Bool
        measure_reconfiguration_time = kwargs.get("measure_reconfiguration_time", False)

        self.set_current_mode("instr")

        if self.multidnn_mode != "PartialReconfiguration":
            if self.multidnn_mode == "SelectableWeights":
                # TODO: also excercise different weight sets in this mode..
                pass

            self.start_accelerator(avg_window_size=avg_window_size)
            time.sleep(instr_runtime)
            (
                overflow_err,
                underflow_err,
                _frame,
                checksum,
                min_latency,
                latency,
                interval,
                avg_latency,
                avg_interval,
                run_cycles,
                run_frames,
            ) = self.observe_instrumentation(debug_print=False)
            self.stop_accelerator()
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
                "min_latency_ms": round(min_latency / fclk * 1e3, 6),
                "latency_ms": round(latency / fclk * 1e3, 6),
                "avg_latency_ms": round(avg_latency / fclk * 1e3, 6),
                "throughput_fps": round(fclk / interval) if interval != 0 else 0,
                "avg_throughput_fps": round(fclk / avg_interval) if avg_interval != 0 else 0,
                "run_avg_throughput_fps": round(run_frames / (run_cycles / fclk))
                if run_cycles > 0
                else 0,
                "min_pipeline_depth": round(min_latency / interval, 2) if interval != 0 else 0,
                "pipeline_depth": round(latency / interval, 2) if interval != 0 else 0,
            }
            mode_tag = (self.multidnn_mode or "single").lower()
            reportfile = Path(report_dir) / f"report_{mode_tag}.json"
            with reportfile.open("w") as f:
                json.dump(report, f, indent=2)
            return 0

        pattern = rf".*_{re.escape(socket_prefix)}_(\d+)_"
        socket_names = []
        for entry in Path(pr_bitstream_folder).iterdir():
            match = re.search(pattern, entry.name)
            if match:
                name = f"{socket_prefix}_{match.group(1)}"
                if name not in socket_names:
                    socket_names.append(name)
        socket_names = sorted(socket_names, key=lambda x: int(x.split("_")[-1]))
        self.enable_icap()
        dfx = self.DFXController(
            self.dfx_controller_0, sockets=socket_names, bitstream_folder=pr_bitstream_folder
        )

        # Sweep mux_interval: how many frames each tUSER value is held before the
        # instrumentation wrapper advances to the next one in round-robin order.
        # The dfx_wrapper detects the tUSER change and triggers partial reconfiguration.
        # mux_interval=0 means tUSER stays at 0 (no reconfiguration, baseline measurement).
        mux_intervals = kwargs.get(
            "mux_intervals",
            [
                0,
                200000,
                100000,
                50000,
                20000,
                10000,
                5000,
                2000,
                1000,
                500,
                200,
                100,
                50,
                20,
                10,
                5,
                2,
                1,
            ],
        )

        test_results = {}
        for mux_interval in mux_intervals:
            # reset instrumentation and accelerator (not DFX controller) for clean measurement:
            self.reset_accelerator()
            self.set_current_mode("instr")  # need to set FINN_switch mode again after reset
            self.start_accelerator(avg_window_size=avg_window_size, mux_interval=mux_interval)
            samples = []
            any_error = False
            for _ in range(num_measurements):
                time.sleep(instr_runtime)
                (
                    overflow_err,
                    underflow_err,
                    _frame,
                    checksum,
                    min_latency,
                    latency,
                    interval,
                    avg_latency,
                    avg_interval,
                    run_cycles,
                    run_frames,
                ) = self.observe_instrumentation(debug_print=False)
                any_error = any_error or overflow_err or underflow_err or interval == 0
                samples.append(
                    (
                        min_latency,
                        latency,
                        interval,
                        avg_latency,
                        avg_interval,
                        checksum,
                        run_cycles,
                        run_frames,
                    )
                )
            self.stop_accelerator()
            time.sleep(1)  # ensure accelerator is flushed and DFX controller is idle before reset
            fclk = self.fclk_mhz_actual * 1e6
            n = len(samples)
            avg_min_latency = sum(s[0] for s in samples) / n
            avg_latency_mean = sum(s[1] for s in samples) / n
            avg_interval_mean = sum(s[2] for s in samples) / n
            avg_avg_latency = sum(s[3] for s in samples) / n
            avg_avg_interval = sum(s[4] for s in samples) / n
            # Use last checksum (frame counter) as reference
            last_checksum = samples[-1][5]
            # run_cycles/run_frames are cumulative since start; use last sample
            last_run_cycles = samples[-1][6]
            last_run_frames = samples[-1][7]
            test_results[mux_interval] = {
                "mux_interval": mux_interval,
                "error": any_error,
                "checksum": last_checksum,
                "num_measurements": num_measurements,
                "min_latency_cycles": avg_min_latency,
                "latency_cycles": avg_latency_mean,
                "interval_cycles": avg_interval_mean,
                "avg_latency_cycles": avg_avg_latency,
                "avg_interval_cycles": avg_avg_interval,
                "run_cycles": last_run_cycles,
                "run_frames": last_run_frames,
                "frequency_mhz": round(self.fclk_mhz_actual),
                "min_latency_ms": round(avg_min_latency / fclk * 1e3, 6),
                "latency_ms": round(avg_latency_mean / fclk * 1e3, 6),
                "avg_latency_ms": round(avg_avg_latency / fclk * 1e3, 6),
                "throughput_fps": round(fclk / avg_interval_mean) if avg_interval_mean != 0 else 0,
                "avg_throughput_fps": round(fclk / avg_avg_interval)
                if avg_avg_interval != 0
                else 0,
                "run_avg_throughput_fps": round(last_run_frames / (last_run_cycles / fclk))
                if last_run_cycles > 0
                else 0,
                "min_pipeline_depth": round(avg_min_latency / avg_interval_mean, 2)
                if avg_interval_mean != 0
                else 0,
                "pipeline_depth": round(avg_latency_mean / avg_interval_mean, 2)
                if avg_interval_mean != 0
                else 0,
            }
        report["test"] = test_results
        del dfx

        if measure_reconfiguration_time:
            # PCAP test - dry run to buffer bitstreams in RAM
            self.enable_pcap()

            full_bs = []
            full_bs_pattern = re.compile(r"^config_(\d+)\.bit$")
            for entry in sorted(Path(pr_bitstream_folder).iterdir()):
                m = full_bs_pattern.match(entry.name)
                if m:
                    full_bs += [str(entry)]

            for p in full_bs:
                pb = Bitstream(p, None, False)
                pb.download()

            full_configuration_time = []
            for _ in range(num_measurements):
                for p in full_bs:
                    pb = Bitstream(p, None, False)
                    start = time.time()
                    pb.download()
                    end = time.time()
                    full_configuration_time.append(end - start)
            fct = sorted(full_configuration_time)
            fn = len(fct)
            full_configuration_report = {
                "avg": sum(fct) / fn,
                "min": fct[0],
                "q1": fct[fn // 4],
                "q3": fct[(3 * fn) // 4],
                "max": fct[-1],
                "bitfile_sizes_bytes": {Path(p).name: Path(p).stat().st_size for p in full_bs},
            }
            report["full_configuration"] = full_configuration_report

            partial_bs_by_rm = {}
            partial_bs_pattern = re.compile(
                rf"^partial_{re.escape(socket_prefix)}_(\d+)_(\d+)\.bit$"
            )
            for entry in sorted(Path(pr_bitstream_folder).iterdir()):
                m = partial_bs_pattern.match(entry.name)
                if m:
                    socket_id = int(m.group(1))
                    rm_id = int(m.group(2))
                    partial_bs_by_rm.setdefault(rm_id, []).append((socket_id, str(entry)))
            for rm_id in partial_bs_by_rm:
                partial_bs_by_rm[rm_id].sort(key=lambda t: t[0])

            # Dry run
            for _rm_id, sockets in sorted(partial_bs_by_rm.items()):
                for _, path in sockets:
                    pb = Bitstream(path, None, True)
                    pb.download()

            # Measure reconfiguration time for one full id (all sockets for a given RM id)
            partial_configuration_time = []
            for _ in range(num_measurements):
                for _rm_id, sockets in sorted(partial_bs_by_rm.items()):
                    start = time.time()
                    for _, path in sockets:
                        pb = Bitstream(path, None, True)
                        pb.download()
                    end = time.time()
                    partial_configuration_time.append(end - start)
            pct = sorted(partial_configuration_time)
            pn = len(pct)
            partial_configuration_report = {
                "avg": sum(pct) / pn,
                "min": pct[0],
                "q1": pct[pn // 4],
                "q3": pct[(3 * pn) // 4],
                "max": pct[-1],
                "bitfile_sizes_bytes": {
                    Path(path).name: Path(path).stat().st_size
                    for sockets in partial_bs_by_rm.values()
                    for _, path in sockets
                },
            }
            report["partial_configuration"] = partial_configuration_report

        report["fclk_mhz"] = self.fclk_mhz
        reportfile = Path(report_dir) / "report_pr.json"
        with reportfile.open("w") as f:
            json.dump(report, f, indent=2)
        return 0

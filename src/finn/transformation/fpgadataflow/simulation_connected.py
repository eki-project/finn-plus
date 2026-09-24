"""Node connected parallel simulations."""

import json
import math
import os
import pandas as pd
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback
from ast import literal_eval
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from rich.console import Console
from shutil import which
from threading import Barrier
from typing import Any, cast

import finn.transformation.fpgadataflow.fifo_depth_search as _fifo_depth_search
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.transformation.fpgadataflow.fifo_depth_search import (
    BRAM_FIFO_PIPELINE_OVERHEAD,
    MinimizationOrder,
    calculate_bram_blocks,
    calculate_bram_depth_range,
    count_bram_sub_fifos,
    get_valid_block_counts,
    minimize_fifo_depth,
    needs_minimization,
    safe_bram_starting_depth,
)
from finn.transformation.fpgadataflow.simulation import (
    Simulation,
    SimulationController,
    SimulationType,
    store_fifo_data,
)
from finn.transformation.fpgadataflow.simulation_build import (
    SimulationCommMode,
    distribute_mpi_ranks,
)
from finn.util.basic import getHWCustomOp, make_build_dir
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log

# The BRAM overhead constant, the cost model and the search skeleton are shared with the
# TEG-based abstract simulation and live in fifo_depth_search; they are re-exported here for
# backwards compatibility.
_count_bram_sub_fifos = count_bram_sub_fifos
_safe_bram_starting_depth = safe_bram_starting_depth
calculate_srl16e_depth_range = _fifo_depth_search.calculate_srl16e_depth_range
calculate_srl16e_luts = _fifo_depth_search.calculate_srl16e_luts
calculate_uram_blocks = _fifo_depth_search.calculate_uram_blocks
calculate_uram_depth_range = _fifo_depth_search.calculate_uram_depth_range

# Number of retries (at ~0.1s apart) when connecting to each MPI rank's control socket,
# i.e. ~120s total per rank before giving up.
MPI_SOCKET_CONNECT_RETRIES = 1200


@dataclass
class MpiSimConfig:
    """Hybrid-MPI simulation settings, bundled to avoid threading a growing list of
    individual keyword arguments through NodeConnectedSimulation.
    """

    sim_comm_mode: str = SimulationCommMode.SHM.value
    mpi_launcher: str = "mpirun"
    mpi_oversubscribe: bool = True
    mpi_args: str = ""

    @classmethod
    def from_build_config(cls, cfg: DataflowBuildConfig) -> "MpiSimConfig":
        """Build an MpiSimConfig from the relevant fields of a DataflowBuildConfig."""
        return cls(
            sim_comm_mode=cfg.fifosim_comm_mode,
            mpi_launcher=cfg.fifosim_mpi_launcher,
            mpi_oversubscribe=cfg.fifosim_mpi_oversubscribe,
            mpi_args=cfg.fifosim_mpi_args,
        )


@dataclass
class MpiLaunchParams:
    """Resolved MPI launch parameters for NodeConnectedSimulationController.

    Distinct from MpiSimConfig: this also carries state resolved at runtime (whether
    MPI is actually needed, and the model's host/rank assignment), not just the
    user-configured settings.
    """

    use_mpi_launcher: bool = False
    mpi_launcher: str = "mpirun"
    mpi_hosts: str | None = None
    mpi_host_map: dict[str, str] | None = None
    mpi_oversubscribe: bool = True
    mpi_args: str = ""


class NodeConnectedSimulationController(SimulationController):
    """Run simulations for node connected cases."""

    def __init__(
        self,
        parallel_simulations: int,
        names: list[str],
        binaries: list[Path],
        console: Console,
        shm_prefix: str,
        mpi_params: MpiLaunchParams | None = None,
        poll_interval: float = 1.0,
        with_progressbar: bool = True,
    ) -> None:
        """Set up node connected simulation."""
        super().__init__(
            parallel_simulations,
            names,
            binaries,
            console,
            poll_interval,
            with_progressbar,
            enable_core_pinning=True,
        )
        mpi_params = mpi_params or MpiLaunchParams()
        # Synchronization barrier for configuration phase
        self.sync_barrier: Barrier | None = None
        self.shm_prefix = shm_prefix
        self.use_mpi_launcher = mpi_params.use_mpi_launcher
        self.mpi_launcher = mpi_params.mpi_launcher
        self.mpi_hosts = mpi_params.mpi_hosts
        self.mpi_oversubscribe = mpi_params.mpi_oversubscribe
        self.mpi_args = mpi_params.mpi_args
        self.mpi_launcher_proc: subprocess.Popen | None = None
        self.mpi_host_map = mpi_params.mpi_host_map
        self.mpi_launcher_stdout: Any = None
        self.mpi_launcher_stderr: Any = None
        self.mpi_launcher_pgid: int | None = None
        for binary in binaries:
            if not binary.exists():
                console.log(f"Binary {binary} does not exist!")
                raise FINNUserError(f"Binary {binary} does not exist!")

    def _start_mpi_group(self) -> None:
        """Start one MPI job that launches one simulation backend per rank."""
        if self.mpi_host_map is None:
            raise FINNInternalError(
                "NodeConnectedSimulationController was constructed with "
                "use_mpi_launcher=True but mpi_host_map=None."
            )

        launcher_path = which(self.mpi_launcher)
        if launcher_path is None:
            raise FINNUserError(
                f"Requested MPI launcher '{self.mpi_launcher}' was not found in PATH."
            )

        def _build_launcher_cmd(
            launcher_script: Path,
            mapping_file: Path,
            endpoint_map: dict[str, list[tuple[int, str]]],
        ) -> list[str]:
            """Build the mpirun command line that launches mpi_run_script.py per rank."""
            cmd = [launcher_path]
            # Use slot-based mapping by default because FINN rank allocation is based on
            # logical CPU slots, while MPI implementations often default to by-core mapping.
            # This avoids launch failures like "failed to map ... Mapping policy: BYCORE".
            cmd.extend(["--map-by", "slot", "--bind-to", "none"])
            # In SLURM allocations, OpenMPI may infer available slots from --ntasks
            # (often just a small number), while FINN launches one MPI rank per layer.
            # Oversubscribe by default so mapping does not fail with "Out of resource".
            if self.mpi_oversubscribe:
                cmd.append("--oversubscribe")
            # The number of ranks per host must match the number of endpoints assigned to
            # that host in endpoint_map, so that (with --map-by slot fill-first) local rank
            # i on a host always corresponds to endpoint_map[host][i]. Using a fixed slot
            # count (e.g. local core count) here would misalign ranks with endpoints
            # whenever hosts don't get an even share of the simulated nodes.
            hosts_with_slots = ",".join(f"{host}:{len(v)}" for host, v in endpoint_map.items())
            cmd.extend(["-H", hosts_with_slots])
            mpi_extra_args = self.mpi_args.strip()
            if mpi_extra_args != "":
                cmd.extend(shlex.split(mpi_extra_args))
            # World size must be the total number of simulation endpoints, not the size of
            # the Slurm/MPI allocation FINN itself runs in (e.g. SLURM_NTASKS): FINN launches
            # one MPI rank per simulated node, which is typically far more than the number of
            # allocated Slurm tasks.
            size = sum(len(v) for v in endpoint_map.values())
            cmd.extend(
                [
                    "-np",
                    str(size),
                    sys.executable,
                    str(launcher_script),
                    str(mapping_file),
                ]
            )
            return cmd

        mapping_file = self.logdir / "mpi_rank_mapping.json"

        launcher_script = (
            Path(__file__).parent.parent.parent / "templates" / "mpi_scripts" / "mpi_run_script.py"
        )

        stdout_log = self.logdir / "mpi_launcher_stdout.log"
        stderr_log = self.logdir / "mpi_launcher_stderr.log"

        get_free_ports_script = (
            Path(__file__).parent.parent.parent / "templates" / "mpi_scripts" / "get_free_ports.py"
        )

        # Preserve first-occurrence order (matches self.names, which is grouped in
        # contiguous per-host blocks). A plain set() here would give hash-randomized
        # order, so self.sockets (built by iterating hosts below) would end up
        # misaligned with self.names, and _run_binary's self.sockets[self.names.index(name)]
        # would talk to the wrong rank's control socket.
        hosts = list(dict.fromkeys(self.mpi_host_map.values()))
        host_map: dict[str, list[str]] = {}
        for host in hosts:
            host_map[host] = []
            for k, v in self.mpi_host_map.items():
                if v == host:
                    host_map[v].append(k)
        count = 0
        for v in host_map.values():
            count = max(count, len(v))
        if count == 0:
            raise FINNInternalError(
                "No simulation nodes were assigned to any MPI worker host "
                "(mpi_host_map is non-empty but every host list is empty)."
            )

        probe_cmd = [launcher_path]
        nodes = len(hosts)
        probe_cmd.extend(
            [
                "--map-by",
                "ppr:1:node",
                "--prtemca",
                "prte_silence_shared_fs",
                "1",
                "-np",
                str(nodes),
                sys.executable,
                str(get_free_ports_script),
                "--start",
                "8000",
                "--end",
                "65535",
                "--count",
                str(count),
            ]
        )
        res = subprocess.run(probe_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise FINNInternalError(
                f"Failed to probe free ports for MPI simulation: {res.stderr.strip()}, "
                f"stdout: {res.stdout.strip()}"
            )
        # Returns one line per node, each line is a JSON object with a "ports" list.
        raw = res.stdout.strip()
        host_to_ports: dict[str, list[int]] = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not obj.get("ok", False):
                raise FINNInternalError(
                    f"Failed to probe free ports for MPI simulation: "
                    f"{obj.get('error', 'unknown error')}"
                )
            host_to_ports[obj["hostname"]] = obj["ports"]

        # Exact (not substring) lookup: matching by "name in str(binary_path)" would
        # mis-assign a binary whenever one node's name is a substring of another's
        # directory name (e.g. "MVAU_hls_1" is a substring of ".../MVAU_hls_10_.../").
        name_to_binary = dict(zip(self.names, self.binaries, strict=True))

        endpoint_map: dict[str, list[tuple[int, str]]] = {}
        for k, v in host_map.items():
            if k not in host_to_ports:
                raise FINNInternalError(
                    f"Free-port probe did not report a result for host {k!r}. "
                    f"Hosts probed: {sorted(host_to_ports.keys())}."
                )
            endpoint_map[k] = []
            for i, name in enumerate(v):
                port = host_to_ports[k][i]
                binary = name_to_binary.get(name)
                if binary is None:
                    raise FINNInternalError(
                        f"No simulation binary found for node {name!r} while building "
                        "the MPI rank mapping."
                    )
                endpoint_map[k].append((port, str(binary)))
        mapping_file.write_text(
            json.dumps(
                endpoint_map,
                indent=2,
            )
        )

        launcher_cmd = _build_launcher_cmd(launcher_script, mapping_file, endpoint_map)

        self.mpi_launcher_stdout = stdout_log.open("w")
        self.mpi_launcher_stderr = stderr_log.open("w")
        self.mpi_launcher_proc = subprocess.Popen(
            launcher_cmd,
            stdout=self.mpi_launcher_stdout,
            stderr=self.mpi_launcher_stderr,
            text=True,
            start_new_session=True,
        )
        with suppress(ProcessLookupError):
            self.mpi_launcher_pgid = os.getpgid(self.mpi_launcher_proc.pid)

        # Connect to each rank's socket server.
        self.sockets = []
        max_retries = MPI_SOCKET_CONNECT_RETRIES
        for host, endpoint in endpoint_map.items():
            for port, _bin in endpoint:
                endpoint_port = cast("int", port)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                connected = False
                for _ in range(max_retries):
                    if self.mpi_launcher_proc.poll() is not None:
                        raise FINNInternalError(
                            "MPI launcher process exited before all sockets connected"
                        )
                    try:
                        sock.connect((host, endpoint_port))
                        connected = True
                        break
                    except (ConnectionRefusedError, socket.gaierror, OSError):
                        time.sleep(0.1)
                if not connected:
                    self._cleanup_sockets()
                    raise FINNInternalError(
                        f"Failed to connect to MPI rank control endpoint: {host}:{endpoint_port}"
                    )
                self.sockets.append((sock, f"tcp://{host}:{endpoint_port}"))

        # Provide one process entry for existing error handling path.
        if self.mpi_launcher_proc is not None:
            self.processes.append(
                (
                    self.mpi_launcher_proc,
                    self.mpi_launcher_stdout,
                    self.mpi_launcher_stderr,
                )
            )

    def _cleanup_sockets(self) -> None:
        """Close sockets and terminate processes; MPI mode has custom cleanup."""
        if not self.use_mpi_launcher:
            super()._cleanup_sockets()
            return

        # Send stop command to all ranks (best effort, non-blocking ack handling).
        # In hybrid MPI mode some ranks may be blocked in MPI recv/send and therefore
        # cannot acknowledge stop immediately. Waiting for each ack can deadlock the
        # controller and prevent subsequent iterations from starting.
        for i in range(len(self.sockets)):
            with suppress(Exception):
                self._send_command(i, "stop", {})

        # Close sockets.
        for sock, _endpoint in self.sockets:
            with suppress(Exception):
                sock.shutdown(socket.SHUT_RDWR)
            with suppress(Exception):
                sock.close()

        self.sockets = []

        # Let launcher/ranks exit gracefully first; then force-stop full process
        # group if anything is still alive.
        if self.mpi_launcher_proc is not None:
            try:
                self.mpi_launcher_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if self.mpi_launcher_pgid is not None:
                    with suppress(ProcessLookupError):
                        os.killpg(self.mpi_launcher_pgid, signal.SIGTERM)
                else:
                    with suppress(ProcessLookupError):
                        self.mpi_launcher_proc.terminate()
                try:
                    self.mpi_launcher_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    if self.mpi_launcher_pgid is not None:
                        with suppress(ProcessLookupError):
                            os.killpg(self.mpi_launcher_pgid, signal.SIGKILL)
                    else:
                        with suppress(ProcessLookupError):
                            self.mpi_launcher_proc.kill()
                    with suppress(Exception):
                        self.mpi_launcher_proc.wait(timeout=3)

        if self.mpi_launcher_stdout is not None:
            self.mpi_launcher_stdout.close()
        if self.mpi_launcher_stderr is not None:
            self.mpi_launcher_stderr.close()

        self.processes = []
        self.mpi_launcher_proc = None
        self.mpi_launcher_stdout = None
        self.mpi_launcher_stderr = None
        self.mpi_launcher_pgid = None

    def _cleanup_shm_resources(self) -> None:
        """Remove any existing shared memory segments and semaphores from /dev/shm."""
        if self.shm_prefix is None:
            return
        try:
            removed_count = 0
            for filepath in Path("/dev/shm").glob("*"):
                try:
                    if not filepath.name.startswith(self.shm_prefix):
                        continue
                    filepath.unlink()
                    removed_count += 1
                except (FileNotFoundError, PermissionError):
                    # File might already be removed or we don't have permission
                    pass

            if removed_count > 0:
                log.info(f"Cleaned up {removed_count} existing shared memory resources")
        except Exception as e:
            # Don't fail if cleanup fails - just log it
            self.console.log(f"Warning: Error during shared memory cleanup: {e}")

    def run(
        self,
        depth: list[list[int]] | None = None,
        output_json: Path | None = None,
        max_cycles: int | None = None,
        fifo_first_valid_cycles: list[list[int]] | None = None,
    ) -> dict[str, list[int]]:
        """Run the simulation entirely with the given depth and sample count.

        Args:
            depth: FIFO depth to configure for simulations.
            samples: Number of samples to simulate.
            output_json: Optional path to write merged simulation data as JSON.
            max_cycles: Max cycles
            fifo_first_valid_cycles: First valid cycle for each FIFO (used for timeout detection)

        Returns:
            Dictionary mapping simulation names to their FIFO utilization arrays.
        """
        futures: list[Future] = []
        fifo_results: dict[str, list[int]] = {}
        cycles_results: dict[str, int] = {}
        samples_results: dict[str, int] = {}
        intervals_results: dict[str, list[int]] = {}
        timeout_result = False
        fifo_depths: dict[str, list[int]] = {}
        fifo_cycles_until_first_valid_results: dict[str, list[int]] = {}
        latency_cycles_results: dict[str, list[int]] = {}

        # Clean up any existing shared memory resources before starting
        self._cleanup_shm_resources()

        # Ensure a fresh stop state for this run.
        with self.stop_lock:
            self.should_stop = False

        # Initialize barrier for all simulations to synchronize after configuration
        self.sync_barrier = Barrier(len(self.names))

        if self.use_mpi_launcher:
            self._start_mpi_group()

        if self.progress is not None:
            self.progress.start()
        try:
            with ThreadPoolExecutor(self.workers) as pool:
                for i, (name, binary) in enumerate(zip(self.names, self.binaries, strict=True)):
                    is_last_node = i == len(self.names) - 1
                    is_special_for_display = i == 0 or is_last_node
                    futures.append(
                        pool.submit(
                            self._run_binary,
                            binary,
                            name,
                            depth[i] if depth is not None else None,
                            is_last_node,  # Only last node has no output FIFOs
                            is_special_for_display,  # First and last get special coloring
                            max_cycles,
                            fifo_first_valid_cycles[i]
                            if fifo_first_valid_cycles is not None
                            else None,
                        )
                    )

                # Wait for first completion or error
                from concurrent.futures import FIRST_COMPLETED, wait

                all_futures = list(futures)  # Keep track of all futures
                while futures:
                    done, futures_s = wait(futures, return_when=FIRST_COMPLETED)
                    futures = list(futures_s)  # Remaining futures that are still running

                    # Check if any completed task indicates we should stop
                    for future in done:
                        try:
                            result = future.result()  # This will raise if there was an exception
                            if result is not None:
                                (
                                    sim_name,
                                    fifo_util,
                                    cycles,
                                    samps,
                                    intervals,
                                    timeout,
                                    fifo_depth,
                                    fifo_cycles_until_first_valid,
                                    latency_cycles,
                                ) = result
                                fifo_depths[sim_name] = fifo_depth
                                fifo_results[sim_name] = fifo_util
                                cycles_results[sim_name] = cycles
                                samples_results[sim_name] = samps
                                intervals_results[sim_name] = intervals
                                fifo_cycles_until_first_valid_results[
                                    sim_name
                                ] = fifo_cycles_until_first_valid
                                latency_cycles_results[sim_name] = latency_cycles
                                timeout_result = timeout_result or timeout
                        except Exception as e:  # noqa
                            self.console.log(f"Simulation failed: {e}")
                            # Set stop flag and break
                            with self.stop_lock:
                                self.should_stop = True
                            break

                    # If we should stop, signal all remaining simulations
                    with self.stop_lock:
                        if self.should_stop:
                            # Don't cancel - let them finish with early stop
                            break

                # Wait for all futures to complete and collect their results
                pool.shutdown(wait=True)
                for future in all_futures:
                    if not future.done():
                        continue
                    try:
                        result = future.result()
                        if result is not None:
                            (
                                sim_name,
                                fifo_util,
                                cycles,
                                samps,
                                intervals,
                                timeout,
                                fifo_depth,
                                fifo_cycles_until_first_valid,
                                latency_cycles,
                            ) = result
                            # Only update if not already collected
                            if sim_name not in fifo_results:
                                fifo_cycles_until_first_valid_results[
                                    sim_name
                                ] = fifo_cycles_until_first_valid
                                fifo_depths[sim_name] = fifo_depth
                                fifo_results[sim_name] = fifo_util
                                cycles_results[sim_name] = cycles
                                samples_results[sim_name] = samps
                                intervals_results[sim_name] = intervals
                                latency_cycles_results[sim_name] = latency_cycles
                                timeout_result = timeout_result or timeout
                    except Exception as e:
                        self.console.log(f"Error collecting result: {e}")

                # Detect nodes whose _run_binary returned None (subprocess
                # crash / unhandled exception).  Their names were never inserted into
                # fifo_results, so the merged JSON would contain empty 'intervals' lists
                # for those nodes.  _check_performance would then silently return False
                # (no degradation detected) and the minimisation algorithm would treat a
                # failed simulation as a successful one.  Mark the run as timed-out so
                # that _test_depth correctly rejects the candidate depth.
                missing_nodes = [name for name in self.names if name not in fifo_results]
                if missing_nodes:
                    self.console.log(
                        f"[bold red]WARNING: simulation results missing for node(s) "
                        f"{missing_nodes} (subprocess likely crashed). "
                        f"Marking run as timed-out to prevent false-success "
                        f"classification.[/bold red]"
                    )
                    timeout_result = True
        finally:
            if self.progress is not None:
                self.progress.stop()
            self._cleanup_sockets()

        # Merge all simulation data
        if output_json is not None:
            merged_data = {
                "simulations": [
                    {
                        "name": name,
                        "fifo_utilization": fifo_results.get(name, []),
                        "fifo_depth": fifo_depths.get(name, []),
                        "cycles": cycles_results.get(name, 0),
                        "samples": samples_results.get(name, 0),
                        "intervals": intervals_results.get(name, []),
                        "fifo_cycles_until_first_valid": fifo_cycles_until_first_valid_results.get(
                            name, []
                        ),
                        "latency_cycles": latency_cycles_results.get(name, []),
                    }
                    for name in self.names
                ],
                "depth_configured": depth,
                "timeout_occurred": timeout_result,
            }
            output_json.write_text(json.dumps(merged_data, indent=2))

        return fifo_results

    def _run_binary(
        self,
        binary: Path,
        name: str | None,
        depth: list[int] | None = None,
        is_last_node: bool = False,
        is_special_for_display: bool = False,
        max_cycles: int | None = None,
        fifo_first_valid_cycles: list[int] | None = None,
    ) -> tuple[str, list[int], int, int, list[int], bool, list[int], list[int], list[int]] | None:
        """Run the specified simulation binary in a new subprocess and communicate with it.

        Args:
            binary: Path to simulation binary
            name: Name of simulation node
            depth: List of FIFO depths for this node's output FIFOs
            is_last_node: True if this is the last node (no output FIFOs to configure)
            is_special_for_display: True if this node should get special color in logs
            max_cycles: Maximum cycles to simulate
            fifo_first_valid_cycles: First valid cycle for each FIFO (used for timeout detection)

        Returns:
            Tuple of (simulation_name, fifo_utilization, cycles, samples, intervals, timeout,
            fifo_depth, fifo_cycles_until_first_valid) on success,
            None on failure.
        """
        cwd = binary.parent
        if name is None:
            name = cwd.name.replace("rtlsim_", "")

        process_index = self.names.index(name)

        with (self.logdir / f"{name}_{process_index}_of_{self.total}.txt").open("w+") as logfile:

            def _print(msg: str, color: str = "green") -> None:
                """Return formatted print."""
                if self.progress is None:
                    if is_special_for_display:
                        color = "orange3"
                    if "ERROR" in msg:
                        color = "red"
                    log.debug(
                        f"[bold {color}]{name:<35}"
                        f"[/bold {color}][cornflower_blue]{process_index} "
                        f"/ {len(self.names) - 1}[/cornflower_blue] {msg:<35}"
                    )
                logfile.write(f"{msg}\n")
                logfile.flush()

            try:
                # Start the simulation process with socket communication
                proc_idx = (
                    process_index
                    if self.use_mpi_launcher
                    else self._start_process(binary, process_index)
                )

                # Send configuration commands
                # Last node has no output FIFOs, so don't configure FIFO depths
                config_payload: dict[str, list[int] | int] = {}
                if not is_last_node and depth is not None:
                    config_payload["fifo_depth"] = depth
                if max_cycles is not None:
                    config_payload["max_cycles"] = max_cycles
                if not is_last_node and fifo_first_valid_cycles is not None:
                    config_payload["fifo_first_valid_cycles"] = fifo_first_valid_cycles

                response = self._send_and_receive(proc_idx, "configure", config_payload)

                if not response or response.get("status") != "success":
                    error_msg = (
                        response.get("message", "Unknown error") if response else "No response"
                    )
                    _print(f"Configuration failed: {error_msg}", "red")
                    return None

                # Wait for all simulations to complete configuration before starting
                _print("Waiting for all simulations to complete configuration...")
                if self.sync_barrier is not None:
                    self.sync_barrier.wait()
                _print("All simulations configured, starting...")

                # Start the simulation
                response = self._send_and_receive(proc_idx, "start", {})

                if not response or response.get("status") != "success":
                    error_msg = (
                        response.get("message", "Unknown error") if response else "No response"
                    )
                    _print(f"Failed to start simulation: {error_msg}", "red")
                    return None

                cycles = 0
                samps = 0
                intervals: list[int] = []
                timeout = False
                fifo_util: list[int] = []
                fifo_depth: list[int] = []
                fifo_cycles_until_first_valid: list[int] = []
                latency_cycles: list[int] = []

                # Poll for status updates
                while True:
                    # Check if we should stop early
                    with self.stop_lock:
                        if self.should_stop:
                            try:
                                stop_response = self._send_and_receive(proc_idx, "stop", {})
                            except (BrokenPipeError, ConnectionResetError, FINNInternalError):
                                # Process may have already exited - that's ok during shutdown
                                stop_response = None
                            if stop_response:
                                cycles = stop_response.get("cycles", 0)
                                samps = stop_response.get("samples", 0)
                                fifo_util = stop_response.get("fifo_utilization", [])
                                intervals = stop_response.get("intervals", [])
                                fifo_depth = stop_response.get("fifo_depth", [])
                                timeout = stop_response.get("timeout", False)
                                fifo_cycles_until_first_valid = stop_response.get(
                                    "fifo_cycles_until_first_valid", []
                                )
                                latency_cycles = stop_response.get("latency_cycles", [])
                                if fifo_util:
                                    logfile.write(f"Final FIFO utilization: {fifo_util}\n")
                            return (
                                name,
                                fifo_util,
                                cycles,
                                samps,
                                intervals,
                                timeout,
                                fifo_depth,
                                fifo_cycles_until_first_valid,
                                latency_cycles,
                            )
                    time.sleep(self.poll_interval)

                    response = self._send_and_receive(proc_idx, "status", {})

                    if not response:
                        _print("Lost connection to simulation", "red")
                        with self.stop_lock:
                            self.should_stop = True
                        raise FINNInternalError("Lost connection to simulation")

                    state = response.get("state", "unknown")

                    if state == "finished" or state == "timeout":
                        cycles = response.get("cycles", 0)
                        samps = response.get("samples", 0)
                        fifo_util = response.get("fifo_utilization", [])
                        fifo_depth = response.get("fifo_depth", [])
                        intervals = response.get("intervals", [])
                        timeout = response.get("timeout", False)
                        fifo_cycles_until_first_valid = response.get(
                            "fifo_cycles_until_first_valid", []
                        )
                        latency_cycles = response.get("latency_cycles", [])
                        # Stop all simulations once any node reaches terminal
                        # state. In node-connected mode only the last node can
                        # naturally reach "finished"; other nodes must be
                        # terminated by controller stop broadcast.
                        with self.stop_lock:
                            self.should_stop = True
                        break

                    if state == "running":
                        # Update progress if available
                        cycles = response.get("cycles", 0)

                    if state == "error":
                        error_msg = response.get("message", "Unknown error")
                        _print(f"Simulation error: {error_msg}", "red")
                        # Signal other simulations to stop
                        with self.stop_lock:
                            self.should_stop = True
                        raise FINNInternalError(f"Simulation error: {error_msg}")

                # Stop the simulation
                stop_response = self._send_and_receive(proc_idx, "stop", {})

                if stop_response:
                    fifo_util = stop_response.get("fifo_utilization", [])
                    fifo_depth = stop_response.get("fifo_depth", [])
                    cycles = stop_response.get("cycles", 0)
                    samps = stop_response.get("samples", 0)
                    fifo_cycles_until_first_valid = stop_response.get(
                        "fifo_cycles_until_first_valid", []
                    )
                    latency_cycles = stop_response.get("latency_cycles", [])
                    if fifo_util:
                        logfile.write(f"Final FIFO utilization: {fifo_util}\n")

                return (
                    name,
                    fifo_util,
                    cycles,
                    samps,
                    intervals,
                    timeout,
                    fifo_depth,
                    fifo_cycles_until_first_valid,
                    latency_cycles,
                )

            except Exception as e:
                self.console.log(f"Exception caught during simulation execution ({name}): {e}")
                self.console.log(traceback.format_exc())
                logfile.write(f"Exception: {e}\n")
                logfile.write(traceback.format_exc())
                with self.stop_lock:
                    self.should_stop = True
                return None


class NodeConnectedSimulation(Simulation):
    """Run node-connected simulations for all layers in parallel."""

    def __init__(
        self,
        model: ModelWrapper,
        simulation_type: SimulationType,
        fpgapart: str,
        clk_ns: float,
        functional_sim: bool,
        shm_prefix: str | None,
        workers: int | None = None,
        max_qsrl_depth: int = 256,
        performance_sim: bool = False,
        mpi_config: MpiSimConfig | None = None,
    ) -> None:
        """Initialize node-connected simulation.

        mpi_config is only used to supply defaults: if the model already has
        sim_comm_mode metadata set (e.g. by a prior BuildSimulation run), that value
        wins over mpi_config.sim_comm_mode.
        """
        mpi_config = mpi_config or MpiSimConfig()
        sim_comm_mode = (
            model.get_metadata_prop("sim_comm_mode") or mpi_config.sim_comm_mode
        ).lower()
        allowed_modes = {SimulationCommMode.SHM.value, SimulationCommMode.HYBRID_MPI.value}
        if sim_comm_mode not in allowed_modes:
            raise FINNUserError(
                f"Unsupported simulation communication mode '{sim_comm_mode}'. "
                f"Allowed values are: {', '.join(sorted(allowed_modes))}."
            )
        model.set_metadata_prop("sim_comm_mode", sim_comm_mode)
        self.sim_comm_mode = sim_comm_mode

        def _parse_list_metadata(name: str) -> list[Any] | None:
            """Return the model metadata prop `name` parsed as a Python list literal."""
            val = model.get_metadata_prop(name)
            if val is None:
                return None
            parsed = literal_eval(val)
            if not isinstance(parsed, list):
                raise FINNUserError(
                    f"Expected model metadata '{name}' to be a Python list literal."
                )
            return parsed

        # Computes and stores sim_rank_map/sim_worker_map/sim_host_map/sim_mpi_hosts
        # metadata if missing or stale; a no-op otherwise (e.g. already computed by
        # BuildSimulation, which normally runs before this).
        distribute_mpi_ranks(model)

        input_channel_types = _parse_list_metadata("sim_input_channel_types")
        output_channel_types = _parse_list_metadata("sim_output_channel_types")
        has_mpi_from_channel_metadata = False
        if input_channel_types is not None and any(ch == "mpi" for ch in input_channel_types):
            has_mpi_from_channel_metadata = True
        if output_channel_types is not None and any(ch == "mpi" for ch in output_channel_types):
            has_mpi_from_channel_metadata = True

        has_mpi_border_channels_meta = model.get_metadata_prop("sim_has_mpi_border_channels")
        has_mpi_from_border_meta: bool | None = None
        if has_mpi_border_channels_meta is not None:
            has_mpi_from_border_meta = has_mpi_border_channels_meta.lower() == "true"

        # Prefer exact per-channel metadata when available, because it directly captures
        # whether this node model has any MPI links. Fall back to coarse border metadata
        # for backward compatibility.
        if input_channel_types is not None or output_channel_types is not None:
            self.sim_has_mpi_border_channels = has_mpi_from_channel_metadata
            if (
                has_mpi_from_border_meta is not None
                and has_mpi_from_border_meta != has_mpi_from_channel_metadata
            ):
                log.warning(
                    "[Simulation] Inconsistent MPI metadata: "
                    f"sim_has_mpi_border_channels={has_mpi_from_border_meta} but channel "
                    f"metadata implies {has_mpi_from_channel_metadata}. "
                    "Using channel metadata."
                )
        else:
            self.sim_has_mpi_border_channels = bool(has_mpi_from_border_meta)

        if (
            sim_comm_mode == SimulationCommMode.HYBRID_MPI.value
            and not self.sim_has_mpi_border_channels
        ):
            log.info(
                "[Simulation] sim_comm_mode=hybrid_mpi enabled, but no cross-worker "
                "border channels were detected. Falling back to pure SHM behavior."
            )

        self.use_mpi_launcher = (
            sim_comm_mode == SimulationCommMode.HYBRID_MPI.value
            and self.sim_has_mpi_border_channels
        )
        self.mpi_launcher = mpi_config.mpi_launcher
        self.mpi_oversubscribe = mpi_config.mpi_oversubscribe
        self.mpi_args = mpi_config.mpi_args
        self.mpi_hosts = model.get_metadata_prop("sim_mpi_hosts")

        super().__init__(
            model, simulation_type, fpgapart, clk_ns, functional_sim, workers, performance_sim
        )
        if shm_prefix is None:
            shm_prefix = model.get_metadata_prop("shm_prefix")
        self.max_qsrl_depth = max_qsrl_depth
        self.performance_sim = performance_sim
        self.shm_prefix = cast("str", shm_prefix)

    def simulate(
        self,
        depth: int | list[list[int]] | None = None,
        max_cycles: int | None = None,
        fifo_first_valid_cycles: list[list[int]] | None = None,
    ) -> tuple[list[dict[str, list[int]]], bool]:
        """Simulate the given number of samples for every layer. Layers are completely isolated
        and simulated in parallel.
        Simulation data is returned as a list of dicts (by node name as index).
        """
        if self.simulation_type != SimulationType.NODE_BASED_CONNECTED:
            raise FINNInternalError(
                f"Called simulation function 'simulate_node_connected' "
                f"does not match provided simulation type "
                f"{self.simulation_type}"
            )
        names = (
            [node.name for node in self.model.graph.node if "FIFO" not in node.op_type]
            if self.performance_sim
            else [node.name for node in self.model.graph.node]
        )
        initial_depth: Any = [[depth]] * len(self.binaries) if isinstance(depth, int) else depth

        # For BRAM FIFOs (depth > max_qsrl_depth), hardware loses BRAM_FIFO_PIPELINE_OVERHEAD
        # entries to internal pipeline registers *per BRAM sub-FIFO*.  Non-power-of-two depths
        # are decomposed into several power-of-two sub-FIFOs (see get_fifo_split_configs), so
        # the total overhead is num_bram_sub_fifos * BRAM_FIFO_PIPELINE_OVERHEAD.
        # Rounding to a full BRAM block before calling get_fifo_split_configs is NOT needed:
        # the decomposition works on any depth, and we want the sub-FIFO count for the exact
        # depth under test.
        if initial_depth is not None and not isinstance(initial_depth, int):
            adjusted_depth: Any = [
                [
                    d - _count_bram_sub_fifos(d, self.max_qsrl_depth) * BRAM_FIFO_PIPELINE_OVERHEAD
                    if d > self.max_qsrl_depth
                    else d
                    for d in node_depths
                ]
                for node_depths in initial_depth
            ]
        else:
            adjusted_depth = initial_depth

        # Run simulation
        start = time.time()
        output_json = Path(make_build_dir("simulation_results_")) / "simulation_data.json"
        sim_host_map: dict[str, str] | None = None
        if self.use_mpi_launcher:
            sim_host_map_meta = self.model.get_metadata_prop("sim_host_map")
            if sim_host_map_meta is not None:
                sim_host_map = cast("dict[str, str]", literal_eval(sim_host_map_meta))
            else:
                raise FINNInternalError("MPI Host Map missing.")
        mpi_params = MpiLaunchParams(
            use_mpi_launcher=self.use_mpi_launcher,
            mpi_launcher=self.mpi_launcher,
            mpi_hosts=self.mpi_hosts,
            mpi_host_map=sim_host_map,
            mpi_oversubscribe=self.mpi_oversubscribe,
            mpi_args=self.mpi_args,
        )
        controller = NodeConnectedSimulationController(
            len(self.binaries),
            names,
            list(self.binaries.values()),
            Console(),
            self.shm_prefix,
            mpi_params=mpi_params,
            poll_interval=0.1,
            with_progressbar=False,
        )
        controller.run(adjusted_depth, output_json, max_cycles, fifo_first_valid_cycles)
        end = time.time()
        log.debug(f"Simulation took {end - start} seconds!")

        # Load the merged data from JSON
        merged_data = json.loads(output_json.read_text())

        # Return the collected data indexed by node index
        data = []
        for sim_entry in merged_data["simulations"]:
            data.append(
                {
                    "name": sim_entry["name"],
                    "fifo_utilization": sim_entry["fifo_utilization"],
                    "fifo_depth": sim_entry["fifo_depth"],
                    "cycles": sim_entry["cycles"],
                    "samples": sim_entry["samples"],
                    "intervals": sim_entry["intervals"],
                    "fifo_cycles_until_first_valid": sim_entry["fifo_cycles_until_first_valid"],
                    "latency_cycles": sim_entry["latency_cycles"],
                }
            )
        json.dump(data, output_json.open("w"), indent=4)
        return data, merged_data.get("timeout_occurred", False)


class RunLayerParallelSimulation(Transformation):
    """Transformation for running Layer Parallel Simulation."""

    def __init__(
        self,
        fpgapart: str,
        clk_ns: float,
        cfg: DataflowBuildConfig,
        minimization_orders: list[MinimizationOrder] | None = None,
        max_qsrl_depth: int = 256,
        vivado_ram_style: str = "auto",
        quality_of_results: str = "default",
    ) -> None:
        """Run layer parallel simulations."""
        super().__init__()
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns
        self.cfg = cfg
        self.max_qsrl_depth = max_qsrl_depth
        self.vivado_ram_style = vivado_ram_style
        self.quality_of_results = quality_of_results
        if minimization_orders is not None:
            self.minimization_orders = minimization_orders
        else:
            self.minimization_orders = [MinimizationOrder.NODE_ORDER]

        self.final_depths: dict[MinimizationOrder, list[list[int]] | None] = dict.fromkeys(
            self.minimization_orders
        )

    def create_starting_fifo_depths(
        self, initial_fifo_depths: list[dict[str, list[int]]]
    ) -> tuple[list[list[int]], list[list[int]]]:
        """From the given initial_fifo_depths returned by the simulation, create a starting
        FIFO depth configuration that can be modified sequentially by the minimization algorithm.
        Also return the fifo_first_valid_cycles.
        """
        # Create fifo_depths (indexed by layer index and then stream index)
        fifo_depths: list[list[int]] = []  # Each entry is a list of fifo sizes for that node
        for val in initial_fifo_depths:
            # Use _safe_bram_starting_depth so that simulate() (which subtracts
            # num_sub_fifos*BRAM_FIFO_PIPELINE_OVERHEAD for BRAM depths) still sees a depth
            # that covers the observed peak utilisation.  A flat +2 is insufficient when a
            # depth decomposes into multiple BRAM sub-FIFOs (e.g. depth 1537 → 2 sub-FIFOs
            # → 4 entries of overhead).
            fifo_depths.append(
                [_safe_bram_starting_depth(v, self.max_qsrl_depth) for v in val["fifo_utilization"]]
            )
        fifo_first_valid_cycles: list[list[int]] = []
        for val in initial_fifo_depths:
            fifo_first_valid_cycles.append(
                [v + math.ceil(v * 0.01) for v in val["fifo_cycles_until_first_valid"]]
            )  # Add 1% cycles grace period
        return fifo_depths, fifo_first_valid_cycles

    def get_minimization_order_indices(
        self,
        min_order: MinimizationOrder,
        model: ModelWrapper,
        bitwidths: list[int],
    ) -> list[int]:
        """Given a MinimizationOrder, return the list of indices to
        access/minimize `fifo_depths` for that order. For example, NODE_ORDER would return
        [0,1,2,...] and NODE_ORDER_REVERSED [N, N-1, N-2, ..., 0].
        """
        assert len(model.graph.node) == len(bitwidths)
        match min_order:
            case MinimizationOrder.NODE_ORDER:
                return list(range(len(model.graph.node)))
            case MinimizationOrder.REVERSE_NODE_ORDER:
                return list(range(len(model.graph.node)))[::-1]
            case (
                MinimizationOrder.LARGEST_BITWIDTH_DIFF_FIRST
                | MinimizationOrder.SMALLEST_BITWIDTH_DIFF_FIRST
            ):
                diffs: list[tuple[int, int]] = []  # (index, diff)
                for i in range(len(model.graph.node)):
                    hw: HWCustomOp = getHWCustomOp(model.graph.node[i])
                    in_width = max(
                        [hw.get_instream_width(j) for j in range(len(model.graph.node[i].input))]
                    )
                    out_width = max(
                        [hw.get_outstream_width(j) for j in range(len(model.graph.node[i].output))]
                    )
                    diffs.append((i, in_width - out_width))
                sorted_order = sorted(
                    diffs,
                    key=lambda x: x[1],
                    reverse=(min_order == MinimizationOrder.LARGEST_BITWIDTH_DIFF_FIRST),
                )
                return [idx for idx, diff in sorted_order]
            case _:
                raise NotImplementedError()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Run layer parallel simulations."""
        shm_prefix = model.get_metadata_prop("shm_prefix")
        if shm_prefix is None or shm_prefix == "":
            raise FINNInternalError(
                "Expected model to have non-empty 'shm_prefix' metadata property "
                "for node-connected simulation"
            )
        sim = NodeConnectedSimulation(
            model,
            SimulationType.NODE_BASED_CONNECTED,
            self.fpgapart,
            self.clk_ns,
            self.cfg.functional_simulation,
            shm_prefix=shm_prefix,
            max_qsrl_depth=self.max_qsrl_depth,
            mpi_config=MpiSimConfig.from_build_config(self.cfg),
        )
        model = sim.model  # TODO:clean up

        work_folder = cast("Path", make_build_dir("fifo_results_", True))

        # Create empty table for datapoints that will be collected
        # First create as a nested dict, since not all data is avilable at the same time
        # It is then flattened when creating the dataframe, so that node and stream are columns too
        # df_data[node][stream_idx][columnm] = ...
        df_data: dict[str, list[dict[str, Any]]] = {}
        for nodeindex, node in enumerate(model.graph.node):
            df_data[node.name] = []
            for node_idx in range(len(node.output)):
                df_data[node.name].append(
                    {
                        "onnx_index": nodeindex,
                        "out_bitwidth": -1,
                        "out_initial_fifo_depths": -1,
                        "fifo_cycles_until_first_valid": -1,
                        "successor_node": ", ".join(
                            [node.name for node in model.find_consumers(node.output[node_idx])]
                        ),
                    }
                )
                for min_order in self.minimization_orders:
                    df_data[node.name][-1][f"out_final_depth_{min_order.name}"] = -1
                    df_data[node.name][-1][f"simulation_time_{min_order.name}"] = -1
                    df_data[node.name][-1][f"minimization_iterations_{min_order.name}"] = -1

        # Running the initial simulation
        log.info("Running initial node-connected simulation.")
        initial_fifo_depths, _ = sim.simulate()

        # Store the initial sizes as a report
        initial_sizes_path = work_folder / "initial_fifo_sizes_sim_connected.json"
        initial_sizes_path.write_text(json.dumps(initial_fifo_depths, indent=4))
        log.debug(f"Wrote initial sizes to: {initial_sizes_path}")

        # Store initial sizes in dataframe as well
        for layerdata in initial_fifo_depths:
            for idx in range(len(layerdata["fifo_utilization"])):
                name: str = cast("str", layerdata["name"])
                df_data[name][idx]["out_initial_fifo_depths"] = layerdata["fifo_utilization"][idx]
                df_data[name][idx]["fifo_cycles_until_first_valid"] = layerdata[
                    "fifo_cycles_until_first_valid"
                ][idx]

        # List of list of fifo depths
        fifo_depths, fifo_first_valid_cycles = self.create_starting_fifo_depths(initial_fifo_depths)

        # Max cycles for any simulation
        sim_cycles: int = cast("int", max([val["cycles"] for val in initial_fifo_depths]))

        # Extract bitwidths from outstream widths of hw nodes
        bit_widths = []
        for node_idx in range(len(fifo_depths)):
            bit_widths.append([])
            hw_node = getHWCustomOp(model.graph.node[node_idx])
            if isinstance(hw_node, HWCustomOp):
                for fifo_idx in range(len(fifo_depths[node_idx])):
                    bit_widths[node_idx].append(hw_node.get_outstream_width(fifo_idx))
            else:
                raise FINNInternalError("Non-HW node found in dataflow graph during simulation")

        # Store bitwidths into dataframe as well
        for node_idx in range(len(bit_widths)):
            for fifo_idx in range(len(bit_widths[node_idx])):
                df_data[model.graph.node[node_idx].name][fifo_idx]["out_bitwidth"] = bit_widths[
                    node_idx
                ][fifo_idx]

        # Run minimization for every layer/stream
        log.info("Minimizing layers...")
        needs_minimization = []
        for node_idx in range(len(fifo_depths)):
            needs_minimization.append([True] * len(fifo_depths[node_idx]))
        for node_idx in range(len(fifo_depths)):
            for fifo_idx in range(len(fifo_depths[node_idx])):
                # Check if we can reduce the fifo size

                used_size = fifo_depths[node_idx][fifo_idx]
                bw = bit_widths[node_idx][fifo_idx]

                needs_minimization[node_idx][fifo_idx] = self._needs_minimization(used_size, bw)

        # Total minimizations
        total_minimizations = sum(len(streams) for streams in fifo_depths)

        for k, minimization_order in enumerate(self.minimization_orders):
            # Create a new empty FIFO depth list
            fifo_depths, fifo_first_valid_cycles = self.create_starting_fifo_depths(
                initial_fifo_depths
            )

            # Minimize FIFO depths using binary search over BRAM block counts
            idx_order = self.get_minimization_order_indices(minimization_order, model, bit_widths)
            if len(idx_order) != len(model.graph.node):
                raise FINNInternalError(
                    f"Expected index order length {len(model.graph.node)}, but got {len(idx_order)}"
                )

            log.info(
                f"Minimizing using order: {minimization_order.name}. Index order is: {idx_order}"
            )

            done = 0
            for node_idx in idx_order:
                for fifo_idx in range(len(fifo_depths[node_idx])):
                    if not needs_minimization[node_idx][fifo_idx]:
                        df_data[model.graph.node[node_idx].name][fifo_idx][
                            f"simulation_time_{minimization_order.name}"
                        ] = 0.0
                        df_data[model.graph.node[node_idx].name][fifo_idx][
                            f"out_final_depth_{minimization_order.name}"
                        ] = fifo_depths[node_idx][fifo_idx]
                        df_data[model.graph.node[node_idx].name][fifo_idx][
                            f"minimization_iterations_{minimization_order.name}"
                        ] = 0
                        log.info(
                            f"[ {node_idx}.{fifo_idx} / {len(fifo_depths) - 1} ] "
                            f"Skipping minimization for this stream."
                        )
                        done += 1
                        continue

                    minimization_start = time.time()
                    minimized_depth, iterations_needed = self._minimize_fifo_depth(
                        node_idx,
                        fifo_idx,
                        fifo_depths,  # current_depths: evolves as FIFOs are minimised
                        bit_widths,
                        initial_fifo_depths,
                        sim,
                        sim_cycles,
                        fifo_first_valid_cycles,
                    )
                    minimization_time = time.time() - minimization_start

                    # Store the minimized size
                    fifo_depths[node_idx][fifo_idx] = minimized_depth
                    done += 1

                    # Store data into dataframe
                    df_data[model.graph.node[node_idx].name][fifo_idx][
                        f"simulation_time_{minimization_order.name}"
                    ] = minimization_time
                    df_data[model.graph.node[node_idx].name][fifo_idx][
                        f"minimization_iterations_{minimization_order.name}"
                    ] = iterations_needed
                    df_data[model.graph.node[node_idx].name][fifo_idx][
                        f"out_final_depth_{minimization_order.name}"
                    ] = fifo_depths[node_idx][fifo_idx]
                    log.debug(
                        f"Set node/stream {node_idx}.{fifo_idx} to "
                        f"depth {fifo_depths[node_idx][fifo_idx]}, in "
                        f"{iterations_needed} iterations and {minimization_time} "
                        f"seconds. (To {minimization_order.name})"
                    )

                    percentage = int(100.0 * float(done) / float(total_minimizations))
                    log.info(
                        f"[ {percentage}% ] "
                        f"[ {node_idx}.{fifo_idx} / {len(fifo_depths) - 1} ] Simulation completed "
                        f"({iterations_needed} iterations).",
                        extra={"markup": True, "highlighter": None},
                    )

            self.final_depths[minimization_order] = deepcopy(fifo_depths)

            order_percent = int(100.0 * float(k + 1) / float(len(self.minimization_orders)))
            log.info(
                f"[ {order_percent}% ] "
                f"-----  Minimization order {minimization_order.name} completed -----",
                extra={"markup": True, "highlighter": None},
            )

        # Store dataframe
        df_keys = list(df_data[model.graph.node[0].name][0].keys())
        log.debug(f"Saving keys: {df_keys} + [node, stream]")
        df_dict = {}
        df_dict["node"] = []
        df_dict["stream"] = []
        for k in df_keys:
            df_dict[k] = []
        for node, nodedata in df_data.items():
            for streamindex, streamdata in enumerate(nodedata):
                df_dict["node"].append(node)
                df_dict["stream"].append(streamindex)
                for key in streamdata.keys():
                    df_dict[key].append(streamdata[key])

        df = pd.DataFrame(df_dict)
        model = store_fifo_data(
            model,
            df,
            work_folder / "fifo_data.csv",
            delete_existing=False,
            store_html=True,
        )

        # Use the smallest fifo depths found (by total bytes)
        smallest_order = self.minimization_orders[0]
        smallest_size = None
        for order in self.minimization_orders:
            current_size = 0
            depths = self.final_depths[order]
            if depths is None:
                raise FINNInternalError(
                    f"Expected FIFO sizes for minimization order {order.name}, but found None."
                )
            for node_idx in range(len(depths)):
                for fifo_idx in range(len(depths[node_idx])):
                    current_size += depths[node_idx][fifo_idx] * bit_widths[node_idx][fifo_idx]

            if smallest_size is None or current_size < smallest_size:
                smallest_size = current_size
                smallest_order = order

        # Set the result fifo depths
        fifo_depths = self.final_depths[smallest_order]
        assert fifo_depths is not None

        # Make sure that all FIFOs with depth > 256 use a full BRAM block,
        # since partial blocks are not supported by Vivado HLS
        for node_idx in range(len(fifo_depths)):
            for fifo_idx in range(len(fifo_depths[node_idx])):
                if fifo_depths[node_idx][fifo_idx] > self.max_qsrl_depth:
                    bw = bit_widths[node_idx][fifo_idx]
                    blocks = calculate_bram_blocks(fifo_depths[node_idx][fifo_idx], bw)
                    _, max_d = calculate_bram_depth_range(blocks, bw)
                    fifo_depths[node_idx][fifo_idx] = max_d

        log.info("Running final end-to-end validation simulation with minimised FIFO depths...")
        validation_data, validation_timeout = sim.simulate(
            fifo_depths,
            max_cycles=math.ceil(sim_cycles * 1.05),
            fifo_first_valid_cycles=fifo_first_valid_cycles,
        )
        if validation_timeout:
            raise FINNUserError(
                "Final validation simulation timed out with the jointly-minimised FIFO depths. "
                "The per-FIFO minimisation may have produced a configuration that is "
                "collectively too small.  Re-run with a larger initial depth or fewer "
                "minimisation orders."
            )
        if self._check_performance(validation_data, initial_fifo_depths):
            raise FINNUserError(
                "Final validation simulation detected throughput degradation with the "
                "jointly-minimised FIFO depths (intervals exceeded baseline). "
                "The per-FIFO minimisation may have produced a configuration that is "
                "collectively too small.  Re-run with a larger initial depth or fewer "
                "minimisation orders."
            )
        log.info("Final validation simulation passed - minimised depths are correct.")

        # Write back results. By default write to output_dir / "fifo_config.json"
        writeback_path = work_folder / "fifo_config.json"
        json_results = []
        for node_idx, node in enumerate(model.graph.node):
            json_results.append({"node": node.name, "depths": fifo_depths[node_idx]})
        with writeback_path.open("w") as f:
            json.dump(json_results, f)
        log.info(f"Wrote results back to {writeback_path}")
        model.set_metadata_prop("fifo_data", str(writeback_path))

        return model, False

    def _check_performance(
        self, new_data: list[dict[str, list[int]]], initial_fifo_depths: list[dict[str, list[int]]]
    ) -> bool:
        """Check if performance has degraded compared to baseline.

        Args:
            new_data: Simulation results to check
            initial_fifo_depths: Baseline performance data

        Returns:
            True if performance degraded, False otherwise
        """
        for new, initial in zip(new_data, initial_fifo_depths, strict=True):
            if len(new["intervals"]) != len(initial["intervals"]):
                raise FINNInternalError(
                    "New simulation data has different number of streams than baseline."
                )
            for idx in range(len(new["intervals"])):
                if new["intervals"][idx] > initial["intervals"][idx]:
                    return True
        return False

    def _test_depth(
        self,
        test_depth: int,
        node_idx: int,
        fifo_idx: int,
        current_depths: list[list[int]],
        initial_fifo_depths: list[dict[str, list[int]]],
        sim: NodeConnectedSimulation,
        sim_cycles: float,
        fifo_first_valid_cycles: list[list[int]],
    ) -> tuple[bool, bool]:
        """Test a specific FIFO depth.

        Args:
            test_depth: Depth to test
            node_idx: Node index
            fifo_idx: FIFO index within node
            current_depths: Current working FIFO depth configuration.  FIFOs that have
                already been minimised contain their final minimised depth; FIFOs not yet
                processed still carry the safe starting depth.  This list is never
                modified by this method - a deep copy is made before inserting
                ``test_depth``.
            initial_fifo_depths: Baseline performance data
            sim: Simulation controller
            sim_cycles: Maximum simulation cycles
            fifo_first_valid_cycles: First valid cycle for each FIFO
        Returns:
            Tuple of (success, timeout) where success means depth works without degradation
        """
        test_depths = deepcopy(current_depths)
        test_depths[node_idx][fifo_idx] = test_depth

        new_simulation_data, timeout = sim.simulate(
            test_depths,
            max_cycles=min(
                math.ceil(sim_cycles * 1.05), math.ceil(sim_cycles) + 10 * len(test_depths)
            ),
            fifo_first_valid_cycles=fifo_first_valid_cycles,
        )

        if timeout:
            return False, True

        performance_degraded = self._check_performance(new_simulation_data, initial_fifo_depths)
        return not performance_degraded, False

    def _get_valid_block_counts(self, min_blocks: int, max_blocks: int, bitwidth: int) -> list[int]:
        """Get all valid BRAM block counts in the specified range (see fifo_depth_search)."""
        return get_valid_block_counts(min_blocks, max_blocks, bitwidth)

    def _minimize_fifo_depth(
        self,
        node_idx: int,
        fifo_idx: int,
        current_depths: list[list[int]],
        bit_widths: list[list[int]],
        initial_fifo_depths: list[dict[str, list[int]]],
        sim: NodeConnectedSimulation,
        sim_cycles: int,
        fifo_first_valid_cycles: list[list[int]],
    ) -> tuple[int, int]:
        """Minimize a single FIFO depth using the shared block-granular search.

        Args:
            node_idx: Node index
            fifo_idx: FIFO index within node
            current_depths: Current working FIFO depth configuration.  FIFOs that have
                already been minimised in this pass carry their final minimised depth;
                FIFOs not yet processed still carry the safe starting depth.  This list
                is mutated by the caller (``apply``) after each call to store the
                minimised result, so successive calls see the evolving state.
            bit_widths: Bitwidths for all FIFOs
            initial_fifo_depths: Baseline performance data
            sim: Simulation controller
            sim_cycles: Maximum simulation cycles
            fifo_first_valid_cycles: First valid cycle for each FIFO
        Returns:
            Tuple: Minimized FIFO depth, Iterations required to arrive at the result
        """
        original_size = current_depths[node_idx][fifo_idx]
        bw = bit_widths[node_idx][fifo_idx]
        log.debug(f"Minimizing Node {node_idx} FIFO {fifo_idx}: original depth {original_size}")

        def test_depth(depth: int) -> tuple[bool, bool]:
            return self._test_depth(
                depth,
                node_idx,
                fifo_idx,
                current_depths,
                initial_fifo_depths,
                sim,
                sim_cycles,
                fifo_first_valid_cycles,
            )

        return minimize_fifo_depth(original_size, bw, test_depth, self.max_qsrl_depth)

    def _needs_minimization(self, fifo_depth: int, bitwidth: int) -> bool:
        """Determine whether a FIFO can be minimized further (see fifo_depth_search)."""
        return needs_minimization(fifo_depth, bitwidth, self.max_qsrl_depth)

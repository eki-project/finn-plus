"""Control (node based) simulations via unix sockets."""

import json
import multiprocessing
import socket
import subprocess
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from rich.console import Console
from threading import Lock
from typing import Any

from finn.util.basic import make_build_dir
from finn.util.exception import FINNInternalError
from finn.util.logging import ThreadsafeProgressDisplay


class NodeConnectedSimulationController:
    """Control a node-node IPC connected simulation in threads."""

    def __init__(
        self,
        parallel_simulations: int,
        names: list[str],
        binaries: list[Path],
        console: Console,
        poll_interval: float = 1.0,
        with_progressbar: bool = True,
    ) -> None:
        """Create a new controller, without starting the simulation.

        Args:
            parallel_simulations: Number of simulations to run in parallel.
            names: List of names for the simulations.
            binaries: List of paths to the simulation binaries.
            console: The rich.console.Console to print with.
            poll_interval: How long the wait between checks of the processes stdout/stdin is.
            with_progressbar: Whether or not to display a progressbar for the cycle count.
        """
        if len(names) != len(binaries):
            raise FINNInternalError(
                f"Simulation controller received non-matching "
                f"name and binary count: {len(names)} and {len(binaries)}"
            )
        self.binaries = binaries
        self.names = names
        self.console = console
        self.poll_interval = poll_interval
        self.workers = parallel_simulations
        self.progress = None
        if with_progressbar:
            self.progress = ThreadsafeProgressDisplay(names, [0] * len(names), names)
        self.running_lock = Lock()
        self.running = 0
        self.total = len(names)
        self.logdir = Path(make_build_dir("node_connected_simulation_logfiles_"))

        # Socket communication management
        self.processes: list[tuple[subprocess.Popen, Any, Any]] = []
        self.sockets: list[tuple[socket.socket, str]] = []

        # Early termination flag
        self.should_stop = False
        self.stop_lock = Lock()

    def _start_process(self, binary: Path, process_id: int) -> int:
        """Start a single C++ simulation process with its own Unix socket.

        Args:
            binary: Path to the simulation executable
            process_id: Unique identifier for this process

        Returns:
            Index of the started process
        """
        thread_id = threading.get_ident()

        # Create unique socket path which includes thread ID to avoid conflicts
        # with multiple threads
        socket_path = Path(f"/tmp/{thread_id}/")
        socket_path.mkdir(parents=True, exist_ok=True)
        socket_path = socket_path / f"sim_socket_{process_id}.sock"

        # Remove socket if it exists
        if socket_path.exists():
            socket_path.unlink()

        # Build command arguments
        cmd = [str(binary), "--socket", socket_path]

        # Create log files for stdout and stderr
        stdout_log = self.logdir / f"{process_id}_stdout.log"
        stderr_log = self.logdir / f"{process_id}_stderr.log"

        stdout_file = stdout_log.open("w")
        stderr_file = stderr_log.open("w")

        # Start C++ process - redirect stdout/stderr to files
        cwd = binary.parent
        proc = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file, text=True, cwd=cwd)

        # Check if process started successfully
        time.sleep(0.2)  # Give process time to fail if there's an immediate error
        if proc.poll() is not None:
            stderr_output = stderr_log.read_text() if stderr_log.exists() else "No stderr"
            stdout_output = stdout_log.read_text() if stdout_log.exists() else "No stdout"
            stdout_file.close()
            stderr_file.close()
            msg = (
                f"C++ process exited immediately with code {proc.returncode}\n"
                f"Stderr: {stderr_output}\nStdout: {stdout_output}"
            )
            raise RuntimeError(msg)

        # Create Unix socket and connect
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        # Wait for C++ process to create socket (with timeout)
        max_retries = 100  # 20 seconds total
        connected = False
        for i in range(max_retries):
            # Check if process is still alive
            if proc.poll() is not None:
                stderr_output = stderr_log.read_text() if stderr_log.exists() else "No stderr"
                stdout_output = stdout_log.read_text() if stdout_log.exists() else "No stdout"
                stdout_file.close()
                stderr_file.close()
                msg = (
                    f"C++ process died during socket wait with code {proc.returncode}\n"
                    f"Stderr: {stderr_output}\nStdout: {stdout_output}"
                )
                raise RuntimeError(msg)

            try:
                sock.connect(str(socket_path))
                connected = True
                break
            except (FileNotFoundError, ConnectionRefusedError) as e:
                if i == max_retries - 1:
                    stderr_output = stderr_log.read_text() if stderr_log.exists() else "No stderr"
                    stdout_output = stdout_log.read_text() if stdout_log.exists() else "No stdout"
                    stdout_file.close()
                    stderr_file.close()
                    msg = (
                        f"Failed to connect to socket after {max_retries} retries\n"
                        f"Stderr: {stderr_output}\nStdout: {stdout_output}"
                    )
                    raise RuntimeError(msg) from e
                time.sleep(0.2)

        if not connected:
            stderr_output = stderr_log.read_text() if stderr_log.exists() else "No stderr"
            stdout_output = stdout_log.read_text() if stdout_log.exists() else "No stdout"
            stdout_file.close()
            stderr_file.close()
            msg = (
                f"Failed to connect to socket {socket_path}\n"
                f"Stderr: {stderr_output}\nStdout: {stdout_output}"
            )
            raise RuntimeError(msg)

        self.processes.append((proc, stdout_file, stderr_file))
        self.sockets.append((sock, str(socket_path)))
        return len(self.processes) - 1

    def _send_command(self, process_idx: int, command: str, payload: dict[str, Any]) -> None:
        """Send command and payload to a specific process.

        Args:
            process_idx: Index of the process to send to
            command: Command string (e.g., "start", "status", "stop")
            payload: Dictionary containing command-specific data
        """
        sock, _ = self.sockets[process_idx]

        message = {"command": command, "payload": payload}

        # Send length-prefixed message
        msg_str = json.dumps(message)
        msg_bytes = msg_str.encode("utf-8")
        length = len(msg_bytes)

        # Send 4-byte length prefix (little-endian)
        sock.sendall(length.to_bytes(4, byteorder="little"))
        # Send actual message
        sock.sendall(msg_bytes)

    def _receive_response(self, process_idx: int) -> dict[str, Any] | None:
        """Receive response from a specific process.

        Args:
            process_idx: Index of the process to receive from

        Returns:
            Dictionary containing the response, or None if error
        """
        sock, _ = self.sockets[process_idx]

        # Read 4-byte length prefix
        length_bytes = sock.recv(4)
        if not length_bytes:
            return None

        length = int.from_bytes(length_bytes, byteorder="little")

        # Read message data
        msg_bytes = b""
        while len(msg_bytes) < length:
            chunk = sock.recv(length - len(msg_bytes))
            if not chunk:
                break
            msg_bytes += chunk

        return json.loads(msg_bytes.decode("utf-8"))

    def _send_and_receive(
        self, process_idx: int, command: str, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Send command and wait for response (convenience method).

        Args:
            process_idx: Index of the process
            command: Command string
            payload: Command payload

        Returns:
            Response dictionary
        """
        self._send_command(process_idx, command, payload)
        return self._receive_response(process_idx)

    def _cleanup_sockets(self) -> None:
        """Close all sockets and terminate all processes."""
        # Send stop command to all processes
        errors = []
        for i in range(len(self.processes)):
            try:
                self._send_command(i, "stop", {})
                self._receive_response(i)
            except Exception as e:  # noqa
                errors.append((i, e))

        # Close sockets
        for sock, socket_path in self.sockets:
            sock.close()
            socket_path_obj = Path(socket_path)
            if socket_path_obj.exists():
                socket_path_obj.unlink(True)

        # Terminate processes and close file handles
        for proc, stdout_file, stderr_file in self.processes:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            finally:
                stdout_file.close()
                stderr_file.close()

    def run(
        self,
        depth: list[list[int]] | None = None,
        output_json: Path | None = None,
        max_cycles: int | None = None,
    ) -> dict[str, list[int]]:
        """Run the simulation entirely with the given depth and sample count.

        Args:
            depth: FIFO depth to configure for simulations.
            samples: Number of samples to simulate.
            output_json: Optional path to write merged simulation data as JSON.

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
                            i % multiprocessing.cpu_count(),
                            depth[i] if depth is not None else None,
                            is_last_node,  # Only last node has no output FIFOs
                            is_special_for_display,  # First and last get special coloring
                            max_cycles,
                        )
                    )

                # Wait for first completion or error
                from concurrent.futures import FIRST_COMPLETED, wait

                all_futures = list(futures)  # Keep track of all futures
                while futures:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)

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
                                ) = result
                                fifo_depths[sim_name] = fifo_depth
                                fifo_results[sim_name] = fifo_util
                                cycles_results[sim_name] = cycles
                                samples_results[sim_name] = samps
                                intervals_results[sim_name] = intervals
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
                            ) = result
                            # Only update if not already collected
                            if sim_name not in fifo_results:
                                fifo_depths[sim_name] = fifo_depth
                                fifo_results[sim_name] = fifo_util
                                cycles_results[sim_name] = cycles
                                samples_results[sim_name] = samps
                                intervals_results[sim_name] = intervals
                                timeout_result = timeout_result or timeout
                    except Exception as e:
                        self.console.log(f"Error collecting result: {e}")
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
        _cpu: int | None,
        depth: list[int] | None = None,
        is_last_node: bool = False,
        is_special_for_display: bool = False,
        max_cycles: int | None = None,
    ) -> tuple[str, list[int], int, int, list[int], bool, list[int]] | None:
        """Run the specified simulation binary in a new subprocess and communicate with it.

        Args:
            binary: Path to simulation binary
            name: Name of simulation node
            _cpu: CPU affinity (unused)
            depth: List of FIFO depths for this node's output FIFOs
            is_last_node: True if this is the last node (no output FIFOs to configure)
            is_special_for_display: True if this node should get special color in logs
            max_cycles: Maximum cycles to simulate

        Returns:
            Tuple of (simulation_name, fifo_utilization, cycles, samples, intervals, timeout,
            fifo_depth) on success,
            None on failure.
        """
        cwd = binary.parent
        if name is None:
            name = cwd.name.replace("rtlsim_", "")

        process_index = self.names.index(name)

        with (self.logdir / f"{name}_{process_index}_of_{self.total}.txt").open("w+") as logfile:

            def _print(msg: str, color: str = "green") -> None:
                if self.progress is None:
                    if is_special_for_display:
                        color = "orange3"
                    if "ERROR" in msg:
                        color = "red"
                    self.console.log(
                        f"[bold {color}]{name:<35}"
                        f"[/bold {color}][cornflower_blue]{process_index} "
                        f"/ {len(self.names) - 1}[/cornflower_blue] {msg:<35}"
                    )
                logfile.write(f"{msg}\n")
                logfile.flush()

            try:
                # Start the simulation process with socket communication
                proc_idx = self._start_process(binary, process_index)

                # Send configuration commands
                # Last node has no output FIFOs, so don't configure FIFO depths
                config_payload: dict[str, list[int] | int] = {}
                if not is_last_node and depth is not None:
                    config_payload["fifo_depth"] = depth
                if max_cycles is not None:
                    config_payload["max_cycles"] = max_cycles

                response = self._send_and_receive(proc_idx, "configure", config_payload)

                if not response or response.get("status") != "success":
                    error_msg = (
                        response.get("message", "Unknown error") if response else "No response"
                    )
                    _print(f"Configuration failed: {error_msg}", "red")
                    return None

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

                # Poll for status updates
                while True:
                    # Check if we should stop early
                    with self.stop_lock:
                        if self.should_stop:
                            stop_response = self._send_and_receive(proc_idx, "stop", {})
                            if stop_response:
                                cycles = stop_response.get("cycles", 0)
                                samps = stop_response.get("samples", 0)
                                fifo_util = stop_response.get("fifo_utilization", [])
                                intervals = stop_response.get("intervals", [])
                                fifo_depth = stop_response.get("fifo_depth", [])
                                timeout = stop_response.get("timeout", False)
                                if fifo_util:
                                    logfile.write(f"Final FIFO utilization: {fifo_util}\n")
                            return (name, fifo_util, cycles, samps, intervals, timeout, fifo_depth)
                    time.sleep(self.poll_interval)

                    response = self._send_and_receive(proc_idx, "status", {})

                    if not response:
                        _print("Lost connection to simulation", "red")
                        with self.stop_lock:
                            self.should_stop = True
                        raise RuntimeError("Lost connection to simulation")

                    state = response.get("state", "unknown")

                    if state == "finished" or state == "timeout":
                        cycles = response.get("cycles", 0)
                        samps = response.get("samples", 0)
                        fifo_util = response.get("fifo_utilization", [])
                        fifo_depth = response.get("fifo_depth", [])
                        intervals = response.get("intervals", [])
                        timeout = response.get("timeout", False)
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
                        raise RuntimeError(f"Simulation error: {error_msg}")

                # Stop the simulation
                stop_response = self._send_and_receive(proc_idx, "stop", {})
                fifo_util = []

                if stop_response:
                    fifo_util = stop_response.get("fifo_utilization", [])
                    fifo_depth = stop_response.get("fifo_depth", [])
                    cycles = stop_response.get("cycles", 0)
                    samps = stop_response.get("samples", 0)
                    if fifo_util:
                        logfile.write(f"Final FIFO utilization: {fifo_util}\n")

                return (name, fifo_util, cycles, samps, intervals, timeout, fifo_depth)

            except Exception as e:
                self.console.log(f"Exception caught during simulation execution ({name}): {e}")
                self.console.log(traceback.format_exc())
                logfile.write(f"Exception: {e}\n")
                logfile.write(traceback.format_exc())
                with self.stop_lock:
                    self.should_stop = True
                return None

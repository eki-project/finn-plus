"""Control (node based) simulations via unix sockets."""

import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from rich.console import Console
from threading import Lock
from typing import Any

from finn.util.basic import make_build_dir
from finn.util.exception import FINNInternalError
from finn.util.logging import ThreadsafeProgressDisplay


class SimulationController:
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
        self.logdir = Path(make_build_dir("simulation_logfiles_"))

        # Socket communication management
        self.processes: list[tuple[subprocess.Popen, Any, Any]] = []
        self.sockets: list[tuple[socket.socket, str]] = []

        # Early termination flag
        self.should_stop = False
        self.stop_lock = Lock()

    def _start_process(self, binary: Path, process_id: int, cpu: int = -1) -> int:
        """Start a single C++ simulation process with its own Unix socket.

        Args:
            binary: Path to the simulation executable
            process_id: Unique identifier for this process
            cpu: CPU core to bind to (if -1, no binding)

        Returns:
            Index of the started process
        """
        thread_id = threading.get_ident()

        # Create unique socket path which includes thread ID to avoid conflicts
        # with multiple threads
        socket_path = Path(f"/tmp/fifosim_sockets/{thread_id}/")
        socket_path.mkdir(parents=True, exist_ok=True)
        socket_path = socket_path / f"sim_socket_{process_id}.sock"

        # Remove socket if it exists
        if socket_path.exists():
            socket_path.unlink()

        # Build command arguments
        cmd = [str(binary), "--socket", socket_path]

        # Create log files for stdout and stderr
        stdout_log = self.logdir / f"{process_id}_stdout_cpp.log"
        stderr_log = self.logdir / f"{process_id}_stderr_cpp.log"

        stdout_file = stdout_log.open("w")
        stderr_file = stderr_log.open("w")

        # Start C++ process - redirect stdout/stderr to files
        cwd = binary.parent
        # Set CPU affinity if a specific core is requested
        preexec_fn = (lambda: os.sched_setaffinity(0, {cpu})) if cpu != -1 else None
        proc = subprocess.Popen(
            cmd, stdout=stdout_file, stderr=stderr_file, text=True, cwd=cwd, preexec_fn=preexec_fn
        )

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
            self.console.log(str(process_id) + ": " + msg)
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
                self.console.log(str(process_id) + ": " + msg)
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
                    self.console.log(str(process_id) + ": " + msg)
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
            self.console.log(str(process_id) + ": " + msg)
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

        Raises:
            TimeoutError: If socket times out waiting for response
        """
        sock, _ = self.sockets[process_idx]

        # Set 120 second timeout to prevent deadlocks
        # Needs to be rather larger to give the simulation IO thread time to answer
        sock.settimeout(120.0)

        # Read 4-byte length prefix
        length_bytes = sock.recv(4)
        if not length_bytes:
            self.console.log(f"{process_idx}: Client disconnected.")
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

        Raises:
            RuntimeError: If the subprocess has terminated with an error
        """
        try:
            self._send_command(process_idx, command, payload)
            response = self._receive_response(process_idx)

            # If we got None (timeout or connection error), check if process crashed
            if response is None:
                proc, stdout_file, stderr_file = self.processes[process_idx]
                returncode = proc.poll()

                if returncode is not None and returncode != 0:
                    # Process has terminated with an error
                    # Flush and read error logs
                    stdout_file.flush()
                    stderr_file.flush()

                    stdout_log = self.logdir / f"{process_idx}_stdout_cpp.log"
                    stderr_log = self.logdir / f"{process_idx}_stderr_cpp.log"

                    stderr_output = stderr_log.read_text() if stderr_log.exists() else "No stderr"
                    stdout_output = stdout_log.read_text() if stdout_log.exists() else "No stdout"

                    # Raise the actual error from the subprocess
                    msg = (
                        f"Subprocess (process_idx={process_idx}) terminated with"
                        f" exit code {returncode}.\n"
                        f"Stderr:\n{stderr_output}\n"
                        f"Stdout:\n{stdout_output}"
                    )
                    raise RuntimeError(msg) from None

            return response
        except (BrokenPipeError, ConnectionResetError, TimeoutError) as err:
            # Connection error or timeout means the subprocess may have died
            # Check if it exited with an error and raise that instead
            proc, stdout_file, stderr_file = self.processes[process_idx]
            returncode = proc.poll()

            if returncode is not None and returncode != 0:
                # Process has terminated with an error
                # Flush and read error logs
                stdout_file.flush()
                stderr_file.flush()

                stdout_log = self.logdir / f"{process_idx}_stdout_cpp.log"
                stderr_log = self.logdir / f"{process_idx}_stderr_cpp.log"

                stderr_output = stderr_log.read_text() if stderr_log.exists() else "No stderr"
                stdout_output = stdout_log.read_text() if stdout_log.exists() else "No stdout"

                # Raise the actual error from the subprocess
                msg = (
                    f"Subprocess (process_idx={process_idx}) terminated with"
                    f" exit code {returncode}.\n"
                    f"Stderr:\n{stderr_output}\n"
                    f"Stdout:\n{stdout_output}"
                )
                raise RuntimeError(msg) from err  # from None

            # If process exited cleanly (returncode == 0) or hasn't exited yet,
            # this is an unexpected connection error
            return None

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

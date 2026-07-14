"""Manages the Simulation superclass as well as general simulation related transforms."""

import json
import pandas as pd
import socket
import subprocess
import threading
import time
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from rich.console import Console
from threading import Lock
from typing import Any, TypeAlias

from finn.transformation.fpgadataflow.simulation_build import BuildSimulation, SimulationType
from finn.util.basic import make_build_dir
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import ThreadsafeProgressDisplay, log

FIFODepthConfig: TypeAlias = list[dict[str, list[int]]]


def store_fifo_data(
    model: ModelWrapper,
    data: pd.DataFrame,
    default_path: Path,
    delete_existing: bool,
    sort_on: str = "onnx_index",
    merge_on: list[str] | None = None,
    merge_how: str = "inner",
    store_html: bool = True,
) -> ModelWrapper:
    """Store the given dataframe in a CSV file.

    If the model already points to data, merge with it and store at the
    path used before (unless delete_existing=True, then simply overwrite at that same path).
    If no data is stored beforehand, use the `default_path` and simply store
    the data there. The path is then entered into the `"fifo_data_path"` metadata prop of the model.

    The function can be used to aggregate benchmarking data across several flow steps.

    Args:
        model: The model that we check for a path to existing FIFO data.
        data: The data to store.
        default_path: Path to use in case that the model doesn't reference a data file yet.
           Is then stored as a metadata prop in the model.
        delete_existing: If true, delete the table and start a new one.
        sort_on: The column to sort on after merging.
        merge_on: What columns to merge on. If "None", use `["onnx_index", "node", "stream"]`
        merge_how: How to merge. Forwarded to pd.merge().
        store_html: If True, also store the data as a HTML with the same name next to the CSV.

    Returns:
        model: Return the model since we might have modified its metadata.
    """
    # Check if all layers are accounted for
    # Note: data may have multiple rows per node (one per output stream)
    if "node" in data.columns:
        num_unique_nodes = len(data["node"].unique())
        if num_unique_nodes != len(model.graph.node):
            raise FINNInternalError(
                f"Tried storing FIFO data for {num_unique_nodes} unique nodes "
                f"but expected {len(model.graph.node)}"
            )
    elif len(data.index) != len(model.graph.node):
        raise FINNInternalError(
            f"Tried storing FIFO data for {len(data.index)} "
            f"values but expected {len(model.graph.node)}"
        )
    fifo_data_path = model.get_metadata_prop("fifo_data_path")
    if fifo_data_path is not None:
        if not fifo_data_path.endswith(".csv"):
            raise FINNInternalError(
                f"It seems the model saved path to store "
                f"the dataframe does not point to a csv file: {fifo_data_path}"
            )
        if delete_existing:
            Path(fifo_data_path).unlink(missing_ok=True)
            merged = data
        else:
            merged = pd.merge(
                data,
                pd.read_csv(fifo_data_path),
                on=merge_on,
                how=merge_how,  # type: ignore
            )
            merged = merged.sort_values(sort_on)
        merged.to_csv(fifo_data_path, index=False)
        if store_html:
            merged.to_html(fifo_data_path.replace(".csv", ".html"))
        log.info(f"Stored FIFO dataframe to {fifo_data_path}.")
    else:
        if not default_path.suffix == ".csv":
            raise FINNInternalError(
                f"It seems the provided default path to store "
                f"the dataframe does not point to a csv file: {fifo_data_path}"
            )
        if delete_existing:
            default_path.unlink(missing_ok=True)
        data.to_csv(default_path, index=False)
        if store_html:
            data.to_html(str(default_path).replace(".csv", ".html"))
        model.set_metadata_prop("fifo_data_path", str(default_path))
        log.info(f"Stored FIFO dataframe to {default_path}.")
    return model


class Simulation:
    """Manage simulation (runs) in FINN. Upon instance creation, the simulation will be built.
    Simulations should inherit from this class and expand for their specific needs.

    IMPORTANT: If the modelwrapper was somehow changed, create a NEW simulation object!
    """

    def __init__(
        self,
        model: ModelWrapper,
        simulation_type: SimulationType,
        fpgapart: str,
        clk_ns: float,
        functional_sim: bool,
        workers: int | None = None,  # noqa: ARG002
        performance_sim: bool = False,
    ) -> None:
        """Create a new simulation instance. Read simulation binary paths
        from the simulation_binaries metadata prop field."""
        self.simulation_type = simulation_type
        self.model = model
        self.model = self.model.transform(
            BuildSimulation(fpgapart, clk_ns, functional_sim, performance_sim)
        )
        sim_binaries_str = self.model.get_metadata_prop("simulation_binaries")

        if sim_binaries_str is None:
            raise FINNUserError(
                "No field simulation_binaries found in the model. Make "
                "sure to run the BuildSimulation transformation beforehand."
            )
        sim_binaries: list[Path] = [Path(p) for p in str(sim_binaries_str).split("\n")]
        nodes = (
            [n for n in self.model.graph.node if "FIFO" not in n.op_type]
            if performance_sim
            else self.model.graph.node
        )
        if len(sim_binaries) != len(nodes):
            raise FINNUserError(
                "The number of found simulation binaries does not match the number "
                "of nodes in the graph. Make sure to run BuildSimulation just "
                "before."
            )
        if any(not p.exists() for p in sim_binaries):
            raise FINNUserError(
                "Simulation binary data points to invalid paths. Please rerun BuildSimulation."
            )
        self.binaries: dict[int, Path] = {i: sim_binaries[i] for i in range(len(sim_binaries))}
        match simulation_type:
            case SimulationType.NODE_BASED_CONNECTED:
                self.binaries = {
                    i: self.binaries[i] / "LayerSimulationBackend" for i in self.binaries.keys()
                }
            case SimulationType.NODE_BASED_ISOLATED:
                self.binaries = {
                    i: self.binaries[i] / "IsolatedSimulationBackend" for i in self.binaries.keys()
                }
            case _:
                raise FINNInternalError(f"Unsupported simulation type: {simulation_type}")

        errors = []
        for binary in self.binaries.values():
            if not binary.exists():
                errors.append(f"Binary {binary} does not exist! Please rerun BuildSimulation!")
        if len(errors) > 0:
            raise FINNInternalError("Errors occurred: \n" + "\n\t".join(errors))

    def simulate(self) -> Any:
        """Run the simulation and return the results.
        The type of the results may differ based on the simulation type."""
        raise NotImplementedError("Call simulate() on subclasses.")


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

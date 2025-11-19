"""Control (node based) simulations via stdio."""
import multiprocessing
import subprocess
import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from rich.console import Console
from subprocess import Popen
from threading import Lock

from finn.util.basic import get_vivado_root, make_build_dir
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
        poll_interval: float = 0.1,
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

    def run(self, depth: int, samples: int) -> None:
        """Run the simulation entirely with the given depth and sample count."""
        futures: list[Future] = []
        for i, name in enumerate(self.names):
            print(f"{i}: {name}")
        if self.progress is not None:
            self.progress.start()
        with ThreadPoolExecutor(self.workers) as pool:
            for i, (name, binary) in enumerate(zip(self.names, self.binaries, strict=True)):
                futures.append(
                    pool.submit(
                        self._run_binary,
                        binary,
                        name,
                        i % multiprocessing.cpu_count(),
                        depth,
                        samples,
                        i == 0 or i == len(self.names) - 1,
                    )
                )
            pool.shutdown()
        if self.progress is not None:
            self.progress.stop()

    def _send(self, proc: Popen[bytes], cmd: str) -> None:
        """Send a command to the given process stdin and flush the buffer."""
        if not cmd.endswith("\n"):
            cmd += "\n"
        proc.stdin.write(cmd.encode())
        proc.stdin.flush()

    def _run_binary(
        self,
        binary: Path,
        name: str | None,
        cpu: int | None,
        depth: int,
        samples: int,
        is_end_node: bool = False,
    ) -> None:
        """Run the specified simulation binary in a new subprocess and communicate with it."""

        # TODO: Seperate into multiple methods

        cwd = binary.parent
        if name is None:
            name = cwd.name.replace("rtlsim_", "")
        with (self.logdir / f"{name}_{self.names.index(name)}_of_{self.total}.txt").open(
            "w+"
        ) as logfile:

            def _print(msg: str, color: str = "green") -> None:
                if self.progress is None:
                    if is_end_node:
                        color = "orange3"
                    if "ERROR" in msg:
                        color = "red"
                    self.console.log(
                        f"[bold {color}]{name:<35}"
                        f"[/bold {color}][cornflower_blue]{self.names.index(name)} "  # type:ignore
                        f"/ {len(self.names)-1}[/cornflower_blue] {msg:<35}"
                    )

            ld_library_path = (
                "LD_LIBRARY_PATH=" + get_vivado_root() + "/lib/lnx64.o:$LD_LIBRARY_PATH"
            )
            taskset = ""
            if cpu is not None:
                taskset += f"taskset --cpu-list {cpu}"  # TODO: numactl?
            command = f"{ld_library_path} {taskset} {binary}"
            _print(f"Running command: {command}")
            try:
                proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=cwd,
                    shell=True,
                )

                def _send(cmd: str) -> None:
                    self._send(proc, cmd)
                    logfile.write(f"SIM: {cmd}")
                    logfile.flush()

                received = ""
                while received != "end":
                    time.sleep(self.poll_interval)
                    received = (
                        proc.stdout.readline().decode("UTF-8").strip().split()
                    )  # type: ignore
                    logfile.write(f"CTRL: {' '.join(received)}\n")
                    logfile.flush()
                    if len(received) == 0:
                        continue
                    if received[0] == "end":
                        _print("Ending simulation.")
                        return
                    if received[0] == "log":
                        _print(" ".join(received[1:]))
                    elif received[0] == "ready":
                        _print("Received ready signal from simulation")
                        _send(f"fifodepth {depth}")
                        _send(f"runSamples {samples}")
                        # _send(f"runCycles 10000000")
                        _print("Settings sent to simulation.")
                    elif received[0] == "cycles":
                        if self.progress is None:
                            _print(" ".join(received[1:]))
                        else:
                            self.progress.update(name, int(received[1]), int(received[2]))
                    elif received[0] == "samples":
                        if self.progress is None:
                            _print(" ".join(received[1:]))
                        else:
                            self.progress.update(name, int(received[1]), int(received[2]))
                    elif received[0] == "started":
                        with self.running_lock:
                            self.running += 1
                            _print(f"Running: {self.running} / {self.total}")
                    elif received[0] == "stopped":
                        with self.running_lock:
                            self.running -= 1
                            _print(f"Running: {self.running} / {self.total}")
                    # elif received[0] == "error":
                    #     _print("ERROR: " + " ".join(received[1:]))
                    #     self.stop_flag = True
                    #     return
                    else:
                        raise FINNInternalError(
                            f"Simulation {name}: Unrecognized command: {received}"
                        )
            except Exception as e:
                self.console.log(f"Exception caught during simulation execution ({name}): {e}")
                self.console.log(traceback.format_exc())
                sys.exit(1)

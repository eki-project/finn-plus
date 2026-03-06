"""Node connected parallel simulations."""

import glob
import json
import math
import os
import pandas as pd
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from enum import Enum
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from rich.console import Console
from threading import Barrier
from typing import Any, cast

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.transformation.fpgadataflow.set_fifo_depths import get_fifo_split_configs
from finn.transformation.fpgadataflow.simulation import Simulation, SimulationType, store_fifo_data
from finn.transformation.fpgadataflow.simulation_controller import SimulationController
from finn.util.basic import make_build_dir
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log


# Hardware BRAM FIFOs lose entries to internal pipeline registers compared to the software FIFO
# model (which has exact capacity). This constant accounts for that overhead so that the
# minimization algorithm finds depths that are safe to deploy on hardware.
BRAM_FIFO_PIPELINE_OVERHEAD = 2


def _count_bram_sub_fifos(depth: int, max_qsrl_depth: int) -> int:
    """Return the number of BRAM (vivado) sub-FIFOs that *depth* decomposes into.

    Non-power-of-two BRAM FIFOs are decomposed into several power-of-two sub-FIFOs by
    get_fifo_split_configs.  Each sub-FIFO whose style is "vivado" has its own pipeline
    register overhead, so the total overhead scales with the sub-FIFO count.
    """
    return sum(1 for _, style in get_fifo_split_configs(depth, max_qsrl_depth) if style == "vivado")


def _safe_bram_starting_depth(peak_util: int, max_qsrl_depth: int) -> int:
    """Return the smallest depth d such that d minus its BRAM pipeline overhead >= peak_util + 1.

    For LUTRAM depths (d <= max_qsrl_depth) the software model is exact so no overhead is needed.
    For BRAM depths the overhead depends on how many sub-FIFOs the decomposition produces,
    which itself depends on d.  We iterate (typically 1-2 steps) until the overhead stabilises.
    """
    d = max(peak_util + 1, 32)
    if d <= max_qsrl_depth:
        return d
    # Iteratively find d where d - num_vivado(d)*overhead >= peak_util + 1
    overhead = 0
    while True:
        d = peak_util + 1 + overhead
        num_vivado = _count_bram_sub_fifos(d, max_qsrl_depth)
        new_overhead = num_vivado * BRAM_FIFO_PIPELINE_OVERHEAD
        if new_overhead <= overhead:
            break
        overhead = new_overhead
    return max(d, 32)


class MinimizationOrder(Enum):
    """The order in which the search algorithm minimizes the FIFO depths."""

    NODE_ORDER = 0
    REVERSE_NODE_ORDER = 1
    LARGEST_BITWIDTH_DIFF_FIRST = 2
    SMALLEST_BITWIDTH_DIFF_FIRST = 3

    # Non black-box model orders
    AFTER_THRESHOLDS_FIRST = 4
    AFTER_DWC_FIRST = 5

    # Half black-box
    # If we ran a sim before, we know the largest FIFOs, so start with these.
    # This strategy might work, if the changes to the model are small enough
    REUSE_PREVIOUS_ORDER = 6


class NodeConnectedSimulationController(SimulationController):
    """Run simulations for node connected cases."""

    def __init__(
        self,
        parallel_simulations: int,
        names: list[str],
        binaries: list[Path],
        console: Console,
        poll_interval: float = 1.0,
        with_progressbar: bool = True,
    ) -> None:
        """Set up node connected simulation."""
        super().__init__(
            parallel_simulations, names, binaries, console, poll_interval, with_progressbar
        )
        # Synchronization barrier for configuration phase
        self.sync_barrier: Barrier | None = None
        for binary in binaries:
            if not binary.exists():
                console.log(f"Binary {binary} does not exist!")
                raise FINNUserError(f"Binary {binary} does not exist!")

    def _cleanup_shm_resources(self) -> None:
        """Remove any existing shared memory segments and semaphores from /dev/shm."""
        try:
            # Collect potential shared memory and semaphore names based on node names
            shm_patterns = []
            # Pattern for shared memory segments (e.g., /nodename_0, /nodename_1)
            shm_patterns.append("/dev/shm/*")

            removed_count = 0
            for pattern in shm_patterns:
                for filepath in glob.glob(pattern):
                    try:
                        Path(filepath).unlink()
                        removed_count += 1
                    except (FileNotFoundError, PermissionError):  # noqa: PERF203
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

        # Clean up any existing shared memory resources before starting
        self._cleanup_shm_resources()

        # Initialize barrier for all simulations to synchronize after configuration
        self.sync_barrier = Barrier(len(self.names))

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
                            i % len(os.sched_getaffinity(0))
                            if len(os.sched_getaffinity(0)) < len(self.names)
                            else -1,  # sched_getaffinity needed, because
                            # cpu_count does not handle well with workload schedulers.
                            # We only pin the core if we have more simulations than cores to avoid
                            # simulations moving around too much and hurting performance. If we have
                            # more cores than simulations, we leave it to the OS to schedule.
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
                                    fifo_cycles_until_first_valid,
                                ) = result
                                fifo_depths[sim_name] = fifo_depth
                                fifo_results[sim_name] = fifo_util
                                cycles_results[sim_name] = cycles
                                samples_results[sim_name] = samps
                                intervals_results[sim_name] = intervals
                                fifo_cycles_until_first_valid_results[sim_name] = (
                                    fifo_cycles_until_first_valid
                                )
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
                            ) = result
                            # Only update if not already collected
                            if sim_name not in fifo_results:
                                fifo_cycles_until_first_valid_results[sim_name] = (
                                    fifo_cycles_until_first_valid
                                )
                                fifo_depths[sim_name] = fifo_depth
                                fifo_results[sim_name] = fifo_util
                                cycles_results[sim_name] = cycles
                                samples_results[sim_name] = samps
                                intervals_results[sim_name] = intervals
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
        fifo_first_valid_cycles: list[int] | None = None,
    ) -> tuple[str, list[int], int, int, list[int], bool, list[int], list[int]] | None:
        """Run the specified simulation binary in a new subprocess and communicate with it.

        Args:
            binary: Path to simulation binary
            name: Name of simulation node
            _cpu: CPU affinity (unused)
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
                proc_idx = self._start_process(
                    binary, process_index, cpu=_cpu if _cpu is not None else -1
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

                # Poll for status updates
                while True:
                    # Check if we should stop early
                    with self.stop_lock:
                        if self.should_stop:
                            try:
                                stop_response = self._send_and_receive(proc_idx, "stop", {})
                            except (BrokenPipeError, ConnectionResetError, RuntimeError):
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
                            )
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
                        fifo_cycles_until_first_valid = response.get(
                            "fifo_cycles_until_first_valid", []
                        )
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

                if stop_response:
                    fifo_util = stop_response.get("fifo_utilization", [])
                    fifo_depth = stop_response.get("fifo_depth", [])
                    cycles = stop_response.get("cycles", 0)
                    samps = stop_response.get("samples", 0)
                    fifo_cycles_until_first_valid = stop_response.get(
                        "fifo_cycles_until_first_valid", []
                    )
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
    def __init__(
        self,
        model: ModelWrapper,
        simulation_type: SimulationType,
        fpgapart: str,
        clk_ns: float,
        functional_sim: bool,
        workers: int | None = None,
        max_qsrl_depth: int = 256,
    ) -> None:
        super().__init__(model, simulation_type, fpgapart, clk_ns, functional_sim, workers)
        self.max_qsrl_depth = max_qsrl_depth

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
        names = [node.name for node in self.model.graph.node]
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
        controller = NodeConnectedSimulationController(
            len(self.binaries), names, list(self.binaries.values()), Console(), 0.1, False
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
                }
            )
        json.dump(data, output_json.open("w"), indent=4)
        return data, merged_data.get("timeout_occurred", False)


class RunLayerParallelSimulation(Transformation):  # noqa
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
            # TODO: Set to ALL search orders
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
                    hw: HWCustomOp = getCustomOp(model.graph.node[i])
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
        sim = NodeConnectedSimulation(
            model,
            SimulationType.NODE_BASED_CONNECTED,
            self.fpgapart,
            self.clk_ns,
            self.cfg.functional_simulation,
            max_qsrl_depth=self.max_qsrl_depth,
        )
        model = sim.model  # TODO:clean up

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

        # TODO: The final depths contained a lot of -1 (default values).
        # Did we need to write the initial depths into there?
        # Or in case of minimization skip we likely need to write the values still.

        # Running the initial simulation
        log.info("Running initial node-connected simulation.")
        initial_fifo_depths, _ = sim.simulate()

        # Store the initial sizes as a report
        initial_sizes_path = (
            Path(self.cfg.output_dir) / "report" / "initial_fifo_sizes_sim_connected.json"
        )
        initial_sizes_path.write_text(json.dumps(initial_fifo_depths, indent=4))
        log.info(f"Wrote initial sizes to: {initial_sizes_path}")

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
            hw_node = getCustomOp(model.graph.node[node_idx])
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
                        f"[ [bold green]{percentage}%[/bold green] ] "
                        f"[ {node_idx}.{fifo_idx} / {len(fifo_depths) - 1} ] Simulation completed "
                        f"({iterations_needed} iterations).",
                        extra={"markup": True, "highlighter": None},
                    )

            self.final_depths[minimization_order] = deepcopy(fifo_depths)

            order_percent = int(100.0 * float(k + 1) / float(len(self.minimization_orders)))
            log.info(
                f"[ [bold gold1]{order_percent}%[/bold gold1] ] "
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
            Path(self.cfg.output_dir) / "report" / "fifo_data.csv",
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
                    # if len(fifo_depths[i]) > 1:
                    #     blocks_plus_one = self._get_valid_block_counts(
                    #         blocks + 1, blocks + 1000, bw
                    #     )
                    #     _, max_d = calculate_bram_depth_range(blocks_plus_one[0], bw)
                    # else:
                    _, max_d = calculate_bram_depth_range(blocks, bw)
                    fifo_depths[node_idx][fifo_idx] = max_d

        log.info("Final FIFO depths:")
        for node_idx in range(len(fifo_depths)):
            log.info(f"{node_idx}: {fifo_depths[node_idx]}")

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
        writeback_path = Path(self.cfg.output_dir) / "fifo_config.json"
        assert len(fifo_depths) == len(model.graph.node)
        json_results = []
        for node_idx, node in enumerate(model.graph.node):
            json_results.append({"node": node.name, "depths": fifo_depths[node_idx]})
        with writeback_path.open("w") as f:
            json.dump(json_results, f)
        log.info(f"Wrote results back to {writeback_path}")

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
        """Get all valid BRAM block counts in the specified range.

        Some block counts are invalid for certain bitwidths due to quantization.
        This method returns only the valid configurations.

        Args:
            min_blocks: Minimum block count (inclusive)
            max_blocks: Maximum block count (inclusive)
            bitwidth: Data bitwidth

        Returns:
            Sorted list of valid block counts
        """
        valid_blocks = []
        for blocks in range(min_blocks, max_blocks + 1):
            _, max_d = calculate_bram_depth_range(blocks, bitwidth)
            if max_d > 0:  # Valid configuration
                valid_blocks.append(blocks)
        return valid_blocks

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
        """Minimize a single FIFO depth using binary search.

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
        iterations = 0
        original_size = current_depths[node_idx][fifo_idx]
        bw = bit_widths[node_idx][fifo_idx]

        log.debug(f"Minimizing Node {node_idx} FIFO {fifo_idx}: original depth {original_size}")

        # If FIFO depth of 32 works, use it because it fits into bw/2 LUTs
        success, timeout = self._test_depth(
            32,
            node_idx,
            fifo_idx,
            current_depths,
            initial_fifo_depths,
            sim,
            sim_cycles,
            fifo_first_valid_cycles,
        )
        iterations += 1
        if success:
            return 32, iterations

        if original_size <= self.max_qsrl_depth:
            upper_luts = calculate_srl16e_luts(original_size, bw)
            # LUTRAM based FIFOs have block sizes of 32, so smallest after 32 is 64
            lower_luts = calculate_srl16e_luts(64, bw)

            # Binary search if there's room to search
            if upper_luts > lower_luts:
                best_working_depth, bin_it = self._binary_search_srl_depth(
                    node_idx,
                    fifo_idx,
                    current_depths,
                    bw,
                    initial_fifo_depths,
                    sim,
                    sim_cycles,
                    fifo_first_valid_cycles,
                    lower_luts=lower_luts,
                    upper_luts=upper_luts,
                )
                iterations += bin_it
                return best_working_depth, iterations
            return original_size, iterations

        # Try FIFO depth of 256 next (fits into LUTRAM)
        success, timeout = self._test_depth(
            self.max_qsrl_depth,
            node_idx,
            fifo_idx,
            current_depths,
            initial_fifo_depths,
            sim,
            sim_cycles,
            fifo_first_valid_cycles,
        )
        iterations += 1
        if success:
            upper_luts = calculate_srl16e_luts(self.max_qsrl_depth, bw)
            # LUTRAM based FIFOs have block sizes of 32, so smallest after 32 is 64
            lower_luts = calculate_srl16e_luts(64, bw)

            # Binary search if there's room to search
            if upper_luts > lower_luts:
                best_working_depth, bin_it = self._binary_search_srl_depth(
                    node_idx,
                    fifo_idx,
                    current_depths,
                    bw,
                    initial_fifo_depths,
                    sim,
                    sim_cycles,
                    fifo_first_valid_cycles,
                    lower_luts=lower_luts,
                    upper_luts=upper_luts,
                )
                iterations += bin_it
                return best_working_depth, iterations
            return self.max_qsrl_depth, iterations

        # We know 256 doesn't work, so we have to use BRAMs
        # Try one BRAM block less than current
        upper_blocks = calculate_bram_blocks(original_size, bw)
        # Get all valid block counts in the range
        valid_blocks = self._get_valid_block_counts(1, upper_blocks - 1, bw)
        if not valid_blocks:
            # No valid configurations exist
            return original_size, iterations
        # Test the maximum valid block count first
        # (largest depth below original, most likely to succeed)
        max_valid_blocks = valid_blocks[-1]
        _, max_d = calculate_bram_depth_range(max_valid_blocks, bw)

        success, timeout = self._test_depth(
            max_d,
            node_idx,
            fifo_idx,
            current_depths,
            initial_fifo_depths,
            sim,
            sim_cycles,
            fifo_first_valid_cycles,
        )
        iterations += 1

        if timeout or not success:
            return original_size, iterations

        best_working_depth = max_d

        # Binary search if there's room to search and multiple valid configs
        if len(valid_blocks) > 1:
            best_working_depth, bin_it = self._exponential_binary_search_depth(
                node_idx,
                fifo_idx,
                current_depths,
                bw,
                initial_fifo_depths,
                sim,
                sim_cycles,
                fifo_first_valid_cycles,
                valid_blocks=valid_blocks,
            )
            iterations += bin_it

        return best_working_depth, iterations

    def _exponential_binary_search_depth(
        self,
        node_idx: int,
        fifo_idx: int,
        current_depths: list,
        bitwidth: int,
        initial_fifo_depths: list[dict[str, list[int]]],
        sim: NodeConnectedSimulation,
        sim_cycles: float,
        fifo_first_valid_cycles: list[list[int]],
        valid_blocks: list[int],
    ) -> tuple[int, int]:
        """Perform exponential + binary search over valid block configurations.

        Uses exponential search to quickly find the range, then binary search within it.
        This is more efficient when smaller block counts are more likely.
        Only searches over pre-validated block counts.

        Args:
            node_idx: Node index
            fifo_idx: FIFO index within node
            current_depths: Current working FIFO depth configuration.  FIFOs already
                minimised in this pass carry their final depth; this list must not be
                modified directly (``_test_depth`` deep-copies it before trial edits).
            bitwidth: Data bitwidth
            initial_fifo_depths: Baseline performance data
            sim: Simulation controller
            sim_cycles: Maximum simulation cycles
            fifo_first_valid_cycles: First valid cycle for each FIFO
            valid_blocks: Sorted list of valid block counts to search over

        Returns:
            Tuple: Best working depth found, Number of iterations required to arrive at this result.
        """
        iterations = 0
        if not valid_blocks:
            raise FINNInternalError("valid_blocks list cannot be empty")

        # Start with the largest valid block count (known to work from caller)
        _, max_d = calculate_bram_depth_range(valid_blocks[-1], bitwidth)
        best_working_depth = max_d

        # Exponential search phase: find range where solution exists
        # Check positions: 0, 1, 2, 4, 8, ... indices in valid_blocks list
        lower_idx = 0
        upper_idx = len(valid_blocks) - 1
        exp_idx = 0
        last_failed_idx = -1

        while exp_idx < upper_idx:
            blocks = valid_blocks[exp_idx]
            _, max_d = calculate_bram_depth_range(blocks, bitwidth)

            success, _ = self._test_depth(
                max_d,
                node_idx,
                fifo_idx,
                current_depths,
                initial_fifo_depths,
                sim,
                sim_cycles,
                fifo_first_valid_cycles,
            )
            iterations += 1

            if success:
                # Found a working depth, now binary search in [last_failed_idx+1, exp_idx]
                best_working_depth = max_d
                lower_idx = last_failed_idx + 1
                upper_idx = exp_idx
                break
            # This doesn't work, try exponentially larger index
            last_failed_idx = exp_idx
            exp_idx = min(exp_idx * 2 if exp_idx > 0 else 1, upper_idx)

        # Binary search phase: refine the range
        while lower_idx < upper_idx:
            mid_idx = (lower_idx + upper_idx) // 2
            blocks = valid_blocks[mid_idx]
            _, max_d = calculate_bram_depth_range(blocks, bitwidth)

            success, _ = self._test_depth(
                max_d,
                node_idx,
                fifo_idx,
                current_depths,
                initial_fifo_depths,
                sim,
                sim_cycles,
                fifo_first_valid_cycles,
            )
            iterations += 1

            if success:
                # This depth works, try smaller (lower indices)
                best_working_depth = max_d
                upper_idx = mid_idx
            else:
                # This depth doesn't work, need larger (higher indices)
                lower_idx = mid_idx + 1

        return best_working_depth, iterations

    def _binary_search_srl_depth(
        self,
        node_idx: int,
        fifo_idx: int,
        current_depths: list,
        bitwidth: int,
        initial_fifo_depths: list[dict[str, list[int]]],
        sim: NodeConnectedSimulation,
        sim_cycles: float,
        fifo_first_valid_cycles: list[list[int]],
        lower_luts: int,
        upper_luts: int,
    ) -> tuple[int, int]:
        """Perform binary search to find minimal working FIFO depth in LUTRAM range.

        Args:
            node_idx: Node index
            fifo_idx: FIFO index within node
            current_depths: Current working FIFO depth configuration.  FIFOs already
                minimised in this pass carry their final depth; this list must not be
                modified directly (``_test_depth`` deep-copies it before trial edits).
            bitwidth: Data bitwidth
            initial_fifo_depths: Baseline performance data
            sim: Simulation controller
            sim_cycles: Maximum simulation cycles
            fifo_first_valid_cycles: First valid cycle for each FIFO
            lower_luts: Lower bound for LUT count
            upper_luts: Upper bound for LUT count (known to work)

        Returns:
            Tuple: Best working depth found, Number of Iterations required to arrive at this result
        """
        iterations = 0
        _, max_d = calculate_srl16e_depth_range(upper_luts, bitwidth)
        best_working_depth = max_d

        while lower_luts < upper_luts:
            mid_luts = (lower_luts + upper_luts) // 2

            # Prevent infinite loop
            if mid_luts == upper_luts:
                mid_luts = upper_luts - 1
            if mid_luts < lower_luts:
                break

            # Find valid depth for this LUT count
            _, max_d = calculate_srl16e_depth_range(mid_luts, bitwidth)

            if max_d == 0:
                # No valid configuration, try more LUTs
                lower_luts = mid_luts + 1
                continue

            success, _ = self._test_depth(
                max_d,
                node_idx,
                fifo_idx,
                current_depths,
                initial_fifo_depths,
                sim,
                sim_cycles,
                fifo_first_valid_cycles,
            )
            iterations += 1

            if success:
                # This depth works, try smaller
                best_working_depth = max_d
                upper_luts = mid_luts
            else:
                # This depth doesn't work, need larger
                lower_luts = mid_luts + 1

        return best_working_depth, iterations

    def _needs_minimization(self, fifo_depth: int, bitwidth: int) -> bool:
        """Determine whether a FIFO can be minimized further.

        Args:
            fifo_depth: Current FIFO depth
            bitwidth: Data bitwidth

        Returns:
            True if the FIFO can be minimized further, False otherwise.
        """
        # Qsrl FIFO Formula: LUTs = ⌈depth/32⌉ x ⌈bitwidth/2⌉
        if fifo_depth <= 32:  # FIFOs of depth <=32 fit into bitwidth/2 LUTs
            return False
        # Return False if exactly the minimum number of possible BRAM blocks is used for this
        # bitwidth and depth is sufficiently large that further optimization is unlikely to succeed
        return not (
            calculate_bram_blocks(fifo_depth, bitwidth)
            <= self._get_valid_block_counts(1, bitwidth, bitwidth)[0]
            and fifo_depth > math.floor(self.max_qsrl_depth * 1.1)
        )


def calculate_bram_blocks(depth: int, bitwidth: int) -> int:
    """Calculate the number of BRAM blocks required for a BRAM FIFO.

    Args:
        depth: FIFO depth
        bitwidth: Data bitwidth
    """
    if bitwidth == 1:
        return math.ceil(depth / 16384)
    if bitwidth == 2:
        return math.ceil(depth / 8192)
    if bitwidth <= 4:
        return (math.ceil(depth / 4096)) * (math.ceil(bitwidth / 4))
    if bitwidth <= 9:
        return (math.ceil(depth / 2048)) * (math.ceil(bitwidth / 9))
    if bitwidth <= 18 or depth > 512:
        return (math.ceil(depth / 1024)) * (math.ceil(bitwidth / 18))
    return (math.ceil(depth / 512)) * (math.ceil(bitwidth / 36))


def calculate_bram_depth_range(blocks: int, bitwidth: int) -> tuple[int, int]:
    """Calculate the range of FIFO depths that use exactly the given number of BRAM blocks.

    Args:
        blocks: Number of BRAM blocks
        bitwidth: Data bitwidth

    Returns:
        Tuple of (min_depth, max_depth) that uses exactly 'blocks' BRAM blocks.
    """
    if blocks < 1:
        raise FINNInternalError("Number of BRAM blocks must be at least 1")

    # Invert the formula from calculate_bram_blocks based on bitwidth
    if bitwidth == 1:
        # blocks = ⌈depth/16384⌉
        # Inversion: (blocks-1)*16384 < depth ≤ blocks*16384
        min_depth = (blocks - 1) * 16384 + 1 if blocks > 1 else 1
        max_depth = blocks * 16384
    elif bitwidth == 2:
        # blocks = ⌈depth/8192⌉
        # Inversion: (blocks-1)*8192 < depth ≤ blocks*8192
        min_depth = (blocks - 1) * 8192 + 1 if blocks > 1 else 1
        max_depth = blocks * 8192
    elif bitwidth <= 4:
        # blocks = ⌈depth/4096⌉ * ⌈bitwidth/4⌉
        bitwidth_factor = math.ceil(bitwidth / 4)
        depth_blocks = math.ceil(blocks / bitwidth_factor)
        min_depth = (depth_blocks - 1) * 4096 + 1 if depth_blocks > 1 else 1
        max_depth = depth_blocks * 4096
    elif bitwidth <= 9:
        # blocks = ⌈depth/2048⌉ * ⌈bitwidth/9⌉
        bitwidth_factor = math.ceil(bitwidth / 9)
        depth_blocks = math.ceil(blocks / bitwidth_factor)
        min_depth = (depth_blocks - 1) * 2048 + 1 if depth_blocks > 1 else 1
        max_depth = depth_blocks * 2048
    elif bitwidth <= 18:
        # blocks = ⌈depth/1024⌉ * ⌈bitwidth/18⌉
        bitwidth_factor = math.ceil(bitwidth / 18)
        depth_blocks = math.ceil(blocks / bitwidth_factor)
        min_depth = (depth_blocks - 1) * 1024 + 1
        max_depth = depth_blocks * 1024
    else:
        # bitwidth > 18, split into two cases from original function
        # Case 1: depth > 512 uses ⌈depth/1024⌉ * ⌈bitwidth/18⌉
        # Case 2: depth ≤ 512 uses ⌈depth/512⌉ * ⌈bitwidth/36⌉

        # Try the depth > 512 case first (⌈depth/1024⌉ * ⌈bitwidth/18⌉)
        bitwidth_factor = math.ceil(bitwidth / 18)
        depth_blocks = math.ceil(blocks / bitwidth_factor)

        # Check if blocks is achievable with this bitwidth factor
        if blocks % bitwidth_factor != 0 or depth_blocks < 1:
            # Try the depth ≤ 512 case instead
            pass
        else:
            min_depth = max((depth_blocks - 1) * 1024 + 1, 513)  # Must be > 512
            max_depth = depth_blocks * 1024
            # Check if this range is valid (entirely > 512)
            if min_depth > 512 and calculate_bram_blocks(min_depth, bitwidth) == blocks:
                return (min_depth, max_depth)

        # Try the depth ≤ 512 case (⌈depth/512⌉ * ⌈bitwidth/36⌉)
        bitwidth_factor = math.ceil(bitwidth / 36)
        depth_blocks = math.ceil(blocks / bitwidth_factor)

        # Check if blocks is achievable with this bitwidth factor
        if blocks % bitwidth_factor != 0 or depth_blocks < 1:
            return (0, 0)  # Invalid block count for this bitwidth

        min_depth = (depth_blocks - 1) * 512 + 1 if depth_blocks > 1 else 1
        max_depth = min(depth_blocks * 512, 512)  # Must be ≤ 512

        # Verify the range is valid (entirely ≤ 512 and produces correct block count)
        if max_depth <= 512 and calculate_bram_blocks(min_depth, bitwidth) == blocks:
            return (min_depth, max_depth)

        return (0, 0)  # No valid range found

    # Verify the range is valid
    if calculate_bram_blocks(min_depth, bitwidth) != blocks:
        raise FINNInternalError("Calculated BRAM depth range is invalid!")
    return (min_depth, max_depth)


def calculate_uram_blocks(depth: int, bitwidth: int) -> int:
    """Calculate the number of URAM blocks required for a URAM FIFO.

    Args:
        depth: FIFO depth
        bitwidth: Data bitwidth
    """
    return (math.ceil(depth / 4096)) * (math.ceil(bitwidth / 72))


def calculate_uram_depth_range(blocks: int, bitwidth: int) -> tuple[int, int]:
    """Calculate the range of FIFO depths that use exactly the given number of URAM blocks.

    Args:
        blocks: Number of URAM blocks
        bitwidth: Data bitwidth

    Returns:
        Tuple of (min_depth, max_depth) that uses exactly 'blocks' URAM blocks.
        Returns (0, 0) if no valid range exists.
    """
    if blocks < 1:
        return (0, 0)

    # URAM formula: blocks = ⌈depth/4096⌉ * ⌈bitwidth/72⌉
    bitwidth_factor = math.ceil(bitwidth / 72)

    # Calculate depth range
    # Minimum depth: (blocks / bitwidth_factor - 1) * 4096 + 1
    # Maximum depth: (blocks / bitwidth_factor) * 4096

    if blocks % bitwidth_factor != 0:
        return (0, 0)  # Invalid block count for this bitwidth

    depth_blocks = blocks // bitwidth_factor
    min_depth = (depth_blocks - 1) * 4096 + 1 if depth_blocks > 1 else 1
    max_depth = depth_blocks * 4096

    # Verify
    if calculate_uram_blocks(min_depth, bitwidth) != blocks:
        return (0, 0)

    return (min_depth, max_depth)


def calculate_srl16e_luts(depth: int, bitwidth: int) -> int:
    """Calculate the number of SRL16E LUTs required for a FIFO.

    Args:
        depth: FIFO depth (must be >= 2)
        bitwidth: Data bitwidth

    Returns:
        Number of SRL16E LUTs required without adress LUTs.

    Formula: LUTs = ⌈depth/32⌉ x ⌈bitwidth/2⌉
    """
    ram_luts = (math.ceil(depth / 32)) * (math.ceil(bitwidth / 2))
    return ram_luts


def calculate_srl16e_depth_range(luts: int, bitwidth: int) -> tuple[int, int]:
    """Calculate the range of FIFO depths that use exactly the given number of SRL16E LUTs.

    Args:
        luts: Number of SRL16E LUTs
        bitwidth: Data bitwidth

    Returns:
        Tuple of (min_depth, max_depth) that uses exactly 'luts' LUTs.
        Returns (0, 0) if no valid range exists.
    """
    if luts < 1:
        return (0, 0)

    # SRL16E formula: luts = ⌈depth/32⌉ * ⌈bitwidth/2⌉
    bitwidth_factor = math.ceil(bitwidth / 2)

    # Calculate depth range
    if luts % bitwidth_factor != 0:
        return (0, 0)  # Invalid LUT count for this bitwidth

    depth_blocks = luts // bitwidth_factor
    min_depth = (depth_blocks - 1) * 32 + 1 if depth_blocks > 1 else 2
    max_depth = depth_blocks * 32

    # Verify
    if calculate_srl16e_luts(min_depth, bitwidth) != luts:
        return (0, 0)

    return (min_depth, max_depth)

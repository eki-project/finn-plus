"""Simulating layers on their own to observe their behaviour."""
import io
import json
import pandas as pd
import re
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path, PosixPath, PurePath
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from rich.console import Console
from threading import Lock
from typing import Any, Literal, TypeAlias

from finn.transformation.fpgadataflow.simulation import Simulation, store_fifo_data
from finn.transformation.fpgadataflow.simulation_build import SimulationType
from finn.transformation.fpgadataflow.simulation_controller import SimulationController
from finn.util.exception import FINNInternalError
from finn.util.logging import log


def get_time() -> str:
    """Return the current time in a formatted hour:minutes:second string."""
    return f"[{time.strftime('%H:%M:%S')}]"


class NodeIsolatedSimulationController(SimulationController):
    """Run simulations for node isolated cases."""

    IsolatedSimLogData = dict[Literal["ready", "valid"], list[dict[str, int]]]

    def __init__(
        self,
        parallel_simulations: int,
        names: list[str],
        binaries: list[Path],
        console: Console,
        poll_interval: float = 1.0,
        with_progressbar: bool = False,
    ) -> None:
        """Set up node isolated simulation."""
        super().__init__(
            parallel_simulations, names, binaries, console, poll_interval, with_progressbar
        )
        log.info("Started simulation controller")

    def get_logfile_path(self, binary_or_idx: Path | int) -> Path:
        """Get the logfile for the given binary or process index."""
        if type(binary_or_idx) is int:
            return (
                self.logdir / f"{binary_or_idx}_log_isolated_"
                f"{self.names[binary_or_idx]}_python.txt"
            )
        elif type(binary_or_idx) in [Path, PurePath, PosixPath]:  # noqa
            process_idx = self.binaries.index(binary_or_idx)  # type: ignore
            return self.logdir / f"{process_idx}_log_isolated_{self.names[process_idx]}_python.txt"
        raise TypeError("Pass either a simulation binary path of an index")

    def write_log(self, logfile: io.TextIOWrapper, msg: str, flush: bool = True) -> None:
        """Write a timestamped message to log."""
        logfile.write(f"{get_time()} {msg}\n")
        if flush:
            logfile.flush()

    def collect_results(
        self, d: Path, readylog_name: str = "readylog.txt", validlog_name: str = "validlog.txt"
    ) -> IsolatedSimLogData:
        """Recieve the directory containing a binary and the simulation logs.
        If no logs are found raises an error, otherwise return the postprocessed logs
        read from JSON.
        """
        readylog = d / readylog_name
        validlog = d / validlog_name
        if not readylog.exists() or not validlog.exists():
            raise FINNInternalError(f"Could not find simulation logs at {readylog} and {validlog}")
        return {
            "ready": json.loads(readylog.read_text()),
            "valid": json.loads(validlog.read_text()),
        }

    def run(self) -> dict[str, IsolatedSimLogData]:
        """Run a node isolated simulation and return the collected
        input ready / output valid data, indexed based on node names."""
        futures: list[Future] = []
        datalock = Lock()
        total = len(self.binaries)
        done = 0

        # Important to initialize from names. Otherwise the results are added into the dict
        # in the order in which they finished simulating. But we want to keep the model order.
        data: dict[str, self.IsolatedSimLogData] = {name: {} for name in self.names}

        # TODO: Lock not needed; futures are not consumed just by
        # TODO: using the callback, so we can unpack them later

        # Callback to show progress and save the simulation result
        def _done_callback_generator(name: str) -> Callable:
            nonlocal total, done, data, datalock

            def _f(future: Future) -> None:
                nonlocal total, done, data, datalock
                with datalock:
                    done += 1
                    log.info(
                        f"[ [bold green]{int(100 * float(done)/float(total))}%"
                        f"[/bold green] ] {name} done!",
                        extra={"markup": True, "highlighter": None},
                    )
                    data[name] = future.result()

            return _f

        # Running the simulation threads
        assert len(self.names) == len(self.binaries)
        with self.console.status(f"Running simulation on every node. Log directory: {self.logdir}"):
            start = time.time()
            with ThreadPoolExecutor(len(self.binaries)) as tpe:
                for i, binary in enumerate(self.binaries):
                    futures.append(tpe.submit(self._run_binary, binary))
                    futures[-1].add_done_callback(_done_callback_generator(self.names[i]))
                tpe.shutdown(wait=True)
            elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start))
            log.info("Thread pool closed. Closing sockets and postprocessing data")
            log.info(f"Simulations took {elapsed}")

        # Finish the logs and clean up the sockets
        for binary in self.binaries:
            with self.get_logfile_path(binary).open("a") as logfile:
                self.write_log(logfile, "Cleaning up socket.")
        self._cleanup_sockets()

        # Check for invalid data points
        invalid = []
        for i, name in enumerate(data.keys()):
            if data[name] is None:
                invalid.append((name, i))
        if len(invalid) > 0:
            raise FINNInternalError(
                f"Lost connection / malformed response from nodes: "
                f"{', '.join([str(x) for x in invalid])}"
            )
        return data

    def _run_binary(self, binary: Path) -> IsolatedSimLogData | None:
        """Thread routine: Run a single simulation from the given path and return
        the collected results. Returns None if connection is lost."""
        process_index = self.binaries.index(binary)
        with self.get_logfile_path(binary).open("w+") as logfile:
            # Logging helper
            def write_log(msg: str) -> None:
                self.write_log(logfile, msg)

            # Initialize: Start simulation process and give the start command
            write_log("Initializing simulation")
            write_log(f"Binary is: {binary}")
            proc_idx = self._start_process(binary, process_index)
            response = self._send_and_receive(proc_idx, "start", {})
            if response is None:
                write_log(
                    "No answer for the clients 'start' " "command received. Timeout or disconnect."
                )
                return None
            write_log(f"Start response: {response}")

            # Main loop
            write_log("Beginning main loop")
            logfile.flush()
            total_status_requests = 0
            while True:
                # Request status in regular intervals
                time.sleep(self.poll_interval)
                write_log("Sending status request")
                response = self._send_and_receive(proc_idx, "status", {})
                total_status_requests += 1
                write_log(f"Status request {total_status_requests} sent.")

                # Process response
                if response is None:
                    write_log("Status request answered with None: Timeout or connection lost.")
                    return None
                state = response["state"]
                write_log(f"Received answer for status request ({total_status_requests})")

                # If the simulation is done, postprocess and return the collected data
                if state == "done":
                    write_log("Received done status. Sending stop signal to simulation.")
                    resp = self._send_and_receive(proc_idx, "stop", {})
                    if resp is None:
                        write_log("No stop response received.")
                    else:
                        write_log("Stop successfully received.")
                    return self.collect_results(binary.parent)

                # Otherwise log the current status
                # TODO: Field name - meaning wrong?
                in_done = response["inputCyclesDone"]
                in_target = response["inputCyclesTarget"]
                out_done = response["outputCyclesDone"]
                out_target = response["outputCyclesTarget"]
                total_cycles = response["totalCycles"]
                percent_simulated_input = int(100.0 * float(in_done) / float(in_target))
                percent_simulated_output = int(100.0 * float(out_done) / float(out_target))
                write_log("Status response:")
                write_log(f"\tTotal cycles: {total_cycles}")
                write_log(
                    f"\tInput data simulated: {percent_simulated_input}% "
                    f"({in_done} / {in_target})"
                )
                write_log(
                    f"\tOutput data simulated: {percent_simulated_output}% "
                    f"({out_done} / {out_target})"
                )


FIFODepthConfig: TypeAlias = dict[int, dict[str, str | list[int]]]
IsoSimLogData = NodeIsolatedSimulationController.IsolatedSimLogData
IsoSimLogDataByLayer = dict[str, IsoSimLogData]  # Indexed by layer name


class IsolatedSimulation(Simulation):
    def __init__(
        self,
        model: ModelWrapper,
        simulation_type: SimulationType,
        fpgapart: str,
        clk_ns: float,
        functional_sim: bool,
        workers: int | None = None,
    ) -> None:
        super().__init__(model, simulation_type, fpgapart, clk_ns, functional_sim, workers)

    def simulate(self) -> IsoSimLogDataByLayer:
        """Simulate isolated nodes."""
        if self.simulation_type != SimulationType.NODE_BASED_ISOLATED:
            raise FINNInternalError(
                f"Called simulation function 'simulate_node_isolated' "
                f"does not match provided simulation type "
                f"{self.simulation_type}"
            )
        names = [node.name for node in self.model.graph.node]
        console = Console()
        controller = NodeIsolatedSimulationController(
            len(self.binaries), names, list(self.binaries.values()), console, 0.1, False
        )
        return controller.run()


class RunLayerIsolatedSimulation(Transformation):
    """Run a layer isolated simulation and calculate some information for a
    later layer parallel simulation.

    This modifies or creates a pandas DF and stores it in a csv file. This file can be
    modified by the node connected simulation as well."""

    def __init__(
        self, fpgapart: str, clk_ns: float, functional_sim: bool, output_dir: Path
    ) -> None:
        """Run isolated layer simulations. The
        default location is at cfg.output_dir/report/fifo_data.csv."""
        super().__init__()
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns
        self.functional_sim = functional_sim
        self.output_dir = output_dir

        # Read / create dataframe with default path
        self.default_fifo_data_path = self.output_dir / "report" / "fifo_data.csv"

    def calculate_upper_bounds(self, data: IsoSimLogDataByLayer) -> dict[str, dict[str, int]]:
        """Try to calculate an upper bound for the incoming FIFO size of the layers.
        Return size indexed by layer name and stream name.

        >>> step = RunLayerIsolatedSimulation("", 0.0, False)
        >>> bounds = step.calculate_upper_bounds({
        ... "A": {
        ...         "ready": [
        ...             {"totalCycles": 43, "inputCyclesDone": 12,
        ...             "inputCyclesTarget": 24, "s_axi_0": 1, "s_axi_1": 0},
        ...             {"totalCycles": 44, "inputCyclesDone": 13,
        ...             "inputCyclesTarget": 24, "s_axi_0": 0, "s_axi_1": 0},
        ...         ], "valid": []
        ... },
        ... "B": {
        ...         "ready": [
        ...             {"totalCycles": 100, "inputCyclesDone": 3,
        ...             "inputCyclesTarget": 10, "s_axi_0": 1, "s_axi_1": 1,
        ...             "s_axi_2": 0},
        ...         ], "valid": []
        ... },
        ... "C": {
        ...         "ready": [
        ...             {"totalCycles": 43, "inputCyclesDone": 14,
        ...             "inputCyclesTarget": 24, "s_axi_0": 1, "s_axi_1": 0},
        ...             {"totalCycles": 44, "inputCyclesDone": 15,
        ...             "inputCyclesTarget": 24, "s_axi_0": 0, "s_axi_1": 0},
        ...         ], "valid": []
        ... }
        ... })
        >>> bounds["A"]
        {'s_axi_0': 1, 's_axi_1': 2}
        >>> bounds["B"]
        {'s_axi_0': 0, 's_axi_1': 0, 's_axi_2': 1}
        >>> bounds["C"]
        {'s_axi_0': 0, 's_axi_1': 0}
        """

        # TODO: Proper pytest tests
        def _any_ready(cycle_data: dict[str, int]) -> bool:
            for key in cycle_data.keys():
                if (
                    key not in ["totalCycles", "inputCyclesDone", "inputCyclesTarget"]
                    and cycle_data[key] == 1
                ):
                    return True
            return False

        results: dict[str, dict[str, int]] = {}
        for layer in data.keys():
            # Save all keys that are not
            results[layer] = {
                stream_name: 0
                for stream_name in data[layer]["ready"][0].keys()
                if stream_name not in ["inputCyclesDone", "inputCyclesTarget", "totalCycles"]
            }
            for cycle_data_ready, cycle_data_valid in zip(
                data[layer]["ready"], data[layer]["valid"], strict=True
            ):
                if cycle_data_ready["inputCyclesDone"] > int(
                    cycle_data_ready["inputCyclesTarget"] / 2.0
                ) and cycle_data_valid["outputCyclesDone"] > int(
                    cycle_data_valid["outputCyclesTarget"] / 2.0
                ):
                    break
                for stream_name in results[layer].keys():
                    # TODO: Currently on the C++ side we multiply the
                    # TODO: target cycles by 2, to get two samples
                    # TODO: We keep track of ready signals until we see
                    # TODO: the first ready after half of all cycles were seen.
                    # TODO: This might change in the future
                    if (
                        cycle_data_ready["inputCyclesTarget"] % 2 != 0
                        or cycle_data_valid["outputCyclesTarget"] % 2 != 0
                    ):
                        raise FINNInternalError(
                            f"An 'inputCyclesTarget' / 'outputCyclesTarget' of layer {layer} seems "
                            f"to not be an even number. Currently, we double "
                            f"the target simulation cycles for every layer "
                            f"on the C++ side. This error may point towards "
                            f"a change on the C++ side, which may cause the "
                            f"need to update this function accordingly!"
                        )
                    results[layer][stream_name] += int(cycle_data_ready[stream_name] == 0)

        # TODO: This calculation assumes, that if the producer does NOT fire the entire time,
        # TODO: the consumer can read at least at the same speed as
        #       if the producer did, and not slower.
        # TODO: (Since this would mean that less data pressure from
        #       the producer makes the consumer _slower_.)
        # TODO: This should usually be the case, but is important to keep in mind.
        return results

    def sanity_check_logged_data(self, data: IsoSimLogDataByLayer) -> None:
        """Do checks on the returned data to make sure it is in spec.

        A correctly formatted example would be:
        >>> data = {
        ...     "layer1": {
        ...         "ready": [{"totalCycles": 10, "inputCyclesDone": 5,
        ...                 "inputCyclesTarget": 10, "s_axi_0": 1}],
        ...         "valid": [{"totalCycles": 10, "outputCyclesDone": 5,
        ...                 "outputCyclesTarget": 10, "m_axi_0": 1}]
        ...     }
        ... }
        >>> sim = RunLayerIsolatedSimulation("", 0.0, False)
        >>> sim.sanity_check_logged_data(data)
        >>>
        """
        # 0. Valid and ready are present
        for layer, ldata in data.items():
            if "valid" not in ldata.keys():
                raise FINNInternalError(
                    f"Simulation log data of layer {layer} is missing the VALID log."
                )
            if "ready" not in ldata.keys():
                raise FINNInternalError(
                    f"Simulation log data of layer {layer} is missing the READY log."
                )
        # 1. All cycle datas are uniform and have at least one stream signal
        for i, (layer, ldata) in enumerate(data.items()):
            cycle_data = ldata["ready"] + ldata["valid"]
            lengths: set[int] = {len(cycle.keys()) for cycle in cycle_data}
            if len(lengths) != 1:
                raise FINNInternalError(
                    f"Simulation log data inconsistent for layer "
                    f"{layer} ({i}). Differing number of fields per cycle."
                )
            if next(iter(lengths)) < 4:
                raise FINNInternalError(
                    f"Simulation for layer {layer} must contain "
                    f"atleast 4 fields (total cycles, AXI cycles "
                    f"done, AXI cycles target and at least one AXI "
                    f"ready/valid signal)!"
                )
        # 2. All ready logs contain the required keywords
        readykeys = ["inputCyclesDone", "inputCyclesTarget", "totalCycles"]
        for rlayer, rdata in data.items():
            for cycle in rdata["ready"]:
                if any(keyword not in cycle.keys() for keyword in readykeys):
                    raise FINNInternalError(
                        f"Simulation READY log of layer {rlayer} "
                        f"contains cycles that are missing a required key."
                    )
                if any(key not in readykeys and "axi" not in key for key in cycle.keys()):
                    raise FINNInternalError(
                        f"In the READY simulation log of layer "
                        f"{rlayer} there seem to be fields that "
                        f"are not expected keywords or AXI streams!"
                    )
        # 3. All valid logs contain the required keywords
        validkeys = ["outputCyclesDone", "outputCyclesTarget", "totalCycles"]
        for vlayer, vdata in data.items():
            for cycle in vdata["valid"]:
                if any(keyword not in cycle.keys() for keyword in validkeys):
                    raise FINNInternalError(
                        f"Simulation VALID log of layer {vlayer} "
                        f"contains cycles that are missing a required key."
                    )
                if any(key not in validkeys and "axi" not in key for key in cycle.keys()):
                    raise FINNInternalError(
                        f"In the VALID simulation log of layer "
                        f"{vlayer} there seem to be fields that "
                        f"are not expected keywords or AXI streams!"
                    )
        # 4. Cycles done can never be larger then the number of total cycles passed in the sim
        for layer, cdata in data.items():
            for line in cdata["ready"] + cdata["valid"]:
                if (
                    "inputCyclesDone" in line.keys()
                    and line["inputCyclesDone"] > line["totalCycles"]
                ):
                    raise FINNInternalError(
                        f"Simulation log of layer {layer} looks incorrect: "
                        f"Number of active receiving cycles "
                        f"({line['inputCyclesDone']}) larger than number of "
                        f"total cycles passed ({line['totalCycles']})."
                    )
                if (
                    "outputCyclesDone" in line.keys()
                    and line["outputCyclesDone"] > line["totalCycles"]
                ):
                    raise FINNInternalError(
                        f"Simulation log of layer {layer} looks incorrect: "
                        f"Number of active producing cycles "
                        f"({line['outputCyclesDone']}) larger than number of "
                        f"total cycles passed ({line['totalCycles']})."
                    )
        # 5. Stream keywords can never have any other value than 1 (HIGH) or 0 (LOW)
        reserved_keywords = readykeys + validkeys
        for layer, ldata in data.items():
            for cycle_data in ldata["ready"] + ldata["valid"]:
                for key in cycle_data.keys():
                    if key not in reserved_keywords and cycle_data[key] not in [0, 1]:
                        raise FINNInternalError(
                            f"Layer {layer} has data point where a "
                            f"non-reserved field (thus an axi stream "
                            f"ready/valid signal) is neither 0 nor 1: "
                            f"Key: {key}, Value: {cycle_data[key]}"
                        )
        # 6. Data is not empty
        for layer, ldata in data.items():
            if len(ldata["ready"]) == 0:
                raise FINNInternalError(f"Layer {layer} has no ready data!")
            if len(ldata["valid"]) == 0:
                raise FINNInternalError(f"Layer {layer} has no valid data!")
        # 7. Check that the order of axi streams corresponds to their names. This helps
        # somewhat to guarantee that the order always stayed the same from building the simulations
        # to evaluating their data

        # The number in the name should increase with every stream, from 0, without gaps
        # and streams should be called "s_axis_<number>"
        readykeys = ["inputCyclesDone", "inputCyclesTarget", "totalCycles"]
        for layer, ldata in data.items():
            for cycledict in ldata["ready"]:
                current_stream_idx = 0
                for key in cycledict.keys():
                    if key not in readykeys:
                        m = re.fullmatch(r"^s_axis_(\d+)$", key)
                        if m is None:
                            raise FINNInternalError(
                                f"Layer {layer} has a non-expected key that "
                                f"does not match the names of streams expected "
                                f"(s_axis_<number>).\n\tKey is: {key}"
                            )
                        stream_idx = m.group(1)
                        if int(stream_idx) != current_stream_idx:
                            raise FINNInternalError(
                                f"Layer {layer} has non-expected stream key "
                                f"that does not follow the expected index "
                                f"scheme: Current expected index is "
                                f"{current_stream_idx}. Got instead: "
                                f"{stream_idx}"
                            )
                        current_stream_idx += 1
        # TODO: Check that names match vivado_stitch_ifnames.
        # TODO: Currently there is no easy way to do this, since we never save the isolated
        # TODO: node-models and vivado_stitch_ifnames is a metadata prop of that isolated model

    def percent_ready(self, data: IsoSimLogDataByLayer) -> dict[str, float]:
        """Calculate how many percent of the time the layer was ready for input data.
        Return indexed by layer name."""
        # TODO: Implement
        return dict.fromkeys(data, 0)

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Run isolated layer simulations."""
        # Run the simulation
        sim = IsolatedSimulation(
            model,
            SimulationType.NODE_BASED_ISOLATED,
            self.fpgapart,
            self.clk_ns,
            self.functional_sim,
        )
        data: IsoSimLogDataByLayer = sim.simulate()

        # Check if data looks good
        log.info("Checking validity of received simulation data...")
        start = time.time()
        self.sanity_check_logged_data(data)
        log.info(f"Validity check took {time.time() - start} seconds.")

        # Calculate upper bounds
        log.info("Estimating upper bounds...")
        start = time.time()
        in_fifo_upper_bound = self.calculate_upper_bounds(data)
        log.info(f"Estimation took {time.time() - start} seconds.")

        # Write into report file
        upper_bounds_file = self.output_dir / "report" / "estimate_upper_fifo_bound.json"
        upper_bounds_file.write_text(json.dumps(in_fifo_upper_bound, indent=4))
        log.info(f"Wrote results to: {upper_bounds_file}")

        # Save data into dataframe
        # NOTE: We actually have to swap the order here: We recorded the _incoming_ FIFO sizes
        # However the connected simulation stores the depths on the layers before it, so
        # essentially _outgoing_ FIFO sizes.

        # NOTE: For this mapping to work, ordering has to be kept correctly in each step:
        # 1. Mapping node.inputs to vivado_stitch_ifnames metadata prop (CreateStitchedIP)
        # 2. Mapping IO shapes to ifnames from before (simulation_builder.py)
        # 3. Mapping stream_descrs to M/S_AXIS_CONTROL array (C++ simulation creation)
        # 4. Writing the data to json. Order of S_AXIS_CONTROL -> order in which JSON gets written
        #       IMPORTANT: Use nlohmann::ordered_json to keep the insertion order!
        # 5. Reading the JSON into python (python dicts are ordered since 3.7)
        #       According to docs, the Python JSON module also keeps order
        # 6. Syncing node.inputs to order of s_axi_... streams read from the JSON.
        edited_bounds = {}

        # Fill edited_bounds with empty values
        for node in model.graph.node:
            suc = model.find_direct_successors(node)
            if suc is None:
                edited_bounds[node.name] = [-1]
            else:
                edited_bounds[node.name] = [-1] * len(suc)

        # For every node check its predecessors.
        # Find the index/tensor that connects the predecessor and the current one
        # Use that index to retrieve the fifo depth between them and save it
        def get_index(a: Any, values: Any) -> int | None:
            for i, val in enumerate(values):
                if val == a:
                    return i
            return None

        for node in model.graph.node:
            # Rely on the fact that find_direct_predecessors gives the streams in-order
            predecessors = model.find_direct_predecessors(node)
            if predecessors is None:
                continue
            for predecessor in predecessors:
                # Find out which m_axis stream of the predecessor leads to node
                for producer_idx, pre_out in enumerate(predecessor.output):
                    if pre_out in node.input:
                        consumer_idx = get_index(pre_out, node.input)
                        if consumer_idx is None:
                            raise FINNInternalError(
                                f"Could not find index of "
                                f"{predecessor.name}'s output and "
                                f"{node.name}'s input: {pre_out}. "
                                f"Index in predecessor.output is "
                                f"{producer_idx}"
                            )
                        # TODO: Switch to array instead of dict?
                        # We have to conver the string-key (s_axi_...) into the index of the dict
                        key = list(in_fifo_upper_bound[node.name].keys())[consumer_idx]
                        # TODO: Tests
                        edited_bounds[predecessor.name][producer_idx] = in_fifo_upper_bound[
                            node.name
                        ][
                            key
                        ]  # noqa
                        log.info(
                            f"Incoming FIFO {node.name}[{key}/{consumer_idx}] "
                            f"-> outgoing FIFO {predecessor.name}[{producer_idx}]"
                        )

        # Prepare the data
        df_data = {
            "onnx_index": [],
            "node": [],
            "stream": [],
            "out_fifo_upper_bound": [],
            "input_ready_percent": [],
        }
        for layer, layerdata in edited_bounds.items():
            for idx in range(len(layerdata)):
                df_data["onnx_index"].append([n.name for n in model.graph.node].index(layer))
                df_data["node"].append(layer)
                df_data["stream"].append(idx)
                df_data["out_fifo_upper_bound"].append(layerdata[idx])
                # TODO: Remove input_ready_percent?
                # df_data["input_ready_percent"].append(self.percent_ready(data)[layer])
                df_data["input_ready_percent"].append(0.0)

        # Create the DF
        self.fifo_data = pd.DataFrame(df_data)
        log.info("First few entries of collected data:")
        log.info(str(self.fifo_data))

        # Save in dataframe and model
        model = store_fifo_data(
            model,
            self.fifo_data,
            self.default_fifo_data_path,
            delete_existing=True,
            store_html=True,
        )

        # TODO: Integrate data into the layer parallel simulation
        return model, False

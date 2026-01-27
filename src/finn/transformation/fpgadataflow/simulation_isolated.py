"""Simulating layers on their own to observe their behaviour."""
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from rich.console import Console
from typing import Literal, TypeAlias

from finn.transformation.fpgadataflow.simulation import Simulation
from finn.transformation.fpgadataflow.simulation_build import SimulationType
from finn.transformation.fpgadataflow.simulation_controller import SimulationController
from finn.util.exception import FINNInternalError
from finn.util.logging import DisabledLoggingConsole, log


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
        self.console.log("Started simulation controller")

    def postprocess_logs(
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
        with self.console.status(f"Running simulation on every node. Log directory: {self.logdir}"):
            with ThreadPoolExecutor(len(self.binaries)) as tpe:
                for binary in self.binaries:
                    futures.append(tpe.submit(self._run_binary, binary))
            tpe.shutdown(wait=True)
        self._cleanup_sockets()

        # Read data
        data: dict[str, self.IsolatedSimLogData] = {}
        invalid = []
        for i, future in enumerate(futures):
            data[self.names[i]] = future.result()
            if data[self.names[i]] is None:
                invalid.append(self.names[i])
        if len(invalid) > 0:
            raise FINNInternalError(
                f"Lost connection / malformed response from nodes: {', '.join(invalid)}"
            )
        return data

    def _run_binary(self, binary: Path) -> IsolatedSimLogData | None:
        """Run simulation. Returning None if connection is lost."""
        process_index = self.binaries.index(binary)
        with (
            self.logdir / f"{process_index}_log_isolated_{self.names[process_index]}_python.txt"
        ).open("w+") as logfile:
            # Initialize
            logfile.write("Initializing simulation.\n")
            proc_idx = self._start_process(binary, process_index)
            response = self._send_and_receive(proc_idx, "start", {})
            if response is None:
                logfile.write("Client disconnected / No answer received to start command!\n")
                return None
            logfile.write(f"Start response: {response}\n")

            if response is None:
                logfile.write("Failed to start simulation: No response\n")
                return None

            # Main loop
            logfile.write("Beginning main loop\n")
            logfile.write(
                "totalCycles,inputCyclesDone,inputCyclesTarget,"
                "outputCyclesDone,outputCyclesTarget\n"
            )
            logfile.flush()

            while True:
                time.sleep(self.poll_interval)
                logfile.write("Sending status request\n")
                response = self._send_and_receive(proc_idx, "status", {})
                if response is None:
                    self.console.log(f"Empty response from {proc_idx} at {binary.parent}")
                    logfile.write("Empty response. Returning.\n")
                    return None
                state = response["state"]
                if state == "done":
                    self.console.log(f"{process_index} is done and postprocessing data.")
                    return self.postprocess_logs(binary.parent)

                # TODO: Order seems wrong
                logfile.write(
                    f"{response['totalCycles']}, "
                    f"{response['inputCyclesDone']}, "
                    f"{response['inputCyclesTarget']}, "
                    f"{response['outputCyclesDone']}, "
                    f"{response['outputCyclesTarget']}\n"
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
        with DisabledLoggingConsole() as console:
            controller = NodeIsolatedSimulationController(
                len(self.binaries), names, list(self.binaries.values()), console, 0.1, False
            )
            return controller.run()


class RunLayerIsolatedSimulation(Transformation):
    """Run a layer isolated simulation and calculate some information for a
    later layer parallel simulation."""

    def __init__(self, fpgapart: str, clk_ns: float, functional_sim: bool) -> None:
        """Run isolated layer simulations."""
        super().__init__()
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns
        self.functional_sim = functional_sim

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
            for cycle_data in data[layer]["ready"]:
                if cycle_data["inputCyclesDone"] > int(
                    cycle_data["inputCyclesTarget"] / 2
                ) and _any_ready(cycle_data):
                    break
                for stream_name in results[layer].keys():
                    # TODO: Currently on the C++ side we multiply the
                    # TODO: target cycles by 2, to get two samples
                    # TODO: We keep track of ready signals until we see
                    # TODO: the first ready after half of all cycles were seen.
                    # TODO: This might change in the future
                    if cycle_data["inputCyclesTarget"] % 2 != 0:
                        raise FINNInternalError(
                            f"An 'inputCyclesTarget' of layer {layer} seems "
                            f"to not be an even number. Currently, we double "
                            f"the target simulation cycles for every layer "
                            f"on the C++ side. This error may point towards "
                            f"a change on the C++ side, which may cause the "
                            f"need to update this function accordingly!"
                        )
                    results[layer][stream_name] += int(cycle_data[stream_name] == 0)

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
        ...                 "inputCyclesTarget": 10, "s_axi0_ready": 1}],
        ...         "valid": [{"totalCycles": 10, "outputCyclesDone": 5,
        ...                 "outputCyclesTarget": 10, "m_axi0_valid": 1}]
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
        for layer, ldata in data.items():
            cycle_data = ldata["ready"] + ldata["valid"]
            lengths: set[int] = {len(cycle.keys()) for cycle in cycle_data}
            if len(lengths) != 1:
                raise FINNInternalError(
                    f"Simulation log data inconsistent for layer "
                    f"{layer}. Differing number of fields per cycle."
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

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Run isolated layer simulations."""
        sim = IsolatedSimulation(
            model,
            SimulationType.NODE_BASED_ISOLATED,
            self.fpgapart,
            self.clk_ns,
            self.functional_sim,
        )
        data: IsoSimLogDataByLayer = sim.simulate()
        self.sanity_check_logged_data(data)
        in_fifo_upper_bound = self.calculate_upper_bounds(data)
        formatted_upper_bounds = "\n\t".join(
            [f"{name}: {in_fifo_upper_bound[name]}" for name in in_fifo_upper_bound.keys()]
        )
        log.info("Upper bounds: \n" + formatted_upper_bounds)

        raise NotImplementedError()
        # TODO: Integrate data into the layer parallel simulation
        return model, False

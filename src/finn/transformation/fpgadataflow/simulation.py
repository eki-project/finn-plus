"""Manage FINN simulation variants."""

import finn_xsi.adapter as finnxsi
import json
import math
import numpy as np
import onnx
import os
import psutil
import shlex
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from enum import Enum
from onnx import NodeProto, TensorProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from random import Random
from subprocess import CalledProcessError
from typing import TYPE_CHECKING, Any, cast

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.simulation_controller import NodeConnectedSimulationController
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import launch_process_helper, make_build_dir
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import DisabledLoggingConsole, ThreadsafeProgressDisplay, log

if TYPE_CHECKING:
    from collections.abc import Sequence


class SimulationType(str, Enum):
    # Individual node simulations connected by IPC
    NODE_BASED_CONNECTED = "NODE_BASED_CONNECTED"

    # Individual node simulations, isolated. E.g. for analysis purposes
    NODE_BASED_ISOLATED = "NODE_BASED_ISOLATED"

    # Legacy method (deprecated)
    COMPLETE_DESIGN = "COMPLETE_DESIGN"


class SimulationBuilder:
    """Build simulations in FINN."""

    def __init__(self, model: ModelWrapper, fpgapart: str, clk_ns: float) -> None:
        """Create a new simulation instance."""
        self.model = model
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns
        self.progress_bar = ThreadsafeProgressDisplay([], [], [])

    def _isolated_node_model(self, by_node: int | str | NodeProto) -> ModelWrapper:
        """Return a modelwrapper that has only the specified node.

        Args:
            by_node: If int, used as the index of the specified node. If string, assumed to be
                        the name of the node.

        Returns:
            ModelWrapper: The isolated-node modelwrapper.
        """
        # Find the node
        index = 0
        if type(by_node) is int:
            if by_node < 0 or by_node >= len(self.model.graph.node):
                raise FINNInternalError(
                    f"Cannot isolate node index {by_node}. Model has"
                    f"{len(self.model.graph.node)} nodes."
                )
            index = by_node
        elif type(by_node) is str:
            node_name = self.model.get_node_from_name(by_node)
            if node_name is None:
                raise FINNInternalError(f"Cannot isolate node {by_node}. No such node found.")
            index = [n.name for n in self.model.graph.node].index(cast("str", node_name))
        elif type(by_node) is NodeProto:
            try:
                index = self.model.graph.node.index(by_node)
            except Exception as e:
                raise FINNInternalError(f"Node {by_node.name} not found in the model.") from e
        else:
            raise FINNInternalError(
                f"Cannot find node to isolate: {by_node}. Specify either "
                f"the index (int), node name (str) or the object itself "
                f"(NodeProto)."
            )

        # Copy model to modify
        node_model = deepcopy(self.model)

        # Remove any other node
        # TODO: Refactor this following section
        for i, node in enumerate(self.model.graph.node):
            if i != index:
                node_model.graph.node.remove(node)
        target_op = getCustomOp(node_model.graph.node[0])
        if not isinstance(target_op, HWCustomOp):
            raise FINNInternalError(
                f"Node {node_model.graph.node[0].name} is not a HWCustomOp, cannot "
                f"isolate for simulation."
            )
        inp = onnx.helper.make_tensor_value_info(
            "inp", TensorProto.FLOAT, cast("Sequence[int]", target_op.get_folded_input_shape())
        )
        inp_dummy_out = onnx.helper.make_tensor_value_info(  # noqa
            "inp_dummy_out",
            TensorProto.FLOAT,
            cast("Sequence[int]", target_op.get_folded_input_shape()),
        )
        outp = onnx.helper.make_tensor_value_info(  # noqa
            "outp", TensorProto.FLOAT, cast("Sequence[int]", target_op.get_normal_output_shape())
        )
        outp_dummy_out = onnx.helper.make_tensor_value_info(
            "outp_dummy_out",
            TensorProto.FLOAT,
            cast("Sequence[int]", target_op.get_normal_output_shape()),
        )
        input_dummy_node = onnx.helper.make_node(
            "RemoveDataPath_rtl",
            inputs=["inp"],
            outputs=["inp_dummy_out"],
            domain="finn.custom_op.fpgadataflow.rtl",
            backend="fpgadataflow",
            folded_shape=target_op.get_folded_input_shape(),
            normal_shape=target_op.get_normal_input_shape(),
            dataType=target_op.get_input_datatype().name,
            name=node_model.graph.node[0].name + "_input_dummy",
        )
        output_dummy_node = onnx.helper.make_node(
            "RemoveDataPath_rtl",
            inputs=["outp"],
            outputs=["outp_dummy_out"],
            domain="finn.custom_op.fpgadataflow.rtl",
            backend="fpgadataflow",
            folded_shape=target_op.get_folded_output_shape(),
            normal_shape=target_op.get_normal_output_shape(),
            dataType=target_op.get_output_datatype().name,
            name=node_model.graph.node[0].name + "_output_dummy",
        )

        node_model.graph.node.insert(0, input_dummy_node)
        node_model.graph.node.append(output_dummy_node)

        # Remove old io
        for _ in range(len(node_model.graph.node[1].input)):
            node_model.graph.node[1].input.pop()
        for _ in range(len(node_model.graph.node[1].output)):
            node_model.graph.node[1].output.pop()

        # Set new io
        node_model.graph.node[1].input.append("inp_dummy_out")
        node_model.graph.node[1].output.append("outp")

        # Remove graph io
        for _ in range(len(node_model.graph.input)):
            node_model.graph.input.pop()
        for _ in range(len(node_model.graph.output)):
            node_model.graph.output.pop()

        # Set new graph io
        node_model.graph.input.append(inp)
        node_model.graph.output.append(outp_dummy_out)

        return node_model

    def _get_stream_descriptions(self, model: ModelWrapper) -> tuple[str, int, str, int]:
        """Return the stream descriptions for the given model for the C++ sim config header.

        Used by for example _build_single_node_simulation().

        Returns:
            tuple[str, int, str, int]: Strings of stream descriptions together with
                                        their count (in, out)
        """
        # Get IO iterations required
        instream_iters = []
        outstream_iters = []
        for top_inp in model.graph.input:
            iname = top_inp.name
            first_node = model.find_consumer(iname)
            assert first_node is not None, "Failed to find consumer for " + iname
            top_ind = list(first_node.input).index(iname)
            ishape_folded = getCustomOp(first_node).get_folded_input_shape(ind=top_ind)
            instream_iters.append(int(np.prod(ishape_folded[:-1])))
        for top_out in model.graph.output:
            oname = top_out.name
            last_node = model.find_producer(oname)
            assert last_node is not None, "Failed to find producer for " + oname
            top_ind = list(last_node.output).index(oname)
            oshape_folded = getCustomOp(last_node).get_folded_output_shape(ind=top_ind)
            outstream_iters.append(int(np.prod(oshape_folded[:-1])))
        interface_names = model.get_metadata_prop("vivado_stitch_ifnames")
        if interface_names is None:
            raise FINNInternalError(
                f"{model}: Could not find stitched-IP interface names. "
                f"Did you run IP Stitching first?"
            )

        # TODO: Copied from rtlsim_exec_cppxsi. Remove eval().
        interface_names = eval(interface_names)
        if "aximm" in interface_names.keys() and interface_names["aximm"] != []:
            raise FINNInternalError(
                f"{model}: CPP XSI Sim does not know how to handle full "
                f"AXI MM interfaces: {interface_names['aximm']}"
            )
        instream_names = [x[0] for x in interface_names["s_axis"]]
        outstream_names = [x[0] for x in interface_names["m_axis"]]

        # Format stream descriptions
        def _format_descr_name(s: str) -> str:
            for old, new in [("[", ""), ("]", ""), ("(", "{"), (")", "}"), ("'", '"')]:
                s = s.replace(old, new)
            return s

        # TODO: Change this since we don't have throttling
        instream_descrs = [
            (instream_names[i], instream_iters[i], instream_iters[i])
            for i in range(len(instream_names))
        ]
        instream_descrs_str = _format_descr_name(str(instream_descrs))

        outstream_descrs = [
            (outstream_names[i], outstream_iters[i], outstream_iters[i])
            for i in range(len(outstream_names))
        ]
        outstream_descrs_str = _format_descr_name(str(outstream_descrs))
        return instream_descrs_str, len(instream_names), outstream_descrs_str, len(outstream_names)

    def _create_sim_so(
        self,
        model: ModelWrapper,
        top_module_name: str,
        vivado_stitched_proj: Path,
        build_dir: Path | None,
        debug: bool,
    ) -> tuple[Path, Path]:
        """Create a new RTLSim .so file. If one exists already it is used.

        Returns:
            tuple[Path, Path]: Return sim_base and sim_rel.
        """
        rtlsim_so_str = model.get_metadata_prop("rtlsim_so")
        if (rtlsim_so_str is None) or not Path(rtlsim_so_str).exists():
            all_verilog_srcs = (
                (Path(vivado_stitched_proj) / "all_verilog_srcs.txt").read_text().split()
            )
            sim_dir = (
                make_build_dir(f"rtlsim_{model.graph.node[0].name}_")
                if build_dir is None
                else build_dir
            )
            sim_base, sim_rel = finnxsi.compile_sim_obj(
                top_module_name, all_verilog_srcs, str(sim_dir), debug=debug
            )
            rtlsim_so = Path(sim_base) / Path(sim_rel)
            model.set_metadata_prop("rtlsim_so", str(rtlsim_so))
        else:
            sim_base, sim_rel = cast("str", rtlsim_so_str.split("xsim.dir"))
            sim_rel = "xsim.dir" + sim_rel
        return Path(sim_base), Path(sim_rel)

    def _compile_simulation(self, sim_base: Path, silent: bool = False) -> Path:
        """Compile an existing RTLSIM directory. Requires _create_sim_so to be run before. Expects
        rtlsim_config.hpp to be templated already.

        Returns:
            Path: Path to the executable shell script to run the binary
        """
        finnxsi_dir = os.environ["FINN_XSI"]
        # Running CMake first
        cmake_call = f"{sys.executable} -m cmake -S {finnxsi_dir} -B {sim_base}"
        log.info(f"Running cmake on RTLSIM Wrapper in {sim_base}")
        try:
            launch_process_helper(
                shlex.split(cmake_call),
                cwd=finnxsi_dir,
                print_stdout=not silent,
                print_stderr=not silent,
                proc_env=os.environ.copy(),
            )
        except CalledProcessError as e:
            raise FINNInternalError(f"Failed to run cmake in {sim_base}") from e
        self.progress_bar.update("CMake")

        # Calling make to actually build the simulation
        makefile = Path(sim_base) / "Makefile"
        if not makefile.exists():
            raise FINNInternalError(f"Failed to create Makefile in {sim_base}!")
        try:
            launch_process_helper(
                ["make"],
                proc_env=os.environ.copy(),
                cwd=sim_base,
                print_stdout=not silent,
                print_stderr=not silent,
            )
        except CalledProcessError as e:
            raise FINNInternalError(f"Failed to create executable in {sim_base}!") from e

        # TODO: Fix name for general rtlsim
        simulation_executable = Path(sim_base) / "LayerSimulationBackend"
        if not simulation_executable.exists():
            raise FINNInternalError(f"Make call in {sim_base} failed!")
        self.progress_bar.update("Make")
        return simulation_executable

    def _template_rtlsim_config(
        self,
        model: ModelWrapper,
        sim_base: Path,
        node_name: str,
        previous_node_name: str | None,
        node_index: int,
        total_nodes: int,
        timeout_cycles: int,
        top_module_name: str,
        trace_file: str | None,
    ) -> Path:
        """Template finn_xsi/finn_xsi/rtlsim_config.hpp.template with the correct values and
        return the templated file.
        """
        finnxsi_dir = os.environ["FINN_XSI"]
        # Prepare the C++ driver config template
        (
            instream_descrs_str,
            len_instreams,
            outstream_descrs_str,
            len_outstreams,
        ) = self._get_stream_descriptions(model)
        template_dict = {
            "TIMEOUT_CYCLES": timeout_cycles,
            # name of the top-level HDL module
            "TOP_MODULE_NAME": top_module_name,
            # top-level AXI stream descriptors
            "ISTREAM_DESC": instream_descrs_str,
            "ISTREAM_LEN": len_instreams,
            "OSTREAM_DESC": outstream_descrs_str,
            "OSTREAM_LEN": len_outstreams,
            # control tracing and trace filename
            "TRACE_FILE": "std::nullopt" if trace_file is None else f'"{trace_file}"',
            # sim kernel .so to use (depends on Vivado version)
            "SIMKERNEL_SO": finnxsi.get_simkernel_so(),
            # log file for xsi (not the sim driver)
            "XSIM_LOG_FILE": '"xsi.log"',
            # Node name in case of single-node simulation
            "NODE_NAME": node_name,
            # Previous node name (for single node simulation)
            "PREVIOUS_NODE_NAME": (
                "std::nullopt" if previous_node_name is None else f'"{previous_node_name}"'
            ),
            "NODE_INDEX": node_index,
            "TOTAL_NODES": total_nodes,
        }

        fifosim_config_fname = Path(finnxsi_dir) / "rtlsim_config.hpp.template"
        fsim_config = fifosim_config_fname.read_text()
        for key, val in template_dict.items():
            fsim_config = fsim_config.replace(f"@{key}@", str(val))

        # Write the config to the simulation directory
        rtlsim_config = Path(sim_base) / "rtlsim_config.hpp"
        rtlsim_config.write_text(fsim_config)
        return rtlsim_config

    def build_single_node_simulation(
        self,
        node_name: str,
        node_model: ModelWrapper,
        node_index: int,
        total_nodes: int,
        previous_node_name: str | None,
        build_dir: Path | None,
        timeout_cycles: int = 0,
        silent: bool = False,
    ) -> Path:
        """Build the simulation binary for a single node.

        This can be used both by the connected node-by-node sim and the isolated node sim.

        Much of this is from the rtlsim_exec.py in core/

        Args:
            node_name: Despite the fact that we receive an isolated node model, we can still
                    manually pass a node name. This is useful to give unique names (e.g. for IPC)
            node_model: The single node ModelWrapper to build the simulation from.
            node_index: The index of the simulated node. Used to determine whether a node shares IO
                        with successors or predecessors.
            total_nodes: The total number of nodes in the complete design.
            previous_node_name: Required by the connected simulation. In the simulation binary this
                                is used to get access to the correct shared memory segment between
                                this node and the previous one.
            build_dir: If given, use this directory for building the simulation. Otherwise one is
                        created from the nodes name.
            timeout_cycles: Number of cycles until simulation timeout. When set to 0 (default), no
                            timeout is given.
            silent: If True, silences the Cmake and make output (including stderr)

        Returns:
            Path: The path to the simulation binary (shell script).
        """
        # TODO: Check if something is an output node instead of checking the node index
        # TODO: Requires changes in the C++ code as well

        # Sanity checks (2 Dummy nodes + 1 target node)
        if len(node_model.graph.node) != 3:
            raise FINNUserError(
                "Cannot create single-node simulation for a model with more than "
                "1 node. Make sure to pass the ModelWrapper containing only"
                "the relevant node."
            )

        # Check that the relevant data exists
        wrapper_filename = node_model.get_metadata_prop("wrapper_filename")
        if wrapper_filename is None or not Path(wrapper_filename).exists():
            raise FINNUserError(
                f"Call CreateStitchedIP prior to building "
                f"the simulation for {node_name}. "
                f"wrapper_filename is set to {wrapper_filename}!"
            )

        vivado_stitched_proj = node_model.get_metadata_prop("vivado_stitch_proj")
        if vivado_stitched_proj is None or not Path(vivado_stitched_proj).exists():
            raise FINNUserError(
                f"Call CreateStitchedIP prior to building "
                f"the simulation for {node_name}. (vivado_stitch_proj not set!)"
            )

        trace_file = cast("str | None", node_model.get_metadata_prop("rtlsim_trace"))
        debug = not (trace_file is None or trace_file == "")

        # Get the module name and path
        top_module_file = Path(wrapper_filename).resolve().absolute()
        top_module_name = top_module_file.name.strip(".v")

        # Build the simulation .so and save it in the "rtlsim_so" metadata prop
        sim_base, _ = self._create_sim_so(
            node_model, top_module_name, Path(vivado_stitched_proj), build_dir, debug
        )

        # Fill out the simulation config header
        _ = self._template_rtlsim_config(
            node_model,
            sim_base,
            node_name,
            previous_node_name,
            node_index,
            total_nodes,
            timeout_cycles,
            top_module_name,
            trace_file,
        )

        # Building the whole simulation
        return self._compile_simulation(sim_base, silent).absolute()

    def _get_randomized_names(self, model: ModelWrapper, suffix_length: int = 5) -> dict[int, str]:
        """Add a randomized suffix to every name in the model. Used to avoid interference with
        previous or parallel running IPC simulations."""
        rand = Random()
        rand.seed()
        return {
            i: model.graph.node[i].name
            + "".join(
                rand.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(suffix_length)
            )
            for i in range(len(model.graph.node))
        }

    def _build_simulation_node_connected(
        self, workers: int, with_live_display: bool, functional_sim: bool
    ) -> dict[int, Path]:
        """Build all nodes in the model in parallel, as isolated simulations, ready for usage in
        an IPC connected simulation chain.

        Args:
            workers: Number of parallel workers to use.
            with_live_display: If True, display the building progress in a rich progress bar.

        Returns:
            Dict of executables that start the simulation of the given nodes,
            indexed by the node-index. These are in their respective FINN_TMP
            directories.
        """

        def _build(
            node_name: str,
            node_index: int,
            total_nodes: int,
            prev_node_name: str | None,
            build_dir: Path,
        ) -> Any:
            nodemodel = self._isolated_node_model(node_index)
            nodemodel = nodemodel.transform(InferShapes())
            nodemodel = nodemodel.transform(PrepareIP(self.fpgapart, self.clk_ns))
            nodemodel = nodemodel.transform(
                CreateStitchedIP(self.fpgapart, self.clk_ns, functional_simulation=functional_sim)
            )
            self.progress_bar.update("StitchedIP")
            return self.build_single_node_simulation(
                node_name,
                nodemodel,
                node_index,
                total_nodes,
                prev_node_name,
                build_dir,
                silent=with_live_display,
            )

        # Create randomized names to avoid clashes with old IPC shared memory
        randomized_names = self._get_randomized_names(self.model)

        # TODO: Currently ignores workers argument
        total_nodes = len(self.model.graph.node)
        futures: dict[int, Future] = {}

        # Build sims in parallel
        synth_workers = max(
            1, cast("int", (psutil.virtual_memory().free / 1024 / 1024 / 1024) // 20)
        )  # 20GB per synthesis
        if not functional_sim:
            # When not having to do synthesis, the build is not memory bottlenecked and
            # can be executed as parallel as possible
            synth_workers = int(os.environ.get("NUM_DEFAULT_WORKERS", len(self.model.graph.node)))

        # Build (stitched IP, cmake, make) all sims in parallel and return paths to
        # the compiled executables
        with DisabledLoggingConsole(), self.progress_bar if with_live_display else nullcontext():
            self.progress_bar.progress.console.log(
                f"Building simulations using {int(synth_workers)} workers.."
            )
            with ThreadPoolExecutor(max_workers=synth_workers) as pool:
                for i in range(total_nodes):
                    futures[i] = pool.submit(
                        _build,
                        randomized_names[i],
                        i,
                        total_nodes,
                        randomized_names[i - 1] if i >= 1 else None,  # type: ignore
                        Path(make_build_dir(f"rtlsim_{randomized_names[i]}_")),
                    )
                return {i: future.result() for i, future in futures.items()}

    def build_simulation(
        self, simtype: SimulationType, workers: int, with_live_display: bool, functional_sim: bool
    ) -> dict[int, Path]:
        """Build a simulation of the given type, return the resulting executable.

        Args:
            simtype: Simulation type to build.
            workers: Number of workers to use in parallel.
                Normally set by the Simulation() class automatically.
            with_live_display: If True, display a live progress-bar.
            functional_sim: If True, use functional simulation (faster but takes some time to build)
        """
        match simtype:
            case SimulationType.NODE_BASED_CONNECTED:
                node_count = len(self.model.graph.node)
                self.progress_bar = ThreadsafeProgressDisplay(
                    ["StitchedIP", "CMake", "Make"],
                    [node_count] * 3,
                    [
                        "[bold blue](1)[/bold blue] Creating stitched IPs",
                        "[bold blue](2)[/bold blue] Configuring project with CMake",
                        "[bold blue](3)[/bold blue] Building simulation binaries",
                    ],
                )
                return self._build_simulation_node_connected(
                    workers, with_live_display, functional_sim
                )
            case SimulationType.NODE_BASED_ISOLATED:
                raise NotImplementedError()
            case SimulationType.COMPLETE_DESIGN:
                raise FINNUserError(f"Simulation method {simtype} is deprecated!")


class Simulation:
    """Manage simulation (runs) in FINN.

    IMPORTANT: If the modelwrapper was somehow changed, create a NEW simulation object!
    """

    def __init__(
        self,
        model: ModelWrapper,
        fpgapart: str,
        clk_ns: float,
        functional_sim: bool,
        workers: int | None = None,
    ) -> None:
        """Create a new simulation instance. If workers is None, NUM_DEFAULT_WORKERS are used."""
        self.model = model
        self.workers = int(os.environ["NUM_DEFAULT_WORKERS"]) if workers is None else workers
        self.functional_sim = functional_sim
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns
        # TODO: Caching of existing simulations
        # Prepare the model for simulation
        with DisabledLoggingConsole() as console:  # noqa
            with console.status("Preparing model for the simulation step..."):
                self._prepare_model()
        self.builder = SimulationBuilder(self.model, fpgapart, clk_ns)

        sys.stdout = sys.stdout.console
        sys.stderr = sys.stderr.console

        self.binaries = self.builder.build_simulation(
            SimulationType.NODE_BASED_CONNECTED,
            self.workers,
            with_live_display=True,
            functional_sim=self.functional_sim,
        )

    def _prepare_model(self) -> None:
        """Execute some preparation transformations on the model."""
        self.model = self.model.transform(InsertDWC())
        self.model = self.model.transform(SpecializeLayers(self.fpgapart))
        self.model = self.model.transform(GiveUniqueNodeNames())
        self.model = self.model.transform(GiveReadableTensorNames())
        self.model = self.model.transform(PrepareIP(self.fpgapart, self.clk_ns))
        self.model = self.model.transform(HLSSynthIP())

    def simulate_node_connected(
        self, depth: int | list[list[int]] | None = None, max_cycles: int | None = None
    ) -> tuple[dict[int, dict[str, list[int]]], bool]:
        """Simulate the given number of samples for every layer. Layers are completely isolated
        and simulated in parallel. Simulation data is returned as a dict (by node name as index).
        """
        names = [node.name for node in self.model.graph.node]
        initial_depth = [[depth]] * len(self.binaries) if isinstance(depth, int) else depth

        # Run simulation
        start = time.time()
        output_json = Path(make_build_dir("simulation_results_")) / "simulation_data.json"
        with DisabledLoggingConsole() as console:
            controller = NodeConnectedSimulationController(
                len(self.binaries), names, list(self.binaries.values()), console, 0.1, False
            )
            controller.run(initial_depth, output_json, max_cycles)
        end = time.time()
        log.info(f"Simulation took {end - start} seconds!")

        # Load the merged data from JSON
        merged_data = json.loads(output_json.read_text())

        # Return the collected data indexed by node index
        data = {}
        for i, sim_entry in enumerate(merged_data["simulations"]):
            data[i] = {
                "name": sim_entry["name"],
                "fifo_utilization": sim_entry["fifo_utilization"],
                "fifo_depth": sim_entry["fifo_depth"],
                "cycles": sim_entry["cycles"],
                "samples": sim_entry["samples"],
                "intervals": sim_entry["intervals"],
            }
        json.dump(data, output_json.open("w"), indent=4)
        return data, merged_data.get("timeout_occurred", False)


# TODO: Just a test transformation. Will be integrated properly later
class RunLayerParallelSimulation(Transformation):  # noqa
    def __init__(
        self,
        fpgapart: str,
        clk_ns: float,
        cfg: DataflowBuildConfig,
        max_qsrl_depth: int = 256,
        vivado_ram_style: str = "auto",
        quality_of_results: str = "default",
    ) -> None:
        super().__init__()
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns
        self.cfg = cfg
        self.max_qsrl_depth = max_qsrl_depth
        self.vivado_ram_style = vivado_ram_style
        self.quality_of_results = quality_of_results

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        sim = Simulation(model, self.fpgapart, self.clk_ns, self.cfg.functional_simulation)
        model = sim.model  # TODO:clean up

        initial_fifo_depths, _ = sim.simulate_node_connected()

        fifo_depths = []  # Each entry is a list of fifo sizes for that node
        for val in initial_fifo_depths.values():
            fifo_depths.append([v + 1 for v in val["fifo_utilization"]])

        # Max cycles for any simulation
        sim_cycles = max([val["cycles"] for val in initial_fifo_depths.values()])

        bit_widths = []
        for i in range(len(fifo_depths)):
            bit_widths.append([])
            hw_node = getCustomOp(model.graph.node[i])
            if isinstance(hw_node, HWCustomOp):
                for j in range(len(fifo_depths[i])):
                    bit_widths[i].append(hw_node.get_outstream_width(j))
            else:
                raise FINNInternalError("Non-HW node found in dataflow graph during simulation")

        needs_minimization = []
        for i in range(len(fifo_depths)):
            needs_minimization.append([True] * len(fifo_depths[i]))
        for i in range(len(fifo_depths)):
            for j in range(len(fifo_depths[i])):
                # Check if we can reduce the fifo size

                used_size = fifo_depths[i][j]
                bw = bit_widths[i][j]

                needs_minimization[i][j] = self.needs_minimization(used_size, bw)

        # Preserve original baseline depths for testing (deep copy)
        original_fifo_depths = [row[:] for row in fifo_depths]

        # Minimize FIFO depths using binary search over BRAM block counts
        for i in range(len(fifo_depths)):
            for j in range(len(fifo_depths[i])):
                if not needs_minimization[i][j]:
                    continue

                minimized_depth = self._minimize_fifo_depth(
                    i,
                    j,
                    fifo_depths,
                    original_fifo_depths,
                    bit_widths,
                    initial_fifo_depths,
                    sim,
                    sim_cycles,
                )
                fifo_depths[i][j] = minimized_depth

        print("Final FIFO depths:")
        for i in range(len(fifo_depths)):
            print(f"{i}: {fifo_depths[i]}")
            log.info(f"{i}: {fifo_depths[i]}")

        return model, False

    def _check_performance(self, new_data: dict, initial_fifo_depths: dict) -> bool:
        """Check if performance has degraded compared to baseline.

        Args:
            new_data: Simulation results to check
            initial_fifo_depths: Baseline performance data

        Returns:
            True if performance degraded, False otherwise
        """
        for k, v in new_data.items():
            for idx in range(len(v["intervals"])):
                if v["intervals"][idx] > initial_fifo_depths[k]["intervals"][idx]:
                    return True
        return False

    def _test_depth(
        self,
        test_depth: int,
        node_idx: int,
        fifo_idx: int,
        baseline_depths: list,
        initial_fifo_depths: dict,
        sim: Simulation,
        sim_cycles: float,
    ) -> tuple[bool, bool]:
        """Test a specific FIFO depth.

        Args:
            test_depth: Depth to test
            node_idx: Node index
            fifo_idx: FIFO index within node
            baseline_depths: Original baseline FIFO depths (unchanged during minimization)
            initial_fifo_depths: Baseline performance data
            sim: Simulation controller
            sim_cycles: Maximum simulation cycles

        Returns:
            Tuple of (success, timeout) where success means depth works without degradation
        """
        test_depths = [row[:] for row in baseline_depths]  # Deep copy from baseline
        test_depths[node_idx][fifo_idx] = test_depth

        new_data, timeout = sim.simulate_node_connected(
            test_depths, max_cycles=math.ceil(sim_cycles * 1.1)
        )

        if timeout:
            return False, True

        performance_degraded = self._check_performance(new_data, initial_fifo_depths)
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
        current_depths: list,
        baseline_depths: list,
        bit_widths: list,
        initial_fifo_depths: dict,
        sim: Simulation,
        sim_cycles: int,
    ) -> int:
        """Minimize a single FIFO depth using binary search.

        Args:
            node_idx: Node index
            fifo_idx: FIFO index within node
            current_depths: Current working FIFO depth configuration
            (may have already-minimized values)
            baseline_depths: Original baseline FIFO depths (unchanged during minimization)
            bit_widths: Bitwidths for all FIFOs
            initial_fifo_depths: Baseline performance data
            sim: Simulation controller
            sim_cycles: Maximum simulation cycles

        Returns:
            Minimized FIFO depth
        """
        original_size = baseline_depths[node_idx][fifo_idx]
        bw = bit_widths[node_idx][fifo_idx]

        print(f"Minimizing Node {node_idx} FIFO {fifo_idx}: original depth {original_size}")

        # If FIFO depth of 2 works, we dont need FIFOs at all, because AXI buffers some values
        success, timeout = self._test_depth(
            2, node_idx, fifo_idx, baseline_depths, initial_fifo_depths, sim, sim_cycles
        )
        if success:
            return 2

        if original_size <= self.max_qsrl_depth:
            upper_luts = calculate_srl16e_luts(original_size, bw)
            # Smallest depth that is reasonable is 32 (Fits into bw LUTRAMs)
            lower_luts = calculate_srl16e_luts(32, bw)

            # Binary search if there's room to search
            if upper_luts > lower_luts:
                best_working_depth = self._binary_search_srl_depth(
                    node_idx,
                    fifo_idx,
                    baseline_depths,
                    bw,
                    initial_fifo_depths,
                    sim,
                    sim_cycles,
                    lower_luts=lower_luts,
                    upper_luts=upper_luts,
                )
                return best_working_depth
            return original_size

        # Try FIFO depth of 256 next (fits into LUTRAM)
        success, timeout = self._test_depth(
            self.max_qsrl_depth,
            node_idx,
            fifo_idx,
            baseline_depths,
            initial_fifo_depths,
            sim,
            sim_cycles,
        )
        if success:
            upper_luts = calculate_srl16e_luts(original_size, bw)
            # Smallest depth that is reasonable is 32 (Fits into bw LUTRAMs)
            lower_luts = calculate_srl16e_luts(32, bw)

            # Binary search if there's room to search
            if upper_luts > lower_luts:
                best_working_depth = self._binary_search_srl_depth(
                    node_idx,
                    fifo_idx,
                    baseline_depths,
                    bw,
                    initial_fifo_depths,
                    sim,
                    sim_cycles,
                    lower_luts=lower_luts,
                    upper_luts=upper_luts,
                )
                return best_working_depth
            return self.max_qsrl_depth

        # We know 256 doesn't work, so we have to use BRAMs
        # Try one BRAM block less than current
        upper_blocks = calculate_bram_blocks(original_size, bw)
        # Get all valid block counts in the range
        valid_blocks = self._get_valid_block_counts(1, upper_blocks - 1, bw)
        if not valid_blocks:
            # No valid configurations exist
            return original_size
        # Test the maximum valid block count first (smallest depth)
        max_valid_blocks = valid_blocks[-1]
        _, max_d = calculate_bram_depth_range(max_valid_blocks, bw)

        success, timeout = self._test_depth(
            max_d, node_idx, fifo_idx, baseline_depths, initial_fifo_depths, sim, sim_cycles
        )

        if timeout or not success:
            return original_size

        best_working_depth = max_d

        # Binary search if there's room to search and multiple valid configs
        if len(valid_blocks) > 1:
            best_working_depth = self._exponential_binary_search_depth(
                node_idx,
                fifo_idx,
                baseline_depths,
                bw,
                initial_fifo_depths,
                sim,
                sim_cycles,
                valid_blocks=valid_blocks,
            )

        return best_working_depth

    def _exponential_binary_search_depth(
        self,
        node_idx: int,
        fifo_idx: int,
        baseline_depths: list,
        bitwidth: int,
        initial_fifo_depths: dict,
        sim: Simulation,
        sim_cycles: float,
        valid_blocks: list[int],
    ) -> int:
        """Perform exponential + binary search over valid block configurations.

        Uses exponential search to quickly find the range, then binary search within it.
        This is more efficient when smaller block counts are more likely.
        Only searches over pre-validated block counts.

        Args:
            node_idx: Node index
            fifo_idx: FIFO index within node
            baseline_depths: Original baseline FIFO depths (unchanged during minimization)
            bitwidth: Data bitwidth
            initial_fifo_depths: Baseline performance data
            sim: Simulation controller
            sim_cycles: Maximum simulation cycles
            valid_blocks: Sorted list of valid block counts to search over

        Returns:
            Best working depth found
        """
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
                max_d, node_idx, fifo_idx, baseline_depths, initial_fifo_depths, sim, sim_cycles
            )

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
                max_d, node_idx, fifo_idx, baseline_depths, initial_fifo_depths, sim, sim_cycles
            )

            if success:
                # This depth works, try smaller (lower indices)
                best_working_depth = max_d
                upper_idx = mid_idx
            else:
                # This depth doesn't work, need larger (higher indices)
                lower_idx = mid_idx + 1

        return best_working_depth

    def _binary_search_srl_depth(
        self,
        node_idx: int,
        fifo_idx: int,
        baseline_depths: list,
        bitwidth: int,
        initial_fifo_depths: dict,
        sim: Simulation,
        sim_cycles: float,
        lower_luts: int,
        upper_luts: int,
    ) -> int:
        """Perform binary search to find minimal working FIFO depth in LUTRAM range.

        Args:
            node_idx: Node index
            fifo_idx: FIFO index within node
            baseline_depths: Original baseline FIFO depths (unchanged during minimization)
            bitwidth: Data bitwidth
            initial_fifo_depths: Baseline performance data
            sim: Simulation controller
            sim_cycles: Maximum simulation cycles
            lower_luts: Lower bound for LUT count
            upper_luts: Upper bound for LUT count (known to work)

        Returns:
            Best working depth found
        """
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
                max_d, node_idx, fifo_idx, baseline_depths, initial_fifo_depths, sim, sim_cycles
            )

            if success:
                # This depth works, try smaller
                best_working_depth = max_d
                upper_luts = mid_luts
            else:
                # This depth doesn't work, need larger
                lower_luts = mid_luts + 1

        return best_working_depth

    def needs_minimization(self, fifo_depth: int, bitwidth: int) -> bool:
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

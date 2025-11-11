"""Manage FINN simulation variants."""
import finn_xsi.adapter as finnxsi
import multiprocessing
import numpy as np
import onnx
import os
import psutil
import shlex
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from onnx import NodeProto, TensorProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from random import Random
from subprocess import CalledProcessError
from typing import Any, Sequence, cast

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import get_vivado_root, launch_process_helper, make_build_dir
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log


class Simulation:
    """Manage simulations in FINN."""

    def __init__(self, model: ModelWrapper, fpgapart: str, clk_ns: float) -> None:
        """Create a new simulation instance."""
        self.model = model
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns

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
            instream_iters.append(np.prod(ishape_folded[:-1]))
        for top_out in model.graph.output:
            oname = top_out.name
            last_node = model.find_producer(oname)
            assert last_node is not None, "Failed to find producer for " + oname
            top_ind = list(last_node.output).index(oname)
            oshape_folded = getCustomOp(last_node).get_folded_output_shape(ind=top_ind)
            outstream_iters.append(np.prod(oshape_folded[:-1]))
        interface_names = model.get_metadata_prop("vivado_stitch_ifnames")
        if interface_names is None:
            raise FINNUserError(
                f"{model}: Could not find stitched-IP interface names. "
                f"Did you run IP Stitching first?"
            )

        # TODO: Copied from rtlsim_exec_cppxsi. Remove eval().
        interface_names = eval(interface_names)
        if "aximm" in interface_names.keys() and interface_names["aximm"] != []:
            raise FINNUserError(
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

    def _compile_simulation(self, sim_base: Path) -> Path:
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
                print_stdout=True,
                proc_env=os.environ.copy(),
            )
        except CalledProcessError as e:
            raise FINNUserError(f"Failed to run cmake in {sim_base}") from e

        # Calling make to actually build the simulation
        makefile = Path(sim_base) / "Makefile"
        if not makefile.exists():
            raise FINNUserError(f"Failed to create Makefile in {sim_base}!")
        try:
            launch_process_helper(["make"], proc_env=os.environ.copy(), cwd=sim_base)
        except CalledProcessError as e:
            raise FINNUserError(f"Failed to create executable in {sim_base}!") from e

        # TODO: Fix name for general rtlsim
        simulation_executable = Path(sim_base) / "LayerSimulationBackend"
        if not simulation_executable.exists():
            raise FINNUserError(f"Make call in {sim_base} failed!")

        # Prepare the script to run the simulation
        # (important to specify LD_LIBRARY_PATH here for XSI to work correctly)
        runsim = Path(sim_base) / "run_fifosim.sh"
        ld_library_path = get_vivado_root() + "/lib/lnx64.o"
        runsim.write_text(
            f"LD_LIBRARY_PATH={ld_library_path}:$LD_LIBRARY_PATH {simulation_executable} --depth 2"
        )
        return runsim

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

    def _build_single_node_simulation(
        self,
        node_name: str,
        node_model: ModelWrapper,
        node_index: int,
        total_nodes: int,
        previous_node_name: str | None,
        build_dir: Path | None,
        timeout_cycles: int = 0,
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
        return self._compile_simulation(sim_base).absolute()

    def run_sim_node_parallel_isolated(self, inputs: int) -> None:  # noqa: ARG002
        """Simulate the given number of inputs for every layer. Layers are completely isolated
        and simulated in parallel.
        """
        for i, node in enumerate(self.model.graph.node):
            print(f"{i}: {node.name}")

        def _build_simulation(
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
                CreateStitchedIP(self.fpgapart, self.clk_ns, functional_simulation=True)
            )
            return self._build_single_node_simulation(
                node_name, nodemodel, node_index, total_nodes, prev_node_name, build_dir
            )

        def _run_simulation(binary: Path, cpu: int | None) -> None:
            command = ""
            if cpu is not None:
                command += f"taskset --cpu-list {cpu} "
                # TODO: numactl
            command += f"bash {binary}"
            subprocess.run(
                shlex.split(command), stdout=sys.stdout, stderr=sys.stderr, cwd=binary.parent
            )

        # Create randomized names to avoid clashing with old IPC shared memory segments.
        rand = Random()
        rand.seed()
        randomized_names = {
            i: self.model.graph.node[i].name
            + "".join(rand.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(5))
            for i in range(len(self.model.graph.node))
        }

        # Build simulations in parallel
        # TODO: Change to info when done
        log.warning("BUILDING NODE SIMULATIONS")
        workers = int(os.environ["NUM_DEFAULT_WORKERS"])
        total_nodes = len(self.model.graph.node)
        futures: dict[int, Future] = {}
        binaries: dict[int, Path] = {}
        self.model = self.model.transform(InsertDWC())
        self.model = self.model.transform(SpecializeLayers(self.fpgapart))
        self.model = self.model.transform(GiveUniqueNodeNames())
        self.model = self.model.transform(GiveReadableTensorNames())
        self.model = self.model.transform(PrepareIP(self.fpgapart, self.clk_ns))
        self.model = self.model.transform(HLSSynthIP())
        synth_workers = max(
            1, cast("int", (psutil.virtual_memory().free / 1024 / 1024 / 1024) // 16)
        )  # 16GB per synthesis
        with ThreadPoolExecutor(max_workers=synth_workers) as pool:
            for i in range(total_nodes):
                futures[i] = pool.submit(
                    _build_simulation,
                    randomized_names[i],
                    i,
                    total_nodes,
                    randomized_names[i - 1] if i >= 1 else None,  # type: ignore
                    Path(make_build_dir(f"rtlsim_{randomized_names[i]}_")),
                )
            pool.shutdown(wait=True)
            for i, future in futures.items():
                binaries[i] = future.result()

        # Create a script to build and run the entire simulation again
        run_simulation = make_build_dir("run_simulation")
        run_all_simulations = Path(run_simulation) / "run.sh"
        build_all_simulations = Path(run_simulation) / "build.sh"
        log.info(f"Storing run-all-simulations script in {run_simulation}")
        with (run_all_simulations).open("w+") as f:
            f.write("#!/bin/bash\n")
            f.write('echo "Running simulation"\n')
            for binary in binaries.values():
                f.write(f"bash {binary} &\n")
            f.write("wait\n")
        with build_all_simulations.open("w+") as f:
            f.write("#!/bin/bash\n")
            for binary in binaries.values():
                # Build each binary new. Done in parallel in the background
                f.write(f"{{ cd {binary.parent};cmake . && make; }} &\n")
            f.write("wait\n")

        # TODO: Change to info when done
        log.warning("RUNNING NODE SIMULATIONS")
        # TODO: Might be unnecessary. Remove later
        sys.stdout = sys.stdout.console
        sys.stderr = sys.stderr.console
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, binary in binaries.items():
                print(
                    f"Submitting thread for running simulation {i} / {total_nodes} "
                    f"({self.model.graph.node[i].name})"
                )
                # TODO: If more processes than CPU cores, group processes to their adjacent nodes
                pool.submit(_run_simulation, binary, i % multiprocessing.cpu_count())
            pool.shutdown(wait=True)

    def run_sim_node_parallel_connected(self, inputs: int) -> Any:
        """Simulate a whole model, with all layers simulated in parallel."""
        # TODO: Enable control through either Python or a seperate C++ driver
        raise NotImplementedError()

    def run_sim_complete(self) -> Any:
        raise NotImplementedError()

    def run_sim_single_node(self, node: Any) -> Any:
        raise NotImplementedError()


# TODO: Just a test transformation. Will be integrated properly later
class RunLayerParallelSimulation(Transformation):  # noqa
    def __init__(self, fpgapart: str, clk_ns: float) -> None:  # noqa
        super().__init__()
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        sim = Simulation(model, self.fpgapart, self.clk_ns)
        sim.run_sim_node_parallel_isolated(1)
        return model, False

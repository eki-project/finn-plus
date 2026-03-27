"""Build FINN Simulations."""

import finn_xsi.adapter as finnxsi
import numpy as np
import onnx
import os
import psutil
import shlex
import subprocess
import sys
import time
from ast import literal_eval
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from onnx import NodeProto, TensorProto, ValueInfoProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name
from subprocess import CalledProcessError
from typing import TYPE_CHECKING, Any, cast

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import launch_process_helper, make_build_dir
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log

if TYPE_CHECKING:
    from collections.abc import Sequence


# TODO: Fix that BuildSimulation has to return binaries for either SimulationType
# TODO: Just store the directory instead - since we build all targets anyways


class SimulationType(str, Enum):
    """Type of simulation."""

    # Individual node simulations connected by IPC
    NODE_BASED_CONNECTED = "NODE_BASED_CONNECTED"

    # Individual node simulations, isolated. E.g. for analysis purposes
    NODE_BASED_ISOLATED = "NODE_BASED_ISOLATED"


class SimulationBuilder:
    """Build simulations in FINN."""

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

        target_op = getCustomOp(self.model.graph.node[index])
        if not isinstance(target_op, HWCustomOp):
            raise FINNInternalError(
                f"Node {target_op.name} is not a HWCustomOp, cannot isolate for simulation."
            )

        initializers: list[TensorProto] = []
        value_info_protos: list[ValueInfoProto] = []
        inputs_graph: list[ValueInfoProto] = []
        inputs_node: list[ValueInfoProto] = []
        outputs_graph: list[ValueInfoProto] = []
        outputs_node: list[ValueInfoProto] = []
        nodes_graph: list[NodeProto] = []

        preds_list: list | None = self.model.find_direct_predecessors(self.model.graph.node[index])
        succs_list: list | None = self.model.find_direct_successors(self.model.graph.node[index])

        num_preds = len(preds_list) if preds_list is not None else 0
        num_succs = len(succs_list) if succs_list is not None else 0

        input_node = False
        output_node = False

        # Set correct input/output count for input and output nodes, since they have no pred/succ.
        if num_preds == 0:
            inputs = self.model.graph.input
            ret = get_by_name(
                inputs, self.model.graph.node[index].input[0]
            )  # Check that node is graph input
            if ret is not None:
                num_preds = 1
                input_node = True
        if num_succs == 0:
            outputs = self.model.graph.output
            ret = get_by_name(
                outputs, self.model.graph.node[index].output[0]
            )  # Check that node is graph output
            if ret is not None:
                num_succs = 1
                output_node = True

        num_inputs = len(self.model.graph.node[index].input)
        num_outputs = len(self.model.graph.node[index].output)

        if num_outputs != num_succs:
            raise FINNInternalError(
                f"Node {self.model.graph.node[index].name} has {num_outputs} outputs but "
                f"{num_succs} successor nodes. This is not supported for isolation."
            )

        initializer_inputs_list = [
            self.model.graph.node[index].input[i]
            for i in range(num_inputs)
            if self.model.get_initializer(self.model.graph.node[index].input[i]) is not None
        ]

        # Handle initializers of nodes
        initializer_inputs = []
        for init in initializer_inputs_list:
            ret = self.model.get_initializer(init, return_dtype=True)
            info = self.model.get_tensor_valueinfo(init)
            if ret is None or info is None:
                raise FINNInternalError(
                    f"Failed to get initializer for {init} "
                    f"while isolating node {self.model.graph.node[index].name}."
                )
            vals, dtype = cast("tuple[np.ndarray, int]", ret)
            initializers.append(onnx.helper.make_tensor(info.name, dtype, vals.shape, vals))
            val_info = onnx.helper.make_tensor_value_info(info.name, dtype, vals.shape)
            value_info_protos.append(val_info)
            initializer_inputs.append(val_info)

        pred_count = 0
        for i in range(num_inputs):
            if self.model.graph.node[index].input[i] in initializer_inputs_list:
                continue  # This input is handled as an initializer, skip
            pred_count += 1
            info = self.model.get_tensor_valueinfo(self.model.graph.node[index].input[i])
            if info is None:
                raise FINNInternalError(
                    f"Failed to get value info for {self.model.graph.node[index].input[i]} "
                    f"while isolating node {self.model.graph.node[index].name}."
                )
            # Setup new input tensors
            new_input_info = onnx.helper.make_tensor_value_info(
                info.name,
                TensorProto.FLOAT,
                cast("Sequence[int]", target_op.get_normal_input_shape(i)),
            )
            new_input_dummy_info = onnx.helper.make_tensor_value_info(
                info.name + "_dummy",
                TensorProto.FLOAT,
                cast("Sequence[int]", target_op.get_normal_input_shape(i)),
            )
            # value_info_protos.append(new_input_info)
            value_info_protos.append(new_input_dummy_info)
            inputs_graph.append(new_input_info)
            inputs_node.append(new_input_dummy_info)

            # Create new dummy node to remove data path for input i
            dummy_node = onnx.helper.make_node(
                "RemoveDataPath_rtl",
                inputs=[new_input_info.name],
                outputs=[new_input_dummy_info.name],
                domain="finn.custom_op.fpgadataflow.rtl",
                backend="fpgadataflow",
                folded_shape=target_op.get_folded_input_shape(i),
                normal_shape=target_op.get_normal_input_shape(i),
                dataType=target_op.get_input_datatype(i).name,
                name=self.model.graph.node[index].name + f"_input_dummy_{i}",
            )

            nodes_graph.append(dummy_node)
        inputs_node.extend(initializer_inputs)
        if pred_count != num_preds:
            raise FINNInternalError(
                f"Node {self.model.graph.node[index].name} has {num_preds} pred. nodes but only "
                f"{pred_count} inputs have been handled."
            )
        for i in range(num_succs):
            info = self.model.get_tensor_valueinfo(self.model.graph.node[index].output[i])
            if info is None:
                raise FINNInternalError(
                    f"Failed to get value info for {self.model.graph.node[index].output[i]} "
                    f"while isolating node {self.model.graph.node[index].name}."
                )
            # Setup new input tensors
            new_output_info = onnx.helper.make_tensor_value_info(
                info.name,
                TensorProto.FLOAT,
                cast("Sequence[int]", target_op.get_normal_output_shape(i)),
            )
            new_output_dummy_info = onnx.helper.make_tensor_value_info(
                info.name + "_dummy",
                TensorProto.FLOAT,
                cast("Sequence[int]", target_op.get_normal_output_shape(i)),
            )
            # value_info_protos.append(new_output_info)
            value_info_protos.append(new_output_dummy_info)
            outputs_graph.append(new_output_info)
            outputs_node.append(new_output_dummy_info)

            # Create new dummy node to remove data path for output i
            dummy_node = onnx.helper.make_node(
                "RemoveDataPath_rtl",
                inputs=[new_output_dummy_info.name],
                outputs=[new_output_info.name],
                domain="finn.custom_op.fpgadataflow.rtl",
                backend="fpgadataflow",
                folded_shape=target_op.get_folded_output_shape(i),
                normal_shape=target_op.get_normal_output_shape(i),
                dataType=target_op.get_output_datatype(i).name,
                name=self.model.graph.node[index].name + f"_output_dummy_{i}",
            )

            nodes_graph.append(dummy_node)

        target_op_attrs = target_op.get_nodeattr_types()
        params = {}
        for attr in target_op_attrs.keys():
            attr_val = target_op.get_nodeattr(attr)
            if (
                (isinstance(attr_val, np.ndarray) and attr_val.size == 0)
                or attr_val == ""
                or attr_val == []
            ):  # Empty value, skip
                continue
            params[attr] = target_op.get_nodeattr(attr)
        new_node = onnx.helper.make_node(
            self.model.graph.node[index].op_type,
            inputs=[inp.name for inp in inputs_node],
            outputs=[outp.name for outp in outputs_node],
            domain=self.model.graph.node[index].domain,
            name=self.model.graph.node[index].name,
            **params,
        )
        nodes_graph.append(new_node)

        graph = onnx.helper.make_graph(
            nodes_graph,
            f"isolated_node_graph_{self.model.graph.node[index].name}",
            inputs_graph,
            outputs_graph,
            initializer=initializers,
            value_info=value_info_protos,
        )

        node_model = onnx.helper.make_model(graph)
        node_model = ModelWrapper(node_model)

        node_model.set_metadata_prop("predecessors", str([pred.name for pred in inputs_graph]))
        node_model.set_metadata_prop("successors", str([succ.name for succ in outputs_graph]))
        node_model.set_metadata_prop("input_node", str(input_node).lower())
        node_model.set_metadata_prop("output_node", str(output_node).lower())

        # node_model.save(f"isolated_node_model_{self.model.graph.node[index].name}.onnx")

        return node_model

    def _get_stream_descriptions(self, model: ModelWrapper) -> tuple[str, str]:
        """Return the stream descriptions for the given model for the C++ sim config header.

        Used by for example _build_single_node_simulation().

        Returns:
            tuple[str, str]: Strings of stream descriptions
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
        interface_names = literal_eval(interface_names)
        if "aximm" in interface_names.keys() and interface_names["aximm"] != []:
            raise FINNInternalError(
                f"{model}: CPP XSI Sim does not know how to handle full "
                f"AXI MM interfaces: {interface_names['aximm']}"
            )
        instream_names = [x[0] for x in interface_names["s_axis"]]
        outstream_names = [x[0] for x in interface_names["m_axis"]]

        # Convert to the format required by the C++ sim config header
        # (initializer list of pairs of name and iters)
        def _format_descr_name(s: list[tuple[str, int]]) -> str:
            return ", ".join([f'StreamDescriptor{{"{name}", {iters}}}' for name, iters in s])

        instream_descrs = [
            (instream_names[i], instream_iters[i]) for i in range(len(instream_names))
        ]
        instream_descrs_str = _format_descr_name(instream_descrs)

        outstream_descrs = [
            (outstream_names[i], outstream_iters[i]) for i in range(len(outstream_names))
        ]
        outstream_descrs_str = _format_descr_name(outstream_descrs)
        return instream_descrs_str, outstream_descrs_str

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

    def _compile_simulation(self, sim_base: Path, silent: bool = True) -> Path:
        """Compile an existing RTLSIM directory. Requires _create_sim_so to be run before. Expects
        rtlsim_config.hpp to be templated already.

        Returns:
            Path: Path to the executable shell script to run the binary
        """
        # Determine executable name
        compile_targets = ["LayerSimulationBackend", "IsolatedSimulationBackend"]
        if all((Path(sim_base) / execname).exists() for execname in compile_targets):
            # Simulation was already compiled, we can return early
            return Path(sim_base)

        # Check where FINNXSI is
        finnxsi_dir = os.environ["FINN_XSI"]

        # Running CMake first
        cmake_call = f"{sys.executable} -m cmake -S {finnxsi_dir} -B {sim_base}"
        log.debug(f"Running cmake on RTLSIM Wrapper in {sim_base}")
        try:
            launch_process_helper(
                shlex.split(cmake_call),
                cwd=finnxsi_dir,
                print_stdout=silent,
                print_stderr=silent,
                proc_env=os.environ.copy(),
            )
        except CalledProcessError as e:
            raise FINNInternalError(f"Failed to run cmake in {sim_base}") from e

        # Calling make to actually build the simulation
        makefile = Path(sim_base) / "Makefile"
        if not makefile.exists():
            raise FINNInternalError(f"Failed to create Makefile in {sim_base}!")
        try:
            launch_process_helper(
                ["make"],
                proc_env=os.environ.copy(),
                cwd=sim_base,
                print_stdout=silent,
                print_stderr=silent,
            )
        except CalledProcessError as e:
            raise FINNInternalError(f"Failed to create executable in {sim_base}!") from e

        errors = []
        for target in compile_targets:
            simulation_executable = Path(sim_base) / target
            if not simulation_executable.exists():
                errors.append(
                    f"Simulation compile target {target} was not created. "
                    f"Check {sim_base} to run make manually."
                )
        if len(errors) > 0:
            raise FINNInternalError("Error compiling simulations: \n" + "\n\t".join(errors))
        return sim_base

    def _template_rtlsim_config(
        self,
        model: ModelWrapper,
        sim_base: Path,
        input_interface_names: list[str] | None,
        output_interface_names: list[str] | None,
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
            outstream_descrs_str,
        ) = self._get_stream_descriptions(model)
        template_dict = {
            "TIMEOUT_CYCLES": timeout_cycles,
            # name of the top-level HDL module
            "TOP_MODULE_NAME": top_module_name,
            # top-level AXI stream descriptors
            "ISTREAM_DESC": instream_descrs_str,
            "OSTREAM_DESC": outstream_descrs_str,
            # control tracing and trace filename
            "TRACE_FILE": "std::nullopt" if trace_file is None else f'"{trace_file}"',
            # sim kernel .so to use (depends on Vivado version)
            "SIMKERNEL_SO": finnxsi.get_simkernel_so(),
            # log file for xsi (not the sim driver)
            "XSIM_LOG_FILE": '"xsi.log"',
            "INPUT_INTERFACE_NAMES": ",".join(['"' + name + '"' for name in input_interface_names])
            if input_interface_names is not None
            else "",
            "OUTPUT_INTERFACE_NAMES": ",".join(
                ['"' + name + '"' for name in output_interface_names]
            )
            if output_interface_names is not None
            else "",
            "INPUT_INTERFACE_COUNT": len(input_interface_names)
            if input_interface_names is not None
            else 0,
            "OUTPUT_INTERFACE_COUNT": len(output_interface_names)
            if output_interface_names is not None
            else 0,
            "NODE_INDEX": node_index,
            "TOTAL_NODES": total_nodes,
            "IS_INPUT_NODE": model.get_metadata_prop("input_node"),
            "IS_OUTPUT_NODE": model.get_metadata_prop("output_node"),
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
        node_model: ModelWrapper,
        node_index: int,
        total_nodes: int,
        input_interface_names: list[str] | None,
        output_interface_names: list[str] | None,
        build_dir: Path | None,
        timeout_cycles: int = 0,
        silent: bool = False,
    ) -> Path:
        """Build the simulation binary for a single node.

        This can be used both by the connected node-by-node sim and the isolated node sim.

        Much of this is from the rtlsim_exec.py in core/

        Args:
            node_model: The single node ModelWrapper to build the simulation from.
            node_index: The index of the simulated node. Used to determine whether a node shares IO
                        with successors or predecessors.
            total_nodes: The total number of nodes in the complete design.
            input_interface_names: Names of input interfaces for IPC communication. Required by the
                                connected simulation to access the correct shared memory segment
                                between this node and its predecessors.
            output_interface_names: Names of output interfaces for IPC communication. Required by
                                the connected simulation to access the correct shared memory segment
                                between this node and its successors.
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

        # Check that the relevant data exists
        wrapper_filename = node_model.get_metadata_prop("wrapper_filename")
        if wrapper_filename is None or not Path(wrapper_filename).exists():
            raise FINNUserError(
                f"Call CreateStitchedIP prior to building "
                f"the simulation for {self.model.graph.node[node_index].name}. "
                f"wrapper_filename is set to {wrapper_filename}!"
            )

        vivado_stitched_proj = node_model.get_metadata_prop("vivado_stitch_proj")
        if vivado_stitched_proj is None or not Path(vivado_stitched_proj).exists():
            raise FINNUserError(
                f"Call CreateStitchedIP prior to building "
                f"the simulation for {self.model.graph.node[node_index].name}."
                "(vivado_stitch_proj not set!)"
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
            input_interface_names,
            output_interface_names,
            node_index,
            total_nodes,
            timeout_cycles,
            top_module_name,
            trace_file,
        )

        # Building the whole simulation
        return self._compile_simulation(sim_base, silent=silent).absolute()

    def _build_simulations_parallel(
        self, with_live_display: bool, functional_sim: bool
    ) -> dict[int, Path]:
        """Build all nodes in the model in parallel, as isolated simulations, ready for usage in
        an IPC connected simulation chain.

        Args:
            workers: Number of parallel workers to use.
            with_live_display: If True, display the building progress in a rich progress bar.
            functional_sim: Use a functional simulation (faster but takes time to build)
            sim_type: Type of simulation

        Returns:
            Dict of executables that start the simulation of the given nodes,
            indexed by the node-index. These are in their respective FINN_TMP
            directories.
        """
        log.info(f"Building simulation binaries for {len(self.model.graph.node)} layers.")

        def _build(
            node_index: int,
            total_nodes: int,
            build_dir: Path,
        ) -> Any:
            nodemodel = self._isolated_node_model(node_index)
            nodemodel = nodemodel.transform(InferShapes())
            nodemodel = nodemodel.transform(PrepareIP(self.fpgapart, self.clk_ns))
            nodemodel = nodemodel.transform(
                CreateStitchedIP(self.fpgapart, self.clk_ns, functional_simulation=functional_sim)
            )
            input_interface_names = nodemodel.get_metadata_prop("predecessors")
            if input_interface_names is not None:
                input_interface_names = literal_eval(input_interface_names)
            output_interface_names = nodemodel.get_metadata_prop("successors")
            if output_interface_names is not None:
                output_interface_names = literal_eval(output_interface_names)
            return self.build_single_node_simulation(
                nodemodel,
                node_index,
                total_nodes,
                input_interface_names,
                output_interface_names,
                build_dir,
                silent=with_live_display,
            )

        total_nodes = len(self.model.graph.node)
        log.info(f"[BuildSimulation] Preparing to build {total_nodes} nodes for the simulation.")
        futures: dict[int, Future] = {}
        built_nodes = 0

        # Progress display callback
        def _callback_progress(name: str) -> Callable:
            nonlocal total_nodes, built_nodes

            def _f(f: Future) -> None:
                nonlocal total_nodes, built_nodes
                built_nodes += 1
                log.info(
                    f"[ [bold green]"
                    f"{int(100.0 * float(built_nodes) / float(total_nodes))}%[/bold green]"
                    f" ] {name}",
                    extra={"markup": True, "highlighter": None},
                )
                # Unpack result once so that the pool fails immediately, instead of waiting for
                # all futures to be completed.
                f.result()

            return _f

        # Build sims in parallel
        synth_workers = max(
            1, cast("int", (psutil.virtual_memory().free / 1024 / 1024 / 1024) // 10)
        )  # 10GB per synthesis
        if not functional_sim:
            # When not having to do synthesis, the build is not memory bottlenecked and
            # can be executed as parallel as possible
            synth_workers = int(os.environ.get("NUM_DEFAULT_WORKERS", len(self.model.graph.node)))

        # Build (stitched IP, cmake, make) all sims in parallel and return paths to
        # the compiled executables
        log.info("[BuildSimulation] Starting the build process.")
        with ThreadPoolExecutor(max_workers=synth_workers) as pool:
            for i in range(total_nodes):
                node_name = self.model.graph.node[i].name
                futures[i] = pool.submit(
                    _build,
                    i,
                    total_nodes - 1,
                    Path(make_build_dir(f"rtlsim_{node_name}_")),
                )
                futures[i].add_done_callback(_callback_progress(node_name))
            pool.shutdown(wait=True)

        # Check if all binaries were compiled successfully
        binaries = {i: future.result() for i, future in futures.items()}
        not_found_binaries = []
        for i, binary in binaries.items():
            if binary is None:
                not_found_binaries.append(i)
        if len(not_found_binaries) > 0:
            raise FINNInternalError(
                "Building simulations failed. "
                "Failed simulation binaries: " + ", ".join(not_found_binaries)
            )
        return binaries

    def build_simulation(self, with_live_display: bool, functional_sim: bool) -> dict[int, Path]:
        """Build a simulation of the given type, return the path to the executable directory
        (indexed by the corresponding node index in the graph).

        Args:
            simtype: Simulation type to build.
            workers: Number of workers to use in parallel.
                Normally set by the Simulation() class automatically.
            with_live_display: If True, display a live progress-bar.
            functional_sim: If True, use functional simulation (faster but takes some time to build)
        """
        return self._build_simulations_parallel(with_live_display, functional_sim)


class BuildSimulation(Transformation):
    """Build a simulation of the given type for the model.
    Puts the model into a prepared state (changes the graph).
    If simulation binaries already exist, enter their directory and only re-compile."""

    def __init__(
        self,
        fpgapart: str,
        clk_ns: float,
        functional_sim: bool,
    ) -> None:
        """Create a new BuildSimulation transform."""
        self.functional_sim = functional_sim
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Build / compile the model. Modifies the model."""
        self.model = model

        # Check if we already have stitched IPs and built simulations. If so, rerun only cmake/make
        needs_rebuild = True
        sim_binaries = self.model.get_metadata_prop("simulation_binaries")

        # 1. Check if binary paths are saved in the model
        if sim_binaries is not None:
            sim_binaries = sim_binaries.split("\n")

            # 2. Check that the model size hasn't changed since creating the binaries. Otherwise
            # we should rebuild.
            if len(sim_binaries) != len(self.model.graph.node):
                log.info(
                    f"[BuildSimulation] Found existing binaries, but number ({len(sim_binaries)}) "
                    f"does not match number of nodes in the graph "
                    f"({len(self.model.graph.node)}). Rebuilding..."
                )
            else:
                log.info("Existing simulations found. Re-running only CMake/Make..")
                needs_rebuild = False
        else:
            log.info("[BuildSimulation] No simulation binaries found, building now.")

        # If needed, call the Builder to create the layer simulation binaries.
        # This creates both the isolated and connected binaries in one go.
        if needs_rebuild:
            log.info("[BuildSimulation] Starting model preparation.")
            self._prepare_model()
            self.builder = SimulationBuilder(self.model, self.fpgapart, self.clk_ns)
            sys.stdout = sys.stdout.console  # type: ignore
            self.binaries = self.builder.build_simulation(
                with_live_display=False,
                functional_sim=self.functional_sim,
            )
            self.model.set_metadata_prop(
                "simulation_binaries", "\n".join([str(p) for p in self.binaries.values()])
            )
        else:
            # Run only compilation again, and avoid repeating building of the stitched IPs
            def _compile(binary: Path) -> None:
                result = subprocess.run(
                    "cmake .;make",
                    shell=True,
                    cwd=str(binary),
                    text=True,
                    capture_output=True,
                )
                if result.returncode != 0:
                    raise FINNUserError(f"Failed compilation in {binary}: {result.stderr}")

            # Since we dont need a rebuild, sim_binaries contains the paths to the binaries
            sim_binaries = [Path(p) for p in sim_binaries]
            total = len(sim_binaries)

            # Prepare compiling the binaries again
            done = 0

            def _progress_callback(binary: str | Path) -> Callable:
                nonlocal done, total

                def _f(future: Future) -> None:
                    nonlocal done, total
                    done += 1
                    log.info(
                        f"[ [bold green]{int(100.0 * float(done) / float(total))}%[/bold green] ] "
                        f"Simulation [green italic]{binary}[/green italic] built.",
                        extra={"markup": True, "highlighter": None},
                    )
                    future.result()

                return _f

            # Run the compilation in parallel with the number of workers specified.
            # If not specified, use 8
            compile_start = time.time()
            futures: list[Future] = []
            with ThreadPoolExecutor(int(os.environ.get("NUM_DEFAULT_WORKERS", "8"))) as tpe:
                for binary in sim_binaries:
                    futures.append(tpe.submit(_compile, binary))
                    futures[-1].add_done_callback(_progress_callback(binary.name))
            tpe.shutdown()
            compile_end = time.time()
            log.info(f"Compilation done. Took {compile_end - compile_start} seconds")
        return self.model, False

    def _prepare_model(self) -> None:
        """Execute some preparation transformations on the model."""
        log.info("[BuildSimulation] Inserting DataWidthConverters...")
        self.model = self.model.transform(InsertDWC())
        log.info("[BuildSimulation] Specializing layers...")
        self.model = self.model.transform(SpecializeLayers(self.fpgapart))
        log.info("[BuildSimulation] Assigning unique and readable node and tensor names...")
        self.model = self.model.transform(GiveUniqueNodeNames())
        self.model = self.model.transform(GiveReadableTensorNames())
        log.info("[BuildSimulation] Preparing IPs...")
        self.model = self.model.transform(PrepareIP(self.fpgapart, self.clk_ns))
        log.info("[BuildSimulation] Synthesizing IPs...")
        self.model = self.model.transform(HLSSynthIP())
        log.info("[BuildSimulation] Model preparation done.")

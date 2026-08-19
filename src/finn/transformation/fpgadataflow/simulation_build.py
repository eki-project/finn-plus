"""Build FINN Simulations."""

import contextlib
import math
import numpy as np
import onnx
import os
import psutil
import random
import re
import shlex
import string
import subprocess
import sys
import time
from ast import literal_eval
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from jinja2 import Environment
from onnx import NodeProto, TensorProto, ValueInfoProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, get_by_name
from subprocess import CalledProcessError
from typing import TYPE_CHECKING, Any, cast

import finn.xsi as finnxsi
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import (
    getHWCustomOp,
    launch_process_helper,
    make_build_dir,
    wait_for_dir,
    wait_for_file,
)
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from collections.abc import Sequence


class SimulationType(str, Enum):
    """Type of simulation."""

    # Individual node simulations connected by IPC
    NODE_BASED_CONNECTED = "NODE_BASED_CONNECTED"

    # Individual node simulations, isolated. E.g. for analysis purposes
    NODE_BASED_ISOLATED = "NODE_BASED_ISOLATED"


class SimulationCommMode(str, Enum):
    """Communication backend mode for simulation interconnects."""

    # Current default behavior: all streams use local shared memory.
    SHM = "shm"

    # Hybrid mode: intra-node streams use SHM, border streams may use MPI.
    HYBRID_MPI = "hybrid_mpi"


def _get_local_cores() -> int:
    """Return the number of CPU cores available to this process."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    cpu_count = os.cpu_count()
    return max(1, int(cpu_count) if cpu_count is not None else 1)


def _parse_hosts_with_slots(raw: str | None) -> list[tuple[str, int]]:
    """Parse a "host[:slots],host[:slots],..." string into (host, slots) pairs.

    Hosts without an explicit slot count default to the local core count.
    """
    if raw is None or raw.strip() == "":
        return [("localhost", _get_local_cores())]

    hosts_with_slots: list[tuple[str, int]] = []
    default_slots = _get_local_cores()
    for entry in raw.split(","):
        token = entry.strip()
        if token == "":
            continue
        host = token
        slots = default_slots
        if ":" in token:
            host_part, slot_part = token.split(":", 1)
            host = host_part.strip()
            try:
                parsed_slots = int(slot_part.strip())
                if parsed_slots > 0:
                    slots = parsed_slots
            except ValueError:
                pass
        if host != "":
            hosts_with_slots.append((host, slots))

    if len(hosts_with_slots) == 0:
        return [("localhost", default_slots)]
    return hosts_with_slots


def distribute_mpi_ranks(
    model: ModelWrapper,
) -> tuple[dict[str, int], dict[str, int], dict[str, str]]:
    """Return the hybrid-MPI rank/worker/host maps for the given model, computing and
    storing them in the model's metadata if missing or stale.

    Maps assign each non-FIFO node a global rank, a worker (host) index, and a host
    name, filling each host to its slot capacity before moving to the next one.
    Recomputed whenever the stored maps don't cover every non-FIFO node currently in the
    graph, since graph mutations between the initial computation and later use (e.g.
    BuildSimulation inserting DWCs) can leave a stale map misaligned with the current
    node set.

    For sim_comm_mode other than "hybrid_mpi", this only loads (never computes) whatever
    maps happen to already be stored, returning empty dicts if none are set.

    Returns:
        (rank_map, worker_map, host_map), each keyed by node name.
    """
    rank_map_meta = model.get_metadata_prop("sim_rank_map")
    worker_map_meta = model.get_metadata_prop("sim_worker_map")
    host_map_meta = model.get_metadata_prop("sim_host_map")
    sim_comm_mode_meta = model.get_metadata_prop("sim_comm_mode")
    sim_comm_mode = cast("str", sim_comm_mode_meta) if sim_comm_mode_meta is not None else "shm"

    rank_map = cast(
        "dict[str, int]", literal_eval(rank_map_meta) if rank_map_meta is not None else {}
    )
    worker_map = cast(
        "dict[str, int]", literal_eval(worker_map_meta) if worker_map_meta is not None else {}
    )
    host_map = cast(
        "dict[str, str]", literal_eval(host_map_meta) if host_map_meta is not None else {}
    )

    if sim_comm_mode != SimulationCommMode.HYBRID_MPI.value:
        return rank_map, worker_map, host_map

    # BuildSimulation mutates the graph (e.g. inserts DWCs), so metadata precomputed
    # earlier can become stale/misaligned. Recompute if coverage is incomplete.
    sim_nodes_for_build = [n for n in model.graph.node if "FIFO" not in n.name]
    need_recompute = (len(rank_map) == 0) or (len(worker_map) == 0)
    if not need_recompute:
        for n in sim_nodes_for_build:
            if n.name not in rank_map or n.name not in worker_map:
                need_recompute = True
                break

    if not need_recompute:
        return rank_map, worker_map, host_map

    hosts_with_slots = _parse_hosts_with_slots(model.get_metadata_prop("sim_mpi_hosts"))
    rank_map = {}
    worker_map = {}
    host_map = {}

    # Compute how many simulations per node to distribute evenly across hosts.
    # This is used to determine how many ranks to assign to each host.
    simulations_per_node = math.ceil(len(sim_nodes_for_build) / len(hosts_with_slots))

    for rank, node in enumerate(sim_nodes_for_build):
        worker_id = rank // simulations_per_node
        host = hosts_with_slots[worker_id][0]
        rank_map[node.name] = rank
        worker_map[node.name] = worker_id
        host_map[node.name] = host

    has_border_channels = any(
        worker_map[consumer.name] != worker_map[node.name]
        for node in sim_nodes_for_build
        for outp in node.output
        for consumer in model.find_consumers(outp)
        if consumer.name in worker_map
    )

    model.set_metadata_prop("sim_rank_map", str(rank_map))
    model.set_metadata_prop("sim_worker_map", str(worker_map))
    model.set_metadata_prop("sim_host_map", str(host_map))
    model.set_metadata_prop("sim_has_mpi_border_channels", str(has_border_channels).lower())
    model.set_metadata_prop(
        "sim_mpi_hosts",
        ",".join([f"{host}:{slots}" for host, slots in hosts_with_slots]),
    )
    log.info(
        "[BuildSimulation] Recomputed hybrid MPI rank/worker/host maps "
        "to match the current transformed graph."
    )

    return rank_map, worker_map, host_map


class SimulationBuilder:
    """Build simulations in FINN."""

    def __init__(
        self,
        model: ModelWrapper,
        fpgapart: str,
        clk_ns: float,
        shm_prefix: str,
        performance_sim: bool = False,
    ) -> None:
        """Create a new simulation instance."""
        self.model = model
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns
        self.shm_prefix = shm_prefix
        self.performance_sim = performance_sim

    def _create_existing_initializer_input(
        self,
        inp_name: str,
        target_node: NodeProto,
    ) -> tuple[TensorProto, ValueInfoProto]:
        """Create tensor/valueinfo for an input that already has an initializer."""
        init_ret = self.model.get_initializer(inp_name, return_dtype=True)
        info = self.model.get_tensor_valueinfo(inp_name)
        if init_ret is None or info is None:
            raise FINNInternalError(
                f"Failed to get initializer for {inp_name} while isolating node {target_node.name}."
            )
        vals, dtype = cast("tuple[np.ndarray, int]", init_ret)
        init_tensor = onnx.helper.make_tensor(info.name, dtype, vals.shape, vals)
        val_info = onnx.helper.make_tensor_value_info(info.name, dtype, vals.shape)
        return init_tensor, val_info

    def _create_mlo_dummy_initializer_input(
        self,
        inp_name: str,
        target_node: NodeProto,
    ) -> tuple[TensorProto, ValueInfoProto]:
        """Create dummy initializer tensor/valueinfo for an MLO parameter input."""
        info = self.model.get_tensor_valueinfo(inp_name)
        if info is None:
            raise FINNInternalError(
                f"Failed to get value info for {inp_name} while isolating node {target_node.name}."
            )

        dtype = info.type.tensor_type.elem_type
        if dtype == TensorProto.UNDEFINED:
            dtype = TensorProto.FLOAT
        tdt = self.model.get_tensor_datatype(inp_name)
        tshape = self.model.get_tensor_shape(inp_name)
        if tshape is None:
            raise FINNInternalError(
                f"Failed to get shape for {inp_name} while isolating node {target_node.name}."
            )
        vals = gen_finn_dt_tensor(tdt, tuple(tshape))
        vals = np.sort(vals)

        init_tensor = onnx.helper.make_tensor(info.name, dtype, vals.shape, vals)
        val_info = onnx.helper.make_tensor_value_info(info.name, dtype, vals.shape)
        return init_tensor, val_info

    def _create_dynamic_input_with_dummy(
        self,
        inp_name: str,
        input_index: int,
        target_node: NodeProto,
        target_op: HWCustomOp,
    ) -> tuple[ValueInfoProto, ValueInfoProto, NodeProto]:
        """Create graph input and dummy node for a non-initializer input."""
        info = self.model.get_tensor_valueinfo(inp_name)
        if info is None:
            raise FINNInternalError(
                f"Failed to get value info for {inp_name} while isolating node {target_node.name}."
            )

        new_input_info = onnx.helper.make_tensor_value_info(
            info.name,
            TensorProto.FLOAT,
            cast("Sequence[int]", target_op.get_normal_input_shape(input_index)),
        )
        new_input_dummy_info = onnx.helper.make_tensor_value_info(
            info.name + "_dummy",
            TensorProto.FLOAT,
            cast("Sequence[int]", target_op.get_normal_input_shape(input_index)),
        )

        dummy_node = onnx.helper.make_node(
            "RemoveDataPath_rtl",
            inputs=[new_input_info.name],
            outputs=[new_input_dummy_info.name],
            domain="finn.custom_op.fpgadataflow.rtl",
            backend="fpgadataflow",
            folded_shape=target_op.get_folded_input_shape(input_index),
            normal_shape=target_op.get_normal_input_shape(input_index),
            dataType=target_op.get_input_datatype(input_index).name,
            name=target_node.name + f"_input_dummy_{input_index}",
        )
        return new_input_info, new_input_dummy_info, dummy_node

    def _create_output_with_dummy(
        self,
        out_name: str,
        output_index: int,
        target_node: NodeProto,
        target_op: HWCustomOp,
    ) -> tuple[ValueInfoProto, ValueInfoProto, NodeProto]:
        """Create graph output and dummy node for an output tensor."""
        info = self.model.get_tensor_valueinfo(out_name)
        if info is None:
            raise FINNInternalError(
                f"Failed to get value info for {out_name} while isolating node {target_node.name}."
            )

        new_output_info = onnx.helper.make_tensor_value_info(
            info.name,
            TensorProto.FLOAT,
            cast("Sequence[int]", target_op.get_normal_output_shape(output_index)),
        )
        new_output_dummy_info = onnx.helper.make_tensor_value_info(
            info.name + "_dummy",
            TensorProto.FLOAT,
            cast("Sequence[int]", target_op.get_normal_output_shape(output_index)),
        )

        dummy_node = onnx.helper.make_node(
            "RemoveDataPath_rtl",
            inputs=[new_output_dummy_info.name],
            outputs=[new_output_info.name],
            domain="finn.custom_op.fpgadataflow.rtl",
            backend="fpgadataflow",
            folded_shape=target_op.get_folded_output_shape(output_index),
            normal_shape=target_op.get_normal_output_shape(output_index),
            dataType=target_op.get_output_datatype(output_index).name,
            name=target_node.name + f"_output_dummy_{output_index}",
        )
        return new_output_info, new_output_dummy_info, dummy_node

    def _isolated_node_model(self, by_node: int | str | NodeProto) -> ModelWrapper:
        """Return a modelwrapper that has only the specified node.

        Args:
            by_node: If int, used as the index of the specified node. If string, assumed to be
                        the name of the node.
            performance_sim: Specifies if the simulation is used for FIFO sizing or performance
                        simulation. In the case of a performance simulation, FIFOs are already
                        in the graph and need to be skipped.

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
            node_obj = self.model.get_node_from_name(by_node)
            if node_obj is None:
                raise FINNInternalError(f"Cannot isolate node {by_node}. No such node found.")
            try:
                index = next(
                    i for i, node in enumerate(self.model.graph.node) if node.name == by_node
                )
            except Exception as e:
                raise FINNInternalError(
                    f"Cannot isolate node {by_node}. No such node found."
                ) from e
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

        target_node = self.model.graph.node[index]
        target_op = getHWCustomOp(target_node)

        is_mlo_node = False
        mlo_flag = self.model.get_metadata_prop("is_mlo")
        if mlo_flag is not None and mlo_flag == "1":
            is_mlo_node = True
        mlo_parameter_input_names = []
        if is_mlo_node:
            if mlo_parameter_input_names is None:
                raise FINNInternalError(
                    f"Node {target_node.name} is an MLO node, but no "
                    f"mlo_input_parameter_names metadata found in the model."
                )
            mlo_param = self.model.get_metadata_prop("mlo_input_parameter_names")
            mlo_parameter_input_names = literal_eval(mlo_param) if mlo_param is not None else None
            if (
                mlo_parameter_input_names is None
                or not isinstance(mlo_parameter_input_names, list)
                or not all(isinstance(name, str) for name in mlo_parameter_input_names)
            ):
                raise FINNInternalError(
                    f"mlo_input_parameter_names metadata is not a"
                    f"list of strings: {mlo_parameter_input_names}"
                )

        initializers: list[TensorProto] = []
        value_info_protos: list[ValueInfoProto] = []
        inputs_graph: list[ValueInfoProto] = []
        inputs_node: list[ValueInfoProto] = []
        outputs_graph: list[ValueInfoProto] = []
        outputs_node: list[ValueInfoProto] = []
        nodes_graph: list[NodeProto] = []

        def _find_first_non_fifo_pred(pred: NodeProto) -> NodeProto | None:
            """Return the first non fifo predecessor."""
            if "FIFO" in pred.op_type:
                # Replace FIFOs with their predecessor
                pred_fifo = self.model.find_direct_predecessors(pred)
                if pred_fifo is not None:
                    if len(pred_fifo) > 1:
                        raise FINNInternalError(
                            f"FIFOs are expected to have exactly one predecessor, "
                            f"but found multiple: {pred_fifo}"
                        )
                    if "FIFO" in pred_fifo[0].op_type:
                        return _find_first_non_fifo_pred(pred_fifo[0])
                    return pred_fifo[0]
                if get_by_name(self.model.graph.input, pred.input[0]) is not None:
                    return None  # Reached the end of the graph, return None to indicate graph input
            return pred

        def _find_first_non_fifo_succ(succ: NodeProto) -> NodeProto | None:
            """Return the first non fifo successor."""
            if "FIFO" in succ.op_type:
                # Replace FIFOs with their successor
                succ_fifo = self.model.find_direct_successors(succ)
                if succ_fifo is not None:
                    if len(succ_fifo) > 1:
                        raise FINNInternalError(
                            f"FIFOs are expected to have exactly one successor, "
                            f"but found multiple: {succ_fifo}"
                        )
                    if "FIFO" in succ_fifo[0].op_type:
                        return _find_first_non_fifo_succ(succ_fifo[0])
                    return succ_fifo[0]
                if get_by_name(self.model.graph.output, succ.output[0]) is not None:
                    return (
                        None  # Reached the end of the graph, return None to indicate graph output
                    )
            return succ

        preds_list: list[NodeProto] | None = self.model.find_direct_predecessors(target_node)
        succs_list: list[NodeProto] | None = self.model.find_direct_successors(target_node)

        preds = []
        if preds_list is not None:
            for pred in preds_list:
                preds.append(_find_first_non_fifo_pred(pred))
        succs = []
        if succs_list is not None:
            for succ in succs_list:
                succs.append(_find_first_non_fifo_succ(succ))
        preds_list = [x for x in preds if x is not None]
        succs_list = [x for x in succs if x is not None]

        num_preds = len(preds_list)
        num_succs = len(succs_list)

        input_node = False
        output_node = False

        def _get_first_non_fifo_input(node: NodeProto, inp: str) -> str:
            """Return the first output by name that is not produced by a FIFO.
            If the input is not produced by a FIFO, return the input itself."""
            pred = self.model.find_direct_predecessors(node)
            if pred is None:
                return inp
            for p in pred:
                if inp in p.output and "FIFO" in p.op_type:
                    fifo_input = p.input[0]  # FIFOs are expected to have exactly one input
                    return _get_first_non_fifo_input(p, fifo_input)
            return inp

        def _get_first_non_fifo_output(node: NodeProto, outp: str) -> str:
            """Return the first output by name that is not consumed by a FIFO.
            If the output is not consumed by a FIFO, return the output itself."""
            succ = self.model.find_direct_successors(node)
            if succ is None:
                return outp
            for s in succ:
                if outp in s.input and "FIFO" in s.op_type:
                    fifo_output = s.output[0]  # FIFOs are expected to have exactly one output
                    return _get_first_non_fifo_output(s, fifo_output)
            return outp

        # Set correct input/output count for input and output nodes, since they have no pred/succ.
        if num_preds == 0:
            inputs = self.model.graph.input
            for i in range(len(target_node.input)):
                inp = _get_first_non_fifo_input(target_node, target_node.input[i])
                ret = get_by_name(inputs, inp)  # Check that node is graph input
                if ret is not None and (not is_mlo_node or inp not in mlo_parameter_input_names):
                    num_preds += 1
                    input_node = True
        if num_succs == 0:
            outputs = self.model.graph.output
            for i in range(len(target_node.output)):
                out = _get_first_non_fifo_output(target_node, target_node.output[i])
                ret = get_by_name(outputs, out)  # Check that node is graph output
                if ret is not None:
                    num_succs += 1
                    output_node = True

        num_inputs = len(target_node.input)
        num_outputs = len(target_node.output)

        if num_outputs != num_succs:
            raise FINNInternalError(
                f"Node {target_node.name} has {num_outputs} outputs but "
                f"{num_succs} successor nodes. This is not supported for isolation."
            )

        # Process each input exactly once: either keep as initializer input or isolate via dummy
        pred_count = 0
        converted_initializer_input_indices: list[int] = []
        for i in range(num_inputs):
            inp_name = target_node.input[i]
            inp_name = _get_first_non_fifo_input(
                target_node, inp_name
            )  # Replace FIFO-produced inputs with their predecessor's input,
            # this will skip all FIFOs in the isolation
            is_mlo_parameter_input = is_mlo_node and inp_name in mlo_parameter_input_names
            init_vals_only = self.model.get_initializer(inp_name)
            if init_vals_only is not None or is_mlo_parameter_input:
                if init_vals_only is not None:
                    init_tensor, val_info = self._create_existing_initializer_input(
                        inp_name,
                        target_node,
                    )
                else:
                    init_tensor, val_info = self._create_mlo_dummy_initializer_input(
                        inp_name,
                        target_node,
                    )
                initializers.append(init_tensor)
                value_info_protos.append(val_info)
                inputs_node.append(val_info)
                converted_initializer_input_indices.append(i)
                continue

            pred_count += 1
            (
                new_input_info,
                new_input_dummy_info,
                dummy_node,
            ) = self._create_dynamic_input_with_dummy(
                inp_name,
                i,
                target_node,
                target_op,
            )
            value_info_protos.append(new_input_dummy_info)
            inputs_graph.append(new_input_info)
            inputs_node.append(new_input_dummy_info)
            nodes_graph.append(dummy_node)

        if pred_count != num_preds:
            raise FINNInternalError(
                f"Node {target_node.name} has {num_preds} pred. nodes but "
                f"{pred_count} inputs have been handled."
            )

        # Process each output exactly once and isolate via dummy
        succ_count = 0
        for i in range(num_outputs):
            out_name = target_node.output[i]
            new_output_info, new_output_dummy_info, dummy_node = self._create_output_with_dummy(
                out_name,
                i,
                target_node,
                target_op,
            )
            succ_count += 1
            value_info_protos.append(new_output_dummy_info)
            outputs_graph.append(new_output_info)
            outputs_node.append(new_output_dummy_info)
            nodes_graph.append(dummy_node)

        if succ_count != num_succs:
            raise FINNInternalError(
                f"Node {target_node.name} has {num_succs} succ. nodes but only "
                f"{succ_count} outputs have been handled."
            )

        # Copy the target node and create a new model with the target node and dummy nodes
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

        params_changed = False
        if len(converted_initializer_input_indices) > 0:
            if target_node.op_type.startswith("Elementwise"):
                if 0 in converted_initializer_input_indices:
                    params["lhs_style"] = "const"
                    params_changed = True
                if 1 in converted_initializer_input_indices:
                    params["rhs_style"] = "const"
                    params_changed = True
            if target_node.op_type.startswith("MVAU"):
                params["mem_mode"] = "internal_decoupled"
                params_changed = True
        if "mlo_max_iter" in params:
            del params["mlo_max_iter"]
            params_changed = True
        if "runtime_writeable_weights" in params and params["runtime_writeable_weights"] == 1:
            params["runtime_writeable_weights"] = 0
            if "ram_style" in params and params["ram_style"] == "ultra":
                params["ram_style"] = "block"
            params_changed = True
        if "dynamic_mode" in params and params["dynamic_mode"] == 1:
            params["dynamic_mode"] = 0
            params_changed = False
        if params_changed:
            params["code_gen_dir_ipgen"] = ""
            params["ipgen_path"] = ""
            params["ip_path"] = ""

        # Add support for hierachical models. FINN returns ModelWrapper,
        # but onnx needs GraphProto for subgraphs.
        # We need to convert any ModelWrapper parameters to GraphProto.
        for i in params.keys():
            if isinstance(params[i], ModelWrapper):
                params[i] = params[i].model.graph

        new_node = onnx.helper.make_node(
            target_node.op_type,
            inputs=[inp.name for inp in inputs_node],
            outputs=[outp.name for outp in outputs_node],
            domain=target_node.domain,
            name=target_node.name,
            **params,
        )
        nodes_graph.append(new_node)

        graph = onnx.helper.make_graph(
            nodes_graph,
            f"isolated_node_graph_{target_node.name}",
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

        rtlsim_trace_str = self.model.get_metadata_prop("rtlsim_trace")
        if rtlsim_trace_str is not None:
            rtlsim_trace_path = Path(rtlsim_trace_str)
            # Add suffix to name depending on node name
            rtlsim_trace_path = rtlsim_trace_path.with_name(
                rtlsim_trace_path.stem + f"_{target_node.name}" + rtlsim_trace_path.suffix
            )
            node_model.set_metadata_prop("rtlsim_trace", str(rtlsim_trace_path))

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
            ishape_folded = getHWCustomOp(first_node).get_folded_input_shape(ind=top_ind)
            instream_iters.append(int(np.prod(ishape_folded[:-1])))
        for top_out in model.graph.output:
            oname = top_out.name
            last_node = model.find_producer(oname)
            assert last_node is not None, "Failed to find producer for " + oname
            top_ind = list(last_node.output).index(oname)
            oshape_folded = getHWCustomOp(last_node).get_folded_output_shape(ind=top_ind)
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
            """Return formated Stream Descriptor."""
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
            sim_dir = Path(
                make_build_dir(f"rtlsim_{model.graph.node[0].name}_")
                if build_dir is None
                else build_dir
            )
            sim_base, sim_rel = finnxsi.compile_sim_obj(
                top_module_name, all_verilog_srcs, sim_dir, debug=debug, fifosim=True
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
        sim_comm_mode = (
            cast("str", self.model.get_metadata_prop("sim_comm_mode") or "shm")
        ).lower()
        enable_mpi = sim_comm_mode == SimulationCommMode.HYBRID_MPI.value

        # Determine executable name
        compile_targets = ["LayerSimulationBackend", "IsolatedSimulationBackend"]
        if all((Path(sim_base) / execname).exists() for execname in compile_targets):
            # Simulation was already compiled, we can return early
            return Path(sim_base)

        # Check where FINNXSI is
        finnxsi_dir = get_settings().finn_xsi

        # Running CMake first
        cmake_call = f"{sys.executable} -m cmake -S {finnxsi_dir} -B {sim_base}"
        if enable_mpi:
            cmake_call += " -DFIFOSIM_ENABLE_MPI=ON"
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
        finnxsi_dir = get_settings().finn_xsi
        # Prepare the C++ driver config template
        (
            instream_descrs_str,
            outstream_descrs_str,
        ) = self._get_stream_descriptions(model)

        sim_comm_mode = cast(
            "str",
            model.get_metadata_prop("sim_comm_mode") or SimulationCommMode.SHM.value,
        )
        if sim_comm_mode not in {m.value for m in SimulationCommMode}:
            raise FINNUserError(
                "Invalid simulation communication mode "
                f"'{sim_comm_mode}'. Expected one of: "
                f"{', '.join(m.value for m in SimulationCommMode)}"
            )

        sim_input_channel_types_str = model.get_metadata_prop("sim_input_channel_types")
        sim_output_channel_types_str = model.get_metadata_prop("sim_output_channel_types")
        sim_input_peer_ranks_str = model.get_metadata_prop("sim_input_peer_ranks")
        sim_output_peer_ranks_str = model.get_metadata_prop("sim_output_peer_ranks")
        input_channel_types = cast(
            "list[str] | None",
            literal_eval(sim_input_channel_types_str)
            if sim_input_channel_types_str is not None
            else None,
        )
        output_channel_types = cast(
            "list[str] | None",
            literal_eval(sim_output_channel_types_str)
            if sim_output_channel_types_str is not None
            else None,
        )
        input_peer_ranks = cast(
            "list[int] | None",
            literal_eval(sim_input_peer_ranks_str)
            if sim_input_peer_ranks_str is not None
            else None,
        )
        output_peer_ranks = cast(
            "list[int] | None",
            literal_eval(sim_output_peer_ranks_str)
            if sim_output_peer_ranks_str is not None
            else None,
        )

        if input_interface_names is None:
            input_interface_names = []
        if output_interface_names is None:
            output_interface_names = []

        if input_channel_types is None:
            input_channel_types = [SimulationCommMode.SHM.value] * len(input_interface_names)
        if output_channel_types is None:
            output_channel_types = [SimulationCommMode.SHM.value] * len(output_interface_names)
        if input_peer_ranks is None:
            input_peer_ranks = [-1] * len(input_interface_names)
        if output_peer_ranks is None:
            output_peer_ranks = [-1] * len(output_interface_names)

        if len(input_channel_types) != len(input_interface_names):
            raise FINNUserError(
                "Length mismatch: sim_input_channel_types has "
                f"{len(input_channel_types)} entries, but input interfaces have "
                f"{len(input_interface_names)} entries."
            )
        if len(output_channel_types) != len(output_interface_names):
            raise FINNUserError(
                "Length mismatch: sim_output_channel_types has "
                f"{len(output_channel_types)} entries, but output interfaces have "
                f"{len(output_interface_names)} entries."
            )
        if len(input_peer_ranks) != len(input_interface_names):
            raise FINNUserError(
                "Length mismatch: sim_input_peer_ranks has "
                f"{len(input_peer_ranks)} entries, but input interfaces have "
                f"{len(input_interface_names)} entries."
            )
        if len(output_peer_ranks) != len(output_interface_names):
            raise FINNUserError(
                "Length mismatch: sim_output_peer_ranks has "
                f"{len(output_peer_ranks)} entries, but output interfaces have "
                f"{len(output_interface_names)} entries."
            )

        allowed_channel_types = {SimulationCommMode.SHM.value, "mpi"}
        if any(
            ch not in allowed_channel_types for ch in input_channel_types + output_channel_types
        ):
            raise FINNUserError(
                "Invalid channel type in sim_input_channel_types/sim_output_channel_types. "
                "Allowed values are 'shm' and 'mpi'."
            )

        has_mpi_channels = any(ch == "mpi" for ch in input_channel_types + output_channel_types)
        if sim_comm_mode == SimulationCommMode.SHM.value and has_mpi_channels:
            raise FINNUserError(
                "sim_comm_mode='shm' is incompatible with channel type 'mpi'. "
                "Set sim_comm_mode='hybrid_mpi' or use only 'shm' channels."
            )
        if sim_comm_mode == SimulationCommMode.HYBRID_MPI.value:
            for ch, rank in zip(input_channel_types, input_peer_ranks, strict=True):
                if ch == "mpi" and rank < 0:
                    raise FINNUserError(
                        "MPI input channel detected but sim_input_peer_ranks contains "
                        "an invalid peer rank (< 0)."
                    )
            for ch, rank in zip(output_channel_types, output_peer_ranks, strict=True):
                if ch == "mpi" and rank < 0:
                    raise FINNUserError(
                        "MPI output channel detected but sim_output_peer_ranks contains "
                        "an invalid peer rank (< 0)."
                    )

        template_dict = {
            "TIMEOUT_CYCLES": timeout_cycles,
            "PRECISE_TIMEOUT": str(self.performance_sim).lower(),
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
            "INPUT_INTERFACE_NAMES": ",".join(
                ['"' + self.shm_prefix + name + '"' for name in input_interface_names]
            )
            if input_interface_names != []
            else "",
            "OUTPUT_INTERFACE_NAMES": ",".join(
                ['"' + self.shm_prefix + name + '"' for name in output_interface_names]
            )
            if output_interface_names != []
            else "",
            "INPUT_CHANNEL_TYPES": ",".join(['"' + ch + '"' for ch in input_channel_types]),
            "OUTPUT_CHANNEL_TYPES": ",".join(['"' + ch + '"' for ch in output_channel_types]),
            "INPUT_PEER_RANKS": ",".join([str(v) for v in input_peer_ranks]),
            "OUTPUT_PEER_RANKS": ",".join([str(v) for v in output_peer_ranks]),
            "INPUT_INTERFACE_COUNT": len(input_interface_names),
            "OUTPUT_INTERFACE_COUNT": len(output_interface_names),
            "SIM_COMM_MODE": sim_comm_mode,
            "NODE_INDEX": node_index,
            "TOTAL_NODES": total_nodes,
            "IS_INPUT_NODE": model.get_metadata_prop("input_node"),
            "IS_OUTPUT_NODE": model.get_metadata_prop("output_node"),
        }

        fifosim_config_fname = Path(finnxsi_dir) / "rtlsim_config.hpp.template"
        fsim_config = fifosim_config_fname.read_text()
        env = Environment()
        fsim_config = env.from_string(fsim_config).render(**template_dict)
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
        # Check that the relevant data exists
        wrapper_filename = node_model.get_metadata_prop("wrapper_filename")
        if wrapper_filename is None or not wait_for_file(Path(wrapper_filename), timeout=5.0):
            raise FINNUserError(
                f"Call CreateStitchedIP prior to building "
                f"the simulation for {self.model.graph.node[node_index].name}. "
                f"wrapper_filename is set to {wrapper_filename}!"
            )

        vivado_stitched_proj = node_model.get_metadata_prop("vivado_stitch_proj")
        if vivado_stitched_proj is None or not wait_for_dir(
            Path(vivado_stitched_proj), timeout=5.0
        ):
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

        sim_comm_mode_meta = self.model.get_metadata_prop("sim_comm_mode")
        sim_comm_mode = cast("str", sim_comm_mode_meta) if sim_comm_mode_meta is not None else "shm"
        rank_map, worker_map, host_map = distribute_mpi_ranks(self.model)

        def _build(
            node_index: int,
            total_nodes: int,
            build_dir: Path,
        ) -> Any:
            """Build simulation for a single node."""
            nodemodel = self._isolated_node_model(node_index)
            nodemodel = nodemodel.transform(InferShapes())
            nodemodel = nodemodel.transform(PrepareIP(self.fpgapart, self.clk_ns))
            nodemodel = nodemodel.transform(HLSSynthIP(self.fpgapart))
            nodemodel = nodemodel.transform(
                CreateStitchedIP(self.fpgapart, self.clk_ns, functional_simulation=functional_sim)
            )
            input_interface_names = nodemodel.get_metadata_prop("predecessors")
            if input_interface_names is not None:
                input_interface_names = literal_eval(input_interface_names)
            output_interface_names = nodemodel.get_metadata_prop("successors")
            if output_interface_names is not None:
                output_interface_names = literal_eval(output_interface_names)

            # Optional automatic channel/rank assignment for hybrid MPI mode.
            if sim_comm_mode == SimulationCommMode.HYBRID_MPI.value and rank_map and worker_map:
                node_name = self.model.graph.node[node_index].name
                node_worker = worker_map.get(node_name, -1)

                input_channel_types: list[str] = []
                output_channel_types: list[str] = []
                input_peer_ranks: list[int] = []
                output_peer_ranks: list[int] = []
                input_route_log: list[str] = []
                output_route_log: list[str] = []

                def _classify_channel(
                    peer: NodeProto | None, iface_name: str
                ) -> tuple[str, int, str]:
                    """Classify a single input/output channel relative to this node.

                    peer is the producer (for an input channel) or the first consumer
                    (for an output channel; only one stream per successor is expected
                    by this flow), or None if the channel has no in-graph peer.

                    Returns (channel_type, peer_rank, route_log_entry).
                    """
                    if peer is None:
                        return "shm", -1, f"{iface_name}:shm(peer=external)"
                    peer_worker = worker_map.get(peer.name)
                    peer_rank = rank_map.get(peer.name)
                    peer_host = host_map.get(peer.name, "localhost")
                    if (
                        peer_worker is not None
                        and peer_rank is not None
                        and peer_worker != node_worker
                    ):
                        return (
                            "mpi",
                            peer_rank,
                            f"{iface_name}:mpi(peer={peer.name},rank={peer_rank},host={peer_host})",
                        )
                    return "shm", -1, f"{iface_name}:shm(peer={peer.name},host={peer_host})"

                if input_interface_names is not None:
                    for inp_name in input_interface_names:
                        producer = self.model.find_producer(inp_name)
                        ch_type, peer_rank, route = _classify_channel(producer, inp_name)
                        input_channel_types.append(ch_type)
                        input_peer_ranks.append(peer_rank)
                        input_route_log.append(route)

                if output_interface_names is not None:
                    for out_name in output_interface_names:
                        consumers = self.model.find_consumers(out_name)
                        # One stream per successor is expected by this flow.
                        consumer = consumers[0] if consumers else None
                        ch_type, peer_rank, route = _classify_channel(consumer, out_name)
                        output_channel_types.append(ch_type)
                        output_peer_ranks.append(peer_rank)
                        output_route_log.append(route)

                nodemodel.set_metadata_prop("sim_input_channel_types", str(input_channel_types))
                nodemodel.set_metadata_prop("sim_output_channel_types", str(output_channel_types))
                nodemodel.set_metadata_prop("sim_input_peer_ranks", str(input_peer_ranks))
                nodemodel.set_metadata_prop("sim_output_peer_ranks", str(output_peer_ranks))
                nodemodel.set_metadata_prop("sim_comm_mode", sim_comm_mode)

            return self.build_single_node_simulation(
                nodemodel,
                node_index,
                total_nodes,
                input_interface_names,
                output_interface_names,
                build_dir,
                silent=with_live_display,
            )

        nodes = (
            [n for n in self.model.graph.node if "FIFO" not in n.op_type]
            if self.performance_sim
            else self.model.graph.node
        )
        total_nodes = len(nodes)
        log.info(f"[BuildSimulation] Preparing to build {total_nodes} nodes for the simulation.")
        futures: dict[int, Future] = {}
        built_nodes = 0

        # Progress display callback
        def _callback_progress(name: str) -> Callable:
            """Return formatted progress callback."""
            nonlocal total_nodes, built_nodes

            def _f(f: Future) -> None:
                """Return callback function for progress display."""
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
        def _try_int(value: str | None) -> int | None:
            """Cast value to int, return None if it fails or if value is None."""
            if value is None:
                return None
            try:
                parsed = int(value)
            except ValueError:
                return None
            return parsed if parsed > 0 else None

        def _parse_slurm_job_cpus_per_node(value: str | None) -> int | None:
            """Return parse slurm job cpus per node."""
            if value is None:
                return None
            # Example values: "16", "16(x2)", "16(x2),8"
            first_chunk = value.split(",")[0].strip()
            match = re.match(r"^(\d+)", first_chunk)
            if match is None:
                return None
            parsed = int(match.group(1))
            return parsed if parsed > 0 else None

        def _get_slurm_cpus() -> int | None:
            """Return slurm workers while considering cpu allocation."""
            cpus_per_task = _try_int(os.environ.get("SLURM_CPUS_PER_TASK"))
            if cpus_per_task is not None:
                return cpus_per_task

            cpus_on_node = _try_int(os.environ.get("SLURM_CPUS_ON_NODE"))
            if cpus_on_node is not None:
                return cpus_on_node

            return _parse_slurm_job_cpus_per_node(os.environ.get("SLURM_JOB_CPUS_PER_NODE"))

        def _get_slurm_mem_workers(cpus_alloc: int | None) -> int | None:
            # SLURM memory env vars are in MB.
            """Return number of slurm workers while considering memory allocation."""
            mem_per_node_mb = _try_int(os.environ.get("SLURM_MEM_PER_NODE"))
            if mem_per_node_mb is not None:
                return max(1, mem_per_node_mb // (10 * 1024))  # 10GB per synthesis

            mem_per_cpu_mb = _try_int(os.environ.get("SLURM_MEM_PER_CPU"))
            if mem_per_cpu_mb is not None and cpus_alloc is not None:
                return max(1, (mem_per_cpu_mb * cpus_alloc) // (10 * 1024))

            return None

        slurm_detected = os.environ.get("SLURM_JOB_ID") is not None
        if slurm_detected:
            cpus_alloc = _get_slurm_cpus()
            if cpus_alloc is None:
                cpus_alloc = 1

            if functional_sim:
                mem_workers = _get_slurm_mem_workers(cpus_alloc)
                if mem_workers is not None:
                    synth_workers = max(1, min(cpus_alloc, mem_workers))
                else:
                    synth_workers = max(1, cpus_alloc)
            else:
                synth_workers = max(1, cpus_alloc)

            synth_workers = min(
                synth_workers,
                len(nodes),
                int(os.environ.get("NUM_DEFAULT_WORKERS", len(nodes))),
            )
            log.info(
                "[BuildSimulation] SLURM job detected, using "
                f"{synth_workers} workers based on allocation."
            )
        else:
            synth_workers = min(
                max(1, cast("int", (psutil.virtual_memory().free / 1024 / 1024 / 1024) // 10)),
                int(os.environ.get("NUM_DEFAULT_WORKERS", len(self.model.graph.node))),
            )  # 10GB per synthesis
            if not functional_sim:
                # When not having to do synthesis, the build is not memory bottlenecked and
                # can be executed as parallel as possible
                synth_workers = int(
                    os.environ.get("NUM_DEFAULT_WORKERS", len(self.model.graph.node))
                )

        # Build (stitched IP, cmake, make) all sims in parallel and return paths to
        # the compiled executables
        log.info("[BuildSimulation] Starting the build process.")
        fifos = self.model.get_nodes_by_op_type("StreamingFIFO")
        fifos.extend(self.model.get_nodes_by_op_type("StreamingFIFO_rtl"))
        if len(fifos) != 0 and not self.performance_sim:
            raise FINNUserError(
                "FIFOs detected in model during FIFO sizing. "
                "Did you call the steps in the correct order?"
            )
        with ThreadPoolExecutor(max_workers=synth_workers) as pool:
            for i in range(len(self.model.graph.node)):
                node_name = self.model.graph.node[i].name
                if "FIFO" in node_name:
                    continue
                futures[i] = pool.submit(
                    _build,
                    i,
                    len(self.model.graph.node) - 1,
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
        self, fpgapart: str, clk_ns: float, functional_sim: bool, performance_sim: bool = False
    ) -> None:
        """Create a new BuildSimulation transform."""
        self.functional_sim = functional_sim
        self.fpgapart = fpgapart
        self.clk_ns = clk_ns
        self.performance_sim = performance_sim

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Build / compile the model. Modifies the model."""
        self.model = model
        sim_comm_mode = (
            cast("str", self.model.get_metadata_prop("sim_comm_mode") or "shm")
        ).lower()
        enable_mpi = sim_comm_mode == SimulationCommMode.HYBRID_MPI.value

        # Check if we already have stitched IPs and built simulations. If so, rerun only cmake/make
        needs_rebuild = True
        sim_binaries = self.model.get_metadata_prop("simulation_binaries")

        if self.performance_sim:
            log.info("[BuildSimulation] Performance simulation mode enabled.")

        # 1. Check if binary paths are saved in the model
        if sim_binaries is not None:
            sim_binaries = sim_binaries.split("\n")

            # 2. Check that the model size hasn't changed since creating the binaries. Otherwise
            # we should rebuild.
            nodes = (
                [n for n in self.model.graph.node if "FIFO" not in n.op_type]
                if self.performance_sim
                else self.model.graph.node
            )
            if len(sim_binaries) != len(nodes):
                log.info(
                    f"[BuildSimulation] Found existing binaries, but number ({len(sim_binaries)}) "
                    f"does not match number of nodes in the graph "
                    f"({len(nodes)}). Rebuilding..."
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
            # For rtlsim performance, we assume, that we already have a complete model.
            if not self.performance_sim:
                self._prepare_model()

            # Set shm prefix
            prefix = f"finn_sim_{os.getpid()}_"
            self.shm_prefix = (
                prefix + "".join(random.choices(string.ascii_letters + string.digits, k=12)) + "_"
            )
            self.model.set_metadata_prop("shm_prefix", self.shm_prefix)

            self.builder = SimulationBuilder(
                self.model, self.fpgapart, self.clk_ns, self.shm_prefix, self.performance_sim
            )
            with contextlib.suppress(AttributeError):
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
                """Compile binary in path binary."""
                cmake_cmd = "cmake"
                if enable_mpi:
                    cmake_cmd += " -DFIFOSIM_ENABLE_MPI=ON"
                result = subprocess.run(
                    f"{cmake_cmd} .;make",
                    shell=True,
                    cwd=str(binary),
                    text=True,
                    capture_output=True,
                )
                if result.returncode != 0:
                    raise FINNUserError(f"Failed compilation in {binary}: {result.stderr}")

            # Since we dont need a rebuild, sim_binaries contains the paths to the binaries
            sim_binaries = [Path(p) for p in cast("list[str]", sim_binaries)]
            total = len(sim_binaries)

            # Prepare compiling the binaries again
            done = 0

            def _progress_callback(binary: str | Path) -> Callable:
                """Return progress callback formatted."""
                nonlocal done, total

                def _f(future: Future) -> None:
                    """Return progress callback."""
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
        old_input_names = [i.name for i in self.model.graph.input]
        self.model = self.model.transform(GiveReadableTensorNames())
        for old_name, node in zip(old_input_names, self.model.graph.input, strict=True):
            self.model.rename_tensor(node.name, old_name)
        log.info("[BuildSimulation] Preparing IPs...")
        self.model = self.model.transform(PrepareIP(self.fpgapart, self.clk_ns))
        log.info("[BuildSimulation] Synthesizing IPs...")
        self.model = self.model.transform(HLSSynthIP())
        log.info("[BuildSimulation] Model preparation done.")

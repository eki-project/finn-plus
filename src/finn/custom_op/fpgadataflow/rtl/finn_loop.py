# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""RTL backend for the FINN loop meta-operator (MLO)."""

import copy
import itertools
import math
import numpy as np
import numpy.typing as npt
import os
import re
import shutil
import subprocess
from onnx import GraphProto
from pathlib import Path
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp, is_custom_op
from qonnx.util.basic import get_by_name, qonnx_make_model, roundup_to_integer_multiple
from typing import TYPE_CHECKING, Any, cast

import finn.core.onnx_exec as oxe
import finn.xsi as finnxsi
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.templates import load_codegen_template
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.util.basic import getHWCustomOp, make_build_dir, resolve_xilinx_tool
from finn.util.create import adjacency_list
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.rtlsim import dat_file_to_numpy_array, mlo_prehook_func_factory
from finn.util.settings import get_settings

if TYPE_CHECKING:
    from onnx import NodeProto

    from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU
    from finn.custom_op.fpgadataflow.rtl.elementwise_binary_rtl import ElementwiseBinary_rtl

# Value types accepted by the base ``get_nodeattr`` / ``set_nodeattr`` API.
BaseNodeAttrValue = int | float | str | bool | npt.NDArray | list[str | int | float] | None
# ``set_nodeattr`` on this op additionally accepts a graph value for the ``body``
# attribute (as a ``ModelWrapper`` or a raw ``GraphProto``).
SetNodeAttrValue = ModelWrapper | GraphProto | BaseNodeAttrValue
# Shape of the dict returned by ``get_nodeattr_types``: attribute name ->
# (dtype, required, default[, allowed_values]). Must match the base classes.
NodeAttrTypes = dict[
    str,
    tuple[str, bool, int | float | str | bool | npt.NDArray | list]
    | tuple[str, bool, int | float | str | bool | npt.NDArray | list, set | None],
]


def collect_ip_dirs(model: ModelWrapper, ipstitch_path: str) -> list[str]:
    """Return the IP directories of all nodes in the model plus the stitched IP and memstream dirs.

    Args:
        model: The (loop body) model whose nodes' IP directories are collected.
        ipstitch_path: Path of the stitched IP project of the model.

    Returns:
        The list of IP directories.
    """
    ip_dirs = []
    need_memstreamer = False
    for node in model.graph.node:
        node_inst = getCustomOp(node)
        ip_dir_value = node_inst.get_nodeattr("ip_path")
        if not Path(cast("str", ip_dir_value)).is_dir():
            raise FINNInternalError(
                "The directory that should contain the generated ip blocks doesn't exist."
            )
        ip_dirs += [cast("str", ip_dir_value)]
        if (
            node.op_type.startswith("MVAU") or node.op_type == "Thresholding_hls"
        ) and node_inst.get_nodeattr("mem_mode") == "internal_decoupled":
            need_memstreamer = True
    ip_dirs += [ipstitch_path + "/ip"]
    if need_memstreamer:
        # add RTL streamer IP
        ip_dirs.append(str(Path(get_settings().finn_rtllib) / "memstream"))
    return ip_dirs


@register_custom_op
class FINNLoop(RTLBackend, HWCustomOp):
    """Meta/container node for a group of fpgadataflow nodes executed in a loop.

    The wrapped nodes have been separated out into a FINN-ONNX model of their own
    and are meant to be executed ``iteration`` times.
    """

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return node attribute types for the loop body, iteration count, datatypes and memory."""
        my_attrs: NodeAttrTypes = {
            "body": ("g", True, ""),
            "iteration": ("i", False, 1),
            # FINN input datatype
            "inputDataType": ("s", True, ""),
            # FINN output datatype
            "outputDataType": ("s", True, ""),
            # Path to save per-iteration execution context (cppsim only).
            # If non-empty, each iteration's full context is saved to this path.
            "iteration_context_path": ("s", False, ""),
            # Memory type used to stream this loop's weights when running MLO.
            # "DDR" enables DDR address offset assignment, any other value skips it.
            # Empty means it is resolved from the target board at build
            "mem_type": ("s", False, ""),
        }
        my_attrs.update(HWCustomOp.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def get_nodeattr(self, name: str) -> BaseNodeAttrValue:
        """Get a node attribute by name.

        Data is stored inside the ONNX node's AttributeProto container. The
        attribute must be part of ``get_nodeattr_types``. The default value is
        returned if the attribute is not set.

        Note: the graph-typed ``body`` attribute is returned as a
        ``ModelWrapper`` (use the :attr:`body` property for a typed accessor).
        The return type is kept identical to the base method; callers that need
        the ``ModelWrapper`` cast it themselves.
        """
        try:
            (dtype, req, def_val, _allowed_values) = self.get_nodeattr_def(name)
            attr = get_by_name(self.onnx_node.attribute, name)
            if attr is not None:
                # dtype indicates which ONNX Attribute member to use
                # g : graph
                if dtype == "g":
                    ret = attr.__getattribute__(dtype)
                    return cast("BaseNodeAttrValue", ModelWrapper(qonnx_make_model(ret)))
                return super().get_nodeattr(name)
            if req:
                raise FINNUserError(
                    f"Required attribute {name} unspecified in a {self.onnx_node.op_type} node"
                )
            # not set, return default value
            return def_val
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name) from None

    def set_nodeattr(self, name: str, value: SetNodeAttrValue) -> None:
        """Set a node attribute by name.

        Data is stored inside the ONNX node's AttributeProto container. The
        attribute must be part of ``get_nodeattr_types``. In addition to the
        base value types, the graph-typed ``body`` attribute may be set from a
        ``ModelWrapper`` or a raw ``GraphProto``.
        """
        try:
            (dtype, _req, _def_val, _allowed_values) = self.get_nodeattr_def(name)
            attr = get_by_name(self.onnx_node.attribute, name)
            if attr is not None:
                # dtype indicates which ONNX Attribute member to use
                # g : graph
                if dtype == "g":
                    if isinstance(value, ModelWrapper):
                        value = value.model.graph
                    if not isinstance(value, GraphProto):
                        raise FINNInternalError(
                            "Value for graph attribute must be a GraphProto or ModelWrapper"
                        )
                    attr.g.CopyFrom(value)
                else:
                    super().set_nodeattr(name, cast("BaseNodeAttrValue", value))
            else:
                super().set_nodeattr(name, cast("BaseNodeAttrValue", value))
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name) from None

    @property
    def body(self) -> ModelWrapper:
        """Return the loop body model (the ``body`` graph attribute)."""
        return cast("ModelWrapper", self.get_nodeattr("body"))

    @property
    def iteration(self) -> int:
        """Return the number of loop iterations."""
        return cast("int", self.get_nodeattr("iteration"))

    @property
    def code_gen_dir_ipgen(self) -> str:
        """Return the code generation directory for IP generation."""
        return cast("str", self.get_nodeattr("code_gen_dir_ipgen"))

    @property
    def has_address_offset(self) -> bool:
        """Return whether an ``address_offset`` was assigned (DDR-streamed loop).

        Checks for the presence of the attribute on the ONNX node, not its value.
        """
        return get_by_name(self.onnx_node.attribute, "address_offset") is not None

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...] | list[int]:
        """Return the unfolded shape of input ``ind`` as seen by its consumer in the loop body."""
        loop_body = self.body
        if ind == 0:
            # get first node in loop body and return
            # normal input shape
            node = loop_body.graph.node[0]
            if is_custom_op(node.domain):
                inst = cast("HWCustomOp", getCustomOp(node))
                ishape = inst.get_normal_input_shape(0)
            else:
                ishape = loop_body.get_tensor_shape(node.input[0])
        else:
            tensor = loop_body.graph.input[ind].name
            # get consumer, assuming the second input is the parameter input
            param_node = loop_body.find_consumer(tensor)
            if param_node is not None and is_custom_op(param_node.domain):
                inst = cast("HWCustomOp", getCustomOp(param_node))
                ishape = inst.get_normal_input_shape(1)
            else:
                ishape = loop_body.get_tensor_shape(tensor)
        return cast("tuple[int, ...] | list[int]", ishape)

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...] | list[int]:  # noqa: ARG002
        """Return the unfolded output shape of the last node in the loop body."""
        loop_body = self.body
        # get last node in loop body and return
        # normal output shape
        node = loop_body.graph.node[-1]
        if is_custom_op(node.domain):
            inst = cast("HWCustomOp", getCustomOp(node))
            oshape = inst.get_normal_output_shape(0)
        else:
            oshape = loop_body.get_tensor_shape(node.output[0])
        return cast("tuple[int, ...] | list[int]", oshape)

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...] | list[int]:
        """Return the folded shape of input ``ind`` as seen by its consumer in the loop body."""
        loop_body = self.body
        if ind == 0:
            # get first node in loop body and return
            # normal input shape
            node = loop_body.graph.node[0]
            inst = cast("HWCustomOp", getCustomOp(node))
            ishape = inst.get_folded_input_shape(0)
        else:
            tensor = loop_body.graph.input[ind].name
            # get consumer, assuming the second input is the parameter input
            param_node = loop_body.find_consumer(tensor)
            if param_node is None:
                raise FINNInternalError(f"No consumer found for loop parameter tensor {tensor}")
            inst = cast("HWCustomOp", getCustomOp(param_node))
            ishape = inst.get_folded_input_shape(1)
        return cast("tuple[int, ...] | list[int]", ishape)

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...] | list[int]:  # noqa: ARG002
        """Return the folded output shape of the last node in the loop body."""
        loop_body = self.body
        # get last node in loop body and return
        # normal output shape
        node = loop_body.graph.node[-1]
        inst = cast("HWCustomOp", getCustomOp(node))
        return cast("tuple[int, ...] | list[int]", inst.get_folded_output_shape(0))

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Do nothing; datatypes are fixed by the inputDataType/outputDataType attributes."""

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:
        """Return the FINN DataType of the input at the given index."""
        if ind == 0:
            idt = DataType[cast("str", self.get_nodeattr("inputDataType"))]
        else:
            loop_body = self.body
            tensor = loop_body.graph.input[ind].name
            # get consumer, assuming the second input is the parameter input
            param_node = loop_body.find_consumer(tensor)
            if param_node is not None and is_custom_op(param_node.domain):
                inst = cast("HWCustomOp", getCustomOp(param_node))
                idt = inst.get_input_datatype(1)
            else:
                idt = loop_body.get_tensor_datatype(tensor)
        return idt

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return the FINN DataType of the output."""
        return DataType[cast("str", self.get_nodeattr("outputDataType"))]

    def get_instream_width(self, ind: int = 0) -> int:
        """Return the width of input stream ``ind`` as seen by its consumer in the loop body."""
        loop_body = self.body
        if ind == 0:
            # get first node in loop body and return
            # normal input shape
            node = loop_body.graph.node[0]
            inst = cast("HWCustomOp", getCustomOp(node))
            iwidth = inst.get_instream_width(0)
        else:
            tensor = loop_body.graph.input[ind].name
            # get consumer, assuming the second input is the parameter input
            param_node = loop_body.find_consumer(tensor)
            if param_node is None:
                raise FINNInternalError(f"No consumer found for loop parameter tensor {tensor}")
            inst = cast("HWCustomOp", getCustomOp(param_node))
            iwidth = inst.get_instream_width(1)
        return iwidth

    def get_exp_cycles(self) -> int:
        """Return the expected cycle count from the body latency, throughput and iteration count."""
        loop_body = self.body
        check_if_cycles_annotated = False

        for node in loop_body.graph.node:
            cnode = getCustomOp(node)
            if cnode.get_nodeattr("cycles_estimate"):
                check_if_cycles_annotated = True
                break
        if not check_if_cycles_annotated:
            loop_body = loop_body.transform(AnnotateCycles())

        iteration = self.iteration
        perf = loop_body.analysis(dataflow_performance)
        body_latency = cast("int", perf["critical_path_cycles"])
        body_throughput = cast("int", perf["max_cycles"])
        overhead_per_iter = 40
        return body_latency + (iteration - 1) * body_throughput + overhead_per_iter * iteration

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return the output stream width of the last node in the loop body."""
        loop_body = self.body
        # get last node in loop body and return
        # normal output shape
        node = loop_body.graph.node[-1]
        inst = cast("HWCustomOp", getCustomOp(node))
        return inst.get_outstream_width(0)

    def get_number_output_values(self) -> int | dict[str, int]:
        """Return the number of output values of the last node in the loop body."""
        loop_body = self.body
        # get last node in loop body and return
        # normal output values
        node = loop_body.graph.node[-1]
        inst = cast("HWCustomOp", getCustomOp(node))
        return inst.get_number_output_values()

    def prepare_rtlsim(self, behav: bool = False) -> None:
        """Create an xsi emulation library for this node's RTL code.

        Sets the ``rtlsim_so`` attribute to the resulting path.
        """
        vivado_stitch_proj_dir = self.code_gen_dir_ipgen
        with (Path(vivado_stitch_proj_dir) / "all_verilog_srcs.txt").open() as f:
            all_verilog_srcs = f.read().split()
        top_module_file_name = Path(cast("str", self.get_nodeattr("ipgen_path"))).resolve().name
        top_module_name = top_module_file_name.strip(".v")
        single_src_dir = Path(make_build_dir("rtlsim_" + top_module_name + "_"))
        trace_file = self.get_nodeattr("rtlsim_trace")
        debug = not (trace_file is None or trace_file == "")
        rtlsim_so = finnxsi.compile_sim_obj(
            top_module_name, all_verilog_srcs, single_src_dir, debug, behav
        )
        # save generated lib filename in attribute
        sim_base, sim_rel = rtlsim_so
        self.set_nodeattr("rtlsim_so", str(sim_base / sim_rel))

    def execute_node(
        self, context: dict[str, np.ndarray], graph: GraphProto  # noqa: ARG002
    ) -> None:
        """Execute the loop via rtlsim or by running the body model once per iteration in Python."""
        node = self.onnx_node
        inp_values = context[node.input[0]]
        if self.get_nodeattr("exec_mode") == "rtlsim":
            # prepare input io_dict
            io_dict: dict[str, dict[str, Any]] = {"inputs": {}, "outputs": {}}
            itensor = inp_values.reshape(self.get_folded_input_shape(0))
            idt = self.get_input_datatype(0)
            iwidth = self.get_instream_width(0)
            # pack input for rtlsim
            packed_input = npy_to_rtlsim_input(itensor, idt, iwidth)
            io_dict["inputs"]["in0"] = packed_input
            io_dict["outputs"]["out0"] = []
            mlo_prehook = mlo_prehook_func_factory(self.onnx_node)
            sim = self.get_rtlsim()
            # reset and call rtlsim, including any pre/post hooks
            self.reset_rtlsim(sim)
            mlo_prehook(sim)
            self.rtlsim_multi_io(
                sim,
                io_dict,
            )
            self.close_rtlsim(sim)
            odt = self.get_output_datatype(0)
            o_folded_shape = self.get_folded_output_shape(0)
            owidth = self.get_outstream_width(0)
            packed_output = io_dict["outputs"]["out0"]
            o_folded_tensor = rtlsim_output_to_npy(
                packed_output, None, odt, o_folded_shape, owidth, odt.bitwidth()
            )
            oshape = self.get_normal_output_shape(0)
            result = o_folded_tensor.reshape(oshape)
        else:
            loop_body = self.body
            # for each iteration run execution
            iteration = self.iteration
            # The "iteration_context_path" attr has dtype "s" with default "", so
            # get_nodeattr never returns None here (unlike the pre-typing version,
            # which kept a redundant ``is not None`` guard).
            iteration_context_path = cast("str", self.get_nodeattr("iteration_context_path"))
            save_iteration_context = iteration_context_path != ""
            all_iteration_contexts: dict[str, np.ndarray] = {}
            outp_dict: dict[str, np.ndarray] = {}
            for i_iter in range(iteration):
                # set the right parameters
                input_dict: dict[str, np.ndarray] = {}
                for i, _inp in enumerate(node.input):
                    if i == 0:
                        input_dict[loop_body.graph.input[i].name] = inp_values
                    else:
                        params = context[node.input[i]]
                        input_dict[loop_body.graph.input[i].name] = params[i_iter]
                outp_dict = oxe.execute_onnx(loop_body, input_dict, return_full_exec_context=True)
                inp_values = outp_dict[loop_body.graph.output[0].name]
                # Save iteration context if enabled
                if save_iteration_context:
                    for tensor_name, tensor_val in outp_dict.items():
                        # Skip empty tensor name (dummy entry for default values)
                        if tensor_name:
                            key = f"iter_{i_iter}_{tensor_name}"
                            all_iteration_contexts[key] = tensor_val
            result = outp_dict[loop_body.graph.output[0].name]
            # Save all iteration contexts to file
            if save_iteration_context:
                np.savez(iteration_context_path, allow_pickle=True, **all_iteration_contexts)
        context[node.output[0]] = np.asarray(result, dtype=np.float32)

    def generate_hdl(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:  # noqa: ARG002
        """Generate the loop control wrapper, stream taps, parameters and body IP for this loop."""
        # Generate params as part of IP preparation
        code_gen_dir = self.code_gen_dir_ipgen
        self.generate_hdl_stream_tap()
        self.generate_params(model, code_gen_dir, fpgapart)
        code_gen_dict: dict[str, list[str]] = {}
        code_gen_dict["$LOOP_CONTROL_WRAPPER_NAME$"] = [f"{self.onnx_node.name}_loop_cont_wrapper"]
        code_gen_dict["$N_MAX_LAYERS$"] = [str(self.iteration)]
        code_gen_dict["$N_LAYERS$"] = [str(self.iteration)]
        code_gen_dict["$ELEM_BITS$"] = [str(self.get_input_datatype(0).bitwidth())]
        code_gen_dict["$ILEN_BITS$"] = [str(self.get_instream_width(0))]
        code_gen_dict["$OLEN_BITS$"] = [str(self.get_outstream_width(0))]
        # Intermediate frame address offset
        code_gen_dict["$ADDRESS_OFFSET$"] = [str(self.get_nodeattr("address_offset"))]

        input_elements = int(np.prod(self.get_normal_input_shape(0)))
        code_gen_dict["$INPUT_ELEMS$"] = [str(input_elements)]

        template_path = Path(get_settings().finn_rtllib) / "mlo" / "loop_control_wrapper.v"
        template_wrapper = template_path.read_text()
        for key, lines in code_gen_dict.items():
            # transform list into long string separated by '\n'
            template_wrapper = template_wrapper.replace(key, "\n".join(lines))
        if self.has_address_offset:
            template_wrapper = "`define HAS_BASE_ADDRESS\n" + template_wrapper
        (Path(code_gen_dir) / f"{self.onnx_node.name}_wrapper.v").write_text(template_wrapper)

    def generate_params(
        self, model: ModelWrapper, path: str | Path, fpgapart: str | None = None
    ) -> None:
        """Generate .dat files for loop parameters and concatenate them together.

        For DDR-streamed loops (``address_offset`` assigned), also emit the
        address_config wrapper that exposes the runtime base address.

        Args:
            model: The model containing this loop node (holds the stacked parameters).
            path: Directory the parameter files are written to.
            fpgapart: Target FPGA part, forwarded to body nodes whose parameter layout
                depends on the device (e.g. Requant).
        """
        if self.has_address_offset:
            ac_template_path = Path(get_settings().finn_rtllib) / "mlo" / "address_config_wrapper.v"
            ac_module_name = self.onnx_node.name + "_address_config_wrapper"
            ac_wrapper = ac_template_path.read_text()
            ac_wrapper = ac_wrapper.replace("$MODULE_NAME_AXI_WRAPPER$", ac_module_name)
            (Path(path) / f"{ac_module_name}.v").write_text(ac_wrapper)

        iteration = self.iteration
        loop_node = self.onnx_node
        loop_body = self.body
        # Nodes with multiple parameter inputs (e.g. Requant: scale+bias) are
        # handled once, across all their param inputs, and skipped on later
        # passes of the per-graph-input loop below.
        handled_nodes: set[str] = set()
        for i, inp in enumerate(loop_node.input[1:]):
            params = model.get_initializer(inp)
            param_dtype = model.get_tensor_datatype(inp)
            if params is None or not isinstance(params, np.ndarray):
                raise FINNUserError(
                    f"Expected initializer for loop parameter input {inp} "
                    f"not found or not an ndarray."
                )
            if params.shape[0] != iteration:
                raise FINNUserError(
                    f"Expected first dimension of loop parameter {inp} to "
                    f"be equal to iteration count {iteration}, but got {params.shape[0]}."
                )
            # get node that initializer is attached to
            loop_tensor = loop_body.graph.input[i + 1].name
            param_node = loop_body.find_consumer(loop_tensor)
            if param_node is None:
                raise FINNInternalError(
                    f"Could not find consumer of loop parameter tensor {loop_tensor} in loop body."
                )
            if param_node.name in handled_nodes:
                continue
            if param_node.op_type == "Requant_rtl":
                self._generate_params_requant(
                    model, path, loop_node, loop_body, param_node, fpgapart
                )
                handled_nodes.add(param_node.name)
                continue
            inst = None
            for it in range(iteration):
                loop_body.set_initializer(loop_tensor, params[it])
                loop_body.set_tensor_datatype(loop_tensor, param_dtype)
                inst = getHWCustomOp(param_node)
                inst.generate_params(loop_body, path)
                param_file = f"{path}/memblock.dat"
                new_param_file = f"{path}/{param_node.op_type}_memblock_{it}.dat"
                if param_node.op_type.startswith("MVAU") or param_node.op_type.startswith(
                    "Elementwise"
                ):
                    # rename so it doesn't get overwritten
                    shutil.move(param_file, new_param_file)
                elif param_node.op_type.startswith("Thresholding"):
                    # get all generated Thresholding dat files
                    pe = cast("int", inst.get_nodeattr("PE"))
                    output_data_type = cast("str", inst.get_nodeattr("outputDataType"))
                    o_bitwidth = DataType[output_data_type].bitwidth()
                    param_files = []
                    for stage in range(o_bitwidth):
                        for pe_value in range(pe):
                            param_files.append(
                                f"{path}/{param_node.name}_threshs_{pe_value}_{stage}.dat"
                            )
                    for param_file in param_files:
                        param_path = Path(param_file)
                        new_param_file = param_path.with_name(
                            param_path.stem + "_i" + str(it) + param_path.suffix
                        )
                        shutil.move(param_path, new_param_file)
                else:
                    raise FINNUserError(
                        f"Node of type {param_node.op_type} not supported as loop node."
                    )

            if param_node.op_type.startswith("MVAU"):
                # Concatenate the per-iteration weights, padding each layer up to the
                # AXI-bus offset (LAYER_OFFS in fetch_weights.sv) so that the DDR
                # image built from this .dat reads layer i from i*LAYER_OFFS. The
                # rtlsim memory model (mlo_sim.py) and the driver's DDR upload
                # (make_driver.py) consume this file as-is. Written one hex byte per
                # line.
                param_file = Path(path) / f"memblock_{param_node.op_type}_id_{i + 1}.dat"
                mvau_inst = cast("MVAU", getCustomOp(param_node))
                layer_bytes, layer_offs = mvau_inst.get_weight_mem_bytes()
                padded = np.zeros(iteration * layer_offs, dtype=np.uint8)
                for it in range(iteration):
                    memblock_file = Path(path) / f"{param_node.op_type}_memblock_{it}.dat"
                    layer = dat_file_to_numpy_array(memblock_file)
                    padded[it * layer_offs : it * layer_offs + layer_bytes] = layer
                    memblock_file.unlink()
                with param_file.open("w") as outfile:
                    outfile.write("\n".join(f"{b:02x}" for b in padded))
                    outfile.write("\n")
            elif param_node.op_type.startswith("Elementwise"):
                # Concatenate the per-iteration .dat files back-to-back into the
                # single memblock the memstream reads (no DDR padding here, unlike
                # the MVAU path above).
                param_file = Path(path) / f"memblock_{param_node.op_type}_id_{i + 1}.dat"
                with param_file.open("w") as outfile:
                    for it in range(iteration):
                        memblock_file = Path(path) / f"{param_node.op_type}_memblock_{it}.dat"
                        with memblock_file.open("r") as infile:
                            for line in infile:
                                outfile.write(line)
                        memblock_file.unlink()  # remove the per-iteration file after concatenation
                # Point the ipgen memstream wrapper at the concatenated .dat file.
                # Adapted from transformations.fpgadataflow.replace_verilog_relpaths
                param_customop = getCustomOp(param_node)
                ipgen_path = Path(cast("str", param_customop.get_nodeattr("code_gen_dir_ipgen")))
                if ipgen_path.is_dir():
                    init_file = Path(path) / f"memblock_{param_node.op_type}_id_{i + 1}.dat"
                    pattern = re.compile(
                        r'^(\s*parameter\s+INIT_FILE\s*=\s*")[^"]+(".*)$',
                        re.MULTILINE,
                    )
                    for fpath in ipgen_path.rglob("*_memstream_wrapper.v"):
                        s = fpath.read_text()
                        updated, n = pattern.subn(
                            lambda m, init_file=init_file: (f"{m.group(1)}{init_file}{m.group(2)}"),
                            s,
                            count=1,
                        )
                        if n:
                            fpath.write_text(updated)
            elif param_node.op_type.startswith("Thresholding"):
                # concatinate all .dat files together
                if inst is None:
                    raise FINNInternalError(
                        "Expected inst to be set after loop over iterations, but it was not."
                    )
                pe = cast("int", inst.get_nodeattr("PE"))
                output_data_type = cast("str", inst.get_nodeattr("outputDataType"))
                o_bitwidth = DataType[output_data_type].bitwidth()
                for stage in range(o_bitwidth):
                    for pe_value in range(pe):
                        param_file = (
                            Path(path) / f"Thresholding_id_{i + 1}_threshs_{pe_value}_{stage}.dat"
                        )
                        with param_file.open("w") as outfile:
                            for it in range(iteration):
                                iter_file = (
                                    Path(path)
                                    / f"{param_node.name}_threshs_{pe_value}_{stage}_i{it}.dat"
                                )
                                with iter_file.open("r") as infile:
                                    cnt = 0
                                    hex_len = 0
                                    for line in infile:
                                        if cnt == 0:
                                            hex_len = len(line.strip())
                                        cnt += 1
                                        outfile.write(line)
                                    # is power of 2?
                                    if (cnt & (cnt - 1)) != 0:
                                        # pad with max value
                                        next_pow2 = 2 ** math.ceil(math.log2(cnt))
                                        pad_val = 2**o_bitwidth - 1
                                        for _ in range(next_pow2 - cnt):
                                            # write out as hex of len hex_len
                                            outfile.write(hex(pad_val)[2:].zfill(hex_len) + "\n")
                                iter_file.unlink()

                # Replace the path for the dat files in the ipgen files
                # Adapted from transformations.fpgadataflow.replace_verilog_relpaths
                param_customop = getCustomOp(param_node)
                ipgen_path = Path(cast("str", param_customop.get_nodeattr("ipgen_path")))
                if ipgen_path.is_dir():
                    threshold_path = f"{path}/Thresholding_id_{i + 1}_"
                    pattern = re.compile(
                        r'^(\s*parameter\s+THRESHOLDS_PATH\s*=\s*")[^"]+(".*)$',
                        re.MULTILINE,
                    )
                    for fpath in ipgen_path.rglob("*.v"):
                        s = fpath.read_text()
                        updated, n = pattern.subn(
                            lambda m, threshold_path=threshold_path: (
                                f"{m.group(1)}{threshold_path}{m.group(2)}"
                            ),
                            s,
                            count=1,
                        )
                        if n:
                            fpath.write_text(updated)

    def _generate_params_requant(
        self,
        model: ModelWrapper,
        path: str | Path,
        loop_node: "NodeProto",
        loop_body: ModelWrapper,
        param_node: "NodeProto",
        fpgapart: str | None,
    ) -> None:
        """Generate per-iteration memstream init files for a Requant_rtl body node.

        Requant has two parameter inputs (scale=input[1], bias=input[2]) whose
        fixed-point decomposition is *coupled*: the bias alignment depends on
        the scale-derived per-channel tap of the SAME iteration. Therefore both
        initializers must be set together for each iteration before calling
        ``generate_params``.

        Each iteration emits a single ``params_memblock.dat`` (unified struct
        packing scale+tap+bias per PE lane); these are renamed per iteration,
        then concatenated (set i at offset i*DEPTH) into
        ``params_memblock_id_{gid}.dat``. The single memstream wrapper's
        ``$INIT_FILE$`` path is rewritten to point at the concatenated file.

        Args:
            model: The model containing this loop node (holds the stacked parameters).
            path: Directory the parameter files are written to.
            loop_node: The ONNX node of this loop.
            loop_body: The loop body model.
            param_node: The Requant_rtl node in the loop body.
            fpgapart: Target FPGA part, forwarded to the Requant parameter generation.
        """
        iteration = self.iteration
        inst = getHWCustomOp(param_node)

        # Map the Requant's scale/bias tensors to loop-body graph-input ids.
        graph_input_names = [x.name for x in loop_body.graph.input]
        scale_tensor = param_node.input[1]
        bias_tensor = param_node.input[2]
        scale_gid = graph_input_names.index(scale_tensor)

        # Stacked per-iteration params live on the matching FINNLoop inputs
        # (loop_node.input[k] corresponds positionally to graph.input[k]).
        scale_stack = model.get_initializer(loop_node.input[scale_gid])
        bias_gid = graph_input_names.index(bias_tensor)
        bias_stack = model.get_initializer(loop_node.input[bias_gid])
        scale_dtype = model.get_tensor_datatype(loop_node.input[scale_gid])
        bias_dtype = model.get_tensor_datatype(loop_node.input[bias_gid])
        if not isinstance(scale_stack, np.ndarray) or scale_stack.shape[0] != iteration:
            raise FINNUserError(
                f"Expected Requant scale parameter of {param_node.name} to be an initializer "
                f"with first dimension equal to iteration count {iteration}."
            )
        if not isinstance(bias_stack, np.ndarray) or bias_stack.shape[0] != iteration:
            raise FINNUserError(
                f"Expected Requant bias parameter of {param_node.name} to be an initializer "
                f"with first dimension equal to iteration count {iteration}."
            )

        for it in range(iteration):
            # Set BOTH initializers for this iteration (coupled decomposition).
            loop_body.set_initializer(scale_tensor, scale_stack[it])
            loop_body.set_tensor_datatype(scale_tensor, scale_dtype)
            loop_body.set_initializer(bias_tensor, bias_stack[it])
            loop_body.set_tensor_datatype(bias_tensor, bias_dtype)
            inst.generate_params(loop_body, path, fpgapart)
            src = f"{path}/params_memblock.dat"
            dst = f"{path}/params_memblock_{it}.dat"
            shutil.move(src, dst)

        # Concatenate per-iteration files into one memblock for the param stream.
        # Use scale_gid as the canonical graph-input id for the unified stream.
        concat_file = f"{path}/params_memblock_id_{scale_gid}.dat"
        with Path(concat_file).open("w") as outfile:
            for it in range(iteration):
                memblock_file = Path(path) / f"params_memblock_{it}.dat"
                with memblock_file.open("r") as infile:
                    for line in infile:
                        outfile.write(line)
                memblock_file.unlink()

        # Rewrite the memstream wrapper's $INIT_FILE$ path (Elementwise-style).
        ipgen_path = inst.get_nodeattr("code_gen_dir_ipgen")
        if ipgen_path is not None and Path(cast("str", ipgen_path)).is_dir():
            old = str(Path(cast("str", ipgen_path)) / "params_memblock.dat")
            for fpath in Path(cast("str", ipgen_path)).rglob("*_params_memstream_wrapper.v"):
                s = fpath.read_text()
                s = s.replace(old, concat_file)
                fpath.write_text(s)

    def generate_hdl_stream_tap(self) -> None:
        """Generate verilog code for the stream tap components."""
        template_path = (
            Path(get_settings().finn_rtllib)
            / "stream_tap"
            / "hdl"
            / "stream_tap_wrapper_template.v"
        )
        code_gen_dir = self.code_gen_dir_ipgen
        iteration = self.iteration
        loop_body = self.body
        graph_inputs = [x.name for x in loop_body.graph.input]
        # TODO check if this needs to be padded
        data_width = DataType.get_smallest_possible(iteration).bitwidth()
        # pad to nearest multiple of 8
        data_width = roundup_to_integer_multiple(data_width, 8)
        for node in loop_body.graph.node:
            node_inst = cast("HWCustomOp", getCustomOp(node))
            if node_inst.get_nodeattr("mlo_max_iter"):
                # calculate TAP_REP
                # for Thresholds this value is fm size / pe
                # for all other param nodes it is 1
                tap_rep = 1
                if node.op_type == "Thresholding_rtl":
                    tap_rep = np.prod(node_inst.get_folded_input_shape(0)[:-1])
                elif node.op_type.startswith("Elementwise") and hasattr(
                    node_inst, "calc_wmem_reps"
                ):
                    tap_rep = cast("ElementwiseBinary_rtl", node_inst).calc_wmem_reps()
                elif node.op_type == "Requant_rtl":
                    # The decoupled Requant core consumes one packed param word
                    # per input beat, cycling through CF stored words (memstream
                    # DEPTH=CF). The set index must be replayed once per spatial
                    # position to re-stream the CF-word block.
                    # folded_input_shape is [*numInputVectors, CF, PE].
                    tap_rep = int(np.prod(node_inst.get_folded_input_shape(0)[:-2]))
                # Emit one stream tap per parameter graph-input of this node.
                # Single-param ops (MVAU/Thresholding/Elementwise) yield one
                # tap. Requant packs scale+bias into a single unified memstream
                # driven by the tap for input[1]; input[2] (bias) is internal
                # to the struct and does not get its own tap.
                param_inputs = node.input[1:]
                if node.op_type == "Requant_rtl":
                    param_inputs = node.input[1:2]
                for pinp in param_inputs:
                    if pinp not in graph_inputs:
                        continue
                    stname = f"IN_{graph_inputs.index(pinp)}"
                    code_gen_dict = {
                        "$MODULE_NAME$": [stname],
                        "$DATA_WIDTH$": [str(data_width)],
                        "$TAP_REP$": [str(tap_rep)],
                    }
                    # apply code generation to template
                    template_wrapper = template_path.read_text()
                    for key, lines in code_gen_dict.items():
                        # transform list into long string separated by '\n'
                        template_wrapper = template_wrapper.replace(key, "\n".join(lines))
                    (Path(code_gen_dir) / f"{stname}_stream_tap_wrapper.v").write_text(
                        template_wrapper
                    )

    def ipgen_singlenode_code(self, fpgapart: str | None = None) -> None:
        """Build the Vivado block design and package the loop as a single IP."""
        prjname = "MakeLoopIP"
        block_name = self.onnx_node.name
        vivado_stitch_proj_dir = self.code_gen_dir_ipgen

        cmd = []

        # add all the generated IP dirs to ip_repo_paths
        ip_dirs = ["list"]
        # add RTL streamer IP
        ip_dirs.append(str(Path(get_settings().finn_rtllib) / "memstream"))
        loop_model = self.body
        for node in loop_model.graph.node:
            node_inst = getCustomOp(node)
            ip_dir_value = node_inst.get_nodeattr("ip_path")
            if not Path(cast("str", ip_dir_value)).is_dir():
                raise FINNInternalError("IP generation directory doesn't exist.")
            ip_dirs += [cast("str", ip_dir_value)]
        ip_dirs_str = " ".join(ip_dirs)
        cmd.append(f"set_property ip_repo_paths [{ip_dirs_str}] [current_project]")
        cmd.append("update_ip_catalog")

        # create and instantiate FINNLoop node overarching block design
        cmd.append(f"create_bd_design {self.onnx_node.name}_bd_design")
        cmd.append(f"create_bd_cell -type hier {self.onnx_node.name}")
        clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
        rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
        # clock and reset
        cmd.append(f"create_bd_pin -dir I -type clk /{self.onnx_node.name}/{clk_name}")
        cmd.append(f"create_bd_pin -dir I -type rst /{self.onnx_node.name}/{rst_name}")
        # interfaces
        node_intf = self.get_verilog_top_module_intf_names()
        m_axis_intfs = node_intf["m_axis"]
        s_axis_intfs = node_intf["s_axis"]
        mm_intfs = node_intf["aximm"]
        for intf in m_axis_intfs:
            cmd.append(
                "create_bd_intf_pin -mode Master "
                f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{self.onnx_node.name}/{intf[0]}"
            )
        for intf in s_axis_intfs:
            cmd.append(
                "create_bd_intf_pin -mode Slave "
                f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{self.onnx_node.name}/{intf[0]}"
            )
        for intf in mm_intfs:
            cmd.append(
                "create_bd_intf_pin -mode Master "
                f"-vlnv xilinx.com:interface:aximm_rtl:1.0 /{self.onnx_node.name}/{intf[0]}"
            )
        # instantiate loop shell
        loop_shell_name = f"{self.onnx_node.name}/{self.onnx_node.name}_loop_cont_wrapper"
        cmd.append(
            f"""create_bd_cell -type module -reference \
            {self.onnx_node.name}_loop_cont_wrapper {loop_shell_name}"""
        )
        # connect loop shell to clk and reset
        cmd.append(
            f"connect_bd_net [get_bd_pins {self.onnx_node.name}/{rst_name}] "
            f"[get_bd_pins {loop_shell_name}/{rst_name}]"
        )
        cmd.append(
            f"connect_bd_net [get_bd_pins {self.onnx_node.name}/{clk_name}] "
            f"[get_bd_pins {loop_shell_name}/{clk_name}]"
        )
        # "externalize" some of the loop shell signals
        ext_intf_signals = ["in0_V", "out0_V", "m_axi_intermediate_frame"]
        for sig in ext_intf_signals:
            cmd.append(
                f"connect_bd_intf_net [get_bd_intf_pins {self.onnx_node.name}/{sig}] "
                f"[get_bd_intf_pins {loop_shell_name}/{sig}]"
            )
        # stream tap graph generation
        loop_body = self.body
        code_gen_dir = self.code_gen_dir_ipgen
        # create a hierarchy for this layer, with the same port names
        stg_intf = {}
        stg_intf["clk"] = self.get_verilog_top_module_intf_names()["clk"]
        stg_intf["rst"] = self.get_verilog_top_module_intf_names()["rst"]
        bd_name = f"{self.onnx_node.name}/stream_tap_graph"
        cmd.append(f"create_bd_cell -type hier {bd_name}")
        # clock and reset
        cmd.append(f"create_bd_pin -dir I -type clk /{bd_name}/{clk_name}")
        cmd.append(f"create_bd_pin -dir I -type rst /{bd_name}/{rst_name}")
        # streams
        cmd.append(
            "create_bd_intf_pin -mode Master "
            f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{bd_name}/m_axis_0"
        )
        cmd.append(
            f"connect_bd_intf_net [get_bd_intf_pins {bd_name}/m_axis_0] "
            f"[get_bd_intf_pins {loop_shell_name}/s_axis_core_out_fw_idx]"
        )

        cmd.append(
            "create_bd_intf_pin -mode Slave "
            f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{bd_name}/s_axis_0"
        )
        # One tap per parameter graph-input, mirroring generate_hdl_stream_tap():
        # Requant packs scale (input[1]) and bias (input[2]) into one memstream
        # driven by the scale tap, so the bias gets no tap of its own.
        taps: list[tuple[int, str, str]] = []
        node_to_taps: dict[str, list[int]] = {}
        for gid, inp in enumerate(loop_body.graph.input[1:], start=1):
            consumer = loop_body.find_consumer(inp.name)
            if consumer is None:
                raise FINNInternalError(f"No consumer found for loop input {inp.name}")
            if consumer.op_type == "Requant_rtl" and inp.name in consumer.input[2:]:
                continue
            taps.append((gid, f"IN_{gid}_stream_tap_wrapper", consumer.name))
            node_to_taps.setdefault(consumer.name, []).append(gid)
        # number the master ports densely, to match the loop body IP's s_axis_1..N
        for pid in range(1, len(taps) + 1):
            cmd.append(
                "create_bd_intf_pin -mode Master "
                f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{bd_name}/m_axis_{pid}"
            )
        # get stream tap (+ skid)  components
        skid_file = str(Path(get_settings().finn_rtllib) / "skid" / "skid.sv")
        stream_tap_dir = Path(get_settings().finn_rtllib) / "stream_tap" / "hdl"
        file_suffix = "_stream_tap_wrapper.v"
        # automatically find stream tap verilog components in code generation directory
        st_tmpl_names = []
        st_verilog_files = []
        for entry in Path(code_gen_dir).iterdir():
            fname = entry.name
            if fname.endswith(file_suffix):
                st_verilog_files.append(str(entry))
                st_tmpl_names.append(fname[:-2])
        dwc_rtllib_dir = Path(get_settings().finn_rtllib) / "dwc" / "hdl"
        sourcefiles = [
            *st_verilog_files,
            str(stream_tap_dir / "stream_tap.sv"),
            skid_file,
            str(dwc_rtllib_dir / "vpc.sv"),
        ]
        for f in sourcefiles:
            cmd += [f"add_files -norecurse {f}"]

        adj_list = adjacency_list(
            loop_body,
            lambda node: (
                node.op_type == "Thresholding_rtl"
                and any(attr.name == "mlo_max_iter" and attr.i > 0 for attr in node.attribute)
            )
            or (
                node.op_type == "MVAU_rtl"
                and any(attr.name == "mlo_max_iter" and attr.i > 0 for attr in node.attribute)
            )
            or (
                node.op_type.startswith("Elementwise")
                and any(attr.name == "mlo_max_iter" and attr.i > 0 for attr in node.attribute)
            )
            or (
                node.op_type == "Requant_rtl"
                and any(attr.name == "mlo_max_iter" and attr.i > 0 for attr in node.attribute)
            ),
        )

        def node_entry(node_name: str) -> str:
            """Return the entry tap (lowest gid) of a node.

            This is where the set index enters the node's param taps. A multi-param
            node chains its taps internally (below), so the index flows in here and
            out of ``node_exit``.

            Args:
                node_name: Name of the parameter node in the loop body.

            Returns:
                The name of the node's entry stream tap.
            """
            return f"IN_{min(node_to_taps[node_name])}_stream_tap_wrapper"

        def node_exit(node_name: str) -> str:
            """Return the exit tap (highest gid) of a node.

            This tap forwards the set index onward to the next node in the chain.
            For a single-param node entry == exit.

            Args:
                node_name: Name of the parameter node in the loop body.

            Returns:
                The name of the node's exit stream tap.
            """
            return f"IN_{max(node_to_taps[node_name])}_stream_tap_wrapper"

        def dst_entry_taps(dsts: list[str]) -> list[str]:
            """Return one entry tap per destination node.

            Intra-node params (e.g. Requant scale+bias) are chained in series
            internally, so a multi-param node contributes exactly one sink here --
            fork duplication only applies across distinct destination nodes
            (branching dataflow).

            Args:
                dsts: Names of the destination nodes.

            Returns:
                The names of the destination nodes' entry stream taps.
            """
            return [node_entry(d) for d in dsts]

        # instantiate all stream taps and connect their clk and rst
        for pid, (_gid, tap_name, _consumer) in enumerate(taps, start=1):
            cmd.append(f"create_bd_cell -type hier -reference {tap_name} /{bd_name}/{tap_name}")
            # connect
            cmd.append(
                f"connect_bd_net [get_bd_pins {bd_name}/{clk_name}] "
                f"[get_bd_pins {bd_name}/{tap_name}/ap_clk]"
            )
            cmd.append(
                f"connect_bd_net [get_bd_pins {bd_name}/{rst_name}] "
                f"[get_bd_pins {bd_name}/{tap_name}/ap_rst_n]"
            )
            cmd.append(
                f"connect_bd_intf_net [get_bd_intf_pins {bd_name}/m_axis_{pid}] "
                f"[get_bd_intf_pins {bd_name}/{tap_name}/m_axis_1]"
            )

        # Chain a multi-param node's taps in series (e.g. Requant scale -> bias):
        # each tap taps off its own m_axis_1 to its memstream and forwards the
        # set index on m_axis_0 to the next param tap. This keeps every tap's
        # forward output consumed (the stream_tap stalls if m_axis_0/ordy is left
        # dangling) while still delivering the same index to all param memstreams.
        for gids in node_to_taps.values():
            for a, b in itertools.pairwise(sorted(gids)):
                cmd.append(
                    "connect_bd_intf_net "
                    f"[get_bd_intf_pins {bd_name}/IN_{a}_stream_tap_wrapper/m_axis_0] "
                    f"[get_bd_intf_pins {bd_name}/IN_{b}_stream_tap_wrapper/s_axis_0]"
                )

        # prune adj_list to remove join duplicates
        pruned_adj_list = copy.deepcopy(adj_list)

        for key in adj_list:
            if key.startswith("__INPUT") and "INPUT0" not in key:
                del pruned_adj_list[key]
            if "__OUTPUT0__" in adj_list[key] and len(adj_list[key]) > 1:
                pruned_adj_list[key].remove("__OUTPUT0__")

        pruned_adj_list = {tuple(v): k for k, v in pruned_adj_list.items()}  # exchange keys, values
        pruned_adj_list = {v: list(k) for k, v in pruned_adj_list.items()}

        # look for double edges,
        # e.g. input connected to node_x and intermediate node connected to node_x

        pruned_adj_list_copy = copy.deepcopy(pruned_adj_list)

        for key0, value0 in pruned_adj_list_copy.items():
            for key1, value1 in pruned_adj_list_copy.items():
                for val in value1:
                    if val in value0 and key0 != key1:
                        # check which src is in the topological order last
                        # key0
                        node0 = loop_body.get_node_from_name(key0)
                        idx0 = loop_body.get_node_index(node0) if node0 is not None else None
                        id0 = idx0 if idx0 is not None else -1
                        # key1
                        node1 = loop_body.get_node_from_name(key1)
                        idx1 = loop_body.get_node_index(node1) if node1 is not None else None
                        id1 = idx1 if idx1 is not None else -1
                        # if node0 is earlier in the graph remove val from list
                        if id0 < id1:
                            pruned_adj_list[key0].remove(val)

        # filter pruned_adj_list in case some of the values are now empty lists
        pruned_adj_list = {key: value for key, value in pruned_adj_list.items() if value != []}

        # create stg
        for src, dsts in pruned_adj_list.items():
            if all(x.startswith("__OUTPUT") for x in dsts):
                continue
            if "__INPUT0__" in src:
                src_inst_name = bd_name
                src_intf_name = "s_axis_0"
            else:
                # A multi-param node forwards the set index from its exit (last)
                # tap; its internal params were already chained in series above.
                src_inst_name = bd_name + "/" + node_exit(src)
                src_intf_name = "m_axis_0"

            # One sink per destination node (its entry tap). A multi-param node
            # is a single sink here -- fork duplication is only for branching to
            # distinct downstream nodes.
            sink_taps = dst_entry_taps(dsts)
            dst_intf_name = "s_axis_0"
            if len(sink_taps) == 1:
                dst_inst_name = sink_taps[0]
                cmd.append(
                    f"connect_bd_intf_net [get_bd_intf_pins {src_inst_name}/{src_intf_name}] "
                    f"[get_bd_intf_pins {bd_name}/{dst_inst_name}/{dst_intf_name}]"
                )
            # if node is a fork connect data signals directly
            # and insert AND logic for rdy and vld signals
            elif len(sink_taps) > 1:
                if "__INPUT0__" in src:
                    cmd.append(
                        "create_bd_cell -type ip "
                        "-vlnv xilinx.com:ip:axis_broadcaster:1.1 "
                        f"{src_inst_name}/axi_broadcaster_0"
                    )
                    cmd.append(
                        f"set_property CONFIG.NUM_MI {{{len(sink_taps)}}} "
                        f"[get_bd_cells {src_inst_name}/axi_broadcaster_0]"
                    )
                    # connect component to clk, rst and input
                    cmd.append(
                        "connect_bd_net "
                        f"[get_bd_pins {src_inst_name}/axi_broadcaster_0/aresetn] "
                        f"[get_bd_pins {bd_name}/{rst_name}]"
                    )
                    cmd.append(
                        "connect_bd_net "
                        f"[get_bd_pins {src_inst_name}/axi_broadcaster_0/aclk] "
                        f"[get_bd_pins {bd_name}/{clk_name}]"
                    )
                    cmd.append(
                        "connect_bd_intf_net "
                        f"[get_bd_intf_pins {src_inst_name}/s_axis_0] "
                        f"[get_bd_intf_pins {src_inst_name}/axi_broadcaster_0/S_AXIS]"
                    )
                    for idx, dst_inst_name in enumerate(sink_taps):
                        cmd.append(
                            "connect_bd_intf_net "
                            f"[get_bd_intf_pins {src_inst_name}/axi_broadcaster_0/M0{idx}_AXIS] "
                            f"[get_bd_intf_pins {src_inst_name}/{dst_inst_name}/{dst_intf_name}]"
                        )
                else:
                    for idx, dst_inst_name in enumerate(sink_taps):
                        cmd.append(
                            "connect_bd_net "
                            f"[get_bd_pins {src_inst_name}/{src_intf_name}_TDATA] "
                            f"[get_bd_pins {bd_name}/{dst_inst_name}/{dst_intf_name}_TDATA]"
                        )
                        cmd.append(
                            "create_bd_cell -type ip "
                            f"-vlnv xilinx.com:ip:util_vector_logic:2.0 "
                            f"{src_inst_name}_util_vector_logic_{idx}"
                        )
                        cmd.append(
                            "set_property CONFIG.C_SIZE {1} "
                            f"[get_bd_cells {src_inst_name}_util_vector_logic_{idx}]"
                        )
                        if idx == 0:
                            cmd.append(
                                "connect_bd_net "
                                f"[get_bd_pins {src_inst_name}/{src_intf_name}_TVALID] "
                                f"[get_bd_pins {src_inst_name}_util_vector_logic_{idx}/Op1]"
                            )
                        elif idx < len(sink_taps):
                            cmd.append(
                                "connect_bd_net "
                                f"[get_bd_pins {src_inst_name}_util_vector_logic_{idx - 1}/Res] "
                                f"[get_bd_pins {src_inst_name}_util_vector_logic_{idx}/Op1]"
                            )

                        cmd.append(
                            "connect_bd_net "
                            f"[get_bd_pins {bd_name}/{dst_inst_name}/{dst_intf_name}_TREADY] "
                            f"[get_bd_pins {src_inst_name}_util_vector_logic_{idx}/Op2]"
                        )

                    last_logic = f"{src_inst_name}_util_vector_logic_{len(sink_taps) - 1}"
                    cmd.append(
                        "connect_bd_net "
                        f"[get_bd_pins {last_logic}/Res] "
                        f"[get_bd_pins {src_inst_name}/{src_intf_name}_TREADY]"
                    )
                    for dst_inst_name in sink_taps:
                        dst_intf_name = "s_axis_0"
                        cmd.append(
                            "connect_bd_net "
                            f"[get_bd_pins {last_logic}/Res] "
                            f"[get_bd_pins {bd_name}/{dst_inst_name}/{dst_intf_name}_TVALID]"
                        )
        # connect output of stream tap graph
        last_nodes = [
            key
            for key, value in adj_list.items()
            if all(x.startswith("__OUTPUT0__") for x in value)
        ]
        cmd.append(
            f"connect_bd_intf_net [get_bd_intf_pins {bd_name}/m_axis_0] "
            f"[get_bd_intf_pins {bd_name}/{node_exit(last_nodes[0])}/m_axis_0]"
        )

        # connect stream tap graph to clk and reset
        cmd.append(
            f"connect_bd_net [get_bd_pins {self.onnx_node.name}/{rst_name}] "
            f"[get_bd_pins {bd_name}/{rst_name}]"
        )
        cmd.append(
            f"connect_bd_net [get_bd_pins {self.onnx_node.name}/{clk_name}] "
            f"[get_bd_pins {bd_name}/{clk_name}]"
        )

        loop_body_ipstitch_path = cast("str", loop_body.get_metadata_prop("vivado_stitch_proj"))
        loop_body_vlnv = loop_body.get_metadata_prop("vivado_stitch_vlnv")
        loop_body_intf_names = eval(
            cast("str", loop_body.get_metadata_prop("vivado_stitch_ifnames"))
        )
        ip_dirs = ["list"]
        ip_dirs += collect_ip_dirs(loop_body, loop_body_ipstitch_path)
        ip_dirs_str = f"[{' '.join(ip_dirs)}]"
        cmd.append(
            "set_property ip_repo_paths "
            f"[concat [get_property ip_repo_paths [current_project]] {ip_dirs_str}] "
            "[current_project]"
        )
        cmd.append("update_ip_catalog -rebuild -scan_changes")
        finn_ip_name = f"{self.onnx_node.name}/finn_design_mlo"
        cmd.append(f"create_bd_cell -type ip -vlnv {loop_body_vlnv} {finn_ip_name}")
        # connect finn ip to clk and reset
        cmd.append(
            f"connect_bd_net [get_bd_pins {self.onnx_node.name}/{rst_name}] "
            f"[get_bd_pins {finn_ip_name}/{rst_name}]"
        )
        cmd.append(
            f"connect_bd_net [get_bd_pins {self.onnx_node.name}/{clk_name}] "
            f"[get_bd_pins {finn_ip_name}/{clk_name}]"
        )
        # Expose the loop body's sim_finish control to the top of the FINNLoop IP.
        # The body's stitched IP carries a sim_ctrl (inserted by CreateStitchedIP)
        # whose sim_finish input triggers $finish. Asserting it during rtlsim runs
        # the SystemVerilog final blocks that flush and close the fifo_gauge log
        # files; without this the gauge logs are left unflushed/empty during
        # characterization-based FIFO sizing.
        cmd.append(f"create_bd_pin -dir I /{self.onnx_node.name}/sim_finish")
        cmd.append(
            f"connect_bd_net [get_bd_pins {self.onnx_node.name}/sim_finish] "
            f"[get_bd_pins {finn_ip_name}/sim_finish]"
        )

        if self.has_address_offset:
            ac_module_name = self.onnx_node.name + "_address_config_wrapper"
            ac_inst_name = f"{self.onnx_node.name}/address_config"
            cmd.append(f"add_files -norecurse {code_gen_dir}/{ac_module_name}.v")
            cmd.append(f"create_bd_cell -type module -reference {ac_module_name} {ac_inst_name}")
            cmd.append(
                f"connect_bd_net [get_bd_pins {self.onnx_node.name}/{rst_name}] "
                f"[get_bd_pins {ac_inst_name}/{rst_name}]"
            )
            cmd.append(
                f"connect_bd_net [get_bd_pins {self.onnx_node.name}/{clk_name}] "
                f"[get_bd_pins {ac_inst_name}/{clk_name}]"
            )
            cmd.append(
                f"connect_bd_net [get_bd_pins {ac_inst_name}/base_address] "
                f"[get_bd_pins {finn_ip_name}/base_address]"
            )
            cmd.append(
                f"connect_bd_net [get_bd_pins {ac_inst_name}/base_address] "
                f"[get_bd_pins {loop_shell_name}/base_address]"
            )
            cmd.append(
                "create_bd_intf_pin -mode Slave "
                f"-vlnv xilinx.com:interface:aximm_rtl:1.0 /{self.onnx_node.name}/s_axilite"
            )
            cmd.append(
                f"connect_bd_intf_net [get_bd_intf_pins {self.onnx_node.name}/s_axilite] "
                f"[get_bd_intf_pins {ac_inst_name}/s_axilite]"
            )

        # "externalize" some of the loop shell signals
        ext_signals = loop_body_intf_names["aximm"]
        for sig in ext_signals:
            cmd.append(
                f"connect_bd_intf_net [get_bd_intf_pins {self.onnx_node.name}/{sig[0]}] "
                f"[get_bd_intf_pins {finn_ip_name}/{sig[0]}]"
            )
        # connect components with each other
        # stream tap with finn ip
        connect_signals = loop_body_intf_names["s_axis"]
        if len(taps) != len(connect_signals) - 1:
            raise FINNInternalError("stream tap / loop body param mismatch")
        for idx, _sig in enumerate(connect_signals[:-1]):
            cmd.append(
                "connect_bd_intf_net "
                f"[get_bd_intf_pins {bd_name}/m_axis_{idx + 1}] "
                f"[get_bd_intf_pins {finn_ip_name}/s_axis_{idx + 1}]"
            )
        # connect stream tap with loop wrapper
        cmd.append(
            "connect_bd_intf_net "
            f"[get_bd_intf_pins {bd_name}/s_axis_0] "
            f"[get_bd_intf_pins {loop_shell_name}/m_axis_core_in_fw_idx]"
        )
        # connect loop wrapper with finn ip
        cmd.append(
            "connect_bd_intf_net "
            f"[get_bd_intf_pins {loop_shell_name}/m_axis_core_in] "
            f"[get_bd_intf_pins {finn_ip_name}/s_axis_0]"
        )
        cmd.append(
            "connect_bd_intf_net "
            f"[get_bd_intf_pins {finn_ip_name}/m_axis_0] "
            f"[get_bd_intf_pins {loop_shell_name}/s_axis_core_out]"
        )
        cmd.append(f"make_bd_pins_external  [get_bd_cells {block_name}]")
        cmd.append(f"make_bd_intf_pins_external  [get_bd_cells {block_name}]")
        cmd.append("set_property name in0_V [get_bd_intf_ports in0_V_0]")
        cmd.append("set_property name ap_clk [get_bd_ports ap_clk_0]")
        cmd.append("set_property name ap_rst_n [get_bd_ports ap_rst_n_0]")
        cmd.append("set_property name out0_V [get_bd_intf_ports out0_V_0]")
        cmd.append(
            "set_property name m_axi_intermediate_frame "
            "[get_bd_intf_ports m_axi_intermediate_frame_0]"
        )
        cmd.append("set_property name sim_finish [get_bd_ports sim_finish_0]")
        if self.has_address_offset:
            cmd.append("set_property name s_axilite [get_bd_intf_ports s_axilite_0]")
        # set property name for aximm interfaces
        ext_signals = loop_body_intf_names["aximm"]
        for sig in ext_signals:
            cmd.append(f"set_property name {sig[0]} [get_bd_intf_ports {sig[0]}_0]")
        cmd.append("save_bd_design")
        # cmd.append("validate_bd_design")
        # cmd.append("save_bd_design")
        # create wrapper hdl (for rtlsim later on)
        bd_subpath = f"{prjname}.srcs/sources_1/bd/{block_name}_bd_design"
        bd_base = f"{vivado_stitch_proj_dir}/{bd_subpath}"
        bd_filename = f"{bd_base}/{block_name}_bd_design.bd"
        cmd.append(f"make_wrapper -files [get_files {bd_filename}] -top")
        wrapper_subpath = f"{prjname}.gen/sources_1/bd/{block_name}_bd_design"
        wrapper_base = f"{vivado_stitch_proj_dir}/{wrapper_subpath}"
        wrapper_filename = f"{wrapper_base}/hdl/{block_name}_bd_design_wrapper.v"
        cmd.append(f"add_files -norecurse {wrapper_filename}")
        cmd.append(f"set_property top {block_name}_bd_design_wrapper [current_fileset]")

        # export block design itself as an IP core
        block_vendor = "xilinx_finn"
        block_library = "finn"
        block_vlnv = f"{block_vendor}:{block_library}:{block_name}_bd_design:1.0"
        cmd.append(
            f"ipx::package_project -root_dir {vivado_stitch_proj_dir}/ip -vendor {block_vendor} "
            f"-library {block_library} -taxonomy /UserIP "
            f"-module {block_name}_bd_design -import_files"
        )
        # Allow user to customize clock in deployment of stitched IP
        cmd.append("set_property ipi_drc {ignore_freq_hz true} [ipx::current_core]")
        # in some cases, the IP packager seems to infer an aperture of 64K or 4G,
        # preventing address assignment of the DDR_LOW and/or DDR_HIGH segments
        # the following is a hotfix to remove this aperture during IODMA packaging
        # Also used for MLO in the context of Zynq
        loop_aximm_names = ["m_axi_intermediate_frame"] + [sig[0] for sig in ext_signals]
        for aximm_name in loop_aximm_names:
            cmd.append(
                f"ipx::remove_segment -quiet {aximm_name}:APERTURE_0 "
                f"[ipx::get_address_spaces {aximm_name} -of_objects [ipx::current_core]]"
            )
        cmd.append(f"set_property core_revision 2 [ipx::find_open_core {block_vlnv}]")
        cmd.append(f"ipx::create_xgui_files [ipx::find_open_core {block_vlnv}]")
        # mark bus interface params as user-resolvable to avoid FREQ_MHZ mismatches
        cmd.append(
            "set_property value_resolve_type user [ipx::get_bus_parameters "
            "-of [ipx::get_bus_interfaces -of [ipx::current_core ]]]"
        )

        template = load_codegen_template("mlo_loop_ip.tcl")

        # transform list into long string separated by '\n'
        cmd_str = "\n".join(cmd)
        template = template.replace("@IP_GEN@", cmd_str)
        template = template.replace("@PRJNAME@", prjname)
        template = template.replace("@PRJFOLDER@", vivado_stitch_proj_dir)
        template = template.replace("@FPGAPART@", cast("str", fpgapart))
        template = template.replace(
            "@TOP_VERILOG_FILE@",
            f"{self.code_gen_dir_ipgen}/{self.onnx_node.name}_wrapper.v",
        )
        (Path(vivado_stitch_proj_dir) / "make_loop_ip.tcl").write_text(template)

        # create a shell script and call Vivado
        make_project_sh = vivado_stitch_proj_dir + "/make_loop_ip.sh"
        working_dir = os.environ["PWD"]
        vivado_cmd = resolve_xilinx_tool("vivado")
        with Path(make_project_sh).open("w") as f:
            f.write("#!/bin/bash \n")
            f.write(f"cd {vivado_stitch_proj_dir}\n")
            f.write(f"{vivado_cmd} -mode batch -source make_loop_ip.tcl\n")
            f.write(f"cd {working_dir}\n")
        bash_command = ["bash", make_project_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        if not Path(wrapper_filename).is_file():
            # Vivado's own log is the only record of why the block design was not built;
            # quote its errors so that a failure is diagnosable from the exception alone
            # (e.g. from a CI job log, whose build directory is gone afterwards).
            vivado_log = Path(vivado_stitch_proj_dir) / "vivado.log"
            vivado_errors = ""
            if vivado_log.is_file():
                with vivado_log.open() as f:
                    error_lines = [ln.rstrip() for ln in f if ln.startswith(("ERROR", "CRITICAL"))]
                vivado_errors = "\n".join(error_lines[-20:])
            raise FINNInternalError(
                f"IPGen failed: {wrapper_filename} not found. Vivado log: {vivado_log}\n"
                f"{vivado_errors}"
            )
        self.set_nodeattr("ipgen_path", wrapper_filename)
        self.set_nodeattr("ip_path", vivado_stitch_proj_dir + "/ip")
        self.set_nodeattr("gen_top_module", f"{block_name}_bd_design_wrapper")
        self.set_nodeattr("ip_vlnv", block_vlnv)

    def get_verilog_top_module_intf_names(self) -> dict:
        """Return the top-level interface names grouped by protocol, including body AXI-MM ports."""
        # from wrapper template
        addr_bits = 64

        intf_names = {}
        intf_names["clk"] = ["ap_clk"]
        intf_names["rst"] = ["ap_rst_n"]

        intf_names["s_axis"] = []
        # AXI4S slave interface from outside loop to loop control externalize
        # to block diagram interface port and connect to fetch_start component
        intf_names["s_axis"].append(("in0_V", self.get_instream_width_padded(0)))

        intf_names["m_axis"] = []
        # AXI4S master interface to drive final loop output externalize
        # to block diagram interface port and connect to store_end component
        intf_names["m_axis"].append(("out0_V", self.get_outstream_width_padded(0)))

        intf_names["aximm"] = []
        # AXI4 master interface for intermediate buffering between layers
        intf_names["aximm"].append(["m_axi_intermediate_frame", str(addr_bits)])
        intf_names["axilite"] = ["s_axilite"] if self.has_address_offset else []

        intf_names["ap_none"] = []

        loop_body = self.body
        loop_body_intf = eval(cast("str", loop_body.get_metadata_prop("vivado_stitch_ifnames")))
        for intf in loop_body_intf["aximm"]:
            intf_names["aximm"].append(intf)

        return intf_names

    def code_generation_ipi(self) -> list[str]:
        """Return the TCL for node instantiation in Vivado IPI, adding IP repo paths."""
        vlnv = self.get_nodeattr("ip_vlnv")
        cmd = []
        # add all the generated IP dirs to ip_repo_paths
        ip_dirs = ["list"]
        # add RTL streamer IP
        loop_body = self.body
        loop_body_ipstitch_path = cast("str", loop_body.get_metadata_prop("vivado_stitch_proj"))
        ip_dirs += collect_ip_dirs(loop_body, loop_body_ipstitch_path)
        ip_dirs_str = " ".join(ip_dirs)
        cmd.append(
            "set_property ip_repo_paths "
            f"[concat [get_property ip_repo_paths [current_project]] {ip_dirs_str}] "
            "[current_project]"
        )
        cmd.append("update_ip_catalog -rebuild -scan_changes")
        cmd.append(f"create_bd_cell -type ip -vlnv {vlnv} {self.onnx_node.name}")
        return cmd

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:  # noqa: ARG002
        """Return no RTL files; the loop is packaged as a Vivado IP."""
        return []

    def intermediate_frame_bytes(self) -> int:
        """Return the DDR bytes reserved for intermediate frames.

        Returns:
            The number of outstanding DMAs times the (bus-aligned) frame size in bytes.
        """
        N_OUTSTANDING_DMAS = 128  # noqa: N806  # Currently hard-coded in intermediate_frames.sv
        DATA_BITS = 256  # noqa: N806  # DMA bus width, hard-coded in loop_control_wrapper.v
        input_elements = int(np.prod(self.get_normal_input_shape(0)))
        elem_bits = self.get_input_datatype(0).bitwidth()
        dma_pe = DATA_BITS // elem_bits
        dma_beats = (input_elements + dma_pe - 1) // dma_pe
        dma_beat_bytes = DATA_BITS // 8
        slot_bytes = dma_beats * dma_beat_bytes
        return N_OUTSTANDING_DMAS * slot_bytes

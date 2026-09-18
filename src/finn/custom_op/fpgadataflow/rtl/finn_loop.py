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
from typing import Any, cast

import finn.core.onnx_exec as oxe
import finn.xsi as finnxsi
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.custom_op.fpgadataflow.rtl import register_custom_op
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.templates import load_codegen_template
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.util.basic import getHWCustomOp, make_build_dir
from finn.util.create import adjacency_list
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.mlo_sim import mlo_prehook_func_factory
from finn.util.settings import get_settings

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
    """Collect and return the list of IP directories for a stitched model."""
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
        """Return the dictionary of node attributes for the FINN loop operator."""
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

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...] | list[int]:
        """Return the normal (unfolded) input shape at the given index."""
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
        """Return the normal (unfolded) output shape at the given index."""
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
        """Return the folded input shape at the given index."""
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
        """Return the folded output shape at the given index."""
        loop_body = self.body
        # get last node in loop body and return
        # normal output shape
        node = loop_body.graph.node[-1]
        inst = cast("HWCustomOp", getCustomOp(node))
        return cast("tuple[int, ...] | list[int]", inst.get_folded_output_shape(0))

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer the node datatype (no-op for the loop node)."""

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
        """Return the FINN DataType of the output at the given index."""
        return DataType[cast("str", self.get_nodeattr("outputDataType"))]

    def get_instream_width(self, ind: int = 0) -> int:
        """Return the input stream width in bits at the given index."""
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
        """Return the expected number of cycles for the loop node."""
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
        body_cycles = cast("int", loop_body.analysis(dataflow_performance)["critical_path_cycles"])
        overhead_per_iter = 40
        return (body_cycles + overhead_per_iter) * iteration

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return the output stream width in bits at the given index."""
        loop_body = self.body
        # get last node in loop body and return
        # normal output shape
        node = loop_body.graph.node[-1]
        inst = cast("HWCustomOp", getCustomOp(node))
        return inst.get_outstream_width(0)

    def get_number_output_values(self) -> int | dict[str, int]:
        """Return the number of expected output values from the loop node."""
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
        self.set_nodeattr("rtlsim_so", str(sim_base) + "/" + str(sim_rel))

    def execute_node(
        self, context: dict[str, np.ndarray], graph: GraphProto  # noqa: ARG002
    ) -> None:
        """Execute the loop node (rtlsim or per-iteration Python fallback)."""
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
                np.savez(iteration_context_path, **all_iteration_contexts)
        context[node.output[0]] = np.asarray(result, dtype=np.float32)

    def generate_hdl(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:  # noqa: ARG002
        """Generate the loop control wrapper HDL and the loop parameters."""
        code_gen_dir = self.code_gen_dir_ipgen
        self.generate_hdl_stream_tap()
        self.generate_params(model, code_gen_dir)
        code_gen_dict: dict[str, list[str]] = {}
        code_gen_dict["$LOOP_CONTROL_WRAPPER_NAME$"] = [f"{self.onnx_node.name}_loop_cont_wrapper"]
        code_gen_dict["$N_MAX_LAYERS$"] = [str(self.iteration)]
        code_gen_dict["$N_LAYERS$"] = [str(self.iteration)]
        code_gen_dict["$ILEN_BITS$"] = [str(self.get_instream_width(0))]
        code_gen_dict["$OLEN_BITS$"] = [str(self.get_outstream_width(0))]

        input_elements = np.prod(self.get_normal_input_shape(0))
        input_bytes = (input_elements * self.get_input_datatype(0).bitwidth() + 8 - 1) // 8
        output_elements = np.prod(self.get_normal_output_shape(0))
        output_bytes = (output_elements * self.get_output_datatype(0).bitwidth() + 8 - 1) // 8
        code_gen_dict["$INPUT_BYTES$"] = [str(input_bytes)]
        code_gen_dict["$OUTPUT_BYTES$"] = [str(output_bytes)]

        # round up to next power of 2
        input_bytes_rounded_to_power_of_2 = 2 ** (math.ceil(math.log2(input_bytes)))
        code_gen_dict["$LAYER_OFFS_INT$"] = [
            str(input_bytes_rounded_to_power_of_2)
        ]  # need to get correct value

        template_path = str(Path(get_settings().finn_rtllib) / "mlo" / "loop_control_wrapper.v")
        template_wrapper = Path(template_path).read_text()
        for key, lines in code_gen_dict.items():
            # transform list into long string separated by '\n'
            template_wrapper = template_wrapper.replace(key, "\n".join(lines))
        (Path(code_gen_dir) / f"{self.onnx_node.name}_wrapper.v").write_text(template_wrapper)

    def generate_params(self, model: ModelWrapper, path: str | Path) -> None:
        """Generate .dat files for loop parameters and concatenate them together."""
        iteration = self.iteration
        loop_node = self.onnx_node
        loop_body = self.body
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

            if param_node.op_type.startswith("MVAU") or param_node.op_type.startswith(
                "Elementwise"
            ):
                # concatinate all .dat files together
                param_file = Path(path) / f"memblock_{param_node.op_type}_id_{i + 1}.dat"
                with param_file.open("w") as outfile:
                    for it in range(iteration):
                        memblock_file = Path(path) / f"{param_node.op_type}_memblock_{it}.dat"
                        with memblock_file.open("r") as infile:
                            for line in infile:
                                outfile.write(line)
                        memblock_file.unlink()  # remove the per-iteration file after concatenation
                # Replace the path for the dat files in the ipgen files if Eltwise
                # Adapted from transformations.fpgadataflow.replace_verilog_relpaths
                if param_node.op_type.startswith("Elementwise"):
                    param_customop = getCustomOp(param_node)
                    ipgen_path_str = param_customop.get_nodeattr("code_gen_dir_ipgen")
                    ipgen_path = Path(cast("str", ipgen_path_str))
                    if ipgen_path.is_dir():
                        init_file = Path(path) / f"memblock_{param_node.op_type}_id_{i + 1}.dat"
                        pattern = re.compile(
                            r'^(\s*parameter\s+INIT_FILE\s*=\s*")[^"]+(".*)$',
                            re.MULTILINE,
                        )
                        for fpath in ipgen_path.rglob("*_memstream_wrapper.v"):
                            s = fpath.read_text()
                            updated, n = pattern.subn(
                                lambda m, init_file=init_file: (
                                    f"{m.group(1)}{init_file}{m.group(2)}"
                                ),
                                s,
                                count=1,
                            )
                            if n:
                                print(f"Updating INIT_FILE in {fpath} -> {init_file}")
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
                ipgen_p = cast("str", param_customop.get_nodeattr("ipgen_path"))
                ipgen_path = Path(ipgen_p)
                if ipgen_path.is_dir():
                    threshold_path = f"{path}/Thresholding_id_{i + 1}_"
                    pattern = re.compile(
                        r'^(\s*parameter\s+THRESHOLDS_PATH\s*=\s*")[^"]+(".*)$',
                        re.MULTILINE,
                    )
                    for fpath in ipgen_path.rglob("*.v"):
                        print(f"Checking {fpath} for THRESHOLDS_PATH to update...")
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

    def generate_hdl_stream_tap(self) -> None:
        """Generate verilog code for the stream tap components."""
        template_path = str(
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
                stname = f"IN_{graph_inputs.index(node.input[1])}"
                code_gen_dict = {
                    "$MODULE_NAME$": [stname],
                    "$DATA_WIDTH$": [str(data_width)],
                    "$TAP_REP$": [str(tap_rep)],
                }
                # apply code generation to template
                template_wrapper = Path(template_path).read_text()
                for key, lines in code_gen_dict.items():
                    # transform list into long string separated by '\n'
                    template_wrapper = template_wrapper.replace(key, "\n".join(lines))
                (Path(code_gen_dir) / f"{stname}_stream_tap_wrapper.v").write_text(template_wrapper)

    def ipgen_singlenode_code(self, fpgapart: str | None = None) -> None:
        """Generate and run the Vivado IP-integrator script for the loop node."""
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
        control_intfs = node_intf["ap_none"]
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
        for intf in control_intfs:
            if intf == "done_if":
                cmd.append(
                    "create_bd_pin -from 1 -to 0 -dir O -type data "
                    f"/{self.onnx_node.name}/{intf}"
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
        ext_intf_signals = ["in0_V", "out0_V", "m_axi_hbm"]
        ext_signals = ["done_if"]
        for sig in ext_intf_signals:
            cmd.append(
                f"connect_bd_intf_net [get_bd_intf_pins {self.onnx_node.name}/{sig}] "
                f"[get_bd_intf_pins {loop_shell_name}/{sig}]"
            )
        for sig in ext_signals:
            cmd.append(
                f"connect_bd_net [get_bd_pins {self.onnx_node.name}/{sig}] "
                f"[get_bd_pins {loop_shell_name}/{sig}]"
            )

        # stream tap graph generation
        loop_body = self.body
        source_target = f"./ip/verilog/rtl_ops/{self.onnx_node.name}"
        cmd.append(f"file mkdir {source_target}")
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
        for idx, _inp in enumerate(loop_body.graph.input[1:]):
            cmd.append(
                "create_bd_intf_pin -mode Master "
                f"-vlnv xilinx.com:interface:axis_rtl:1.0 /{bd_name}/m_axis_{idx + 1}"
            )
        # get stream tap (+ skid)  components
        skid_file = str(Path(get_settings().finn_rtllib) / "skid" / "skid.sv")
        stream_tap_dir = str(Path(get_settings().finn_rtllib) / "stream_tap" / "hdl")
        file_suffix = "_stream_tap_wrapper.v"
        # automatically find stream tap verilog components in code generation directory
        st_tmpl_names = []
        st_verilog_files = []
        for entry in Path(code_gen_dir).iterdir():
            fname = entry.name
            if fname.endswith(file_suffix):
                st_verilog_files.append(str(entry))
                st_tmpl_names.append(fname[:-2])
        sourcefiles = [
            *st_verilog_files,
            str(Path(stream_tap_dir) / "stream_tap.sv"),
            skid_file,
        ]
        for f in sourcefiles:
            cmd += [f"add_files -copy_to {source_target} -norecurse {f}"]

        adj_list = adjacency_list(
            loop_body,
            lambda node: (
                node.op_type == "Thresholding_rtl"
                or (
                    node.op_type == "MVAU_rtl"
                    and any(attr.name == "mlo_max_iter" and attr.i > 0 for attr in node.attribute)
                )
                or (
                    node.op_type.startswith("Elementwise")
                    and any(attr.name == "mlo_max_iter" and attr.i > 0 for attr in node.attribute)
                )
            ),
        )

        # create map that maps each stream tap to its param node
        st_map: dict[str, str] = {}
        for idx, inp in enumerate(loop_body.graph.input[1:]):
            consumer = loop_body.find_consumer(inp.name)
            if consumer is None:
                raise FINNInternalError(f"No consumer found for loop input {inp.name}")
            st_map[consumer.name] = f"IN_{idx + 1}_stream_tap_wrapper"

        # instantiate all stream taps and connect their clk and rst
        for idx, st_name in enumerate(st_map.values()):
            cmd.append(f"create_bd_cell -type hier -reference {st_name} /{bd_name}/{st_name}")
            # connect
            cmd.append(
                f"connect_bd_net [get_bd_pins {bd_name}/{clk_name}] "
                f"[get_bd_pins {bd_name}/{st_name}/ap_clk]"
            )
            cmd.append(
                f"connect_bd_net [get_bd_pins {bd_name}/{rst_name}] "
                f"[get_bd_pins {bd_name}/{st_name}/ap_rst_n]"
            )
            cmd.append(
                f"connect_bd_intf_net [get_bd_intf_pins {bd_name}/m_axis_{idx + 1}] "
                f"[get_bd_intf_pins {bd_name}/{st_name}/m_axis_1]"
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
                src_inst_name = bd_name + "/" + st_map[src]
                src_intf_name = "m_axis_0"

            dst_intf_name = "s_axis_0"
            if len(dsts) == 1:
                dst_inst_name = st_map[dsts[0]]
                cmd.append(
                    f"connect_bd_intf_net [get_bd_intf_pins {src_inst_name}/{src_intf_name}] "
                    f"[get_bd_intf_pins {bd_name}/{dst_inst_name}/{dst_intf_name}]"
                )
            # if node is a fork connect data signals directly
            # and insert AND logic for rdy and vld signals
            elif len(dsts) > 1:
                if "__INPUT0__" in src:
                    cmd.append(
                        "create_bd_cell -type ip "
                        "-vlnv xilinx.com:ip:axis_broadcaster:1.1 "
                        f"{src_inst_name}/axi_broadcaster_0"
                    )
                    cmd.append(
                        f"set_property CONFIG.NUM_MI {{{len(dsts)}}} "
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
                    for idx, dst in enumerate(dsts):
                        dst_inst_name = st_map[dst]
                        cmd.append(
                            "connect_bd_intf_net "
                            f"[get_bd_intf_pins {src_inst_name}/axi_broadcaster_0/M0{idx}_AXIS] "
                            f"[get_bd_intf_pins {src_inst_name}/{dst_inst_name}/{dst_intf_name}]"
                        )
                else:
                    for idx, dst in enumerate(dsts):
                        dst_inst_name = st_map[dst]
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
                        elif idx < len(dsts):
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

                    cmd.append(
                        "connect_bd_net "
                        f"[get_bd_pins {src_inst_name}_util_vector_logic_{len(dsts) - 1}/Res] "
                        f"[get_bd_pins {src_inst_name}/{src_intf_name}_TREADY]"
                    )
                    for dst in dsts:
                        dst_inst_name = st_map[dst]
                        dst_intf_name = "s_axis_0"
                        cmd.append(
                            "connect_bd_net "
                            f"[get_bd_pins {src_inst_name}_util_vector_logic_{len(dsts) - 1}/Res] "
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
            f"[get_bd_intf_pins {bd_name}/{st_map[last_nodes[0]]}/m_axis_0]"
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
        cmd.append("set_property name m_axi_hbm [get_bd_intf_ports m_axi_hbm_0]")
        cmd.append("set_property name done_if [get_bd_ports done_if_0]")
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
        cmd.append(
            "ipx::remove_segment -quiet m_axi_gmem0:APERTURE_0 "
            "[ipx::get_address_spaces m_axi_gmem0 -of_objects [ipx::current_core]]"
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
        with (Path(vivado_stitch_proj_dir) / "make_loop_ip.sh").open("w") as f:
            f.write("#!/bin/bash \n")
            f.write(f"cd {vivado_stitch_proj_dir}\n")
            f.write("vivado -mode batch -source make_loop_ip.tcl\n")
            f.write(f"cd {working_dir}\n")
        bash_command = ["bash", make_project_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
        if not Path(wrapper_filename).is_file():
            raise FINNInternalError(f"IPGen failed: {wrapper_filename} not found")
        self.set_nodeattr("ipgen_path", wrapper_filename)
        self.set_nodeattr("ip_path", vivado_stitch_proj_dir + "/ip")
        self.set_nodeattr("gen_top_module", f"{block_name}_bd_design_wrapper")
        self.set_nodeattr("ip_vlnv", block_vlnv)

    def get_verilog_top_module_intf_names(self) -> dict:
        """Return the verilog top module interface names (from wrapper template)."""
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
        # TODO: rename because it might not be hbm?
        intf_names["aximm"].append(["m_axi_hbm", str(addr_bits)])
        intf_names["axilite"] = []

        # using ap_none field to add control signals
        intf_names["ap_none"] = []
        # done_if should be externalize to a block diagram port
        # and connected to the axil_iw_slv_mlo component
        intf_names["ap_none"].append("done_if")

        loop_body = self.body
        loop_body_intf = eval(cast("str", loop_body.get_metadata_prop("vivado_stitch_ifnames")))
        for intf in loop_body_intf["aximm"]:
            intf_names["aximm"].append(intf)

        return intf_names

    def code_generation_ipi(self) -> list[str]:
        """Return the IP-integrator commands to instantiate this node."""
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
        """Return the list of RTL files for this node."""
        return []

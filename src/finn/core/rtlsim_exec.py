"""Manage execution of RTL based simulation."""
# Copyright (c) 2020 Xilinx, Inc.
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

import numpy as np
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from finn import xsi as finnxsi
from finn.util.basic import get_liveness_threshold_cycles, getHWCustomOp, make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.xsi import SimEngine

if TYPE_CHECKING:
    from qonnx.core.datatype import BaseDataType
    from qonnx.core.modelwrapper import ModelWrapper

from ast import literal_eval

from finn.util.exception import FINNUserError


def prep_rtlsim_io_dict(
    model: "ModelWrapper", execution_context: dict[str, np.ndarray]
) -> tuple[
    dict[str, dict[str, list[int]]],
    dict[str, list[tuple[str, int]] | list[str]],
    dict[str, int] | int,
    list[tuple[int, "BaseDataType", tuple[int, ...], tuple[int, ...]]],
    int,
]:
    """Prepare the input/output dictionary for RTLSim execution."""
    # extract i/o info to prepare io_dict
    io_dict = {"inputs": {}, "outputs": {}}
    if_names = model.get_metadata_prop("vivado_stitch_ifnames")
    if if_names is None:
        raise FINNUserError(
            "Vivado stitch interface names not found in model metadata. "
            "Did you run step_create_stitched_ip first?"
        )
    if_dict: dict[str, list[tuple[str, int]] | list[str]] = literal_eval(if_names)
    # go over and prepare inputs
    batchsize = None
    first_node = None
    if_name = None
    for i, i_vi in enumerate(model.graph.input):
        i_name = i_vi.name
        i_tensor = execution_context[i_name]
        i_dt = model.get_tensor_datatype(i_name)
        first_node_onnx = model.find_consumer(i_name)
        if first_node_onnx is None:
            raise FINNUserError(
                f"Input {i_name} has no consumer node in the model. "
                f"Check that the inputs are all properly connected."
            )
        first_node = getHWCustomOp(first_node_onnx)
        node_inp_ind = list(first_node_onnx.input).index(i_name)
        if node_inp_ind == 0:
            # default node input (input 0)
            i_stream_w = first_node.get_instream_width()
            i_folded_shape = first_node.get_folded_input_shape()
        else:
            # not input 0; node must support specifying inp index
            # for these functions
            i_stream_w = first_node.get_instream_width(node_inp_ind)
            i_folded_shape = first_node.get_folded_input_shape(node_inp_ind)

        if first_node.onnx_node.op_type == "InnerShuffle_rtl":
            batchsize = 1
        else:
            batchsize = i_tensor.shape[0]
            i_folded_shape = list(i_folded_shape)
            i_folded_shape[0] = batchsize
            # override batch size for input
        i_folded_shape = tuple(i_folded_shape)

        # TODO any other layout transformations need to happen here!
        i_tensor = i_tensor.reshape(i_folded_shape)
        # pack input for rtlsim
        packed_input = npy_to_rtlsim_input(i_tensor, i_dt, i_stream_w)
        # add to io_dict
        if_name = if_dict["s_axis"][i][0]
        io_dict["inputs"][if_name] = packed_input
    # go over outputs to determine how many values will be produced
    num_out_values: dict[str, int] | int = {}
    o_tensor_info: list[tuple[int, BaseDataType, tuple[int, ...], tuple[int, ...]]] = []
    if first_node is None or batchsize is None or if_name is None:
        raise FINNUserError(
            "No consumer node found for first input. "
            "Cannot determine output stream widths and number of output values. "
            "Check that the inputs are all properly connected and consumed by a node."
        )
    for o, o_vi in enumerate(model.graph.output):
        # output in io_dict just needs an empty list
        if_name = if_dict["m_axis"][o][0]
        io_dict["outputs"][if_name] = []
        # extract output shape
        o_name = o_vi.name
        o_shape = model.get_tensor_shape(o_name)
        if o_shape is None:
            raise FINNUserError(
                f"Shape of output {o_name} is not known. "
                f"Cannot determine number of output values. "
                f"Check that the model is properly inferred and shapes are known."
            )
        o_dt = model.get_tensor_datatype(o_name)
        last_node_onnx = model.find_producer(o_name)
        if last_node_onnx is None:
            raise FINNUserError(
                f"Output {o_name} has no producer node in the model. "
                f"Check that the outputs are all properly connected."
            )
        last_node = getHWCustomOp(last_node_onnx)
        o_folded_shape = last_node.get_folded_output_shape()
        # override batch size from actual input
        o_shape = list(o_shape)
        if o_shape[0] != batchsize and (first_node.onnx_node.op_type != "InnerShuffle_rtl"):
            o_shape[0] = batchsize
            num_out_values[if_name] = batchsize * last_node.get_number_output_values()
        else:
            num_out_values[if_name] = last_node.get_number_output_values()
        o_shape = tuple(o_shape)
        o_folded_shape = list(o_folded_shape)
        if o_folded_shape[0] != batchsize and (first_node.onnx_node.op_type != "InnerShuffle_rtl"):
            o_folded_shape[0] = batchsize
        o_folded_shape = tuple(o_folded_shape)
        o_stream_w = last_node.get_outstream_width()
        o_tensor_info.append((o_stream_w, o_dt, o_folded_shape, o_shape))
    if len(num_out_values.keys()) == 1:
        num_out_values = num_out_values[if_name]
    return io_dict, if_dict, num_out_values, o_tensor_info, batchsize


def rtlsim_exec_finnxsi(
    model: "ModelWrapper",
    execution_context: dict[str, np.ndarray],
    pre_hook: Callable[[SimEngine], None] | None = None,
    post_hook: Callable[[SimEngine], None] | None = None,
) -> None:
    """Use finnxsi to execute given model with stitched IP. The execution
    context contains the input values. Hook functions can be optionally
    specified to observe/alter the state of the circuit
    - pre_hook : hook function to be called before sim start (after reset)
    - post_hook : hook function to be called after sim end.
    """
    # ensure stitched ip project already exists
    wrapper_filename = model.get_metadata_prop("wrapper_filename")
    if wrapper_filename is None:
        wrapper_filename = ""
    wrapper_filename = Path(wrapper_filename)
    if not wrapper_filename.is_file():
        raise FINNUserError(
            f"Wrapper file {wrapper_filename} doesn't exist. "
            f"Did you run step_create_stitched_ip first?"
        )
    vivado_stitch_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
    if vivado_stitch_proj_dir is None:
        vivado_stitch_proj_dir = ""
    vivado_stitch_proj_dir = Path(vivado_stitch_proj_dir)
    if not vivado_stitch_proj_dir.is_dir():
        raise FINNUserError(
            f"Directory {vivado_stitch_proj_dir} doesn't exist. "
            f"Did you run step_create_stitched_ip first?"
        )
    trace_file = model.get_metadata_prop("rtlsim_trace")
    io_dict, if_dict, num_out_values, o_tensor_info, batchsize = prep_rtlsim_io_dict(
        model, execution_context
    )

    # prepare rtlsim model
    rtlsim_so = model.get_metadata_prop("rtlsim_so")
    if (rtlsim_so is None) or (not Path(rtlsim_so).is_file()):
        with (vivado_stitch_proj_dir / "all_verilog_srcs.txt").open() as f:
            all_verilog_srcs = f.read().split()
        top_module_file_name = wrapper_filename.name
        top_module_name = top_module_file_name.strip(".v")
        single_src_dir = Path(make_build_dir("rtlsim_" + top_module_name + "_"))
        debug = not (trace_file is None or trace_file == "")
        rtlsim_so = finnxsi.compile_sim_obj(
            top_module_name, all_verilog_srcs, single_src_dir, debug=debug
        )
        # save generated lib filename in attribute
        model.set_metadata_prop("rtlsim_so", str(rtlsim_so[0] / rtlsim_so[1]))
        sim_base, sim_rel = rtlsim_so
        # pass in correct tracefile from attribute
        if trace_file == "default":
            trace_file = top_module_file_name + ".wdb"
        sim = finnxsi.load_sim_obj(sim_base, sim_rel, trace_file)
    else:
        sim_base, sim_rel = rtlsim_so.split("xsim.dir")
        sim_rel = "xsim.dir" + sim_rel
        sim = finnxsi.load_sim_obj(Path(sim_base), Path(sim_rel), trace_file)

    # reset and call rtlsim, including any pre/post hooks
    finnxsi.reset_rtlsim(sim)
    if pre_hook is not None:
        pre_hook(sim)
    n_cycles = finnxsi.rtlsim_multi_io(
        sim,
        io_dict,
        cast("dict[str, int | np.integer]|int", num_out_values),
        sname="",
        liveness_threshold=get_liveness_threshold_cycles() * batchsize,
    )
    if post_hook is not None:
        post_hook(sim)
    # important to call close_rtlsim for finnxsi to flush traces and stop
    finnxsi.close_rtlsim(sim)

    # unpack outputs and put back into execution context
    for o, o_vi in enumerate(model.graph.output):
        o_name = o_vi.name
        if_name = if_dict["m_axis"][o][0]
        o_stream_w, o_dt, o_folded_shape, o_shape = o_tensor_info[o]
        packed_output = io_dict["outputs"][if_name]
        o_folded_tensor = rtlsim_output_to_npy(
            packed_output, None, o_dt, o_folded_shape, o_stream_w, o_dt.bitwidth()
        )
        execution_context[o_name] = o_folded_tensor.reshape(o_shape)

    model.set_metadata_prop("cycles_rtlsim", str(n_cycles))


def rtlsim_exec(
    model: "ModelWrapper",
    execution_context: dict[str, np.ndarray],
    pre_hook: Callable[[SimEngine], None] | None = None,
    post_hook: Callable[[SimEngine], None] | None = None,
) -> None:
    """Use XSI to execute given model with stitched IP. The execution
    context contains the input values. Hook functions can be optionally
    specified to observe/alter the state of the circuit, receiving the
    sim object as their first argument:
    - pre_hook : hook function to be called before sim start (after reset)
    - post_hook : hook function to be called after sim end.
    """
    rtlsim_exec_finnxsi(model, execution_context, pre_hook, post_hook)

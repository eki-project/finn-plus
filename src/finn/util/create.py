"""Utility functions for creating ONNX models, including random MLPs and adjacency lists."""
# Copyright (c) 2020 Xilinx, Inc.
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

import numpy as np
from collections import defaultdict, deque
from collections.abc import Callable
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import calculate_signed_dot_prod_range, gen_finn_dt_tensor, qonnx_make_model
from typing import Any


def hls_random_mlp_maker(layer_spec: list[dict[str, Any]]) -> ModelWrapper:
    """Create an MLP of given specification using HLSCustomOp instances.
    Generate random weights/thresholds of appropriate size."""
    ret = []
    rng = np.random.default_rng()
    for lyr in layer_spec:
        idt = lyr["idt"]
        wdt = lyr["wdt"]
        mw = lyr["mw"]
        mh = lyr["mh"]
        act = lyr["act"]
        lyr["W"] = gen_finn_dt_tensor(wdt, (mw, mh))
        if act is None:
            # no activation, produce accumulators
            thresholds = None
            threshold_dtype = None
            if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
                output_dtype = DataType["UINT32"]
            else:
                output_dtype = DataType["INT32"]
        else:
            output_dtype = act
            (min_val, max_val) = calculate_signed_dot_prod_range(idt, wdt, mw)
            min_val_int = int(min_val)
            max_val_int = int(max_val)
            n_steps = act.get_num_possible_values() - 1
            thresholds = rng.integers(min_val_int, max_val_int - 1, size=(mh, n_steps)).astype(
                np.float32
            )
            # provide non-decreasing thresholds
            thresholds = np.sort(thresholds, axis=1)
            # generate thresholds for activation
            if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
                threshold_dtype = DataType["UINT32"]
                # bias thresholds to be positive
                thresholds = np.ceil((thresholds + mw) / 2)
                assert (thresholds >= 0).all()
            else:
                threshold_dtype = DataType["INT32"]
        lyr["T"] = thresholds
        lyr["tdt"] = threshold_dtype
        lyr["odt"] = output_dtype
        ret.append(lyr)

    return hls_mlp_maker(ret)


def hls_mlp_maker(layer_spec: list[dict[str, Any]]) -> ModelWrapper:
    """Create an MLP of given specification using HLSCustomOp instances."""
    current_in_name = ""
    current_out_name = ""

    graph = helper.make_graph(nodes=[], name="mlp", inputs=[], outputs=[])

    model = qonnx_make_model(graph, producer_name="finn")
    model = ModelWrapper(model)

    for i, lyr in enumerate(layer_spec):
        current_w_name = f"W_{i}"
        current_t_name = f"T_{i}"
        current_in_name = f"act_{i}"
        current_out_name = f"act_{i + 1}"

        weights = lyr["W"]
        (mw, mh) = weights.shape
        thresholds = lyr["T"]
        pe = lyr["pe"]
        simd = lyr["simd"]
        wdt = lyr["wdt"]
        idt = lyr["idt"]
        tdt = lyr["tdt"]
        odt = lyr["odt"]

        if i == 0:
            global_in = helper.make_tensor_value_info(current_in_name, TensorProto.FLOAT, [1, mw])
            model.graph.input.append(global_in)

        if i == len(layer_spec) - 1:
            global_out = helper.make_tensor_value_info(current_out_name, TensorProto.FLOAT, [1, mh])
            model.graph.output.append(global_out)

        # there are two ways to implement bipolar weights and inputs for
        # MatrixVectorActivation:
        # - specify their datatypes as such
        # - specify their datatypes as BINARY as use binaryXnorMode
        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            # we'll internally convert weights/inputs to binary and specify the
            # datatypes as such, and also set the binaryXnorMode attribute to 1
            export_wdt = DataType["BINARY"]
            export_idt = DataType["BINARY"]
            binary_xnor_mode = 1
        else:
            export_wdt = wdt
            export_idt = idt
            binary_xnor_mode = 0

        if thresholds is not None:
            no_act = 0
            node_inp_list = [current_in_name, current_w_name, current_t_name]
            actval = 0 if odt == DataType["BIPOLAR"] else odt.min()
        else:
            # no thresholds
            node_inp_list = [current_in_name, current_w_name]
            actval = 0
            no_act = 1
        fc_layer_node = helper.make_node(
            "MVAU",
            node_inp_list,
            [current_out_name],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            MW=mw,
            MH=mh,
            SIMD=simd,
            PE=pe,
            inputDataType=export_idt.name,
            weightDataType=export_wdt.name,
            outputDataType=odt.name,
            ActVal=actval,
            binaryXnorMode=binary_xnor_mode,
            noActivation=no_act,
        )

        model.graph.node.append(fc_layer_node)
        model.set_tensor_datatype(current_in_name, idt)
        model.set_tensor_datatype(current_out_name, odt)
        model.set_tensor_datatype(current_w_name, wdt)
        if binary_xnor_mode:
            # convert bipolar to binary
            model.set_initializer(current_w_name, (weights + 1) / 2)
        else:
            model.set_initializer(current_w_name, weights)
        if thresholds is not None:
            model.set_tensor_datatype(current_t_name, tdt)
            model.set_initializer(current_t_name, thresholds)

    return model


def adjacency_list(
    model: ModelWrapper, filter_function: Callable[[Any], bool]
) -> dict[str, list[str]]:
    """Return adjacency list of nodes based on filter function."""
    graph = model.graph

    full_graph = defaultdict(list)
    # Build full DAG across all nodes
    for node in graph.node:
        for input_tensor in node.input:
            producer = model.find_producer(input_tensor)
            if producer:
                full_graph[producer.name].append(node.name)
            elif (
                hasattr(graph, "input")
                and graph.input
                and input_tensor in [inp.name for inp in graph.input]
            ):
                full_graph[input_tensor].append(node.name)
        for output_tensor in node.output:
            producer = model.find_producer(output_tensor)
            if (
                producer
                and hasattr(graph, "output")
                and graph.output
                and output_tensor in [output.name for output in graph.output]
            ):
                full_graph[producer.name].append(output_tensor)

    # Apply filtering logic
    if not callable(filter_function):
        raise ValueError("filter_function must be callable")
    filter_nodes = [node.name for node in graph.node if filter_function(node)]
    graph_inputs = (
        [inp.name for inp in graph.input] if hasattr(graph, "input") and graph.input else []
    )
    graph_outputs = (
        [output.name for output in graph.output]
        if hasattr(graph, "output") and graph.output
        else []
    )

    relevant_nodes = filter_nodes + graph_inputs + graph_outputs
    filtered_adjacency = defaultdict(list)

    for node in relevant_nodes:
        visited = set()
        queue = deque(full_graph.get(node, []))
        while queue:
            source = node
            sink = queue.popleft()
            if sink in visited:
                continue
            visited.add(sink)

            if sink in relevant_nodes:
                if sink in graph_outputs:
                    sink = f"__OUTPUT{graph_outputs.index(sink)}__"
                if source in graph_inputs:
                    source = f"__INPUT{graph_inputs.index(source)}__"
                filtered_adjacency[source].append(sink)
            else:
                queue.extend(full_graph.get(sink, []))

    return dict(filtered_adjacency)

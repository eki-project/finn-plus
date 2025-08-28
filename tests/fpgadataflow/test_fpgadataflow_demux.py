# ruff: noqa: N803
"""Tests to validate correct functionality of the (de)mux operators and the inserting transform."""
from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
from typing import cast

from finn.core.onnx_exec import execute_onnx
from finn.custom_op.fpgadataflow.hls.demux_hls import DeMuxBase_hls
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

# TODO: Test to expect failure when having an input width larger than the available bitwidth
# TODO: Test different strategies


def make_identity_model(
    streamNames: list[str],
    streamTypes: list[str],
    streamNormalShapes: list[str],
    streamFoldedShapes: list[str],
    streamWidths: list[int],
    muxed_bitwidth: int,
    multiplexStrategy: str,
    input_name_prefix: str = "in",
    output_name_prefix: str = "out",
) -> ModelWrapper:
    """Create and return a simple model with a Mux and Demux."""
    different_lengths = len(
        set(
            map(
                len,
                [streamNames, streamTypes, streamNormalShapes, streamFoldedShapes, streamWidths],
            )
        )
    )
    assert different_lengths == 1, "stream<...> arguments all need to have the same length!"

    # Create network inputs and outputs for all streams
    inputs = []
    outputs = []
    for i in range(len(streamNames)):
        normalShape = [int(x) for x in streamNormalShapes[i].split(",")]
        inputs.append(
            helper.make_tensor_value_info(
                f"{input_name_prefix}_{streamNames[i]}", TensorProto.FLOAT, normalShape
            )
        )
        outputs.append(
            helper.make_tensor_value_info(
                f"{output_name_prefix}_{streamNames[i]}", TensorProto.FLOAT, normalShape
            )
        )
    mux_node = helper.make_node(
        "AnnotatedMux_hls",
        [inp.name for inp in inputs],
        ["network"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        streamNames=streamNames,
        streamTypes=streamTypes,
        streamNormalShapes=streamNormalShapes,
        streamFoldedShapes=streamFoldedShapes,
        streamWidths=streamWidths,
        muxed_bitwidth=muxed_bitwidth,
        multiplexStrategy=multiplexStrategy,
    )
    demux_node = helper.make_node(
        "AnnotatedDemux_hls",
        ["network"],
        [outp.name for outp in outputs],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        streamNames=streamNames,
        streamTypes=streamTypes,
        streamNormalShapes=streamNormalShapes,
        streamFoldedShapes=streamFoldedShapes,
        streamWidths=streamWidths,
        muxed_bitwidth=muxed_bitwidth,
    )
    graph = helper.make_graph(
        nodes=[mux_node, demux_node],
        name="graph",
        inputs=inputs,
        outputs=outputs,  # value_info=[network]
    )
    model = qonnx_make_model(graph, producer_name="model")
    model = ModelWrapper(model)
    for i, inp in enumerate(inputs):
        model.set_tensor_datatype(inp.name, DataType[streamTypes[i]])
    for i, outp in enumerate(outputs):
        model.set_tensor_datatype(outp.name, DataType[streamTypes[i]])
    return model


def test_simple_ident():
    name = ["stream0", "stream1"]
    dt = ["UINT4", "UINT5"]
    shapes = ["1,2,10", "1,3,9"]
    widths = [80, 27 * 5]
    network = 512
    strat = "ROUND_ROBIN"
    model = make_identity_model(name, dt, shapes, shapes, widths, network, strat)
    for node in model.graph.node:
        inst = getCustomOp(node)
        inst.set_nodeattr("inFIFODepths", [100, 100])
        inst.set_nodeattr("outFIFODepths", [100, 100])
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(SpecializeLayers("xcu280-fsvh2892-2l-e"))
    model = model.transform(PrepareIP("xcu280-fsvh2892-2l-e", 2.5))
    model = model.transform(HLSSynthIP())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers("xcu280-fsvh2892-2l-e"))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP("xcu280-fsvh2892-2l-e", 2.5))
    model = model.transform(HLSSynthIP())
    model = model.transform(GiveUniqueNodeNames())
    for node in model.graph.node:
        print("NODE " + node.name)
    model = model.transform(CreateStitchedIP("xcu280-fsvh2892-2l-e", 2.5))
    model = model.transform(PrepareRTLSim())
    model = model.transform(SetExecMode("rtlsim"))
    model.set_metadata_prop("exec_mode", "rtlsim")

    inputs = {
        "in_stream0": gen_finn_dt_tensor(DataType[dt[0]], [int(x) for x in shapes[0].split(",")]),
        "in_stream1": gen_finn_dt_tensor(DataType[dt[1]], [int(x) for x in shapes[1].split(",")]),
    }
    print(inputs)
    for node in model.graph.node:
        if node.op_type == "AnnotatedMux_hls":
            mux: DeMuxBase_hls = cast(DeMuxBase_hls, getCustomOp(node))
            mux.set_sim_output_numbers([2, 3])
            assert mux.has_sim_output_numbers()
        if node.op_type == "AnnotatedDemux_hls":
            demux: DeMuxBase_hls = cast(DeMuxBase_hls, getCustomOp(node))
            demux.set_sim_output_numbers([2, 3])
            assert demux.has_sim_output_numbers()

    res = execute_onnx(model, inputs)
    print(res)
    for key in inputs.keys():
        assert np.array_equal(inputs[key], res[key.replace("in", "out")])

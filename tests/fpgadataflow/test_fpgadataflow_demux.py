# ruff: noqa: N803
from __future__ import annotations

import pytest

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


def make_single_node_model(
    nodetype: str,
    streamNames: list[str],
    streamTypes: list[str],
    streamNormalShapes: list[str],
    streamFoldedShapes: list[str],
    streamWidths: list[str],
    muxed_bitwidth: int,
    multiplexStrategy: str | None = None,
) -> ModelWrapper:
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, (1, 10))
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, (1, 10))
    node = helper.make_node(
        nodetype,
        ["input0", "input1", "input2"],
        ["output"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        streamNames=streamNames,
        streamTypes=streamTypes,
        streamNormalShapes=streamNormalShapes,
        streamFoldedShapes=streamFoldedShapes,
        streamWidths=streamWidths,
        muxed_bitwidth=muxed_bitwidth,
    )
    if multiplexStrategy is not None:
        node.attribute.append(helper.make_attribute("multiplexStrategy", multiplexStrategy))

    graph = helper.make_graph(nodes=[node], name="graph", inputs=[inp], outputs=[outp])
    model = qonnx_make_model(graph, producer_name="model")
    return ModelWrapper(model)


def make_identity_model(
    streamNames: list[str],
    streamTypes: list[str],
    streamNormalShapes: list[str],
    streamFoldedShapes: list[str],
    streamWidths: list[str],
    muxed_bitwidth: int,
    multiplexStrategy: str,
    input_name_prefix: str = "in",
    output_name_prefix: str = "out",
) -> ModelWrapper:
    """Create a model that takes some input streams, muxes them into one connection and demuxes
    them again. Can be used to verify that multiplexing is working correctly."""
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


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.parametrize("fpgapart", ["xcu280-fsvh2892-2l-e"])
@pytest.mark.parametrize("streamNames", [["in0", "in1", "in2"]])
@pytest.mark.parametrize("streamTypes", [["UINT4", "UINT8", "INT3"]])
@pytest.mark.parametrize("streamNormalShapes", [["1,2,5", "1,3,10", "1,20"]])
@pytest.mark.parametrize("streamFoldedShapes", [["1,2,5", "1,3,10", "1,20"]])
@pytest.mark.parametrize("streamWidths", [["128", "200", "412"]])
@pytest.mark.parametrize("muxType", ["AnnotatedMux_hls", "AnnotatedDemux_hls"])
@pytest.mark.parametrize("muxed_bitwidth", [512])
@pytest.mark.parametrize("multiplexStrategy", ["ROUND_ROBIN"])
def test_fpgadataflow_de_mux_ipgen(
    fpgapart: str,
    streamNames: list[str],
    streamTypes: list[str],
    streamNormalShapes: list[str],
    streamFoldedShapes: list[str],
    streamWidths: list[str],
    muxed_bitwidth: int,
    muxType: str,
    multiplexStrategy: str,
) -> None:
    model = make_single_node_model(
        muxType,
        streamNames,
        streamTypes,
        streamNormalShapes,
        streamFoldedShapes,
        streamWidths,
        muxed_bitwidth,
        multiplexStrategy if muxType == "AnnotatedMux_hls" else None,
    )
    model = model.transform(SpecializeLayers(fpgapart))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(fpgapart, 2.5))
    model = model.transform(GiveUniqueNodeNames())

    # HLSSynth contains the assert to check for the existance of the generated IP
    model = model.transform(HLSSynthIP())


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.parametrize("fpgapart", ["xcu280-fsvh2892-2l-e"])
@pytest.mark.parametrize("streamNames", [["in0", "in1", "in2"]])
@pytest.mark.parametrize("streamTypes", [["UINT4", "UINT8", "INT3"]])
@pytest.mark.parametrize("streamNormalShapes", [["1,2,5", "1,3,10", "1,20"]])
@pytest.mark.parametrize("streamFoldedShapes", [["1,2,5", "1,3,10", "1,20"]])
@pytest.mark.parametrize("streamWidths", [["128", "200", "412"]])
@pytest.mark.parametrize("muxed_bitwidth", [512])
@pytest.mark.parametrize("multiplexStrategy", ["ROUND_ROBIN"])
@pytest.mark.parametrize("samples", [10, 100])
def test_fpgadataflow_demux_identity_model(
    fpgapart: str,
    streamNames: list[str],
    streamTypes: list[str],
    streamNormalShapes: list[str],
    streamFoldedShapes: list[str],
    streamWidths: list[str],
    muxed_bitwidth: int,
    multiplexStrategy: str,
    samples: int,
) -> None:
    in_prefix = "in"
    out_prefix = "out"

    # Prepare model
    model = make_identity_model(
        streamNames,
        streamTypes,
        streamNormalShapes,
        streamFoldedShapes,
        streamWidths,
        muxed_bitwidth,
        multiplexStrategy,
        input_name_prefix=in_prefix,
        output_name_prefix=out_prefix,
    )
    assert len(model.graph.node) == 2, "Model has the wrong number of nodes for this test!"  #
    assert model.graph.node[0].op_type == "AnnotatedMux_hls"
    assert model.graph.node[1].op_type == "AnnotatedDemux_hls"
    assert getCustomOp(model.graph.node[0]).get_nodeattr("streamNames") == streamNames
    assert getCustomOp(model.graph.node[1]).get_nodeattr("streamNames") == streamNames
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(SpecializeLayers(fpgapart))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(fpgapart, 2.5))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(HLSSynthIP())
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareRTLSim())

    input_data = {}
    output_data = {}
    for iteration in range(samples):
        # Generate random data
        for i, streamName in enumerate(streamNames):
            input_stream_name = in_prefix + "_" + streamName
            output_stream_name = out_prefix + "_" + streamName
            tensor_shape = [int(x) for x in streamNormalShapes[i].split(",")]
            input_data[input_stream_name] = gen_finn_dt_tensor(
                DataType[streamTypes[i]], tensor_shape
            )
            output_data[output_stream_name] = input_data[input_stream_name]

        # Execute simulation
        print("Executing sample " + str(iteration))
        simulated_output = execute_onnx(model, input_data, return_full_exec_context=False)

        # Check outputs
        for streamName in simulated_output.keys():
            assert simulated_output[streamName] == output_data[streamName], (
                f"[Iteration {iteration}]  Data mismatch on stream {streamName}. "
                "Got {simulated_output[streamName]}, expected {output_data[streamName]}"
            )

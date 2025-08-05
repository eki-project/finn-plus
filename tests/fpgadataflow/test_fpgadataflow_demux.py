# ruff: noqa: N803
from __future__ import annotations
import pytest

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

from deps.qonnx.src.qonnx.util.basic import gen_finn_dt_tensor
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
        inputs.append(
            helper.make_tensor_value_info(
                f"{input_name_prefix}_{streamNames[i]}", TensorProto.FLOAT, streamNormalShapes[i]
            )
        )
        outputs.append(
            helper.make_tensor_value_info(
                f"{output_name_prefix}_{streamNames[i]}", TensorProto.FLOAT, streamNormalShapes[i]
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
        nodes=[mux_node, demux_node], name="graph", inputs=inputs, outputs=outputs
    )
    model = qonnx_make_model(graph, producer_name="model")
    return ModelWrapper(model)


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
    model = model.transform(SpecializeLayers(fpgapart))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(fpgapart, 2.5))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(HLSSynthIP())
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareRTLSim())

    # Generate random data
    input_data = dict(zip([f"{in_prefix}_{name}" for name in streamNames], []))
    for _ in range(samples):
        for i, streamName in enumerate(streamNames):
            input_data[streamName].append(
                gen_finn_dt_tensor(DataType[streamTypes[i]], streamNormalShapes[i])
            )

    # Create expected output dict
    output_data = {}
    for streamName, datas in input_data.items():
        output_data[streamName.replace("_" + in_prefix, "_" + out_prefix)] = datas

    simulated_output = execute_onnx(model, input_data, return_full_exec_context=False)

    # Check outputs
    for streamName in simulated_output.keys():
        assert simulated_output[streamName] == output_data[streamName], (
            f"Data mismatch on stream {streamName}. "
            "Got {simulated_output[streamName]}, expected {output_data[streamName]}"
        )

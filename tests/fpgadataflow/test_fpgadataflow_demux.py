# ruff: noqa: N803
"""Tests to validate correct functionality of the (de)mux operators and the inserting transform."""
from __future__ import annotations

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
from typing import TYPE_CHECKING, Any, cast

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.hls.demux_hls import DeMuxBase_hls

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


@pytest.mark.parametrize(
    "stream_config",
    [
        (["stream0", "stream1"], ["UINT4", "UINT5"], ["1,2,10", "1,3,9"], [80, 27 * 5]),
        (["stream0", "stream1"], ["BIPOLAR", "BIPOLAR"], ["1,2,10", "1,3,9"], [20, 27]),
        (
            ["stream0", "stream1", "stream2"],
            ["UINT3", "UINT6", "INT2"],
            ["1,3,2,10", "1,3,3,9", "1,10,1,3"],
            [180, 81 * 6, 60],
        ),
        (["stream0"], ["UINT3"], ["1,7,2"], [35]),
    ],
)
@pytest.mark.parametrize("network", [512])
@pytest.mark.parametrize("strategy", ["ROUND_ROBIN"])
@pytest.mark.parametrize("fpgapart", ["xcu280-fsvh2892-2l-e"])
@pytest.mark.parametrize("clk", [2.5])
@pytest.mark.parametrize("samples_per_stream", [1, 10])
def test_manual_demux_model(  # noqa: C901
    stream_config: tuple[list[str], list[str], list[str], list[str]],
    network: int,
    strategy: str,
    fpgapart: str,
    clk: float,
    samples_per_stream: int,
) -> None:
    """Test a simple manually constructed model. Contains only a mux followed by a demux."""
    names, dts, shapes, widths = stream_config

    def get_shape(index: int) -> list[int]:
        return [int(x) for x in shapes[index].split(",")]

    lengths: set[int] = {len(x) for x in [names, dts, shapes, widths]}
    assert len(lengths) == 1, "Arguments to test don't have the same length!"
    in_prefix = "in"
    out_prefix = "out"
    model: ModelWrapper = make_identity_model(
        names, dts, shapes, shapes, widths, network, strategy, in_prefix, out_prefix
    )

    # Prepare Model:
    # Set FIFO sizes manually
    for node in model.graph.node:
        inst = getCustomOp(node)
        inst.set_nodeattr("inFIFODepths", [100] * len(names))
        inst.set_nodeattr("outFIFODepths", [100] * len(names))

    # Infer shapes and DTs
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(GiveUniqueNodeNames())

    # Synth the mux and demux nodes
    model = model.transform(SpecializeLayers(fpgapart))
    model = model.transform(PrepareIP(fpgapart, clk))
    model = model.transform(HLSSynthIP())
    model = model.transform(GiveUniqueNodeNames())

    # Insert FIFOs and synthesize them
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers(fpgapart))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(fpgapart, clk))
    model = model.transform(HLSSynthIP())
    model = model.transform(GiveUniqueNodeNames())

    # Create stitched IP for simulation, prepare RTLSIM
    model = model.transform(CreateStitchedIP(fpgapart, clk))
    model = model.transform(PrepareRTLSim())
    model = model.transform(SetExecMode("rtlsim"))
    model.set_metadata_prop("exec_mode", "rtlsim")

    # Prepare model inputs
    inputs: dict[str, list[Any]] = {}
    expected: dict[str, list[Any]] = {}
    output_numbers = []
    for i, name in enumerate(names):
        in_name = f"{in_prefix}_{name}"
        out_name = f"{out_prefix}_{name}"
        inputs[in_name] = []
        expected[out_name] = []
        for _ in range(samples_per_stream):
            shape: list[int] = get_shape(i)
            dt: BaseDataType = DataType[dts[i]]
            tensor = gen_finn_dt_tensor(dt, shape)
            inputs[in_name].append(tensor)
            expected[out_name].append(tensor)

        # Save what number of outputs this stream will generate
        # In shape (1,2,10) we want the 2
        output_numbers.append(inputs[in_name][0].shape[-2])

    # Set simulation output numbers (required for this operator)
    for node in model.graph.node:
        if node.op_type == "AnnotatedMux_hls":
            mux: DeMuxBase_hls = cast("DeMuxBase_hls", getCustomOp(node))
            mux.set_sim_output_numbers(output_numbers)
            assert mux.has_sim_output_numbers()
        if node.op_type == "AnnotatedDemux_hls":
            demux: DeMuxBase_hls = cast("DeMuxBase_hls", getCustomOp(node))
            demux.set_sim_output_numbers(output_numbers)
            assert demux.has_sim_output_numbers()

    # Run the actual simluation
    for j in range(samples_per_stream):
        model_inputs = {k: v[j] for k, v in inputs.items()}
        res = execute_onnx(model, model_inputs)
        for key in expected.keys():
            assert np.array_equal(expected[key][j], res[key])

    # Test unbalanced inputs
    for _ in range(samples_per_stream):
        in_tensor: Any = gen_finn_dt_tensor(DataType[dts[0]], get_shape(0))
        res = execute_onnx(model, {f"{in_prefix}_{names[0]}": in_tensor})
        assert np.array_equal(res[f"{out_prefix}_{names[0]}"], in_tensor)
        for i in range(1, len(names)):
            out_name = f"{out_prefix}_{names[i]}"
            assert np.array_equal(res[out_name], np.zeros(get_shape(i)))

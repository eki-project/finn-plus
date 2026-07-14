############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ##########################################################################

"""End-to-end test for out-of-context synthesis."""

import pytest

import numpy as np
import numpy.typing as npt
from onnx import TensorProto, helper
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import ApplySimulatedFIFOSizes
from finn.transformation.fpgadataflow.simulation_build import BuildSimulation
from finn.transformation.fpgadataflow.simulation_connected import RunLayerParallelSimulation
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext

fpga_part = "xczu7ev-ffvc1156-2-e"
clk_ns = 10


def insert_and_set_fifo_depths(model: ModelWrapper, fpga_part: str, clk_ns: float) -> ModelWrapper:
    """Run FIFO sizing for testing."""
    cfg = DataflowBuildConfig()
    cfg.fpga_part = fpga_part
    cfg.synth_clk_period_ns = clk_ns
    model = model.transform(
        BuildSimulation(
            fpga_part,
            clk_ns,
            True,
            performance_sim=False,
        )
    )
    model = model.transform(RunLayerParallelSimulation(fpga_part, clk_ns, cfg))
    model = model.transform(ApplySimulatedFIFOSizes(cfg))
    return model


def generate_random_threshold_values(
    data_type: BaseDataType, num_input_channels: int, num_steps: int
) -> npt.NDArray[np.floating]:
    """Generate random threshold values for a given datatype."""
    rng = np.random.default_rng()
    if data_type.is_integer():
        low = int(data_type.min())
        high = int(data_type.max()) + 1
        return rng.integers(
            low,
            high,
            size=(num_input_channels, num_steps),
        ).astype(np.float32)
    return (rng.standard_normal(size=(num_input_channels, num_steps)) * 1000).astype(
        data_type.to_numpy_dt()
    )


def create_test_model() -> ModelWrapper:
    """Create a small model used for OOC synthesis testing."""
    weights = gen_finn_dt_tensor(DataType["INT4"], (16, 32))
    thresholds = np.sort(
        generate_random_threshold_values(
            DataType["FLOAT32"],
            1,
            DataType["INT8"].get_num_possible_values() - 1,
        ),
        axis=1,
    )
    mul_param = gen_finn_dt_tensor(DataType["FLOAT32"], [1])
    add_param = gen_finn_dt_tensor(DataType["FLOAT32"], [1, 4, 32])

    # Initialize a new graph
    nodes = []

    # Add nodes
    mt_op = helper.make_node(
        "MultiThreshold",
        inputs=["inp", "thresh"],
        outputs=["mt_output"],
        domain="qonnx.custom_op.general",
        out_dtype="INT8",
        out_bias=float(DataType["INT8"].min()),
    )
    nodes.append(mt_op)

    matmul_op = helper.make_node(
        "MatMul",
        inputs=["mt_output", "matmul_weight"],
        outputs=["matmul_output"],
    )
    nodes.append(matmul_op)

    scalar_mul_op = helper.make_node(
        "Mul",
        inputs=["matmul_output", "scalar_input"],
        outputs=["scalar_output"],
    )
    nodes.append(scalar_mul_op)

    channel_add_op = helper.make_node(
        "Add",
        inputs=["scalar_output", "channelwise_bias"],
        outputs=["final_output"],
    )
    nodes.append(channel_add_op)

    # Define inputs
    inputs = [
        helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4, 16]),
    ]

    # Define outputs
    outputs = [helper.make_tensor_value_info("final_output", TensorProto.FLOAT, [1, 4, 32])]

    value_info = [
        helper.make_tensor_value_info("mt_output", TensorProto.FLOAT, [1, 4, 16]),
        helper.make_tensor_value_info("thresh", TensorProto.FLOAT, [1, 255]),
        helper.make_tensor_value_info("matmul_output", TensorProto.FLOAT, [1, 4, 32]),
        helper.make_tensor_value_info("matmul_weight", TensorProto.FLOAT, [16, 32]),
        helper.make_tensor_value_info("scalar_input", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("scalar_output", TensorProto.FLOAT, [1, 4, 32]),
        helper.make_tensor_value_info("channelwise_bias", TensorProto.FLOAT, [1, 4, 32]),
    ]

    # Create the graph
    graph = helper.make_graph(
        nodes=nodes, name="TestModelGraph", inputs=inputs, outputs=outputs, value_info=value_info
    )

    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model = ModelWrapper(model)

    # Set initializers and datatypes
    model.set_initializer("matmul_weight", weights)
    model.set_initializer("thresh", thresholds)
    model.set_initializer("scalar_input", mul_param)
    model.set_initializer("channelwise_bias", add_param)

    model.set_tensor_datatype("inp", DataType["FLOAT32"])
    model.set_tensor_datatype("matmul_weight", DataType["INT4"])
    model.set_tensor_datatype("thresh", DataType["FLOAT32"])
    model.set_tensor_datatype("scalar_input", DataType["FLOAT32"])
    model.set_tensor_datatype("channelwise_bias", DataType["FLOAT32"])

    return model


@pytest.mark.end2end
@pytest.mark.vivado
@pytest.mark.slow
def test_ooc_synthesis() -> None:
    """Run OOC synthesis flow and validate expected outputs and reports."""
    model = create_test_model()
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # generate reference output
    x = gen_finn_dt_tensor(DataType["FLOAT32"], (1, 4, 16))
    y_dict = oxe.execute_onnx(model, {model.get_first_global_in(): x})
    y_ref = y_dict[model.get_first_global_out()]

    # infer and specialize layers
    model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(SpecializeLayers(fpga_part))

    # node-by-node rtlsim
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(PrepareIP(fpga_part, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())

    y_dict = oxe.execute_onnx(model, {model.get_first_global_in(): x})
    y_prod = y_dict[model.get_first_global_out()]
    assert (y_prod == y_ref).all()

    # FIFO sizing
    model = insert_and_set_fifo_depths(model, fpga_part, clk_ns)

    # stitched IP rtlsim
    model = model.transform(PrepareIP(fpga_part, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(fpga_part, clk_ns))
    model = model.transform(SynthOutOfContext(fpga_part, clk_ns))
    ret = model.get_metadata_prop("res_total_ooc_synth")
    assert ret is not None
    # example expected output: (details may differ based on Vivado version etc)
    # "{'vivado_proj_folder': ...,
    # 'LUT': 708.0, 'FF': 1516.0, 'DSP': 0.0, 'BRAM': 0.0, 'WNS': 0.152, '': 0,
    # 'fmax_mhz': 206.27062706270627}"
    ret = eval(ret)
    assert ret["LUT"] > 0
    assert ret["FF"] > 0
    assert ret["DSP"] > 0
    assert ret["BRAM"] > 0
    assert ret["fmax_mhz"] > 100

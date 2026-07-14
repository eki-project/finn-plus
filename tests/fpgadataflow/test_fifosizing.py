# Copyright (c) 2022 Xilinx, Inc.
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
"""Test FIFO sizing functionality."""

import pytest

import json
import shutil
import torch
from brevitas.export import export_qonnx
from onnx import TensorProto, helper
from pathlib import Path
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import qonnx_make_model
from typing import Literal

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.transformation.fpgadataflow.set_fifo_depths import ApplySimulatedFIFOSizes
from finn.transformation.fpgadataflow.simulation_build import BuildSimulation
from finn.transformation.fpgadataflow.simulation_connected import RunLayerParallelSimulation
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import make_build_dir
from tests.testing_util.test import get_trained_network_and_ishape


def insert_and_set_fifo_depths(model: ModelWrapper, fpga_part: str, clk_ns: float) -> ModelWrapper:
    """Run FIFO sizing for testing."""
    cfg = build_cfg.DataflowBuildConfig()
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


def fetch_test_model(topology: str, wbits: int = 2, abits: int = 2) -> Path:
    """Fetch the test model for the given topology and bitwidths,
    export it to QONNX, and return the output directory."""
    tmp_output_dir = Path(make_build_dir(f"build_fifosizing_{topology}_"))
    (model, ishape) = get_trained_network_and_ishape(topology, wbits, abits)
    chkpt_name = tmp_output_dir / "model.onnx"
    export_qonnx(model, torch.randn(ishape), chkpt_name)
    return tmp_output_dir


def make_multi_io_modelwrapper(ch: int, pe: int, idt: BaseDataType) -> ModelWrapper:
    """Make a simple ONNX model with one addstreams node and one duplicate streams node,
    with multiple inputs and outputs, for testing multi-IO FIFO sizing."""
    in0 = helper.make_tensor_value_info("in0", TensorProto.FLOAT, [1, ch])
    in1 = helper.make_tensor_value_info("in1", TensorProto.FLOAT, [1, ch])
    mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, ch])
    out0 = helper.make_tensor_value_info("out0", TensorProto.FLOAT, [1, ch])
    out1 = helper.make_tensor_value_info("out1", TensorProto.FLOAT, [1, ch])

    addstreams_node = helper.make_node(
        "ElementwiseAdd",
        ["in0", "in1"],
        ["mid"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        lhs_shape=[1, ch],
        rhs_shape=[1, ch],
        out_shape=[1, ch],
        lhs_dtype=idt.name,
        rhs_dtype=idt.name,
        out_dtype=idt.name,
        lhs_style="input",
        rhs_style="input",
        PE=pe,
        inFIFODepths=[2, 2],
    )
    duplicate_node = helper.make_node(
        "DuplicateStreams",
        ["mid"],
        ["out0", "out1"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=ch,
        NumOutputStreams=2,
        PE=pe,
        inputDataType=idt.name,
        numInputVectors=[1],
        outFIFODepths=[2, 2],
    )
    graph = helper.make_graph(
        nodes=[addstreams_node, duplicate_node],
        name="graph",
        inputs=[in0, in1],
        outputs=[out0, out1],
        value_info=[mid],
    )

    model = qonnx_make_model(graph, producer_name="multi-io-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("in0", idt)
    model.set_tensor_datatype("in1", idt)

    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    return model


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
@pytest.mark.parametrize("topology", ["tfc", "cnv"])
def test_fifosizing_linear(topology: Literal["tfc", "cnv"]) -> None:
    """Test FIFO sizing on a simple linear topology, and check that the generated FIFO config."""
    tmp_output_dir = fetch_test_model(topology)
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        auto_fifo_depths=True,
        target_fps=10000 if topology == "tfc" else 1000,
        synth_clk_period_ns=10.0,
        board="Pynq-Z1",
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        ],
        steps=[
            "step_qonnx_to_finn",
            "step_tidy_up",
            "step_streamline",
            "step_convert_to_hw",
            "step_create_dataflow_partition",
            "step_specialize_layers",
            "step_target_fps_parallelization",
            "step_apply_folding_config",
            "step_minimize_bit_width",
            "step_generate_estimate_reports",
            "step_generate_hardware",
            "step_measure_rtlsim_performance",
        ],
    )
    build.build_dataflow_cfg(str(tmp_output_dir / "model.onnx"), cfg)

    if topology == "tfc":
        expected_fifos = {
            "fifo_depths": {
                "StreamingFIFO_rtl_0": 2,
                "StreamingFIFO_rtl_1": 32,
                "StreamingFIFO_rtl_2": 32,
                "StreamingFIFO_rtl_3": 32,
                "StreamingFIFO_rtl_4": 32,
                "StreamingFIFO_rtl_5": 32,
                "StreamingFIFO_rtl_6": 32,
                "StreamingFIFO_rtl_7": 32,
                "StreamingFIFO_rtl_8": 32,
                "StreamingFIFO_rtl_9": 32,
            },
            "fifo_sizes": {
                "StreamingFIFO_rtl_0": 1024,
                "StreamingFIFO_rtl_1": 1024,
                "StreamingFIFO_rtl_2": 64,
                "StreamingFIFO_rtl_3": 448,
                "StreamingFIFO_rtl_4": 64,
                "StreamingFIFO_rtl_5": 64,
                "StreamingFIFO_rtl_6": 64,
                "StreamingFIFO_rtl_7": 256,
                "StreamingFIFO_rtl_8": 1024,
                "StreamingFIFO_rtl_9": 1024,
            },
            "impl_style": {
                "StreamingFIFO_rtl_0": "rtl",
                "StreamingFIFO_rtl_1": "rtl",
                "StreamingFIFO_rtl_2": "rtl",
                "StreamingFIFO_rtl_3": "rtl",
                "StreamingFIFO_rtl_4": "rtl",
                "StreamingFIFO_rtl_5": "rtl",
                "StreamingFIFO_rtl_6": "rtl",
                "StreamingFIFO_rtl_7": "rtl",
                "StreamingFIFO_rtl_8": "rtl",
                "StreamingFIFO_rtl_9": "rtl",
            },
            "ram_style": {
                "StreamingFIFO_rtl_0": "block",
                "StreamingFIFO_rtl_1": "block",
                "StreamingFIFO_rtl_2": "block",
                "StreamingFIFO_rtl_3": "block",
                "StreamingFIFO_rtl_4": "block",
                "StreamingFIFO_rtl_5": "block",
                "StreamingFIFO_rtl_6": "block",
                "StreamingFIFO_rtl_7": "block",
                "StreamingFIFO_rtl_8": "block",
                "StreamingFIFO_rtl_9": "block",
            },
            "total_fifo_size_kiB": 0.6171875,
        }
    else:
        expected_fifos = {
            "fifo_depths": {
                "StreamingFIFO_rtl_0": 2,
                "StreamingFIFO_rtl_1": 32,
                "StreamingFIFO_rtl_2": 32,
                "StreamingFIFO_rtl_3": 32,
                "StreamingFIFO_rtl_4": 32,
                "StreamingFIFO_rtl_5": 1024,
                "StreamingFIFO_rtl_6": 32,
                "StreamingFIFO_rtl_7": 32,
                "StreamingFIFO_rtl_8": 32,
                "StreamingFIFO_rtl_9": 32,
                "StreamingFIFO_rtl_10": 32,
                "StreamingFIFO_rtl_11": 32,
                "StreamingFIFO_rtl_12": 32,
                "StreamingFIFO_rtl_13": 512,
                "StreamingFIFO_rtl_14": 32,
                "StreamingFIFO_rtl_15": 32,
                "StreamingFIFO_rtl_16": 32,
                "StreamingFIFO_rtl_17": 32,
                "StreamingFIFO_rtl_18": 32,
                "StreamingFIFO_rtl_19": 32,
                "StreamingFIFO_rtl_20": 32,
                "StreamingFIFO_rtl_21": 32,
                "StreamingFIFO_rtl_22": 32,
                "StreamingFIFO_rtl_23": 512,
                "StreamingFIFO_rtl_24": 32,
                "StreamingFIFO_rtl_25": 32,
                "StreamingFIFO_rtl_26": 32,
                "StreamingFIFO_rtl_27": 32,
                "StreamingFIFO_rtl_28": 32,
                "StreamingFIFO_rtl_29": 32,
                "StreamingFIFO_rtl_30": 32,
                "StreamingFIFO_rtl_31": 32,
                "StreamingFIFO_rtl_32": 32,
                "StreamingFIFO_rtl_33": 32,
                "StreamingFIFO_rtl_34": 32,
                "StreamingFIFO_rtl_35": 32,
            },
            "fifo_sizes": {
                "StreamingFIFO_rtl_0": 1024,
                "StreamingFIFO_rtl_1": 256,
                "StreamingFIFO_rtl_2": 256,
                "StreamingFIFO_rtl_3": 6912,
                "StreamingFIFO_rtl_4": 64,
                "StreamingFIFO_rtl_5": 16384,
                "StreamingFIFO_rtl_6": 512,
                "StreamingFIFO_rtl_7": 1152,
                "StreamingFIFO_rtl_8": 2048,
                "StreamingFIFO_rtl_9": 64,
                "StreamingFIFO_rtl_10": 64,
                "StreamingFIFO_rtl_11": 64,
                "StreamingFIFO_rtl_12": 64,
                "StreamingFIFO_rtl_13": 18432,
                "StreamingFIFO_rtl_14": 512,
                "StreamingFIFO_rtl_15": 128,
                "StreamingFIFO_rtl_16": 128,
                "StreamingFIFO_rtl_17": 1152,
                "StreamingFIFO_rtl_18": 1024,
                "StreamingFIFO_rtl_19": 64,
                "StreamingFIFO_rtl_20": 64,
                "StreamingFIFO_rtl_21": 64,
                "StreamingFIFO_rtl_22": 64,
                "StreamingFIFO_rtl_23": 18432,
                "StreamingFIFO_rtl_24": 128,
                "StreamingFIFO_rtl_25": 64,
                "StreamingFIFO_rtl_26": 64,
                "StreamingFIFO_rtl_27": 384,
                "StreamingFIFO_rtl_28": 64,
                "StreamingFIFO_rtl_29": 128,
                "StreamingFIFO_rtl_30": 64,
                "StreamingFIFO_rtl_31": 256,
                "StreamingFIFO_rtl_32": 64,
                "StreamingFIFO_rtl_33": 352,
                "StreamingFIFO_rtl_34": 1024,
                "StreamingFIFO_rtl_35": 1024,
            },
            "impl_style": {
                "StreamingFIFO_rtl_0": "rtl",
                "StreamingFIFO_rtl_1": "rtl",
                "StreamingFIFO_rtl_2": "rtl",
                "StreamingFIFO_rtl_3": "rtl",
                "StreamingFIFO_rtl_4": "rtl",
                "StreamingFIFO_rtl_5": "vivado",
                "StreamingFIFO_rtl_6": "rtl",
                "StreamingFIFO_rtl_7": "rtl",
                "StreamingFIFO_rtl_8": "rtl",
                "StreamingFIFO_rtl_9": "rtl",
                "StreamingFIFO_rtl_10": "rtl",
                "StreamingFIFO_rtl_11": "rtl",
                "StreamingFIFO_rtl_12": "rtl",
                "StreamingFIFO_rtl_13": "vivado",
                "StreamingFIFO_rtl_14": "rtl",
                "StreamingFIFO_rtl_15": "rtl",
                "StreamingFIFO_rtl_16": "rtl",
                "StreamingFIFO_rtl_17": "rtl",
                "StreamingFIFO_rtl_18": "rtl",
                "StreamingFIFO_rtl_19": "rtl",
                "StreamingFIFO_rtl_20": "rtl",
                "StreamingFIFO_rtl_21": "rtl",
                "StreamingFIFO_rtl_22": "rtl",
                "StreamingFIFO_rtl_23": "vivado",
                "StreamingFIFO_rtl_24": "rtl",
                "StreamingFIFO_rtl_25": "rtl",
                "StreamingFIFO_rtl_26": "rtl",
                "StreamingFIFO_rtl_27": "rtl",
                "StreamingFIFO_rtl_28": "rtl",
                "StreamingFIFO_rtl_29": "rtl",
                "StreamingFIFO_rtl_30": "rtl",
                "StreamingFIFO_rtl_31": "rtl",
                "StreamingFIFO_rtl_32": "rtl",
                "StreamingFIFO_rtl_33": "rtl",
                "StreamingFIFO_rtl_34": "rtl",
                "StreamingFIFO_rtl_35": "rtl",
            },
            "ram_style": {
                "StreamingFIFO_rtl_0": "block",
                "StreamingFIFO_rtl_1": "block",
                "StreamingFIFO_rtl_2": "block",
                "StreamingFIFO_rtl_3": "block",
                "StreamingFIFO_rtl_4": "block",
                "StreamingFIFO_rtl_5": "block",
                "StreamingFIFO_rtl_6": "block",
                "StreamingFIFO_rtl_7": "block",
                "StreamingFIFO_rtl_8": "block",
                "StreamingFIFO_rtl_9": "block",
                "StreamingFIFO_rtl_10": "block",
                "StreamingFIFO_rtl_11": "block",
                "StreamingFIFO_rtl_12": "block",
                "StreamingFIFO_rtl_13": "block",
                "StreamingFIFO_rtl_14": "block",
                "StreamingFIFO_rtl_15": "block",
                "StreamingFIFO_rtl_16": "block",
                "StreamingFIFO_rtl_17": "block",
                "StreamingFIFO_rtl_18": "block",
                "StreamingFIFO_rtl_19": "block",
                "StreamingFIFO_rtl_20": "block",
                "StreamingFIFO_rtl_21": "block",
                "StreamingFIFO_rtl_22": "block",
                "StreamingFIFO_rtl_23": "block",
                "StreamingFIFO_rtl_24": "block",
                "StreamingFIFO_rtl_25": "block",
                "StreamingFIFO_rtl_26": "block",
                "StreamingFIFO_rtl_27": "block",
                "StreamingFIFO_rtl_28": "block",
                "StreamingFIFO_rtl_29": "block",
                "StreamingFIFO_rtl_30": "block",
                "StreamingFIFO_rtl_31": "block",
                "StreamingFIFO_rtl_32": "block",
                "StreamingFIFO_rtl_33": "block",
                "StreamingFIFO_rtl_34": "block",
                "StreamingFIFO_rtl_35": "block",
            },
            "total_fifo_size_kiB": 8.85546875,
        }

    with (tmp_output_dir / "report/fifo_sizing.json").open() as f:
        fifo_sizing_report = json.load(f)
    assert fifo_sizing_report == expected_fifos
    # now run the same build using the generated folding and FIFO config
    tmp_output_dir_cmp = fetch_test_model(topology)
    cfg_cmp = cfg
    cfg_cmp.output_dir = tmp_output_dir_cmp
    cfg_cmp.auto_fifo_depths = False
    cfg_cmp.target_fps = None
    cfg_cmp.folding_config_file = tmp_output_dir / "report/auto_folding_config.json"
    cfg_cmp.fifo_config_file = tmp_output_dir / "report/fifo_sizing.json"
    build.build_dataflow_cfg(str(tmp_output_dir_cmp / "model.onnx"), cfg_cmp)

    model0 = ModelWrapper(str(tmp_output_dir / "intermediate_models/step_generate_hardware.onnx"))
    model1 = ModelWrapper(
        str(tmp_output_dir_cmp / "intermediate_models/step_generate_hardware.onnx")
    )

    assert len(model0.graph.node) == len(model1.graph.node)
    for i in range(len(model0.graph.node)):
        node0 = model0.graph.node[i]
        node1 = model1.graph.node[i]
        assert node0.op_type == node1.op_type
        if node0.op_type == "StreamingFIFO":
            node0_inst = getCustomOp(node0)
            node1_inst = getCustomOp(node1)
            assert node0_inst.get_nodeattr("depth") == node1_inst.get_nodeattr("depth")

    shutil.rmtree(tmp_output_dir)
    shutil.rmtree(tmp_output_dir_cmp)


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
def test_fifosizing_multi_io() -> None:
    """Construct small onnx graph with addstreams, followed by duplicate streams
    to have test model with multiple inputs and multiple outputs."""
    model = make_multi_io_modelwrapper(2, 2, DataType["INT4"])
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))
    model = model.transform(GiveUniqueNodeNames())
    model = insert_and_set_fifo_depths(model, "xc7z020clg400-1", 5)
    fifos = model.get_nodes_by_op_type("StreamingFIFO_rtl")
    assert len(fifos) > 1, "No FIFOs inserted"

import pytest

import shutil
from onnx import TensorProto, helper
from onnx.onnx_ml_pb2 import NodeProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


def make_single_node_model(idt, odt, nodetype="AnnotatedMux") -> ModelWrapper:
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, (1, 10))
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, (1, 10))
    node = helper.make_node(
        nodetype,
        ["input0", "input1", "input2"],
        ["output"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        inputDataType=idt.name,
        outputDataType=odt.name,
        streamNames=["in0", "in1", "in2"],
        streamTypes=["UINT4", "UINT8", "INT3"],
        streamNormalShapes=["1,2,5", "1,3,10", "1,20"],
        streamFoldedShapes=["1,2,5", "1,3,10", "1,20"],
        streamWidths=["128", "200", "412"],
        muxed_bitwidth=512,
    )
    graph = helper.make_graph(nodes=[node], name="graph", inputs=[inp], outputs=[outp])
    model = qonnx_make_model(graph, producer_name="model")
    model = ModelWrapper(model)
    return model


@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_arbiter_mux() -> None:
    idt = DataType["UINT4"]
    odt = DataType["UINT4"]
    fpgapart = "xcu280-fsvh2892-2l-e"
    model = make_single_node_model(idt, odt, "AnnotatedMux_hls")
    model = model.transform(SpecializeLayers(fpgapart))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(fpgapart, 2.5))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(HLSSynthIP())


@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_arbiter_demux() -> None:
    raise AssertionError()

"""Shared helpers for building small dataflow models in floorplanner tests."""

import numpy as np
from onnx import NodeProto, TensorProto
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

#: An Alveo U55C part, i.e. a three-SLR device with HBM in SLR0.
U55C_PART: str = "xcu55c-fsvh2892-2L-e"

#: A single-SLR Zynq part.
ZYNQ_PART: str = "xc7z020clg400-1"

#: Matrix dimensions used by every MVAU built here. Small enough to keep estimates
#: tiny and the tests fast.
MATRIX_DIM: int = 4


def make_mvau(name: str, inp: str, outp: str) -> NodeProto:
    """Build an abstract MVAU node with a matching weight input."""
    return oh.make_node(
        "MVAU",
        [inp, name + "_w"],
        [outp],
        name=name,
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=MATRIX_DIM,
        MH=MATRIX_DIM,
        SIMD=1,
        PE=1,
        inputDataType="INT2",
        weightDataType="INT2",
        outputDataType="INT2",
        ActVal=0,
        binaryXnorMode=0,
        noActivation=1,
    )


def make_fifo(name: str, inp: str, outp: str) -> NodeProto:
    """Build an abstract StreamingFIFO node."""
    return oh.make_node(
        "StreamingFIFO",
        [inp],
        [outp],
        name=name,
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        depth=16,
        folded_shape=[1, MATRIX_DIM, 1],
        normal_shape=[1, MATRIX_DIM],
        dataType="INT2",
        impl_style="rtl",
    )


def make_dwc(name: str, inp: str, outp: str) -> NodeProto:
    """Build an abstract StreamingDataWidthConverter node."""
    return oh.make_node(
        "StreamingDataWidthConverter",
        [inp],
        [outp],
        name=name,
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        inShape=[1, MATRIX_DIM],
        outShape=[1, MATRIX_DIM],
        inWidth=2,
        outWidth=2,
        dataType="INT2",
    )


def make_iodma(name: str, inp: str, outp: str, direction: str = "in") -> NodeProto:
    """Build an IODMA node, i.e. a node with a real external-memory AXI-MM port."""
    return oh.make_node(
        "IODMA_hls",
        [inp],
        [outp],
        name=name,
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        numInputVectors=[1],
        NumChannels=MATRIX_DIM,
        dataType="UINT8",
        intfWidth=32,
        streamWidth=32,
        direction=direction,
    )


def make_model(nodes: list[NodeProto], part: str = U55C_PART) -> ModelWrapper:
    """Assemble, specialize and name a dataflow model from a linear list of nodes.

    The nodes are wired in the order given: the first node's first input is the graph
    input and the last node's output is the graph output.

    Args:
        nodes: The nodes to place in the graph, already wired to each other.
        part: FPGA part to specialize the layers for.

    Returns:
        A specialized model with unique node names.
    """
    inp = oh.make_tensor_value_info(nodes[0].input[0], TensorProto.FLOAT, [1, MATRIX_DIM])
    outp = oh.make_tensor_value_info(nodes[-1].output[0], TensorProto.FLOAT, [1, MATRIX_DIM])
    graph = oh.make_graph(nodes=nodes, name="floorplan_test", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="floorplan-test"))
    for node in nodes:
        for tensor in list(node.input) + list(node.output):
            model.set_tensor_datatype(tensor, DataType["INT2"])
        if node.op_type == "MVAU":
            weights = node.input[1]
            model.set_initializer(weights, np.ones((MATRIX_DIM, MATRIX_DIM), dtype=np.float32))
            model.set_tensor_datatype(weights, DataType["INT2"])
    model = model.transform(SpecializeLayers(part))
    return model.transform(GiveUniqueNodeNames())


def make_chain(length: int, part: str = U55C_PART) -> ModelWrapper:
    """Build a linear chain of `length` MVAUs."""
    nodes = []
    for i in range(length):
        inp = "inp" if i == 0 else f"t{i}"
        outp = "outp" if i == length - 1 else f"t{i + 1}"
        nodes.append(make_mvau(f"mvau{i}", inp, outp))
    return make_model(nodes, part)


def attr(model: ModelWrapper, node_name: str, name: str) -> int | str:
    """Read a nodeattr off the node with the given name."""
    node = next(n for n in model.graph.node if n.name == node_name)
    return getCustomOp(node).get_nodeattr(name)


def set_device_ids(model: ModelWrapper, device_of: dict[str, int]) -> None:
    """Set `device_id` on the named nodes, as a prior multi-FPGA partitioning would."""
    for node in model.graph.node:
        if node.name in device_of:
            getCustomOp(node).set_nodeattr("device_id", device_of[node.name])

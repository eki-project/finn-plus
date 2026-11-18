"""Tests for the results that SetFolding records (unhandled op types, achieved cycles per
frame), its strict mode and the typed errors it raises.
"""

import pytest

import logging
import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.util.exception import FINNFoldingError, FINNUserError
from tests.fpgadataflow.test_set_folding import make_multi_fclayer_model

# Three 128x128 MVAUs with 4-bit weights: SIMD is limited to 8 by the default mvau_wwidth_max
# of 36 bits, so that they cannot get faster than 128 / 8 = 16 cycles per frame
MVAU_MIN_CYCLES = 16
# A target that SIMD alone reaches, i.e. without running into the limit of any attribute
MVAU_EASY_TARGET = 5000


def make_mvau_chain() -> ModelWrapper:
    """Return a chain of three MVAUs."""
    return make_multi_fclayer_model(128, DataType["INT4"], DataType["INT2"], DataType["INT16"], 3)


def make_swg_dw_model(
    consumer: str = "VVAU_hls", with_swg: bool = True, kernel: int = 3
) -> ModelWrapper:
    """Return a depthwise 3x3 convolution on 8 channels: a sliding window generator (or a
    FIFO in its place, if with_swg is False) followed by a VVAU or Pool layer."""
    ch, dim = 8, 8
    odim = dim - kernel + 1
    window = kernel * kernel
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, dim, dim, ch])
    mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, odim, odim, window * ch])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, odim, odim, ch])
    if with_swg:
        producer = helper.make_node(
            "ConvolutionInputGenerator_rtl",
            ["inp"],
            ["mid"],
            domain="finn.custom_op.fpgadataflow.rtl",
            backend="fpgadataflow",
            ConvKernelDim=[kernel, kernel],
            IFMChannels=ch,
            IFMDim=[dim, dim],
            OFMDim=[odim, odim],
            SIMD=1,
            Stride=[1, 1],
            Dilation=[1, 1],
            inputDataType="UINT4",
            outputDataType="UINT4",
            depthwise=1,
        )
    else:
        producer = helper.make_node(
            "StreamingFIFO_rtl",
            ["inp"],
            ["mid"],
            domain="finn.custom_op.fpgadataflow.rtl",
            backend="fpgadataflow",
            depth=2,
            folded_shape=[1, odim, odim, window * ch, 1],
            normal_shape=[1, odim, odim, window * ch],
            dataType="UINT4",
        )
    if consumer == "VVAU_hls":
        consumer_node = helper.make_node(
            "VVAU_hls",
            ["mid", "weights"],
            ["outp"],
            domain="finn.custom_op.fpgadataflow.hls",
            backend="fpgadataflow",
            PE=1,
            SIMD=1,
            Dim=[odim, odim],
            Channels=ch,
            Kernel=[kernel, kernel],
            resType="lut",
            ActVal=0,
            inputDataType="UINT4",
            weightDataType="INT4",
            outputDataType="INT16",
            noActivation=1,
        )
    else:
        consumer_node = helper.make_node(
            "Pool_hls",
            ["mid"],
            ["outp"],
            domain="finn.custom_op.fpgadataflow.hls",
            backend="fpgadataflow",
            Channels=ch,
            PE=1,
            KernelSize=[kernel, kernel],
            Function="MaxPool",
            OutImgDims=[odim, odim],
            InputDataType="UINT4",
            OutputDataType="UINT4",
        )
    graph = helper.make_graph([producer, consumer_node], "graph", [inp], [outp], value_info=[mid])
    model = ModelWrapper(qonnx_make_model(graph))
    if consumer == "VVAU_hls":
        model.set_initializer("weights", np.zeros((ch, 1, kernel, kernel), dtype=np.float32))
        model.set_tensor_datatype("weights", DataType["INT4"])
    for tensor in ("inp", "mid"):
        model.set_tensor_datatype(tensor, DataType["UINT4"])
    return model


@pytest.mark.fpgadataflow
def test_set_folding_results_target_met() -> None:
    """The results record a met target and no unhandled op types."""
    transform = SetFolding(MVAU_EASY_TARGET)
    assert transform.unhandled_op_types == []
    assert transform.achieved_cycles_per_frame is None
    assert transform.any_throughput_target_missed is False
    model = make_mvau_chain().transform(transform)
    assert transform.unhandled_op_types == []
    assert transform.any_throughput_target_missed is False
    assert transform.achieved_cycles_per_frame == max(model.analysis(exp_cycles_per_layer).values())
    assert transform.achieved_cycles_per_frame <= MVAU_EASY_TARGET


@pytest.mark.fpgadataflow
@pytest.mark.parametrize("two_pass_relaxation", [False, True])
def test_set_folding_results_target_missed(two_pass_relaxation: bool) -> None:
    """The results describe the requested target, also with the two-pass relaxation."""
    target = MVAU_MIN_CYCLES // 2
    transform = SetFolding(target, two_pass_relaxation=two_pass_relaxation)
    model = make_mvau_chain().transform(transform)
    # the results describe the requested target, also if the relaxation ran a second pass
    assert transform.target_cycles_per_frame == target
    assert transform.achieved_cycles_per_frame == MVAU_MIN_CYCLES
    assert transform.any_throughput_target_missed is True
    assert transform.unhandled_op_types == []
    assert max(model.analysis(exp_cycles_per_layer).values()) == MVAU_MIN_CYCLES


@pytest.mark.fpgadataflow
def test_set_folding_results_are_reset() -> None:
    """Every call of apply() starts from empty results."""
    transform = SetFolding(MVAU_MIN_CYCLES // 2)
    make_swg_dw_model(with_swg=False, kernel=1).transform(transform)
    assert transform.unhandled_op_types == ["StreamingFIFO_rtl"]
    make_mvau_chain().transform(transform)
    assert transform.unhandled_op_types == []
    assert transform.any_throughput_target_missed is True
    assert transform.achieved_cycles_per_frame == MVAU_MIN_CYCLES
    transform.target_cycles_per_frame = MVAU_EASY_TARGET
    make_mvau_chain().transform(transform)
    assert transform.any_throughput_target_missed is False
    assert transform.achieved_cycles_per_frame <= MVAU_EASY_TARGET


@pytest.mark.fpgadataflow
def test_set_folding_target_missed_hint_is_conservative() -> None:
    """any_throughput_target_missed is set although the target is reached in the end."""
    # PE of the VVAU is exhausted before SIMD is raised, which sets the hint although the
    # target is met in the end: only achieved_cycles_per_frame tells whether it was missed
    transform = SetFolding(300, strict=True)
    model = make_swg_dw_model().transform(transform)
    vvau = getCustomOp(model.get_nodes_by_op_type("VVAU_hls")[0])
    assert (vvau.get_nodeattr("PE"), vvau.get_nodeattr("SIMD")) == (8, 3)
    assert transform.any_throughput_target_missed is True
    assert transform.achieved_cycles_per_frame <= 300
    # the same holds for an MVAU whose SIMD is limited by mvau_wwidth_max before PE is raised
    transform = SetFolding(1000, strict=True)
    make_mvau_chain().transform(transform)
    assert transform.any_throughput_target_missed is True
    assert transform.achieved_cycles_per_frame <= 1000


@pytest.mark.fpgadataflow
def test_set_folding_strict_target_missed() -> None:
    """Strict mode raises for a target that cannot be reached."""
    target = MVAU_MIN_CYCLES // 2
    with pytest.raises(FINNFoldingError) as excinfo:
        make_mvau_chain().transform(SetFolding(target, strict=True))
    assert isinstance(excinfo.value, FINNUserError)
    assert excinfo.value.op_type == "MVAU_hls"
    assert excinfo.value.node_name == "MVAU_hls_0"
    assert f"missed the target of {target} cycles per frame" in str(excinfo.value)
    assert f"estimate of {MVAU_MIN_CYCLES} cycles" in str(excinfo.value)


@pytest.mark.fpgadataflow
def test_set_folding_strict_target_met() -> None:
    """Strict mode passes if the target is reached exactly."""
    transform = SetFolding(MVAU_MIN_CYCLES, strict=True)
    model = make_mvau_chain().transform(transform)
    assert max(model.analysis(exp_cycles_per_layer).values()) == MVAU_MIN_CYCLES
    assert transform.achieved_cycles_per_frame == MVAU_MIN_CYCLES


@pytest.mark.fpgadataflow
def test_set_folding_unhandled_op_types(caplog: pytest.LogCaptureFixture) -> None:
    """Op types SetFolding cannot fold are recorded, logged and rejected in strict mode."""
    # pointwise VVAU behind a FIFO, which SetFolding does not know how to fold
    transform = SetFolding(1000)
    with caplog.at_level(logging.WARNING):
        make_swg_dw_model(with_swg=False, kernel=1).transform(transform)
    assert transform.unhandled_op_types == ["StreamingFIFO_rtl"]
    messages = [record.getMessage() for record in caplog.records]
    assert "SetFolding doesn't know how to handle op_type StreamingFIFO_rtl" in messages

    with pytest.raises(FINNFoldingError) as excinfo:
        make_swg_dw_model(with_swg=False, kernel=1).transform(SetFolding(1000, strict=True))
    assert excinfo.value.op_type == "StreamingFIFO_rtl"
    assert excinfo.value.node_name == "StreamingFIFO_rtl_0"
    assert "SetFolding doesn't know how to handle op_type(s) StreamingFIFO_rtl" in str(
        excinfo.value
    )
    assert "missed the target" not in str(excinfo.value)


@pytest.mark.fpgadataflow
@pytest.mark.parametrize("consumer", ["VVAU_hls", "Pool_hls"])
def test_set_folding_depthwise_without_swg(consumer: str) -> None:
    """A depthwise layer without sliding window generator raises a typed error."""
    # a depthwise layer with a kernel larger than 1x1 needs a sliding window generator
    model = make_swg_dw_model(consumer=consumer, with_swg=False)
    with pytest.raises(FINNFoldingError) as excinfo:
        model.transform(SetFolding(1000))
    assert isinstance(excinfo.value, FINNUserError)
    assert excinfo.value.op_type == consumer
    assert excinfo.value.node_name == model.graph.node[1].name
    assert "Expected SWU on DW op input, found StreamingFIFO_rtl" in str(excinfo.value)


@pytest.mark.fpgadataflow
def test_set_folding_depthwise_without_producer() -> None:
    """A depthwise layer without any producer raises a typed error."""
    model = make_swg_dw_model()
    model.graph.node.remove(model.graph.node[0])
    model.graph.input[0].name = "mid"
    with pytest.raises(FINNFoldingError) as excinfo:
        model.transform(SetFolding(1000))
    assert excinfo.value.op_type == "VVAU_hls"
    assert "Expected producer node for VVAU/Pool input" in str(excinfo.value)

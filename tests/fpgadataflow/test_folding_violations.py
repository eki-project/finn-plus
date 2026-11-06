"""Tests for the folding legality check of the hardware layers (``check_folding``), the
``folding_violations`` analysis pass and the early failure in ``step_apply_folding_config``.
"""

import pytest

import json
import numpy as np
from collections.abc import Callable, Iterator
from onnx import TensorProto, helper
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

from finn.analysis.fpgadataflow.folding_violations import folding_violations
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.builder.build_dataflow_steps import step_apply_folding_config
from finn.util.basic import make_build_dir, robust_rmtree
from finn.util.exception import FINNFoldingError, FINNUnsupportedOpError, FINNUserError


def make_single_node_model(
    op_type: str,
    attrs: dict,
    in_shape: list[int],
    out_shapes: list[list[int]],
    inits: dict | None = None,
    name: str = "node",
) -> ModelWrapper:
    """Return a model with a single specialized hardware layer of the given op type, which
    consumes the graph input "inp" followed by the given initializers."""
    inits = inits or {}
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)
    outs = [
        helper.make_tensor_value_info(f"outp{i}", TensorProto.FLOAT, shape)
        for i, shape in enumerate(out_shapes)
    ]
    node = helper.make_node(
        op_type,
        ["inp", *inits.keys()],
        [out.name for out in outs],
        domain="finn.custom_op.fpgadataflow." + op_type.rsplit("_", 1)[1],
        backend="fpgadataflow",
        name=name,
        **attrs,
    )
    model = ModelWrapper(qonnx_make_model(helper.make_graph([node], "graph", [inp], outs)))
    for init_name, (values, dtype) in inits.items():
        model.set_initializer(init_name, values.astype(np.float32))
        model.set_tensor_datatype(init_name, DataType[dtype])
    return model


def make_mvau(variant: str = "hls", **attrs: int) -> ModelWrapper:
    """MVAU with MW=12 and MH=8."""
    node_attrs = {
        "MW": 12,
        "MH": 8,
        "SIMD": 4,
        "PE": 4,
        "inputDataType": "UINT4",
        "weightDataType": "INT4",
        "outputDataType": "INT16",
        "ActVal": 0,
        "binaryXnorMode": 0,
        "noActivation": 1,
        "numInputVectors": [1],
    }
    node_attrs.update(attrs)
    weights = {"weights": (np.ones((12, 8)), "INT4")}
    return make_single_node_model("MVAU_" + variant, node_attrs, [1, 12], [[1, 8]], weights)


def make_vvau(variant: str = "hls", **attrs: int) -> ModelWrapper:
    """VVAU with 8 channels and a 3x3 kernel."""
    node_attrs = {
        "PE": 4,
        "SIMD": 3,
        "Dim": [4, 4],
        "Channels": 8,
        "Kernel": [3, 3],
        "resType": "lut",
        "ActVal": 0,
        "inputDataType": "UINT4",
        "weightDataType": "INT4",
        "outputDataType": "INT16",
        "noActivation": 1,
    }
    node_attrs.update(attrs)
    weights = {"weights": (np.ones((8, 1, 3, 3)), "INT4")}
    return make_single_node_model(
        "VVAU_" + variant, node_attrs, [1, 4, 4, 72], [[1, 4, 4, 8]], weights
    )


def make_swg(k: int = 3, dim: int = 8, stride: int = 1, **attrs: int) -> ModelWrapper:
    """RTL sliding window generator with 8 channels."""
    odim = max((dim - k) // stride + 1, 1)
    node_attrs = {
        "ConvKernelDim": [k, k],
        "IFMChannels": 8,
        "IFMDim": [dim, dim],
        "OFMDim": [odim, odim],
        "SIMD": 4,
        "Stride": [stride, stride],
        "Dilation": [1, 1],
        "inputDataType": "UINT4",
        "outputDataType": "UINT4",
        "depthwise": 0,
        "parallel_window": 0,
    }
    node_attrs.update(attrs)
    return make_single_node_model(
        "ConvolutionInputGenerator_rtl",
        node_attrs,
        [1, dim, dim, 8],
        [[1, odim, odim, k * k * 8]],
    )


def make_fmpadding(**attrs: int) -> ModelWrapper:
    """RTL feature map padding with 8 channels."""
    node_attrs = {
        "ImgDim": [4, 4],
        "Padding": [1, 1, 1, 1],
        "NumChannels": 8,
        "inputDataType": "UINT4",
        "SIMD": 4,
    }
    node_attrs.update(attrs)
    return make_single_node_model("FMPadding_rtl", node_attrs, [1, 4, 4, 8], [[1, 6, 6, 8]])


def make_fmpadding_pixel(**attrs: int) -> ModelWrapper:
    """Pixel padding with 8 channels."""
    node_attrs = {
        "ImgDim": [4, 4],
        "Stride": [2, 2],
        "NumChannels": 8,
        "inputDataType": "UINT4",
        "SIMD": 4,
    }
    node_attrs.update(attrs)
    return make_single_node_model("FMPadding_Pixel_hls", node_attrs, [1, 4, 4, 8], [[1, 7, 7, 8]])


def make_pool(**attrs: int) -> ModelWrapper:
    """2x2 max pooling with 8 channels."""
    node_attrs = {
        "Channels": 8,
        "PE": 4,
        "KernelSize": [2, 2],
        "Function": "MaxPool",
        "OutImgDims": [2, 2],
        "InputDataType": "UINT4",
        "OutputDataType": "UINT4",
    }
    node_attrs.update(attrs)
    return make_single_node_model("Pool_hls", node_attrs, [1, 2, 2, 32], [[1, 2, 2, 8]])


def make_globalaccpool(**attrs: int) -> ModelWrapper:
    """Global accumulation pooling with 8 channels."""
    node_attrs = {"NumChannels": 8, "PE": 4, "inputDataType": "UINT4", "numInputVectors": [1, 4, 4]}
    node_attrs.update(attrs)
    return make_single_node_model("GlobalAccPool_hls", node_attrs, [1, 4, 4, 8], [[1, 1, 1, 8]])


def make_duplicatestreams(**attrs: int) -> ModelWrapper:
    """Stream duplication with 8 channels."""
    node_attrs = {
        "NumChannels": 8,
        "PE": 4,
        "NumOutputStreams": 2,
        "inputDataType": "UINT4",
        "numInputVectors": [1],
    }
    node_attrs.update(attrs)
    return make_single_node_model("DuplicateStreams_hls", node_attrs, [1, 8], [[1, 8], [1, 8]])


def make_labelselect(**attrs: int) -> ModelWrapper:
    """Label selection among 10 labels."""
    node_attrs = {"Labels": 10, "PE": 5, "K": 1, "inputDataType": "UINT4", "numInputVectors": [1]}
    node_attrs.update(attrs)
    return make_single_node_model("LabelSelect_hls", node_attrs, [1, 10], [[1, 1]])


# (model builder, attributes of an illegal folding, expected part of the message)
ILLEGAL_FOLDINGS = [
    pytest.param(make_mvau, {"PE": 3}, "MH (8) must be divisable by PE", id="mvau-pe"),
    pytest.param(make_mvau, {"SIMD": 5}, "MW (12) must be divisable by SIMD", id="mvau-simd"),
    pytest.param(
        make_mvau, {"variant": "rtl", "PE": 3}, "MH (8) must be divisable by PE", id="mvau_rtl-pe"
    ),
    pytest.param(
        make_vvau, {"PE": 3}, "Requirement Channels divisable by PE is violated.", id="vvau-pe"
    ),
    pytest.param(
        make_vvau,
        {"SIMD": 2},
        "Requirement kernel (k_h * k_w) divisable by SIMD is violated.",
        id="vvau-simd",
    ),
    pytest.param(
        make_vvau,
        {"variant": "rtl", "SIMD": 2},
        "Requirement kernel (k_h * k_w) divisable by SIMD is violated.",
        id="vvau_rtl-simd",
    ),
    pytest.param(make_swg, {"SIMD": 3}, "SIMD must divide IFMChannels", id="swg-simd"),
    pytest.param(
        make_swg,
        {"SIMD": 4, "parallel_window": 1},
        "SIMD must be equal to IFMChannels",
        id="swg-parallel-window",
    ),
    pytest.param(
        make_swg,
        {"SIMD": 3, "parallel_window": 1, "depthwise": 1},
        "SIMD must divide IFMChannels",
        id="swg-parallel-window-depthwise",
    ),
    pytest.param(make_fmpadding, {"SIMD": 3}, "SIMD must divide input channels", id="fmpadding"),
    pytest.param(
        make_fmpadding_pixel, {"SIMD": 3}, "SIMD must divide input channels", id="fmpadding_pixel"
    ),
    pytest.param(make_pool, {"PE": 3}, "PE must divide input channels", id="pool"),
    pytest.param(make_globalaccpool, {"PE": 3}, "PE must divide NumChannels", id="globalaccpool"),
    pytest.param(
        make_duplicatestreams, {"PE": 3}, "PE must divide NumChannels", id="duplicatestreams"
    ),
    pytest.param(make_labelselect, {"PE": 3}, "PE must divide Labels", id="labelselect"),
]

# (model builder, attributes of a legal folding)
LEGAL_FOLDINGS = [
    pytest.param(make_mvau, {}, id="mvau"),
    pytest.param(make_mvau, {"PE": 8, "SIMD": 12}, id="mvau-unfolded"),
    pytest.param(make_mvau, {"variant": "rtl", "PE": 1, "SIMD": 1}, id="mvau_rtl-folded"),
    pytest.param(make_vvau, {}, id="vvau"),
    pytest.param(make_vvau, {"variant": "rtl", "PE": 8, "SIMD": 9}, id="vvau_rtl-unfolded"),
    pytest.param(make_swg, {}, id="swg"),
    pytest.param(make_swg, {"SIMD": 8, "parallel_window": 1}, id="swg-parallel-window"),
    pytest.param(
        make_swg,
        {"SIMD": 4, "parallel_window": 1, "depthwise": 1},
        id="swg-parallel-window-depthwise",
    ),
    pytest.param(make_swg, {"k": 1, "SIMD": 4, "parallel_window": 1}, id="swg-parallel-window-1x1"),
    pytest.param(make_fmpadding, {}, id="fmpadding"),
    pytest.param(make_fmpadding_pixel, {}, id="fmpadding_pixel"),
    pytest.param(make_pool, {}, id="pool"),
    pytest.param(make_globalaccpool, {}, id="globalaccpool"),
    pytest.param(make_duplicatestreams, {}, id="duplicatestreams"),
    pytest.param(make_labelselect, {}, id="labelselect"),
]


@pytest.mark.fpgadataflow
@pytest.mark.parametrize("make_model, attrs, message", ILLEGAL_FOLDINGS)
def test_check_folding_illegal(make_model: Callable, attrs: dict, message: str) -> None:
    """An illegal folding raises a FINNFoldingError that names the node."""
    model = make_model(**attrs)
    node = model.graph.node[0]
    with pytest.raises(FINNFoldingError) as excinfo:
        getCustomOp(node).check_folding()
    assert isinstance(excinfo.value, FINNUserError)
    assert excinfo.value.node_name == "node"
    assert excinfo.value.op_type == node.op_type
    assert str(excinfo.value).startswith("node: ")
    assert message in str(excinfo.value)
    assert model.analysis(folding_violations) == {"node": str(excinfo.value)}


@pytest.mark.fpgadataflow
@pytest.mark.parametrize("make_model, attrs", LEGAL_FOLDINGS)
def test_check_folding_legal(make_model: Callable, attrs: dict) -> None:
    """A legal folding passes the check."""
    model = make_model(**attrs)
    assert getCustomOp(model.graph.node[0]).check_folding() is None
    assert model.analysis(folding_violations) == {}


# The code paths that rejected an illegal folding before check_folding() existed. They
# share its conditions and have to keep failing, so that no illegal folding gets further
# than before if the check is skipped.
@pytest.mark.fpgadataflow
@pytest.mark.parametrize(
    "make_model, attrs, method",
    [
        pytest.param(make_mvau, {"PE": 3}, "calc_wmem", id="mvau-pe"),
        pytest.param(make_mvau, {"SIMD": 5}, "calc_wmem", id="mvau-simd"),
        pytest.param(make_vvau, {"PE": 3}, "get_folded_input_shape", id="vvau-pe"),
        pytest.param(make_vvau, {"SIMD": 2}, "get_folded_input_shape", id="vvau-simd"),
        pytest.param(make_swg, {"SIMD": 3}, "select_impl_style", id="swg-impl-style"),
        pytest.param(make_swg, {"SIMD": 3}, "get_exp_cycles", id="swg-exp-cycles"),
        pytest.param(make_swg, {"SIMD": 3}, "get_folded_input_shape", id="swg-ishape"),
        pytest.param(make_swg, {"SIMD": 3}, "get_folded_output_shape", id="swg-oshape"),
        pytest.param(make_swg, {"SIMD": 3}, "get_instream_width", id="swg-width"),
        pytest.param(make_fmpadding, {"SIMD": 3}, "get_folded_input_shape", id="fmpadding-ishape"),
        pytest.param(make_fmpadding, {"SIMD": 3}, "get_folded_output_shape", id="fmpadding-oshape"),
        pytest.param(
            make_fmpadding_pixel, {"SIMD": 3}, "get_folded_input_shape", id="pixel-ishape"
        ),
        pytest.param(
            make_fmpadding_pixel, {"SIMD": 3}, "get_folded_output_shape", id="pixel-oshape"
        ),
        pytest.param(make_pool, {"PE": 3}, "get_folded_input_shape", id="pool-ishape"),
        pytest.param(make_pool, {"PE": 3}, "get_folded_output_shape", id="pool-oshape"),
        pytest.param(make_globalaccpool, {"PE": 3}, "get_folded_input_shape", id="gap-ishape"),
        pytest.param(make_globalaccpool, {"PE": 3}, "get_folded_output_shape", id="gap-oshape"),
        pytest.param(make_duplicatestreams, {"PE": 3}, "get_folded_input_shape", id="dup-ishape"),
        pytest.param(make_duplicatestreams, {"PE": 3}, "get_folded_output_shape", id="dup-oshape"),
        pytest.param(make_labelselect, {"PE": 3}, "get_folded_input_shape", id="labelselect"),
    ],
)
def test_illegal_folding_still_fails_in_existing_code_paths(
    make_model: Callable, attrs: dict, method: str
) -> None:
    """The code paths that rejected an illegal folding before keep rejecting it."""
    inst = getCustomOp(make_model(**attrs).graph.node[0])
    with pytest.raises(FINNFoldingError) as excinfo:
        getattr(inst, method)()
    assert excinfo.value.node_name == "node"


@pytest.mark.fpgadataflow
def test_threshold_tensors_only_depend_on_pe() -> None:
    """The threshold memory is folded by PE only, so an illegal SIMD is not reported there."""
    # the threshold memory is folded by PE only, so an illegal SIMD is not reported there
    thresholds = np.zeros((8, 3), dtype=np.float32)
    for make_model in (make_mvau, make_vvau):
        inst = getCustomOp(make_model(SIMD=5, noActivation=0).graph.node[0])
        assert inst.get_hw_compatible_threshold_tensor(thresholds).shape == (1, 4, 2, 3)
        inst = getCustomOp(make_model(PE=3, noActivation=0).graph.node[0])
        with pytest.raises(FINNFoldingError, match="PE"):
            inst.get_hw_compatible_threshold_tensor(thresholds)


@pytest.mark.fpgadataflow
@pytest.mark.parametrize(
    "attrs", [{"k": 3, "dim": 2}, {"k": 1, "dim": 2, "stride": 3}], ids=["kernel", "stride"]
)
def test_swg_window_exceeding_feature_map_is_not_a_folding_error(attrs: dict) -> None:
    """A window larger than the feature map is unsupported rather than a folding error."""
    # no folding can implement this node, so it is reported as unsupported instead
    model = make_swg(**attrs)
    inst = getCustomOp(model.graph.node[0])
    for check in (inst.check_folding, inst.select_impl_style):
        with pytest.raises(FINNUnsupportedOpError) as excinfo:
            check()
        assert not isinstance(excinfo.value, FINNFoldingError)
        assert excinfo.value.node_name == "node"
        assert excinfo.value.op_type == "ConvolutionInputGenerator_rtl"
        assert "Illegal conv configuration: kernel or stride > FM dimension" in str(excinfo.value)
    with pytest.raises(FINNUnsupportedOpError):
        model.analysis(folding_violations)


@pytest.mark.fpgadataflow
def test_check_folding_without_constraints() -> None:
    """Layers that do not implement the check accept any folding."""
    # layers that do not implement the check accept any folding
    fifo = helper.make_node(
        "StreamingFIFO_rtl",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        depth=2,
        folded_shape=[1, 2, 4],
        normal_shape=[1, 8],
        dataType="UINT4",
    )
    assert getCustomOp(fifo).check_folding() is None


@pytest.mark.fpgadataflow
def test_folding_violations_reports_all_nodes() -> None:
    """The analysis reports every node with an illegal folding in graph order."""
    legal = make_mvau().graph.node[0]
    illegal_pe = make_mvau(PE=3).graph.node[0]
    illegal_simd = make_mvau(SIMD=5).graph.node[0]
    software = helper.make_node("Mul", ["t2", "scale"], ["outp"], name="mul")
    tensors = ["inp", "t0", "t1", "t2"]
    for idx, (node, name) in enumerate([(legal, "a"), (illegal_pe, "b"), (illegal_simd, "c")]):
        node.name = name
        node.input[0], node.output[0] = tensors[idx], tensors[idx + 1]
    graph = helper.make_graph(
        [legal, illegal_pe, illegal_simd, software],
        "graph",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 12])],
        [helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 8])],
    )
    violations = ModelWrapper(qonnx_make_model(graph)).analysis(folding_violations)
    assert list(violations) == ["b", "c"]
    assert "MH (8) must be divisable by PE (currently 3)" in violations["b"]
    assert "MW (12) must be divisable by SIMD (currently 5)" in violations["c"]


@pytest.mark.fpgadataflow
def test_folding_violations_in_subgraphs() -> None:
    """The analysis descends into subgraphs on request."""
    body = make_mvau(PE=3).graph
    loop = helper.make_node("Loop", ["inp"], ["outp"], name="loop", body=body)
    graph = helper.make_graph(
        [loop],
        "graph",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 12])],
        [helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 8])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    assert model.analysis(folding_violations) == {}
    assert list(model.analysis(folding_violations, apply_to_subgraphs=True)) == ["node"]


@pytest.fixture
def build_dir() -> Iterator[Path]:
    """Provide a scratch directory for the build configuration."""
    path = make_build_dir("test_folding_violations_")
    yield Path(path)
    robust_rmtree(path)


def make_build_config(build_dir: Path, folding_config: dict | None = None) -> DataflowBuildConfig:
    """Return a build configuration, optionally with the given folding configuration."""
    folding_config_file = None
    if folding_config is not None:
        folding_config_file = build_dir / "folding_config.json"
        folding_config_file.write_text(json.dumps({"Defaults": {}, **folding_config}))
    return DataflowBuildConfig(
        output_dir=str(build_dir / "output"),
        synth_clk_period_ns=10.0,
        fpga_part="xc7z020clg400-1",
        generate_outputs=[],
        folding_config_file=folding_config_file,
    )


@pytest.mark.fpgadataflow
def test_step_apply_folding_config_rejects_illegal_config(build_dir: Path) -> None:
    """An illegal folding configuration file is rejected at the end of the step."""
    # GiveUniqueNodeNames renames the node to MVAU_hls_0 before the config is applied
    cfg = make_build_config(build_dir, {"MVAU_hls_0": {"PE": 3, "SIMD": 5}})
    with pytest.raises(FINNFoldingError) as excinfo:
        step_apply_folding_config(make_mvau(), cfg)
    assert isinstance(excinfo.value, FINNUserError)
    assert excinfo.value.node_name == "MVAU_hls_0"
    assert excinfo.value.op_type == "MVAU_hls"
    assert list(excinfo.value.violations) == ["MVAU_hls_0"]
    assert excinfo.value.violations["MVAU_hls_0"] in str(excinfo.value)
    assert "MVAU_hls_0: MH (8) must be divisable by PE (currently 3)" in str(excinfo.value)


@pytest.mark.fpgadataflow
def test_step_apply_folding_config_rejects_illegal_folding_without_config(build_dir: Path) -> None:
    """The folding is also checked if it was not set by this step."""
    # the folding is also checked if it was not set by this step
    cfg = make_build_config(build_dir)
    with pytest.raises(FINNFoldingError, match="Illegal folding configuration of 1 node"):
        step_apply_folding_config(make_vvau(SIMD=2), cfg)


@pytest.mark.fpgadataflow
def test_step_apply_folding_config_accepts_legal_config(build_dir: Path) -> None:
    """A legal folding configuration is applied as before."""
    cfg = make_build_config(build_dir, {"MVAU_hls_0": {"PE": 2, "SIMD": 6}})
    model = step_apply_folding_config(make_mvau(), cfg)
    inst = getCustomOp(model.graph.node[0])
    assert (inst.get_nodeattr("PE"), inst.get_nodeattr("SIMD")) == (2, 6)
    assert (cfg.get_report_directory() / "applied_folding_config.json").is_file()
    # and without a config file the model passes unchanged
    model = step_apply_folding_config(make_mvau(), make_build_config(build_dir))
    inst = getCustomOp(model.graph.node[0])
    assert (inst.get_nodeattr("PE"), inst.get_nodeattr("SIMD")) == (4, 4)

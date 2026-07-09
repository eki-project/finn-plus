"""Unit tests for Reduce conversion to FINN fpgadataflow Reduce hardware op."""

import pytest

import numpy as np
from onnx import TensorProto
from onnx import helper as oh
from onnx import numpy_helper
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
from typing import cast

import finn.core.onnx_exec as oxe
from finn.custom_op.fpgadataflow.reduce import Reduce
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferReduce
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.exception import FINNUserError

test_fpga_part: str = "xczu7ev-ffvc1156-2-e"
target_clk_ns = 5


def make_reduce_model(
    op_type: str,
    ishape: list[int],
    axes: list[int],
    keepdims: int,
    idt: BaseDataType,
    axes_as_initializer: bool = True,
) -> ModelWrapper:
    """Create a small ONNX model with a single Reduce* node using opset-19 style axes input."""
    inp = oh.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    out = oh.make_tensor_value_info("out", TensorProto.FLOAT, None)

    axes_name = "axes"
    axes_arr = np.asarray(axes, dtype=np.int64)
    graph_inputs = [inp]
    initializers = []
    value_info = []

    if axes_as_initializer:
        axes_init = numpy_helper.from_array(axes_arr, name=axes_name)
        initializers.append(axes_init)
        value_info.append(
            oh.make_tensor_value_info(axes_name, TensorProto.INT64, list(axes_arr.shape))
        )
    else:
        graph_inputs.append(
            oh.make_tensor_value_info(axes_name, TensorProto.INT64, list(axes_arr.shape))
        )

    node = oh.make_node(
        op_type,
        ["inp", axes_name],
        ["out"],
        keepdims=int(keepdims),
        name=f"{op_type}_node",
    )

    graph = oh.make_graph(
        [node],
        name="reduce_graph",
        inputs=graph_inputs,
        outputs=[out],
        initializer=initializers,
        value_info=value_info,
    )
    model = qonnx_make_model(graph, opset_imports=[oh.make_opsetid("", 19)], producer_name="reduce")
    model = ModelWrapper(model, fix_missing_initializer_valueinfo=True)
    model.set_tensor_datatype("inp", idt)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


@pytest.mark.parametrize(
    "op_type,exp_op",
    [
        ("ReduceSum", "sum"),
        ("ReduceMin", "min"),
        ("ReduceMax", "max"),
        ("ReduceProd", "product"),
    ],
)
@pytest.mark.parametrize("ishape", [[1, 4, 8], [1, 3, 5, 7], [2, 3, 4]])
@pytest.mark.parametrize("axes", [[2], [-1], [-2, -1], [1, 2], [-2]])
@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.fpgadataflow
def test_infer_reduce_conversion(
    op_type: str,
    ishape: list[int],
    axes: list[int],
    keepdims: int,
    exp_op: str,
) -> None:
    """Check InferReduce converts supported reductions into finn.custom_op.fpgadataflow Reduce."""
    model = make_reduce_model(op_type, ishape, axes, keepdims, DataType["INT8"])
    exp_start_axis = axes[0] if axes[0] >= 0 else len(ishape) + axes[0]

    model = model.transform(InferReduce())

    node = model.graph.node[0]
    assert node.op_type == "Reduce"
    assert node.domain == "finn.custom_op.fpgadataflow"
    assert len(node.input) == 1

    inst = getCustomOp(node)
    assert inst.get_nodeattr("op") == exp_op
    assert inst.get_nodeattr("index_start_axis") == exp_start_axis
    assert inst.get_nodeattr("keepdims") == keepdims
    assert inst.get_nodeattr("input_shape") == ishape


@pytest.mark.parametrize("axes", [[2], [1, 2]])
@pytest.mark.parametrize("ishape", [[1, 4, 4, 8]])
@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.fpgadataflow
def test_oshape_inference(axes: list[int], ishape: list[int], keepdims: int) -> None:
    """Check MinimizeAccumulatorWidth can be applied to Reduce."""
    model = make_reduce_model(
        op_type="ReduceSum",
        ishape=ishape,
        axes=axes,
        keepdims=keepdims,
        idt=DataType["INT8"],
    )
    model = model.transform(InferReduce())

    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst_reduce = cast("Reduce", inst)
    oshape = inst_reduce.get_normal_output_shape()

    expected_oshape: list[int | None] = list(ishape)
    for axis in axes:
        expected_oshape[axis] = 1 if keepdims else None
    expected_oshape = [dim for dim in expected_oshape if dim is not None]
    assert oshape == expected_oshape


@pytest.mark.parametrize(
    "ishape, axes",
    [
        ([1, 4, 8], [1]),
        ([1, 4, 4, 8], [1, 2]),
        ([1, 4, 1, 8], [2]),
        ([1, 1, 1, 8], [1, 2]),
        ([1, 4, 4, 5, 8], [1, 2, 3]),
    ],
)
@pytest.mark.fpgadataflow
def test_make_shape_compatible_op_reduce(ishape: list[int], axes: list[int]) -> None:
    """Check Reduce.make_shape_compatible_op emits ONNX ReduceSum with static axes input."""
    model = make_reduce_model(
        op_type="ReduceSum",
        ishape=ishape,
        axes=axes,
        keepdims=1,
        idt=DataType["INT8"],
    )
    model = model.transform(InferReduce())

    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst_reduce = cast("Reduce", inst)
    assert inst_reduce.stop_index == axes[-1]

    init_cnt = len(model.graph.initializer)
    shape_node = inst.make_shape_compatible_op(model)

    assert shape_node.op_type == "ReduceSum"
    assert shape_node.input[0] == node.input[0]
    assert shape_node.output[0] == node.output[0]
    assert len(shape_node.input) == 2
    assert len(model.graph.initializer) == init_cnt + 1
    axis_init = next(x for x in model.graph.initializer if x.name == shape_node.input[1])
    reduce_axis = np.arange(inst_reduce.start_index, inst_reduce.stop_index + 1, dtype=np.int64)
    assert np.array_equal(numpy_helper.to_array(axis_init), reduce_axis)


datatypes: list[BaseDataType] = []
datatypes.append(DataType["FLOAT32"])
for i in range(1, 33):
    datatypes.append(DataType["INT" + str(i)])
    datatypes.append(DataType["UINT" + str(i)])


@pytest.mark.parametrize("idt", datatypes)
@pytest.mark.parametrize("axes", [[2], [1, 2]])
@pytest.mark.parametrize("op_type", ["ReduceSum", "ReduceMin", "ReduceMax", "ReduceProd"])
@pytest.mark.fpgadataflow
def test_minimize_bitwidth_reduce(idt: BaseDataType, axes: list[int], op_type: str) -> None:
    """Check MinimizeAccumulatorWidth can be applied to Reduce."""
    ishape = [1, 4, 4, 8]
    model = make_reduce_model(
        op_type=op_type,
        ishape=ishape,
        axes=axes,
        keepdims=1,
        idt=idt,
    )
    model = model.transform(InferReduce())

    if idt.is_integer():
        reductions = 1
        for axis in axes:
            reductions *= ishape[axis]
        if op_type == "ReduceProd":
            bw = idt.bitwidth() * reductions
        elif op_type == "ReduceSum":
            bw = idt.bitwidth() + int(np.ceil(np.log2(reductions)))
        else:
            bw = idt.bitwidth()
        odt = DataType["INT" + str(bw)] if idt.signed() else DataType["UINT" + str(bw)]
    else:
        odt = idt

    model = model.transform(MinimizeAccumulatorWidth())
    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst_reduce = cast("Reduce", inst)
    assert inst_reduce.odtype == odt


@pytest.mark.parametrize(
    "axes",
    [
        [2, 2],  # duplicate axes
        [4],  # out of range axis
        [-5],  # negative out of range axis
        [1],  # non-trailing axis
        [1, 3],  # non-contiguous axes
    ],
)
@pytest.mark.fpgadataflow
def test_infer_reduce_invalid_axes_raises(axes: list[int]) -> None:
    """Check invalid axis configurations raise a user-facing error."""
    model = make_reduce_model(
        op_type="ReduceSum",
        ishape=[1, 3, 4, 5],
        axes=axes,
        keepdims=1,
        idt=DataType["INT8"],
        axes_as_initializer=True,
    )

    with pytest.raises(FINNUserError):
        model.transform(InferReduce())


@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.parametrize("ishape", [[1, 4, 8], [1, 3, 5, 7]])
@pytest.mark.parametrize("axes", [[-2], [1, 2]])
@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("op_type", ["ReduceSum", "ReduceMin", "ReduceMax", "ReduceProd"])
@pytest.mark.parametrize("pe", [1, 2, 4, "max"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_hwreduce(
    exec_mode: str, ishape: list[int], axes: list[int], keepdims: int, op_type: str, pe: int | str
) -> None:
    """End-to-end cppsim/rtlsim test for InferReduce + Reduce_hls execution."""
    if len(axes) + 1 == len(ishape):
        pytest.skip("Channel reduction not supported in Reduce_hls yet.")
    if isinstance(pe, str) and pe == "max":
        pe = ishape[-1]  # max PE is equal to number of lanes
    pe = int(pe)
    if ishape[-1] // pe != ishape[-1] / pe:
        pytest.skip("PE must divide number of lanes evenly.")
    tolerance = 1e-5
    datatype = DataType["INT8"]
    if ishape == [1, 3, 5, 7] and op_type == "ReduceProd":
        datatype = DataType["INT2"]
    model = make_reduce_model(
        op_type=op_type,
        ishape=ishape,
        axes=axes,
        keepdims=keepdims,
        idt=datatype,
    )

    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    inp = gen_finn_dt_tensor(datatype, ishape)
    input_t = {in_name: inp}

    y_ref = oxe.execute_onnx(model, input_t)[out_name]

    model = model.transform(InferReduce())
    assert model.graph.node[0].op_type == "Reduce"
    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst_reduce = cast("Reduce", inst)
    inst_reduce.set_nodeattr("PE", pe)

    model = model.transform(SpecializeLayers(test_fpga_part))
    assert model.graph.node[0].op_type == "Reduce_hls"

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(SetExecMode(exec_mode))
    model = model.transform(MinimizeAccumulatorWidth(), apply_to_subgraphs=True)
    # make sure the changed datatypes are propagated through the network
    model = model.transform(InferDataTypes(), apply_to_subgraphs=True)
    model = model.transform(InferDataLayouts(), apply_to_subgraphs=True)

    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    elif exec_mode == "rtlsim":
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())

    y_hw = oxe.execute_onnx(model, input_t)[out_name]
    assert np.allclose(y_ref, y_hw, atol=tolerance)


@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.parametrize("ishape", [[1, 4, 8], [1, 3, 5, 7]])
@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize(
    "op_type", ["ReduceSum", "ReduceMin", "ReduceMax"]
)  # No ReduceProd because of overflow
@pytest.mark.parametrize("pe", [1, 2, 4, "max"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_hwreduce_depthwise(
    exec_mode: str, ishape: list[int], keepdims: int, op_type: str, pe: int | str
) -> None:
    """End-to-end cppsim/rtlsim test for depthwise (channel-reducing) Reduce_hls execution."""
    # Reduce over every non-batch axis, including the channel axis, so the
    # conversion picks the depthwise reduction mode.
    axes = list(range(1, len(ishape)))
    if isinstance(pe, str) and pe == "max":
        pe = ishape[-1]  # max PE is equal to number of lanes
    pe = int(pe)
    if ishape[-1] // pe != ishape[-1] / pe:
        pytest.skip("PE must divide number of lanes evenly.")
    tolerance = 1e-5
    datatype = DataType["INT8"]
    model = make_reduce_model(
        op_type=op_type,
        ishape=ishape,
        axes=axes,
        keepdims=keepdims,
        idt=datatype,
    )

    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    inp = gen_finn_dt_tensor(datatype, ishape)
    input_t = {in_name: inp}

    y_ref = oxe.execute_onnx(model, input_t)[out_name]

    model = model.transform(InferReduce())
    assert model.graph.node[0].op_type == "Reduce"
    node = model.graph.node[0]
    inst = getCustomOp(node)
    inst_reduce = cast("Reduce", inst)
    assert inst_reduce.depthwise
    inst_reduce.set_nodeattr("PE", pe)

    model = model.transform(SpecializeLayers(test_fpga_part))
    assert model.graph.node[0].op_type == "Reduce_hls"

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(SetExecMode(exec_mode))
    model = model.transform(MinimizeAccumulatorWidth(), apply_to_subgraphs=True)
    # make sure the changed datatypes are propagated through the network
    model = model.transform(InferDataTypes(), apply_to_subgraphs=True)
    model = model.transform(InferDataLayouts(), apply_to_subgraphs=True)

    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())

    y_hw = oxe.execute_onnx(model, input_t)[out_name]
    assert np.allclose(y_ref, y_hw, atol=tolerance)

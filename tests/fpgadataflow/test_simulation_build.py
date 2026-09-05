"""Unit tests for SimulationBuilder isolated node model generation."""

from __future__ import annotations

import pytest

import numpy as np
from onnx import GraphProto, NodeProto, TensorProto, ValueInfoProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model
from typing import Protocol, TypedDict
from finn.transformation.fpgadataflow.simulation_build import SimulationBuilder
from finn.util.exception import FINNInternalError


class _SimulationBuilderProtocol(Protocol):
    def __init__(
        self,
        model: ModelWrapper,
        fpgapart: str,
        clk_ns: float,
        shm_prefix: str,
        performance_sim: bool = False,
    ) -> None:
        ...

    def _isolated_node_model(self, by_node: int | str | NodeProto) -> ModelWrapper:
        ...


def _vi(name: str, shape: list[int]) -> ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _add_value_info(value_info: list[ValueInfoProto], name: str, shape: list[int]) -> None:
    if any(vi.name == name for vi in value_info):
        return
    value_info.append(_vi(name, shape))


def _make_dwc(name: str, inp: str, out: str, shape: list[int]) -> NodeProto:
    return helper.make_node(
        "StreamingDataWidthConverter",
        [inp],
        [out],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        inShape=list(shape),
        outShape=list(shape),
        inWidth=9,
        outWidth=9,
        dataType="INT9",
        preferred_impl_style="rtl",
        name=name,
    )


def _make_fifo(name: str, inp: str, out: str, shape: list[int]) -> NodeProto:
    return helper.make_node(
        "StreamingFIFO",
        [inp],
        [out],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        depth=32,
        folded_shape=shape,
        normal_shape=shape,
        dataType="INT9",
        impl_style="rtl",
        ram_style="block",
        name=name,
    )


def _make_add_hls(name: str, lhs: str, rhs: str, out: str, shape: list[int]) -> NodeProto:
    return helper.make_node(
        "ElementwiseAdd_hls",
        [lhs, rhs],
        [out],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        numInputVectors=[1],
        lhs_shape=list(shape),
        rhs_shape=list(shape),
        out_shape=list(shape),
        lhs_dtype="INT9",
        rhs_dtype="INT9",
        out_dtype="INT9",
        lhs_style="input",
        rhs_style="input",
        PE=1,
        name=name,
    )


def _make_mvau_rtl(name: str, inp: str, weights: str, out: str) -> NodeProto:
    return helper.make_node(
        "MVAU_rtl",
        [inp, weights],
        [out],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        MW=4,
        MH=4,
        SIMD=1,
        PE=1,
        inputDataType="INT8",
        weightDataType="INT8",
        outputDataType="INT32",
        ActVal=0,
        binaryXnorMode=0,
        noActivation=1,
        name=name,
    )


def _make_duplicate_stream(
    name: str,
    inp: str,
    out_list: list[str],
    shape: list[int],
    num_outputs: int = 2,
    num_channels: int = 4,
    pe: int = 1,
    data_type: str = "INT8",
) -> NodeProto:
    return helper.make_node(
        "DuplicateStreams",
        [inp],
        out_list,
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=num_channels,
        NumOutputStreams=num_outputs,
        PE=pe,
        inputDataType=data_type,
        numInputVectors=shape,
        preferred_impl_style="hls",
        cpp_interface="hls_vector",
        hls_style="freerunning",
        name=name,
    )


def _wrap_model(graph: GraphProto) -> ModelWrapper:
    model = ModelWrapper(qonnx_make_model(graph, producer_name="simulation-build-test"))
    # model = model.transform(InferDataLayouts())
    # model = model.transform(InferDataTypes())
    # # model = model.transform(InferShapes())
    return model


def _build_unary_target_model(
    pre_binary: bool = False,
    succ_binary: bool = False,
    fifos: bool = False,
    include_succ_node: bool = True,
    extra_post_nodes: int = 0,
    fifo_pre: bool | None = None,
    fifo_between: bool | None = None,
    fifo_after: bool | None = None,
    fifo_between_depth: int | None = None,
) -> ModelWrapper:
    shape = [1, 32, 32, 3]
    if (
        include_succ_node
        and extra_post_nodes == 0
        and fifo_pre is None
        and fifo_between is None
        and fifo_after is None
        and fifo_between_depth is None
    ):
        nodes = []
        graph_inputs = []
        value_info = [_vi("target_in", shape), _vi("target_out", shape)]

        # Add Node in front of dwc
        if pre_binary:
            graph_inputs.extend([_vi("pre_in0", shape), _vi("pre_in1", shape)])
            if fifos:
                nodes.append(_make_fifo("FIFO_0", "pre_in0", "fifo_out0", shape))
                _add_value_info(value_info, "fifo_out0", shape)
                node_in_0 = "fifo_out0"
                nodes.append(_make_fifo("FIFO_1", "pre_in1", "fifo_out1", shape))
                _add_value_info(value_info, "fifo_out1", shape)
                node_in_1 = "fifo_out1"
            else:
                node_in_0 = "pre_in0"
                node_in_1 = "pre_in1"
            nodes.append(_make_add_hls("pre_add", node_in_0, node_in_1, "target_in", shape))
        else:
            graph_inputs.append(_vi("pre_in0", shape))
            if fifos:
                nodes.append(_make_fifo("FIFO_0", "pre_in0", "fifo_out0", shape))
                _add_value_info(value_info, "fifo_out0", shape)
                node_in_0 = "fifo_out0"
            else:
                node_in_0 = "pre_in0"
            nodes.append(_make_dwc("pre_dwc", node_in_0, "target_in", shape))

        nodes.append(_make_dwc("target_dwc", "target_in", "target_out", shape))

        if fifos:
            nodes.append(_make_fifo("FIFOOut_0", "target_out", "fifoOut_out0", shape))
            _add_value_info(value_info, "fifoOut_out0", shape)
            node_out_0 = "fifoOut_out0"
        else:
            node_out_0 = "target_out"

        # Add Node after dwc
        if succ_binary:
            graph_inputs.append(_vi("succ_in1", shape))
            nodes.append(_make_add_hls("succ_add", node_out_0, "succ_in1", "graph_out", shape))
            _add_value_info(value_info, "graph_out", shape)
        else:
            nodes.append(_make_dwc("succ_dwc", node_out_0, "graph_out", shape))
            _add_value_info(value_info, "graph_out", shape)

        if fifos:
            nodes.append(_make_fifo("FIFOOut_1", "graph_out", "fifoOut_out1", shape))
            _add_value_info(value_info, "fifoOut_out1", shape)
            node_out_1 = "fifoOut_out1"
        else:
            node_out_1 = "graph_out"

        graph_outputs = [_vi(node_out_1, shape)]

        reserved_names = {vi.name for vi in graph_inputs} | {vi.name for vi in graph_outputs}
        value_info = [vi for vi in value_info if vi.name not in reserved_names]

        graph = helper.make_graph(
            nodes=nodes,
            name="unary_target_graph",
            inputs=graph_inputs,
            outputs=graph_outputs,
            value_info=value_info,
        )
        return _wrap_model(graph)

    nodes = []
    graph_inputs = []
    value_info = [_vi("target_in", shape), _vi("target_out", shape)]
    use_fifo_pre = fifos if fifo_pre is None else fifo_pre
    use_fifo_between = fifos if fifo_between is None else fifo_between
    use_fifo_after = fifos if fifo_after is None else fifo_after
    fifo_between_count = (
        fifo_between_depth if fifo_between_depth is not None else (1 if use_fifo_between else 0)
    )
    fifo_index = 0

    def _append_fifo(prefix: str, inp: str) -> str:
        nonlocal fifo_index
        out_name = f"{prefix}_out{fifo_index}"
        nodes.append(_make_fifo(f"{prefix}_{fifo_index}", inp, out_name, shape))
        _add_value_info(value_info, out_name, shape)
        fifo_index += 1
        return out_name

    def _append_fifo_chain(prefix: str, inp: str, count: int) -> str:
        current = inp
        for _ in range(count):
            current = _append_fifo(prefix, current)
        return current

    if pre_binary:
        graph_inputs.extend([_vi("pre_in0", shape), _vi("pre_in1", shape)])
        node_in_0 = "pre_in0"
        node_in_1 = "pre_in1"
        if use_fifo_pre:
            node_in_0 = _append_fifo("FIFO_pre", node_in_0)
            node_in_1 = _append_fifo("FIFO_pre", node_in_1)
        nodes.append(_make_add_hls("pre_add", node_in_0, node_in_1, "target_in", shape))
    else:
        graph_inputs.append(_vi("pre_in0", shape))
        node_in_0 = "pre_in0"
        if use_fifo_pre:
            node_in_0 = _append_fifo("FIFO_pre", node_in_0)
        nodes.append(_make_dwc("pre_dwc", node_in_0, "target_in", shape))

    nodes.append(_make_dwc("target_dwc", "target_in", "target_out", shape))
    current_out = "target_out"

    if fifo_between_count > 0:
        current_out = _append_fifo_chain("FIFO_mid", current_out, fifo_between_count)

    if include_succ_node:
        if succ_binary:
            graph_inputs.append(_vi("succ_in1", shape))
            nodes.append(_make_add_hls("succ_add", current_out, "succ_in1", "post_0_out", shape))
        else:
            nodes.append(_make_dwc("succ_dwc", current_out, "post_0_out", shape))
        _add_value_info(value_info, "post_0_out", shape)
        current_out = "post_0_out"

    for idx in range(extra_post_nodes):
        if fifo_between_count > 0:
            current_out = _append_fifo_chain("FIFO_mid", current_out, fifo_between_count)
        post_name = f"post_out_{idx}"
        nodes.append(_make_dwc(f"post_dwc_{idx}", current_out, post_name, shape))
        _add_value_info(value_info, post_name, shape)
        current_out = post_name

    if use_fifo_after:
        current_out = _append_fifo("FIFO_out", current_out)

    graph_outputs = [_vi(current_out, shape)]

    reserved_names = {vi.name for vi in graph_inputs} | {vi.name for vi in graph_outputs}
    value_info = [vi for vi in value_info if vi.name not in reserved_names]

    graph = helper.make_graph(
        nodes=nodes,
        name="unary_target_graph",
        inputs=graph_inputs,
        outputs=graph_outputs,
        value_info=value_info,
    )
    return _wrap_model(graph)


def _build_binary_target_model(
    initializer_side: str | None = None,
    mlo: bool = False,
    fifos: bool = False,
    include_succ_node: bool = True,
    extra_post_nodes: int = 0,
    fifo_pre: bool | None = None,
    fifo_between: bool | None = None,
    fifo_after: bool | None = None,
    fifo_between_depth: int | None = None,
) -> ModelWrapper:
    shape = [1, 32, 32, 3]
    if (
        include_succ_node
        and extra_post_nodes == 0
        and fifo_pre is None
        and fifo_between is None
        and fifo_after is None
        and not fifos
        and fifo_between_depth is None
    ):
        lhs_name = "lhs_in"
        rhs_name = "rhs_in"
        nodes = [_make_add_hls("target_add", lhs_name, rhs_name, "target_out", shape)]
        nodes.append(_make_dwc("succ_dwc", "target_out", "graph_out", shape))

        graph_outputs = [_vi("graph_out", shape)]
        graph_inputs = []
        if initializer_side != "lhs":
            graph_inputs.append(_vi(lhs_name, shape))
        if initializer_side != "rhs":
            graph_inputs.append(_vi(rhs_name, shape))

        value_info = [_vi("target_out", shape), _vi("graph_out", shape)]
        if initializer_side == "lhs":
            value_info.append(_vi(lhs_name, shape))
        if initializer_side == "rhs":
            value_info.append(_vi(rhs_name, shape))

        reserved_names = {vi.name for vi in graph_inputs} | {vi.name for vi in graph_outputs}
        value_info = [vi for vi in value_info if vi.name not in reserved_names]

        graph = helper.make_graph(
            nodes=nodes,
            name="binary_target_graph",
            inputs=graph_inputs,
            outputs=graph_outputs,
            value_info=value_info,
        )
        model = _wrap_model(graph)

        if initializer_side is not None:
            init_name = lhs_name if initializer_side == "lhs" else rhs_name
            model.set_initializer(init_name, np.ones(shape, dtype=np.float32))

        if mlo:
            model.set_metadata_prop("is_mlo", "1")
            mlo_inputs = [rhs_name] if initializer_side == "rhs" else [lhs_name]
            model.set_metadata_prop("mlo_input_parameter_names", str(mlo_inputs))

        return model

    lhs_name = "lhs_in"
    rhs_name = "rhs_in"
    nodes = []
    graph_inputs = []
    value_info = [_vi("target_out", shape)]
    use_fifo_pre = fifos if fifo_pre is None else fifo_pre
    use_fifo_between = fifos if fifo_between is None else fifo_between
    use_fifo_after = fifos if fifo_after is None else fifo_after
    fifo_between_count = (
        fifo_between_depth if fifo_between_depth is not None else (1 if use_fifo_between else 0)
    )
    fifo_index = 0

    def _append_fifo(prefix: str, inp: str) -> str:
        nonlocal fifo_index
        out_name = f"{prefix}_out{fifo_index}"
        nodes.append(_make_fifo(f"{prefix}_{fifo_index}", inp, out_name, shape))
        _add_value_info(value_info, out_name, shape)
        fifo_index += 1
        return out_name

    def _append_fifo_chain(prefix: str, inp: str, count: int) -> str:
        current = inp
        for _ in range(count):
            current = _append_fifo(prefix, current)
        return current

    lhs_input = lhs_name
    rhs_input = rhs_name
    if initializer_side != "lhs":
        graph_inputs.append(_vi(lhs_name, shape))
        if use_fifo_pre:
            lhs_input = _append_fifo("FIFO_pre", lhs_input)
    else:
        value_info.append(_vi(lhs_name, shape))

    if initializer_side != "rhs":
        graph_inputs.append(_vi(rhs_name, shape))
        if use_fifo_pre:
            rhs_input = _append_fifo("FIFO_pre", rhs_input)
    else:
        value_info.append(_vi(rhs_name, shape))

    nodes.append(_make_add_hls("target_add", lhs_input, rhs_input, "target_out", shape))
    current_out = "target_out"

    if fifo_between_count > 0:
        current_out = _append_fifo_chain("FIFO_mid", current_out, fifo_between_count)

    if include_succ_node:
        nodes.append(_make_dwc("succ_dwc", current_out, "post_0_out", shape))
        _add_value_info(value_info, "post_0_out", shape)
        current_out = "post_0_out"

    for idx in range(extra_post_nodes):
        if fifo_between_count > 0:
            current_out = _append_fifo_chain("FIFO_mid", current_out, fifo_between_count)
        post_name = f"post_out_{idx}"
        nodes.append(_make_dwc(f"post_dwc_{idx}", current_out, post_name, shape))
        _add_value_info(value_info, post_name, shape)
        current_out = post_name

    if use_fifo_after:
        current_out = _append_fifo("FIFO_out", current_out)

    graph_outputs = [_vi(current_out, shape)]

    reserved_names = {vi.name for vi in graph_inputs} | {vi.name for vi in graph_outputs}
    value_info = [vi for vi in value_info if vi.name not in reserved_names]

    graph = helper.make_graph(
        nodes=nodes,
        name="binary_target_graph",
        inputs=graph_inputs,
        outputs=graph_outputs,
        value_info=value_info,
    )
    model = _wrap_model(graph)

    if initializer_side is not None:
        init_name = lhs_name if initializer_side == "lhs" else rhs_name
        model.set_initializer(init_name, np.ones(shape, dtype=np.float32))

    if mlo:
        model.set_metadata_prop("is_mlo", "1")
        mlo_inputs = [rhs_name] if initializer_side == "rhs" else [lhs_name]
        model.set_metadata_prop("mlo_input_parameter_names", str(mlo_inputs))

    return model


def _build_duplicate_target_model(
    fifos: bool = False,
    branch_nodes: bool = False,
    fifo_pre: bool | None = None,
    fifo_between: bool | None = None,
    fifo_after: bool | None = None,
    fifo_between_depth: int | None = None,
) -> ModelWrapper:
    shape = [1, 32, 32, 3]
    nodes = []
    graph_inputs = [_vi("dup_in", shape)]
    value_info = []
    use_fifo_pre = fifos if fifo_pre is None else fifo_pre
    use_fifo_between = fifos if fifo_between is None else fifo_between
    use_fifo_after = fifos if fifo_after is None else fifo_after
    fifo_between_count = (
        fifo_between_depth if fifo_between_depth is not None else (1 if use_fifo_between else 0)
    )

    current_in = "dup_in"
    if use_fifo_pre:
        nodes.append(_make_fifo("FIFO_pre_0", current_in, "fifo_pre_out0", shape))
        _add_value_info(value_info, "fifo_pre_out0", shape)
        current_in = "fifo_pre_out0"

    dup_outs = ["dup_out0", "dup_out1"]
    nodes.append(_make_duplicate_stream("dup_stream", current_in, dup_outs, shape, num_outputs=2))
    _add_value_info(value_info, dup_outs[0], shape)
    _add_value_info(value_info, dup_outs[1], shape)

    branch_outputs = []
    for idx in range(2):
        branch_in = dup_outs[idx]
        if fifo_between_count > 0:
            for chain_idx in range(fifo_between_count):
                fifo_name = f"FIFO_branch_{idx}_{chain_idx}"
                fifo_out = f"fifo_branch_{idx}_{chain_idx}"
                nodes.append(_make_fifo(fifo_name, branch_in, fifo_out, shape))
                _add_value_info(value_info, fifo_out, shape)
                branch_in = fifo_out

        if branch_nodes:
            node_out = f"branch_out_{idx}"
            nodes.append(_make_dwc(f"branch_dwc_{idx}", branch_in, node_out, shape))
            _add_value_info(value_info, node_out, shape)
            branch_in = node_out

        if use_fifo_after:
            fifo_name = f"FIFO_branch_{idx}_after"
            fifo_out = f"fifo_branch_{idx}_after"
            nodes.append(_make_fifo(fifo_name, branch_in, fifo_out, shape))
            _add_value_info(value_info, fifo_out, shape)
            branch_in = fifo_out

        branch_outputs.append(branch_in)

    graph_outputs = [_vi(branch_outputs[0], shape), _vi(branch_outputs[1], shape)]

    reserved_names = {vi.name for vi in graph_inputs} | {vi.name for vi in graph_outputs}
    value_info = [vi for vi in value_info if vi.name not in reserved_names]

    graph = helper.make_graph(
        nodes=nodes,
        name="duplicate_target_graph",
        inputs=graph_inputs,
        outputs=graph_outputs,
        value_info=value_info,
    )
    return _wrap_model(graph)


def _build_mvau_target_model(mlo: bool = False) -> ModelWrapper:
    shape_ifm = [1, 1, 1, 4]
    shape_out = [1, 1, 1, 4]
    shape_w = [4, 4]

    graph = helper.make_graph(
        nodes=[
            _make_mvau_rtl("target_mvau", "ifm", "weights", "mvau_out"),
            _make_dwc("succ_dwc", "mvau_out", "graph_out", shape_out),
        ],
        name="mvau_target_graph",
        inputs=[_vi("ifm", shape_ifm), _vi("weights", shape_w)],
        outputs=[_vi("graph_out", shape_out)],
        value_info=[_vi("mvau_out", shape_out)],
    )
    model = _wrap_model(graph)

    if mlo:
        model.set_metadata_prop("is_mlo", "1")
        model.set_metadata_prop("mlo_input_parameter_names", str(["weights"]))

    return model


class _DuplicateStreamConfig(TypedDict):
    fifos: bool
    branch_nodes: bool
    fifo_pre: bool
    fifo_between: bool
    fifo_after: bool
    fifo_between_depth: int | None
    expected_input_node: bool
    expected_output_node: bool


_DUPLICATE_STREAM_CONFIGS: list[_DuplicateStreamConfig] = [
    {
        "fifos": False,
        "branch_nodes": False,
        "fifo_pre": False,
        "fifo_between": False,
        "fifo_after": False,
        "fifo_between_depth": None,
        "expected_input_node": True,
        "expected_output_node": True,
    },
    {
        "fifos": False,
        "branch_nodes": False,
        "fifo_pre": True,
        "fifo_between": True,
        "fifo_after": False,
        "fifo_between_depth": 2,
        "expected_input_node": True,
        "expected_output_node": True,
    },
    {
        "fifos": False,
        "branch_nodes": True,
        "fifo_pre": False,
        "fifo_between": False,
        "fifo_after": False,
        "fifo_between_depth": None,
        "expected_input_node": True,
        "expected_output_node": False,
    },
    {
        "fifos": False,
        "branch_nodes": True,
        "fifo_pre": True,
        "fifo_between": True,
        "fifo_after": True,
        "fifo_between_depth": 2,
        "expected_input_node": True,
        "expected_output_node": False,
    },
]


def _assert_isolated_model(
    isolated_model: ModelWrapper,
    source_model: ModelWrapper | None,
    target_name: str,
    expected_graph_inputs: list[str],
    expected_graph_outputs: list[str],
    expected_initializer_inputs: list[str],
    expected_input_node_flag: bool,
    expected_target_inputs: list[str],
    expected_target_outputs: list[str],
    expected_output_node_flag: bool = False,
) -> None:
    reference_model = isolated_model if source_model is None else source_model

    def _resolve_through_fifos(tensor_name: str) -> str:
        name = tensor_name
        if name.endswith("_dummy"):
            name = name[: -len("_dummy")]
        while True:
            producer = reference_model.find_producer(name)
            if producer is None:
                return name
            if producer.op_type == "StreamingFIFO" or "FIFO" in producer.name:
                name = producer.input[0]
                continue
            return name

    graph = isolated_model.graph
    graph_input_names = [x.name for x in graph.input]
    graph_output_names = [x.name for x in graph.output]

    resolved_graph_inputs = [_resolve_through_fifos(name) for name in graph_input_names]
    resolved_graph_outputs = [_resolve_through_fifos(name) for name in graph_output_names]
    resolved_expected_graph_inputs = [
        _resolve_through_fifos(name) for name in expected_graph_inputs
    ]
    resolved_expected_graph_outputs = [
        _resolve_through_fifos(name) for name in expected_graph_outputs
    ]

    assert resolved_graph_inputs == resolved_expected_graph_inputs
    assert resolved_graph_outputs == resolved_expected_graph_outputs

    input_dummy_nodes = [
        n for n in graph.node if n.op_type == "RemoveDataPath_rtl" and "_input_dummy_" in n.name
    ]
    output_dummy_nodes = [
        n for n in graph.node if n.op_type == "RemoveDataPath_rtl" and "_output_dummy_" in n.name
    ]
    target_nodes = [n for n in graph.node if n.name == target_name]

    assert len(target_nodes) == 1
    assert len(input_dummy_nodes) == len(expected_graph_inputs)
    assert len(output_dummy_nodes) == len(expected_graph_outputs)

    initializer_names = [x.name for x in graph.initializer]
    assert initializer_names == expected_initializer_inputs

    target_node = target_nodes[0]
    resolved_target_inputs = [_resolve_through_fifos(inp) for inp in target_node.input]
    resolved_target_outputs = [_resolve_through_fifos(outp) for outp in target_node.output]
    resolved_expected_target_inputs = [
        _resolve_through_fifos(inp) for inp in expected_target_inputs
    ]
    resolved_expected_target_outputs = [
        _resolve_through_fifos(outp) for outp in expected_target_outputs
    ]

    assert resolved_target_inputs == resolved_expected_target_inputs
    assert resolved_target_outputs == resolved_expected_target_outputs

    target_dummy_inputs = [inp for inp in target_node.input if inp.endswith("_dummy")]
    target_initializer_inputs = [inp for inp in target_node.input if inp in initializer_names]
    assert len(target_dummy_inputs) == len(expected_graph_inputs)
    assert target_initializer_inputs == expected_initializer_inputs

    assert isolated_model.get_metadata_prop("predecessors") == str(expected_graph_inputs)
    assert isolated_model.get_metadata_prop("successors") == str(graph_output_names)
    assert isolated_model.get_metadata_prop("input_node") == str(expected_input_node_flag).lower()
    assert isolated_model.get_metadata_prop("output_node") == str(expected_output_node_flag).lower()


def _isolate_node_model(builder: _SimulationBuilderProtocol, by_node: int | str) -> ModelWrapper:
    return builder._isolated_node_model(by_node)  # noqa: SLF001


def _assert_isolated_models_match(
    isolated_no_fifo: ModelWrapper,
    isolated_fifo: ModelWrapper,
    target_name: str,
) -> None:
    assert [inp.name for inp in isolated_no_fifo.graph.input] == [
        inp.name for inp in isolated_fifo.graph.input
    ]
    assert [out.name for out in isolated_no_fifo.graph.output] == [
        out.name for out in isolated_fifo.graph.output
    ]

    def _node_names(model: ModelWrapper, op_type: str) -> list[str]:
        return [node.name for node in model.graph.node if node.op_type == op_type]

    assert _node_names(isolated_no_fifo, "RemoveDataPath_rtl") == _node_names(
        isolated_fifo, "RemoveDataPath_rtl"
    )

    node_no_fifo = next(node for node in isolated_no_fifo.graph.node if node.name == target_name)
    node_fifo = next(node for node in isolated_fifo.graph.node if node.name == target_name)
    assert node_no_fifo.op_type == node_fifo.op_type
    assert list(node_no_fifo.input) == list(node_fifo.input)
    assert list(node_no_fifo.output) == list(node_fifo.output)


@pytest.mark.parametrize(
    "pre_binary,succ_binary",
    [
        (False, False),
        (True, True),
    ],
)
def test_isolated_node_model_unary_target_with_varied_other_node_inputs(
    pre_binary: bool, succ_binary: bool
) -> None:
    """Isolate unary target with unary/binary surrounding nodes."""
    model = _build_unary_target_model(pre_binary=pre_binary, succ_binary=succ_binary)
    builder = SimulationBuilder(model, "xc7z020clg400-1", 5.0, "test_isolated_1_")

    isolated = _isolate_node_model(builder, 1)

    _assert_isolated_model(
        isolated_model=isolated,
        source_model=None,
        target_name="target_dwc",
        expected_graph_inputs=["target_in"],
        expected_graph_outputs=["target_out"],
        expected_initializer_inputs=[],
        expected_input_node_flag=False,
        expected_target_inputs=["target_in_dummy"],
        expected_target_outputs=["target_out_dummy"],
    )


def test_isolated_node_model_select_by_name() -> None:
    """Selecting node by name returns the correct isolated model."""
    model = _build_unary_target_model(pre_binary=False, succ_binary=False)
    builder = SimulationBuilder(model, "xc7z020clg400-1", 5.0, "test_isolated_2_")

    isolated = _isolate_node_model(builder, "target_dwc")

    _assert_isolated_model(
        isolated_model=isolated,
        source_model=None,
        target_name="target_dwc",
        expected_graph_inputs=["target_in"],
        expected_graph_outputs=["target_out"],
        expected_initializer_inputs=[],
        expected_input_node_flag=False,
        expected_target_inputs=["target_in_dummy"],
        expected_target_outputs=["target_out_dummy"],
    )


@pytest.mark.parametrize(
    "initializer_side,mlo,expected_graph_inputs,expected_initializer_inputs,expected_target_inputs",
    [
        (None, False, ["lhs_in", "rhs_in"], [], ["lhs_in_dummy", "rhs_in_dummy"]),
        ("rhs", False, ["lhs_in"], ["rhs_in"], ["lhs_in_dummy", "rhs_in"]),
        ("lhs", False, ["rhs_in"], ["lhs_in"], ["lhs_in", "rhs_in_dummy"]),
        ("rhs", True, ["lhs_in"], ["rhs_in"], ["lhs_in_dummy", "rhs_in"]),
        (None, True, ["rhs_in"], ["lhs_in"], ["lhs_in", "rhs_in_dummy"]),
    ],
)
def test_isolated_node_model_binary_target_with_dynamic_and_fixed_inputs(
    initializer_side: str | None,
    mlo: bool,
    expected_graph_inputs: list[str],
    expected_initializer_inputs: list[str],
    expected_target_inputs: list[str],
) -> None:
    """Isolate binary target for dynamic/fixed lhs-rhs and MLO/non-MLO cases."""
    model = _build_binary_target_model(initializer_side=initializer_side, mlo=mlo)
    builder = SimulationBuilder(model, "xc7z020clg400-1", 5.0, "test_isolated_3_")

    isolated = _isolate_node_model(builder, 0)

    _assert_isolated_model(
        isolated_model=isolated,
        source_model=None,
        target_name="target_add",
        expected_graph_inputs=expected_graph_inputs,
        expected_graph_outputs=["target_out"],
        expected_initializer_inputs=expected_initializer_inputs,
        expected_input_node_flag=True,
        expected_target_inputs=expected_target_inputs,
        expected_target_outputs=["target_out_dummy"],
    )


@pytest.mark.parametrize("fifo_between_depth", [1, 2])
def test_isolated_node_model_unary_succ_fifo_chain_transparency(
    fifo_between_depth: int,
) -> None:
    """FIFOs between target and successor are transparent for isolation checks."""
    model = _build_unary_target_model(
        pre_binary=False,
        succ_binary=False,
        include_succ_node=True,
        fifo_between=True,
        fifo_between_depth=fifo_between_depth,
    )
    builder = SimulationBuilder(model, "xc7z020clg400-1", 5.0, "test_isolated_4_")

    isolated = _isolate_node_model(builder, "succ_dwc")

    _assert_isolated_model(
        isolated_model=isolated,
        source_model=model,
        target_name="succ_dwc",
        expected_graph_inputs=["target_out"],
        expected_graph_outputs=["post_0_out"],
        expected_initializer_inputs=[],
        expected_input_node_flag=False,
        expected_target_inputs=["target_out_dummy"],
        expected_target_outputs=["post_0_out_dummy"],
        expected_output_node_flag=True,
    )


@pytest.mark.parametrize("fifo_between_depth", [1, 2])
def test_isolated_node_model_binary_succ_fifo_chain_transparency(
    fifo_between_depth: int,
) -> None:
    """Binary successor nodes see FIFO chains as transparent."""
    model = _build_binary_target_model(
        initializer_side=None,
        mlo=False,
        include_succ_node=True,
        fifo_between=True,
        fifo_between_depth=fifo_between_depth,
    )
    builder = SimulationBuilder(model, "xc7z020clg400-1", 5.0, "test_isolated_5_")

    isolated = _isolate_node_model(builder, "succ_dwc")

    _assert_isolated_model(
        isolated_model=isolated,
        source_model=model,
        target_name="succ_dwc",
        expected_graph_inputs=["target_out"],
        expected_graph_outputs=["post_0_out"],
        expected_initializer_inputs=[],
        expected_input_node_flag=False,
        expected_target_inputs=["target_out_dummy"],
        expected_target_outputs=["post_0_out_dummy"],
        expected_output_node_flag=True,
    )


@pytest.mark.parametrize(
    "initializer_side,expected_graph_inputs,expected_initializer_inputs,expected_target_inputs",
    [
        (None, ["lhs_in", "rhs_in"], [], ["lhs_in_dummy", "rhs_in_dummy"]),
        ("rhs", ["lhs_in"], ["rhs_in"], ["lhs_in_dummy", "rhs_in"]),
    ],
)
def test_isolated_node_model_binary_target_fifo_pre_transparency(
    initializer_side: str | None,
    expected_graph_inputs: list[str],
    expected_initializer_inputs: list[str],
    expected_target_inputs: list[str],
) -> None:
    """FIFO chains before the target are transparent for inputs."""
    model = _build_binary_target_model(
        initializer_side=initializer_side,
        mlo=False,
        include_succ_node=True,
        fifo_pre=True,
        fifo_between=True,
        fifo_between_depth=2,
    )
    builder = SimulationBuilder(model, "xc7z020clg400-1", 5.0, "test_isolated_6_")

    isolated = _isolate_node_model(builder, "target_add")

    _assert_isolated_model(
        isolated_model=isolated,
        source_model=model,
        target_name="target_add",
        expected_graph_inputs=expected_graph_inputs,
        expected_graph_outputs=["target_out"],
        expected_initializer_inputs=expected_initializer_inputs,
        expected_input_node_flag=True,
        expected_target_inputs=expected_target_inputs,
        expected_target_outputs=["target_out_dummy"],
        expected_output_node_flag=False,
    )

    isolated = _isolate_node_model(builder, "succ_dwc")

    _assert_isolated_model(
        isolated_model=isolated,
        source_model=model,
        target_name="succ_dwc",
        expected_graph_inputs=["target_out"],
        expected_graph_outputs=["post_0_out"],
        expected_initializer_inputs=[],
        expected_input_node_flag=False,
        expected_target_inputs=["target_out_dummy"],
        expected_target_outputs=["post_0_out_dummy"],
        expected_output_node_flag=True,
    )


@pytest.mark.parametrize("config", _DUPLICATE_STREAM_CONFIGS)
def test_isolated_node_model_duplicate_stream_fifo_transparency(
    config: _DuplicateStreamConfig,
) -> None:
    """DuplicateStreams models behave identically with FIFO chains present."""
    model = _build_duplicate_target_model(
        fifos=config["fifos"],
        branch_nodes=config["branch_nodes"],
        fifo_pre=config["fifo_pre"],
        fifo_between=config["fifo_between"],
        fifo_after=config["fifo_after"],
        fifo_between_depth=config["fifo_between_depth"],
    )
    builder = SimulationBuilder(model, "xc7z020clg400-1", 5.0, "test_isolated_7_")

    isolated = _isolate_node_model(builder, "dup_stream")

    _assert_isolated_model(
        isolated_model=isolated,
        source_model=model,
        target_name="dup_stream",
        expected_graph_inputs=["dup_in"],
        expected_graph_outputs=["dup_out0", "dup_out1"],
        expected_initializer_inputs=[],
        expected_input_node_flag=config["expected_input_node"],
        expected_target_inputs=["dup_in_dummy"],
        expected_target_outputs=["dup_out0_dummy", "dup_out1_dummy"],
        expected_output_node_flag=config["expected_output_node"],
    )


def test_isolated_node_model_fifo_transparency_nodes() -> None:
    """Compare isolated node inputs/outputs between FIFO and non-FIFO topologies for all nodes."""
    model_no_fifo = _build_unary_target_model(
        pre_binary=True,
        succ_binary=False,
        include_succ_node=True,
        extra_post_nodes=1,
        fifo_pre=False,
        fifo_between=False,
        fifo_after=False,
    )
    model_fifo = _build_unary_target_model(
        pre_binary=True,
        succ_binary=False,
        include_succ_node=True,
        extra_post_nodes=1,
        fifo_pre=True,
        fifo_between=True,
        fifo_between_depth=2,
        fifo_after=True,
    )

    builder_no_fifo = SimulationBuilder(model_no_fifo, "xc7z020clg400-1", 5.0, "test_isolated_8_")
    builder_fifo = SimulationBuilder(model_fifo, "xc7z020clg400-1", 5.0, "test_isolated_9_")

    node_names = [node.name for node in model_no_fifo.graph.node if node.op_type != "StreamingFIFO"]

    for node_name in node_names:
        print(f"Comparing isolated models for node '{node_name}'...")
        isolated_no_fifo = _isolate_node_model(builder_no_fifo, node_name)
        isolated_fifo = _isolate_node_model(builder_fifo, node_name)

        _assert_isolated_models_match(
            isolated_no_fifo=isolated_no_fifo,
            isolated_fifo=isolated_fifo,
            target_name=node_name,
        )


def test_isolated_node_model_elementwise_sets_const_style_for_mlo_initializer() -> None:
    """Elementwise ops set lhs_style/rhs_style=const for remapped MLO initializer inputs."""
    model = _build_binary_target_model(initializer_side=None, mlo=True)
    builder = SimulationBuilder(model, "xc7z020clg400-1", 5.0, "test_isolated_10_")

    isolated = _isolate_node_model(builder, 0)
    target_node = next(n for n in isolated.graph.node if n.name == "target_add")
    attrs = {attr.name: helper.get_attribute_value(attr) for attr in target_node.attribute}
    lhs_style = (
        attrs["lhs_style"].decode() if isinstance(attrs["lhs_style"], bytes) else attrs["lhs_style"]
    )
    rhs_style = (
        attrs["rhs_style"].decode() if isinstance(attrs["rhs_style"], bytes) else attrs["rhs_style"]
    )

    assert lhs_style == "const"
    assert rhs_style == "input"


def test_isolated_node_model_mvau_sets_internal_decoupled_for_initializer_input() -> None:
    """MVAU ops set mem_mode=internal_decoupled when an input is remapped to initializer."""
    model = _build_mvau_target_model(mlo=True)
    builder = SimulationBuilder(model, "xc7z020clg400-1", 5.0, "test_isolated_11_")

    isolated = _isolate_node_model(builder, 0)
    target_node = next(n for n in isolated.graph.node if n.name == "target_mvau")
    attrs = {attr.name: helper.get_attribute_value(attr) for attr in target_node.attribute}
    mem_mode = (
        attrs["mem_mode"].decode() if isinstance(attrs["mem_mode"], bytes) else attrs["mem_mode"]
    )

    assert mem_mode == "internal_decoupled"


def test_isolated_node_model_rejects_bad_mlo_metadata() -> None:
    """Reject invalid mlo_input_parameter_names metadata values."""
    model = _build_binary_target_model(initializer_side="rhs", mlo=False)
    model.set_metadata_prop("is_mlo", "1")
    model.set_metadata_prop("mlo_input_parameter_names", "42")
    builder = SimulationBuilder(model, "xc7z020clg400-1", 5.0, "test_isolated_12_")

    with pytest.raises(FINNInternalError, match="mlo_input_parameter_names"):
        _isolate_node_model(builder, 0)

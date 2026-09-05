# Copyright (C) 2024, Advanced Micro Devices, Inc.
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

"""Tests for splitting large FIFOs in dataflow graphs."""

import pytest

import numpy as np
from onnx import NodeProto, TensorProto
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model
from typing import Literal, cast

from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.set_fifo_depths import SplitLargeFIFOs, get_fifo_split_configs
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


def _make_elementwise_add(
    name: str,
    lhs: str,
    rhs: str,
    out: str,
    shape: list[int],
    dtype: str = "INT8",
    lhs_style: str = "input",
    rhs_style: str = "input",
) -> NodeProto:
    return oh.make_node(
        "ElementwiseAdd",
        [lhs, rhs],
        [out],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        lhs_dtype=dtype,
        rhs_dtype=dtype,
        out_dtype=dtype,
        lhs_shape=list(shape),
        rhs_shape=list(shape),
        out_shape=list(shape),
        lhs_style=lhs_style,
        rhs_style=rhs_style,
        PE=1,
        name=name,
    )


def _build_elementwise_add_model() -> ModelWrapper:
    shape = [1, 4]
    inp0 = oh.make_tensor_value_info("inp0", TensorProto.INT8, shape)
    rhs0 = oh.make_tensor_value_info("rhs0", TensorProto.INT8, shape)
    rhs1 = oh.make_tensor_value_info("rhs1", TensorProto.INT8, shape)
    mid = oh.make_tensor_value_info("mid", TensorProto.INT8, shape)
    out = oh.make_tensor_value_info("out", TensorProto.INT8, shape)
    nodes = [
        _make_elementwise_add("add_0", "inp0", "rhs0", "mid", shape, rhs_style="const"),
        _make_elementwise_add("add_1", "mid", "rhs1", "out", shape, rhs_style="const"),
    ]
    graph = oh.make_graph(
        nodes=nodes,
        inputs=[inp0],
        outputs=[out],
        value_info=[mid, rhs0, rhs1],
        name="two_elementwise_adds",
    )
    model = ModelWrapper(qonnx_make_model(graph, producer_name="test_split_large_fifos"))
    model.set_initializer("rhs0", np.ones(shape, dtype=np.int8))
    model.set_initializer("rhs1", np.ones(shape, dtype=np.int8))
    for tensor_name in ["inp0", "rhs0", "rhs1", "mid", "out"]:
        model.set_tensor_datatype(tensor_name, DataType["INT8"])
    return model


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
@pytest.mark.parametrize("depth", [16384, 65536, 45000, 1537])
def test_split_large_fifos(depth: Literal[16384, 65536, 45000, 1537]) -> None:
    """Split oversized FIFOs into supported power-of-two depths."""
    model = _build_elementwise_add_model()
    model = model.transform(SpecializeLayers("xcvm1802-vsvd1760-2MP-e-S"))
    for node in model.graph.node:
        n = getCustomOp(node)
        n.set_nodeattr("inFIFODepths", [depth])
        n.set_nodeattr("outFIFODepths", [depth])
    model = model.transform(InsertFIFO(True, 256, "auto"))
    model = model.transform(SplitLargeFIFOs(256, 32768))
    for node in model.get_nodes_by_op_type("StreamingFIFO_rtl"):
        n = getCustomOp(node)
        # Each FIFO needs to be a power of 2 in depth
        assert (
            cast("int", n.get_nodeattr("depth")) & (cast("int", n.get_nodeattr("depth")) - 1) == 0
        ), f"FIFO depth {n.get_nodeattr('depth')} is not a power of 2"


def test_split_large_fifo_configs() -> None:
    """Validate FIFO split configurations for fixed depth inputs."""
    ret0 = get_fifo_split_configs(513, 256, 32768)
    assert ret0 == [(512, "vivado"), (2, "rtl")]
    ret1 = get_fifo_split_configs(1200, 256, 32768)
    assert ret1 == [(1024, "vivado"), (176, "rtl")]
    ret2 = get_fifo_split_configs(45000, 256, 32768)
    assert ret2 == [
        (32768, "vivado"),
        (8192, "vivado"),
        (2048, "vivado"),
        (1024, "vivado"),
        (512, "vivado"),
        (256, "rtl"),
        (200, "rtl"),
    ]

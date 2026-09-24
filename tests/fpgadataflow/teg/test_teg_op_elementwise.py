# Copyright (C) 2026, Paderborn University
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
# * Neither the name of FINN nor the names of its
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

"""Differential test of the HLS ElementwiseBinaryOperation template (join node) against XSI."""

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.analysis.fpgadataflow.teg.patterns import StallPattern
from finn.transformation.fpgadataflow.convert_to_hw.elementwise_binary_operation import (
    InferElementwiseBinaryOperation,
)
from finn.util.basic import getHWCustomOp
from tests.fpgadataflow.teg.stall_injection import (
    is_prepared,
    prepare_node_rtlsim,
    run_abstract,
    run_xsi,
    tokens_in,
    tokens_out,
)
from tests.fpgadataflow.teg.trace_compare import assert_traces_equal, compare_traces

TEST_FPGA_PART = "xc7z020clg400-1"
TARGET_CLK_NS = 5.0

pytestmark = [
    pytest.mark.fpgadataflow,
    pytest.mark.fifo_model,
    pytest.mark.vivado,
    pytest.mark.slow,
]


def make_elementwise_model(
    op: str, lhs_shape: list[int], rhs_shape: list[int], pe: int, rhs_const: bool
) -> ModelWrapper:
    """Single ``Elementwise<op>`` node; ``rhs_const`` turns the right operand into an
    embedded parameter (one stream input), else both operands are streams."""
    dt = DataType["INT4"]
    out_shape = list(np.broadcast_shapes(lhs_shape, rhs_shape))
    node = helper.make_node(op, ["in_x", "in_y"], ["out"])
    lhs = helper.make_tensor_value_info("in_x", TensorProto.FLOAT, lhs_shape)
    rhs = helper.make_tensor_value_info("in_y", TensorProto.FLOAT, rhs_shape)
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)
    graph = helper.make_graph([node], inputs=[lhs, rhs], outputs=[out], name="eltwise")
    model = ModelWrapper(qonnx_make_model(graph, producer_name="teg-eltwise"))
    model.set_tensor_datatype("in_x", dt)
    model.set_tensor_datatype("in_y", dt)
    model.set_tensor_datatype("out", DataType["INT8"])
    if rhs_const:
        model.set_initializer("in_y", gen_finn_dt_tensor(dt, rhs_shape))
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(InferElementwiseBinaryOperation())
    assert model.graph.node[0].op_type == f"Elementwise{op}"
    inst = getHWCustomOp(model.graph.node[0])
    inst.set_nodeattr("PE", pe)
    inst.set_nodeattr("preferred_impl_style", "hls")
    return model


_PREPARED: dict[tuple, ModelWrapper] = {}


def prepared_model(cfg: tuple) -> ModelWrapper:
    """Prepare (and cache per configuration) the XSI library of the node."""
    if not is_prepared(_PREPARED, cfg):
        op, lhs, rhs, pe, const = cfg
        _PREPARED[cfg] = prepare_node_rtlsim(
            make_elementwise_model(op, list(lhs), list(rhs), pe, const),
            TEST_FPGA_PART,
            TARGET_CLK_NS,
        )
    return _PREPARED[cfg]


CONFIGS = [
    # (op, lhs shape, rhs shape, PE, rhs const)
    ("Add", (1, 3, 3, 8), (1, 3, 3, 8), 2, False),  # ResNet residual add
    ("Add", (1, 3, 3, 8), (1, 3, 3, 8), 8, False),
    ("Add", (1, 2, 2, 6), (6,), 3, False),  # broadcast operand: read once per pixel
    ("Mul", (1, 2, 2, 8), (8,), 2, True),  # embedded constant (single stream)
]


def _cfg_id(c: tuple) -> str:
    lhs, rhs = "x".join(map(str, c[1])), "x".join(map(str, c[2]))
    return f"{c[0]}_{lhs}_{rhs}_PE{c[3]}{'_const' if c[4] else ''}"


def stall_kinds_join(seed: int) -> dict[str, tuple[list[StallPattern], StallPattern]]:
    """Patterns for the two operand streams and the output."""
    return {
        "free": ([StallPattern.none(), StallPattern.none()], StallPattern.none()),
        "in0_bernoulli": (
            [StallPattern.bernoulli(0.3, seed), StallPattern.none()],
            StallPattern.none(),
        ),
        "in1_bernoulli": (
            [StallPattern.none(), StallPattern.bernoulli(0.3, seed + 1)],
            StallPattern.none(),
        ),
        "out_bernoulli": (
            [StallPattern.none(), StallPattern.none()],
            StallPattern.bernoulli(0.3, seed + 2),
        ),
        "all_bernoulli": (
            [StallPattern.bernoulli(0.3, seed + 3), StallPattern.bernoulli(0.4, seed + 4)],
            StallPattern.bernoulli(0.4, seed + 5),
        ),
        "out_single_stall": (
            [StallPattern.none(), StallPattern.none()],
            StallPattern.single_stall(12, 25),
        ),
    }


#: like the HLS MVAU: the flp pipeline shows bubbles under simultaneous input starvation and
#: output back-pressure that a fixed TEG does not reproduce (1-2 cycles)
KNOWN_DEVIATIONS = {"all_bernoulli"}


@pytest.mark.parametrize("cfg", CONFIGS, ids=_cfg_id)
@pytest.mark.parametrize("kind", list(stall_kinds_join(0)))
def test_teg_op_elementwise_hls(
    cfg: tuple, kind: str, finn_test_seed: int, request: pytest.FixtureRequest
) -> None:
    """XSI and the abstract model must produce identical handshake traces on all streams."""
    if kind in KNOWN_DEVIATIONS:
        request.applymarker(
            pytest.mark.xfail(
                strict=False, reason="flp pipeline: bubbles under back-pressure (not a fixed TEG)"
            )
        )
    model = prepared_model(cfg)
    in_pats, out_pat = stall_kinds_join(finn_test_seed)[kind]
    if cfg[4]:
        in_pats = in_pats[:1]
    frames = 2
    xsi = run_xsi(model, frames, in_pats, out_pat)
    model_in, model_out = run_abstract(model, frames, in_pats, out_pat)
    xsi_in, xsi_out = xsi.relative()
    inst = getHWCustomOp(model.graph.node[0])
    for key in sorted(xsi_in):
        assert len(xsi_in[key]) == tokens_in(inst, int(key[2:])) * frames, key
    assert len(xsi_out["out0"]) == tokens_out(inst) * frames
    print(
        f"\n{_cfg_id(cfg)} {kind}: XSI {xsi.cycles} cycles, first "
        f"{[v[0] for v in xsi_in.values()]}/{xsi_out['out0'][0]} (model "
        f"{[v[0] for v in model_in.values()]}/{model_out['out0'][0]}), last "
        f"{[v[-1] for v in xsi_in.values()]}/{xsi_out['out0'][-1]} (model "
        f"{[v[-1] for v in model_in.values()]}/{model_out['out0'][-1]})"
    )
    diffs = compare_traces(xsi_in, xsi_out, model_in, model_out)
    assert_traces_equal(diffs, f"Elementwise {_cfg_id(cfg)} stall={kind}")

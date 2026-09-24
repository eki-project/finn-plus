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

"""Differential test of the StreamingDataWidthConverter templates (RTL and HLS) against XSI."""

import pytest

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from finn.analysis.fpgadataflow.teg.hls_report import hls_loop_params
from finn.util.basic import getHWCustomOp
from tests.fpgadataflow.teg.stall_injection import (
    prepare_node_rtlsim,
    run_abstract,
    run_xsi,
    stall_kinds,
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


def make_dwc_model(
    shape: list[int], in_width: int, out_width: int, impl_style: str
) -> ModelWrapper:
    """Single ``StreamingDataWidthConverter`` node over a UINT4 tensor."""
    dt = DataType["UINT4"]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)
    node = helper.make_node(
        "StreamingDataWidthConverter",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        inShape=shape,
        outShape=shape,
        inWidth=in_width,
        outWidth=out_width,
        dataType=dt.name,
        preferred_impl_style=impl_style,
    )
    graph = helper.make_graph(nodes=[node], name="dwc_graph", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="teg-dwc"))
    model.set_tensor_datatype("inp", dt)
    model.set_tensor_datatype("outp", dt)
    return model


_PREPARED: dict[tuple, ModelWrapper] = {}


def prepared_model(cfg: tuple) -> ModelWrapper:
    """Prepare (and cache per configuration) the XSI library of the node."""
    if cfg not in _PREPARED:
        shape, iw, ow, impl = cfg
        _PREPARED[cfg] = prepare_node_rtlsim(
            make_dwc_model(list(shape), iw, ow, impl), TEST_FPGA_PART, TARGET_CLK_NS
        )
    return _PREPARED[cfg]


CONFIGS = [
    # (shape, inWidth, outWidth, impl_style)
    ((1, 2, 2, 12), 4, 12, "rtl"),  # up, K = 3
    ((1, 2, 2, 12), 12, 4, "rtl"),  # down, K = 3
    ((1, 24), 8, 16, "rtl"),  # up, K = 2
    ((1, 24), 16, 8, "rtl"),  # down, K = 2
    ((1, 2, 2, 12), 4, 12, "hls"),  # up
    ((1, 2, 2, 12), 12, 4, "hls"),  # down
    ((1, 24), 8, 12, "hls"),  # non-integer ratio: LCM stage
]


def _cfg_id(c: tuple) -> str:
    return f"{c[3]}_{c[1]}to{c[2]}_{'x'.join(map(str, c[0]))}"


#: known deviation of the RTL down-converter model (see ``dwc_rtl.py``): input bubbles while
#: the output is blocked let the RTL accept the next word one handshake later than the model
KNOWN_DEVIATIONS = {("both_bursty", "rtl", "down")}


@pytest.mark.parametrize("cfg", CONFIGS, ids=_cfg_id)
@pytest.mark.parametrize("kind", list(stall_kinds(0)))
def test_teg_op_dwc(
    cfg: tuple, kind: str, finn_test_seed: int, request: pytest.FixtureRequest
) -> None:
    """XSI and the abstract model must produce identical handshake traces."""
    direction = "down" if cfg[1] > cfg[2] else "up"
    if (kind, cfg[3], direction) in KNOWN_DEVIATIONS:
        request.applymarker(
            pytest.mark.xfail(
                strict=False, reason="RTL down-converter BRdy hold after an input bubble"
            )
        )
    model = prepared_model(cfg)
    in_pat, out_pat = stall_kinds(finn_test_seed)[kind]
    frames = 2
    xsi = run_xsi(model, frames, in_pat, out_pat)
    model_in, model_out = run_abstract(model, frames, in_pat, out_pat)
    xsi_in, xsi_out = xsi.relative()
    inst = getHWCustomOp(model.graph.node[0])
    n_in, n_out = tokens_in(inst) * frames, tokens_out(inst) * frames
    assert len(xsi_in["in0"]) == n_in and len(xsi_out["out0"]) == n_out
    params = hls_loop_params(inst) if cfg[3] == "hls" else None
    stages = (
        ""
        if params is None
        else (
            f" L={params.depth} r={params.read_stage} w={params.write_stage}"
            f" rw={params.rewind_delay}"
        )
    )
    print(
        f"\n{_cfg_id(cfg)} {kind}:{stages} {n_in} in / {n_out} out, XSI {xsi.cycles} cycles, "
        f"first "
        f"in/out {xsi_in['in0'][0]}/{xsi_out['out0'][0]} (model "
        f"{model_in['in0'][0]}/{model_out['out0'][0]}), last {xsi_in['in0'][-1]}/"
        f"{xsi_out['out0'][-1]} (model {model_in['in0'][-1]}/{model_out['out0'][-1]})"
    )
    diffs = compare_traces(xsi_in, xsi_out, model_in, model_out)
    assert_traces_equal(diffs, f"DWC {_cfg_id(cfg)} stall={kind} in={in_pat!r} out={out_pat!r}")

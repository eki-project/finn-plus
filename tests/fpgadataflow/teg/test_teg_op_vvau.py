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

"""Differential test of the HLS VVAU template (depthwise convolution) against XSI."""

import pytest

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

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


def make_vvau_model(
    k: int, channels: int, dim: int, pe: int, simd: int, mem_mode: str
) -> ModelWrapper:
    """Single ``VVAU`` node without activation (``noActivation = 1``)."""
    idt, wdt, odt = DataType["UINT4"], DataType["INT4"], DataType["INT16"]
    in_shape = [1, dim, dim, k * k * channels]
    out_shape = [1, dim, dim, channels]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape)
    node = helper.make_node(
        "VVAU",
        ["inp", "weights"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        PE=pe,
        SIMD=simd,
        Dim=[dim, dim],
        Channels=channels,
        Kernel=[k, k],
        resType="lut",
        ActVal=0,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        noActivation=1,
        mem_mode=mem_mode,
        preferred_impl_style="hls",
    )
    graph = helper.make_graph(nodes=[node], name="vvau", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="teg-vvau"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)
    model.set_initializer("weights", gen_finn_dt_tensor(wdt, (channels, 1, k, k)))
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


_PREPARED: dict[tuple, ModelWrapper] = {}


def prepared_model(cfg: tuple) -> ModelWrapper:
    """Prepare (and cache per configuration) the XSI library of the node."""
    if cfg not in _PREPARED:
        _PREPARED[cfg] = prepare_node_rtlsim(make_vvau_model(*cfg), TEST_FPGA_PART, TARGET_CLK_NS)
    return _PREPARED[cfg]


CONFIGS = [
    # (kernel, channels, dim, PE, SIMD, mem_mode)
    (3, 4, 2, 2, 1, "internal_embedded"),  # SF = 9, NF = 2
    (3, 4, 2, 4, 1, "internal_decoupled"),  # MobileNet style: PE = channels, SIMD = 1
    (2, 4, 3, 2, 2, "internal_embedded"),  # SIMD > 1
]


def _cfg_id(c: tuple) -> str:
    return f"k{c[0]}_C{c[1]}_D{c[2]}_PE{c[3]}_SIMD{c[4]}_{c[5].split('_')[1]}"


#: like the HLS MVAU: the flp pipeline shows bubbles under back-pressure that a fixed TEG
#: does not reproduce (1-3 cycles); with streamed weights also after a plain output stall
KNOWN_DEVIATIONS = {"both_bernoulli", "both_bursty"}
KNOWN_DEVIATIONS_DECOUPLED = KNOWN_DEVIATIONS | {"out_single_stall", "out_bernoulli"}


@pytest.mark.parametrize("cfg", CONFIGS, ids=_cfg_id)
@pytest.mark.parametrize("kind", list(stall_kinds(0)))
def test_teg_op_vvau_hls(
    cfg: tuple, kind: str, finn_test_seed: int, request: pytest.FixtureRequest
) -> None:
    """XSI and the abstract model must produce identical handshake traces."""
    deviations = KNOWN_DEVIATIONS_DECOUPLED if cfg[5] == "internal_decoupled" else KNOWN_DEVIATIONS
    if kind in deviations:
        request.applymarker(
            pytest.mark.xfail(
                strict=False, reason="flp pipeline: bubbles under back-pressure (not a fixed TEG)"
            )
        )
    model = prepared_model(cfg)
    in_pat, out_pat = stall_kinds(finn_test_seed)[kind]
    frames = 2
    inst = getHWCustomOp(model.graph.node[0])
    extra = {}
    if cfg[5] == "internal_decoupled":
        # weight stream of the memstream: one PE x SIMD word per loop iteration, i.e. SF
        # words per output token (vvau.hpp:110-157)
        sf = cfg[0] * cfg[0] // cfg[4]
        extra = {"in1_V": tokens_out(inst) * sf}
    xsi = run_xsi(model, frames, in_pat, out_pat, extra_inputs=extra)
    model_in, model_out = run_abstract(model, frames, in_pat, out_pat)
    xsi_in, xsi_out = xsi.relative()
    n_in, n_out = tokens_in(inst) * frames, tokens_out(inst) * frames
    assert len(xsi_in["in0"]) == n_in and len(xsi_out["out0"]) == n_out
    print(
        f"\n{_cfg_id(cfg)} {kind}: {n_in} in / {n_out} out, XSI {xsi.cycles} cycles, first "
        f"in/out {xsi_in['in0'][0]}/{xsi_out['out0'][0]} (model "
        f"{model_in['in0'][0]}/{model_out['out0'][0]}), last {xsi_in['in0'][-1]}/"
        f"{xsi_out['out0'][-1]} (model {model_in['in0'][-1]}/{model_out['out0'][-1]})"
    )
    diffs = compare_traces(xsi_in, xsi_out, model_in, model_out)
    assert_traces_equal(diffs, f"VVAU {_cfg_id(cfg)} stall={kind}")

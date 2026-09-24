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

"""Differential test of the HLS MVAU template (flushable pipeline) against XSI."""

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.analysis.fpgadataflow.teg.hls_report import hls_loop_params
from finn.util.basic import getHWCustomOp
from tests.fpgadataflow.teg.stall_injection import (
    is_prepared,
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


def make_mvau_model(
    mw: int, mh: int, simd: int, pe: int, vecs: list[int], mem_mode: str
) -> ModelWrapper:
    """Single ``MVAU`` HLS node without activation (INT4 x INT4 -> INT32)."""
    idt, wdt, odt = DataType["INT4"], DataType["INT4"], DataType["INT32"]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [*vecs, mw])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [*vecs, mh])
    node = helper.make_node(
        "MVAU",
        ["inp", "weights"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        SIMD=simd,
        PE=pe,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        ActVal=0,
        binaryXnorMode=0,
        noActivation=1,
        numInputVectors=vecs,
        mem_mode=mem_mode,
        preferred_impl_style="hls",
    )
    graph = helper.make_graph(nodes=[node], name="mvau_graph", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="teg-mvau"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)
    model.set_initializer("weights", gen_finn_dt_tensor(wdt, (mw, mh)).astype(np.float32))
    return model


_PREPARED: dict[tuple, ModelWrapper] = {}


def prepared_model(cfg: tuple) -> ModelWrapper:
    """Prepare (and cache per configuration) the XSI library of the node (HLS synthesis)."""
    if not is_prepared(_PREPARED, cfg):
        mw, mh, simd, pe, vecs, mem_mode = cfg
        _PREPARED[cfg] = prepare_node_rtlsim(
            make_mvau_model(mw, mh, simd, pe, list(vecs), mem_mode), TEST_FPGA_PART, TARGET_CLK_NS
        )
    return _PREPARED[cfg]


CONFIGS = [
    # (MW, MH, SIMD, PE, numInputVectors, mem_mode); the loops have to be long enough for
    # Vitis to pipeline the top function with auto-rewind, as it does for real layers
    (64, 8, 4, 2, (1,), "internal_embedded"),  # SF = 16, NF = 4, 64 iterations
    (12, 6, 6, 2, (1, 2, 2), "internal_embedded"),  # SF = 2, NF = 3, four vectors, 24 its
    (32, 4, 4, 4, (1, 4), "internal_embedded"),  # SF = 8, NF = 1, four vectors, 32 its
    (64, 8, 4, 2, (1,), "internal_decoupled"),  # weights from a stream (driven unstalled)
]


def _cfg_id(c: tuple) -> str:
    return f"MW{c[0]}_MH{c[1]}_SIMD{c[2]}_PE{c[3]}_V{'x'.join(map(str, c[4]))}_{c[5]}"


#: known deviation of the flp loop model (see ``hls_loop.py``): input bubbles that coincide
#: with output back-pressure shift the iteration held in the write stage, which a fixed timed
#: event graph cannot express; the traces deviate by a few cycles in these combined cases
KNOWN_DEVIATIONS = {"both_bernoulli", "both_bursty"}


@pytest.mark.parametrize("cfg", CONFIGS, ids=_cfg_id)
@pytest.mark.parametrize("kind", list(stall_kinds(0)))
def test_teg_op_mvau_hls(
    cfg: tuple, kind: str, finn_test_seed: int, request: pytest.FixtureRequest
) -> None:
    """XSI and the abstract model must produce identical handshake traces."""
    if kind in KNOWN_DEVIATIONS:
        request.applymarker(
            pytest.mark.xfail(
                strict=False, reason="flp pipeline: bubbles under back-pressure (not a fixed TEG)"
            )
        )
    model = prepared_model(cfg)
    inst = getHWCustomOp(model.graph.node[0])
    in_pat, out_pat = stall_kinds(finn_test_seed)[kind]
    frames = 2
    extra = None
    if cfg[5] != "internal_embedded":
        # weight stream: one PE x SIMD word per iteration, i.e. NF * SF per input vector
        sf = inst.get_nodeattr("MW") // inst.get_nodeattr("SIMD")
        extra = {"in1_V": tokens_out(inst) * sf}
    xsi = run_xsi(model, frames, in_pat, out_pat, extra_inputs=extra)
    model_in, model_out = run_abstract(model, frames, in_pat, out_pat)
    xsi_in, xsi_out = xsi.relative()
    n_in, n_out = tokens_in(inst) * frames, tokens_out(inst) * frames
    assert len(xsi_in["in0"]) == n_in and len(xsi_out["out0"]) == n_out
    params = hls_loop_params(inst)
    print(
        f"\n{_cfg_id(cfg)} {kind}: L={params.depth} rewind={params.rewind}/{params.rewind_delay} "
        f"stages r={params.read_stage} w={params.write_stage} {n_in} in / {n_out} out, "
        f"XSI {xsi.cycles} cycles, first in/out {xsi_in['in0'][0]}/{xsi_out['out0'][0]} (model "
        f"{model_in['in0'][0]}/{model_out['out0'][0]}), last {xsi_in['in0'][-1]}/"
        f"{xsi_out['out0'][-1]} (model {model_in['in0'][-1]}/{model_out['out0'][-1]})"
    )
    diffs = compare_traces(xsi_in, xsi_out, model_in, model_out)
    assert_traces_equal(
        diffs, f"MVAU_hls {_cfg_id(cfg)} stall={kind} in={in_pat!r} out={out_pat!r}"
    )

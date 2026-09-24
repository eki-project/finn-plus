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

"""Differential test of the RTL MVAU template (replay buffer, DSP core, output lock) against XSI."""

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.analysis.fpgadataflow.teg.templates.mvau_rtl import wrapper_params
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
        resType="dsp",
        preferred_impl_style="rtl",
    )
    graph = helper.make_graph(nodes=[node], name="mvau_graph", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="teg-mvau-rtl"))
    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)
    # narrow-range weights (no -8): required for the RTL MVU on DSP48E1 parts
    weights = np.clip(gen_finn_dt_tensor(wdt, (mw, mh)), wdt.min() + 1, wdt.max())
    model.set_initializer("weights", weights.astype(np.float32))
    return model


_PREPARED: dict[tuple, ModelWrapper] = {}


def prepared_model(cfg: tuple) -> ModelWrapper:
    """Prepare (and cache per configuration) the XSI library of the node (HLS synthesis)."""
    if cfg not in _PREPARED:
        mw, mh, simd, pe, vecs, mem_mode = cfg
        prepared = prepare_node_rtlsim(
            make_mvau_model(mw, mh, simd, pe, list(vecs), mem_mode), TEST_FPGA_PART, TARGET_CLK_NS
        )
        assert prepared.graph.node[0].op_type == "MVAU_rtl", prepared.graph.node[0].op_type
        _PREPARED[cfg] = prepared
    return _PREPARED[cfg]


CONFIGS = [
    # (MW, MH, SIMD, PE, numInputVectors, mem_mode); decoupled weights come from the memstream
    # generated inside the wrapper, so the single-node design has no weight port
    (16, 8, 4, 2, (1,), "internal_decoupled"),  # SF = 4, NF = 4
    (12, 6, 6, 2, (1, 2, 2), "internal_decoupled"),  # SF = 2, NF = 3, four vectors
    (8, 4, 4, 4, (1, 4), "internal_decoupled"),  # SF = 2, NF = 1: replay buffer bypassed
]


def _cfg_id(c: tuple) -> str:
    return f"MW{c[0]}_MH{c[1]}_SIMD{c[2]}_PE{c[3]}_V{'x'.join(map(str, c[4]))}_{c[5]}"


@pytest.mark.parametrize("cfg", CONFIGS, ids=_cfg_id)
@pytest.mark.parametrize("kind", list(stall_kinds(0)))
@pytest.mark.xfail(strict=False, reason="MVAU_rtl template constants under calibration")
def test_teg_op_mvau_rtl(cfg: tuple, kind: str, finn_test_seed: int) -> None:
    """XSI and the abstract model must produce identical handshake traces."""
    model = prepared_model(cfg)
    inst = getHWCustomOp(model.graph.node[0])
    in_pat, out_pat = stall_kinds(finn_test_seed)[kind]
    frames = 2
    # the weight stream port of the wrapper (fed by the memstream only at IPI level): one
    # PE x SIMD word per core iteration, i.e. NF * SF per input vector, driven without stalls
    sf = inst.get_nodeattr("MW") // inst.get_nodeattr("SIMD")
    extra = {"in1_V": tokens_out(inst) * sf}
    xsi = run_xsi(model, frames, in_pat, out_pat, extra_inputs=extra)
    model_in, model_out = run_abstract(model, frames, in_pat, out_pat)
    xsi_in, xsi_out = xsi.relative()
    n_in, n_out = tokens_in(inst) * frames, tokens_out(inst) * frames
    assert len(xsi_in["in0"]) == n_in and len(xsi_out["out0"]) == n_out
    params = wrapper_params(inst)
    print(
        f"\n{_cfg_id(cfg)} {kind}: VERSION={params.get('VERSION')} SEGMENTLEN="
        f"{params.get('SEGMENTLEN')} {n_in} in / {n_out} out, "
        f"XSI {xsi.cycles} cycles, first in/out {xsi_in['in0'][0]}/{xsi_out['out0'][0]} (model "
        f"{model_in['in0'][0]}/{model_out['out0'][0]}), last {xsi_in['in0'][-1]}/"
        f"{xsi_out['out0'][-1]} (model {model_in['in0'][-1]}/{model_out['out0'][-1]})"
    )
    diffs = compare_traces(xsi_in, xsi_out, model_in, model_out)
    assert_traces_equal(
        diffs, f"MVAU_rtl {_cfg_id(cfg)} stall={kind} in={in_pat!r} out={out_pat!r}"
    )

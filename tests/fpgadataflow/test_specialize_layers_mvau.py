# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from functools import partial

from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import gen_finn_dt_tensor

from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.exception import FINNUserError
from tests.fpgadataflow.test_fpgadataflow_mvau import make_single_fclayer_modelwrapper

DSP48E2_PART = "xczu28dr-ffvg1517-2-e"
DSP58_PART = "xcvc1902-vsva2197-2MP-e-S"


def make_mvau_model(idt, wvals, wdt, mw, mh=8):
    """Single MVAU (no activation) with narrow-range weight values of datatype wvals,
    annotated with the (possibly wider container) datatype wdt."""
    W = gen_finn_dt_tensor(wvals, (mw, mh))
    W[W == wvals.min()] = wvals.min() + 1
    # make sure the extremes are present so the derived widths are deterministic
    W[0, 0], W[0, 1] = wvals.min() + 1, wvals.max()
    model = make_single_fclayer_modelwrapper(W, 1, 1, wdt, idt, DataType["INT32"])
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("preferred_impl_style", "rtl")
    return model


def minimize_bit_widths(model):
    """Mirror step_minimize_bit_width_initial, which precedes specialization in the flow."""
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())
    return model.transform(InferDataTypes())


# The RTL MVU computes on the DSP datapaths: activations on the signed B port (18/24 bit,
# so unsigned activations get one bit less), weights on A (27 bit) and the accumulator on
# P (48/58 bit). Wider configurations must fall back to HLS. The widths are judged after
# bit width minimization, so an oversized container weight datatype (such as the INT64
# placeholder of the ONNX passes frontend) does not matter.
# (part, input dtype, weight value range, weight container dtype, MW, expected variant)
@pytest.mark.parametrize(
    "part, idt, wvals, wdt, mw, expected",
    [
        pytest.param(DSP48E2_PART, "UINT8", "INT8", "INT8", 64, "rtl", id="dsp48-8x8"),
        pytest.param(DSP48E2_PART, "UINT8", "INT8", "INT64", 64, "rtl", id="dsp48-placeholder-wdt"),
        pytest.param(DSP48E2_PART, "UINT17", "INT8", "INT8", 64, "rtl", id="dsp48-act-fits-b"),
        pytest.param(DSP48E2_PART, "UINT18", "INT8", "INT8", 64, "hls", id="dsp48-act-exceeds-b"),
        pytest.param(
            DSP48E2_PART, "INT18", "INT8", "INT8", 64, "rtl", id="dsp48-signed-act-fits-b"
        ),
        pytest.param(DSP48E2_PART, "UINT17", "INT24", "INT64", 64, "rtl", id="dsp48-acc-fits-p"),
        pytest.param(
            DSP48E2_PART, "UINT17", "INT24", "INT64", 1024, "hls", id="dsp48-acc-exceeds-p"
        ),
        pytest.param(DSP58_PART, "UINT23", "INT8", "INT8", 64, "rtl", id="dsp58-act-fits-b"),
        pytest.param(DSP58_PART, "UINT24", "INT8", "INT8", 64, "hls", id="dsp58-act-exceeds-b"),
    ],
)
@pytest.mark.fpgadataflow
def test_specialize_mvau_rtl_dsp_datapath_bounds(part, idt, wvals, wdt, mw, expected):
    model = make_mvau_model(DataType[idt], DataType[wvals], DataType[wdt], mw)
    model = minimize_bit_widths(model)
    model = model.transform(SpecializeLayers(part))
    assert model.graph.node[0].op_type == f"MVAU_{expected}"


@pytest.mark.fpgadataflow
def test_mvau_rtl_codegen_rejects_wide_operands():
    """Widths exceeding the DSP datapaths are caught at code generation as well, e.g.
    when the RTL variant is selected explicitly or the datatypes change afterwards."""
    model = make_mvau_model(DataType["UINT38"], DataType["INT8"], DataType["INT8"], 64)
    model = minimize_bit_widths(model)
    # force the RTL variant regardless of the specialization check
    model.graph.node[0].op_type = "MVAU_rtl"
    model.graph.node[0].domain = "finn.custom_op.fpgadataflow.rtl"
    with pytest.raises(FINNUserError, match="exceeds the .* activation datapath limit"):
        model.transform(PrepareIP(DSP48E2_PART, 10.0))


# ---------------------------------------------------------------------------
# Distributed-arithmetic RTL MVU (mem_mode=internal_embedded)
# ---------------------------------------------------------------------------


def make_embedded_mvau_model(idt="UINT4", wdt="INT4", mw=16, mh=16, preferred="rtl", act=False):
    """Single MVAU with static weights and mem_mode=internal_embedded."""
    W = gen_finn_dt_tensor(DataType[wdt], (mw, mh))
    T, tdt, odt = None, None, DataType["INT32"]
    if act:
        import numpy as np

        odt, tdt = DataType["INT4"], DataType["INT32"]
        T = np.sort(np.random.randint(-64, 64, (mh, 15)).astype(np.float32), axis=1)
    model = make_single_fclayer_modelwrapper(W, 1, 1, DataType[wdt], DataType[idt], odt, T, tdt)
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("mem_mode", "internal_embedded")
    inst.set_nodeattr("resType", "lut")
    inst.set_nodeattr("preferred_impl_style", preferred)
    return minimize_bit_widths(model)


@pytest.mark.fpgadataflow
def test_specialize_mvau_rtl_da_opt_in():
    model = make_embedded_mvau_model(preferred="rtl")
    model = model.transform(SpecializeLayers(DSP48E2_PART))
    assert model.graph.node[0].op_type == "MVAU_rtl"
    inst = getCustomOp(model.graph.node[0])
    assert inst.get_nodeattr("mem_mode") == "internal_embedded"
    assert inst._is_da_mode()


@pytest.mark.parametrize("preferred", ["hls", ""])
@pytest.mark.fpgadataflow
def test_specialize_mvau_embedded_stays_hls_without_opt_in(preferred):
    model = make_embedded_mvau_model(preferred=preferred)
    model = model.transform(SpecializeLayers(DSP48E2_PART))
    assert model.graph.node[0].op_type == "MVAU_hls"
    assert getCustomOp(model.graph.node[0]).get_nodeattr("mem_mode") == "internal_embedded"


@pytest.mark.parametrize(
    "attr, value, reason",
    [
        ("binaryXnorMode", 1, "binaryXnorMode"),
        ("runtime_writeable_weights", 1, "runtime_writeable_weights"),
        ("resType", "dsp", "LUT-only"),
        ("TH", 2, "tiling"),
        ("pumpedMemory", 1, "clock pumping"),
    ],
)
@pytest.mark.fpgadataflow
def test_specialize_mvau_rtl_da_rejects(attr, value, reason):
    model = make_embedded_mvau_model(preferred="rtl")
    getCustomOp(model.graph.node[0]).set_nodeattr(attr, value)
    with pytest.raises(FINNUserError, match=reason):
        model.transform(SpecializeLayers(DSP48E2_PART))


@pytest.mark.fpgadataflow
def test_specialize_mvau_rtl_da_rejects_embedded_thresholds():
    model = make_embedded_mvau_model(preferred="rtl", act=True)
    with pytest.raises(FINNUserError, match="standalone thresholds"):
        model.transform(SpecializeLayers(DSP48E2_PART))


@pytest.mark.fpgadataflow
def test_mvau_rtl_da_codegen_rejects_partial_folding():
    """SIMD=MW and PE=MH is only checked at code generation time."""
    model = make_embedded_mvau_model(preferred="rtl")
    model = model.transform(SpecializeLayers(DSP48E2_PART))
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("SIMD", 8)
    inst.set_nodeattr("PE", 16)
    with pytest.raises(FINNUserError, match="SIMD=MW"):
        model.transform(PrepareIP(DSP48E2_PART, 5))
    # fully unrolled: code generation succeeds and reports the solution
    inst.set_nodeattr("SIMD", 16)
    model = model.transform(PrepareIP(DSP48E2_PART, 5))
    inst = getCustomOp(model.graph.node[0])
    assert inst.get_nodeattr("da_cost") > 0
    assert inst.get_nodeattr("da_latency_cycles") >= 2
    assert inst.lut_estimation(DSP48E2_PART) > 0
    assert inst.dsp_estimation(DSP48E2_PART) == 0
    files = inst.get_rtl_file_list(abspath=True)
    import os

    assert all(os.path.isfile(f) for f in files), files
    assert any(f.endswith("mvu_da_axi.sv") for f in files)
    assert any(f.endswith("_da_core_wrapper.v") for f in files)


@pytest.mark.fpgadataflow
def test_mvau_rtl_da_estimate_runs_solver():
    """The estimate reports run before code generation; the analysis pass runs the
    solver so that the LUT estimate is based on the adder graph."""
    import math

    model = make_embedded_mvau_model(preferred="rtl")
    model = model.transform(SpecializeLayers(DSP48E2_PART))
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("SIMD", 16)
    inst.set_nodeattr("PE", 16)
    fallback = inst.lut_estimation(DSP48E2_PART)
    res = model.analysis(partial(res_estimation, fpgapart=DSP48E2_PART))
    inst = getCustomOp(model.graph.node[0])
    da_cost = inst.get_nodeattr("da_cost")
    assert da_cost > 0
    queue = 16 * inst.get_output_datatype().bitwidth()
    assert res["MVAU_rtl_0"]["LUT"] == math.ceil(da_cost) + queue
    assert res["MVAU_rtl_0"]["LUT"] != fallback
    assert res["MVAU_rtl_0"]["DSP"] == 0
    # partially folded: the solver is skipped and the fallback estimate remains
    model2 = make_embedded_mvau_model(preferred="rtl")
    model2 = model2.transform(SpecializeLayers(DSP48E2_PART))
    res2 = model2.analysis(partial(res_estimation, fpgapart=DSP48E2_PART))
    assert getCustomOp(model2.graph.node[0]).get_nodeattr("da_cost") == 0
    assert res2["MVAU_rtl_0"]["LUT"] > 0

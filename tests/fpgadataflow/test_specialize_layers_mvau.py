# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import gen_finn_dt_tensor

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
            DSP48E2_PART, "UINT17", "INT24", "INT64", 256, "hls", id="dsp48-acc-exceeds-p"
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

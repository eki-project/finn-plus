# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import gen_finn_dt_tensor

from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from tests.fpgadataflow.test_fpgadataflow_mvau import make_single_fclayer_modelwrapper

DSP48E2_PART = "xczu28dr-ffvg1517-2-e"
DSP58_PART = "xcvc1902-vsva2197-2MP-e-S"


# The RTL MVU computes on the DSP datapaths: activations on B (18/24 bit), weights
# on A (27 bit) and the accumulator on P (48/58 bit). Wider configurations must fall
# back to HLS, judged by the widths after bit width minimization (the container
# weight datatype at this point may be an oversized placeholder such as INT64).
# (part, input dtype, weight value range, weight container dtype, MW, expected variant)
@pytest.mark.parametrize(
    "part, idt, wvals, wdt, mw, expected",
    [
        pytest.param(DSP48E2_PART, "UINT8", "INT8", "INT8", 64, "rtl", id="dsp48-8x8"),
        pytest.param(DSP48E2_PART, "UINT8", "INT8", "INT64", 64, "rtl", id="dsp48-placeholder-wdt"),
        pytest.param(DSP48E2_PART, "UINT18", "INT8", "INT8", 64, "rtl", id="dsp48-act-fits-b"),
        pytest.param(DSP48E2_PART, "UINT19", "INT8", "INT8", 64, "hls", id="dsp48-act-exceeds-b"),
        pytest.param(DSP48E2_PART, "UINT18", "INT24", "INT64", 64, "rtl", id="dsp48-acc-fits-p"),
        pytest.param(
            DSP48E2_PART, "UINT18", "INT24", "INT64", 256, "hls", id="dsp48-acc-exceeds-p"
        ),
        pytest.param(DSP58_PART, "UINT24", "INT8", "INT8", 64, "rtl", id="dsp58-act-fits-b"),
        pytest.param(DSP58_PART, "UINT25", "INT8", "INT8", 64, "hls", id="dsp58-act-exceeds-b"),
    ],
)
@pytest.mark.fpgadataflow
def test_specialize_mvau_rtl_dsp_datapath_bounds(part, idt, wvals, wdt, mw, expected):
    idt, wvals, wdt = DataType[idt], DataType[wvals], DataType[wdt]
    # narrow-range weight values, independent of the (container) weight datatype
    W = gen_finn_dt_tensor(wvals, (mw, 8))
    W[W == wvals.min()] = wvals.min() + 1
    # make sure the extremes are present so the derived widths are deterministic
    W[0, 0], W[0, 1] = wvals.min() + 1, wvals.max()
    model = make_single_fclayer_modelwrapper(W, 1, 1, wdt, idt, DataType["INT32"])
    inst = getCustomOp(model.graph.node[0])
    inst.set_nodeattr("preferred_impl_style", "rtl")
    model = model.transform(SpecializeLayers(part))
    assert model.graph.node[0].op_type == f"MVAU_{expected}"

# Copyright (c) 2025, Paderborn University
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-pandas tests of the QoR feature specifications (no FINN needed)."""

import pytest

import pandas as pd

from finn.qor.features import (
    SPECS,
    DataTypeInfo,
    broadcast_kind,
    datatype_features,
    parse_datatype,
    spec_for_op_type,
)

pytestmark = pytest.mark.qor


@pytest.mark.parametrize(
    "name,expected",
    [
        ("INT4", (4, True, "int")),
        ("UINT8", (8, False, "int")),
        ("BINARY", (1, False, "int")),
        ("BIPOLAR", (1, True, "int")),
        ("TERNARY", (2, True, "int")),
        ("FLOAT32", (32, True, "float")),
        ("FLOAT16", (16, True, "float")),
        ("FIXED<8,4>", (8, True, "fixed")),
        ("SCALEDINT<8>", (8, True, "scaledint")),
    ],
)
def test_parse_datatype(name, expected):
    info = parse_datatype(name)
    assert info == DataTypeInfo(*expected)
    # tuple indexing is used by older code
    assert (info[0], info[1]) == expected[:2]


def test_datatype_features_none():
    assert datatype_features("act", None) == {"act_bitwidth": None, "act_sign": None}
    assert datatype_features("x", "INT4", with_kind=True) == {
        "x_bitwidth": 4,
        "x_sign": True,
        "x_kind": "int",
    }


@pytest.mark.parametrize(
    "rhs_shape,out_shape,expected",
    [
        ([1], [1, 8, 8, 32], "scalar"),
        ([32], [1, 8, 8, 32], "channel"),
        ([1, 8, 8, 32], [1, 8, 8, 32], "full"),
        ([8, 32], [1, 8, 8, 32], "other"),
    ],
)
def test_broadcast_kind(rhs_shape, out_shape, expected):
    assert broadcast_kind(rhs_shape, out_shape) == expected


# One synthetic database row (params + dut_info) per operator, in the flattened layout
_ROWS = {
    "mvau": {
        "params.backend": "hls",
        "params.mem_mode": "internal_embedded",
        "params.ram_style": "distributed",
        "params.ram_style_thr": "distributed",
        "params.idt": "INT4",
        "params.wdt": "INT4",
        "params.act": "INT4",
        "params.mw": 64,
        "params.mh": 64,
        "dut_info.simd": 8,
        "dut_info.pe": 8,
        "dut_info.zero_weights": 0.06,
    },
    "thresholding": {
        "params.backend": "rtl",
        "params.idt": "INT8",
        "params.odt": "UINT4",
        "params.ch": 64,
        "params.pe": 8,
        "dut_info.tdt": "INT8",
        "params.mem_mode": None,
        "params.ram_style": None,
        "params.depth_trigger_bram": 256,
        "params.depth_trigger_uram": 0,
    },
    "swg": {
        "params.idt": "UINT4",
        "params.ifm_ch": 16,
        "params.ifm_dim": [16, 16],
        "params.k": [3, 3],
        "params.stride": [1, 1],
        "params.dilation": [1, 1],
        "params.simd": 4,
        "params.depthwise": 0,
        "params.parallel_window": 0,
        "params.m": 1,
        "params.ram_style": "distributed",
        "dut_info.impl_style": "default",
        "dut_info.buffer_depth": 140,
        "dut_info.is1D": 0,
    },
    "vvau": {
        "params.backend": "hls",
        "params.mem_mode": "internal_embedded",
        "params.ram_style": "auto",
        "params.resType": "lut",
        "params.idt": "INT8",
        "params.wdt": "INT4",
        "params.act": None,
        "params.ch": 32,
        "params.k": [3, 3],
        "params.pe": 4,
        "params.simd": 3,
        "dut_info.zero_weights": 0.07,
    },
    "fifo": {
        "params.impl_style": "vivado",
        "params.ram_style": "block",
        "params.dtype": "INT8",
        "dut_info.width_bits": 64,
        "params.depth": 1000,
        "dut_info.depth_adjusted": 1024,
        "dut_info.capacity_bits": 65536,
    },
    "dwc": {
        "params.backend": "rtl",
        "params.dtype": "INT4",
        "params.ch": 32,
        "dut_info.in_width": 16,
        "dut_info.out_width": 32,
        "dut_info.ratio": 2.0,
        "dut_info.integer_ratio": True,
    },
    "pool": {
        "params.function": "QuantAvgPool",
        "params.idt": "UINT8",
        "dut_info.odt": "UINT4",
        "params.ch": 32,
        "params.pe": 4,
        "params.k": [2, 2],
        "dut_info.accum_bits": 10,
    },
    "fmpadding": {
        "params.idt": "UINT4",
        "params.ch": 16,
        "params.simd": 4,
        "params.idim": [8, 8],
        "params.padding": [1, 1, 1, 1],
    },
    "eltwise": {
        "params.backend": "hls",
        "params.op": "Add",
        "params.lhs_dtype": "INT8",
        "dut_info.rhs_dtype": "INT7",
        "dut_info.out_dtype": "INT9",
        "params.pe": 4,
        "params.rhs_bcast": "channel",
        "dut_info.rhs_num_elems": 32,
        "params.shape": [1, 8, 8, 32],
        "params.mem_mode": "internal_embedded",
        "params.ram_style": "auto",
    },
}


def test_all_specs_have_rows():
    assert set(_ROWS) == set(SPECS)


@pytest.mark.parametrize("operator", sorted(SPECS))
def test_spec_derives_all_feature_columns(operator):
    spec = SPECS[operator]
    df = pd.DataFrame([_ROWS[operator], _ROWS[operator]])
    derived = spec.derive_db_columns(df)
    missing = [c for c in spec.feature_cols if c not in derived.columns]
    assert not missing, f"{operator}: derive_db_columns does not produce {missing}"
    # every irrelevant column must be a params column and must not be a feature
    for col in spec.irrelevant_param_cols:
        assert col.startswith("params.")
        assert col not in spec.feature_cols
    # the spec name is the DUT name / database folder: lower case identifier
    assert spec.name.isidentifier() and spec.name == spec.name.lower()


def test_spec_for_op_type():
    assert spec_for_op_type("MVAU_rtl").name == "mvau"
    assert spec_for_op_type("ConvolutionInputGenerator_rtl").name == "swg"
    assert spec_for_op_type("ElementwiseMul_hls").name == "eltwise"
    assert spec_for_op_type("LabelSelect_hls") is None


def test_op_types_unique():
    seen = {}
    for spec in SPECS.values():
        for op_type in spec.op_types:
            assert op_type not in seen, f"{op_type} claimed by {seen[op_type]} and {spec.name}"
            seen[op_type] = spec.name

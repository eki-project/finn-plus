# Copyright (c) 2025, Paderborn University
# SPDX-License-Identifier: BSD-3-Clause

"""Build every microbenchmark DUT's model (no synthesis) and check that the node feature
extractors produce exactly the operator's feature columns with the same values the
database side derives from params + dut_info (train/inference parity)."""

import pytest

import math
import pandas as pd
import random
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.analysis.fpgadataflow.empirical_qor_estimation import (
    NODE_FEATURE_EXTRACTORS,
    node_features,
)
from finn.benchmarking.dut import MICROBENCH_DUTS
from finn.benchmarking.dut.microbench_base import backend_of, resolve_part
from finn.benchmarking.param_space import sample_params
from finn.benchmarking.sampling import expand_config, pop_sampling_info
from finn.qor.features import COMMON_DUT_INFO_KEYS, SPECS
from finn.util.basic import getHWCustomOp
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

pytestmark = [pytest.mark.qor, pytest.mark.fpgadataflow]

RFSOC = "xczu28dr-ffvg1517-2-e"
VERSAL = "xcvc1902-vsva2197-2MP-e-S"

_MVAU = {
    "backend": "hls",
    "idt": "INT4",
    "wdt": "INT4",
    "act": "INT4",
    "nhw": [1],
    "mw": 64,
    "mh": 64,
    "sf": 8,
    "nf": 8,
    "m": 1,
    "sparsity_type": "none",
    "sparsity_amount": 0,
    "mem_mode": "internal_embedded",
    "ram_style": "distributed",
    "ram_style_thr": "distributed",
}
_THR = {
    "backend": "hls",
    "idt": "INT8",
    "odt": "UINT4",
    "ch": 64,
    "pe": 8,
    "nhw": [1, 8, 8],
    "mem_mode": "internal_embedded",
    "ram_style": "distributed",
    "depth_trigger_bram": 0,
    "depth_trigger_uram": 0,
}
_SWG = {
    "idt": "UINT4",
    "ifm_ch": 16,
    "ifm_dim": [16, 16],
    "k": [3, 3],
    "stride": [1, 1],
    "dilation": [1, 1],
    "simd": 4,
    "depthwise": 0,
    "parallel_window": 0,
    "m": 1,
    "ram_style": "distributed",
}
_VVAU = {
    "backend": "hls",
    "idt": "INT8",
    "wdt": "INT4",
    "act": "UINT4",
    "ch": 32,
    "dim": [7, 7],
    "k": [3, 3],
    "pe": 4,
    "simd": 3,
    "mem_mode": "internal_embedded",
    "ram_style": "auto",
    "resType": "lut",
}
_FIFO = {"impl_style": "rtl", "dtype": "INT8", "elems": 8, "n": 16, "depth": 32, "ram_style": None}
_DWC = {"backend": "hls", "dtype": "INT4", "in_elems": 6, "out_elems": 4, "ch": 24, "n": 4}
_POOL = {
    "function": "MaxPool",
    "idt": "INT8",
    "odt": None,
    "ch": 32,
    "pe": 4,
    "k": [2, 2],
    "odim": [8, 8],
}
_FMP = {"idt": "UINT4", "ch": 16, "simd": 4, "idim": [8, 8], "padding": [1, 1, 1, 1]}
_ELT = {
    "backend": "hls",
    "op": "Add",
    "lhs_dtype": "INT8",
    "rhs_dtype": "INT8",
    "shape": [1, 8, 8, 32],
    "rhs_bcast": "channel",
    "pe": 4,
    "mem_mode": "internal_embedded",
    "ram_style": "auto",
}

# (dut, params, part, expected op_type)
CASES = [
    ("mvau", _MVAU, RFSOC, "MVAU_hls"),
    (
        "mvau",
        {
            **_MVAU,
            "backend": "rtl",
            "idt": "INT8",
            "wdt": "INT8",
            "act": None,
            "mem_mode": "internal_decoupled",
        },
        RFSOC,
        "MVAU_rtl",
    ),
    ("thresholding", _THR, RFSOC, "Thresholding_hls"),
    (
        "thresholding",
        {
            **_THR,
            "backend": "rtl",
            "mem_mode": None,
            "ram_style": None,
            "depth_trigger_bram": 256,
            "depth_trigger_uram": 4096,
        },
        RFSOC,
        "Thresholding_rtl",
    ),
    ("swg", _SWG, RFSOC, "ConvolutionInputGenerator_rtl"),
    (
        "swg",
        {**_SWG, "simd": 16, "parallel_window": 1, "ram_style": "block"},
        RFSOC,
        "ConvolutionInputGenerator_rtl",
    ),
    (
        "swg",
        {**_SWG, "ifm_dim": [1, 64], "k": [1, 5], "depthwise": 1},
        RFSOC,
        "ConvolutionInputGenerator_rtl",
    ),
    ("vvau", _VVAU, RFSOC, "VVAU_hls"),
    (
        "vvau",
        {
            **_VVAU,
            "backend": "rtl",
            "act": None,
            "resType": None,
            "idt": "INT8",
            "wdt": "INT8",
            "mem_mode": "internal_decoupled",
            "board": "VCK190",
        },
        VERSAL,
        "VVAU_rtl",
    ),
    ("fifo", _FIFO, RFSOC, "StreamingFIFO_rtl"),
    (
        "fifo",
        {**_FIFO, "impl_style": "vivado", "depth": 1000, "ram_style": "block"},
        RFSOC,
        "StreamingFIFO_rtl",
    ),
    ("dwc", _DWC, RFSOC, "StreamingDataWidthConverter_hls"),
    (
        "dwc",
        {**_DWC, "backend": "rtl", "in_elems": 4, "out_elems": 8, "ch": 32},
        RFSOC,
        "StreamingDataWidthConverter_rtl",
    ),
    ("pool", _POOL, RFSOC, "Pool_hls"),
    (
        "pool",
        {**_POOL, "function": "QuantAvgPool", "idt": "UINT8", "odt": "UINT4"},
        RFSOC,
        "Pool_hls",
    ),
    ("fmpadding", _FMP, RFSOC, "FMPadding_rtl"),
    ("eltwise", _ELT, RFSOC, "ElementwiseAdd_hls"),
    (
        "eltwise",
        {
            **_ELT,
            "op": "Mul",
            "rhs_bcast": "scalar",
            "lhs_dtype": "UINT4",
            "rhs_dtype": "INT16",
            "mem_mode": "internal_decoupled",
        },
        RFSOC,
        "ElementwiseMul_hls",
    ),
    (
        "eltwise",
        {
            **_ELT,
            "backend": "rtl",
            "op": "Mul",
            "lhs_dtype": "FLOAT32",
            "rhs_dtype": "FLOAT32",
            "mem_mode": "internal_decoupled",
            "board": "VCK190",
        },
        VERSAL,
        "ElementwiseMul_rtl",
    ),
]

INVALID = [
    ("mvau", {**_MVAU, "sf": 7}),
    ("mvau", {**_MVAU, "backend": "rtl"}),  # activation not supported by rtl
    ("thresholding", {**_THR, "pe": 7}),
    ("thresholding", {**_THR, "backend": "rtl"}),  # mem_mode/ram_style must be unset for rtl
    ("swg", {**_SWG, "k": [17, 17]}),
    ("swg", {**_SWG, "parallel_window": 1}),  # simd != ifm_ch
    ("vvau", {**_VVAU, "backend": "rtl", "act": None, "resType": None}),  # not Versal
    ("fifo", {**_FIFO, "ram_style": "block"}),
    ("fifo", {**_FIFO, "impl_style": "vivado", "depth": 8, "ram_style": "block"}),
    ("dwc", {**_DWC, "backend": "rtl"}),  # non-integer ratio
    ("pool", {**_POOL, "odt": "INT4"}),
    ("fmpadding", {**_FMP, "padding": [0, 0, 0, 0]}),
    ("eltwise", {**_ELT, "pe": 3}),
    ("eltwise", {**_ELT, "backend": "rtl"}),
]


def _single_hw_node(model):
    nodes = [n for n in model.graph.node if is_hls_node(n) or is_rtl_node(n)]
    assert len(nodes) == 1, [n.op_type for n in model.graph.node]
    return nodes[0]


def _equal(a, b) -> bool:
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return b is None or (isinstance(b, float) and math.isnan(b))
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b))
    return a == b


def test_every_spec_has_a_dut_and_extractor():
    assert set(NODE_FEATURE_EXTRACTORS) == set(SPECS)
    assert set(MICROBENCH_DUTS) == set(SPECS)
    for name, cls in MICROBENCH_DUTS.items():
        assert cls.NAME == name
        assert set(cls.OP_TYPES) == set(SPECS[name].op_types)


@pytest.mark.parametrize("dut,params,part,op_type", CASES, ids=[f"{c[0]}-{c[3]}" for c in CASES])
def test_dut_model_and_feature_parity(dut, params, part, op_type):
    cls = MICROBENCH_DUTS[dut]
    spec = SPECS[dut]
    assert cls.validate(params) is None
    model, info = cls.make_model(params, part)
    model = model.transform(GiveUniqueNodeNames())
    node = _single_hw_node(model)
    assert node.op_type == op_type
    assert set(spec.dut_info_keys) <= set(
        info
    ), f"dut_info misses {set(spec.dut_info_keys) - set(info)}"
    info = {
        **info,
        "dut_node_name": node.name,
        "dut_op_type": node.op_type,
        "dut_backend": backend_of(node),
    }
    assert set(COMMON_DUT_INFO_KEYS) <= set(info)

    # inference side
    inst = getHWCustomOp(node)
    node_row = node_features(model, node, inst)
    assert list(node_row.columns) == spec.feature_cols

    # database side: the same params + dut_info as one flattened row
    db_row = pd.json_normalize([{"params": params, "dut_info": info}])
    db_row = spec.derive_db_columns(db_row)
    mismatches = {
        col: (db_row[col].iloc[0], node_row[col].iloc[0])
        for col in spec.feature_cols
        if not _equal(db_row[col].iloc[0], node_row[col].iloc[0])
    }
    assert not mismatches, f"train/inference feature mismatch: {mismatches}"


@pytest.mark.parametrize("dut,params", INVALID, ids=[f"{c[0]}-{i}" for i, c in enumerate(INVALID)])
def test_validate_rejects(dut, params):
    reason = MICROBENCH_DUTS[dut].validate(params)
    assert isinstance(reason, str) and reason


@pytest.mark.parametrize("dut", sorted(MICROBENCH_DUTS))
def test_param_space_yields_valid_configs(dut):
    cls = MICROBENCH_DUTS[dut]
    rng = random.Random(1234)
    space = cls.param_space()
    n_valid, n_total = 0, 60
    for _ in range(n_total):
        params = sample_params(space, rng)
        if cls.validate(params) is None:
            n_valid += 1
    # the space is meant to be sampled efficiently: at least a third must be valid
    assert n_valid >= n_total // 3, f"{dut}: only {n_valid}/{n_total} sampled configs valid"


@pytest.mark.parametrize("dut", sorted(MICROBENCH_DUTS))
def test_sampled_configs_build(dut):
    cls = MICROBENCH_DUTS[dut]
    rng = random.Random(42)
    space = cls.param_space()
    built = 0
    for _ in range(40):
        params = sample_params(space, rng)
        if cls.validate(params) is not None:
            continue
        model, info = cls.make_model(params, resolve_part(params))
        node = _single_hw_node(model.transform(GiveUniqueNodeNames()))
        if node.op_type not in cls.OP_TYPES:
            continue
        node_features(model, node, getHWCustomOp(node))
        built += 1
        if built == 3:
            break
    assert built == 3


def test_expand_config_with_real_duts(monkeypatch):
    monkeypatch.delenv("SAMPLE_COUNT", raising=False)
    monkeypatch.delenv("SAMPLE_SEED", raising=False)
    monkeypatch.delenv("CI_PIPELINE_ID", raising=False)
    config = [
        {"mode": "sample", "dut": dut, "num_samples": 5, "seed": 3, "skip_existing": False}
        for dut in sorted(MICROBENCH_DUTS)
    ]
    expanded, stats = expand_config(config, MICROBENCH_DUTS, None)
    assert [s.produced for s in stats] == [5] * len(MICROBENCH_DUTS)
    assert len(expanded) == 5 * len(MICROBENCH_DUTS)
    for params in expanded:
        pop_sampling_info(params)
        cls = MICROBENCH_DUTS[params["dut"]]
        assert cls.validate(params) is None
        # the fixed defaults every CI run needs are attached
        assert params["instrumentation_no_dma"] is True
        assert params["store_results_in_dvc_data"] is True
        assert "bitfile" in params["generate_outputs"]

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
_FIFO = {"dtype": "INT8", "elems": 8, "n": 16, "depth": 32, "ram_style": "auto"}
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
    ("fifo", {**_FIFO, "depth": 1024, "ram_style": "block"}, RFSOC, "StreamingFIFO_rtl"),
    (
        "fifo",
        {**_FIFO, "depth": 4096, "elems": 16, "ram_style": "ultra"},
        RFSOC,
        "StreamingFIFO_rtl",
    ),
    ("dwc", _DWC, RFSOC, "StreamingDataWidthConverter_hls"),
    ("dwc", {**_DWC, "backend": "rtl"}, RFSOC, "StreamingDataWidthConverter_rtl"),
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
    (
        "eltwise",
        {
            **_ELT,
            "backend": "rtl",
            "op": "Add",
            "lhs_dtype": "FLOAT32",
            "rhs_dtype": "INT8",
            "mem_mode": "internal_decoupled",
            "board": "VCK190",
        },
        VERSAL,
        "ElementwiseAdd_rtl",
    ),
]

INVALID = [
    ("mvau", {**_MVAU, "sf": 7}),
    ("mvau", {**_MVAU, "backend": "rtl"}),  # activation not supported by rtl
    # 128 x 8 x 8 bit weight tile = 8192 bits > AP_INT_MAX_W (pipeline 152593 run 7)
    (
        "mvau",
        {
            **_MVAU,
            "idt": "INT8",
            "wdt": "INT8",
            "act": "INT8",
            "mw": 1024,
            "mh": 128,
            "sf": 8,
            "nf": 16,
            "mem_mode": "internal_embedded",
        },
    ),
    # URAM weights need runtime-writeable weights on the RFSoC (pipeline 152593 run 1)
    ("mvau", {**_MVAU, "mem_mode": "internal_decoupled", "ram_style": "ultra"}),
    ("thresholding", {**_THR, "pe": 7}),
    ("thresholding", {**_THR, "backend": "rtl"}),  # mem_mode/ram_style must be unset for rtl
    ("swg", {**_SWG, "k": [17, 17]}),
    ("swg", {**_SWG, "parallel_window": 1}),  # simd != ifm_ch
    ("vvau", {**_VVAU, "backend": "rtl", "act": None, "resType": None}),  # not Versal
    ("fifo", {**_FIFO, "ram_style": "vivado"}),
    ("fifo", {**_FIFO, "depth": 1}),
    ("dwc", {**_DWC, "in_elems": 5}),  # does not divide ch
    ("pool", {**_POOL, "odt": "INT4"}),
    ("fmpadding", {**_FMP, "padding": [0, 0, 0, 0]}),
    ("eltwise", {**_ELT, "pe": 3}),
    ("eltwise", {**_ELT, "backend": "rtl"}),
    # --- failure classes of the first sample_all pipeline (152661) ---
    ("mvau", {**_MVAU, "wdt": "BINARY"}),  # MVAU_hls: true binary not supported
    ("thresholding", {**_THR, "ram_style": "ultra"}),  # URAM init from bitfile: Versal only
    ("thresholding", {**_THR, "mem_mode": "internal_decoupled", "ram_style": "ultra"}),
    # 64 PEs x 255 steps x 4 bit = 65280 bit threshold stream > AP_INT_MAX_W
    (
        "thresholding",
        {
            **_THR,
            "idt": "UINT4",
            "odt": "INT8",
            "ch": 512,
            "pe": 64,
            "mem_mode": "internal_decoupled",
        },
    ),
    ("vvau", {**_VVAU, "mem_mode": "internal_decoupled", "ram_style": "ultra"}),
    ("eltwise", {**_ELT, "ram_style": "ultra"}),
    # 192 bit x 16384 deep shift register: Vivado refuses the array
    ("fifo", {"dtype": "INT16", "elems": 12, "n": 64, "depth": 16384, "ram_style": "srl"}),
    # 15 x 4 bit output elements: padded 64 bit word not sliceable into 15 subwords
    (
        "swg",
        {
            **_SWG,
            "idt": "INT4",
            "ifm_ch": 3,
            "ifm_dim": [1, 256],
            "k": [1, 5],
            "depthwise": 1,
            "parallel_window": 1,
            "simd": 3,
            "ram_style": "auto",
        },
    ),
    # 224x224x512 depthwise with SIMD 2: 12.8 M cycles per frame (18 h build)
    (
        "swg",
        {**_SWG, "ifm_ch": 512, "ifm_dim": [224, 224], "k": [5, 5], "depthwise": 1, "simd": 2},
    ),
    (
        "dwc",
        {"backend": "rtl", "dtype": "BINARY", "ch": 96, "in_elems": 32, "out_elems": 3, "n": 16},
    ),
    ("fmpadding", {"idt": "UINT2", "ch": 3, "simd": 3, "idim": [7, 7], "padding": [2, 2, 2, 2]}),
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
        inst = getHWCustomOp(node)
        if is_hls_node(node):
            inst.get_ap_int_max_w()  # raises if a width exceeds AP_INT_MAX_W
        node_features(model, node, inst)
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


def test_smoke_config_runs_are_valid():
    """Every run of the CI smoke config must be buildable (an invalid combination is only
    skipped at build time and would silently reduce the coverage) and cover every DUT."""
    import yaml
    from pathlib import Path

    cfg_path = Path(__file__).resolve().parents[2] / "ci" / "cfg" / "microbenchmark_basic.yml"
    config = yaml.safe_load(cfg_path.read_text())
    expanded, _ = expand_config(config, MICROBENCH_DUTS, None)
    reasons = {
        i: MICROBENCH_DUTS[params["dut"]].validate(params) for i, params in enumerate(expanded)
    }
    assert {i: r for i, r in reasons.items() if r is not None} == {}
    assert {params["dut"] for params in expanded} == set(MICROBENCH_DUTS)


@pytest.mark.parametrize("dut,params", [("fifo", "_FIFO"), ("dwc", "_DWC")])
def test_throughput_estimate_without_cycle_model(dut, params):
    """FIFOs and data width converters report no expected cycles per frame, the builder must
    then skip the throughput estimate instead of dividing by zero."""
    from qonnx.transformation.general import GiveUniqueNodeNames as _Names

    from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
    from finn.builder.build_dataflow_steps import _estimated_throughput_fps
    from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles

    params = dict(globals()[params])
    model, _ = MICROBENCH_DUTS[dut].make_model(params, RFSOC)
    model = model.transform(_Names()).transform(AnnotateCycles())
    perf = model.analysis(dataflow_performance)
    assert perf["max_cycles"] == 0
    assert _estimated_throughput_fps(1e8, perf["max_cycles"]) is None
    assert _estimated_throughput_fps(1e8, 50) == 2e6


@pytest.mark.parametrize("in_elems,out_elems", [(4, 8), (8, 4)])
def test_dwc_dut_floorplan(in_elems, out_elems):
    """A standalone data width converter has no neighbour on its narrow side, which the
    floorplanning of the bitfile flow must tolerate (up- and downsizing)."""
    from finn.transformation.fpgadataflow.floorplan import Floorplan

    params = {**_DWC, "in_elems": in_elems, "out_elems": out_elems, "ch": 32}
    assert MICROBENCH_DUTS["dwc"].validate(params) is None
    model, _ = MICROBENCH_DUTS["dwc"].make_model(params, RFSOC)
    model = model.transform(GiveUniqueNodeNames()).transform(Floorplan())
    assert getHWCustomOp(model.graph.node[0]).get_nodeattr("slr") == -1


def test_instrumentation_and_frame_helpers():
    from finn.benchmarking.dut.microbench_base import (
        MAX_FRAME_CYCLES,
        frame_cycles_ok,
        output_words_ok,
    )

    # padded word width (multiple of 8) must be a multiple of the elements per word
    assert output_words_ok(8, 8) and output_words_ok(4, 2) and output_words_ok(2, 3)
    assert not output_words_ok(15, 4)  # 60 -> 64 bits, 64 % 15 != 0 (SWG run 60)
    assert not output_words_ok(3, 1)  # 3 -> 8 bits (DWC run 130)
    assert not output_words_ok(3, 2)  # 6 -> 8 bits (FMPadding run 191)
    assert not output_words_ok(5, 4)  # 20 -> 24 bits
    assert not output_words_ok(0, 8)
    assert frame_cycles_ok(1) and frame_cycles_ok(MAX_FRAME_CYCLES)
    assert not frame_cycles_ok(MAX_FRAME_CYCLES + 1) and not frame_cycles_ok(0)

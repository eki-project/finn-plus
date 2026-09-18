"""Per-operator feature specifications for empirical QoR estimation.

An :class:`OperatorFeatureSpec` is the single source of truth for which columns of the
microbenchmark database are used as regression features for a given operator. The same
column names must be produced when mapping a node of a FINN dataflow graph to features
(see ``finn.analysis.fpgadataflow.empirical_qor_estimation.node_features``), so that a
model fitted on the database can be applied to a model under construction.

Derived columns (datatype bitwidths, split [H, W] lists, ...) are computed by the pure
helper functions of this module on both sides, which keeps training and inference in sync.

This module must stay importable without FINN/QONNX (only pandas), because the CI job
that fits the models runs outside of the FINN environment.
"""

import pandas as pd
import re
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional

_INT_RE = re.compile(r"^(U?)INT(\d+)$")
_FLOAT_RE = re.compile(r"^FLOAT(\d+)$")
_FIXED_RE = re.compile(r"^FIXED<(\d+),(-?\d+)>$")
_SCALEDINT_RE = re.compile(r"^SCALEDINT<(\d+)>$")


class DataTypeInfo(NamedTuple):
    """Numeric summary of a QONNX datatype name (indexable for backwards compatibility)."""

    bitwidth: int
    signed: bool
    #: One of ``"int"``, ``"float"``, ``"fixed"``, ``"scaledint"``.
    kind: str


def parse_datatype(name: str) -> DataTypeInfo:
    """Return ``(bitwidth, signed, kind)`` for a QONNX datatype name such as ``"INT4"``.

    Handles the datatypes that occur in microbenchmarks without importing QONNX; falls
    back to ``qonnx.core.datatype.DataType`` for anything else if available.
    """
    match = _INT_RE.match(name)
    if match:
        return DataTypeInfo(int(match.group(2)), match.group(1) == "", "int")
    if name == "BINARY":
        return DataTypeInfo(1, False, "int")
    if name == "BIPOLAR":
        return DataTypeInfo(1, True, "int")
    if name == "TERNARY":
        return DataTypeInfo(2, True, "int")
    match = _FLOAT_RE.match(name)
    if match:
        return DataTypeInfo(int(match.group(1)), True, "float")
    match = _FIXED_RE.match(name)
    if match:
        return DataTypeInfo(int(match.group(1)), True, "fixed")
    match = _SCALEDINT_RE.match(name)
    if match:
        return DataTypeInfo(int(match.group(1)), True, "scaledint")
    try:
        from qonnx.core.datatype import DataType
    except ImportError as e:
        raise ValueError(f"Cannot parse datatype '{name}' without QONNX") from e
    dt = DataType[name]
    kind = "int" if dt.is_integer() else "fixed" if dt.is_fixed_point() else "float"
    return DataTypeInfo(dt.bitwidth(), dt.signed(), kind)


def datatype_features(prefix: str, name: Optional[str], with_kind: bool = False) -> dict:
    """Feature columns ``<prefix>_bitwidth``/``<prefix>_sign`` (and ``<prefix>_kind``) for a
    datatype name. ``None`` (e.g. no activation) yields ``None`` values."""
    if name is None:
        bitwidth, signed, kind = None, None, None
    else:
        bitwidth, signed, kind = parse_datatype(name)
    features: dict = {f"{prefix}_bitwidth": bitwidth, f"{prefix}_sign": signed}
    if with_kind:
        features[f"{prefix}_kind"] = kind
    return features


def add_datatype_columns(
    df: pd.DataFrame, col: str, prefix: str, with_kind: bool = False
) -> pd.DataFrame:
    """Add the :func:`datatype_features` columns derived from the datatype strings in ``col``."""
    rows = df[col].map(lambda name: datatype_features(prefix, name, with_kind))
    for key in datatype_features(prefix, "INT8", with_kind):
        df[key] = rows.map(lambda r, k=key: r[k])
    return df


def split_ints(df: pd.DataFrame, col: str, names: list[str]) -> pd.DataFrame:
    """Split a column holding lists (e.g. ``[H, W]``) into one integer column per entry."""
    for i, name in enumerate(names):
        df[name] = df[col].map(lambda v, i=i: v[i] if v is not None else None)
    return df


def broadcast_kind(rhs_shape: list, out_shape: list) -> str:
    """Classify how a constant operand is broadcast against the output of an elementwise op:
    ``"scalar"`` (one element), ``"channel"`` (one value per last-axis element),
    ``"full"`` (one value per output element) or ``"other"``."""
    n_rhs = 1
    for d in rhs_shape:
        n_rhs *= int(d)
    n_out = 1
    for d in out_shape:
        n_out *= int(d)
    if n_rhs == 1:
        return "scalar"
    if n_rhs == int(out_shape[-1]):
        return "channel"
    if n_rhs == n_out:
        return "full"
    return "other"


def _deriver(
    datatype_cols: Optional[dict[str, str]] = None,
    split_cols: Optional[dict[str, list[str]]] = None,
    with_kind: tuple[str, ...] = (),
    extra: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Build a ``derive_db_columns`` function from declarative column mappings."""

    def derive(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for prefix, col in (datatype_cols or {}).items():
            df = add_datatype_columns(df, col, prefix, with_kind=prefix in with_kind)
        for col, names in (split_cols or {}).items():
            df = split_ints(df, col, names)
        if extra is not None:
            df = extra(df)
        return df

    return derive


@dataclass(frozen=True)
class OperatorFeatureSpec:
    """Feature definition for one operator type.

    Attributes:
        name: Operator name, equal to the DUT name and the database subfolder (e.g. ``"mvau"``).
        op_types: FINN node op_types this spec applies to (e.g. ``("MVAU_hls", "MVAU_rtl")``).
        feature_cols: Columns used as regression features, must exist after
            :meth:`derive_db_columns` has been applied.
        derive_db_columns: Adds derived feature columns to a flattened database DataFrame.
        drop_broken_runs: Removes database rows known to contain invalid measurements.
        irrelevant_param_cols: ``params.*`` columns that do not influence the implementation
            (loop bounds such as the number of input vectors); dropped before deduplication.
        dut_info_keys: Keys the microbenchmark DUT writes to ``dut_info.json`` (documentation
            and test assertion; they appear as ``dut_info.<key>`` columns).
    """

    name: str
    op_types: tuple[str, ...]
    feature_cols: list[str]
    derive_db_columns: Callable[[pd.DataFrame], pd.DataFrame] = field(default=lambda df: df)
    drop_broken_runs: Callable[[pd.DataFrame], pd.DataFrame] = field(default=lambda df: df)
    irrelevant_param_cols: list[str] = field(default_factory=list)
    dut_info_keys: list[str] = field(default_factory=list)


#: dut_info keys every microbenchmark DUT writes (see finn.benchmarking.dut.microbench_base)
COMMON_DUT_INFO_KEYS = ["dut_node_name", "dut_op_type", "dut_backend"]


def _mvau_drop_broken_runs(df: pd.DataFrame) -> pd.DataFrame:
    """RTL MVAUs with PE=1 trigger an instrumentation degeneration bug (output stream
    unconnected), so their measurements are invalid."""
    if "params.backend" in df.columns and "dut_info.pe" in df.columns:
        broken = (df["params.backend"] == "rtl") & (df["dut_info.pe"] == 1)
        return df[~broken]
    return df


MVAU_SPEC = OperatorFeatureSpec(
    name="mvau",
    op_types=("MVAU_hls", "MVAU_rtl"),
    feature_cols=[
        "params.backend",
        "params.mem_mode",
        "params.ram_style",
        "params.ram_style_thr",
        # input/weight datatypes as bitwidth + sign instead of the raw string
        "idt_bitwidth",
        "idt_sign",
        "wdt_bitwidth",
        "wdt_sign",
        # activation datatype as string (None if the MVAU has no activation)
        "params.act",
        "params.mw",
        "params.mh",
        # folding as SIMD/PE (the database also holds SF/NF, but these are derived)
        "dut_info.simd",
        "dut_info.pe",
        # fraction of zero weights
        "dut_info.zero_weights",
    ],
    derive_db_columns=_deriver(datatype_cols={"idt": "params.idt", "wdt": "params.wdt"}),
    drop_broken_runs=_mvau_drop_broken_runs,
    # nhw (number of input vectors) is only the outer loop bound of the operator
    irrelevant_param_cols=["params.nhw"],
    dut_info_keys=["simd", "pe", "zero_weights", "easy_weights"],
)

THRESHOLDING_SPEC = OperatorFeatureSpec(
    name="thresholding",
    op_types=("Thresholding_hls", "Thresholding_rtl"),
    feature_cols=[
        "params.backend",
        "idt_bitwidth",
        "idt_sign",
        "odt_bitwidth",
        "odt_sign",
        "params.ch",
        "params.pe",
        # threshold datatype after bit width minimization
        "tdt_bitwidth",
        # HLS only (None for RTL)
        "params.mem_mode",
        "params.ram_style",
        # RTL only (0 for HLS)
        "params.depth_trigger_bram",
        "params.depth_trigger_uram",
    ],
    derive_db_columns=_deriver(
        datatype_cols={"idt": "params.idt", "odt": "params.odt", "tdt": "dut_info.tdt"}
    ),
    irrelevant_param_cols=["params.nhw"],
    dut_info_keys=["pe", "num_steps", "tdt", "tmem"],
)

SWG_SPEC = OperatorFeatureSpec(
    name="swg",
    op_types=("ConvolutionInputGenerator_rtl",),
    feature_cols=[
        "idt_bitwidth",
        "params.ifm_ch",
        "ifm_h",
        "ifm_w",
        "k_h",
        "k_w",
        "stride_h",
        "stride_w",
        "dilation_h",
        "dilation_w",
        "params.simd",
        "params.depthwise",
        "params.parallel_window",
        "params.m",
        "params.ram_style",
        "dut_info.impl_style",
        "dut_info.buffer_depth",
        "dut_info.is1D",
    ],
    derive_db_columns=_deriver(
        datatype_cols={"idt": "params.idt"},
        split_cols={
            "params.ifm_dim": ["ifm_h", "ifm_w"],
            "params.k": ["k_h", "k_w"],
            "params.stride": ["stride_h", "stride_w"],
            "params.dilation": ["dilation_h", "dilation_w"],
        },
    ),
    dut_info_keys=["impl_style", "buffer_depth", "ofm_dim", "out_width", "is1D"],
)

VVAU_SPEC = OperatorFeatureSpec(
    name="vvau",
    op_types=("VVAU_hls", "VVAU_rtl"),
    feature_cols=[
        "params.backend",
        "params.mem_mode",
        "params.ram_style",
        "params.resType",
        "idt_bitwidth",
        "idt_sign",
        "wdt_bitwidth",
        "wdt_sign",
        "params.act",
        "params.ch",
        "k_h",
        "k_w",
        "params.pe",
        "params.simd",
        "dut_info.zero_weights",
    ],
    derive_db_columns=_deriver(
        datatype_cols={"idt": "params.idt", "wdt": "params.wdt"},
        split_cols={"params.k": ["k_h", "k_w"]},
    ),
    # the spatial dimension is only the outer loop bound
    irrelevant_param_cols=["params.dim"],
    dut_info_keys=["pe", "simd", "zero_weights", "wmem", "tmem"],
)

FIFO_SPEC = OperatorFeatureSpec(
    name="fifo",
    op_types=("StreamingFIFO_rtl",),
    feature_cols=[
        "params.impl_style",
        # vivado impl_style only (None for rtl)
        "params.ram_style",
        "dut_info.width_bits",
        "params.depth",
        "dut_info.depth_adjusted",
        "dut_info.capacity_bits",
    ],
    derive_db_columns=_deriver(datatype_cols={"dtype": "params.dtype"}),
    # n (number of beats) is only the loop bound
    irrelevant_param_cols=["params.n"],
    dut_info_keys=["width_bits", "depth_adjusted", "capacity_bits"],
)

DWC_SPEC = OperatorFeatureSpec(
    name="dwc",
    op_types=("StreamingDataWidthConverter_hls", "StreamingDataWidthConverter_rtl"),
    feature_cols=[
        "params.backend",
        "dtype_bitwidth",
        "params.ch",
        "dut_info.in_width",
        "dut_info.out_width",
        "dut_info.ratio",
        "dut_info.integer_ratio",
    ],
    derive_db_columns=_deriver(datatype_cols={"dtype": "params.dtype"}),
    irrelevant_param_cols=["params.n"],
    dut_info_keys=["in_width", "out_width", "ratio", "integer_ratio"],
)

POOL_SPEC = OperatorFeatureSpec(
    name="pool",
    op_types=("Pool_hls",),
    feature_cols=[
        "params.function",
        "idt_bitwidth",
        "idt_sign",
        "odt_bitwidth",
        "params.ch",
        "params.pe",
        "k_h",
        "k_w",
        "dut_info.accum_bits",
    ],
    derive_db_columns=_deriver(
        datatype_cols={"idt": "params.idt", "odt": "dut_info.odt"},
        split_cols={"params.k": ["k_h", "k_w"]},
    ),
    # output image size is only the loop bound
    irrelevant_param_cols=["params.odim"],
    dut_info_keys=["odt", "in_width", "accum_bits", "size"],
)

FMPADDING_SPEC = OperatorFeatureSpec(
    name="fmpadding",
    op_types=("FMPadding_rtl",),
    feature_cols=[
        "idt_bitwidth",
        "params.ch",
        "params.simd",
        "idim_h",
        "idim_w",
        "pad_t",
        "pad_l",
        "pad_b",
        "pad_r",
    ],
    derive_db_columns=_deriver(
        datatype_cols={"idt": "params.idt"},
        split_cols={
            "params.idim": ["idim_h", "idim_w"],
            "params.padding": ["pad_t", "pad_l", "pad_b", "pad_r"],
        },
    ),
    dut_info_keys=["odim", "stream_width"],
)


def _eltwise_extra(df: pd.DataFrame) -> pd.DataFrame:
    df["c"] = df["params.shape"].map(lambda s: s[-1] if s is not None else None)
    return df


ELTWISE_SPEC = OperatorFeatureSpec(
    name="eltwise",
    op_types=(
        "ElementwiseAdd_hls",
        "ElementwiseAdd_rtl",
        "ElementwiseMul_hls",
        "ElementwiseMul_rtl",
    ),
    feature_cols=[
        "params.backend",
        "params.op",
        "lhs_bitwidth",
        "lhs_sign",
        "lhs_kind",
        # constant operand datatype after bit width minimization
        "rhs_bitwidth",
        "rhs_sign",
        "rhs_kind",
        "out_bitwidth",
        "params.pe",
        # how the constant operand is broadcast: scalar, channel or full
        "params.rhs_bcast",
        "dut_info.rhs_num_elems",
        "c",
        "params.mem_mode",
        "params.ram_style",
    ],
    derive_db_columns=_deriver(
        datatype_cols={
            "lhs": "params.lhs_dtype",
            "rhs": "dut_info.rhs_dtype",
            "out": "dut_info.out_dtype",
        },
        with_kind=("lhs", "rhs"),
        extra=_eltwise_extra,
    ),
    dut_info_keys=["rhs_dtype", "out_dtype", "rhs_num_elems", "wmem"],
)

#: All known operator feature specs, keyed by operator name.
SPECS: dict[str, OperatorFeatureSpec] = {
    spec.name: spec
    for spec in (
        MVAU_SPEC,
        THRESHOLDING_SPEC,
        SWG_SPEC,
        VVAU_SPEC,
        FIFO_SPEC,
        DWC_SPEC,
        POOL_SPEC,
        FMPADDING_SPEC,
        ELTWISE_SPEC,
    )
}


def spec_for_op_type(op_type: str) -> Optional[OperatorFeatureSpec]:
    """Return the spec covering a FINN node op_type, or None if the operator is unsupported."""
    for spec in SPECS.values():
        if op_type in spec.op_types:
            return spec
    return None

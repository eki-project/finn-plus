"""Per-operator feature specifications for empirical QoR estimation.

An :class:`OperatorFeatureSpec` is the single source of truth for which columns of the
microbenchmark database are used as regression features for a given operator. The same
column names must be produced when mapping a node of a FINN dataflow graph to features
(see ``finn.analysis.fpgadataflow.empirical_qor_estimation.node_features``), so that a
model fitted on the database can be applied to a model under construction.

This module must stay importable without FINN/QONNX (only pandas), because the CI job
that fits the models runs outside of the FINN environment.
"""

import pandas as pd
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

_DATATYPE_RE = re.compile(r"^(U?)INT(\d+)$")


def parse_datatype(name: str) -> tuple[int, bool]:
    """Return ``(bitwidth, signed)`` for a QONNX datatype name such as ``"INT4"``.

    Handles the integer-like datatypes that occur in microbenchmarks without importing
    QONNX; falls back to ``qonnx.core.datatype.DataType`` for anything else if available.
    """
    match = _DATATYPE_RE.match(name)
    if match:
        return int(match.group(2)), match.group(1) == ""
    if name == "BINARY":
        return 1, False
    if name == "BIPOLAR":
        return 1, True
    if name == "TERNARY":
        return 2, True
    try:
        from qonnx.core.datatype import DataType
    except ImportError as e:
        raise ValueError(f"Cannot parse datatype '{name}' without QONNX") from e
    dt = DataType[name]
    return dt.bitwidth(), dt.signed()


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
    """

    name: str
    op_types: tuple[str, ...]
    feature_cols: list[str]
    derive_db_columns: Callable[[pd.DataFrame], pd.DataFrame] = field(default=lambda df: df)
    drop_broken_runs: Callable[[pd.DataFrame], pd.DataFrame] = field(default=lambda df: df)


def _mvau_derive_db_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Split datatype strings into bitwidth and sign so regressors can exploit them."""
    df = df.copy()
    for prefix, col in (("idt", "params.idt"), ("wdt", "params.wdt")):
        parsed = df[col].map(parse_datatype)
        df[f"{prefix}_bitwidth"] = parsed.map(lambda x: x[0])
        df[f"{prefix}_sign"] = parsed.map(lambda x: x[1])
    return df


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
    derive_db_columns=_mvau_derive_db_columns,
    drop_broken_runs=_mvau_drop_broken_runs,
)

#: All known operator feature specs, keyed by operator name.
SPECS: dict[str, OperatorFeatureSpec] = {MVAU_SPEC.name: MVAU_SPEC}


def spec_for_op_type(op_type: str) -> Optional[OperatorFeatureSpec]:
    """Return the spec covering a FINN node op_type, or None if the operator is unsupported."""
    for spec in SPECS.values():
        if op_type in spec.op_types:
            return spec
    return None

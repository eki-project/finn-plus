"""Loading of the CI microbenchmark database for empirical QoR estimation.

The database is a directory with one subfolder per DUT/operator (``mvau``, ...), each
containing the JSON files written by ``ci/collect/collect.py``::

    {"dut": ..., "date": ..., "commit": ..., "pipeline_id": ..., "pipeline_name": ...,
     "runs": [{"params": {...}, "dut_info": {...}, "metrics": {...}}, ...]}

This module must stay importable without FINN/QONNX (only pandas), because the CI job
that fits the models runs outside of the FINN environment.
"""

import json
import logging
import numpy as np
import os
import pandas as pd
from dataclasses import dataclass
from typing import Any, Optional

from finn.qor.features import SPECS, OperatorFeatureSpec

logger = logging.getLogger(__name__)

#: Environment variable pointing to the microbenchmark database directory.
DATABASE_ENV_VAR = "FINN_MICROBENCHMARK_DATABASE"

#: Measured power columns the power target is derived from, as (column, scale to mW). The
#: schema of the measurement reports changed over time, so for every run the first column
#: with a value is used. The target is the PL/PS core rail (0V85): collect.py currently logs
#: its average over all iterations in mW (avg_0V85_power), older runs stored it in W
#: (power_pl_ps_load). The total board power would be avg_total_power resp. power_total_load.
POWER_COLS: list[tuple[str, float]] = [
    ("metrics.measurement.power.avg_0V85_power", 1.0),
    ("metrics.measurement.power.power_pl_ps_load", 1000.0),
]

#: Name of the derived power target column.
POWER_TARGET_COL = "power"

# Run parameters that do not influence the implementation and only add noise to the
# params-based deduplication.
_IRRELEVANT_PARAM_COLS = [
    "params.generate_outputs",
    "params.store_results_in_dvc_experiment",
    "params.store_results_in_dvc_data",
    # nhw (number of input vectors) is only the outer loop bound of the operator
    "params.nhw",
]


@dataclass
class LoadStats:
    """Row counts at the individual filtering stages of :func:`load_microbenchmark_database`."""

    files: int = 0
    total: int = 0
    after_filters: int = 0
    skipped: int = 0
    failed: int = 0
    broken: int = 0
    duplicates: int = 0
    final: int = 0

    def __str__(self) -> str:
        """One-line summary of the row counts for log output."""
        return (
            f"{self.files} files, {self.total} runs, {self.after_filters} after manual filters, "
            f"-{self.skipped} skipped, -{self.failed} failed, -{self.broken} broken, "
            f"-{self.duplicates} duplicates -> {self.final} samples"
        )


def resolve_database_path(database_path: Optional[str] = None) -> str:
    """Return the given path or fall back to ``$FINN_MICROBENCHMARK_DATABASE``."""
    path = database_path or os.environ.get(DATABASE_ENV_VAR)
    if not path:
        raise ValueError(f"Database path not given and {DATABASE_ENV_VAR} is not set")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Database path does not exist: {path}")
    return path


def make_hashable(obj: Any) -> Any:
    """Recursively convert lists/dicts to tuples so the result is hashable."""
    if isinstance(obj, (list, tuple)):
        return tuple(make_hashable(e) for e in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    return obj


def read_operator_runs(database_path: str, operator: str) -> tuple[pd.DataFrame, int]:
    """Read all JSON files of one operator and flatten them into one row per run.

    Returns the DataFrame and the number of files read. File-level metadata (date, commit,
    pipeline_id, ...) is repeated on every run row.
    """
    operator_dir = os.path.join(database_path, operator)
    if not os.path.isdir(operator_dir):
        raise FileNotFoundError(f"No database folder for operator '{operator}' in {database_path}")
    files = sorted(f for f in os.listdir(operator_dir) if f.endswith(".json"))
    rows = []
    for filename in files:
        with open(os.path.join(operator_dir, filename), "r") as f:
            entry = json.load(f)
        metadata = {k: v for k, v in entry.items() if k != "runs"}
        for run in entry.get("runs", []):
            rows.append({**metadata, **run})
    return pd.json_normalize(rows), len(files)


def derive_power_target(
    df: pd.DataFrame,
    power_cols: Optional[list[tuple[str, float]]] = None,
    out_col: str = POWER_TARGET_COL,
) -> pd.DataFrame:
    """Add a power target column in mW: measured power minus the (minimum observed) baseline.

    For every run the first of ``power_cols`` (default :data:`POWER_COLS`) that has a value is
    used, scaled to mW, so that runs recorded with different report schemas are combined. The
    smallest measured value is treated as the static/baseline power of the platform; a small
    offset keeps the target strictly positive so relative error metrics stay defined. Runs
    without any measurement keep NaN in the target column and are ignored when fitting. If
    none of the columns exists, the DataFrame is returned unchanged with a warning.
    """
    power_cols = POWER_COLS if power_cols is None else power_cols
    available = [(col, scale) for col, scale in power_cols if col in df.columns]
    if not available:
        logger.warning("None of the power columns %s in database", [c for c, _ in power_cols])
        return df
    df = df.copy()
    measured = pd.Series(np.nan, index=df.index, dtype=float)
    for col, scale in available:
        use = measured.isna() & df[col].notna()
        measured[use] = df.loc[use, col].astype(float) * scale
        logger.info("Power target: %d runs from '%s' (x%g)", int(use.sum()), col, scale)
    baseline = measured.min() - 0.01
    df[out_col] = measured - baseline
    logger.info(
        "Power baseline (min observed - 0.01 mW): %.3f mW, shifted into '%s' "
        "(%d of %d runs measured)",
        baseline,
        out_col,
        int(measured.notna().sum()),
        len(df),
    )
    return df


def load_microbenchmark_database(
    operator: str,
    database_path: Optional[str] = None,
    include_commit: Optional[list[str]] = None,
    exclude_commit: Optional[list[str]] = None,
    include_pipeline_id: Optional[list[int]] = None,
    exclude_pipeline_id: Optional[list[int]] = None,
    derive_power: bool = True,
    spec: Optional[OperatorFeatureSpec] = None,
    **column_filters: Any,
) -> tuple[pd.DataFrame, LoadStats]:
    """Load, filter and deduplicate the microbenchmark runs of one operator.

    Processing order:
        1. flatten all runs (nested dicts become dot-separated columns),
        2. drop irrelevant parameter columns,
        3. apply commit/pipeline include/exclude lists (commits match by prefix, so short
           hashes work) and arbitrary ``column == value`` filters from ``column_filters``,
        4. drop skipped/failed runs and runs the operator spec marks as broken,
        5. keep only the newest run for each unique combination of all ``params.*`` columns,
        6. derive the feature columns of the spec and (if ``derive_power``) the power target,
           see :func:`derive_power_target`.

    Returns the DataFrame (index reset) and the :class:`LoadStats` of the run.
    """
    spec = spec or SPECS[operator]
    stats = LoadStats()
    df, stats.files = read_operator_runs(resolve_database_path(database_path), operator)
    stats.total = len(df)
    if df.empty:
        raise ValueError(f"No runs found for operator '{operator}'")

    df["pipeline_id"] = df["pipeline_id"].astype(int)
    df = df.drop(columns=[c for c in _IRRELEVANT_PARAM_COLS if c in df.columns])

    if include_commit:
        df = df[df["commit"].str.startswith(tuple(include_commit))]
    if exclude_commit:
        df = df[~df["commit"].str.startswith(tuple(exclude_commit))]
    if include_pipeline_id:
        df = df[df["pipeline_id"].isin(include_pipeline_id)]
    if exclude_pipeline_id:
        df = df[~df["pipeline_id"].isin(exclude_pipeline_id)]
    for key, value in column_filters.items():
        if key not in df.columns:
            raise KeyError(f"Filter column '{key}' not found in database")
        df = df[df[key] == value]
    stats.after_filters = len(df)

    status = df["metrics.status"]
    stats.skipped = int((status == "skipped").sum())
    stats.failed = int((status == "failed").sum())
    df = df[(status != "skipped") & (status != "failed")]
    n_before = len(df)
    df = spec.drop_broken_runs(df)
    stats.broken = n_before - len(df)

    # Deduplicate: keep only the newest run for each unique parameter set
    n_before = len(df)
    params_cols = sorted(c for c in df.columns if c.startswith("params."))
    if params_cols and "date" in df.columns:
        key = df[params_cols].apply(lambda row: tuple(make_hashable(v) for v in row), axis=1)
        df = df.assign(_params_key=key)
        df = df.sort_values(["date", "_params_key"], ascending=[False, True])
        df = df.drop_duplicates(subset=["_params_key"], keep="first").drop(columns="_params_key")
    stats.duplicates = n_before - len(df)

    df = spec.derive_db_columns(df)
    if derive_power:
        df = derive_power_target(df)
    df = df.reset_index(drop=True)
    stats.final = len(df)
    logger.info("Loaded microbenchmark database for '%s': %s", operator, stats)
    return df, stats

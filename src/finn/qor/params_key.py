"""Canonical identity of a microbenchmark run: its parameter set.

The result database is deduplicated on all ``params.*`` values and the random sampler skips
configurations that already exist, so both must agree on how a parameter set is turned into
a key. This module is the shared definition. It also offers a stdlib-only reader of the
database JSON files for the sampler, which runs on the cluster without pandas.
"""

import json
import os
from typing import Any, Iterable, Iterator, Optional

#: Run parameters that never influence the implementation (infrastructure switches).
IRRELEVANT_PARAMS: tuple[str, ...] = (
    "generate_outputs",
    "store_results_in_dvc_experiment",
    "store_results_in_dvc_data",
)


def irrelevant_param_cols() -> list[str]:
    """:data:`IRRELEVANT_PARAMS` as flattened database column names."""
    return [f"params.{name}" for name in IRRELEVANT_PARAMS]


def make_hashable(obj: Any) -> Any:
    """Recursively convert lists/dicts to tuples so the result is hashable."""
    if isinstance(obj, (list, tuple)):
        return tuple(make_hashable(e) for e in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    return obj


def params_key(
    params: dict, keys: Optional[Iterable[str]] = None, ignore: Iterable[str] = IRRELEVANT_PARAMS
) -> tuple:
    """Hashable key of a parameter set: sorted ``(name, value)`` pairs over ``keys``
    (default: all keys of ``params``) minus ``ignore``. Missing keys count as ``None``."""
    ignore = set(ignore)
    names = sorted(set(keys) if keys is not None else set(params))
    return tuple((name, make_hashable(params.get(name))) for name in names if name not in ignore)


def iter_database_runs(database_path: str, operator: str) -> Iterator[tuple[dict, dict]]:
    """Yield ``(file metadata, run)`` for every run of an operator in the database
    (``<database>/<operator>/*.json``); nothing if the operator folder does not exist."""
    operator_dir = os.path.join(database_path, operator)
    if not os.path.isdir(operator_dir):
        return
    for filename in sorted(os.listdir(operator_dir)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(operator_dir, filename), "r") as f:
            entry = json.load(f)
        metadata = {k: v for k, v in entry.items() if k != "runs"}
        for run in entry.get("runs", []):
            yield metadata, run


def run_status(run: dict) -> Optional[str]:
    """Status (``ok``/``skipped``/``failed``) recorded for a database run."""
    return (run.get("metrics") or {}).get("status")


def existing_param_keys(
    database_path: str,
    operator: str,
    keys: Iterable[str],
    statuses: Optional[Iterable[str]] = ("ok",),
    ignore: Iterable[str] = IRRELEVANT_PARAMS,
) -> tuple[set[tuple], dict]:
    """Keys (see :func:`params_key`, projected onto ``keys``) of all database runs of an
    operator whose status is in ``statuses`` (None = any status). Returns the set and a
    stats dict (files, runs, keys)."""
    keys = list(keys)
    statuses = None if statuses is None else set(statuses)
    found: set[tuple] = set()
    files: set[str] = set()
    n_runs = 0
    for metadata, run in iter_database_runs(database_path, operator):
        n_runs += 1
        files.add(str(metadata.get("pipeline_id")))
        if statuses is not None and run_status(run) not in statuses:
            continue
        found.add(params_key(run.get("params", {}), keys, ignore))
    return found, {"files": len(files), "runs": n_runs, "keys": len(found)}

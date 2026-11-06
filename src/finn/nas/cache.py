"""Result caches for the oracle."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Protocol

from finn.nas.result import EvalResult


class ResultCache(Protocol):
    """Key/value store for :class:`EvalResult` s."""

    def get(self, key: str) -> EvalResult | None:
        """Return the cached result for ``key`` or ``None``."""
        ...

    def put(self, key: str, result: EvalResult) -> None:
        """Store ``result`` under ``key``."""
        ...


class SqliteCache:
    """Cache in a single SQLite file (WAL mode, safe for a few concurrent workers)."""

    def __init__(self, path: str | Path) -> None:
        """Open (and if needed create) the cache database at ``path``."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute(
                "CREATE TABLE IF NOT EXISTS results ("
                "key TEXT PRIMARY KEY, created REAL, result_json TEXT)"
            )

    def _connect(self) -> sqlite3.Connection:
        """Open a connection that waits for concurrent writers."""
        con = sqlite3.connect(str(self.path), timeout=30.0)
        con.execute("PRAGMA busy_timeout=30000")
        return con

    def get(self, key: str) -> EvalResult | None:
        """Return the cached result for ``key`` or ``None``."""
        with self._connect() as con:
            row = con.execute("SELECT result_json FROM results WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return EvalResult.from_dict(json.loads(row[0]))

    def put(self, key: str, result: EvalResult) -> None:
        """Store ``result`` under ``key`` (replacing an existing entry)."""
        payload = json.dumps(result.to_dict())
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO results (key, created, result_json) VALUES (?, ?, ?)",
                (key, time.time(), payload),
            )

    def __len__(self) -> int:
        """Return the number of cached results."""
        with self._connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM results").fetchone()[0])


class JsonDirCache:
    """One JSON file per key. Robust on network file systems where SQLite locking is not."""

    def __init__(self, directory: str | Path) -> None:
        """Use ``directory`` (created if missing) as cache root."""
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        """Return the file that stores ``key``."""
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> EvalResult | None:
        """Return the cached result for ``key`` or ``None``."""
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return EvalResult.from_dict(json.loads(p.read_text()))
        except (OSError, ValueError):
            return None

    def put(self, key: str, result: EvalResult) -> None:
        """Store ``result`` atomically (write to temp file, then rename)."""
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(result.to_dict()))
        tmp.replace(p)

    def __len__(self) -> int:
        """Return the number of cached results."""
        return sum(1 for _ in self.directory.glob("*/*.json"))


def make_cache(path: str | Path) -> ResultCache:
    """Create a cache: a directory (existing or ending in ``/``) gives a JSON cache, else SQLite."""
    p = Path(path)
    if p.is_dir() or str(path).endswith(("/", os.sep)):
        return JsonDirCache(p)
    return SqliteCache(p)

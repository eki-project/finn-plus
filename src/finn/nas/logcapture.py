"""Capture FINN log messages and Python warnings emitted during an evaluation."""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

FINN_LOGGER_NAME = "finn_logger"


class _ListHandler(logging.Handler):
    """Logging handler that appends formatted messages to a list."""

    def __init__(self, records: list[tuple[int, str]]) -> None:
        """Create a handler writing into ``records``."""
        super().__init__()
        self.records = records

    def emit(self, record: logging.LogRecord) -> None:
        """Store the record's level and message."""
        self.records.append((record.levelno, record.getMessage()))


@dataclass
class CapturedLogs:
    """Messages collected by :func:`capture_finn_logs`."""

    records: list[tuple[int, str]] = field(default_factory=list)
    python_warnings: list[str] = field(default_factory=list)

    def messages(self, min_level: int = logging.WARNING) -> list[str]:
        """Return all captured FINN messages at or above ``min_level``."""
        return [msg for lvl, msg in self.records if lvl >= min_level]

    def mark(self) -> int:
        """Return a position marker to be used with :meth:`since`."""
        return len(self.records)

    def since(self, mark: int, min_level: int = logging.WARNING) -> list[str]:
        """Return messages captured after ``mark``."""
        return [msg for lvl, msg in self.records[mark:] if lvl >= min_level]


@contextmanager
def capture_finn_logs(min_level: int = logging.WARNING) -> Iterator[CapturedLogs]:
    """Attach a list handler to the FINN logger for the duration of the context.

    Outside the FINN CLI the logger has no handlers and may be disabled, so this also makes
    sure messages at ``min_level`` actually reach the handler. Python ``warnings`` (used by
    qonnx/onnx) are recorded as well.
    """
    captured = CapturedLogs()
    logger = logging.getLogger(FINN_LOGGER_NAME)
    handler = _ListHandler(captured.records)
    handler.setLevel(min_level)
    old_level = logger.level
    old_disabled = logger.disabled
    logger.addHandler(handler)
    logger.disabled = False
    if logger.level == logging.NOTSET or logger.level > min_level:
        logger.setLevel(min_level)
    try:
        with warnings.catch_warnings(record=True) as pyw:
            warnings.simplefilter("always")
            try:
                yield captured
            finally:
                captured.python_warnings.extend(str(w.message) for w in pyw)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logger.disabled = old_disabled

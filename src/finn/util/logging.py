"""Logging utilities for FINN using Rich console.

This module provides logging configuration and utilities for FINN,
including a Rich console for formatted output.
"""

import logging
from rich.console import Console
from types import TracebackType

# Top level console used by logger
# Can be retrieved to create for example status displays in Rich
_RICH_CONSOLE = Console()


def get_console() -> Console:
    """Get the global Rich console instance used by the FINN logger.

    Returns
    -------
    Console
        The Rich console instance.
    """
    return _RICH_CONSOLE


def set_console(console: Console) -> None:
    """Set the global Rich console instance used by the FINN logger.

    Parameters
    ----------
    console : Console
        The Rich console instance to set.
    """
    global _RICH_CONSOLE
    _RICH_CONSOLE = console


log = logging.getLogger("finn_logger")


class LogDisabledConsole:
    """Use to get a console to use for Rich formatting without logging enabled."""

    def __init__(self) -> None:
        """Initialize the context manager and disable logging."""
        log.disabled = True

    def __enter__(self) -> Console:
        """Enter the context and return the Rich console.

        Returns
        -------
        Console
            The Rich console instance.
        """
        return _RICH_CONSOLE

    def __exit__(
        self,
        tp: type[BaseException] | None,
        vl: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit the context and re-enable logging.

        Parameters
        ----------
        tp : type[BaseException] | None
            Exception type.
        vl : BaseException | None
            Exception value.
        tb : TracebackType | None
            Exception traceback.
        """
        log.disabled = False

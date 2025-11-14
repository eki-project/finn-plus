import logging
from rich.console import Console

# Top level console used by logger
# Can be retrieved to create for example status displays in Rich
_RICH_CONSOLE = Console()


def get_console() -> Console:
    return _RICH_CONSOLE


log = logging.getLogger("finn_logger")


class LogDisabledConsole:
    """Use to get a console to use for Rich formatting without logging enabled."""

    def __init__(self) -> None:
        log.disabled = True

    def __enter__(self) -> Console:
        return _RICH_CONSOLE

    def __exit__(self, tp, vl, tb) -> None:
        log.disabled = False
        return None

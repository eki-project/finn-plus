"""Handle logging related functionality."""
import logging
from rich.console import Console
from rich.progress import Progress, TaskID
from threading import Lock

log = logging.getLogger("finn_logger")

_RICH_CONSOLE = Console()


class DisabledLoggingConsole:
    """Contextmanager to use the current rich console without logging active."""

    def __init__(self) -> None:
        log.disabled = True

    def __enter__(self) -> Console:
        return _RICH_CONSOLE

    def __exit__(self, tp, vl, tb) -> None:
        log.disabled = False


def get_console() -> Console:
    return _RICH_CONSOLE


class ThreadsafeProgressDisplay:
    """Small helper to display multithreaded display bars.
    Logging has to be disabled before usage.
    """

    def __init__(
        self, tasks: list[str], totals: list[int | float], descriptions: list[str]
    ) -> None:
        """Create a new progress display."""
        self.lock = Lock()
        self.state: dict[str, int | float] = dict.fromkeys(tasks, 0)
        self.ptasks: dict[str, TaskID] = {}
        self.totals_state = dict(zip(tasks, totals, strict=True))

        self.tasks: list[str] = tasks
        self.totals: list[float | int] = totals
        self.descriptions: list[str] = descriptions
        assert len(tasks) == len(totals)
        assert len(totals) == len(descriptions)

    def start(self) -> None:
        """Start the display."""
        self.progress = Progress(transient=True, redirect_stdout=False, redirect_stderr=False)
        self.progress.start()
        for task, desc, total in zip(self.tasks, self.descriptions, self.totals, strict=True):
            self.ptasks[task] = self.progress.add_task(desc, total=total)

    def update(self, task: str, value: float | None = None, total: float | None = None) -> None:
        """Update a value and the progress bar. If the task does not exist do nothing.
        This is practical, because it means any method can update the progressbar
        without any danger. Just the initially calling method must create a fitting display object.

        If value is None, the value is incremented once.
        """
        if task in self.state and task in self.ptasks:
            # NOTE: rich.progress at some point apparently became threadsafe,
            # but just to be extra sure we add a lock here.
            with self.lock:
                if value is None:
                    self.state[task] += 1
                else:
                    self.state[task] = value
                if total is not None:
                    self.totals_state[task] = total
                self.progress.update(
                    self.ptasks[task],
                    completed=self.state[task],
                    refresh=True,
                    total=self.totals_state[task],
                )

    def stop(self) -> None:
        """Stop the display."""
        self.progress.stop()

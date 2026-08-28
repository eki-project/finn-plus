"""Multiprocessing start method configuration for FINN.

``fork`` is unsafe in a process that has already loaded threaded native libraries
(PyTorch, ONNX Runtime, Vivado tooling), and Python is moving away from it as the
default. FINN therefore selects a start method that creates a fresh interpreter,
which is set once per process before any pool or worker is created.
"""

import multiprocessing as mp
import os

# "spawn" rather than "forkserver": the forkserver daemon is started at the first
# pool creation and keeps the environment it had at that moment, so later updates
# (e.g. the per-test FINN_BUILD_DIR set by the isolate_build_dir fixture) would
# never reach the workers. "spawn" starts each child fresh and always passes the
# current environment.
DEFAULT_START_METHOD = "spawn"


def get_configured_start_method() -> str:
    """Return the configured start method, defaulting to forkserver."""
    return os.environ.get("FINN_MP_START_METHOD", DEFAULT_START_METHOD)


def configure_start_method(start_method: str | None = None) -> str:
    """Set the process-wide multiprocessing start method and return it.

    This must run before any pool or child process is created, while the process is
    still single-threaded, so that the start method is in effect for every later
    ``multiprocessing`` call, including those made by libraries such as ``qonnx``.

    Args:
        start_method: Start method to use. If None, resolved from the environment.

    Returns:
        The start method now in effect.
    """
    if start_method is None:
        start_method = get_configured_start_method()

    if mp.get_start_method(allow_none=True) != start_method:
        mp.set_start_method(start_method, force=True)

    # Children get a fresh interpreter, so make sure the settings file path survives
    # into them; finn.util.settings rebuilds the settings from it on demand.
    if start_method != "fork":
        _ensure_settings_path_exported()

    return start_method


def _ensure_settings_path_exported() -> None:
    """Export FINN_SETTINGS so child interpreters can rebuild the global settings."""
    if "FINN_SETTINGS" in os.environ:
        return

    from finn.util.settings import _SETTINGS

    if _SETTINGS is None:
        return
    settings_path = _SETTINGS.get_path()
    if settings_path.exists():
        os.environ["FINN_SETTINGS"] = str(settings_path)

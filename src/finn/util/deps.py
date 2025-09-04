import os
from pathlib import Path

from finn.util.exception import FINNInternalError


def get_deps_path() -> Path:
    """Get the dependency path from the environment variable.
    If it is not set, use the default location"""
    if "FINN_DEPS" not in os.environ.keys():
        return Path.home() / ".finn" / "deps"
    return Path(os.environ["FINN_DEPS"])


# TODO: Move to own file?
def get_cache_path() -> Path:
    """Return the path to the cache."""
    if "FINN_IP_CACHE" not in os.environ.keys():
        raise FINNInternalError(
            "FINN_IP_CACHE environment variable not found! This may be a "
            "bug, since the setup (run_finn.py) should always set this "
            "variable!"
        )
    return Path(os.environ["FINN_IP_CACHE"])

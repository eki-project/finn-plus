import os
from pathlib import Path


def get_deps_path() -> Path:
    """Get the dependency path from the environment variable.
    If it is not set, use the default location"""
    if "FINN_DEPS" not in os.environ.keys():
        return Path.home() / ".finn" / "deps"
    return Path(os.environ["FINN_DEPS"])

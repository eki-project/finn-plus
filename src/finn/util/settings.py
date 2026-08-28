"""Global FINN settings management.

This module provides access to the global FINN settings instance that is
initialized when FINN is started via run_finn.py.
"""

import os
from pathlib import Path

from finn.interface.settings import FINNSettings
from finn.util.exception import FINNUserError

_SETTINGS: FINNSettings | None = None


def _initialize_from_environment() -> FINNSettings | None:
    """Rebuild the settings from the environment, or return None if not possible.

    Only ``fork`` copies the parent's memory into the child, so with ``spawn`` and
    ``forkserver`` a child process starts a fresh interpreter in which the global
    settings instance is unset. The environment is inherited in all cases, so
    ``FINN_SETTINGS`` lets the child reconstruct the same settings the parent used.
    """
    settings_path = os.environ.get("FINN_SETTINGS")
    build_dir = os.environ.get("FINN_BUILD_DIR")
    if settings_path is not None and Path(settings_path).exists():
        # FINN_BUILD_DIR is exported as an absolute path, so it can stand in for the
        # flow config that would otherwise be needed to resolve a relative build dir.
        return FINNSettings.init(
            override_settings_path=Path(settings_path),
            flow_config=Path(build_dir) / "dummy.yaml" if build_dir is not None else None,
            auto_set_environment_vars=False,
        )
    if build_dir is not None:
        # No settings file to read, but the build directory alone is enough to rebuild
        # usable settings, which is all a worker needs to place its generated code.
        return FINNSettings.init(
            flow_config=Path(build_dir) / "dummy.yaml",
            finn_build_dir=build_dir,
            auto_set_environment_vars=False,
        )
    return None


def initialize_dummy_settings() -> None:
    """Initialize and set the global settings. This might be useful when for example running
    FINN Transformation outside the FINN CLI context.

    Since this constructs a settings object, if the FINN_SETTINGS environment variable is given,
    it is used for the path to the settings file.
    """
    global _SETTINGS
    _SETTINGS = FINNSettings.init(flow_config=Path("dummy.yaml"))


def get_settings() -> FINNSettings:
    """Get the global FINN settings instance.

    Returns
    -------
    FINNSettings
        The global FINN settings instance

    Raises
    ------
    FINNUserError
        If FINN was not properly started via run_finn.py
    """
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = _initialize_from_environment()
    if _SETTINGS is None:
        raise FINNUserError(
            "Could not find global settings. Was FINN properly started via run_finn.py? "
            "If you are executing parts of FINN outside the typical flow, you might have "
            "to initialize settings using `finn.util.settings.initialize_dummy_settings()` "
            "first. For further information refer to the functions documentation."
        )
    return _SETTINGS

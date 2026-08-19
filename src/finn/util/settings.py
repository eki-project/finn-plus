"""Global FINN settings management.

This module provides access to the global FINN settings instance that is
initialized when FINN is started via run_finn.py.
"""

from pathlib import Path

from finn.interface.settings import FINNSettings
from finn.util.exception import FINNUserError

_SETTINGS: FINNSettings | None = None


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
    if _SETTINGS is None:
        raise FINNUserError(
            "Could not find global settings. Was FINN properly started via run_finn.py? "
            "If you are executing parts of FINN outside the typical flow, you might have "
            "to initialize settings using `finn.util.settings.initialize_dummy_settings()` "
            "first. For further information refer to the functions documentation."
        )
    return _SETTINGS

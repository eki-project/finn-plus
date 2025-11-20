"""Global FINN settings management.

This module provides access to the global FINN settings instance that is
initialized when FINN is started via run_finn.py.
"""

from finn.interface.settings import FINNSettings
from finn.util.exception import FINNUserError

_SETTINGS: FINNSettings | None = None


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
            "Could not find global settings. Was FINN properly started via run_finn.py?"
        )
    return _SETTINGS

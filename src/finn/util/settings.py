from finn.interface.settings import FINNSettings
from finn.util.exception import FINNUserError

_SETTINGS: FINNSettings | None = None


def get_settings() -> FINNSettings:
    if _SETTINGS is None:
        raise FINNUserError(
            "Could not find global settings. Was FINN properly started via run_finn.py?"
        )
    return _SETTINGS

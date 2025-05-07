from __future__ import annotations

import os
from pathlib import Path

from finn.interface.interface_utils import error, read_yaml, write_yaml

# Variables that need to be Path() objects
_SETTINGS_PATH_VARS = ["FINN_DEPS", "FINN_BUILD_DIR"]

# Default fallback settings
# FINN_BUILD_DIR is not existent. It should only be loaded from a settings file!
_SETTINGS: dict[str, Path | str] = {
    "FINN_DEPS": Path.home() / ".finn" / "deps",
}


def _resolve_settings_path() -> Path | None:
    """Best effort to find the settings file. If it is found nowhere and isnt provided
    via an environment variable (FINN_SETTINGS), return None"""
    if "FINN_SETTINGS" in os.environ.keys():
        p = Path(os.environ["FINN_SETTINGS"])
        if p.exists():
            return p
        error(f"Settings path specified via FINN_SETTINGS, but settings could not be found at {p}!")
        return None
    paths = [
        Path(__file__).parent.parent / "settings.yaml",
        Path.home() / ".finn" / "settings.yaml",
        Path.home() / ".config" / "settings.yaml",
    ]
    for path in paths:
        if path.exists():
            return path
    return None


def _update_settings() -> None:
    """Update the settings. This means loading the settings from any of the paths and setting the
    global dictionary to the new value. If no settings are found this returns immediately without
    an error"""
    global _SETTINGS
    settings_path = _resolve_settings_path()
    if settings_path is None:
        return
    temp_settings = read_yaml(settings_path)
    if temp_settings is None:
        return
    for setting_key in temp_settings.keys():
        _SETTINGS[setting_key] = temp_settings[setting_key]
        if setting_key in _SETTINGS_PATH_VARS:
            _SETTINGS[setting_key] = Path(_SETTINGS[setting_key]).expanduser()


def set_settings(s: dict) -> None:
    """Update the global setting dict"""
    global _SETTINGS
    _SETTINGS.update(s)


def write_settings() -> None:
    """Write settings back to the resolved path. If the path cant be resolved, simply return"""
    global _SETTINGS
    settings_path = _resolve_settings_path()
    mod_settings = {key: str(_SETTINGS[key]) for key in _SETTINGS.keys()}
    if settings_path is None:
        return
    write_yaml(mod_settings, settings_path)


def get_settings(force_update: bool = False) -> dict:
    """Retrieve the settings. If you suspect that settings changed, you can force an update"""
    if force_update:
        _update_settings()
    return _SETTINGS


def settings_found() -> bool:
    """Try to resolve the settings path. If none is found, return false, else true."""
    return _resolve_settings_path() is not None


def skip_update_by_default() -> bool:
    """Return whether the dependency update should be skipped. Reads the settings for this.
    Dep updates will NOT be skipped, unless specified otherwise"""
    settings = get_settings()
    if "AUTOMATIC_DEPENDENCY_UPDATES" not in settings.keys():
        return False
    value = settings["AUTOMATIC_DEPENDENCY_UPDATES"]
    assert type(value) is bool, "Field AUTOMATIC_DEPENDENCY_UPDATES in settings must be a bool"
    return not value


# Overwrite Settings when importing this module
_update_settings()

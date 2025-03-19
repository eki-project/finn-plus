from __future__ import annotations

import os
from pathlib import Path

from interface.interface_utils import error, read_yaml

# Variables that need to be Path() objects
_SETTINGS_PATH_VARS = ["DEFAULT_DEPS", "FINN_ROOT", "ENVVAR_CONFIG", "FINN_DEFAULT_BUILD_DIR"]

# Default fallback settings
_SETTINGS: dict[str, Path | str] = {
    "DEFAULT_DEPS": Path.home() / ".finn" / "deps",
    "FINN_ROOT": Path(__file__).parent,
    "ENVVAR_CONFIG": Path.home() / ".finn" / "envvars.yaml",
    "FINN_DEFAULT_BUILD_DIR": Path("/tmp/FINN_TMP"),
}

FINN_DEPS = {
    "finn-experimental": (
        "https://github.com/Xilinx/finn-experimental.git",
        "0724be21111a21f0d81a072fccc1c446e053f851",
    ),
    "brevitas": (
        "https://github.com/iksnagreb/brevitas.git",
        "003f9f4070c20639790c7b406a28612a089fc502",
    ),
    "cnpy": ("https://github.com/rogersce/cnpy.git", "4e8810b1a8637695171ed346ce68f6984e585ef4"),
    "oh-my-xilinx": (
        "https://github.com/maltanar/oh-my-xilinx.git",
        "0b59762f9e4c4f7e5aa535ee9bc29f292434ca7a",
    ),
    "finn-hlslib": (
        "https://github.com/Xilinx/finn-hlslib.git",
        "5c5ad631e3602a8dd5bd3399a016477a407d6ee7",
    ),
    "attention-hlslib": (
        "https://github.com/iksnagreb/attention-hlslib.git",
        "afc9720f10e551e1f734e137b21bb6d0a8342177",
    ),
}

VERILATOR = ("https://github.com/verilator/verilator", "v4.224")

FINN_BOARDFILES = {
    "avnet-bdf": (
        "https://github.com/Avnet/bdf.git",
        "2d49cfc25766f07792c0b314489f21fe916b639b",
        Path(),
    ),
    "xil-bdf": (
        "https://github.com/Xilinx/XilinxBoardStore.git",
        "8cf4bb674a919ac34e3d99d8d71a9e60af93d14e",
        Path("boards/Xilinx/rfsoc2x2"),
    ),
    "rfsoc4x2-bdf": (
        "https://github.com/RealDigitalOrg/RFSoC4x2-BSP.git",
        "13fb6f6c02c7dfd7e4b336b18b959ad5115db696",
        Path("board_files/rfsoc4x2"),
    ),
    "kv260-som-bdf": (
        "https://github.com/Xilinx/XilinxBoardStore.git",
        "98e0d3efc901f0b974006bc4370c2a7ad8856c79",
        Path("boards/Xilinx/kv260_som"),
    ),
}

REQUIRED_VERILATOR_VERSION = VERILATOR[1]

IS_POSIX = os.name == "posix"


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
        Path(__file__).parent / "settings.yaml",
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
            _SETTINGS[setting_key] = Path(_SETTINGS[setting_key])


def get_settings(force_update: bool = False) -> dict:
    """Retrieve the settings. If you suspect that settings changed, you can force an update"""
    if force_update:
        _update_settings()
    return _SETTINGS


def settings_found() -> bool:
    """Try to resolve the settings path. If none is found, return false, else true."""
    return _resolve_settings_path() is not None


# Overwrite Settings when importing this module
_update_settings()

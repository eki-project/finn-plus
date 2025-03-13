from __future__ import annotations
import os
import yaml
from pathlib import Path
from rich.console import Console

console = Console()

# FINN Internal Defaults - for configuring the compiler
SETTINGS_PATH = Path.home() / ".finn" / "settings.yaml"
# Variables that need to be Path() objects
SETTINGS_PATH_VARS = ["DEFAULT_DEPS", "FINN_ROOT", "ENVVAR_CONFIG", "FINN_DEFAULT_BUILD_DIR"]
# Default fallback settings
SETTINGS: dict[str, Path | str] = {
    "DEFAULT_DEPS": Path.home() / ".finn" / "deps",
    "FINN_ROOT": Path(__file__).parent.parent,
    "ENVVAR_CONFIG": Path.home() / ".finn" / "envvars.yaml",
    "FINN_DEFAULT_BUILD_DIR": Path("/tmp/FINN_TMP"),
}

FINN_ENVVARS = {
    "PLATFORM_REPO_PATHS": "/opt/xilinx/platforms",
    "XRT_DEB_VERSION": "xrt_202220.2.14.354_22.04-amd64-xrt",
    "XILINX_LOCAL_USER_DATA": "no",
    "VIVADO_PATH": "/tools/Xilinx/Vivado/2022.1",
    "VITIS_PATH": "/tools/Xilinx/Vitis/2022.1",
    "HLS_PATH": "/tools/Xilinx/Vitis_HLS/2022.1",
}

IS_POSIX = os.name == "posix"


def read_yaml(p: Path) -> dict | None:
    """Read a yaml file and return its contents. If the file does not exist, return None"""
    if p.exists():
        with p.open() as f:
            return yaml.load(f, yaml.Loader)
    else:
        return None


def write_yaml(data: dict, p: Path) -> bool:
    """Try writing the given data to a yaml file. If this fails, return false otherwise
    true"""
    try:
        with p.open("w+") as f:
            yaml.dump(data, f, yaml.Dumper)
            return True
    except (OSError, yaml.error.YAMLError):
        return False


def update_settings() -> None:
    """Update the settings"""
    global SETTINGS, SETTINGS_PATH
    temp_settings = read_yaml(SETTINGS_PATH)
    if temp_settings is None:
        console.print("[bold]STATUS: [bold]Settings file not found, using default fallback values.")
    else:
        for setting_key in temp_settings.keys():
            SETTINGS[setting_key] = temp_settings[setting_key]
            if setting_key in SETTINGS_PATH_VARS:
                SETTINGS[setting_key] = Path(SETTINGS[setting_key])


def update_finn_envvars() -> None:
    """Update FINN environment variables. Still lower priority than passed env vars"""
    global FINN_ENVVARS, SETTINGS
    values = read_yaml(SETTINGS["ENVVAR_CONFIG"])
    if values is not None:
        FINN_ENVVARS.update(values)
    else:
        console.print(
            "[bold orange1]WARNING: [/bold orange1][orange3]Environment variable config "
            "not found. Using defaults. It is highly recommended that you create an environment "
            "variable configuration file![/orange3]"
        )


# Overwrite Settings when importing this module
update_settings()

# Overwrite environment variables now
update_finn_envvars()

"Manage environment variables. Called by run_finn.py"
from __future__ import annotations

import os
import psutil
from pathlib import Path
from rich.console import Console
from rich.table import Table

from interface.interface_globals import FINN_ENVVARS, SETTINGS


def get_finn_envvars(
    deps: Path | None, config_path: Path, local_temps: bool, num_workers: int
) -> dict[str, str]:
    """FINN required environment variables"""
    cpucount = psutil.cpu_count(logical=False)
    if num_workers == -1:
        cpus = int(0.75 * cpucount) if cpucount is not None else 1
    else:
        cpus = num_workers

    # Settings-File > SETTINGS default value
    finn_root = str(SETTINGS["FINN_ROOT"])

    # Cmdline param > Settings-File > SETTINGS default value
    deps_value = SETTINGS["DEFAULT_DEPS"]
    if deps is not None:
        deps_value = str(deps)

    # Envvar > Local-temps > Settings-File > SETTINGS default value
    build_dir = ""
    if "FINN_HOST_BUILD_DIR" in os.environ.keys() and os.environ["FINN_HOST_BUILD_DIR"] != "":
        build_dir = os.environ["FINN_HOST_BUILD_DIR"]
    else:
        if local_temps:
            build_dir = str(config_path.parent / "FINN_TMP")
        else:
            build_dir = SETTINGS["FINN_DEFAULT_BUILD_DIR"]

    # Envvar > Default
    if "OHMYXILINX" in os.environ.keys():
        ohmyxilinx = os.environ["OHMYXILINX"]
    else:
        ohmyxilinx = str(Path(deps_value) / "oh-my-xilinx")

    initial = {
        "FINN_ROOT": finn_root,
        "NUM_DEFAULT_WORKERS": str(cpus),
        "OHMYXILINX": ohmyxilinx,
        "FINN_BUILD_DIR": build_dir,
        "FINN_DEPS": deps_value,
    }

    # Envvar > ENVVAR config default > Default
    for k, v in FINN_ENVVARS.items():
        if k in os.environ.keys():
            initial[k] = os.environ[k]
        else:
            initial[k] = v

    return initial


def print_envvars(envs: dict) -> None:
    console = Console()
    table = Table()
    table.add_column("Environment Variable")
    table.add_column("Value")
    for k, v in envs.items():
        table.add_row(f"[cyan]{k}[/cyan]", f"{v}")
    console.print(table)

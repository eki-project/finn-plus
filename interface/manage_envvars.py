"Manage environment variables. Called by run_finn.py"
import os
import psutil
import yaml
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from interface.interface_globals import (
    DEFAULT_FINN_ROOT,
    DEFAULT_FINN_TMP,
    DEFAULT_FINN_TMP_HOST,
    DEFAULT_GLOBAL_ENVVARS,
)


def get_global_envvars(config_path: Path) -> tuple[dict[str, str], bool]:
    """Get a dictionary of environment variables that are globally used.
    Precedence is: Set variables > Config > Default. Also returns if the env
    var config could be read"""
    envvars = {}
    config_vars = {}
    read_config = False
    if config_path.exists() and config_path.suffix in [".yml", ".yaml"]:
        with config_path.open() as f:
            config_vars = dict(yaml.load(f, yaml.Loader).items())
            read_config = True
    for k, v in DEFAULT_GLOBAL_ENVVARS.items():
        if k in os.environ.keys():
            envvars[k] = os.environ[k]
        elif k in config_vars.keys():
            envvars[k] = config_vars[k]
        else:
            envvars[k] = v
    return envvars, read_config


def get_run_specific_envvars(
    deps: Path, config_path: Path, local_temps: bool, num_workers: int
) -> dict[str, str]:
    """Retun run specific environment variables"""
    cpucount = psutil.cpu_count(logical=False)
    if num_workers == -1:
        cpus = int(0.75 * cpucount) if cpucount is not None else 1
    else:
        cpus = num_workers
    return {
        "FINN_ROOT": str(DEFAULT_FINN_ROOT),
        "NUM_DEFAULT_WORKERS": str(cpus),
        "OHMYXILINX": str((deps / "oh-my-xilinx").absolute()),
        "FINN_BUILD_DIR": str((config_path.parent / "FINN_TMP").absolute())
        if local_temps
        else str(DEFAULT_FINN_TMP),
        "FINN_HOST_BUILD_DIR": str((config_path.parent / "FINN_TMP_HOST").absolute())
        if local_temps
        else str(DEFAULT_FINN_TMP_HOST),
        "FINN_DEPS": str(deps),
    }


def print_environment() -> None:
    """Print a panel in console with all FINN relevant environment variables"""
    s = f"[italic]FINN_ROOT:[/italic] [bold cyan]{os.environ['FINN_ROOT']}[/bold cyan]\n"
    s += f"[italic]FINN_BUILD_DIR:[/italic][bold cyan] {os.environ['FINN_BUILD_DIR']}[/bold cyan]\n"
    s += (
        f"[italic]FINN_HOST_BUILD_DIR:[/italic][bold cyan] {os.environ['FINN_HOST_BUILD_DIR']}"
        "[/bold cyan]\n"
    )
    s += f"[italic]FINN_DEPS:[/italic][bold cyan] {os.environ['FINN_DEPS']}[/bold cyan]\n"
    s += (
        f"[italic]NUM_DEFAULT_WORKERS:[/italic][bold cyan] {os.environ['NUM_DEFAULT_WORKERS']}"
        "[/bold cyan]\n"
    )
    s += f"[italic]OHMYXILINX:[/italic][bold cyan] {os.environ['OHMYXILINX']}[/bold cyan]\n"
    s += (
        f"[italic]PLATFORM_REPO_PATHS:[/italic][bold cyan] {os.environ['PLATFORM_REPO_PATHS']}"
        "[/bold cyan]\n"
    )
    s += (
        f"[italic]XRT_DEB_VERSION:[/italic][bold cyan] {os.environ['XRT_DEB_VERSION']}"
        "[/bold cyan]\n"
    )
    s += f"[italic]VIVADO_PATH:[/italic][bold cyan] {os.environ['VIVADO_PATH']}[/bold cyan]\n"
    s += f"[italic]VITIS_PATH:[/italic][bold cyan] {os.environ['VITIS_PATH']}[/bold cyan]\n"
    s += f"[italic]HLS_PATH:[/italic][bold cyan] {os.environ['HLS_PATH']}[/bold cyan]\n"
    s += (
        f"[italic]XILINX_LOCAL_USER_DATA:[/italic][bold cyan] "
        f"{os.environ['XILINX_LOCAL_USER_DATA']}[/bold cyan]\n"
    )
    Console().print(Panel(s, title="Environment"))

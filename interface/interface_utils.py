from __future__ import annotations

import os
import psutil
import sys
import yaml
from pathlib import Path
from rich.console import Console

from interface import DEBUG
from interface.manage_deps import REQUIRED_VERILATOR_VERSION, check_verilator_version


def error(msg: str) -> None:
    """Print an error"""
    Console().print(f"[bold red]ERROR: [/bold red][red]{msg}[/red]")


def warning(msg: str) -> None:
    """Print a warning"""
    Console().print(f"[bold orange1]WARNING: [/bold orange1][orange3]{msg}[/orange3]")


def status(msg: str) -> None:
    """Print a status message"""
    Console().print(f"[bold cyan]STATUS: [/bold cyan][cyan]{msg}[/cyan]")


def success(msg: str) -> None:
    """Print a success message"""
    Console().print(f"[bold green]SUCCESS: [/bold green][green]{msg}[/green]")


def debug(msg: str) -> None:
    """Print a debug message. Only done when the flag is set"""
    if DEBUG:
        Console().print(f"[bold blue]DEBUG: [/bold blue][blue]{msg}[/blue]")


def assert_path_valid(p: Path) -> None:
    """Check if the path exists, if not print an error message and exit with an error code"""
    if not p.exists():
        Console().print(f"[bold red]File or directory {p} does not exist. Stopping...[/bold red]")
        sys.exit(1)


def check_verilator() -> None:
    """Check that verilator exists and has the right version. Stop execution if not"""
    console = Console()
    verilator_version = check_verilator_version()
    if verilator_version is None:
        console.print(
            "[bold red]ERROR: Verilator could not be found or executed properly after "
            "the local installation. Stopping... [/bold red]"
        )
        sys.exit(1)
    elif verilator_version is not None and verilator_version < REQUIRED_VERILATOR_VERSION:
        console.print(
            f"[bold orange1]WARNING: [/bold orange1][orange3]It seems you are using verilator "
            f"version [bold]{verilator_version}[/bold]. "
            f"The recommended version is [bold]{REQUIRED_VERILATOR_VERSION}[/bold]. "
            "FIFO-Sizing or simulations might fail due to verilator errors.[/orange3]"
        )
    else:
        console.print(f"[bold green]Verilator version {verilator_version} found![/bold green]")


def set_synthesis_tools_paths() -> None:
    """Check that all synthesis tools can be found. If not, give a warning. If they are found, set
    the appropiate environment variables"""
    for envname, toolname in [
        ("XILINX_VIVADO", "vivado"),
        ("XILINX_VITIS", "vitis"),
        ("XILINX_HLS", "vitis_hls"),
    ]:
        if envname not in os.environ.keys():
            warning(
                f"Path to the {toolname} tool could not be resolved from {envname}. "
                "Did you source your settings file?"
            )
        p = Path(os.environ[envname]) / "bin" / toolname
        if not p.exists():
            warning(f"Path for {toolname} found, but executable not found in {p}!")
        else:
            os.environ[envname.replace("XILINX_", "") + "_PATH"] = str(p)


def resolve_build_dir(flow_config: Path, build_dir: Path | None, settings: dict) -> Path | None:
    """Resolve the build dir.
    Priority is command line argument > Environment variable > Settings Default > Fixed default"""
    if build_dir is not None:
        return build_dir
    if "FINN_BUILD_DIR" in os.environ.keys():
        return Path(os.environ["FINN_BUILD_DIR"])
    if "FINN_BUILD_DIR" in settings.keys():
        return Path(settings["FINN_BUILD_DIR"])
    return flow_config.parent / "FINN_TMP"


def resolve_deps_path(deps: Path | None, settings: dict) -> Path | None:
    """Try to resolve the dependency path. If none is valid, return None
    Priority is command line argument > Environment variable > Settings Default > Fixed default"""
    if deps is not None:
        return deps
    if "FINN_DEPS" in os.environ.keys():
        return Path(os.environ["FINN_DEPS"])
    if "FINN_DEPS" in settings.keys():
        return Path(settings["FINN_DEPS"])
    return None


def resolve_num_workers(num: int, settings: dict) -> int:
    """Resolve the number of workers to use. Uses 75% of cores available as default fallback"""
    if num > -1:
        return num
    if "NUM_DEFAULT_WORKERS" in os.environ.keys() and os.environ["NUM_DEFAULT_WORKERS"] != "":
        return int(os.environ["NUM_DEFAULT_WORKERS"])
    if "NUM_DEFAULT_WORKERS" in settings.keys():
        return int(settings["NUM_DEFAULT_WORKERS"])
    cpus = psutil.cpu_count()
    if cpus is None or cpus == 1:
        return 1
    return int(cpus * 0.75)


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

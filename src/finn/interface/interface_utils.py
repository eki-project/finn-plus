from __future__ import annotations

import os
import psutil
import sys
import yaml
from pathlib import Path
from rich.console import Console

from finn.interface import DEBUG


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


def set_synthesis_tools_paths() -> None:
    """Check that all synthesis tools can be found. If not, give a warning."""
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
            continue
        envname_path = os.environ[envname]

        # Exception for Vitis HLS because of changed behavior starting with 2024.2
        # XILINX_HLS no longer points to */Vitis_HLS/VERSION but */Vitis/VERSION
        p = Path(envname_path) / "bin" / toolname
        if not p.exists() and toolname == "vitis_hls":
            envname_path = envname_path.replace("Vitis", "Vitis_HLS")
            p = Path(envname_path) / "bin" / toolname

        if not p.exists():
            warning(f"Path for {toolname} found, but executable not found in {p}!")
        # TODO: simply check "which" instead?

    if (
        "PLATFORM_REPO_PATHS" not in os.environ.keys()
        or not Path(os.environ["PLATFORM_REPO_PATHS"]).exists()
    ):
        p = Path("/opt/xilinx/platforms")
        if p.exists():
            os.environ["PLATFORM_REPO_PATHS"] = str(p.absolute())
        else:
            warning(
                "PLATFORM_REPO_PATHS is not set "
                "and the default path does not exist. Synthesis might fail."
            )


def resolve_build_dir(
    flow_config: Path, build_dir: Path | None, settings: dict, is_test_run: bool = False
) -> Path | None:
    """Resolve the build dir. By default this should return FINN_TMP next to the flow config.
    Priority is command line argument > Environment variable > Settings Default > Fixed default.
    If the given path is relative and found in the envvar or settings file,
    add it to the flow_config_path"""
    if build_dir is not None:
        return build_dir
    if "FINN_BUILD_DIR" in os.environ.keys():
        p = Path(os.environ["FINN_BUILD_DIR"])
        if not p.is_absolute():
            return flow_config.parent / p
        return p
    if "FINN_BUILD_DIR" in settings.keys():
        p = Path(settings["FINN_BUILD_DIR"])
        if not p.is_absolute():
            return flow_config.parent / p
        return p
    if is_test_run:
        # Need a different fallback because tests have no build config
        return Path("/tmp/FINN_TEST_BUILD_DIR")
    return flow_config.parent / "FINN_TMP"


def resolve_deps_path(deps: Path | None, settings: dict) -> Path | None:
    """Try to resolve the dependency path. If none is valid, return None. If the path is relative,
    and not given by command line, append it to the FINN directory.
    Priority is command line argument > Environment variable > Settings Default > Fixed default"""
    if deps is not None:
        return deps
    if "FINN_DEPS" in os.environ.keys():
        p = Path(os.environ["FINN_DEPS"])
        if not p.is_absolute():
            return Path(__file__).parent.parent / p
        return p
    if "FINN_DEPS" in settings.keys():
        p = Path(settings["FINN_DEPS"])
        if not p.is_absolute():
            return Path(__file__).parent.parent / p
        return p
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

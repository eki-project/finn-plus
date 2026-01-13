"""Utilities for the FINN command line interface."""

from __future__ import annotations

import click
import importlib
import os
import rich
import rich.box
import rich.table
import sys
from pathlib import Path
from rich.console import Console
from typing import Any

from finn.interface import DEBUG


def resolve_module_path(name: str) -> str:
    """Resolve the path to modules which are not part of the FINN package hierarchy."""
    # Try to import the module via importlib - allows "-" in names and resolve
    # the absolute path to the first candidate location as a string
    try:
        return str(importlib.import_module(name).__path__[0])
    except ModuleNotFoundError:
        # Try a different location if notebooks have not been found, maybe we
        # are in the Git repository root and should look there as well...
        try:
            return str(importlib.import_module(f"finn.{name}").__path__[0])
        except ModuleNotFoundError:
            warning(f"Could not resolve {name}. FINN might not work properly.")
    # Return the empty string as a default...
    return ""


class NullablePath(click.ParamType):
    """If the passed parameter is an empty string return None, otherwise a Path."""

    name = "NullablePath"

    def __init__(self, expand_user: bool = True) -> None:
        """Initialize a NullablePath."""
        super().__init__()
        self.expand_user = expand_user

    def convert(self, value: str, param: Any, ctx: Any) -> Path | None:  # noqa
        """Convert a string value into a Path or None, depending on the contents."""
        if value == "":
            return None
        p = Path(value)
        if self.expand_user:
            return p.expanduser()
        return p


def error(msg: str) -> None:
    """Print an error."""
    Console().print(f"[bold red]ERROR: [/bold red][red]{msg}[/red]")


def warning(msg: str, critical: bool = False) -> None:
    """Print a warning."""
    color = "bold orange1" if not critical else "bold orange_red1"
    prefix = "" if not critical else "CRITICAL "
    Console().print(f"[{color}]{prefix}WARNING: [/{color}][orange3]{msg}[/orange3]")


def status(msg: str) -> None:
    """Print a status message."""
    Console().print(f"[bold cyan]STATUS: [/bold cyan][cyan]{msg}[/cyan]")


def success(msg: str) -> None:
    """Print a success message."""
    Console().print(f"[bold green]SUCCESS: [/bold green][green]{msg}[/green]")


def debug(msg: str, with_rich: bool = True) -> None:
    """Print a debug message. Only done when the flag is set."""
    if DEBUG:
        if with_rich:
            Console().print(f"[bold blue]DEBUG: [/bold blue][blue]{msg}[/blue]")
        else:
            print(f"DEBUG: {msg}")


def table(data: dict[Any, Any], key_header: str, value_header: str) -> None:
    """Print the data as a table."""
    table = rich.table.Table(key_header, value_header, box=rich.box.SIMPLE)
    for k, v in data.items():
        table.add_row(str(k), str(v))
    Console().print(table)


def assert_path_valid(p: Path) -> None:
    """Check if the path exists, if not print an error message and exit with an error code."""
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

        # Append to the search path just in case it was missing before, if
        # already in there adding again should do nothing
        os.environ["PATH"] += os.pathsep + str((Path(envname_path) / "bin").absolute())

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

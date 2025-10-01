"""Utility functions for the FINN+ interface."""
from __future__ import annotations

import click
import os
import sys
from pathlib import Path
from rich.console import Console
from typing import Any

from finn.interface import DEBUG


class NullablePath(click.ParamType):
    """If the passed parameter is an empty string return None, otherwise a Path."""

    name = "NullablePath"

    def __init__(self, expand_user: bool = True) -> None:  # noqa
        super().__init__()
        self.expand_user = expand_user

    def convert(self, value: str, param: Any, ctx: Any) -> Path | None:  # noqa
        if value == "":
            return None
        p = Path(value)
        if self.expand_user:
            return p.expanduser()
        return p


def error(msg: str) -> None:
    """Print an error."""
    Console().print(f"[bold red]ERROR: [/bold red][red]\t{msg}[/red]")


def warning(msg: str) -> None:
    """Print a warning."""
    Console().print(f"[bold orange1]WARNING: [/bold orange1][orange3]\t{msg}[/orange3]")


def status(msg: str) -> None:
    """Print a status message."""
    Console().print(f"[bold cyan]STATUS: [/bold cyan][cyan]\t{msg}[/cyan]")


def success(msg: str) -> None:
    """Print a success message."""
    Console().print(f"[bold green]SUCCESS: [/bold green][green]\t{msg}[/green]")


def debug(msg: str) -> None:
    """Print a debug message. Only done when the flag is set."""
    if DEBUG:
        Console().print(f"[bold blue]DEBUG: [/bold blue][blue]\t{msg}[/blue]")


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

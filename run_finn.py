from __future__ import annotations

import click
import os
import shlex
import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table

from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig
from interface.interface_globals import (
    IS_POSIX,
    _resolve_settings_path,
    get_settings,
    settings_found,
)
from interface.interface_utils import (
    assert_path_valid,
    check_verilator,
    error,
    resolve_build_dir,
    resolve_deps_path,
    resolve_num_workers,
    set_synthesis_tools_paths,
    status,
    warning,
)
from interface.manage_deps import try_install_verilator, update_dependencies
from interface.manage_tests import run_test


def update_all_deps(path: Path) -> None:
    """Update dependencies and notify the user of the results. If updating fails,
    this ends the execution"""
    console = Console()
    with console.status("[bold cyan]Gathering dependencies...[/bold cyan]") as _:
        update_status = update_dependencies(path)
    with console.status("[bold cyan]Installing verilator...[/bold cyan]") as _:
        vname, vsuc, vmsg = try_install_verilator(path)
    table = Table()
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Message")
    any_failed = False
    for pkg_name, suc, msg in update_status:
        if not suc:
            any_failed = True
        table.add_row(
            pkg_name,
            "[bold green]Success[/bold green]" if suc else "[bold red]Failed[/bold red]",
            msg,
        )
    table.add_row(
        vname, "[bold green]Success[/bold green]" if vsuc else "[bold red]Failed[/bold red]", vmsg
    )
    console.print(table)
    if any_failed:
        error("Dependency update failed. Stopping.")
        sys.exit(1)


def prepare_finn(
    deps: Path | None, flow_config: Path, build_dir: Path | None, num_workers: int
) -> None:
    """Prepare a FINN environment by:
    0. Reading all settings and environment vars
    1. Updating all dependencies
    2. Setting all environment vars
    """
    # Resolve settings and dependencies, error if this doesnt work
    if not settings_found():
        warning("Settings file could not be found. Using defaults.")
    else:
        sp = _resolve_settings_path()
        status(f"Using settings file at {sp}")
    settings = get_settings(force_update=True)
    deps_path = resolve_deps_path(deps, settings)
    if deps_path is None:
        error("Dependency location could not be resolved!")
        sys.exit(1)
    else:
        status(f"Using dependency path: {deps_path}")
    os.environ["FINN_DEPS"] = str(deps_path)

    # Update / Install all dependencies
    update_all_deps(deps_path)
    check_verilator()

    # Check synthesis tools
    set_synthesis_tools_paths()

    # Add OHMYXILINX?
    os.environ["OHMYXILINX"] = str(deps_path / "oh-my-xilinx")
    os.environ["PATH"] = os.environ["PATH"] + ":" + os.environ["OHMYXILINX"]

    # Resolve the build directory
    resolved_build_dir = resolve_build_dir(flow_config, build_dir, settings)
    if resolved_build_dir is None:
        error("Could not resolve the build directory!")
        sys.exit(1)
    os.environ["FINN_BUILD_DIR"] = str(resolved_build_dir)

    # Resolve number of workers
    workers = resolve_num_workers(num_workers, settings)
    status(f"Using {workers} workers.")
    os.environ["NUM_DEFAULT_WORKERS"] = str(workers)


@click.group()
def main_group() -> None:
    pass


@click.command(help="Build a hardware design")
@click.option("--dependency-path", "-d", default="")
@click.option("--build-path", "-b", help="Specify a build temp path of your choice", default="")
@click.option(
    "--num-workers",
    "-n",
    help="Number of parallel workers for FINN to use. When -1, automatically use 75% of cores",
    default=-1,
    show_default=True,
)
@click.argument("config")
@click.argument("model")
def build(dependency_path: str, build_path: str, num_workers: int, config: str, model: str) -> None:
    config_path = Path(config)
    model_path = Path(model)
    build_dir = Path(build_path) if build_path != "" else None
    assert_path_valid(config_path)
    assert_path_valid(model_path)
    dep_path = Path(dependency_path) if dependency_path != "" else None
    status(f"Starting FINN build with config {config_path.name} and model {model_path.name}!")
    prepare_finn(dep_path, config_path, build_dir, num_workers)
    status("Creating dataflow build config...")
    dfbc: DataflowBuildConfig | None = None
    match config_path.suffix:
        case ".yaml" | ".yml":
            raise NotImplementedError("Depends on a pending PR for YAML support.")
        case ".json":
            with config_path.open() as f:
                dfbc = DataflowBuildConfig.from_json(f.read())
        case _:
            error(
                f"Unknown config file type: {config_path.name}. "
                "Valid formats are: .json, .yml, .yaml"
            )
            sys.exit(1)
    if dfbc is None:
        error("Failed to generate dataflow build config!")
        sys.exit(1)
    Console().rule(
        f"[bold cyan]Running FINN with config[/bold cyan][bold orange1] "
        f"{config_path.name}[/bold orange1][bold cyan] on model [/bold cyan]"
        "[bold orange1]{model_path.name}[/bold orange1]"
    )
    build_dataflow_cfg(str(model_path), dfbc)


@click.command(help="Run a script in a FINN environment")
@click.option("--dependency-path", "-d", default="")
@click.option("--build-path", "-b", help="Specify a build temp path of your choice", default="")
@click.option(
    "--num-workers",
    "-n",
    help="Number of parallel workers for FINN to use. When -1, automatically use 75% of cores",
    default=-1,
    show_default=True,
)
@click.argument("script")
def run(dependency_path: str, build_path: str, num_workers: int, script: str) -> None:
    script_path = Path(script)
    build_dir = Path(build_path) if build_path != "" else None
    assert_path_valid(script_path)
    dep_path = Path(dependency_path) if dependency_path != "" else None
    prepare_finn(dep_path, script_path, build_dir, num_workers)
    Console().rule(
        f"[bold cyan]Starting script "
        f"[/bold cyan][bold orange1]{script_path.name}[/bold orange1]"
    )
    subprocess.run(
        shlex.split(f"python3 {script_path.name}", posix=IS_POSIX), cwd=script_path.parent
    )


@click.command(help="Run a given test. Uses /tmp/FINN_TMP as the temporary file location")
@click.option(
    "--variant",
    "-v",
    help="Which test to execute (quick, main, rtlsim, end2end, full)",
    default="quick",
    show_default=True,
)
@click.option("--dependency-path", "-d", default="")
@click.option("--num-workers", "-n", default=-1, show_default=True)
@click.option("--num-test-workers", "-t", default="auto", show_default=True)
@click.option("--build-path", "-b", help="Specify a build temp path of your choice", default="")
def test(
    variant: str, dependency_path: str, num_workers: int, num_test_workers: str, build_path: str
) -> None:
    console = Console()
    build_dir = Path(build_path) if build_path != "" else None
    dep_path = Path(dependency_path) if dependency_path != "" else None
    prepare_finn(dep_path, Path(), build_dir, num_workers)
    console.rule("RUNNING TESTS")
    run_test(variant, num_test_workers)


@click.group(help="Dependency management")
def deps() -> None:
    pass


@click.command(help="Update or install dependencies to the given path")
@click.option(
    "--path",
    "-p",
    help="Path to install to",
    default="",
    show_default=True,
)
def update(path: str) -> None:
    dep_path = Path(path) if path != "" else None
    prepare_finn(dep_path, Path(), None, 1)


@click.group(help="Manage FINN settings")
def config() -> None:
    # TODO: Config remove?
    pass


def main() -> None:
    # config.add_command(config_list)
    # config.add_command(config_get)
    # config.add_command(config_set)
    deps.add_command(update)
    main_group.add_command(config)
    main_group.add_command(deps)
    main_group.add_command(build)
    main_group.add_command(test)
    main_group.add_command(run)
    main_group()


if __name__ == "__main__":
    main()

from __future__ import annotations

import click
import os
import shlex
import subprocess
import sys
from pathlib import Path
from rich.console import Console

from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig
from interface import IS_POSIX
from interface.interface_globals import (
    _resolve_settings_path,
    get_settings,
    set_settings,
    settings_found,
    skip_update_by_default,
    write_settings,
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
    write_yaml,
)
from interface.manage_deps import update_dependencies
from interface.manage_tests import run_test


def prepare_finn(
    deps: Path | None,
    flow_config: Path,
    build_dir: Path | None,
    num_workers: int,
    is_test_run: bool = False,
    skip_dep_update: bool = False,
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
    os.environ["FINN_DEPS"] = str(deps_path.absolute())

    # Update / Install all dependencies
    if not skip_dep_update:
        update_dependencies(deps_path)
    else:
        warning("Skipping dependency updates!")
    check_verilator()

    # Check synthesis tools
    set_synthesis_tools_paths()

    # Add OHMYXILINX?
    os.environ["OHMYXILINX"] = str((deps_path / "oh-my-xilinx").absolute())
    os.environ["PATH"] = os.environ["PATH"] + ":" + os.environ["OHMYXILINX"]

    # Resolve the build directory
    resolved_build_dir = resolve_build_dir(
        flow_config, build_dir, settings, is_test_run=is_test_run
    )
    if resolved_build_dir is None:
        error("Could not resolve the build directory!")
        sys.exit(1)
    os.environ["FINN_BUILD_DIR"] = str(resolved_build_dir.absolute())
    if not resolved_build_dir.exists():
        resolved_build_dir.mkdir(parents=True)
    status(f"Build directory set to: {resolved_build_dir}")

    # Resolve number of workers
    workers = resolve_num_workers(num_workers, settings)
    status(f"Using {workers} workers.")
    os.environ["NUM_DEFAULT_WORKERS"] = str(workers)

    # Set FINN_ROOT
    os.environ["FINN_ROOT"] = str(Path(__file__).parent.absolute())
    status(f"FINN_ROOT set to {Path(__file__).parent}")


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
@click.option(
    "--skip-dep-update",
    "-s",
    default=skip_update_by_default(),
    is_flag=True,
    help="Whether to skip the dependency update. Can be changed in settings via"
    "AUTOMATIC_DEPENDENCY_UPDATES: false",
)
@click.argument("config")
@click.argument("model")
def build(
    dependency_path: str,
    build_path: str,
    num_workers: int,
    skip_dep_update: bool,
    config: str,
    model: str,
) -> None:
    config_path = Path(config).expanduser()
    model_path = Path(model).expanduser()
    build_dir = Path(build_path).expanduser() if build_path != "" else None
    assert_path_valid(config_path)
    assert_path_valid(model_path)
    dep_path = Path(dependency_path).expanduser() if dependency_path != "" else None
    status(f"Starting FINN build with config {config_path.name} and model {model_path.name}!")
    prepare_finn(dep_path, config_path, build_dir, num_workers, skip_dep_update=skip_dep_update)
    status("Creating dataflow build config...")
    dfbc: DataflowBuildConfig | None = None
    match config_path.suffix:
        case ".yaml" | ".yml":
            with config_path.open() as f:
                dfbc = DataflowBuildConfig.from_yaml(f.read())
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
        f"[bold orange1]{model_path.name}[/bold orange1]"
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
    script_path = Path(script).expanduser()
    build_dir = Path(build_path).expanduser() if build_path != "" else None
    assert_path_valid(script_path)
    dep_path = Path(dependency_path).expanduser() if dependency_path != "" else None
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
    help="Which test to execute (quick, quicktest_ci, full_ci)",
    default="quick",
    show_default=True,
)
@click.option("--dependency-path", "-d", default="")
@click.option("--num-workers", "-n", default=-1, show_default=True)
@click.option("--num-test-workers", "-t", default="auto", show_default=True)
@click.option(
    "--build-path",
    "-b",
    help="Specify a build temp path of your choice",
    default="",
)
def test(
    variant: str, dependency_path: str, num_workers: int, num_test_workers: str, build_path: str
) -> None:
    console = Console()
    build_dir = Path(build_path).expanduser() if build_path != "" else None
    dep_path = Path(dependency_path).expanduser() if dependency_path != "" else None
    prepare_finn(dep_path, Path(), build_dir, num_workers, is_test_run=True)
    status(f"Using {num_test_workers} test workers")
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
    dep_path = Path(path).expanduser() if path != "" else None
    prepare_finn(dep_path, Path(), None, 1)


@click.group(help="Manage FINN settings")
def config() -> None:
    # TODO: Config remove?
    pass


def _command_get_settings():
    sp = _resolve_settings_path()
    if sp is None:
        error("Could not find a settings file. Stopping")
        sys.exit(1)
    status(f"Found settings at {sp}")
    return get_settings(force_update=True)


@click.command("list", help="List the settings files contents")
def config_list() -> None:
    console = Console()
    for k, v in _command_get_settings().items():
        console.print(f"[blue]{k}[/blue]: {v}")


@click.command("get", help="Get a specific key from the settings")
@click.argument("key")
def config_get(key: str) -> None:
    settings = _command_get_settings()
    if key not in settings.keys():
        error(f"Key {key} could not be found in the settings file!")
        sys.exit(1)
    Console().print(f"[blue]{key}[/blue]: {settings[key]}")


@click.command("set", help="Set a key to a given value")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    if not settings_found():
        error(
            "Settings file not found. To create a template one at the "
            'default location run "finn config create"'
        )
        sys.exit(1)
    else:
        settings = get_settings(force_update=True)
        settings[key] = value
    set_settings(settings)
    write_settings()


@click.command(
    "create",
    help="Create a template settings file. If one exists at the given path, "
    "its overwritten. Please enter a directory, no filename",
)
@click.argument("path", default="~/.finn/")
def config_create(path: str) -> None:
    p = Path(path).expanduser()
    if p.suffix != "":
        error("Please specify a path to a directory, not a file!")
        sys.exit(1)
    complete_path = p / "settings.yaml"
    settings = get_settings(force_update=False)
    msettings = {key: str(settings[key]) for key in settings.keys()}
    if not write_yaml(msettings, complete_path):
        error(f"Writing to {complete_path} failed!")
        sys.exit(1)
    status(f"File written to {complete_path}")


def main() -> None:
    config.add_command(config_list)
    config.add_command(config_create)
    config.add_command(config_get)
    config.add_command(config_set)
    deps.add_command(update)
    main_group.add_command(config)
    main_group.add_command(deps)
    main_group.add_command(build)
    main_group.add_command(test)
    main_group.add_command(run)
    main_group()


if __name__ == "__main__":
    main()

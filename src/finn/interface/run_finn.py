"""Run FINN+."""
# ruff: noqa: D103
from __future__ import annotations

import click
import importlib
import os
import shlex
import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.pretty import pprint
from typing import Any, Callable

from finn.interface import IS_POSIX
from finn.interface.interface_utils import (
    NullablePath,
    error,
    set_synthesis_tools_paths,
    status,
    warning,
)
from finn.interface.manage_deps import install_pyxsi, update_dependencies
from finn.interface.manage_tests import run_test
from finn.interface.settings import FINNSettings


# Resolves the path to modules which are not part of the FINN package hierarchy
def _resolve_module_path(name: str) -> str:
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


def requires_dependencies(f: Callable) -> Callable[..., Any]:
    """Add a click parameter named --dependency-path (-d) that defaults to
    None if the param is empty, and a path otherwise."""  # noqa
    return click.option("--dependency-path", "-d", default="", type=NullablePath())(f)


def requires_builddir(f: Callable) -> Callable[..., Any]:
    """Add a click parameter named --build-path (-b) that defaults to
    None if the param is empty, and a path otherwise."""  # noqa
    return click.option(
        "--build-path",
        "-b",
        help="Specify a build temp path of your choice",
        default="",
        type=NullablePath(),
    )(f)


def requires_config(f: Callable) -> Callable[..., Any]:
    """Add a click parameter named config (type pathlib.Path)."""
    return click.argument(
        "config",
        type=click.Path(
            exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
        ),
    )(f)


def requires_model(f: Callable) -> Callable[..., Any]:
    """Add a click parameter named model (type pathlib.Path)."""
    return click.argument(
        "model",
        type=click.Path(
            exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
        ),
    )(f)


def with_num_workers_option(f: Callable) -> Callable[..., Any]:
    """Add a click parameter called --num-workers (-n). Defaults to -1."""
    return click.option(
        "--num-workers",
        "-n",
        help="Number of parallel workers for FINN to use. When -1, automatically use 75% of cores",
        default=-1,
        show_default=True,
    )(f)


def with_skip_depenency_update_option(f: Callable) -> Callable[..., Any]:
    """Add a click parameter called --skip-dep-update (-s). Defaults to False."""
    return click.option(
        "--skip-dep-update",
        "-s",
        is_flag=True,
        help="Whether to skip the dependency update. Can be changed in settings via"
        "AUTOMATIC_DEPENDENCY_UPDATES: false",
    )(f)


def prepare_finn(
    settings: FINNSettings,
    deps: Path | None,
    flow_config: Path,
    build_dir: Path | None,
    num_workers: int,
    is_test_run: bool = False,
    cmdparam_skip_dep_update: bool = False,
) -> FINNSettings:
    """Prepare a FINN environment with the given settings. The settings will be adapted
    and modified with the necessary runtime data and returned."""  # noqa
    status(f"Using settings file at {settings.get_path()}")

    # Dependencies
    deps_path = settings.resolve_deps_path(deps)
    if deps_path is None:
        error("Dependency location could not be resolved!")
        sys.exit(1)
    else:
        status(f"Using dependency path: {deps_path}")
    settings["FINN_DEPS"] = deps_path.absolute()
    os.environ["FINN_DEPS"] = str(settings["FINN_DEPS"])

    # Skipping dependency updates
    settings["AUTOMATIC_DEPENDENCY_UPDATES"] = not settings.resolve_skip_update(
        cmdparam_skip_dep_update
    )

    # Pythonpath
    if "PYTHONPATH" not in os.environ.keys():
        os.environ["PYTHONPATH"] = ""

    # Update / Install all dependencies
    if settings["AUTOMATIC_DEPENDENCY_UPDATES"]:
        update_dependencies(deps_path)
    else:
        warning("Skipping dependency updates!")

    # Check synthesis tools
    set_synthesis_tools_paths()

    # Install pyXSI
    pyxsi_status = install_pyxsi()
    if pyxsi_status:
        status("pyXSI installed successfully.")
    else:
        error("pyXSI installation failed.")
        sys.exit(1)

    # Resolve the build directory
    resolved_build_dir = settings.resolve_build_dir(build_dir, flow_config, is_test_run).absolute()
    settings["FINN_BUILD_DIR"] = resolved_build_dir
    os.environ["FINN_BUILD_DIR"] = str(resolved_build_dir)
    if not resolved_build_dir.exists():
        resolved_build_dir.mkdir(parents=True)
    status(f"Build directory set to: {resolved_build_dir}")

    # Resolve the number of workers
    workers = settings.resolve_num_workers(num_workers)
    status(f"Using {workers} workers.")
    settings["NUM_DEFAULT_WORKERS"] = workers
    os.environ["NUM_DEFAULT_WORKERS"] = str(workers)

    # Resolve paths to some not properly packaged components...
    os.environ["FINN_RTLLIB"] = _resolve_module_path("finn-rtllib")
    os.environ["FINN_CUSTOM_HLS"] = _resolve_module_path("custom_hls")
    os.environ["FINN_QNN_DATA"] = _resolve_module_path("qnn-data")
    os.environ["FINN_NOTEBOOKS"] = _resolve_module_path("notebooks")
    os.environ["FINN_TESTS"] = _resolve_module_path("tests")
    return settings


@click.group()
def main_group() -> None:
    pass


@click.command(help="Build a hardware design")
@requires_dependencies
@requires_builddir
@requires_config
@requires_model
@with_num_workers_option
@with_skip_depenency_update_option
@click.option(
    "--start",
    default="",
    help="If no start_step is given in the dataflow build config, "
    "this starts the flow from the given step.",
)
@click.option(
    "--stop",
    default="",
    help="If no stop_step is given in the dataflow build config, "
    "this stops the flow at the given step.",
)
def build(
    dependency_path: str | None,
    build_path: str | None,
    num_workers: int,
    skip_dep_update: bool,
    start: str,
    stop: str,
    config: Path,
    model: Path,
) -> None:
    status(f"Starting FINN build with config {config.name} and model {model.name}!")
    prepare_finn(
        FINNSettings(),
        dependency_path,
        config.expanduser(),
        build_path,
        num_workers,
        cmdparam_skip_dep_update=skip_dep_update,
    )

    # Can import from finn now, since all deps are installed
    # and all environment variables are set correctly
    from finn.builder.build_dataflow import build_dataflow_cfg
    from finn.builder.build_dataflow_config import DataflowBuildConfig

    status("Creating dataflow build config...")
    dfbc: DataflowBuildConfig | None = None
    match config.suffix:
        case ".yaml" | ".yml":
            with config.open() as f:
                dfbc = DataflowBuildConfig.from_yaml(f.read())
        case ".json":
            with config.open() as f:
                dfbc = DataflowBuildConfig.from_json(f.read())
        case _:
            error(
                f"Unknown config file type: {config.name}. " "Valid formats are: .json, .yml, .yaml"
            )
            sys.exit(1)
    if dfbc is None:
        error("Failed to generate dataflow build config!")
        sys.exit(1)

    # Set start and stop steps
    if dfbc.start_step is None and start != "":
        dfbc.start_step = start
    if dfbc.stop_step is None and stop != "":
        dfbc.stop_step = stop

    # Set output directory to where the config lies, not where FINN lies
    if not Path(dfbc.output_dir).is_absolute():
        dfbc.output_dir = str((config.parent / dfbc.output_dir).absolute())
    status(f"Output directory is {dfbc.output_dir}")

    # Add path of config to sys.path so that custom steps can be found
    sys.path.append(str(config.parent.absolute()))

    Console().rule(
        f"[bold cyan]Running FINN with config[/bold cyan][bold orange1] "
        f"{config.name}[/bold orange1][bold cyan] on model [/bold cyan]"
        f"[bold orange1]{model.name}[/bold orange1]"
    )
    build_dataflow_cfg(str(model), dfbc)


@click.command(help="Run a script in a FINN environment")
@requires_dependencies
@requires_builddir
@with_num_workers_option
@with_skip_depenency_update_option
@click.argument(
    "script",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, executable=True, path_type=Path),
)
def run(
    dependency_path: Path | None,
    build_path: Path | None,
    skip_dep_update: bool,
    num_workers: int,
    script: Path,
) -> None:
    script = script.expanduser()
    prepare_finn(
        FINNSettings(),
        dependency_path,
        script,
        build_path,
        num_workers,
        cmdparam_skip_dep_update=skip_dep_update,
    )
    Console().rule(
        f"[bold cyan]Starting script " f"[/bold cyan][bold orange1]{script.name}[/bold orange1]"
    )
    subprocess.run(
        shlex.split(f"{sys.executable} {script.name}", posix=IS_POSIX), cwd=script.parent
    )


@click.command(help="Run a given benchmark configuration.")
@click.option("--bench_config", help="Name or path of experiment configuration file", required=True)
@requires_dependencies
@requires_builddir
@with_num_workers_option
def bench(bench_config: str, dependency_path: str, num_workers: int, build_path: str) -> None:
    console = Console()
    prepare_finn(FINNSettings(), dependency_path, Path(), build_path, num_workers)
    console.rule("RUNNING BENCHMARK")

    # Late import because we need prepare_finn to setup remaining dependencies first
    from finn.benchmarking.bench import start_bench_run

    exit_code = start_bench_run(bench_config)
    sys.exit(exit_code)


@click.command(help="Run a given test. Uses /tmp/FINN_TMP as the temporary file location")
@requires_dependencies
@requires_builddir
@with_num_workers_option
@click.option(
    "--variant",
    "-v",
    help="Which test to execute (quick, quicktest_ci, full_ci)",
    default="quick",
    show_default=True,
    type=click.Choice(["quick", "quicktest_ci", "full_ci"]),
)
@click.option("--num-test-workers", "-t", default="auto", show_default=True)
def test(
    variant: str,
    dependency_path: Path | None,
    num_workers: int,
    num_test_workers: str,
    build_path: Path | None,
) -> None:
    console = Console()
    prepare_finn(FINNSettings(), dependency_path, Path(), build_path, num_workers, is_test_run=True)
    status(f"Using {num_test_workers} test workers")
    console.rule("RUNNING TESTS")
    run_test(variant, num_test_workers)


@click.group(help="Dependency management")
def deps() -> None:
    pass


@click.command(help="Update or install dependencies to the given path")
@click.option(
    "--path", "-p", help="Path to install to", default="", show_default=True, type=NullablePath()
)
def update(path: str) -> None:
    prepare_finn(FINNSettings(), path, Path(), None, 1)


@click.group(help="Manage FINN settings")
def config() -> None:
    # TODO: Config remove?
    pass


def _command_get_settings() -> FINNSettings:
    settings = FINNSettings()
    if not settings.get_path().exists():
        warning("Could not resolve settings file.")
        sys.exit(1)
    return settings


@click.command("list", help="List the settings files contents")
def config_list() -> None:
    pprint(_command_get_settings()._settings)  # noqa


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
    settings = FINNSettings(sync=True)
    settings[key] = value


@click.command(
    "create",
    help="Create a template settings file. If one exists at the given path, "
    "its overwritten. Please enter a directory, no filename",
)
@click.argument("path", default="~/.finn/")
def config_create(path: str) -> None:
    p = Path(path).expanduser()
    if not p.exists():
        error("The given path does not seem to exist!")
        sys.exit(1)
    if not p.is_file():
        error("Please specify a path to a directory, not a file!")
        sys.exit(1)
    p = p / "settings.yaml"
    settings = FINNSettings(sync=True, override_path=p)

    # This should return good defaults for a template
    settings = prepare_finn(
        settings=settings,
        deps=None,
        flow_config=Path(),
        build_dir=Path("FINN_TMP"),
        num_workers=1,
        is_test_run=False,
        cmdparam_skip_dep_update=False,
    )
    if not p.exists():
        error(f"Failed to create config template at {p}!")
        sys.exit(1)
    status(f"File written to {p}")


def main() -> None:
    config.add_command(config_list)
    config.add_command(config_create)
    config.add_command(config_get)
    config.add_command(config_set)
    deps.add_command(update)
    main_group.add_command(config)
    main_group.add_command(deps)
    main_group.add_command(build)
    main_group.add_command(bench)
    main_group.add_command(test)
    main_group.add_command(run)
    main_group()


if __name__ == "__main__":
    main()

"""Run FINN+."""
# ruff: noqa: PIE790, ARG001
from __future__ import annotations

import click
import inspect
import os
import rich
import shlex
import subprocess
import sys
from pathlib import Path
from rich.console import Console
from typing import TYPE_CHECKING, Any

from finn.interface import IS_POSIX
from finn.interface.interface_utils import (
    NullablePath,
    _resolve_module_path,
    error,
    set_synthesis_tools_paths,
    status,
    warning,
)
from finn.interface.manage_deps import DependencyUpdater
from finn.interface.manage_tests import run_test
from finn.interface.settings import FINNSettings
from finn.util.exception import FINNValidationError

if TYPE_CHECKING:
    from collections.abc import Callable


def finn_deps(f: Callable) -> Callable[..., Any]:
    """Add a click parameter named --dependency-path (-d) (finn_deps) that defaults to
    None if the param is empty, and a path otherwise."""  # noqa
    return click.option("--dependency-path", "-d", "finn_deps", default="", type=NullablePath())(f)


def finn_deps_definitions(f: Callable) -> Callable[..., Any]:
    """Add a click parameter named --dependency-definitions (-D) (finn_deps_definitions) that defaults to
    None if the param is empty, and a path otherwise."""  # noqa
    return click.option(
        "--dependency-definitions", "-D", "finn_deps_definitions", default="", type=NullablePath()
    )(f)


def finn_build_dir(f: Callable) -> Callable[..., Any]:
    """Add a click parameter named --build-path (-b) (finn_build_dir) that defaults to
    None if the param is empty, and a path otherwise."""  # noqa
    return click.option(
        "--build-path",
        "-b",
        "finn_build_dir",
        help="Specify a build temp path of your choice",
        default="",
        type=NullablePath(),
    )(f)


def flow_config(f: Callable) -> Callable[..., Any]:
    """Add a click parameter named config (type pathlib.Path)."""
    return click.argument(
        "flow_config",
        type=click.Path(
            exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
        ),
    )(f)


def model(f: Callable) -> Callable[..., Any]:
    """Add a click parameter named model (type pathlib.Path)."""
    return click.argument(
        "model",
        type=click.Path(
            exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
        ),
    )(f)


def num_default_workers(f: Callable) -> Callable[..., Any]:
    """Add a click parameter called --num-workers (-n) (num_default_workers). Defaults to -1."""
    return click.option(
        "--num-workers",
        "-n",
        "num_default_workers",
        help="Number of parallel workers for FINN to use. When -1, automatically use 75% of cores",
        default=-1,
        show_default=True,
    )(f)


def skip_dep_update(f: Callable) -> Callable[..., Any]:
    """Add a click parameter called --skip-dep-update (-s). Defaults to False."""
    return click.option(
        "--skip-dep-update",
        "-s",
        is_flag=True,
        help="Whether to skip the dependency update. Can be changed in settings via"
        "AUTOMATIC_DEPENDENCY_UPDATES: false",
    )(f)


def prepare_finn(settings: FINNSettings) -> None:
    """Prepare FINN to run."""
    status(f"[SETTINGS FILE] {settings._settings_path.absolute()}")  # noqa
    status(f"[FINN BUILD DIRECTORY] {settings.finn_build_dir}")
    status(f"[DEPDENDENCY PATH] {settings.finn_deps}")
    status(f"[DEPDENDENCY DEFINITIONS PATH] {settings.finn_deps_definitions}")
    status(f"[NUM WORKERS] {settings.num_default_workers}")
    if not settings.settingsfile_exists():
        warning("Settings file does not exist yet. Creating file now.")
        settings.save()
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = ""

    # Update / Install all dependencies
    if settings.automatic_dependency_updates:
        updater = DependencyUpdater(
            dependency_location=settings.finn_deps,
            dependency_definition_file=settings.finn_deps_definitions,
            git_timeout_s=settings.deps_git_timeout,
        )
        status(f"[EXTERNAL DEPENDENCY DEFINITION FILE] {updater.depfile.absolute()}")
        updater.update()
    else:
        warning("Skipping dependency updates!")

    # Check synthesis tools
    set_synthesis_tools_paths()

    # Resolve paths to some not properly packaged components...
    os.environ["FINN_RTLLIB"] = _resolve_module_path("finn-rtllib")
    os.environ["FINN_CUSTOM_HLS"] = _resolve_module_path("custom_hls")
    os.environ["FINN_QNN_DATA"] = _resolve_module_path("qnn-data")
    os.environ["FINN_NOTEBOOKS"] = _resolve_module_path("notebooks")
    os.environ["FINN_TESTS"] = _resolve_module_path("tests")


@click.group()
def main_group() -> None:
    """Main click group."""  # noqa
    pass


def get_function_args() -> dict:
    """Return key-values for the calling functions arguments. Filtered, so that no
    arguments accidentally get returned.
    """
    caller = inspect.stack()[1].frame
    args = inspect.getargvalues(caller).args
    d = {arg: caller.f_locals[arg] for arg in args}
    allowed = [
        "finn_deps",
        "finn_deps_definitions",
        "finn_build_dir",
        "num_default_workers",
        "flow_config",
    ]
    keys = list(d.keys())
    for key in keys:
        if key not in allowed:
            del d[key]
    return d


@click.command(help="Build a hardware design")
@finn_deps
@finn_deps_definitions
@finn_build_dir
@flow_config
@model
@num_default_workers
@skip_dep_update
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
    finn_deps: Path | None,
    finn_deps_definitions: Path | None,
    finn_build_dir: Path | None,
    num_default_workers: int,
    skip_dep_update: bool,
    start: str,
    stop: str,
    flow_config: Path,
    model: Path,
) -> None:
    """Click command line option to build a FINN flow from a YAML config and an ONNX model."""
    status(f"Starting FINN build with config {config.name} and model {model.name}!")
    settings = FINNSettings.init(
        auto_set_environment_vars=True,
        automatic_dependency_updates=not skip_dep_update,
        **get_function_args(),
    )
    prepare_finn(settings)

    # Can import from finn now, since all deps are installed
    # and all environment variables are set correctly
    from finn.builder.build_dataflow import build_dataflow_cfg
    from finn.builder.build_dataflow_config import DataflowBuildConfig

    status("Creating dataflow build config...")
    dfbc: DataflowBuildConfig | None = None
    match flow_config.suffix:
        case ".yaml" | ".yml":
            with flow_config.open() as f:
                dfbc = DataflowBuildConfig.from_yaml(f.read())
        case ".json":
            with flow_config.open() as f:
                dfbc = DataflowBuildConfig.from_json(f.read())
        case _:
            error(f"Unknown config file type: {config.name}. Valid formats are: .json, .yml, .yaml")
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
        dfbc.output_dir = str((flow_config.parent / dfbc.output_dir).absolute())
    status(f"Output directory is {dfbc.output_dir}")

    # Add path of config to sys.path so that custom steps can be found
    sys.path.append(str(flow_config.parent.absolute()))

    Console().rule(
        f"[bold cyan]Running FINN with config[/bold cyan][bold orange1] "
        f"{config.name}[/bold orange1][bold cyan] on model [/bold cyan]"
        f"[bold orange1]{model.name}[/bold orange1]"
    )
    build_dataflow_cfg(str(model), dfbc)


@click.command(help="Run a script in a FINN environment")
@finn_deps
@finn_deps_definitions
@finn_build_dir
@num_default_workers
@skip_dep_update
@click.argument(
    "script",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, executable=True, path_type=Path),
)
def run(
    finn_deps: Path | None,
    finn_deps_definitions: Path | None,
    finn_build_dir: Path | None,
    skip_dep_update: bool,
    num_workers: int,
    script: Path,
) -> None:
    """Click command line option to run a script in a FINN+ context.

    Can be used for backwards compability with old FINN build flows.
    """
    script = script.expanduser()
    settings = FINNSettings.init(
        auto_set_environment_vars=True,
        automatic_dependency_updates=not skip_dep_update,
        flow_config=script,
        **get_function_args(),
    )
    prepare_finn(settings)
    Console().rule(
        f"[bold cyan]Starting script [/bold cyan][bold orange1]{script.name}[/bold orange1]"
    )
    subprocess.run(
        shlex.split(f"{sys.executable} {script.name}", posix=IS_POSIX), cwd=script.parent
    )


@click.command(help="Run a given benchmark configuration.")
@click.option("--bench_config", help="Name or path of experiment configuration file", required=True)
@finn_deps
@finn_deps_definitions
@finn_build_dir
@num_default_workers
def bench(
    bench_config: str,
    finn_deps: Path | None,
    finn_deps_definitions: Path | None,
    num_default_workers: int,
    finn_build_dir: Path | None,
) -> None:
    """Run a benchmark."""
    settings = FINNSettings.init(
        auto_set_environment_vars=True,
        automatic_dependency_updates=not skip_dep_update,
        **get_function_args(),
    )
    prepare_finn(settings)
    Console().rule("RUNNING BENCHMARK")

    # Late import because we need prepare_finn to setup remaining dependencies first
    from finn.benchmarking.bench import start_bench_run

    exit_code = start_bench_run(bench_config)
    sys.exit(exit_code)


@click.command(help="Run a given test. Uses /tmp/FINN_TMP as the temporary file location")
@finn_deps
@finn_deps_definitions
@finn_build_dir
@num_default_workers
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
    finn_deps: Path | None,
    finn_deps_definitions: Path | None,
    num_default_workers: int,
    num_test_workers: str,
    build_path: Path | None,
) -> None:
    """Run a selected subset of the FINN(+) testsuite."""
    settings = FINNSettings.init(
        auto_set_environment_vars=True,
        automatic_dependency_updates=not skip_dep_update,
        **get_function_args(),
    )
    if not settings.finn_build_dir.exists():
        settings.finn_build_dir = Path("/tmp/FINN_TEST_BUILD_DIR")
    prepare_finn(settings)
    status(f"Using {num_test_workers} test workers")
    Console().rule("RUNNING TESTS")
    run_test(variant, num_test_workers)


@click.group(help="Dependency management")
def deps() -> None:
    """Click group collecting depenency related commands."""
    pass


@click.command(help="Update or install dependencies to the given path")
@finn_deps
@finn_deps_definitions
def update(finn_deps: Path | None, finn_deps_definitions: Path | None) -> None:
    """Update all FINN+ dependencies and then exit."""
    settings = FINNSettings.init(
        auto_set_environment_vars=True,
        automatic_dependency_updates=True,
        flow_config=Path(),
        **get_function_args(),
    )
    prepare_finn(settings)


@click.group(help="Manage FINN settings")
def config() -> None:
    """Click group for config related commands."""
    # TODO: Config remove?
    pass


def _command_get_settings() -> FINNSettings:
    settings = FINNSettings.init(
        auto_set_environment_vars=True, automatic_dependency_updates=False, flow_config=Path()
    )
    prepare_finn(settings)
    if not settings.settingsfile_exists():
        warning("Could not resolve settings file.")
        sys.exit(1)
    return settings


@click.command("list", help="List the settings files contents")
def config_list() -> None:
    """List all settings found in the current settings file."""
    rich.print(_command_get_settings())


@click.command("get", help="Get a specific key from the settings")
@click.argument("key")
def config_get(key: str) -> None:
    """Get the value of a settings key."""
    settings = _command_get_settings().model_dump()
    if key not in settings.keys():
        error(f"Key {key} could not be found in the settings file!")
        sys.exit(1)
    Console().print(f"[blue]{key}[/blue]: {settings[key]}")


@click.command("set", help="Set a key to a given value")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a setting key to a given value."""
    # settings_original = _command_get_settings()
    # TODO: Fix
    # settings_original.model_copy(update={key: value}, deep=True)
    # settings_original.save()


@click.command("help", help="List all known settings fields and their purpose")
def config_help() -> None:
    """Print a table with all known settings keys and their purpose."""
    # TODO
    # table(FINNSettings.get_settings_keys(), "Settings Key", "Purpose")


@click.command(
    "create",
    help="Create a template settings file. If one exists at the given path, "
    "its overwritten. Please enter a directory, no filename",
)
@click.argument("path", default="~/.finn/")
def config_create(path: str) -> None:
    """Create a template config at the `<given path>/settings.yaml`. Uses the default values."""
    p = Path(path).expanduser().absolute()
    if not p.exists():
        error("The given path does not seem to exist!")
        sys.exit(1)
    if p.suffix != "":
        error("Please specify a path to a directory, not a file!")
        sys.exit(1)
    p = p / "settings.yaml"
    if p.exists():
        status(f"A settings file already exists at {p}")
        return
    settings = FINNSettings.init(override_settings_path=p, flow_config=Path())
    settings.save()
    if not p.exists():
        error(f"Failed to create config template at {p}!")
        sys.exit(1)
    status(f"File written to {p}")


def main() -> None:
    """Clicks entrypoint function."""
    config.add_command(config_list)
    config.add_command(config_create)
    config.add_command(config_get)
    config.add_command(config_set)
    config.add_command(config_help)
    deps.add_command(update)
    main_group.add_command(config)
    main_group.add_command(deps)
    main_group.add_command(build)
    main_group.add_command(bench)
    main_group.add_command(test)
    main_group.add_command(run)
    try:
        main_group()
    except FINNValidationError as e:
        error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()

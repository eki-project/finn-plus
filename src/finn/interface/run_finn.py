"""Run FINN+."""
# ruff: noqa: PIE790
from __future__ import annotations

import click
import os
import shlex
import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.pretty import pprint
from typing import TYPE_CHECKING, Any

from finn.interface import IS_POSIX
from finn.interface.interface_utils import (
    NullablePath,
    _resolve_module_path,
    error,
    set_synthesis_tools_paths,
    status,
    table,
    warning,
)
from finn.interface.manage_deps import DependencyUpdater
from finn.interface.manage_tests import run_test
from finn.interface.settings import FINNSettings

if TYPE_CHECKING:
    from collections.abc import Callable


def requires_dependencies(f: Callable) -> Callable[..., Any]:
    """Add a click parameter named --dependency-path (-d) that defaults to
    None if the param is empty, and a path otherwise."""  # noqa
    return click.option("--dependency-path", "-d", default="", type=NullablePath())(f)


def requires_dependency_definitions(f: Callable) -> Callable[..., Any]:
    """Add a click parameter named --dependency-definitions (-D) that defaults to
    None if the param is empty, and a path otherwise."""  # noqa
    return click.option("--dependency-definitions", "-D", default="", type=NullablePath())(f)


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
    deps_definitions: Path | None,
    flow_config: Path,
    build_dir: Path | None,
    num_workers: int,
    is_test_run: bool = False,
    cmdparam_skip_dep_update: bool = False,
) -> FINNSettings:
    """Prepare a FINN environment. After executing this function, FINN can be run. The function
    receives a `FINNSettings` object which, combined with optional override values,
    resolves all paths and settings. The modified settings are returned.

    Args:
        settings: FINNSettings used for configuring FINN. Doesn't need to be complete.
        deps: Path to the dependency directory. If none resolved via settings and env-vars.
        deps_definitions: Path to the dependency definition file. If none resolved
                            via settings and env-vars.
        flow_config: Path to the build config (yaml or script).
        build_dir: Path to temporary files directory. If none resolved via settings and env-vars.
        num_workers: Number of workers used in parallel tasks. Further resolved by settings.
        is_test_run: True passed when running tests to modify build_dir.
        cmdparam_skip_dep_update: Whether to skip dependency updates.

    Returns:
        FINNSettings: Modified settings. If the settings file didn't exist before, it was created.
    """
    settings_path = settings.get_path()
    status(f"[SETTINGS FILE] {settings_path.absolute()}")

    # Load defaults only if the settings file does not exist.
    # If a settings file exists but a key is not given, it is resolved below.
    if not settings_path.exists():
        warning("Settings file does not exist yet. Creating file with default values now.")
        if not settings.load_defaults():
            error("Could not write settings file. Make sure the parent path exists.")
            sys.exit(1)

    # Set arbitrary git timeout
    if "DEPS_GIT_TIMEOUT" not in settings:
        settings["DEPS_GIT_TIMEOUT"] = 100

    # Dependencies
    deps_path = settings.resolve_deps_path(deps)
    status(f"[DEPENDENCY PATH] {deps_path}")
    if "FINN_DEPS" not in settings:
        settings["FINN_DEPS"] = deps_path.absolute()
    os.environ["FINN_DEPS"] = str(deps_path.absolute())

    # Skipping dependency updates
    settings["AUTOMATIC_DEPENDENCY_UPDATES"] = not settings.resolve_skip_update(
        cmdparam_skip_dep_update
    )

    # Pythonpath
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = ""

    # Update / Install all dependencies
    if settings["AUTOMATIC_DEPENDENCY_UPDATES"]:
        if "FINN_DEPS_DEFINITIONS" not in settings:
            settings["FINN_DEPS_DEFINITIONS"] = settings.resolve_deps_definitions_path(
                deps_definitions
            )
        updater = DependencyUpdater(
            dependency_location=deps_path,
            dependency_definition_file=settings["FINN_DEPS_DEFINITIONS"],
            git_timeout_s=settings["DEPS_GIT_TIMEOUT"],
        )
        status(f"[EXTERNAL DEPENDENCY DEFINITION FILE] {updater.depfile.absolute()}")
        updater.update()
    else:
        warning("Skipping dependency updates!")

    # Check synthesis tools
    set_synthesis_tools_paths()

    # Resolve the build directory
    resolved_build_dir = settings.resolve_build_dir(build_dir, flow_config, is_test_run).absolute()
    settings["FINN_BUILD_DIR"] = resolved_build_dir
    os.environ["FINN_BUILD_DIR"] = str(resolved_build_dir)
    if not resolved_build_dir.exists():
        resolved_build_dir.mkdir(parents=True)
    status(f"[BUILD DIRECTORY] {resolved_build_dir}")

    # Resolve the number of workers
    workers = settings.resolve_num_workers(num_workers)
    status(f"[NUM DEFAULT WORKERS] {workers} workers")
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
    """Main click group."""  # noqa
    pass


@click.command(help="Build a hardware design")
@requires_dependencies
@requires_dependency_definitions
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
    dependency_path: Path | None,
    dependency_definitions: Path | None,
    build_path: Path | None,
    num_workers: int,
    skip_dep_update: bool,
    start: str,
    stop: str,
    config: Path,
    model: Path,
) -> None:
    """Click command line option to build a FINN flow from a YAML config and an ONNX model."""
    status(f"Starting FINN build with config {config.name} and model {model.name}!")
    prepare_finn(
        FINNSettings(),
        dependency_path,
        dependency_definitions,
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
@requires_dependency_definitions
@requires_builddir
@with_num_workers_option
@with_skip_depenency_update_option
@click.argument(
    "script",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, executable=True, path_type=Path),
)
def run(
    dependency_path: Path | None,
    dependency_definitions: Path | None,
    build_path: Path | None,
    skip_dep_update: bool,
    num_workers: int,
    script: Path,
) -> None:
    """Click command line option to run a script in a FINN+ context.

    Can be used for backwards compability with old FINN build flows.
    """
    script = script.expanduser()
    prepare_finn(
        FINNSettings(),
        dependency_path,
        dependency_definitions,
        script,
        build_path,
        num_workers,
        cmdparam_skip_dep_update=skip_dep_update,
    )
    Console().rule(
        f"[bold cyan]Starting script [/bold cyan][bold orange1]{script.name}[/bold orange1]"
    )
    subprocess.run(
        shlex.split(f"{sys.executable} {script.name}", posix=IS_POSIX), cwd=script.parent
    )


@click.command(help="Run a given benchmark configuration.")
@click.option("--bench_config", help="Name or path of experiment configuration file", required=True)
@requires_dependencies
@requires_dependency_definitions
@requires_builddir
@with_num_workers_option
def bench(
    bench_config: str,
    dependency_path: Path | None,
    dependency_definitions: Path | None,
    num_workers: int,
    build_path: Path | None,
) -> None:
    """Run a benchmark."""
    console = Console()
    prepare_finn(
        FINNSettings(), dependency_path, dependency_definitions, Path(), build_path, num_workers
    )
    console.rule("RUNNING BENCHMARK")

    # Late import because we need prepare_finn to setup remaining dependencies first
    from finn.benchmarking.bench import start_bench_run

    exit_code = start_bench_run(bench_config)
    sys.exit(exit_code)


@click.command(help="Run a given test. Uses /tmp/FINN_TMP as the temporary file location")
@requires_dependencies
@requires_dependency_definitions
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
    dependency_definitions: Path | None,
    num_workers: int,
    num_test_workers: str,
    build_path: Path | None,
) -> None:
    """Run a selected subset of the FINN(+) testsuite."""
    console = Console()
    prepare_finn(
        FINNSettings(),
        dependency_path,
        dependency_definitions,
        Path(),
        build_path,
        num_workers,
        is_test_run=True,
    )
    status(f"Using {num_test_workers} test workers")
    console.rule("RUNNING TESTS")
    run_test(variant, num_test_workers)


@click.group(help="Dependency management")
def deps() -> None:
    """Click group collecting depenency related commands."""
    pass


@click.command(help="Update or install dependencies to the given path")
@requires_dependencies
@requires_dependency_definitions
def update(dependency_path: Path | None, dependency_definitions: Path | None) -> None:
    """Update all FINN+ dependencies and then exit."""
    prepare_finn(FINNSettings(), dependency_path, dependency_definitions, Path(), None, 1)


@click.group(help="Manage FINN settings")
def config() -> None:
    """Click group for config related commands."""
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
    """List all settings found in the current settings file."""
    pprint(_command_get_settings()._settings)  # noqa


@click.command("get", help="Get a specific key from the settings")
@click.argument("key")
def config_get(key: str) -> None:
    """Get the value of a settings key."""
    settings = _command_get_settings()
    if key not in settings.keys():
        error(f"Key {key} could not be found in the settings file!")
        sys.exit(1)
    Console().print(f"[blue]{key}[/blue]: {settings[key]}")


@click.command("set", help="Set a key to a given value")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a setting key to a given value."""
    settings = FINNSettings(sync=True)
    settings[key] = value


@click.command("help", help="List all known settings fields and their purpose")
def config_help() -> None:
    """Print a table with all known settings keys and their purpose."""
    table(FINNSettings.KNOWN_KEYS, "Settings Key", "Purpose")


@click.command(
    "create",
    help="Create a template settings file. If one exists at the given path, "
    "its overwritten. Please enter a directory, no filename",
)
@click.argument("path", default="~/.finn/")
def config_create(path: str) -> None:
    """Create a template config at the `<given path>/settings.yaml`. Uses the default values."""
    p = Path(path).expanduser()
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
    settings = FINNSettings(sync=True, override_path=p)

    # This should return good defaults for a template
    settings = prepare_finn(
        settings=settings,
        deps=None,
        deps_definitions=None,
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
    main_group()


if __name__ == "__main__":
    main()

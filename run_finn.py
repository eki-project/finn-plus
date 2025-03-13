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
    SETTINGS,
    SETTINGS_PATH,
    SETTINGS_PATH_VARS,
    update_settings,
    write_yaml,
)
from interface.manage_deps import (
    check_verilator_version,
    try_install_verilator,
    update_dependencies,
)
from interface.manage_envvars import get_finn_envvars, print_envvars
from interface.manage_tests import run_test


def assert_path_valid(p: Path) -> None:
    """Check if the path exists, if not print an error message and exit with an error code"""
    if not p.exists():
        Console().print(f"[bold red]File or directory {p} does not exist. Stopping...[/bold red]")
        sys.exit(1)


def _update(path: Path | None) -> None:
    """Update dependencies and notify the user of the results. If updating fails,
    this ends the execution"""
    console = Console()
    if path is None:
        path = SETTINGS["DEFAULT_DEPS"]
    with console.status("[bold cyan]Gathering dependencies...[/bold cyan]") as _:
        update_status = update_dependencies(path)
    with console.status("[bold cyan]Installing verilator...[/bold cyan]") as _:
        vname, vsuc, vmsg = try_install_verilator(path)
    table = Table()
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Message")
    any_failed = False
    for pkg_name, success, msg in update_status:
        if not success:
            any_failed = True
        table.add_row(
            pkg_name,
            "[bold green]Success[/bold green]" if success else "[bold red]Failed[/bold red]",
            msg,
        )
    table.add_row(
        vname, "[bold green]Success[/bold green]" if vsuc else "[bold red]Failed[/bold red]", vmsg
    )
    console.print(table)
    if any_failed:
        console.print("[bold red]ERROR: [/bold red][red]Dependency update failed. Stopping.[/red]")
        sys.exit(1)


def prepare_finn(deps: Path | None, flow_config: Path, local_temps: bool, num_workers: int) -> None:
    """Prepare a FINN environment (fetch deps, set envvars). Print a summary at the end.
    If deps is None then use the default deps location. This is configured in the settings file or,
    if not present pointing to ~/.finn/deps by default"""
    console = Console()
    envs = get_finn_envvars(
        deps=deps, config_path=flow_config, local_temps=local_temps, num_workers=num_workers
    )
    for key in ["VIVADO_PATH", "VITIS_PATH", "HLS_PATH"]:
        p = Path(envs[key])
        if not p.exists():
            console.print(
                f"[bold orange1]WARNING: [/bold orange1][orange3]Could not "
                f"find executable defined in {key} (at {p})!"
            )
    _update(deps)
    for k, v in envs.items():
        os.environ[k] = str(v)
    verilator_version = check_verilator_version()
    if verilator_version is None:
        console.print(
            "[bold red]ERROR: Verilator could not be found or executed properly after "
            "the local installation. Stopping... [/bold red]"
        )
        sys.exit(1)
    elif verilator_version is not None and verilator_version < "4.224":
        console.print(
            f"[bold orange1]WARNING: [/bold orange1][orange3]It seems you are using verilator "
            f"version [bold]{verilator_version}[/bold]. "
            "The recommended version is [bold]4.224[/bold]. "
            "FIFO-Sizing or simulations might fail due to verilator errors.[/orange3]"
        )
    else:
        console.print(f"[bold green]Verilator version {verilator_version} found![/bold green]")
    print_envvars(envs)


@click.group()
def main_group() -> None:
    pass


@click.command(help="Build a hardware design")
@click.option("--dependency-path", "-d", default="")
@click.option(
    "--no-local-temps",
    "-l",
    help="If given, stores temporary files in the "
    "default global location and not next to the config.",
    is_flag=True,
)
@click.option(
    "--num-workers",
    "-n",
    help="Number of parallel workers for FINN to use. When -1, automatically use 75% of cores",
    default=-1,
    show_default=True,
)
@click.argument("config")
@click.argument("model")
def build(
    dependency_path: str, no_local_temps: bool, num_workers: int, config: str, model: str
) -> None:
    config_path = Path(config)
    model_path = Path(model)
    if dependency_path != "":
        dep_path = Path(dependency_path)
        assert_path_valid(dep_path)
    else:
        dep_path = None
    assert_path_valid(config_path)
    assert_path_valid(model_path)
    console = Console()
    console.print(
        f"[bold cyan]Starting FINN build with config {config_path.name} on model {model_path.name}"
    )
    prepare_finn(dep_path, config_path, not no_local_temps, num_workers)
    console.print("[bold cyan]Creating dataflow build config...[/bold cyan]")
    dfbc: DataflowBuildConfig | None = None
    match config_path.suffix:
        case ".yaml" | ".yml":
            raise NotImplementedError("Depends on a pending PR for YAML support.")
        case ".json":
            with config_path.open() as f:
                dfbc = DataflowBuildConfig.from_json(f.read())
        case _:
            console.print(
                f"[bold red]Unknown config file type: {config_path.name}. "
                "Valid formats are: .json, .yml, .yaml"
            )
            sys.exit(1)

    if dfbc is None:
        console.print("[bold red]Failed to generate dataflow build config!")
        sys.exit(1)

    console.rule(
        f"[bold cyan]Running FINN with config[/bold cyan][bold orange1] "
        f"{config_path.name}[/bold orange1][bold cyan] on model [/bold cyan]"
        "[bold orange1]{model_path.name}[/bold orange1]"
    )
    build_dataflow_cfg(str(model_path), dfbc)


@click.command(help="Run a script in a FINN environment")
@click.option("--dependency-path", "-d", default="")
@click.option(
    "--no-local-temps",
    "-l",
    help="If given, stores temporary files in the "
    "default global location and not next to the config.",
    is_flag=True,
)
@click.option(
    "--num-workers",
    "-n",
    help="Number of parallel workers for FINN to use. When -1, automatically use 75% of cores",
    default=-1,
    show_default=True,
)
@click.argument("script")
def run(dependency_path: str, no_local_temps: bool, num_workers: int, script: str) -> None:
    console = Console()
    script_path = Path(script)
    assert_path_valid(script_path)
    if dependency_path != "":
        dep_path = Path(dependency_path)
        assert_path_valid(dep_path)
    else:
        dep_path = None
    prepare_finn(
        deps=dep_path,
        flow_config=script_path,
        local_temps=not no_local_temps,
        num_workers=num_workers,
    )
    console.rule(
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
def test(variant: str, dependency_path: str, num_workers: int, num_test_workers: str) -> None:
    console = Console()
    if dependency_path != "":
        dep_path = Path(dependency_path)
        assert_path_valid(dep_path)
    else:
        dep_path = None
    prepare_finn(dep_path, Path(), False, num_workers)
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
    default=str(SETTINGS["DEFAULT_DEPS"]),
    show_default=True,
)
def update(path: str) -> None:
    _update(Path(path))


@click.group(help="Manage FINN settings")
def config() -> None:
    # TODO: Config remove?
    pass


@click.command("list", help="List all configuration parameters and their values")
def config_list() -> None:
    console = Console()
    if SETTINGS_PATH.exists():
        update_settings()
        for k, v in SETTINGS.items():
            console.print(f"[italic cyan]{k}: [/italic cyan]{v}")
    else:
        console.print(
            f"[bold red]ERROR: [/bold red]"
            f"[red]Settings file not found at {SETTINGS_PATH}![/red]"
        )


@click.command("get", help="Get the value of a given config parameter")
@click.argument("key")
def config_get(key: str) -> None:
    console = Console()
    if SETTINGS_PATH.exists():
        update_settings()
        if key not in SETTINGS.keys():
            console.print(f"[bold red]ERROR: [/bold red]Key {key} not found in settings file!")
        else:
            console.print(f"[italic cyan]{key}: [/italic cyan]{SETTINGS[key]}")
    else:
        console.print(
            f"[bold red]ERROR: [/bold red]"
            f"[red]Settings file not found at {SETTINGS_PATH}![/red]"
        )


@click.command("set", help="Set the value of a given config parameter")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    global SETTINGS
    console = Console()
    if not SETTINGS_PATH.exists():
        console.print(
            f"[bold orange1]WARNING: [/bold orange1][orange3]"
            f"Settings file at {SETTINGS_PATH} does not yet exist.[/orange3]"
        )
        ans = console.input("Create file? (y/n) > ")
        if ans.lower() == "y":
            result = write_yaml(SETTINGS, SETTINGS_PATH)
            if not result:
                console.print(
                    "[bold red]ERROR: [/bold red][red]Could not create settings file. "
                    "Make sure you ran FINN once before or that ~/.finn exists![/red]"
                )
                sys.exit(1)
            else:
                console.print("[green]Created settings file![/green]")
    if key in SETTINGS_PATH_VARS:
        SETTINGS[key] = Path(value)
    else:
        SETTINGS[key] = value
    console.print(SETTINGS)
    if write_yaml(SETTINGS, SETTINGS_PATH):
        console.print("[green]Key updated![/green]")
    else:
        console.print(
            "[bold red]ERROR: [/bold red][red]Could not write settings file. "
            "Make sure you ran FINN once before or that ~/.finn exists![/red]"
        )


def main() -> None:
    config.add_command(config_list)
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

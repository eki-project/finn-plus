import click
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table

from interface.manage_deps import update_dependencies
from interface.manage_envvars import (
    DEFAULT_FINN_TMP,
    get_global_envvars,
    get_run_specific_envvars,
)

DEFAULT_DEPS = Path.home() / ".finn" / "deps"
DEFAULT_FINN_ROOT = Path(__file__).parent
DEFAULT_ENVVAR_CONFIG = Path.home() / ".finn" / "envvars.yaml"


def _update(path: str) -> None:
    """Update dependencies and notify the user of the results"""
    console = Console()
    with console.status("[bold cyan]Gathering dependencies...[/bold cyan]") as _:
        update_status = update_dependencies(Path(path))
    table = Table()
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Message")
    for pkg_name, success, msg in update_status:
        table.add_row(
            pkg_name,
            "[bold green]Success[/bold green]" if success else "[bold red]Failed[/bold red]",
            msg,
        )
    console.print(table)


def prepare_finn(config: Path, deps: Path, local_temps: bool, num_workers: int) -> None:
    """Prepare a FINN environment (fetch deps, set envvars)"""
    _update(str(deps))
    envs = get_global_envvars(config)
    envs.update(get_run_specific_envvars(deps, config, local_temps, num_workers))
    for k, v in envs.items():
        os.environ[k] = v


@click.group()
def main_group() -> None:
    pass


@click.command(help="Build a hardware design")
@click.option("--dependency-path", "-d", default=str(DEFAULT_DEPS), show_default=True)
@click.option(
    "--no-local-temps",
    "-l",
    help="If given, stores temporary files in the "
    f"default global location at {DEFAULT_FINN_TMP} and not next to the config.",
    is_flag=True,
)
@click.option(
    "--num-workers",
    "-n",
    help="Number of parallel workers for FINN to use. " "When -1, automatically use 75% of cores",
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
    console = Console()
    console.print(
        f"[bold green]Starting FINN build with config {config_path.name} on model {model_path.name}"
    )
    prepare_finn(Path(config), Path(dependency_path), not no_local_temps, num_workers)


@click.command(help="Run a script in a FINN environment")
@click.option("--dependency-path", "-d", default=str(DEFAULT_DEPS), show_default=True)
@click.option(
    "--no-local-temps",
    "-l",
    help="If given, stores temporary files in the "
    f"default global location at {DEFAULT_FINN_TMP} and not next to the config.",
    is_flag=True,
)
@click.option(
    "--num-workers",
    "-n",
    help="Number of parallel workers for FINN to use. " "When -1, automatically use 75% of cores",
    default=-1,
    show_default=True,
)
@click.argument("script")
def run(dependency_path: str, no_local_temps: bool, num_workers: int, script: str) -> None:
    prepare_finn(Path(script), Path(dependency_path), not no_local_temps, num_workers)


@click.group(help="Dependency management")
def deps() -> None:
    pass


@click.command(help="Update or install dependencies to the given path")
@click.option(
    "--path", "-p", help="Path to install to", default=str(DEFAULT_DEPS), show_default=True
)
def update(path: str) -> None:
    _update(path)


def main() -> None:
    main_group()


if __name__ == "__main__":
    deps.add_command(update)
    main_group.add_command(deps)
    main_group.add_command(build)
    main_group.add_command(run)
    main()

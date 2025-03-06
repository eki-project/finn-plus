import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from interface.manage_deps import update_dependencies

DEFAULT_DEPS = Path.home() / ".finn" / "deps"
DEFAULT_FINN_ROOT = Path(__file__).parent
DEFAULT_ENVVAR_CONFIG = Path.home() / ".finn" / "envvars.yaml"


@click.group()
def main_group() -> None:
    pass


@click.command(help="Build a hardware design")
@click.argument("config")
@click.argument("model")
def build(config: str, model: str) -> None:
    config_path = Path(config)
    model_path = Path(model)
    console = Console()
    console.print(
        f"[bold green]Starting FINN build with config {config_path.name} on model {model_path.name}"
    )


@click.command(help="Run a script in a FINN environment")
@click.argument("script")
def run(script: str) -> None:
    pass


@click.group(help="Dependency management")
def deps() -> None:
    pass


@click.command(help="Update or install dependencies to the given path")
@click.option(
    "--path", "-p", help="Path to install to.", default=str(DEFAULT_DEPS), show_default=True
)
def update(path: str) -> None:
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


def main() -> None:
    main_group()


if __name__ == "__main__":
    deps.add_command(update)
    main_group.add_command(deps)
    main_group.add_command(build)
    main_group.add_command(run)
    main()

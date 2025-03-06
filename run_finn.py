import click
from pathlib import Path
from rich.console import Console

DEFAULT_DEPS = Path.home() / ".finn" / "deps"
DEFAULT_FINN_ROOT = Path(__file__).parent


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
@click.option("--path", "-p", help="Path to install to.", default=DEFAULT_DEPS, show_default=True)
def update() -> None:
    pass


def main() -> None:
    main_group()


if __name__ == "__main__":
    deps.add_command(update)
    main_group.add_command(deps)
    main_group.add_command(build)
    main_group.add_command(run)
    main()

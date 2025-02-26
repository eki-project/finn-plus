from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from rich.live import Live

from interface.finn_deps import FINN_BOARDFILES, FINN_DEPS, pull_boardfile, pull_dep

#### GROUPS ####

@click.group()
def main_group(): pass

@click.group()
def deps(): pass


#### DEPENDENCIES ####

@click.command("update")
@click.option("--path", default=str(Path.home() / ".finn" / "deps"), help="Path to directory where dependencies lie")
def update_deps(path):
    deppath = Path(path).absolute()
    console = Console()
    console.print(f"Installing dependencies to: [cyan]{deppath}[/cyan]")
    if not deppath.exists():
        deppath.mkdir(parents=True)
    table = Table(header_style="bold")
    table.add_column("Dependency")
    table.add_column("Status", justify="center")
    table.add_column("Message")
    with console.status("[bold cyan]Gathering dependencies...[/bold cyan]") as status:
        for name, value in FINN_DEPS.items():
            repo, commit = value
            msg, status = pull_dep(name, repo, commit, deppath)
            table.add_row(name, "[green]✔[/green]" if status else "[red]✘[/red]", msg)
        for name, value in FINN_BOARDFILES.items():
            repo, commit, copy_path = value
            msg, status = pull_boardfile(name, repo, commit, Path(copy_path), deppath)
            table.add_row(name, "[green]✔[/green]" if status else "[red]✘[/red]", msg)

    console.print(table)
    console.print("Done.")


@click.command("install")
@click.option("--path", default=str(Path.home() / ".finn" / "deps"), help="Path to directory where dependencies lie")
def install_deps(path):
    update_deps(path)


@click.command()
def show():
    print("my deps...")



#### BUILD ####

@click.command()
@click.argument("configfile")
def build(): pass


deps.add_command(update_deps)
deps.add_command(show)
main_group.add_command(deps)
main_group.add_command(build)

if __name__ == "__main__":
    main_group()
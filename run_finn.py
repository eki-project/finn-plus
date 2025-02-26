from pathlib import Path
import shutil
import subprocess
import sys
import click
import os
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import Progress
from rich.panel import Panel

from interface.finn_deps import FINN_BOARDFILES, FINN_DEPS, deps_exist, pull_boardfile, pull_dep
from interface.finn_envvars import required_envvars, preserve_envvars, restore_envvars, set_missing_envvars

#### GROUPS ####

@click.group()
def main_group(): pass

@click.group()
def deps(): pass


#### DEPENDENCIES ####
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


def check_deps(path):
    console = Console()
    success, missing = deps_exist()
    if success:
        console.print("[bold green]All dependencies found![/bold green]")
    else:
        console.print("[red] Missing some dependencies:[/red]")
        table = Table()
        table.add_column("Dependency")
        table.add_column("Expected location")
        for name, exp in missing:
            table.add_row(f"[red]{name}[/red]", str(exp))
        console.print(table)
        sys.exit(1)


@click.command("update")
@click.option("--path", default=str(Path.home() / ".finn" / "deps"), help="Path to directory where dependencies lie")
def update_deps_command(path):
    update_deps(path)


@click.command("install")
@click.option("--path", default=str(Path.home() / ".finn" / "deps"), help="Path to directory where dependencies lie")
def install_deps_command(path):
    update_deps(path)


@click.command("check")
@click.option("--path", default=str(Path.home() / ".finn" / "deps"), help="Path to directory where dependencies lie")
def check_deps_command(path):
    check_deps(path)



#### BUILD ####

@click.command()
@click.argument("buildfile")
@click.option("--force-update", "-f", help="Force an update of dependencies before starting", default=False, is_flag=True)
@click.option("--deps-path", "-d", default=str(Path.home() / ".finn" / "deps"), help="Path to directory where dependencies lie")
@click.option("--local-temps", "-l", default=True, is_flag=True, help="Whether to store temporary build files local to the model/buildfile. Defaults to true")
def build(buildfile, force_update, deps_path, local_temps):
    # TODO: Keep usage of str vs Path() consistent everywhere

    console = Console()
    buildfile_path = Path(buildfile)
    if not buildfile_path.exists():
        console.print(f"[bold red]Could not find buildfile at: {buildfile_path}[/bold red]")
        sys.exit(1)

    # Check dependencies
    if not force_update:
        check_deps(deps_path)
    else:
        update_deps(deps_path)

    if shutil.which("verilator") is None:
        console.print("[bold red]Could not find [italic]verilator[/italic] in path! Please install verilator before continuing.[/bold red]")
    else:
        console.print("[bold green]Verilator found![/bold green]")
    
    # Conserve environment variables
    preserved = preserve_envvars(buildfile_path, local_temps, Path(deps_path))
    set_missing_envvars(buildfile_path, local_temps, Path(deps_path))

    # Run FINN
    console.print(Panel(f"Dependency directory: {deps_path}\nBuildfile: {buildfile_path.absolute()}"))
    console.print("\n")
    console.rule("RUNNING FINN")
    subprocess.run(f"python {buildfile_path.name}", shell=True, cwd=buildfile_path.parent.absolute())

    # Restore old variables
    restore_envvars(preserved)



@click.command(help="Run a complete setup (Pulling deps, setting envvars, etc)")
def setup():
    console = Console()
    console.print("Running FINN setup...")
    console.print("\n\n")
    console.rule("ENVIRONMENT VARIABLES")
    suggestions = required_envvars(Path("."), False, Path.home() / ".finn" / "deps")
    for varname, varcontent in suggestions.items():
        if varname in ["FINN_BUILD_DIR", "FINN_HOST_BUILD_DIR", "FINN_ROOT"]:
            continue
        if varname in os.environ.keys():
            ans = console.input(f"Variable [bold]{varname}[/bold] already exists with value \"{os.environ[varname]}\". Replace? (leave blank to leave old value) > ")
            if ans != "":
                os.environ[varname] = ans
        else:
            ans = ""
            while ans not in ["d", "l", "r"]:
                ans = console.input(f"Variable [bold]{varname}[/bold] not set. Default is \"{varcontent}\". Replace? (r Replace / l Leave unset / d Default) > ")
            if ans == "l":
                continue
            elif ans == "r":
                ans = console.input("New value > ")
                os.environ[varname] = ans
            elif ans == "d":
                os.environ[varname] = varcontent

    console.print("\n\n")
    console.rule("DEPENDENCIES")
    update_deps(Path.home() / ".finn" / "deps")
    console.print("\n\n")
    console.print(Panel("FINN is now set up!")) 
    


deps.add_command(update_deps_command)
deps.add_command(check_deps_command)
main_group.add_command(deps)
main_group.add_command(build)
main_group.add_command(setup)

if __name__ == "__main__":
    main_group()
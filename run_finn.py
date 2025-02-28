import os
from pathlib import Path
import shutil
import subprocess
import sys
import click
from rich.table import Table
from rich.panel import Panel
from rich.console import Console

from interface.finn_deps import FINN_BOARDFILES, FINN_DEPS, deps_exist, pull_boardfile, pull_dep
from interface.finn_envvars import GLOBAL_FINN_ENVVARS, generate_envvars


########### TODO #############
# - Copy .Xilinx to HOME dir when starting
# - Vivado IP Cache env var (run-docker.sh)


#### GROUPS ####

@click.group()
def main_group(): pass

@click.group(help="Dependency related commands. Use to check and update your non-python based FINN dependencies")
def deps(): pass

@click.group(help="Run tests on the given FINN installation. Use finn test --help to learn more.")
def test(): pass


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


@click.command("update", help="Install or update FINNs non-python dependencies")
@click.option("--path", default=str(Path.home() / ".finn" / "deps"), help="Path to directory where dependencies lie", show_default=True)
def update_deps_command(path):
    update_deps(path)


@click.command("check", help="Check for and print missing dependencies in the given path")
@click.option("--path", default=str(Path.home() / ".finn" / "deps"), help="Path to directory where dependencies lie", show_default=True)
def check_deps_command(path):
    check_deps(path)



#### BUILD ####

def setup_envvars():
    console = Console()
    console.print()
    first = True
    newly_set = {}
    for varname, varcontent in GLOBAL_FINN_ENVVARS.items():
        if varname not in os.environ.keys():
            if first:
                console.print(Panel("[yellow]Some environment variables required by FINN are not set. You can set these variables interactively now. To avoid this in the future, set them permanently, for example in your bashrc![/yellow]", title="[bold]Missing Environment Variables[/bold]"))
                first = False
            ans = console.input(f"\nVariable [bold]{varname}[/bold] is not set. Default is [blue]{varcontent}[/blue] (1 - leave unset, 2 - set to default, 3 - set own) > ")
            if ans == "1":
                continue
            elif ans == "2":
                os.environ[varname] = varcontent
                newly_set[varname] = varcontent
            elif ans == "3":
                os.environ[varname] = console.input("New Value > ")
                newly_set[varname] = os.environ[varname]
            else:
                console.print("[red]Unknown option. Stopping...[/red]")
                sys.exit(1)
    


@click.command(help="Run a FINN build with the given build file")
@click.argument("buildfile")
@click.option("--force-update", "-f", help="Force an update of dependencies before starting", default=False, is_flag=True)
@click.option("--deps-path", "-d", default=str(Path.home() / ".finn" / "deps"), help="Path to directory where dependencies lie", show_default=True)
@click.option("--local-temps", "-l", default=True, is_flag=True, help="Whether to store temporary build files local to the model/buildfile.")
@click.option("--num-workers", "-n", default=-1, help="Number of workers to do parallel tasks. -1 automatically uses 75% of your available cores.", show_default=True)
@click.option("--clean-temps", "-c", default=False, is_flag=True, help="Clean temporary files from previous runs automatically?")
@click.option("--ignore-missing-envvars", "-i", default=False, help="When using this flag, FINN does not interactively ask to set missing environment variables. Useful for starting FINN automatically without user input but may run into errors if variables are not set.", is_flag=True)
def build(buildfile, force_update, deps_path, local_temps, num_workers, clean_temps, ignore_missing_envvars):
    # TODO: Keep usage of str vs Path() consistent everywhere
    if not ignore_missing_envvars:
        setup_envvars()
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
    
    finnbuilddir = buildfile_path.parent.absolute() / "FINN_TMP" if local_temps else Path("/tmp/FINN_TMP")
    if clean_temps:
        for obj in finnbuilddir.iterdir():
            if obj.is_dir():
                shutil.rmtree(str(obj))
            else:
                obj.unlink()
        if len(list(finnbuilddir.iterdir())) == 0:
            console.print("[bold yellow]Deleted all previous temporary build files![/bold yellow]")
        else:
            console.print(f"[bold red]It seems that deleting old run files failed in directory {finnbuilddir}. Stopping...[/bold red]")
            sys.exit(1)

    # Run FINN
    prefix = generate_envvars(Path(__file__).parent.absolute(), buildfile_path, local_temps, Path(deps_path), num_workers)
    splitprefix = prefix.replace(" ", "\n")
    console.print(Panel(f"Prefix:\n{splitprefix}\nDependency directory: {deps_path}\nBuildfile: {buildfile_path.absolute()}"))
    console.print("\n")
    console.rule("RUNNING FINN")
    subprocess.run(f"{prefix} python {buildfile_path.name}", shell=True, cwd=buildfile_path.parent.absolute())



#### TESTS ####
@click.command(name="quicktest", help="Run the quicktests in FINN. Should only take a few minutes")
@click.option("--variant", "-v", help="Which variant of the quicktests to execute. Defaults to standard tests.", default="")
def run_quicktest():
    setup_envvars()

    subprocess.run("pytest -m 'not (vivado or slow or vitis or board or notebooks or bnn_pynq)' --dist=loadfile -n auto", shell=True) 





############### CLICK ###############

test.add_command(run_quicktest)
deps.add_command(update_deps_command)
deps.add_command(check_deps_command)
main_group.add_command(deps)
main_group.add_command(build)
main_group.add_command(test)

if __name__ == "__main__":
    main_group()
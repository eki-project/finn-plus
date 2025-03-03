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
from interface.finn_envvars import GLOBAL_FINN_ENVVARS, generate_envvars, load_preset_envvars
from interface.finn_inspect import inspect_onnx

from rich.traceback import install
install(show_locals=True)


########### TODO #############
# - Copy .Xilinx to HOME dir when starting
# - Vivado IP Cache env var (run-docker.sh)
# - Complete all tests
# - Complete the inspect function

# - Replace environment vars with a configuration that can be passed to BuildDataflowConfig(?)

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
    


@click.command(help="Run a FINN build with the given build file. Tries to use environment variables in scope. If some are missing try to read from ~/.finn/env.yaml. Otherwise asks the user")
@click.argument("buildfile")
@click.option("--force-update", "-f", help="Force an update of dependencies before starting", default=False, is_flag=True)
@click.option("--deps-path", "-d", default=str(Path.home() / ".finn" / "deps"), help="Path to directory where dependencies lie", show_default=True)
@click.option("--local-temps", "-l", default=True, is_flag=True, help="Whether to store temporary build files local to the model/buildfile.")
@click.option("--num-workers", "-n", default=-1, help="Number of workers to do parallel tasks. -1 automatically uses 75% of your available cores.", show_default=True)
@click.option("--clean-temps", "-c", default=False, is_flag=True, help="Clean temporary files from previous runs automatically?")
@click.option("--ignore-missing-envvars", "-i", default=False, help="When using this flag, FINN does not interactively ask to set missing environment variables. Useful for starting FINN automatically without user input but may run into errors if variables are not set.", is_flag=True)
@click.option("--envvar-config", "-e", help="Path to a config file containing values for FINN specific environment variables", default=str(Path.home() / ".finn" / "env.yaml"))
@click.option("--ignore-envvar-config", help="Ignore any environment variable config", default=False, is_flag=True)
def build(buildfile, force_update, deps_path, local_temps, num_workers, clean_temps, ignore_missing_envvars, envvar_config, ignore_envvar_config):
    # TODO: Keep usage of str vs Path() consistent everywhere
    console = Console()
    console.print()
    console.rule("CONFIGURATION")
    if not ignore_missing_envvars:
        if not ignore_envvar_config:
            success = load_preset_envvars(Path(envvar_config))
            if success:
                console.print("[bold green]Loaded environment variable config.[/bold green]")
            else:
                console.print("[bold yellow]Environment variable config not found or has incompatible format. You might be asked for environment variable values later.[/bold yellow]")
        setup_envvars()
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
    
    # Remove previous temp files if wanted
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
    # TODO: Implement all test variants
    subprocess.run("pytest -m 'not (vivado or slow or vitis or board or notebooks or bnn_pynq)' --dist=loadfile -n auto", shell=True) 




#### INSPECT ####

@click.command(help="Inspect something. (Takes ONNX files, build.yaml, build.py, XCLBINs). Only uses options for the given filetype")
@click.argument("obj")
@click.option("--ignore-fifos", help="Dont display FIFOs in the model tree (ONNX)", default=False, is_flag=True, show_default=True)
@click.option("--no-collapse-fifo-names", help="Collapse consecutive FIFOs in the tree to one node (ONNX)", default=False, is_flag=True, show_default=True)
@click.option("--no-ignore-sdp-prefix", help="Removes the SDP name prefix from submodel node names (ONNX)", default=False, is_flag=True, show_default=True)
@click.option("--no-display-cycle-estimates", help="Display cycle estimates next to node names (ONNX)", default=False, is_flag=True, show_default=True)
def inspect(obj, ignore_fifos, no_collapse_fifo_names, no_ignore_sdp_prefix, no_display_cycle_estimates):
    console = Console()
    objpath = Path(obj)
    if not objpath.exists():
        console.print(f"[red]File {obj} does not exist![/red]")
        sys.exit(1)
    split_name = obj.split(".")
    if len(split_name) < 2:
        console.print(f"[bold red]Cannot determine filetype since appropiate file ending is missing![/bold red]")
        sys.exit(1)
    
    ending = split_name[-1].lower()
    match ending:
        case "onnx":
            inspect_onnx(objpath, not no_ignore_sdp_prefix, not no_display_cycle_estimates, not no_collapse_fifo_names, ignore_fifos)
        case "yaml" | "yml":
            raise NotImplementedError()
        case "py":
            raise NotImplementedError()
        case "xclbin":
            raise NotImplementedError()
    


############### CLICK ###############

test.add_command(run_quicktest)
deps.add_command(update_deps_command)
deps.add_command(check_deps_command)
main_group.add_command(deps)
main_group.add_command(build)
main_group.add_command(test)
main_group.add_command(inspect)

def main():
    main_group()
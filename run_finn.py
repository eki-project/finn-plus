import click
import os
import shutil
import subprocess
import sys
from inspect import isclass
from pathlib import Path
from qonnx.transformation.base import Transformation
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install

from interface.finn_deps import (
    FINN_BOARDFILES,
    FINN_DEPS,
    deps_exist,
    pull_boardfile,
    pull_dep,
)
from interface.finn_envvars import (
    GLOBAL_FINN_ENVVARS,
    generate_envvars,
    load_preset_envvars,
    make_envvar_prefix_str,
)
from interface.finn_inspect import inspect_onnx

install(show_locals=True)


# - Copy .Xilinx to HOME dir when starting
# - Vivado IP Cache env var (run-docker.sh)
# - Complete all tests
# - Complete the inspect function

# - Replace environment vars with a configuration that can be passed to BuildDataflowConfig(?)


@click.group()
def main_group() -> None:
    pass


@click.group(
    help="Dependency related commands. Use to check and "
    "update your non-python based FINN dependencies"
)
def deps() -> None:
    pass


@click.group(help="Run tests on the given FINN installation. Use finn test --help to learn more.")
def test() -> None:
    pass


def update_deps(path: str) -> None:
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


def check_deps(path: str) -> None:
    console = Console()
    success, missing = deps_exist(path)
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
@click.option(
    "--path",
    default=str(Path.home() / ".finn" / "deps"),
    help="Path to directory where dependencies lie",
    show_default=True,
)
def update_deps_command(path: str) -> None:
    update_deps(path)


@click.command("check", help="Check for and print missing dependencies in the given path")
@click.option(
    "--path",
    default=str(Path.home() / ".finn" / "deps"),
    help="Path to directory where dependencies lie",
    show_default=True,
)
def check_deps_command(path: str) -> None:
    check_deps(path)


def setup_envvars() -> None:
    console = Console()
    first = True
    newly_set = {}
    for varname, varcontent in GLOBAL_FINN_ENVVARS.items():
        if varname not in os.environ.keys():
            if first:
                console.print(
                    Panel(
                        "[yellow]Some environment variables required by FINN are not set. You can "
                        "set these variables interactively now. To avoid this in the future, "
                        "set them permanently, for example in your bashrc![/yellow]",
                        title="[bold]Missing Environment Variables[/bold]",
                    )
                )
                first = False
            ans = console.input(
                f"\nVariable [bold]{varname}[/bold] is not set. Default is "
                "[blue]{varcontent}[/blue] (1 - leave unset, 2 - set to default, 3 - set own) > "
            )
            if ans == "1":
                continue
            if ans == "2":
                os.environ[varname] = varcontent
                newly_set[varname] = varcontent
            elif ans == "3":
                os.environ[varname] = console.input("New Value > ")
                newly_set[varname] = os.environ[varname]
            else:
                console.print("[red]Unknown option. Stopping...[/red]")
                sys.exit(1)


def prepare_finn_environment(
    targetfile: str,
    force_update: bool,
    deps_path: str,
    local_temps: bool,
    num_workers: int,
    clean_temps: bool,
    ignore_missing_envvars: bool,
    envvar_config: str,
    ignore_envvar_config: bool,
) -> str:
    """Prepare a build or run in a FINN environment. Returns a dict of required env variables.
    In detail this function does the following:
    1. Load preset environment variables at the given or default path. This _sets_ them
    2. If env vars are missing, ask the user to set them manually
    3. Check if the given target file exists
    4. Check if all deps are available and if tasked to update them
    5. Check if verilator exists
    6. Generate required env vars
    7. If asked to, delete old temporary datas
    8. Return environment variables as a dict"""
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
                console.print(
                    "[bold yellow]Environment variable config not found or has incompatible format."
                    "You might be asked for environment variable values later.[/bold yellow]"
                )
        setup_envvars()
    targetfile_path = Path(targetfile)
    if not targetfile_path.exists():
        console.print(f"[bold red]Could not find file at: {targetfile_path}[/bold red]")
        sys.exit(1)

    # Check dependencies
    if not force_update:
        check_deps(deps_path)
    else:
        update_deps(deps_path)

    if shutil.which("verilator") is None:
        console.print(
            "[bold red]Could not find [italic]verilator[/italic] in path! Please install "
            "verilator before continuing.[/bold red]"
        )
    else:
        console.print("[bold green]Verilator found![/bold green]")

    # Generate required envvars for running finn
    envvars = generate_envvars(
        Path(__file__).parent.absolute(), targetfile_path, local_temps, Path(deps_path), num_workers
    )

    # Remove previous temp files if wanted
    finnbuilddir = envvars["FINN_BUILD_DIR"]
    if clean_temps:
        for obj in finnbuilddir.iterdir():
            if obj.is_dir():
                shutil.rmtree(str(obj))
            else:
                obj.unlink()
        if len(list(finnbuilddir.iterdir())) == 0:
            console.print("[bold yellow]Deleted all previous temporary build files![/bold yellow]")
        else:
            console.print(
                f"[bold red]It seems that deleting old run files failed in directory "
                f"{finnbuilddir}. Stopping...[/bold red]"
            )
            sys.exit(1)

    # Return the environment variables required for running FINN
    return envvars


@click.command(
    help="Run a python script in a FINN environment. Usually is a FINN build script but can be"
    "anything. Checks environment variables and if some are missing tries to "
    "read from ~/.finn/env.yaml. Otherwise asks the user"
)
@click.argument("buildfile")
@click.option(
    "--force-update",
    "-f",
    help="Force an update of dependencies before starting",
    default=False,
    is_flag=True,
)
@click.option(
    "--deps-path",
    "-d",
    default=str(Path.home() / ".finn" / "deps"),
    help="Path to directory where dependencies lie",
    show_default=True,
)
@click.option(
    "--local-temps",
    "-l",
    default=True,
    is_flag=True,
    help="Whether to store temporary build files local to the model/buildfile.",
)
@click.option(
    "--num-workers",
    "-n",
    default=-1,
    help="Number of workers to do parallel tasks. -1 automatically uses "
    "75% of your available cores.",
    show_default=True,
)
@click.option(
    "--clean-temps",
    "-c",
    default=False,
    is_flag=True,
    help="Clean temporary files from previous runs automatically?",
)
@click.option(
    "--ignore-missing-envvars",
    "-i",
    default=False,
    help="When using this flag, FINN does not interactively ask to set missing environment "
    "variables. Useful for starting FINN automatically without user input but may run into errors "
    "if variables are not set.",
    is_flag=True,
)
@click.option(
    "--envvar-config",
    "-e",
    help="Path to a config file containing values for FINN specific environment variables",
    default=str(Path.home() / ".finn" / "env.yaml"),
)
@click.option(
    "--ignore-envvar-config",
    help="Ignore any environment variable config",
    default=False,
    is_flag=True,
)
def run(
    targetfile: str,
    force_update: bool,
    deps_path: str,
    local_temps: bool,
    num_workers: int,
    clean_temps: bool,
    ignore_missing_envvars: bool,
    envvar_config: str,
    ignore_envvar_config: bool,
) -> None:
    console = Console()
    console.print("\n")
    console.rule("RUNNING")
    envvars = prepare_finn_environment(
        targetfile,
        force_update,
        deps_path,
        local_temps,
        num_workers,
        clean_temps,
        ignore_missing_envvars,
        envvar_config,
        ignore_envvar_config,
    )
    prefix = make_envvar_prefix_str(envvars)
    subprocess.run(
        f"{prefix} python {targetfile.name}", shell=True, cwd=Path(targetfile).parent.absolute()
    )


@click.command(name="quicktest", help="Run the quicktests in FINN. Should only take a few minutes")
@click.option(
    "--variant",
    "-v",
    help="Which variant of the quicktests to execute. Defaults to standard tests.",
    default="",
)
@click.option("--num-workers", "-n", help="Number of pytest workers in parallel", default="auto")
@click.option(
    "--envvar-config",
    "-e",
    help="Path to a config file containing values for FINN specific environment variables",
    default=str(Path.home() / ".finn" / "env.yaml"),
)
@click.option(
    "--deps-path",
    "-d",
    default=str(Path.home() / ".finn" / "deps"),
    help="Path to directory where dependencies lie",
    show_default=True,
)
def run_quicktest(variant: str, num_workers: int, envvar_config: str, deps_path: str) -> None:
    console = Console()
    success = load_preset_envvars(Path(envvar_config))
    if success:
        console.print("[bold green]Loaded environment variable config.[/bold green]")
    else:
        console.print(
            "[bold yellow]Environment variable config not found or has incompatible format. "
            "You might be asked for environment variable values later.[/bold yellow]"
        )
    setup_envvars()
    prefix = generate_envvars(
        Path(__file__).parent.absolute(), Path.home(), False, Path(deps_path), num_workers
    )
    match variant:
        case "":
            console.print("[bold green]Starting default tests[/bold green]")
            subprocess.run(
                f"{prefix} pytest -m 'not "
                "(vivado or slow or vitis or board or notebooks or bnn_pynq)' "
                "--dist=loadfile -n {num_workers}",
                shell=True,
            )
        case "main":
            console.print("[bold green]Starting main tests[/bold green]")
            subprocess.run(
                f"{prefix} pytest -k 'not (rtlsim or end2end)' --dist=loadfile -n {num_workers}",
                shell=True,
            )
        case "rtlsim":
            console.print("[bold green]Starting RTLSIM tests[/bold green]")
            subprocess.run(f"{prefix} pytest -k rtlsim --workers {num_workers}", shell=True)
        case "end2end":
            console.print("[bold green]Starting end2end tests[/bold green]")
            subprocess.run(f"{prefix} pytest -k end2end", shell=True)
        case "full":
            console.print("[bold green]Running all tests. This might take a while[/bold green]")
            subprocess.run(
                f"{prefix} pytest -k 'not (rtlsim or end2end)' --dist=loadfile -n {num_workers}",
                shell=True,
            )
            subprocess.run(f"{prefix} pytest -k rtlsim --workers {num_workers}", shell=True)
            subprocess.run(f"{prefix} pytest -k end2end", shell=True)
        case "brevitas":
            console.print("[bold green]Brevitas tests...[/bold green]")
            subprocess.run(f"{prefix} pytest -k brevitas_export", shell=True)


@click.command(
    help="Inspect something. (Takes ONNX files, build.yaml, build.py, XCLBINs). "
    "Only uses options for the given filetype"
)
@click.argument("obj")
@click.option(
    "--ignore-fifos",
    help="Dont display FIFOs in the model tree (ONNX)",
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option(
    "--no-collapse-fifo-names",
    help="Collapse consecutive FIFOs in the tree to one node (ONNX)",
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option(
    "--no-ignore-sdp-prefix",
    help="Removes the SDP name prefix from submodel node names (ONNX)",
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option(
    "--no-display-cycle-estimates",
    help="Display cycle estimates next to node names (ONNX)",
    default=False,
    is_flag=True,
    show_default=True,
)
def inspect(
    obj: str,
    ignore_fifos: bool,
    no_collapse_fifo_names: bool,
    no_ignore_sdp_prefix: bool,
    no_display_cycle_estimates: bool,
) -> None:
    console = Console()
    objpath = Path(obj)
    if not objpath.exists():
        console.print(f"[red]File {obj} does not exist![/red]")
        sys.exit(1)
    split_name = obj.split(".")
    if len(split_name) < 2:
        console.print(
            "[bold red]Cannot determine filetype since file ending is missing![/bold red]"
        )
        sys.exit(1)

    ending = split_name[-1].lower()
    match ending:
        case "onnx":
            inspect_onnx(
                objpath,
                not no_ignore_sdp_prefix,
                not no_display_cycle_estimates,
                not no_collapse_fifo_names,
                ignore_fifos,
            )
        case "yaml" | "yml":
            raise NotImplementedError()
        case "py":
            raise NotImplementedError()
        case "xclbin":
            raise NotImplementedError()


@click.group(help="Documentation related commands")
def docs() -> None:
    pass


@click.command()
@click.argument("transformation")
@click.option(
    "--relaxed", "-r", help="Look for substring matches and display all found.", is_flag=True
)
@click.option("--show-location", "-l", help="Also show location of the class", is_flag=True)
def get(transformation: str, relaxed: bool, show_location: bool) -> None:
    console = Console()
    docs = {}
    locs = {}
    transformation_pkgs = [
        "finn.transformation.fpgadataflow",
        "finn.transformation.qonnx",
        "finn.transformation.streamline",
    ]
    with console.status("Looking up docs...") as status:
        for transformation_package in transformation_pkgs:
            actual_path = (
                Path(__file__).parent / "src" / Path(transformation_package.replace(".", "/"))
            )
            modules = [
                str(m.name).replace(".py", "")
                for m in actual_path.iterdir()
                if str(m).endswith(".py")
            ]
            modules = [m for m in modules if m != "__init__"]
            mod = __import__(transformation_package, globals(), locals(), modules, 0)
            for modname in modules:
                for potential_class_name in dir(mod.__dict__[modname]):
                    potential_class = mod.__dict__[modname].__dict__[potential_class_name]
                    if isclass(potential_class) and issubclass(potential_class, Transformation):
                        docs[potential_class_name] = potential_class.__doc__
                        locs[
                            potential_class_name
                        ] = f"{transformation_package}.{modname}.{potential_class_name}"

    for class_name, class_doc in docs.items():
        doc_text = class_doc
        if show_location:
            doc_text = (
                f"[italic orange1]Class location: {locs[class_name]}"
                "[/italic orange1]\n\n{class_doc}"
            )
        if relaxed:
            if transformation.lower() in class_name.lower():
                console.print(
                    Panel(
                        doc_text, title=f"[bold cyan]{class_name}[/bold cyan]", border_style="cyan"
                    )
                )
        else:
            if class_name.lower() == transformation.lower():
                console.print(
                    Panel(
                        doc_text, title=f"[bold cyan]{class_name}[/bold cyan]", border_style="cyan"
                    )
                )
                return
    if not relaxed:
        console.print(
            "[bold red]No documentation found for transformation "
            "[reverse]{transformation}[/reverse][/bold red]"
        )


test.add_command(run_quicktest)
deps.add_command(update_deps_command)
deps.add_command(check_deps_command)
docs.add_command(get)
main_group.add_command(deps)
main_group.add_command(run)
main_group.add_command(test)
main_group.add_command(inspect)
main_group.add_command(docs)


def main() -> None:
    main_group()


if __name__ == "__main__":
    main()

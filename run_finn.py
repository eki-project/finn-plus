import click
import os
import shutil
import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table

from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig
from interface.manage_deps import update_dependencies
from interface.manage_envvars import (
    DEFAULT_FINN_TMP,
    get_global_envvars,
    get_run_specific_envvars,
)

DEFAULT_DEPS = Path.home() / ".finn" / "deps"
DEFAULT_FINN_ROOT = Path(__file__).parent
DEFAULT_ENVVAR_CONFIG = Path.home() / ".finn" / "envvars.yaml"


def assert_path_valid(p: Path) -> None:
    """Check if the path exists, if not print an error message and exit with an error code"""
    if not p.exists():
        Console().print(f"[bold red]File or directory {p} does not exist. Stopping...[/bold red]")
        sys.exit(1)


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


def prepare_finn(
    envvar_config: Path, flow_config: Path, deps: Path, local_temps: bool, num_workers: int
) -> None:
    """Prepare a FINN environment (fetch deps, set envvars)"""
    console = Console()
    _update(str(deps))
    envs, read_config = get_global_envvars(envvar_config)
    if not read_config:
        console.print(
            f"[bold orange1]Could not read the default "
            f"environment variable config at {envvar_config}![/bold orange1]"
        )
    envs.update(get_run_specific_envvars(deps, flow_config, local_temps, num_workers))
    for k, v in envs.items():
        os.environ[k] = v
    if shutil.which("verilator") is None:
        console.print("[bold red]Verilator could not be found! Stopping...[/bold red]")
        sys.exit(1)


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
    dep_path = Path(dependency_path)
    assert_path_valid(config_path)
    assert_path_valid(model_path)
    assert_path_valid(dep_path)
    console = Console()
    console.print(
        f"[bold cyan]Starting FINN build with config {config_path.name} on model {model_path.name}"
    )
    console.print("[bold cyan]Setting up the FINN environment...[/bold cyan]")
    prepare_finn(DEFAULT_ENVVAR_CONFIG, config_path, dep_path, not no_local_temps, num_workers)
    console.print("[bold green]Done![/bold green]")
    console.print("[bold cyan]Creating dataflow build config...[/bold cyan]")
    dfbc = None
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

    console.rule(
        f"[bold cyan]Running FINN with config[/bold cyan][bold orange1] "
        f"{config_path.name}[/bold orange1][bold cyan] on model [/bold cyan]"
        "[bold orange1]{model_path.name}[/bold orange1]"
    )
    build_dataflow_cfg(str(model_path), dfbc)


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
    console = Console()
    script_path = Path(script)
    dep_path = Path(dependency_path)
    assert_path_valid(script_path)
    assert_path_valid(dep_path)
    console.print("[bold cyan]Setting up the FINN environment...[/bold cyan]")
    prepare_finn(DEFAULT_ENVVAR_CONFIG, script_path, dep_path, not no_local_temps, num_workers)
    console.print("[bold green]Done![/bold green]")
    console.rule(
        f"[bold cyan]Starting script "
        f"[/bold cyan][bold orange1]{script_path.name}[/bold orange1]"
    )
    subprocess.run(f"python3 {script_path.name}", cwd=script_path.parent, shell=True)


@click.command
@click.option(
    "--variant",
    "-v",
    help="Which test to execute (quick, main, rtlsim, end2end, full)",
    default="quick",
    show_default=True,
)
@click.option("--dependency-path", "-d", default=str(DEFAULT_DEPS), show_default=True)
@click.option("--num-workers", "-n", default="auto", show_default=True)
def test(variant: str, dependency_path: str, num_workers: int) -> None:
    console = Console()
    prepare_finn(
        DEFAULT_ENVVAR_CONFIG, Path(), Path(dependency_path), local_temps=False, num_workers=-1
    )
    console.rule("RUNNING TESTS")
    match variant:
        case "quick":
            subprocess.run(
                f"pytest -m 'not "
                f"(vivado or slow or vitis or board or notebooks or bnn_pynq)' "
                f"--dist=loadfile -n {num_workers}",
                shell=True,
            )
        case "main":
            subprocess.run(
                f"pytest -k 'not (rtlsim or end2end)' --dist=loadfile -n {num_workers}",
                shell=True,
            )
        case "rtlsim":
            subprocess.run(f"pytest -k rtlsim --workers {num_workers}", shell=True)
        case "end2end":
            subprocess.run("pytest -k end2end", shell=True)
        case "full":
            subprocess.run(
                f"pytest -k 'not (rtlsim or end2end)' --dist=loadfile -n {num_workers}",
                shell=True,
            )
            subprocess.run(f"pytest -k rtlsim --workers {num_workers}", shell=True)
            subprocess.run("pytest -k end2end", shell=True)
        case "brevitas":
            console.print("[bold green]Brevitas tests...[/bold green]")
            subprocess.run("pytest -k brevitas_export", shell=True)


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
    main_group.add_command(test)
    main_group.add_command(run)
    main()

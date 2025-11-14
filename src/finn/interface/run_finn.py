"""Run FINN+."""
# ruff: noqa: PIE790, ARG001
from __future__ import annotations

import click
import inspect
import json
import os
import rich
import shlex
import subprocess
import sys
import yaml
from pathlib import Path
from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt
from typing import TYPE_CHECKING, Any, cast

import finn.util.settings
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    LogLevel,
    ShellFlowType,
    default_build_dataflow_steps,
)
from finn.interface import IS_POSIX
from finn.interface.interface_utils import (
    NullablePath,
    error,
    resolve_module_path,
    set_synthesis_tools_paths,
    status,
    warning,
)
from finn.interface.manage_deps import DependencyUpdater
from finn.interface.manage_tests import run_test
from finn.interface.settings import FINNSettings
from finn.util.exception import FINNUserError, FINNValidationError

if TYPE_CHECKING:
    from collections.abc import Callable


def output(f: Callable) -> Callable[..., Any]:
    """Add a click parameter named --output (-o) that defaults to
    None if the param is empty, and a path otherwise."""  # noqa
    return click.option("--output", "-o", "output", default="", type=NullablePath())(f)


def finn_deps(f: Callable) -> Callable[..., Any]:
    """Add a click parameter named --dependency-path (-d) (finn_deps) that defaults to
    None if the param is empty, and a path otherwise."""  # noqa
    return click.option("--dependency-path", "-d", "finn_deps", default="", type=NullablePath())(f)


def finn_deps_definitions(f: Callable) -> Callable[..., Any]:
    """Add a click parameter named --dependency-definitions (-D) (finn_deps_definitions) that defaults to
    None if the param is empty, and a path otherwise."""  # noqa
    return click.option(
        "--dependency-definitions", "-D", "finn_deps_definitions", default="", type=NullablePath()
    )(f)


def finn_build_dir(f: Callable) -> Callable[..., Any]:
    """Add a click parameter named --build-path (-b) (finn_build_dir) that defaults to
    None if the param is empty, and a path otherwise."""  # noqa
    return click.option(
        "--build-path",
        "-b",
        "finn_build_dir",
        help="Specify a build temp path of your choice",
        default="",
        type=NullablePath(),
    )(f)


def flow_config(f: Callable) -> Callable[..., Any]:
    """Add a click parameter named config (type pathlib.Path)."""
    return click.argument(
        "flow_config",
        type=click.Path(
            exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
        ),
    )(f)


def model(f: Callable) -> Callable[..., Any]:
    """Add a click parameter named model (type pathlib.Path)."""
    return click.argument(
        "model",
        required=False,
        type=click.Path(
            exists=True, file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
        ),
    )(f)


def verify_input(f: Callable) -> Callable[..., Any]:
    """Add a click parameter named verify_input (type NullablePath)."""
    return click.option(
        "--verify-input",
        default="",
        help="Path to .npy  file that will be used as the input for verification.",
        type=NullablePath(),
    )(f)


def verify_output(f: Callable) -> Callable[..., Any]:
    """Add a click parameter named verify_output (type NullablePath)."""
    return click.option(
        "--verify-output",
        default="",
        help="Path to .npy  file that will be used as the expected output for verification.",
        type=NullablePath(),
    )(f)


def num_default_workers(f: Callable) -> Callable[..., Any]:
    """Add a click parameter called --num-workers (-n) (num_default_workers). Defaults to -1."""
    return click.option(
        "--num-workers",
        "-n",
        "num_default_workers",
        help="Number of parallel workers for FINN to use. When -1, automatically use 75% of cores",
        default=-1,
        show_default=True,
    )(f)


def skip_dep_update(f: Callable) -> Callable[..., Any]:
    """Add a click parameter called --skip-dep-update (-s). Defaults to False."""
    return click.option(
        "--skip-dep-update",
        "-s",
        is_flag=True,
        help="Whether to skip the dependency update. Can be changed in settings via"
        "AUTOMATIC_DEPENDENCY_UPDATES: false",
    )(f)


def accept_defaults(f: Callable) -> Callable[..., Any]:
    """Add a click parameter called --accept-defaults. Defaults to False."""
    return click.option(
        "--accept-defaults",
        is_flag=True,
        help="If set, skip the setup wizard in case that no settings files were found.",
    )(f)


def run_flow_wizard() -> None:
    """Interactively create a flow config with the user and save it."""
    dfbc = DataflowBuildConfig()
    console = Console()
    console.clear()
    console.print("[bold green]Welcome to FINN+s configuration wizard.[/bold green]\n")
    console.print("[bold]Core[/bold]")
    console.rule()
    output_path = Prompt.ask(
        "Where do you want to store the results of the flow, "
        "relative to the configuration file?",
        default="build_output",
    )
    dfbc.output_dir = output_path
    if dfbc.output_dir == "":
        console.print("[red]Please specify a valid path for the output directory[/red]")
        sys.exit(1)
    console.print()
    flow = IntPrompt.ask(
        "How much of the flow do you want to execute?\n"
        "1) Run until nodes are folded and first estimates are available\n"
        "2) Run until a stitched IP of synthesized cores is available\n"
        "3) Run until a full accelerator bitstream is available. (This will take "
        "a significant amount of time)",
        default=3,
        choices=["1", "2", "3"],
    )
    final_step_lookup = {
        1: "step_generate_estimate_reports",
        2: "step_measure_rtlsim_performance",
        3: "step_deployment_package",
    }
    dfbc.steps = default_build_dataflow_steps[
        : default_build_dataflow_steps.index(final_step_lookup[flow]) + 1
    ]
    console.print()
    loglevel = Prompt.ask(
        "What log level do you want to see during execution? (every log level "
        "includes all previous levels, for example DEBUG includes INFO)? ",
        default="WARNING",
        choices=["ERROR", "WARNING", "INFO", "DEBUG"],
    )
    dfbc.console_log_level = LogLevel(loglevel)
    console.print()
    dfbc.verbose = Confirm.ask("Do you want verbose output?", default=False)
    console.print()
    console.print("[bold]FPGA Settings[/bold]")
    console.rule()
    select_board = Confirm.ask(
        "Do you want to specify a board, or the FPGA part directly? (yes=board)"
    )
    if select_board:
        dfbc.board = Prompt.ask("Board")
    else:
        dfbc.fpga_part = Prompt.ask("FPGA Part")
    console.print()
    shell = Prompt.ask(
        "Which shell type would you like to use? (can be left empty, "
        "if no bitstream will be generated)",
        choices=["alveo", "zynq"] + ([""] if flow != 3 else []),
    )
    if shell == "alveo":
        dfbc.shell_flow_type = ShellFlowType.VITIS_ALVEO
    elif shell == "zynq":
        dfbc.shell_flow_type = ShellFlowType.VIVADO_ZYNQ
    console.print()

    clk_ns_str = Prompt.ask(
        "What clock frequency should Vivado target? (specify in ns or MHz: e.g. 7.0ns or 125mhz)"
    )
    if "ns" in clk_ns_str:
        dfbc.synth_clk_period_ns = float(clk_ns_str.replace("ns", ""))
    elif "mhz" in clk_ns_str:
        dfbc.synth_clk_period_ns = 1000.0 / float(clk_ns_str.replace("mhz", ""))
    else:
        console.print(
            "[red]Unknown unit. Please specify according to the instructions above![/red]"
        )
        sys.exit(1)
    console.print("\n[bold]Accelerator Settings[/bold]")
    console.rule()
    fps = IntPrompt.ask(
        "What target FPS should the accelerator achieve? (Leave at 0 to omit)", default=0
    )
    if fps != 0:
        dfbc.target_fps = fps
    console.print()
    dfbc.mvau_wwidth_max = IntPrompt.ask(
        "What mvau_wwidth_max value should the accelerator have?", default=dfbc.mvau_wwidth_max
    )
    console.print()
    console.print()
    console.rule()
    path = Path(
        Prompt.ask("Where do you want to save this flow configuration?", default="cfg.yaml")
    )
    with path.open("w+") as f:
        f.write(dfbc.to_yaml())
    console.print(
        "[green]You flow configuration has been saved. There are many more options to be set, "
        "this was just the minimal setup. Feel free to adjust the config further.[/green]"
    )


def run_setup_wizard(settings: FINNSettings) -> None:
    """Interactively ask the user to confirm / change / reject the default settings if no
    settings file was found. Saves the edited settings.
    IMPORTANT: This method does not check whether the file exists, or if the passed settings
    contain the default values. It simply runs the wizard without questions."""
    console = Console()
    console.clear()
    console.print(
        "[bold green]Welcome to FINN+.\nIt seems that you don't have a "
        "settings.yaml file yet. If this "
        "is on purpose, restart FINN+ with --accept-defaults to skip the "
        "wizard and use the predefined "
        "default values. Otherwise follow the next instructions.[/bold green]\n"
    )
    do_setup = Confirm.ask("Continue?")
    if not do_setup:
        return
    console.clear()
    console.print(
        "[bold]FINN_BUILD_DIR[/bold] stores the location of the path at "
        "which FINN puts temporary files "
        "generated during the build. If a relative path is given, the "
        "path is searched for from the provided "
        "model path.\n"
    )
    settings.finn_build_dir = Prompt.ask("FINN_BUILD_DIR", default="./FINN_TMP")
    console.clear()
    console.print(
        "[bold]FINN_DEPS[/bold] points to the location of the FINN dependency "
        "directory. If relative, this path is "
        "searched for from the root of the used FINN repository / installation.\n"
    )
    settings.finn_deps = Prompt.ask("FINN_DEPS", default=str(settings.finn_deps))
    console.clear()
    console.print(
        "[bold]FINN_DEPS_DEFINITIONS[/bold] points to the YAML file "
        "containing the dependency definitions. These are "
        "normally placed as external_dependencies.yaml in the FINN+ root.\n"
    )
    settings.finn_deps_definitions = Prompt.ask(
        "FINN_DEPS_DEFINITIONS", default=str(settings.finn_deps_definitions)
    )
    console.clear()
    console.print(
        "[bold]NUM_DEFAULT_WORKERS[/bold] specifies the number of "
        "workers to use in multithreaded contexts. By default "
        "this is set to 75% of your available CPU cores.\n"
    )
    settings.num_default_workers = IntPrompt.ask(
        "NUM_DEFAULT_WORKERS", default=settings.num_default_workers
    )
    console.clear()
    console.print(
        "[bold]AUTOMATIC_DEPENDENCY_UPDATES[/bold] specifies whether FINN+ "
        "will try to update the dependencies every time "
        "it is run. For fast iteration it is recommended to be turned off, but on otherwise.\n"
    )
    settings.automatic_dependency_updates = Confirm.ask(
        "AUTOMATIC_DEPENDENCY_UPDATES?", default=settings.automatic_dependency_updates
    )
    console.clear()
    console.print(
        "[bold]DEPS_GIT_TIMEOUT[/bold] specifies how many seconds the "
        "update waits until a Git dependency is fetched.\n"
    )
    settings.deps_git_timeout = IntPrompt.ask("DEPS_GIT_TIMEOUT", default=settings.deps_git_timeout)
    console.clear()
    console.print(
        "[italic]Some other settings were automatically inferred from your setup. "
        "They are listed below. Please check if they are correct. "
        "If not, these can be modified in the afterwards generated settings file\n"
    )
    console.print(f"[bold]FINN_CUSTOM_HLS[/bold]: {settings.finn_custom_hls}")
    console.print(f"[bold]FINN_NOTEBOOKS[/bold]: {settings.finn_notebooks}")
    console.print(f"[bold]FINN_RTLLIB[/bold]: {settings.finn_rtllib}")
    console.print(f"[bold]FINN_QNNDATA[/bold]: {settings.finn_qnn_data}")
    console.print(f"[bold]FINN_TESTS[/bold]: {settings.finn_tests}")
    console.print(
        "\n\n[bold green]Please check your edited settings and confirm them.[/bold green]"
    )
    console.print(f"[bold]FINN_BUILD_DIR[/bold]: {settings.finn_build_dir}")
    console.print(f"[bold]FINN_DEPS[/bold]: {settings.finn_deps}")
    console.print(f"[bold]FINN_DEPS_DEFINITIONS[/bold]: {settings.finn_deps_definitions}")
    console.print(f"[bold]NUM_DEFAULT_WORKERS[/bold]: {settings.num_default_workers}")
    console.print(
        f"[bold]AUTOMATIC_DEPENDENCY_UPDATES[/bold]: {settings.automatic_dependency_updates}"
    )
    console.print(f"[bold]DEPS_GIT_TIMEOUT[/bold]: {settings.deps_git_timeout}")
    console.print()
    do_save = Confirm.ask(f"Do you want to save these settings (at {settings._settings_path})? ")
    if not do_save:
        console.print(
            "[bold orange3]The settings were not saved. Either restart the wizard,"
            " or pass --accept-defaults to FINN.[/bold orange3]"
        )
        return
    settings.save()
    console.print(
        "[bold green]Settings written. Please restart FINN+ for the changes "
        "to take effect.[/bold green]"
    )
    console.rule()


def prepare_finn(settings: FINNSettings, accept_defaults: bool) -> None:
    """Prepare FINN to run."""
    if not settings.settingsfile_exists() and not accept_defaults:
        run_setup_wizard(settings)
        sys.exit(0)
    status(f"[SETTINGS FILE] {settings._settings_path.absolute()}")  # noqa
    status(f"[FINN BUILD DIRECTORY] {settings.finn_build_dir}")
    status(f"[DEPDENDENCY PATH] {settings.finn_deps}")
    status(f"[DEPDENDENCY DEFINITIONS PATH] {settings.finn_deps_definitions}")
    status(f"[NUM WORKERS] {settings.num_default_workers}")
    finn.util.settings._SETTINGS = settings
    if not settings.settingsfile_exists():
        warning("Settings file does not exist yet. Creating file now.")
        settings.save()
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = ""

    # Update / Install all dependencies
    if settings.automatic_dependency_updates:
        try:
            updater = DependencyUpdater(
                dependency_location=settings.finn_deps,
                dependency_definition_file=settings.finn_deps_definitions,
                git_timeout_s=settings.deps_git_timeout,
            )
            status(f"[EXTERNAL DEPENDENCY DEFINITION FILE] {updater.depfile.absolute()}")
            updater.update()
        except FINNUserError as e:
            error(f"FINN ERROR: {e}")
            sys.exit(1)
    else:
        warning("Skipping dependency updates!")

    # Check synthesis tools
    set_synthesis_tools_paths()

    os.environ["FINN_RTLLIB"] = resolve_module_path("finn-rtllib")
    os.environ["FINN_CUSTOM_HLS"] = resolve_module_path("custom_hls")
    os.environ["FINN_QNN_DATA"] = resolve_module_path("qnn-data")
    os.environ["FINN_NOTEBOOKS"] = resolve_module_path("notebooks")
    os.environ["FINN_TESTS"] = resolve_module_path("tests")


@click.group(
    help='Produce hardware designs from ONNX models. To get started use "finn build" '
    "(or run or auto) to start a FINN flow."
)
def main_group() -> None:
    """Main click group."""  # noqa
    pass


def get_function_args() -> dict:
    """Return key-values for the calling functions arguments. Filtered, so that no
    arguments accidentally get returned.
    """
    caller = inspect.stack()[1].frame
    args = inspect.getargvalues(caller).args
    d = {arg: caller.f_locals[arg] for arg in args}
    allowed = [
        "finn_deps",
        "finn_deps_definitions",
        "finn_build_dir",
        "num_default_workers",
        "flow_config",
    ]
    keys = list(d.keys())
    for key in keys:
        if key not in allowed:
            del d[key]
    return d


def read_model_path(flowconfig: Path) -> Path | None:
    """Try to read the model path from the flow config."""
    if flowconfig.suffix in [".yaml", ".yml"]:
        with flowconfig.open() as f:
            data: dict = yaml.load(f, yaml.Loader)
    elif flowconfig.suffix == ".json":
        data: dict = json.loads(flowconfig.read_text())
    else:
        raise FINNUserError(
            "Pass the flowconfig either as a YAML " "(.yaml, .yml) or JSON (.json) file!"
        )
    np = data.get("model_path")
    if np is None:
        return None
    p = Path(np)
    if p.is_absolute():
        return p
    return Path.cwd() / p


# The build function is separated from its command so that it can be reused
# without decorators in other commands
def _build(
    output: Path | None,
    accept_defaults: bool,
    finn_deps: Path | None,
    finn_deps_definitions: Path | None,
    finn_build_dir: Path | None,
    verify_input: Path | None,
    verify_output: Path | None,
    num_default_workers: int,
    skip_dep_update: bool,
    start: str,
    stop: str,
    flow_config: Path,
    model: Path | None,
) -> None:
    """Click command line option to build a FINN flow from a YAML config and an ONNX model."""
    # Check for the model path in the DFBC in case we dont pass one over CLI
    if model is None:
        mp = read_model_path(flow_config)
        if mp is None:
            error(
                "Please pass the model either via CLI "
                "or by setting model_path in your flow config!"
            )
            sys.exit(1)
        model = mp
    status(f"Starting FINN build with config {config.name} and model {model.name}!")
    settings = FINNSettings.init(
        auto_set_environment_vars=True,
        automatic_dependency_updates=not skip_dep_update,
        **get_function_args(),
    )
    prepare_finn(settings, accept_defaults)

    # Can import from finn now, since all deps are installed
    # and all environment variables are set correctly
    from finn.builder.build_dataflow import build_dataflow_cfg
    from finn.builder.build_dataflow_config import DataflowBuildConfig

    status("Creating dataflow build config...")
    dfbc: DataflowBuildConfig | None = None
    match flow_config.suffix:
        case ".yaml" | ".yml":
            with flow_config.open() as f:
                dfbc = DataflowBuildConfig.from_yaml(f.read())
        case ".json":
            with flow_config.open() as f:
                dfbc = DataflowBuildConfig.from_json(f.read())
        case _:
            error(f"Unknown config file type: {config.name}. Valid formats are: .json, .yml, .yaml")
            sys.exit(1)
    if dfbc is None:
        error("Failed to generate dataflow build config!")
        sys.exit(1)

    # Append the model path
    dfbc.model_path = model

    # Set start and stop steps
    if dfbc.start_step is None and start != "":
        dfbc.start_step = start
    if dfbc.stop_step is None and stop != "":
        dfbc.stop_step = stop

    # Set output directory to where the config lies, not where FINN lies
    if output is not None:
        dfbc.output_dir = output
    if not Path(dfbc.output_dir).is_absolute():
        dfbc.output_dir = str((flow_config.parent / dfbc.output_dir).absolute())
    status(f"Output directory is {dfbc.output_dir}")

    # Set verification steps
    # Override paths to verification input/output if specified
    if verify_input is not None:
        dfbc.verify_input_npy = verify_input

    if verify_output is not None:
        dfbc.verify_expected_output_npy = verify_output

    # Add path of config to sys.path so that custom steps can be found
    sys.path.append(str(flow_config.parent.absolute()))

    Console().rule(
        f"[bold cyan]Running FINN with config[/bold cyan][bold orange1] "
        f"{config.name}[/bold orange1][bold cyan] on model [/bold cyan]"
        f"[bold orange1]{model.name}[/bold orange1]"
    )
    build_dataflow_cfg(str(model), dfbc)


@click.command(help="Build a hardware design")
@output
@accept_defaults
@finn_deps
@finn_deps_definitions
@finn_build_dir
@flow_config
@model
@verify_input
@verify_output
@num_default_workers
@skip_dep_update
@click.option(
    "--start",
    default="",
    help="If no start_step is given in the dataflow build config, "
    "this starts the flow from the given step.",
)
@click.option(
    "--stop",
    default="",
    help="If no stop_step is given in the dataflow build config, "
    "this stops the flow at the given step.",
)
def build(
    output: Path | None,
    accept_defaults: bool,
    finn_deps: Path | None,
    finn_deps_definitions: Path | None,
    finn_build_dir: Path | None,
    verify_input: Path | None,
    verify_output: Path | None,
    num_default_workers: int,
    skip_dep_update: bool,
    start: str,
    stop: str,
    flow_config: Path,
    model: Path | None,
) -> None:
    """Launch a FINN hardware build."""
    _build(
        output,
        accept_defaults,
        finn_deps,
        finn_deps_definitions,
        finn_build_dir,
        verify_input,
        verify_output,
        num_default_workers,
        skip_dep_update,
        start,
        stop,
        flow_config,
        model,
    )


@click.command(help="Run a script in a FINN environment", deprecated=True)
@accept_defaults
@finn_deps
@finn_deps_definitions
@finn_build_dir
@num_default_workers
@skip_dep_update
@click.argument(
    "script",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, executable=True, path_type=Path),
)
def run(
    accept_defaults: bool,
    finn_deps: Path | None,
    finn_deps_definitions: Path | None,
    finn_build_dir: Path | None,
    skip_dep_update: bool,
    num_workers: int,
    script: Path,
) -> None:
    """Click command line option to run a script in a FINN+ context.

    Can be used for backwards compability with old FINN build flows.
    """
    script = script.expanduser()
    settings = FINNSettings.init(
        auto_set_environment_vars=True,
        automatic_dependency_updates=not skip_dep_update,
        flow_config=script,
        **get_function_args(),
    )
    prepare_finn(settings, accept_defaults)
    Console().rule(
        f"[bold cyan]Starting script [/bold cyan][bold orange1]{script.name}[/bold orange1]"
    )
    subprocess.run(
        shlex.split(f"{sys.executable} {script.name}", posix=IS_POSIX), cwd=script.parent
    )


@click.command(help="Best effort to automatically start FINN without further configuration.")
def auto() -> None:
    flow_config: Path | None = None
    model: Path | None = None

    # Search a configuration
    files: list[Path] = list(Path.cwd().iterdir())
    potential_configs = [p for p in files if p.suffix in [".yaml", ".yml", ".json"]]
    if len(potential_configs) == 0:
        error("Could not find a suitable configuration file (YAML or JSON).")
        sys.exit(1)
    for candidate in [
        "cfg.yaml",
        "cfg.yml",
        "cfg.json",
        "config.yaml",
        "config.yml",
        "config.json",
    ]:
        if candidate in [p.name for p in potential_configs]:
            flow_config = cast("Path", Path.cwd() / candidate)
    if flow_config is None:
        flow_config = potential_configs[0]
    status(f"Trying to use {flow_config} as a flow configuration file")

    # Search the model
    mp = read_model_path(flow_config)  # type: ignore
    if mp is not None:
        model = mp
    else:
        potential_models: list[Path] = [p for p in files if p.suffix == ".onnx"]
        if len(potential_models) == 0:
            error("No ONNX files found in this directory.")
            sys.exit(1)
        if "model.onnx" in [p.name for p in potential_models]:
            model = Path.cwd() / "model.onnx"
        else:
            model = potential_models[0]
    status(f"Trying to use {model} as a model file\n\n")
    _build(None, False, None, None, None, None, None, -1, False, "", "", flow_config, model)


@click.group(help="Run setup wizards for various tasks.")
def wizard() -> None:
    pass


@click.command(name="flow")
def flow_wizard() -> None:
    run_flow_wizard()


@click.command(name="config")
def config_wizard() -> None:
    settings = FINNSettings.init(
        auto_set_environment_vars=True,
        automatic_dependency_updates=not skip_dep_update,
        **get_function_args(),
        flow_config=Path(),
        finn_build_dir=Path("FINN_TMP"),
    )
    run_setup_wizard(settings)


@click.command(help="Run a given benchmark configuration.")
@click.option("--bench_config", help="Name or path of experiment configuration file", required=True)
@finn_deps
@finn_deps_definitions
@finn_build_dir
@num_default_workers
def bench(
    bench_config: str,
    finn_deps: Path | None,
    finn_deps_definitions: Path | None,
    num_default_workers: int,
    finn_build_dir: Path | None,
) -> None:
    """Run a benchmark."""
    settings = FINNSettings.init(
        auto_set_environment_vars=True,
        automatic_dependency_updates=not skip_dep_update,
        **get_function_args(),
    )
    prepare_finn(settings, True)
    Console().rule("RUNNING BENCHMARK")

    # Late import because we need prepare_finn to setup remaining dependencies first
    from finn.benchmarking.bench import start_bench_run

    exit_code = start_bench_run(bench_config)
    sys.exit(exit_code)


@click.command(help="Run a given test. Uses /tmp/FINN_TMP as the temporary file location")
@finn_deps
@finn_deps_definitions
@finn_build_dir
@num_default_workers
@click.option(
    "--variant",
    "-v",
    help="Which test to execute (quick, quicktest_ci, full_ci)",
    default="quick",
    show_default=True,
    type=click.Choice(["quick", "quicktest_ci", "full_ci"]),
)
@click.option("--num-test-workers", "-t", default="auto", show_default=True)
def test(
    variant: str,
    finn_deps: Path | None,
    finn_deps_definitions: Path | None,
    num_default_workers: int,
    num_test_workers: str,
    build_path: Path | None,
) -> None:
    """Run a selected subset of the FINN(+) testsuite."""
    settings = FINNSettings.init(
        auto_set_environment_vars=True,
        automatic_dependency_updates=not skip_dep_update,
        **get_function_args(),
    )
    if not settings.finn_build_dir.exists():
        settings.finn_build_dir = Path("/tmp/FINN_TEST_BUILD_DIR")
    prepare_finn(settings, True)
    status(f"Using {num_test_workers} test workers")
    Console().rule("RUNNING TESTS")
    run_test(variant, num_test_workers)


@click.group(help="Dependency management")
def deps() -> None:
    """Click group collecting depenency related commands."""
    pass


@click.command(help="Update or install dependencies to the given path")
@accept_defaults
@finn_deps
@finn_deps_definitions
def update(
    accept_defaults: bool, finn_deps: Path | None, finn_deps_definitions: Path | None
) -> None:
    """Update all FINN+ dependencies and then exit."""
    settings = FINNSettings.init(
        auto_set_environment_vars=True,
        automatic_dependency_updates=True,
        flow_config=Path(),
        **get_function_args(),
    )
    prepare_finn(settings, accept_defaults)


@click.group(help="Manage FINN settings")
def config() -> None:
    """Click group for config related commands."""
    # TODO: Config remove?
    pass


def _command_get_settings() -> FINNSettings:
    """Return a settings instance for use in config commands."""
    settings = FINNSettings.init(
        auto_set_environment_vars=True, automatic_dependency_updates=False, flow_config=Path()
    )
    prepare_finn(settings, True)
    if not settings.settingsfile_exists():
        warning("Could not resolve settings file.")
        sys.exit(1)
    return settings


@click.command("list", help="List the settings files contents")
def config_list() -> None:
    """List all settings found in the current settings file."""
    rich.print(_command_get_settings())


@click.command("get", help="Get a specific key from the settings")
@click.argument("key")
def config_get(key: str) -> None:
    """Get the value of a settings key."""
    settings = _command_get_settings().model_dump()
    if key not in settings.keys():
        error(f"Key {key} could not be found in the settings file!")
        sys.exit(1)
    Console().print(f"[blue]{key}[/blue]: {settings[key]}")


@click.command("set", help="Set a key to a given value")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a setting key to a given value."""
    # settings_original = _command_get_settings()
    # TODO: Fix
    # settings_original.model_copy(update={key: value}, deep=True)
    # settings_original.save()


@click.command("help", help="List all known settings fields and their purpose")
def config_help() -> None:
    """Print a table with all known settings keys and their purpose."""
    # TODO
    rich.print(_command_get_settings().get_settings_keys())


@click.command(
    "create",
    help="Create a template settings file. If one exists at the given path, "
    "its overwritten. Please enter a directory, no filename",
)
@click.argument("path", default="~/.finn/")
def config_create(path: str) -> None:
    """Create a template config at the `<given path>/settings.yaml`. Uses the default values."""
    p = Path(path).expanduser().absolute()
    if not p.exists():
        error("The given path does not seem to exist!")
        sys.exit(1)
    if p.suffix != "":
        error("Please specify a path to a directory, not a file!")
        sys.exit(1)
    p = p / "settings.yaml"
    if p.exists():
        status(f"A settings file already exists at {p}")
        return
    settings = FINNSettings.init(override_settings_path=p, flow_config=Path())
    settings.save()
    if not p.exists():
        error(f"Failed to create config template at {p}!")
        sys.exit(1)
    status(f"File written to {p}")


def main() -> None:
    """Clicks entrypoint function."""
    config.add_command(config_list)
    config.add_command(config_create)
    config.add_command(config_get)
    config.add_command(config_set)
    config.add_command(config_help)
    deps.add_command(update)
    wizard.add_command(flow_wizard)
    wizard.add_command(config_wizard)
    main_group.add_command(auto)
    main_group.add_command(wizard)
    main_group.add_command(config)
    main_group.add_command(deps)
    main_group.add_command(build)
    main_group.add_command(bench)
    main_group.add_command(test)
    main_group.add_command(run)
    try:
        main_group()
    except FINNValidationError as e:
        error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()

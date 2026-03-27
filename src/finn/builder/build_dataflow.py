# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""FINN dataflow build system.

This module provides the main build infrastructure for converting ONNX models
to FINN dataflow accelerators. It handles step resolution, logging, error handling,
and the complete build pipeline from ONNX input to hardware accelerator output.
"""

import datetime
import importlib
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

import pdb  # isort: split
import sys
import time
from qonnx.core.modelwrapper import ModelWrapper
from rich import print as rprint
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import Traceback
from typing import Any, TextIO

import finn.util.logging
from finn.util.exception import (
    FINNConfigurationError,
    FINNDataflowError,
    FINNError,
    FINNUserError,
)
from finn.util.exception_snapshot import snapshot_on_exception
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    LogLevel,
    default_build_dataflow_steps,
    to_logging_level,
)
from finn.builder.build_dataflow_steps import build_dataflow_step_lookup
from finn.util.basic import get_vivado_root
from finn.util.logging import log
from finn.util.settings import get_settings


def get_logfile_path(cfg: DataflowBuildConfig) -> str:
    """Return the path to the logfile in the build dir."""
    return str(Path(cfg.output_dir) / "build_dataflow.log")


# adapted from https://stackoverflow.com/a/39215961
class PrintLogger:
    """Custom stream handler that writes to both console and log file with timestamps."""

    def __init__(self, logger: logging.Logger, level: int, originalstream: TextIO | Any) -> None:
        """Initialize the print logger with logger, level, and original stream."""
        self.logger = logger
        self.level = level
        self.console = originalstream
        self.linebuf = ""

    def write(self, buf: str) -> None:
        """Write buffer content to both logger and console with timestamp."""
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())
            timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
            self.console.write(f"[{timestamp}] " + line + "\n")

    def flush(self) -> None:
        """Flush the console stream."""
        self.console.flush()


BuildStep = Callable[[ModelWrapper, DataflowBuildConfig], ModelWrapper]


def resolve_build_steps(cfg: DataflowBuildConfig, partial: bool = True) -> list[BuildStep]:
    """Resolve build step names to callable functions.

    Converts string step names to callable functions by looking up in the step registry,
    importing from modules, or using direct callable objects. Supports partial execution
    between start_step and stop_step if specified in config.

    Args:
        cfg: Build configuration containing step definitions
        partial: If True, respect start_step/stop_step boundaries

    Returns:
        List of callable step functions ready for execution
    """
    steps = cfg.steps
    if steps is None:
        steps = default_build_dataflow_steps
    steps_as_fxns = []
    for transform_step in steps:
        if type(transform_step) is str:
            # lookup step function from step name
            if transform_step in build_dataflow_step_lookup.keys():
                steps_as_fxns.append(build_dataflow_step_lookup[transform_step])
            else:
                if "." not in transform_step:
                    if transform_step not in globals().keys():
                        msg = (
                            f"Step {transform_step} is not a default step, not in globals() "
                            "and not an importable name!"
                        )
                        raise FINNConfigurationError(msg)
                    else:  # noqa
                        fxn_step = globals()[transform_step]
                        if not callable(fxn_step):
                            msg = (
                                f"Step {transform_step} was resolved in globals(), but is "
                                "not callable object. If the name was already in use, consider "
                                "moving your custom step into it's own module and importing it "
                                "via yourmodule.yourstep!"
                            )
                            raise FINNConfigurationError(msg)
                        steps_as_fxns.append(fxn_step)
                        continue
                else:
                    split_step = transform_step.split(".")
                    module_path, fxn_step_name = split_step[:-1], split_step[-1]
                    try:
                        imported_module = importlib.import_module(".".join(module_path))
                        fxn_step = getattr(imported_module, fxn_step_name)
                        if callable(fxn_step):
                            steps_as_fxns.append(fxn_step)
                            continue
                        else:  # noqa
                            msg = (
                                f"Could import custom step module, but final name is not a "
                                f"callable object. Path was {transform_step}"
                            )
                            raise FINNConfigurationError(msg)
                    except ModuleNotFoundError as mnf:
                        msg = (
                            f"Could not resolve build step: {transform_step}. "
                            "The given step is neither importable nor a default step. "
                            "This can also happen if an error occurs while importing your module. "
                        )
                        raise FINNConfigurationError(msg) from mnf
        elif callable(transform_step):
            # treat step as function to be called as-is
            steps_as_fxns.append(transform_step)
        else:
            raise FINNConfigurationError("Could not resolve build step: " + str(transform_step))
    if partial:
        step_names = [x.__name__ for x in steps_as_fxns]
        start_ind = 0 if cfg.start_step is None else step_names.index(cfg.start_step)
        stop_ind = len(step_names) - 1 if cfg.stop_step is None else step_names.index(cfg.stop_step)
        steps_as_fxns = steps_as_fxns[start_ind : (stop_ind + 1)]

    # Add the exception snapshot decorator if needed
    return [
        snapshot_on_exception(
            snapshot_finn=False, snapshot_config=True, snapshot_buildlog=True, build_dir_prefix=None
        )(transform_step)
        if (
            cfg.enable_exception_snapshots
            and "snapshot_on_exception_enabled" not in dir(transform_step)
        )
        else transform_step
        for transform_step in steps_as_fxns
    ]


def resolve_step_filename(step_name: str, cfg: DataflowBuildConfig, step_delta: int = 0) -> Path:
    """Resolve the intermediate model filename for a given build step.

    Args:
        step_name: Name of the build step
        cfg: Build configuration
        step_delta: Offset from the step (0=current, -1=previous, +1=next)

    Returns:
        Path to the intermediate model file for the specified step
    """
    # Find the correct file
    step_names = [x.__name__ for x in resolve_build_steps(cfg, partial=False)]
    if step_name not in step_names:
        raise FINNConfigurationError(
            f"Cannot restart from unknown step '{step_name}'. Your flow configuration "
            f"contains the following steps: \n\t" + "\n\t".join(step_names)
        )
    try:
        step_index_original = step_names.index(step_name)
        step_no = step_index_original + step_delta
    except ValueError:
        raise FINNUserError(
            f"Step filename could not be resolved. Step "
            f"{step_name} was not found in your flow configuration"
        )
    if step_no < 0 and step_delta != 0 and step_names.index(step_name) == 0:
        # We simply assume that --start was given, since this method is only called in that case
        # TODO: Move the error (check) to the creation of the modelwrapper
        raise FINNUserError(
            f"Could not resolve the model filename for a step before "
            f"'{step_name}' because it is the first step in your flow "
            f"config. To start FINN from the first step, simply run it "
            f"without the '--start' parameter."
        )
    if step_no < 0 or step_no >= len(step_names):
        raise FINNDataflowError(
            f"Invalid combination of step index ({step_index_original}) and "
            f"delta ({step_delta}): {step_no} (must be in the range from 0 "
            f"to {len(step_names)-1})"
        )

    # Return if it exists
    filename = Path(cfg.output_dir) / "intermediate_models" / f"{step_names[step_no]}.onnx"
    if not filename.exists():
        raise FINNConfigurationError(
            f"Expected model file at {filename} to start from step "
            f"{step_name}, but could not find it!"
        )
    return filename


def setup_logging(cfg: DataflowBuildConfig) -> logging.Logger:
    """Configure logging for the build process.

    Sets up file logging, console mirroring, and rich console handlers
    based on the build configuration settings.

    Args:
        cfg: Build configuration with logging settings

    Returns:
        Configured logger instance
    """
    # Set up global logger, the force=True has the following effects:
    # - If multiple build are run in a row, the log file will be re-created for each,
    #   which is needed if the file was deleted/moved or the output dir changed
    # - In a PyTest session, this logger will replace the PyTest log handlers, so logs
    #   (+ captured warnings!) will end up in the log file instead of being collected by PyTest
    logpath = get_logfile_path(cfg)
    if cfg.verbose:
        loglevel = logging.DEBUG
        detailed_location = " %(pathname)s:%(lineno)d: "
    else:
        loglevel = logging.INFO
        detailed_location = ""
    logging.basicConfig(
        level=loglevel,
        format=f"[%(asctime)s]%(levelname)s:{detailed_location}%(message)s",
        filename=logpath,
        filemode="w",
        force=True,
    )

    # Capture all warnings.warn calls of qonnx, ...
    logging.captureWarnings(True)

    # Mirror stdout and stderr to log
    log = logging.getLogger("build_dataflow")
    log.setLevel(loglevel)

    # Redirect stdout/stderr
    # Prevent rediricting stdout/sterr multiple times
    if not isinstance(sys.stdout, PrintLogger):
        sys.stdout = PrintLogger(log, logging.INFO, sys.stdout)
        sys.stderr = PrintLogger(log, logging.ERROR, sys.stderr)
    console = Console(file=sys.stdout.console)
    finn.util.logging.set_console(console)

    # Mirror a configurable log level to console (default = ERROR)
    if cfg.console_log_level != LogLevel.NONE:
        consolehandler = RichHandler(
            show_time=True, log_time_format="[%Y-%m-%d %H:%M:%S]", show_path=False, console=console
        )
        consolehandler.setLevel(to_logging_level(cfg.console_log_level))
        logging.getLogger().addHandler(consolehandler)
    return log


def exit_buildflow(
    cfg: DataflowBuildConfig, time_per_step: dict[str, float] | None = None, exit_code: int = 0
) -> int:
    """Create metadata_builder.json and time_per_step.json files with build results.

    Args:
        cfg: Build configuration
        time_per_step: Dictionary of step execution times
        exit_code: Build exit code (0=success, non-zero=failure)

    Returns:
        The provided exit code
    """
    # Generate metadata_builder.json
    metadata = {
        "status": "failed" if exit_code else "ok",
        "tool_version": Path(get_vivado_root()).name,
    }
    metadata_builder = Path(cfg.output_dir) / "report" / "metadata_builder.json"
    metadata_builder.write_text(json.dumps(metadata, indent=2))

    # Generate time_per_step.json
    time_per_step_json = Path(cfg.output_dir) / "report" / "time_per_step.json"
    if time_per_step is not None:
        time_per_step["total_build_time"] = sum(time_per_step.values())
        time_per_step_json.write_text(json.dumps(time_per_step, indent=2))
    return exit_code


def create_model_wrapper(model_filename: str, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Create a modelwrapper from the given config and filename. If a start-step
    is given, the ModelWrapper is constructed from a previous intermediate model.
    """
    if cfg.start_step is None:
        print(f"Building dataflow accelerator from {model_filename}")
        return ModelWrapper(model_filename)
    if model_filename != "":
        log.warning(
            "When using a start-step, FINN automatically searches "
            "for the correct model to use from previous runs, overwriting your "
            "passed model file (but still using it's path for the location of the "
            "temporary file directory, etc.). This behaviour might change "
            "in future versions!"
        )
    intermediate_model_filename = resolve_step_filename(cfg.start_step, cfg, -1)
    print(
        f"Building dataflow accelerator from intermediate"
        f" checkpoint {intermediate_model_filename}"
    )
    return ModelWrapper(str(intermediate_model_filename))


def build_dataflow_cfg(model_filename: str, cfg: DataflowBuildConfig) -> int:
    """Build a dataflow accelerator using the given configuration.

    Main entry point for building FINN dataflow accelerators. Handles step execution,
    logging, error handling, and intermediate model saving.

    Args:
        model_filename: ONNX model filename to build
        cfg: Build configuration specifying steps and options

    Returns:
        Exit code (0=success, non-zero=failure)
    """
    # Create the output directories
    output_dir = Path(cfg.output_dir)
    intermediate_model_dir = output_dir / "intermediate_models"
    intermediate_model_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    log = setup_logging(cfg)
    logfile = get_logfile_path(cfg)
    print(f"Intermediate outputs will be generated in {get_settings().finn_build_dir}")
    print(f"Final outputs will be generated in {cfg.output_dir}")
    print(f"Build log is at {logfile}")

    # Setup done, start build flow
    time_per_step: dict[str, float] = {}
    try:
        model = create_model_wrapper(model_filename, cfg)
        build_dataflow_steps: list[BuildStep] = resolve_build_steps(cfg)

        # Execute all steps
        for step_num, transform_step in enumerate(build_dataflow_steps):
            step_name = transform_step.__name__
            print(f"Running step: {step_name} [{step_num + 1}/{len(build_dataflow_steps)}]")

            # Run the step
            step_start = time.time()
            model = transform_step(model, cfg)
            step_end = time.time()
            time_per_step[step_name] = round(step_end - step_start)
            if cfg.save_intermediate_models:
                model.save(str(intermediate_model_dir / f"{step_name}.onnx"))

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Aborting...")
        return exit_buildflow(cfg, time_per_step, -1)

    except (Exception, FINNError) as e:
        # Re-raise exception if we are in a PyTest session so we don't miss it
        if "PYTEST_CURRENT_TEST" in os.environ:
            raise
        if issubclass(type(e), FINNUserError):
            # Handle FINN USER ERROR
            log.error(f"FINN ERROR: {e}")
        else:
            # Handle remaining errors (= FINN INTERNAL COMPILER ERROR)
            log.error(f"FINN INTERNAL COMPILER ERROR: {e}")

        # Print traceback for interal errors or if in debug mode
        if not issubclass(type(e), FINNUserError) or log.level == logging.DEBUG:
            # Restoring stdout and stderr
            if type(sys.stdout) is PrintLogger:
                sys.stdout = sys.stdout.console
            if type(sys.stderr) is PrintLogger:
                sys.stderr = sys.stderr.console

            # Print traceback both to console and logfile
            rprint(Traceback(show_locals=False))
            with Path(get_logfile_path(cfg)).open("a") as f:
                rprint(Traceback(show_locals=False), file=f)

            # Start postmortem debug if configured
            if cfg.enable_build_pdb_debug:
                pdb.post_mortem(e.__traceback__)

        return exit_buildflow(cfg, time_per_step, -1)
    return exit_buildflow(cfg, time_per_step, 0)

"""Here we organize FINN+`s exceptions and error handling. It also contains a decorator to
snapshot FINN+ when it crashes for debugging purposes.
"""
from __future__ import annotations

import functools
import inspect
import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from typing import Callable

import finn
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.util.basic import make_build_dir

"""
FINNError is the base class for all errors.
FINNUserError is a purely user-facing error that has nothing to do with FINNs internals
FINNInternalError is a compiler internal error

Every error should subclass FINNUserError or FINNInternalError
"""


class FINNError(Exception):
    """Base-class for all FINN exceptions. Useful to differentiate exceptions while catching."""

    def __init__(self, *args: object) -> None:
        """Create a new FINNError."""
        super().__init__(*args)


class FINNInternalError(FINNError):
    """Custom exception class for internal compiler errors."""

    def __init__(self, *args: object) -> None:
        """Create a new FINNInternalError."""
        super().__init__(*args)


class FINNUserError(FINNError):
    """Custom exception class which should be used to
    print errors without stacktraces if debug is disabled.
    """

    def __init__(self, *args: object) -> None:
        """Create a new FINNUserError."""
        super().__init__(*args)


class FINNConfigurationError(FINNUserError):
    """Error emitted if FINN is configured incorrectly."""

    def __init__(self, *args: object) -> None:
        """Create a new FINNConfigurationError."""
        super().__init__(*args)


class FINNDataflowError(FINNInternalError):
    """Errors regarding the dataflow, dataflow config, step resolution, etc."""

    def __init__(self, *args: object) -> None:
        """Create a new FINNDataflowError."""
        super().__init__(*args)


# Alias for a build flow step function apply function
StepFunction = Callable[[ModelWrapper, DataflowBuildConfig], ModelWrapper]


def snapshot_on_exception(
    snapshot_finn: bool = False,
    snapshot_config: bool = True,
    snapshot_model: bool = True,
    include_every_modelwrapper: bool = True,
    snapshot_buildlog: bool = True,
    snapshot_finn_envvars: bool = True,
    additional_envvars: list[str] | None = None,
    build_dir_prefix: str | None = None,
) -> Callable[[StepFunction], StepFunction]:
    """Apply this decorator to any step function with the signature
    ```
    @snapshot_on_exception()
    def step_...(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
            ...
    ```
    to, in case an exception is raised, snapshot the ONNX model directly after the crash, as
    well as a snapshot of FINN itself, as well as the build config and the dataflow build log.

    The items that can be snapshot are these:
    - FINN itself (the source tree)
    - The build config
    - The current ONNX model
        - (Optionally) every ONNX model in the current scope
    - The build log file
    - FINN specific environment variables
    - Other environment variables

    Everything is stored in a new directory in your configured output directory for the run, with a
    timestamp. For example: .../<cfg.output_dir>/crash_reports/crash_<date+time>_<random suffix>/

    For the ONNX model, the function first tries to find a ModelWrapper object called model in the
    scope where the exception was raised. If this is not found, the first object of type
    ModelWrapper is used. If this is not available, the ModelWrapper of the step function is used.
    Furthermore, if include_every_modelwrapper is set to True, every ModelWrapper object
    in the scope where the exception was raised will be saved as well.

    Keep in mind that when an exception is raised while working on a SDP submodel, the submodel, not
    the parent model, will be saved. The parent model can then be found in intermediate_models, if
    enabled.

    additional_envvars can be set to include environment variables that don't natively belong to
    FINN, but might help in debugging
    certain steps (for example LD_LIBRARY_PATH), into the snapshot.
    """
    if additional_envvars is None:
        additional_envvars = []

    def decorator(step: StepFunction) -> StepFunction:
        """Construct the snapshot_on_exception decorator."""

        @functools.wraps(step)
        def wrapped(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
            """Wrap the step function in the snapshot code."""
            try:
                return step(model, cfg)
            except Exception as e:
                date = datetime.today().strftime("%d-%m-%Y__%I-%M-%S")
                if build_dir_prefix is None:
                    prefix = f"crash_{date}_"
                else:
                    prefix = f"{build_dir_prefix}_{date}_"
                temp_path = Path(make_build_dir(prefix))
                crash_report_dir = Path(cfg.output_dir) / "crash_reports"
                if not crash_report_dir.exists():
                    crash_report_dir.mkdir(parents=True)
                path = crash_report_dir / temp_path.name
                temp_path.rename(path.absolute())

                if snapshot_model:
                    # Get the frame where the exception was raised, get it's frame object,
                    # and from it all locals, hopefully containing our ModelWrapper object
                    error_locals = inspect.trace()[-1][0].f_locals
                    modelwrappers = {
                        k: v for k, v in error_locals.items() if isinstance(v, ModelWrapper)
                    }
                    actual_model: ModelWrapper | None = None
                    if "model" in modelwrappers.keys():
                        actual_model = modelwrappers["model"]
                        actual_model.save(str(path / "model_exception_scope.onnx"))
                    elif len(modelwrappers.keys()) > 0:
                        actual_model = next(iter(modelwrappers.values()))
                        actual_model.save(str(path / "first_modelwrapper_exception_scope.onnx"))
                    else:
                        actual_model = model
                        model.save(str(path / "model_buildflow_step.onnx"))
                    if include_every_modelwrapper:
                        all_modelwrappers_path = path / "all_modelwrappers"
                        all_modelwrappers_path.mkdir()
                        for modelwrapper_name, modelwrapper in modelwrappers.items():
                            modelwrapper.save(
                                all_modelwrappers_path / (modelwrapper_name + ".onnx")
                            )
                    if actual_model is not None:
                        for node in model.graph.node:
                            if node.op_type == "StreamingDataflowPartition":
                                submodel = Path(getCustomOp(node).get_nodeattr("model"))
                                if submodel.exists():
                                    submodel_dir = Path(path / "submodels")
                                    submodel_dir.mkdir()
                                    shutil.copy(submodel, submodel_dir)
                if snapshot_config:
                    (path / "cfg.yaml").write_text(str(cfg.to_yaml()))
                if snapshot_finn:
                    finn_root = Path(finn.__file__).parent
                    shutil.copytree(finn_root, path / "finn")
                if snapshot_buildlog and cfg.output_dir is not None:
                    dataflow_log = (Path(cfg.output_dir) / "build_dataflow.log").read_text()
                    dataflow_log += traceback.format_exc()
                    (path / "build_dataflow.log").write_text(dataflow_log)
                if snapshot_finn_envvars:
                    env: dict[str, str] = {}
                    non_finn_envvars: list[str] = [*additional_envvars, "NUM_DEFAULT_WORKERS"]
                    for key in non_finn_envvars:
                        if key in os.environ.keys():
                            env[key] = os.environ[key]
                        else:
                            env[key] = f"Environment variable {key} was not found!"
                    env.update({k: v for k, v in os.environ.items() if k.startswith("FINN_")})
                    with (path / "finn_env_vars").open("w+") as f:
                        for k, v in env.items():
                            f.write(f"{k}={v}\n")
                raise e

        # Mark the function as already decorated
        # If automatic deocoration of all steps is enabled, this
        # avoids making multiple snapshots for steps that are
        # additionally manually marked
        wrapped.snapshot_on_exception_enabled = True  # type: ignore
        return wrapped

    return decorator

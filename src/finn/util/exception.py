from __future__ import annotations

import inspect
import shutil
from datetime import datetime
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
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
    """Base-class for FINN exceptions. Useful to differentiate exceptions while catching."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FINNInternalError(FINNError):
    """Custom exception class for internal compiler errors"""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FINNUserError(FINNError):
    """Custom exception class which should be used to
    print errors without stacktraces if debug is disabled."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FINNConfigurationError(FINNUserError):
    """Error emitted when FINN is configured incorrectly"""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FINNDataflowError(FINNInternalError):
    """Errors regarding the dataflow, dataflow config, step resolution, etc."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


# Alias for a build flow step function apply function
StepFunction = Callable[[ModelWrapper, DataflowBuildConfig], ModelWrapper]


def snapshot_on_exception(
    snapshot_finn: bool = False,
    snapshot_config: bool = True,
    snapshot_model: bool = True,
    snapshot_buildlog: bool = True,
) -> Callable[[StepFunction], StepFunction]:
    """Apply this decorator to any step function with the signature
    ```
    def step_...(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
            ...
    ```
    to, in case an exception is raised, snapshot the ONNX model directly after the crash, as
    well as a snapshot of FINN itself, as well as the build config and the dataflow build log"""

    def decorator(step: StepFunction) -> StepFunction:
        def wrapped(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
            try:
                return step(model, cfg)
            except Exception as e:
                date = datetime.today().strftime("%d_%m_%Y__%I_%M_%S")
                path = Path(make_build_dir(f"crash_{date}_"))
                if snapshot_model:
                    # Get the frame where the exception was raised, get it's frame object,
                    # and from it all locals, hopefully containing our ModelWrapper object
                    error_locals = inspect.trace()[-1][0].f_locals
                    modelwrappers = {
                        k: v for k, v in error_locals.items() if isinstance(v, ModelWrapper)
                    }
                    if "model" in modelwrappers.keys():
                        modelwrappers["model"].save(
                            str(path / "transformation_model_snapshot.onnx")
                        )
                    elif len(modelwrappers.keys()) > 0:
                        next(iter(modelwrappers.values())).save(
                            str(path / "likely_transformation_model_snapshot.onnx")
                        )
                    else:
                        model.save(str(path / "before_failed_transformation_model_snapshot.onnx"))
                if snapshot_config:
                    yaml = str(cfg.to_yaml())
                    with (path / "cfg.yaml").open("w+") as f:
                        f.write(yaml)
                if snapshot_finn:
                    finn_root = Path(finn.__file__).parent
                    shutil.copytree(finn_root, path / "finn")
                if snapshot_buildlog and cfg.output_dir is not None:
                    shutil.copy(
                        Path(cfg.output_dir) / "build_dataflow.log", path / "build_dataflow.log"
                    )
                raise e

        wrapped.__name__ = step.__name__
        return wrapped

    return decorator

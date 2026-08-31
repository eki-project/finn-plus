"""Dataset-specific accuracy validation for FINN-generated accelerators."""

from importlib import import_module
from typing import Any

#: Supported validation datasets, mapped to the module implementing their ``validate``
#: function. Modules are imported lazily so that a missing optional dataset dependency only
#: affects the dataset that needs it.
VALIDATION_DATASETS: dict[str, str] = {
    "mnist": "finn_plus_driver.validate.mnist",
    "cifar": "finn_plus_driver.validate.cifar",
    "imagenet": "finn_plus_driver.validate.imagenet",
    "radioml": "finn_plus_driver.validate.radioml",
    "unswnb15": "finn_plus_driver.validate.unswnb15",
}

__all__ = ["VALIDATION_DATASETS", "run_validate"]


def run_validate(validation_dataset: str, cls_inst: Any, *args: Any, **kwargs: Any) -> None:
    """Dispatch validation to the appropriate dataset-specific validate function.

    Args:
        validation_dataset: Name of the dataset to validate against.
        cls_inst: Driver instance used to run inference on the accelerator.
        *args: Positional arguments forwarded to the dataset's validate function.
        **kwargs: Keyword arguments forwarded to the dataset's validate function.
    """
    print(f"Running validation with Dataset: {validation_dataset}")
    print(f"Report directory: {kwargs.get('report_dir')}")

    if validation_dataset not in VALIDATION_DATASETS:
        print(f"WARNING: SKIPPING VALIDATION FOR UNKNOWN DATASET: {validation_dataset}")
        return

    module = import_module(VALIDATION_DATASETS[validation_dataset])
    module.validate(cls_inst, *args, **kwargs)

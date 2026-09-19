"""Validation script for CIFAR-10/100 datasets."""

import os
from dataset_loading import cifar

from finn_plus_driver.validate.common import ArrayDataset, run_validation, validation_kwargs


def validate(cls_inst, *args, **kwargs):
    """Run CIFAR dataset validation and report top-1 accuracy."""
    report_dir = kwargs.get("report_dir")
    dataset_path = kwargs.get("dataset_path", os.path.dirname(os.path.realpath(__file__)))
    # Dataset name "cifar" selects CIFAR-10, "cifar100" selects CIFAR-100 (fine labels)
    cifar10 = kwargs.get("validation_dataset") != "cifar100"

    trainx, trainy, testx, testy, valx, valy = cifar.load_cifar_data(
        dataset_path, download=True, one_hot=False, cifar10=cifar10
    )

    print("Starting validation..")
    dataset = ArrayDataset(testx, testy)
    run_validation(cls_inst, dataset, report_dir, **validation_kwargs(kwargs))

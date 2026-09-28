"""Validation script for the MNIST dataset."""

import os
from dataset_loading import mnist

from finn_plus_driver.validate.common import ArrayDataset, run_validation, validation_kwargs


def validate(cls_inst, *args, **kwargs):
    """Run MNIST validation and report top-1 accuracy."""
    report_dir = kwargs.get("report_dir")
    dataset_path = kwargs.get("dataset_path", os.path.dirname(os.path.realpath(__file__)))

    trainx, trainy, testx, testy, valx, valy = mnist.load_mnist_data(
        dataset_path, download=True, one_hot=False
    )

    print("Starting validation..")
    dataset = ArrayDataset(testx, testy)
    run_validation(cls_inst, dataset, report_dir, **validation_kwargs(kwargs))

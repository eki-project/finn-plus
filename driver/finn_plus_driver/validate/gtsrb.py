"""Validation script for the German Traffic Sign Recognition Benchmark (GTSRB) dataset."""

import numpy as np
import os

from finn_plus_driver.validate.common import ArrayDataset, run_validation, validation_kwargs

#: The 12630 images of the official test set, scaled to 32x32 pixels: ``features`` holds the
#: raw RGB pixels as ``(num_samples, 32, 32, 3)`` uint8 and ``labels`` the class (0 to 42).
#: Created by scripts/prepare_validation_datasets.py in the finn-plus repository.
DATASET_FILE = "gtsrb_test_32x32.npz"


def validate(cls_inst, *args, **kwargs):
    """Run GTSRB validation and report top-1 accuracy.

    The accelerator is fed the raw ``UINT8`` pixels, the scaling to [0, 1] is part of the model
    (see ``add_preproc_divide_by_255`` in the build flow of the ``gtsrb`` DUT).
    """
    report_dir = kwargs.get("report_dir")
    dataset_path = kwargs.get("dataset_path") or os.path.join(
        os.environ["DATASET_DIR"], DATASET_FILE
    )
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"{dataset_path} not found, create it with scripts/prepare_validation_datasets.py"
        )
    data = np.load(dataset_path)

    dataset = ArrayDataset(data["features"], data["labels"], drop_remainder=False)
    run_validation(cls_inst, dataset, report_dir, **validation_kwargs(kwargs))

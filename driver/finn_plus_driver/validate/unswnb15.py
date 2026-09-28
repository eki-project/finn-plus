"""Validation script for the UNSW-NB15 intrusion detection dataset."""

import numpy as np
import os

from finn_plus_driver.validate.common import ArrayDataset, run_validation, validation_kwargs


def to_bipolar(x):
    """Map {0, 1} data to bipolar {-1, +1} float32 values as expected by the accelerator."""
    return 2 * x.astype(np.float32) - 1


# From finn examples
def validate(cls_inst, *args, **kwargs):
    """Run UNSW-NB15 validation and report accuracy."""
    report_dir = kwargs.get("report_dir")
    dataset_path = kwargs.get(
        "dataset_path", os.path.join(os.environ["DATASET_DIR"], "unsw_nb15_binarized.npz")
    )
    unsw_nb15_data = np.load(dataset_path)["test"][:82000]

    test_imgs = unsw_nb15_data[:, :-1]
    test_imgs = np.pad(test_imgs, [(0, 0), [0, 7]], mode="constant")
    # labels are bipolar as well, so they compare directly with the accelerator output
    test_labels = to_bipolar(unsw_nb15_data[:, -1]).astype(np.int64)

    dataset = ArrayDataset(test_imgs, test_labels, transform=to_bipolar)
    run_validation(cls_inst, dataset, report_dir, **validation_kwargs(kwargs))

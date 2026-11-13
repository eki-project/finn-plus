"""Validation script for the Google Speech Commands v2 dataset (keyword spotting)."""

import functools
import numpy as np
import os

from finn_plus_driver.validate.common import ArrayDataset, run_validation, validation_kwargs

#: The validation split as published with the finn-examples KWS model: MFCC features computed
#: with python_speech_features (10 coefficients x 49 frames per one-second clip) of the 9981
#: validation clips plus 121 silence samples, labeled with the 12 classes the model was trained
#: on (10 keywords, silence, unknown).
#: See scripts/prepare_validation_datasets.py in the finn-plus repository.
DATASET_FILE = "python_speech_preprocessing_all_validation_KWS_data.npz"

#: Scale of the 8-bit input quantizer of the finn-examples KWS MLP. Only needed for accelerators
#: that take the quantized integers, see :func:`quantize`.
INPUT_SCALE = 0.8298503756523132


def quantize(features, scale=INPUT_SCALE):
    """Quantize float MFCC features to the int8 values of the model's (narrow) input quantizer."""
    return np.clip(np.round(features / scale), -127, 127).astype(np.int8)


def validate(cls_inst, *args, **kwargs):
    """Run Speech Commands v2 validation and report top-1 accuracy.

    Accelerators with a float input (the input quantizer is part of the hardware) are fed the
    MFCC features as they are. Accelerators with an integer input (the input quantizer was
    left out of the hardware, as in the finn-examples build) are fed the features quantized
    with ``input_scale``, which defaults to :data:`INPUT_SCALE`.
    """
    report_dir = kwargs.get("report_dir")
    dataset_path = kwargs.get("dataset_path") or os.path.join(
        os.environ["DATASET_DIR"], DATASET_FILE
    )
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"{dataset_path} not found, fetch it with scripts/prepare_validation_datasets.py"
        )
    data = np.load(dataset_path)

    transform = None
    if cls_inst.idt().is_integer():
        scale = kwargs.get("input_scale", INPUT_SCALE)
        print("Quantizing inputs with scale %s" % scale)
        transform = functools.partial(quantize, scale=scale)

    dataset = ArrayDataset(
        data["data_arr"], data["label_arr"], transform=transform, drop_remainder=False
    )
    run_validation(cls_inst, dataset, report_dir, **validation_kwargs(kwargs))

"""Tests for the validators of the datasets stored as arrays in ``$DATASET_DIR`` (Speech
Commands, GTSRB), run on small stand-in files (no board, PYNQ or real dataset required).

Run from the driver directory with: python -m pytest tests
"""

import pytest

import json
import numpy as np
import os
from finn_plus_driver.validate import VALIDATION_DATASETS, gtsrb, run_validate, speechcommands
from finn_plus_driver.validate.common import REPORT_NAME, ArrayDataset, run_validation
from qonnx.core.datatype import DataType


class FakeAccelerator:
    """Mimics the driver: records the inputs and predicts a class from each sample's sum."""

    def __init__(self, batch_size, sample_shape, idt, num_classes):
        """Set up the fake with the accelerator's batch size, input shape and datatype."""
        self.batch_size = batch_size
        self.sample_shape = tuple(sample_shape)
        self._idt = idt
        self.num_classes = num_classes
        self.inputs_seen = []

    def idt(self, ind=0):
        """Input datatype."""
        return self._idt

    def ishape_normal(self, ind=0):
        """Input shape for the current batch size."""
        return (self.batch_size, *self.sample_shape)

    def execute(self, inputs):
        """Return the top-1 class the way an accelerator with a TopK layer does."""
        assert inputs.shape == self.ishape_normal()
        self.inputs_seen.append(inputs)
        return predict(inputs, self.num_classes).reshape(self.batch_size, 1)


def predict(inputs, num_classes):
    """Class of every sample: the sum over its (integer) values modulo the class count."""
    sums = np.asarray(inputs, dtype=np.int64).reshape(len(inputs), -1).sum(axis=1)
    return sums % num_classes


def read_report(report_dir):
    """Load the validation report."""
    with open(os.path.join(str(report_dir), REPORT_NAME)) as f:
        return json.load(f)


def test_datasets_are_registered():
    assert VALIDATION_DATASETS["speechcommands"] == speechcommands.__name__
    assert VALIDATION_DATASETS["gtsrb"] == gtsrb.__name__


def test_array_dataset_processes_remainder_as_partial_batch(tmp_path):
    inputs = np.arange(25 * 4).reshape(25, 4)
    labels = predict(inputs, 3)
    labels[[3, 22]] += 1  # one wrong sample in a full batch, one in the partial batch
    accel = FakeAccelerator(10, (4,), DataType["INT8"], 3)
    dataset = ArrayDataset(inputs, labels, drop_remainder=False)
    report = run_validation(accel, dataset, str(tmp_path), passes=2)
    assert [len(x) for x in accel.inputs_seen] == [10, 10, 5, 10, 10, 5]
    assert accel.batch_size == 10
    assert report["top-1_accuracy_per_pass"] == [92.0, 92.0]
    assert report["num_prediction_mismatches"] == 0


def test_array_dataset_drops_remainder_by_default(tmp_path):
    inputs = np.arange(25 * 4).reshape(25, 4)
    accel = FakeAccelerator(10, (4,), DataType["INT8"], 3)
    report = run_validation(accel, ArrayDataset(inputs, predict(inputs, 3)), str(tmp_path))
    assert [len(x) for x in accel.inputs_seen] == [10, 10]
    assert report["top-1_accuracy"] == 80.0


def make_speechcommands_file(path, num_samples=23):
    """Stand-in for the published feature file: same keys, layout and value range."""
    rng = np.random.default_rng(0)
    features = rng.uniform(-290, 45, size=(num_samples, 1, 10, 49)).astype(np.float32)
    labels = rng.integers(0, 12, size=num_samples)
    np.savez(path, data_arr=features, label_arr=labels)
    return features, labels


def test_speechcommands_float_input(tmp_path):
    features, _ = make_speechcommands_file(tmp_path / "kws.npz")
    # the accelerator takes the features channels-last, the file holds them channels-first
    accel = FakeAccelerator(10, (10, 49, 1), DataType["FLOAT32"], 12)
    run_validate(
        "speechcommands", accel, report_dir=str(tmp_path), dataset_path=tmp_path / "kws.npz"
    )
    seen = np.concatenate([x.reshape(len(x), -1) for x in accel.inputs_seen])
    assert seen.dtype == np.float32
    np.testing.assert_array_equal(seen, features.reshape(len(features), -1))
    assert read_report(tmp_path)["num_samples"] == 23


def test_speechcommands_integer_input_is_quantized(tmp_path):
    features, _ = make_speechcommands_file(tmp_path / "kws.npz")
    accel = FakeAccelerator(10, (490,), DataType["INT8"], 12)
    run_validate(
        "speechcommands", accel, report_dir=str(tmp_path), dataset_path=tmp_path / "kws.npz"
    )
    seen = np.concatenate(accel.inputs_seen)
    assert seen.dtype == np.int8
    assert seen.min() == -127  # narrow range: -128 is not a quantizer level
    expected = np.round(features.reshape(len(features), -1) / speechcommands.INPUT_SCALE)
    np.testing.assert_array_equal(seen, np.clip(expected, -127, 127))


def test_gtsrb_validates_all_samples(tmp_path):
    rng = np.random.default_rng(0)
    features = rng.integers(0, 256, size=(23, 32, 32, 3), dtype=np.uint8)
    labels = predict(features, 43).astype(np.uint8)
    labels[[0, 21]] = (labels[[0, 21]] + 1) % 43
    np.savez_compressed(tmp_path / "gtsrb.npz", features=features, labels=labels)
    accel = FakeAccelerator(10, (32, 32, 3), DataType["UINT8"], 43)
    run_validate("gtsrb", accel, report_dir=str(tmp_path), dataset_path=tmp_path / "gtsrb.npz")
    seen = np.concatenate(accel.inputs_seen)
    assert seen.dtype == np.uint8
    np.testing.assert_array_equal(seen, features)
    report = read_report(tmp_path)
    assert report["num_samples"] == 23
    assert report["top-1_accuracy"] == pytest.approx(100.0 * 21 / 23)


@pytest.mark.parametrize("dataset", ["speechcommands", "gtsrb"])
def test_missing_dataset_file_is_an_error(tmp_path, dataset):
    accel = FakeAccelerator(10, (4,), DataType["INT8"], 3)
    with pytest.raises(FileNotFoundError, match="prepare_validation_datasets.py"):
        run_validate(dataset, accel, report_dir=str(tmp_path), dataset_path=tmp_path / "x.npz")

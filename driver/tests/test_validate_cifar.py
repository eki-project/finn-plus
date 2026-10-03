"""Tests for the CIFAR validator's input normalization (no board or PYNQ required).

Run from the driver directory with: python -m pytest tests
"""

import pytest

import numpy as np
import sys
import types
from finn_plus_driver.validate import run_validate
from finn_plus_driver.validate.cifar import CIFAR10_MEAN, CIFAR10_STD, normalize_batch
from qonnx.core.datatype import DataType


def test_normalize_batch_defaults():
    rng = np.random.default_rng(0)
    batch = rng.integers(0, 256, size=(4, 32, 32, 3), dtype=np.uint8)
    out = normalize_batch(batch)
    assert out.shape == batch.shape
    assert out.dtype == np.float32
    expected = (
        batch.astype(np.float32) / 255.0 - np.array(CIFAR10_MEAN, dtype=np.float32)
    ) / np.array(CIFAR10_STD, dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-6)
    # the value range must fit a signed 8-bit input quantizer with a scale of ~0.016
    assert out.min() > -2.1 and out.max() < 2.2


def test_normalize_batch_custom_constants():
    batch = np.full((1, 2, 2, 3), 255, dtype=np.uint8)
    batch[..., 0] = 0
    out = normalize_batch(batch, mean=(0.5, 0.5, 0.5), std=(0.25, 0.5, 1.0))
    np.testing.assert_allclose(out[..., 0], -2.0)
    np.testing.assert_allclose(out[..., 1], 1.0)
    np.testing.assert_allclose(out[..., 2], 0.5)


class FakeAccelerator:
    """Mimics the driver: records the inputs and returns class 0 for every sample."""

    def __init__(self, idt):
        """Set up the fake with the accelerator's input datatype."""
        self.batch_size = 4
        self._idt = idt
        self.inputs_seen = []

    def idt(self, ind=0):
        """Input datatype."""
        return self._idt

    def ishape_normal(self, ind=0):
        """Input shape for the current batch size."""
        return (self.batch_size, 32, 32, 3)

    def execute(self, inputs):
        """Record the input batch."""
        self.inputs_seen.append(inputs)
        return np.zeros((self.batch_size, 1), dtype=np.float32)


@pytest.fixture
def cifar_images(monkeypatch):
    """Replace the dataset_loading package by a stand-in that serves eight random images."""
    images = np.random.default_rng(0).integers(0, 256, size=(8, 32, 32, 3), dtype=np.uint8)
    labels = np.zeros(8, dtype=np.int64)

    def load_cifar_data(path, download, one_hot, cifar10):
        """Return the images as the test split, like dataset_loading.cifar.load_cifar_data."""
        return None, None, images, labels, None, None

    cifar = types.SimpleNamespace(load_cifar_data=load_cifar_data)
    monkeypatch.setitem(sys.modules, "dataset_loading", types.SimpleNamespace(cifar=cifar))
    return images


@pytest.mark.parametrize("dataset", ["cifar", "cifar100"])
def test_float_input_is_normalized_by_default(tmp_path, cifar_images, dataset):
    accel = FakeAccelerator(DataType["FLOAT32"])
    run_validate(dataset, accel, report_dir=str(tmp_path), dataset_path=str(tmp_path))
    np.testing.assert_allclose(np.concatenate(accel.inputs_seen), normalize_batch(cifar_images))


def test_integer_input_gets_raw_pixels_by_default(tmp_path, cifar_images):
    accel = FakeAccelerator(DataType["UINT8"])
    run_validate("cifar", accel, report_dir=str(tmp_path), dataset_path=str(tmp_path))
    seen = np.concatenate(accel.inputs_seen)
    assert seen.dtype == np.uint8
    np.testing.assert_array_equal(seen, cifar_images)


def test_normalize_kwarg_overrides_the_default(tmp_path, cifar_images):
    accel = FakeAccelerator(DataType["FLOAT32"])
    run_validate(
        "cifar", accel, report_dir=str(tmp_path), dataset_path=str(tmp_path), normalize=False
    )
    np.testing.assert_array_equal(np.concatenate(accel.inputs_seen), cifar_images)

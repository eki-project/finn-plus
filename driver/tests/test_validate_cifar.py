"""Tests for the CIFAR validator's input normalization (no board or PYNQ required).

Run from the driver directory with: python -m pytest tests
"""

import numpy as np

from finn_plus_driver.validate.cifar import CIFAR10_MEAN, CIFAR10_STD, normalize_batch


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

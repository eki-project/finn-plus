"""Tests for the driver's runtime weight loading. They need the ``pynq`` package (board image)
because the overlay module imports it, but no bitstream: the overlay is not instantiated.

Run from the driver directory with: python -m pytest tests
"""

import pytest

import os

pynq = pytest.importorskip("pynq")

from finn_plus_driver.overlays.dma import FINNDMAOverlay  # noqa: E402


def _bare_overlay(runtime_weight_dir, runtime_weights_expected):
    """An overlay object without running __init__ (no bitstream download)."""
    inst = FINNDMAOverlay.__new__(FINNDMAOverlay)
    inst.runtime_weight_dir = runtime_weight_dir
    inst.runtime_weights_expected = runtime_weights_expected
    return inst


def test_missing_dir_is_an_error_when_runtime_weights_expected(tmp_path):
    inst = _bare_overlay(os.path.join(str(tmp_path), "does_not_exist"), True)
    with pytest.raises(FileNotFoundError):
        inst.load_runtime_weights(flush_accel=False, verify=False)


def test_missing_dir_is_ignored_without_runtime_weights(tmp_path):
    inst = _bare_overlay(os.path.join(str(tmp_path), "does_not_exist"), False)
    inst.load_runtime_weights(flush_accel=False, verify=False)


def test_empty_dir_is_an_error_when_runtime_weights_expected(tmp_path):
    inst = _bare_overlay(str(tmp_path), True)
    with pytest.raises(FileNotFoundError):
        inst.load_runtime_weights(flush_accel=False, verify=False)

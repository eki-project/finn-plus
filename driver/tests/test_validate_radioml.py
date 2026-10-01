"""Tests for the RadioML validator's sample selection and input handling (no board, PYNQ or
real dataset required).

Run from the driver directory with: python -m pytest tests
"""

import pytest

import hashlib
import importlib
import numpy as np

# the validator needs h5py, which is only installed where the driver actually runs
h5py = pytest.importorskip("h5py")
radioml = importlib.import_module("finn_plus_driver.validate.radioml")

SAMPLES_PER_SNR = 4096


def test_transformer_eval_split_is_reproduced():
    indices = np.array(radioml.select_transformer_eval_indices())
    # 10% of the 24 classes x 19 noise levels x 4096 samples from -6 dB upwards
    assert len(indices) == 186778
    assert (np.diff(indices) > 0).all()
    assert ((indices // SAMPLES_PER_SNR) % 26 >= 7).all()
    # The split is cut from a seeded numpy permutation. The checksum is the one of the split
    # whose class labels equal, sample by sample, the classes.csv published with the model,
    # so a numpy version that permutes differently is noticed here and not as an accuracy
    # that silently includes training samples.
    checksum = hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
    assert checksum == "f91aa4f44ed4afbbec5d06e9e432de9acba9a35f5187d54ad971a902ca115952"


def test_vgg10_test_split_is_the_highest_snr():
    indices = np.array(radioml.select_test_indices())
    assert len(indices) == 24 * 410
    assert ((indices // SAMPLES_PER_SNR) % 26 == 25).all()


class FakeAccelerator:
    """Provides the input shape of an accelerator with the given per-sample shape."""

    def __init__(self, batch_size, sample_shape):
        """Set up the fake with a batch size and the input shape of one sample."""
        self.batch_size = batch_size
        self.sample_shape = tuple(sample_shape)

    def ishape_normal(self, ind=0):
        """Input shape for the current batch size."""
        return (self.batch_size, *self.sample_shape)


@pytest.fixture
def radioml_file(tmp_path):
    """Small stand-in for the dataset file: 40 samples of 16 I/Q values, 24 classes."""
    data = np.random.default_rng(0).uniform(-3, 3, size=(40, 16, 2)).astype(np.float32)
    path = tmp_path / "radioml.hdf5"
    with h5py.File(path, "w") as f:
        f["X"] = data
        f["Y"] = np.eye(24, dtype=np.int64)[np.arange(40) % 24]
    return path, data


def test_float_inputs_are_passed_unquantized(radioml_file):
    path, data = radioml_file
    dataset = radioml.RadioMLDataset(path, [3, 7, 8, 30, 39], quantize_inputs=False)
    accel = FakeAccelerator(2, (1, 16, 2))
    batches = list(dataset.iter_batches(accel))
    assert [len(indices) for indices, _, _ in batches] == [2, 2, 1]
    assert accel.batch_size == 1  # shrunk for the last batch, run_validation restores it
    inputs = np.concatenate([x for _, x, _ in batches])
    assert inputs.dtype == np.float32
    np.testing.assert_array_equal(inputs, data[[3, 7, 8, 30, 39]].reshape(5, 1, 16, 2))
    labels = np.concatenate([y for _, _, y in batches])
    np.testing.assert_array_equal(labels, np.array([3, 7, 8, 30, 39]) % 24)
    assert dataset.sample_id(3) == "30"


def test_integer_inputs_are_quantized(radioml_file):
    path, data = radioml_file
    dataset = radioml.RadioMLDataset(path, [3, 7, 8, 30, 39])
    accel = FakeAccelerator(3, (16, 1, 2))
    # duplicates and unsorted indices, as the re-runs of mismatching samples request them
    inputs = dataset.load(accel, [4, 0, 4])
    assert inputs.dtype == np.int8
    np.testing.assert_array_equal(inputs, radioml.quantize(data[[39, 3, 39]]).reshape(3, 16, 1, 2))


def test_predict_accepts_scores_and_top1(radioml_file):
    dataset = radioml.RadioMLDataset(radioml_file[0], [0])
    scores = np.zeros((3, 24), dtype=np.float32)
    scores[[0, 1, 2], [5, 23, 0]] = 1
    np.testing.assert_array_equal(dataset.predict(scores, 3), [5, 23, 0])
    top1 = np.array([[5.0], [23.0], [0.0]], dtype=np.float32)
    np.testing.assert_array_equal(dataset.predict(top1, 3), [5, 23, 0])

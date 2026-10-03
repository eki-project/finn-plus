"""Tests for the masked token validation of the TinyStories validator (no board, PYNQ or real
dataset required).

Run from the driver directory with: python -m pytest tests
"""

import numpy as np

from finn_plus_driver.validate import run_validate
from finn_plus_driver.validate.common import run_validation
from finn_plus_driver.validate.tinystories import IGNORE_INDEX, MaskedTokenDataset

VOCAB_SIZE = 50
SEQUENCE_LENGTH = 16


def expected_token(input_ids):
    """The token the fake accelerator predicts at every position of the given sequences."""
    positions = np.arange(input_ids.shape[-1])
    return (input_ids.astype(np.int64) * 7 + positions) % VOCAB_SIZE


class FakeAccelerator:
    """Mimics the driver: returns scores over the vocabulary for every token of a sequence.

    ``glitches`` maps a call counter (0-based execute() call) to a (row, position) whose
    prediction is corrupted in that call.
    """

    def __init__(self, batch_size, glitches=None):
        """Set up the fake with a batch size and an optional glitch schedule."""
        self.batch_size = batch_size
        self.glitches = glitches or {}
        self.calls = 0
        self.batch_sizes_seen = []

    def ishape_normal(self, ind=0):
        """Input shape for the current batch size."""
        return (self.batch_size, SEQUENCE_LENGTH)

    def execute(self, inputs):
        """Return one-hot scores of the expected token, corrupted where a glitch is due."""
        assert inputs.shape == self.ishape_normal() and inputs.dtype == np.uint64
        self.batch_sizes_seen.append(self.batch_size)
        tokens = expected_token(inputs)
        if self.calls in self.glitches:
            row, position = self.glitches[self.calls]
            tokens[row, position] = (tokens[row, position] + 1) % VOCAB_SIZE
        self.calls += 1
        return np.eye(VOCAB_SIZE, dtype=np.float32)[tokens]


def make_sequences(num_sequences, num_wrong, seed=0):
    """Random sequences with every fourth token masked, ``num_wrong`` of them mislabeled."""
    rng = np.random.default_rng(seed)
    input_ids = rng.integers(0, VOCAB_SIZE, size=(num_sequences, SEQUENCE_LENGTH)).astype(np.int16)
    labels = np.full(input_ids.shape, IGNORE_INDEX, dtype=np.int16)
    masked = rng.random(input_ids.shape) < 0.25
    masked[2] = False  # a sequence without any masked token
    labels[masked] = expected_token(input_ids)[masked]
    wrong = rng.choice(np.argwhere(masked), size=num_wrong, replace=False)
    labels[wrong[:, 0], wrong[:, 1]] = (labels[wrong[:, 0], wrong[:, 1]] + 1) % VOCAB_SIZE
    return input_ids, labels


def test_accuracy_counts_masked_tokens_only(tmp_path):
    input_ids, labels = make_sequences(11, num_wrong=3)
    num_masked = int(np.count_nonzero(labels != IGNORE_INDEX))
    accel = FakeAccelerator(4)
    report = run_validation(accel, MaskedTokenDataset(input_ids, labels), str(tmp_path))
    assert accel.batch_sizes_seen == [4, 4, 3]
    assert accel.batch_size == 4
    assert report["num_samples"] == num_masked
    assert report["top-1_accuracy"] == 100.0 * (num_masked - 3) / num_masked


def test_mismatching_token_is_rerun(tmp_path):
    input_ids, labels = make_sequences(8, num_wrong=0)
    dataset = MaskedTokenDataset(input_ids, labels)
    # corrupt a masked token of the second batch in the second pass (execute() call 3)
    sample = int(dataset.first_sample[5])
    glitch = (int(dataset.sequences[sample]) - 4, int(dataset.positions[sample]))
    accel = FakeAccelerator(4, glitches={3: glitch})
    report = run_validation(accel, dataset, str(tmp_path), passes=2, rerun_repeats=2)
    assert report["top-1_accuracy_per_pass"][0] == 100.0
    assert report["top-1_accuracy_per_pass"][1] < 100.0
    (entry,) = report["mismatches"]
    assert entry["index"] == sample
    assert entry["id"] == "sequence 5 token %d" % glitch[1]
    label = int(dataset.labels[sample])
    assert entry["predictions_per_pass"] == [label, (label + 1) % VOCAB_SIZE]
    assert entry["batch_rerun_predictions"] == [label, label]
    assert entry["isolated_rerun_predictions"] == {str(label): 8}
    assert entry["reproduced"] is False


def test_validate_reads_the_prepared_file(tmp_path):
    input_ids, labels = make_sequences(6, num_wrong=0)
    np.savez_compressed(tmp_path / "stories.npz", input_ids=input_ids, labels=labels)
    accel = FakeAccelerator(4)
    run_validate(
        "tinystories", accel, report_dir=str(tmp_path), dataset_path=tmp_path / "stories.npz"
    )
    assert accel.batch_sizes_seen == [4, 2]

"""Tests for the shared multi-pass validation loop (no board or PYNQ required).

Run from the driver directory with: python -m pytest tests
"""

import json
import numpy as np
import os

from finn_plus_driver.validate.common import (
    PREDICTIONS_NAME,
    REPORT_NAME,
    ArrayDataset,
    run_validation,
    validation_kwargs,
)


class FakeAccelerator:
    """Mimics the driver: predicts argmax over the flattened input, with optional glitches.

    ``glitches`` maps a call counter (0-based execute() call) to a batch position whose output
    is corrupted in that call. This models a rare, non-reproducible on-board error.
    """

    def __init__(self, batch_size, num_classes, glitches=None):
        self._batch_size = batch_size
        self.num_classes = num_classes
        self.glitches = glitches or {}
        self.calls = 0
        self.batch_sizes_seen = []

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    def ishape_normal(self, ind=0):
        return (self._batch_size, self.num_classes)

    def execute(self, inputs):
        assert inputs.shape == self.ishape_normal()
        self.batch_sizes_seen.append(self._batch_size)
        pred = np.argmax(inputs, axis=1).astype(np.float32).reshape(self._batch_size, 1)
        if self.calls in self.glitches:
            pos = self.glitches[self.calls]
            pred[pos, 0] = (pred[pos, 0] + 1) % self.num_classes
        self.calls += 1
        return pred


def make_dataset(num_samples, num_classes, seed=0):
    rng = np.random.default_rng(seed)
    inputs = rng.random((num_samples, num_classes)).astype(np.float32)
    labels = np.argmax(inputs, axis=1)
    # make some samples "wrong" by relabeling them
    labels[::7] = (labels[::7] + 1) % num_classes
    return ArrayDataset(inputs, labels)


def test_single_pass_matches_previous_behavior(tmp_path):
    dataset = make_dataset(40, 5)
    accel = FakeAccelerator(10, 5)
    report = run_validation(accel, dataset, str(tmp_path), passes=1)
    expected_acc = 100.0 * np.count_nonzero(np.arange(40) % 7 != 0) / 40
    assert report["top-1_accuracy"] == expected_acc
    assert report["num_passes"] == 1
    assert report["num_prediction_mismatches"] == 0
    assert report["mismatches"] == []
    with open(os.path.join(str(tmp_path), REPORT_NAME)) as f:
        assert json.load(f)["top-1_accuracy"] == expected_acc
    dump = np.load(os.path.join(str(tmp_path), PREDICTIONS_NAME))
    assert dump["predictions"].shape == (1, 40)
    assert (dump["order"][0] == np.arange(40)).all()
    assert (dump["labels"] == dataset.labels).all()


def test_two_passes_report_and_rerun_mismatch(tmp_path):
    dataset = make_dataset(40, 5)
    # 4 batches per pass; corrupt position 3 of the second batch of the second pass (call 5)
    accel = FakeAccelerator(10, 5, glitches={5: 3})
    report = run_validation(accel, dataset, str(tmp_path), passes=2, rerun_repeats=2)
    assert report["num_passes"] == 2
    assert report["top-1_accuracy"] == report["top-1_accuracy_per_pass"][0]
    assert report["top-1_accuracy_min"] <= report["top-1_accuracy_max"]
    assert report["num_prediction_mismatches"] == 1
    (entry,) = report["mismatches"]
    assert entry["index"] == 13
    assert entry["id"] == "13"
    assert entry["label"] == int(dataset.labels[13])
    correct = int(np.argmax(dataset.inputs[13]))
    assert entry["predictions_per_pass"] == [correct, (correct + 1) % 5]
    # the glitch did not repeat, so the batch re-runs and isolated re-runs are all stable
    assert entry["batch_rerun_predictions"] == [correct, correct]
    assert entry["isolated_rerun_predictions"] == {str(correct): 20}
    assert entry["reproduced"] is False
    assert report["num_mismatches_reproduced"] == 0
    dump = np.load(os.path.join(str(tmp_path), PREDICTIONS_NAME))
    assert (dump["mismatch_indices"] == [13]).all()
    assert dump["predictions"][0, 13] != dump["predictions"][1, 13]
    # batch size is left untouched
    assert accel.batch_size == 10


def test_reproduced_mismatch_is_flagged(tmp_path):
    dataset = make_dataset(20, 4)
    # corrupt the sample in every re-run as well: batch re-runs (calls 4, 5) and isolated (6, 7)
    accel = FakeAccelerator(10, 4, glitches={3: 1, 5: 1, 6: 4})
    report = run_validation(accel, dataset, str(tmp_path), passes=2, rerun_repeats=2)
    (entry,) = report["mismatches"]
    assert entry["index"] == 11
    assert entry["reproduced"] is True
    assert report["num_mismatches_reproduced"] == 1
    assert len(entry["isolated_rerun_predictions"]) == 2


def test_partial_batch_dataset_restores_batch_size(tmp_path):
    class ShrinkingDataset(ArrayDataset):
        def iter_batches(self, cls_inst):
            for start in range(0, len(self), 10):
                indices = np.arange(start, min(start + 10, len(self)))
                cls_inst.batch_size = len(indices)
                yield indices, self.load(cls_inst, indices), self.labels[indices]

    dataset = ShrinkingDataset(*(lambda d: (d.inputs, d.labels))(make_dataset(25, 3)))
    accel = FakeAccelerator(10, 3)
    report = run_validation(accel, dataset, str(tmp_path), passes=2)
    assert accel.batch_size == 10
    assert accel.batch_sizes_seen == [10, 10, 5, 10, 10, 5]
    assert report["num_samples"] == 25
    assert report["num_prediction_mismatches"] == 0


def test_validation_kwargs_filters_unknown_keys():
    kwargs = {"report_dir": "x", "passes": 3, "max_reruns": 2, "validation_dataset": "cifar"}
    assert validation_kwargs(kwargs) == {"passes": 3, "max_reruns": 2}

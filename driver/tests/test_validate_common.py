"""Tests for the shared multi-pass validation loop (no board or PYNQ required).

Run from the driver directory with: python -m pytest tests
"""

import pytest

import json
import numpy as np
import os
import queue

try:  # only installed where the driver actually runs (PYNQ board image, CI test venv)
    import dataset_loading
    from PIL import Image
except ImportError:  # pragma: no cover - exercised by the skip below
    dataset_loading = None

from finn_plus_driver.validate.common import (
    PREDICTIONS_NAME,
    REPORT_NAME,
    ArrayDataset,
    run_validation,
    shutdown_loaders,
    validation_kwargs,
)


class FakeAccelerator:
    """Mimics the driver: predicts argmax over the flattened input, with optional glitches.

    ``glitches`` maps a call counter (0-based execute() call) to a batch position whose output
    is corrupted in that call. This models a rare, non-reproducible on-board error.
    """

    def __init__(self, batch_size, num_classes, glitches=None):
        """Set up the fake with a batch size, class count and optional glitch schedule."""
        self._batch_size = batch_size
        self.num_classes = num_classes
        self.glitches = glitches or {}
        self.calls = 0
        self.batch_sizes_seen = []

    @property
    def batch_size(self):
        """Current batch size."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        """Set the batch size (the real driver reallocates buffers here)."""
        self._batch_size = value

    def ishape_normal(self, ind=0):
        """Input shape for the current batch size."""
        return (self._batch_size, self.num_classes)

    def execute(self, inputs):
        """Predict the argmax of every input, corrupting one position if a glitch is scheduled."""
        assert inputs.shape == self.ishape_normal()
        self.batch_sizes_seen.append(self._batch_size)
        pred = np.argmax(inputs, axis=1).astype(np.float32).reshape(self._batch_size, 1)
        if self.calls in self.glitches:
            pos = self.glitches[self.calls]
            pred[pos, 0] = (pred[pos, 0] + 1) % self.num_classes
        self.calls += 1
        return pred


def make_dataset(num_samples, num_classes, seed=0):
    """Random inputs whose argmax is the label, except every seventh sample which is relabeled."""
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
        """Dataset whose last batch is smaller than the driver batch size."""

        def iter_batches(self, cls_inst):
            """Yield batches of ten, shrinking the driver batch size for the remainder."""
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


class KillLoadersQueue:
    """Stand-in for dataset_loading 0.0.4, whose ImgQueue offers kill_loaders()."""

    def __init__(self, fail=False):
        """Optionally make the shutdown call fail."""
        self.calls = []
        self.fail = fail

    def kill_loaders(self):
        """Record the call and optionally fail."""
        self.calls.append("kill_loaders")
        if self.fail:
            raise RuntimeError("loader shutdown failed")


class JoinQueue(queue.Queue):
    """Stand-in for finn-dataset-loading 0.0.5, whose ImgQueue overrides queue.Queue.join()."""

    def __init__(self):
        """Create the queue and the call log."""
        super().__init__()
        self.calls = []

    def join(self):
        """Record the call."""
        self.calls.append("join")


class BothApisQueue(KillLoadersQueue):
    """A queue offering both methods: kill_loaders() must win."""

    def join(self):
        """Record the call."""
        self.calls.append("join")


def test_shutdown_loaders_prefers_kill_loaders():
    """dataset_loading 0.0.4 exposes kill_loaders(), which must be preferred over join()."""
    q = BothApisQueue()
    shutdown_loaders(q)
    assert q.calls == ["kill_loaders"]


def test_shutdown_loaders_uses_overridden_join():
    """finn-dataset-loading 0.0.5, installed on the board, renamed the method to join()."""
    q = JoinQueue()
    shutdown_loaders(q)
    assert q.calls == ["join"]


def test_shutdown_loaders_ignores_inherited_queue_join():
    """queue.Queue.join() waits for task_done() on every item and would block forever."""

    class PlainQueue(queue.Queue):
        """A queue that does not override join()."""

    q = PlainQueue()
    q.put(("img", 0))  # an unfinished task: queue.Queue.join() would never return
    shutdown_loaders(q)  # must return immediately instead of calling join()


def test_shutdown_loaders_survives_failure():
    """A failing shutdown must not abort a completed validation pass."""
    q = KillLoadersQueue(fail=True)
    shutdown_loaders(q)
    assert q.calls == ["kill_loaders"]


def test_results_are_saved_after_every_pass(tmp_path):
    """A crash in a later pass must not discard the results already measured."""

    class FailingAccelerator(FakeAccelerator):
        """Fake accelerator that fails from the fifth execute() call on."""

        def execute(self, inputs):
            """Fail from the fifth call on, otherwise predict normally."""
            if self.calls >= 4:  # fails at the start of the second pass
                raise RuntimeError("board fell over")
            return super().execute(inputs)

    dataset = make_dataset(40, 5)
    accel = FailingAccelerator(10, 5)
    with pytest.raises(RuntimeError):
        run_validation(accel, dataset, str(tmp_path), passes=2)

    with open(os.path.join(str(tmp_path), REPORT_NAME)) as f:
        report = json.load(f)
    assert report["num_passes"] == 1
    assert report["num_passes_requested"] == 2
    assert report["top-1_accuracy"] == report["top-1_accuracy_per_pass"][0]
    dump = np.load(os.path.join(str(tmp_path), PREDICTIONS_NAME))
    assert dump["predictions"].shape == (1, 40)


def test_rerun_failure_keeps_accuracies(tmp_path):
    """If the diagnostic re-runs fail, the measured accuracies are still reported."""

    class FailingRerunDataset(ArrayDataset):
        """Dataset whose load() fails once both validation passes are done."""

        def __init__(self, *args, **kwargs):
            """Track the number of completed passes."""
            super().__init__(*args, **kwargs)
            self.passes_done = 0

        def iter_batches(self, cls_inst):
            """Yield the batches and count the completed passes."""
            yield from super().iter_batches(cls_inst)
            self.passes_done += 1

        def load(self, cls_inst, indices):
            """Fail during the re-runs, otherwise load normally."""
            if self.passes_done >= 2:  # only fail during the re-runs
                raise RuntimeError("cannot reload sample")
            return super().load(cls_inst, indices)

    base = make_dataset(40, 5)
    dataset = FailingRerunDataset(base.inputs, base.labels)
    accel = FakeAccelerator(10, 5, glitches={5: 3})
    report = run_validation(accel, dataset, str(tmp_path), passes=2)
    assert report["num_prediction_mismatches"] == 1
    assert report["mismatches"] == []
    assert report["num_mismatches_reproduced"] == 0
    assert len(report["top-1_accuracy_per_pass"]) == 2


@pytest.mark.skipif(dataset_loading is None, reason="dataset_loading is not installed")
def test_imagenet_dataset_with_real_loader(tmp_path):
    """Drive ImageNetDataset through the real (multi-threaded) dataset_loading package.

    The loader delivers images in non-deterministic order, so this checks that every sample is
    processed exactly once per pass and that image, index and label stay together. It also
    exercises the loader shutdown, whose API differs between the installed package versions.
    """
    from finn_plus_driver.validate.imagenet import ImageNetDataset

    num_images = 12
    batch_size = 4
    # solid-color images whose gray value encodes the sample index, so that a prediction
    # derived from the pixels reveals a mixed-up index/image association
    for i in range(num_images):
        value = 10 * (i + 1)
        Image.fromarray(np.full((240, 320, 3), value, dtype=np.uint8)).save(
            tmp_path / f"ILSVRC2012_val_{i + 1:08d}.JPEG", quality=100
        )
    label_file = tmp_path / "val.txt"
    label_file.write_text(
        "".join(f"ILSVRC2012_val_{i + 1:08d}.JPEG {i % 3}\n" for i in range(num_images))
    )

    class PixelAccelerator(FakeAccelerator):
        """Returns the index encoded in the image (as the hardware returns the top-1 class)."""

        def ishape_normal(self, ind=0):
            """Input shape of the image accelerator."""
            return (self._batch_size, 224, 224, 3)

        def execute(self, inputs):
            """Return the index encoded in the pixel values of every image."""
            assert inputs.shape == self.ishape_normal()
            means = inputs.reshape(self._batch_size, -1).mean(axis=1)
            self.calls += 1
            return np.round(means / 10.0 - 1).reshape(self._batch_size, 1)

    dataset = ImageNetDataset(str(tmp_path), str(label_file), n_images=num_images, num_threads=3)
    assert len(dataset) == num_images
    assert dataset.sample_id(0) == "ILSVRC2012_val_00000001.JPEG"

    accel = PixelAccelerator(batch_size, 3)
    report = run_validation(accel, dataset, str(tmp_path / "report"), passes=2)

    dump = np.load(os.path.join(str(tmp_path / "report"), PREDICTIONS_NAME))
    # every sample seen exactly once per pass, despite the non-deterministic loader order
    for p in range(2):
        assert sorted(dump["order"][p].tolist()) == list(range(num_images))
    # the prediction read from the image equals the sample index: nothing got mixed up
    assert (dump["predictions"] == np.arange(num_images)).all()
    assert (dump["labels"] == np.arange(num_images) % 3).all()
    assert report["num_prediction_mismatches"] == 0
    assert report["num_passes"] == 2
    # samples 0, 1 and 2 predict their own index, which is also their label (i % 3)
    assert report["top-1_accuracy"] == 100.0 * 3 / num_images


@pytest.mark.skipif(dataset_loading is None, reason="dataset_loading is not installed")
def test_imagenet_dataset_epoch_boundary(tmp_path):
    """Regression test for the flaky ImageNet top-1 accuracy (70.406 / 70.404 / 70.402 %).

    The original validation loop queued the file names with an unlimited number of epochs,
    so after the last file the loader threads immediately started on the first files of the
    next epoch. If the last image of the epoch decoded more slowly than those, one of the
    first images was counted in its place, i.e. once too often, and the last image was
    dropped. With a slow last image and small first images this is reproduced reliably; the
    dataset must nevertheless deliver every image exactly once per pass.
    """
    from finn_plus_driver.validate.imagenet import ImageNetDataset

    num_images = 12
    batch_size = 4
    for i in range(1, num_images + 1):
        size = (1600, 1600) if i == num_images else (8, 8)  # slow last image, fast others
        Image.fromarray(np.full((*size, 3), 10 * i, dtype=np.uint8)).save(
            tmp_path / f"ILSVRC2012_val_{i:08d}.JPEG"
        )
    label_file = tmp_path / "val.txt"
    label_file.write_text(
        "".join(f"ILSVRC2012_val_{i:08d}.JPEG {i}\n" for i in range(1, num_images + 1))
    )

    class Accelerator:
        """Minimal driver stand-in providing batch size and input shape."""

        def __init__(self):
            """Set the batch size."""
            self.batch_size = batch_size

        def ishape_normal(self, ind=0):
            """Input shape of the image accelerator."""
            return (batch_size, 224, 224, 3)

    dataset = ImageNetDataset(str(tmp_path), str(label_file), n_images=num_images, num_threads=4)
    for _ in range(3):
        seen = []
        for indices, inputs, labels in dataset.iter_batches(Accelerator()):
            # the label travels with its image: index i carries label i + 1
            assert list(labels) == [i + 1 for i in indices]
            seen += list(indices)
        assert sorted(seen) == list(range(num_images)), seen

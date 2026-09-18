"""Shared validation loop for the dataset-specific validators.

Runs the accelerator over a dataset one or more times, records the prediction of every sample
and, when more than one pass is requested, reports the samples whose prediction differs between
the passes. Those samples are re-run afterwards to check whether the deviation reproduces.

Background: the CI observed top-1 accuracies of the same bitfile that differ by single images
between measurement runs. A single accuracy number can neither tell which sample flipped nor
whether the deviation is reproducible, so the per-sample predictions of every pass are stored
next to the report (``validate_predictions.npz``) and the mismatches are listed in the report.
"""

import json
import numpy as np
import os
import queue

#: Name of the report written by run_validation() into the report directory
REPORT_NAME = "report_dma_validate.json"
#: Name of the per-sample prediction dump written next to the report
PREDICTIONS_NAME = "validate_predictions.npz"


def shutdown_loaders(img_queue):
    """Stop the loader threads/processes filling a dataset_loading ImgQueue, best effort.

    The shutdown method differs between the loader versions in use: ``dataset_loading`` 0.0.4
    (multiprocessing-based) offers ``kill_loaders()``, while ``finn-dataset-loading`` 0.0.5
    (thread-based), which is installed on the PYNQ board images, renamed it to ``join()``.

    Only an overridden ``join()`` is called: the inherited ``queue.Queue.join()`` waits for
    ``task_done()`` on every item and would block forever. Failures are reported but never
    raised, because the loaders also terminate on their own once the file queue is exhausted
    (the queues are loaded with ``max_epochs=1``), and losing a completed validation pass to a
    cleanup error would defeat the purpose of the measurement.
    """
    if hasattr(img_queue, "kill_loaders"):
        method_name = "kill_loaders"
    elif getattr(type(img_queue), "join", None) not in (None, queue.Queue.join):
        method_name = "join"
    else:
        print("WARNING: image loaders offer no known shutdown method, letting them finish")
        return
    try:
        getattr(img_queue, method_name)()
    except Exception as e:
        print(f"WARNING: could not stop the image loaders via {method_name}(): {e}")


class ValidationDataset:
    """Interface that the dataset-specific validators implement for :func:`run_validation`.

    Samples are addressed by an index in ``range(len(dataset))``. Labels and predictions are
    integers (or anything that compares with ``==`` after conversion to ``int64``).
    """

    def __len__(self):
        """Number of samples the accuracy is computed over."""
        raise NotImplementedError

    def sample_id(self, index):
        """Human-readable identifier of a sample (e.g. the file name), used in the report."""
        return str(index)

    def iter_batches(self, cls_inst):
        """Yield ``(indices, inputs, labels)`` batches for one pass over the dataset.

        ``inputs`` must be ready to be passed to ``cls_inst.execute()`` and ``indices`` and
        ``labels`` must be 1D and of equal length. A batch may only be smaller than
        ``cls_inst.batch_size`` if the dataset adjusts ``cls_inst.batch_size`` itself
        (:func:`run_validation` restores the original batch size after every pass).
        """
        raise NotImplementedError

    def load(self, cls_inst, indices):
        """Return the accelerator inputs of the given samples (as one batch, duplicates allowed)."""
        raise NotImplementedError

    def predict(self, obuf, batch_size):
        """Convert the accelerator output of a batch into one prediction per sample."""
        raise NotImplementedError


class ArrayDataset(ValidationDataset):
    """Dataset held in memory as ``(num_samples, ...)`` input and ``(num_samples,)`` label arrays.

    ``transform`` is applied to every input batch before it is reshaped to the accelerator's
    input shape. Only full batches are processed; a remainder that does not fill a batch is
    skipped but still counts towards the total (matching the previous behavior of the
    validators).
    """

    def __init__(self, inputs, labels, transform=None):
        assert inputs.shape[0] == labels.shape[0], "Number of inputs and labels differ"
        self.inputs = inputs
        self.labels = np.asarray(labels).flatten()
        self.transform = transform

    def __len__(self):
        return self.inputs.shape[0]

    def iter_batches(self, cls_inst):
        batch_size = cls_inst.batch_size
        n_batches = len(self) // batch_size
        if n_batches * batch_size != len(self):
            print(
                "WARNING: %d samples do not fill a batch of %d and are skipped (counted as wrong)"
                % (len(self) - n_batches * batch_size, batch_size)
            )
        for i in range(n_batches):
            indices = np.arange(i * batch_size, (i + 1) * batch_size)
            yield indices, self.load(cls_inst, indices), self.labels[indices]

    def load(self, cls_inst, indices):
        batch = self.inputs[np.asarray(indices)]
        if self.transform is not None:
            batch = self.transform(batch)
        return batch.reshape(cls_inst.ishape_normal())

    def predict(self, obuf, batch_size):
        obuf = obuf.reshape(batch_size, -1)
        if obuf.shape[1] > 1:
            # accelerator returns logits/scores: pick the top-1 class
            return np.argmax(obuf, axis=1)
        return obuf[:, 0]


def _as_index_array(values):
    return np.asarray(values, dtype=np.int64).flatten()


def _build_report(accuracies, num_samples, passes, mismatch_indices, mismatches, num_reproduced):
    """Assemble the report dict. ``num_passes`` is the number of passes actually completed."""
    return {
        # accuracy of the first pass, i.e. what a single-pass validation reports
        "top-1_accuracy": accuracies[0] if accuracies else 0.0,
        "num_samples": num_samples,
        "num_passes": len(accuracies),
        "num_passes_requested": passes,
        "top-1_accuracy_min": min(accuracies) if accuracies else 0.0,
        "top-1_accuracy_max": max(accuracies) if accuracies else 0.0,
        "top-1_accuracy_per_pass": accuracies,
        "num_prediction_mismatches": int(len(mismatch_indices)),
        "num_mismatches_reproduced": num_reproduced,
        "mismatches": mismatches,
    }


def _save_results(report_dir, report, labels, predictions, order, mismatch_indices):
    """Write the report and the per-sample prediction dump.

    Called after every pass so that a failure in a later pass (or in the re-run analysis)
    cannot discard the results that were already measured.
    """
    os.makedirs(report_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(report_dir, PREDICTIONS_NAME),
        labels=labels,
        predictions=predictions,
        order=order,
        mismatch_indices=mismatch_indices,
    )
    with open(os.path.join(report_dir, REPORT_NAME), "w") as f:
        json.dump(report, f, indent=2)


def _rerun_batches(cls_inst, dataset, index, order, batch_size):
    """Re-run the batches (one per pass) in which the sample was processed, keeping the batch
    composition of the pass. Returns one prediction per pass, None where the batch could not be
    reconstructed (e.g. a partial last batch)."""
    predictions = []
    for pass_order in order:
        position = np.where(pass_order == index)[0]
        if position.size == 0:
            predictions.append(None)
            continue
        start = (int(position[0]) // batch_size) * batch_size
        batch_indices = pass_order[start : start + batch_size]
        batch_indices = batch_indices[batch_indices >= 0]
        if len(batch_indices) != batch_size:
            predictions.append(None)
            continue
        inputs = dataset.load(cls_inst, batch_indices)
        pred = _as_index_array(dataset.predict(cls_inst.execute(inputs), batch_size))
        predictions.append(int(pred[np.where(batch_indices == index)[0][0]]))
    return predictions


def run_validation(cls_inst, dataset, report_dir, passes=1, rerun_repeats=3, max_reruns=20):
    """Validate the accelerator on a dataset and write ``report_dma_validate.json``.

    Parameters
    ----------
    cls_inst : driver instance
        Provides ``batch_size``, ``ishape_normal()`` and ``execute()``.
    dataset : ValidationDataset
        The dataset to validate on.
    report_dir : str
        Directory the report and the per-sample prediction dump are written to.
    passes : int
        Number of passes over the dataset. With more than one pass, samples whose prediction
        differs between the passes are reported and re-run.
    rerun_repeats : int
        How often each mismatching sample is re-run in isolation (a batch filled with copies
        of the sample) to check whether its prediction is stable.
    max_reruns : int
        Maximum number of mismatching samples that are re-run and listed in detail.

    Returns
    -------
    dict
        The report that was written to ``report_dir``.
    """
    passes = max(1, int(passes))
    num_samples = len(dataset)
    batch_size = cls_inst.batch_size
    labels = np.full(num_samples, -1, dtype=np.int64)
    predictions = np.full((passes, num_samples), -1, dtype=np.int64)
    # sample indices in processing order, one row per pass (batch composition may vary)
    order = np.full((passes, num_samples), -1, dtype=np.int64)
    accuracies = []

    for p in range(passes):
        if cls_inst.batch_size != batch_size:
            cls_inst.batch_size = batch_size
        print("Starting validation pass %d/%d.." % (p + 1, passes))
        ok = 0
        nok = 0
        position = 0
        for b, (indices, inputs, exp) in enumerate(dataset.iter_batches(cls_inst)):
            indices = _as_index_array(indices)
            exp = _as_index_array(exp)
            pred = _as_index_array(dataset.predict(cls_inst.execute(inputs), len(indices)))
            assert pred.shape == exp.shape == indices.shape, "Batch shapes do not match"
            matches = pred == exp
            ok += int(np.count_nonzero(matches))
            nok += int(np.count_nonzero(~matches))
            labels[indices] = exp
            predictions[p, indices] = pred
            order[p, position : position + len(indices)] = indices
            position += len(indices)
            print("batch %d : total OK %d NOK %d" % (b + 1, ok, nok))
        if position != num_samples:
            print("WARNING: pass %d processed %d of %d samples" % (p + 1, position, num_samples))
        acc = 100.0 * ok / num_samples
        accuracies.append(acc)
        print("Final top-1 accuracy (pass %d/%d): %f%%" % (p + 1, passes, acc))
        # persist after every pass, a later pass must not be able to discard these results
        _save_results(
            report_dir,
            _build_report(accuracies, num_samples, passes, [], [], 0),
            labels,
            predictions[: p + 1],
            order[: p + 1],
            np.zeros(0, dtype=np.int64),
        )
    if cls_inst.batch_size != batch_size:
        cls_inst.batch_size = batch_size

    # Samples whose prediction is not identical in all passes
    if passes > 1:
        mismatch_indices = np.where(np.any(predictions != predictions[0], axis=0))[0]
    else:
        mismatch_indices = np.zeros(0, dtype=np.int64)
    if passes > 1:
        print(
            "Predictions of %d/%d samples differ between the %d passes"
            % (len(mismatch_indices), num_samples, passes)
        )

    mismatches, num_reproduced = _analyze_mismatches(
        cls_inst,
        dataset,
        mismatch_indices[:max_reruns],
        labels,
        predictions,
        order,
        batch_size,
        rerun_repeats,
    )

    report = _build_report(
        accuracies, num_samples, passes, mismatch_indices, mismatches, num_reproduced
    )
    _save_results(report_dir, report, labels, predictions, order, mismatch_indices)
    return report


def _analyze_mismatches(
    cls_inst, dataset, mismatch_indices, labels, predictions, order, batch_size, rerun_repeats
):
    """Re-run the samples whose prediction differed between passes.

    Each sample is re-run twice: once in the batch composition of every pass, and once in a
    batch consisting only of copies of the sample. Returns the per-sample findings and how
    many of them produced differing predictions again.

    Re-running exercises the accelerator, so a failure here is caught: the measured accuracies
    are the primary result and must survive a problem in the diagnostic re-runs.
    """
    mismatches = []
    num_reproduced = 0
    for index in mismatch_indices:
        index = int(index)
        try:
            entry = _rerun_mismatch(
                cls_inst, dataset, index, labels, predictions, order, batch_size, rerun_repeats
            )
        except Exception as e:
            print(f"WARNING: could not re-run sample {index}: {e}")
            continue
        num_reproduced += int(entry["reproduced"])
        mismatches.append(entry)
        print(
            "Sample %d (%s): label %d, predictions per pass %s, batch re-runs %s, "
            "isolated re-runs %s"
            % (
                index,
                entry["id"],
                entry["label"],
                entry["predictions_per_pass"],
                entry["batch_rerun_predictions"],
                entry["isolated_rerun_predictions"],
            )
        )
    return mismatches, num_reproduced


def _rerun_mismatch(
    cls_inst, dataset, index, labels, predictions, order, batch_size, rerun_repeats
):
    """Collect the re-run predictions of a single mismatching sample."""
    entry = {
        "index": index,
        "id": dataset.sample_id(index),
        "label": int(labels[index]),
        "predictions_per_pass": predictions[:, index].tolist(),
    }
    # (a) re-run the batches the sample was part of, with the batch composition of each pass
    entry["batch_rerun_predictions"] = _rerun_batches(cls_inst, dataset, index, order, batch_size)
    # (b) re-run the sample in isolation: a batch consisting only of copies of the sample
    inputs = dataset.load(cls_inst, np.full(batch_size, index))
    isolated = [
        _as_index_array(dataset.predict(cls_inst.execute(inputs), batch_size))
        for _ in range(max(0, int(rerun_repeats)))
    ]
    isolated = np.stack(isolated) if isolated else np.zeros((0, batch_size), dtype=np.int64)
    isolated_values, isolated_counts = np.unique(isolated, return_counts=True)
    entry["isolated_rerun_predictions"] = {
        str(int(v)): int(c) for v, c in zip(isolated_values, isolated_counts)
    }
    batch_values = {v for v in entry["batch_rerun_predictions"] if v is not None}
    entry["reproduced"] = bool(len(isolated_values) > 1 or len(batch_values) > 1)
    return entry


def validation_kwargs(kwargs):
    """Extract the :func:`run_validation` options from the validate() keyword arguments."""
    return {k: kwargs[k] for k in ("passes", "rerun_repeats", "max_reruns") if k in kwargs}

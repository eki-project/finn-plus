"""Compare the per-sample prediction dumps of several accelerator validation runs.

The Pynq driver's ``validate`` function writes ``validate_predictions.npz`` next to its report
(see ``driver/finn_plus_driver/validate/common.py``). Given the dumps of several measurement
jobs of the same bitfile, this script lists every sample whose prediction is not identical in
all passes of all runs, together with the context needed to tell hypotheses apart:

- the label, the prediction in every pass, and the accuracy of every pass
- the position of the sample in the batch of each pass and the predictions of its batch
  neighbours (a wrong prediction equal to the previous frame's class points at a stale output
  buffer in the driver/DMA rather than at the accelerator)

Usage::

    python scripts/compare_validation_predictions.py run_a.npz run_b.npz [...]
    python scripts/compare_validation_predictions.py --imagenet run_a.npz run_b.npz
"""

import argparse
import numpy as np
from pathlib import Path


def load_runs(paths):
    """Load the dumps and check that they describe the same dataset."""
    runs = []
    labels = None
    for path in paths:
        dump = np.load(path)
        if labels is None:
            labels = dump["labels"]
        elif not np.array_equal(labels, dump["labels"]):
            raise SystemExit(f"{path}: labels differ from {paths[0]}, not the same dataset")
        runs.append((Path(path).stem, dump["predictions"], dump["order"]))
    return labels, runs


def batch_context(order_row, index, batch_size, predictions_row):
    """Position of a sample within its batch and the predictions of the whole batch."""
    position = np.where(order_row == index)[0]
    if position.size == 0:
        return None, []
    position = int(position[0])
    start = (position // batch_size) * batch_size
    batch = order_row[start : start + batch_size]
    batch = batch[batch >= 0]
    return position - start, [int(predictions_row[i]) for i in batch]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dumps", nargs="+", help="validate_predictions.npz files to compare")
    parser.add_argument(
        "--batch-size", type=int, default=100, help="driver batch size used for validation"
    )
    parser.add_argument(
        "--imagenet",
        action="store_true",
        help="print ILSVRC2012 validation file names instead of plain sample indices",
    )
    parser.add_argument(
        "--max-samples", type=int, default=50, help="maximum number of samples to list"
    )
    args = parser.parse_args()

    labels, runs = load_runs(args.dumps)
    num_samples = len(labels)
    # one row per pass over all runs, plus a (run, pass) tag per row
    rows = []
    tags = []
    for name, predictions, order in runs:
        for p in range(predictions.shape[0]):
            rows.append((predictions[p], order[p]))
            tags.append(f"{name}#{p + 1}")
    all_predictions = np.stack([r[0] for r in rows])

    print(f"{len(runs)} runs, {len(rows)} passes, {num_samples} samples")
    for tag, (predictions, _) in zip(tags, rows):
        acc = 100.0 * np.count_nonzero(predictions == labels) / num_samples
        print(f"  {tag:<40} top-1 accuracy {acc:.3f}%")

    differing = np.where(np.any(all_predictions != all_predictions[0], axis=0))[0]
    print(f"\nSamples with differing predictions: {len(differing)}")
    for index in differing[: args.max_samples]:
        index = int(index)
        name = f"ILSVRC2012_val_{index + 1:08d}.JPEG" if args.imagenet else str(index)
        print(f"\nsample {index} ({name}), label {int(labels[index])}")
        for tag, (predictions, order) in zip(tags, rows):
            pred = int(predictions[index])
            verdict = "ok " if pred == labels[index] else "BAD"
            pos, batch = batch_context(order, index, args.batch_size, predictions)
            if pos is None:
                print(f"  {tag:<40} {verdict} pred {pred:>5}   (not processed)")
                continue
            prev_pred = batch[pos - 1] if pos > 0 else None
            note = ""
            if pred != labels[index] and prev_pred is not None and prev_pred == pred:
                note = "  <- equals previous frame's prediction (stale output?)"
            print(
                f"  {tag:<40} {verdict} pred {pred:>5}   batch position {pos:>3}, "
                f"previous frame pred {prev_pred}{note}"
            )
    if len(differing) > args.max_samples:
        print(f"\n... {len(differing) - args.max_samples} more samples not shown")


if __name__ == "__main__":
    main()

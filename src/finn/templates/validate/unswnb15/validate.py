"""Validation script for the UNSW-NB15 intrusion detection dataset."""
import json
import numpy as np
import os


# From finn examples
def validate(cls_inst, *args, **kwargs):
    """Run UNSW-NB15 validation and report accuracy."""
    report_dir = kwargs.get("report_dir")
    dataset_path = kwargs.get(
        "dataset_path", os.path.join(os.environ["DATASET_DIR"], "unsw_nb15_binarized.npz")
    )
    unsw_nb15_data = np.load(dataset_path)["test"][:82000]
    batch_size = cls_inst.batch_size

    test_imgs = unsw_nb15_data[:, :-1]
    test_imgs = np.pad(test_imgs, [(0, 0), [0, 7]], mode="constant")
    test_labels = unsw_nb15_data[:, -1]
    n_batches = int(test_imgs.shape[0] / batch_size)
    test_imgs = test_imgs.reshape(n_batches, batch_size, -1)
    test_labels = test_labels.reshape(n_batches, batch_size)

    ok = 0
    nok = 0
    n_batches = test_imgs.shape[0]
    total = batch_size * n_batches

    for i in range(n_batches):
        inp = test_imgs[i].astype(np.float32)
        exp = test_labels[i].astype(np.float32)
        inp = 2 * inp - 1
        exp = 2 * exp - 1
        out = cls_inst.execute(inp)
        matches = np.count_nonzero(out.flatten() == exp.flatten())
        nok += batch_size - matches
        ok += matches
        print("batch %d / %d : total OK %d NOK %d" % (i + 1, n_batches, ok, nok))

    acc = 100.0 * ok / (total)
    print("Final accuracy: {:.2f}%".format(acc))

    # write report to file
    report = {
        "top-1_accuracy": acc,
    }
    reportfile = os.path.join(report_dir, "report_dma_validate.json")
    with open(reportfile, "w") as f:
        json.dump(report, f, indent=2)

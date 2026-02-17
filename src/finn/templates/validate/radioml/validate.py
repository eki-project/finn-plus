import h5py
import json
import math
import numpy as np
import os


def quantize(data):
    quant_min = -2.0
    quant_max = 2.0
    quant_range = quant_max - quant_min
    data_quant = (data - quant_min) / quant_range
    data_quant = np.round(data_quant * 256) - 128
    data_quant = np.clip(data_quant, -128, 127)
    data_quant = data_quant.astype(np.int8)
    return data_quant


def validate(cls_inst, *args, **kwargs):
    report_dir = kwargs.get("report_dir")
    dataset_path = kwargs.get(
        "dataset_path", os.path.join(os.environ["DATASET_DIR"], "GOLD_XYZ_OSC.0001_1024.hdf5")
    )
    h5_file = h5py.File(dataset_path, "r", locking=False)
    data_h5 = h5_file["X"]
    label_mod = np.argmax(h5_file["Y"], axis=1)  # comes in one-hot encoding

    # assemble list of test set indices
    # do not pre-load large dataset into memory
    np.random.seed(2018)
    test_indices = []
    for mod in range(0, 24):  # all modulations (0 to 23)
        for snr_idx in range(0, 26):  # all SNRs (0 to 25 = -20dB to +30dB)
            start_idx = 26 * 4096 * mod + 4096 * snr_idx
            indices_subclass = list(range(start_idx, start_idx + 4096))

            split = int(np.ceil(0.1 * 4096))  # 90%/10% split
            np.random.shuffle(indices_subclass)
            val_indices_subclass = indices_subclass[:split]

            if snr_idx >= 25:  # select which SNRs to test on
                test_indices.extend(val_indices_subclass)

    test_indices = sorted(test_indices)

    ok = 0
    nok = 0
    total = len(test_indices)
    batch_size = cls_inst.batch_size
    for i_batch in range(math.ceil(total / batch_size)):
        i_frame = i_batch * batch_size
        if i_frame + batch_size > total:
            batch_size = total - i_frame
            cls_inst.batch_size = batch_size
        batch_indices = test_indices[i_frame : i_frame + batch_size]
        data, mod = data_h5[batch_indices], label_mod[batch_indices]

        ibuf = quantize(data).reshape(cls_inst.ishape_normal(0))
        obuf = cls_inst.execute(ibuf)

        pred = obuf.reshape(batch_size).astype(int)

        ok += np.equal(pred, mod).sum().item()
        nok += np.not_equal(pred, mod).sum().item()

        print("batch %d : total OK %d NOK %d" % (i_batch, ok, nok))

    acc = 100.0 * ok / (total)
    print("Measured top-1 accuracy: {}%".format(acc))

    # write report to file
    report = {
        "top-1_accuracy": acc,
    }
    reportfile = os.path.join(report_dir, "report_dma_validate.json")
    with open(reportfile, "w") as f:
        json.dump(report, f, indent=2)

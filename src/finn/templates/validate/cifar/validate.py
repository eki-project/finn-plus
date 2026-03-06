import json
import numpy as np
import os
from dataset_loading import cifar


def validate(cls_inst, *args, **kwargs):
    report_dir = kwargs.get("report_dir")
    dataset_path = kwargs.get("dataset_path", os.path.dirname(os.path.realpath(__file__)))
    bsize = cls_inst.batch_size
    cifar10 = kwargs.get("cifar10", True)

    trainx, trainy, testx, testy, valx, valy = cifar.load_cifar_data(
        dataset_path, download=True, one_hot=False, cifar10=cifar10
    )

    test_imgs = testx
    test_labels = testy

    ok = 0
    nok = 0
    total = test_imgs.shape[0]

    n_batches = int(total / bsize)

    test_imgs = test_imgs.reshape(n_batches, bsize, -1)
    test_labels = test_labels.reshape(n_batches, bsize)

    print("Starting validation..")
    for i in range(n_batches):
        ibuf_normal = test_imgs[i].reshape(cls_inst.ishape_normal())
        exp = test_labels[i]
        obuf_normal = cls_inst.execute(ibuf_normal)
        # obuf_normal = obuf_normal.reshape(bsize, -1)[:,0]
        if obuf_normal.shape[1] > 1:
            obuf_normal = np.argmax(obuf_normal, axis=1)
        ret = np.bincount(obuf_normal.flatten() == exp.flatten(), minlength=2)
        nok += ret[0]
        ok += ret[1]
        print("batch %d / %d : total OK %d NOK %d" % (i + 1, n_batches, ok, nok))

    # calculate top-1 accuracy
    acc = 100.0 * ok / (total)
    print("Final accuracy: %f" % acc)

    # write report to file
    report = {
        "top-1_accuracy": acc,
    }
    reportfile = os.path.join(report_dir, "report_dma_validate.json")
    with open(reportfile, "w") as f:
        json.dump(report, f, indent=2)

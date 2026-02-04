import json
import numpy as np
import os
from dataset_loading import FileQueue, ImgQueue
from PIL import Image


def img_resize(img, size):
    w, h = img.size
    if (w <= h and w == size) or (h <= w and h == size):
        return img
    if w < h:
        ow = size
        oh = int(size * h / w)
        return img.resize((ow, oh), Image.BILINEAR)
    else:
        oh = size
        ow = int(size * w / h)
        return img.resize((ow, oh), Image.BILINEAR)


def img_center_crop(img, size):
    crop_height, crop_width = (size, size)
    image_width, image_height = img.size
    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))


def pre_process(img_np):
    img = Image.fromarray(img_np.astype(np.uint8))
    img = img_resize(img, 256)
    img = img_center_crop(img, 224)
    img = np.array(img, dtype=np.uint8)
    return img


def setup_dataloader(val_path, label_file_path=None, batch_size=100, n_images=50000):
    files = ["ILSVRC2012_val_{:08d}.JPEG".format(i) for i in range(1, n_images + 1)]
    labels = np.loadtxt(label_file_path, dtype=int, usecols=1)
    file_queue = FileQueue()
    file_queue.load_epochs(list(zip(files, labels)), shuffle=False)
    img_queue = ImgQueue(maxsize=batch_size)
    img_queue.start_loaders(file_queue, num_threads=4, img_dir=val_path, transform=pre_process)
    return img_queue


def validate(cls_inst, *args, **kwargs):
    report_dir = kwargs.get("report_dir")
    dataset_path = kwargs.get(
        "dataset_path",
        os.path.join(os.environ["DATASET_DIR"], "ImageNet2012", "ILSVRC2012_img_val"),
    )
    batch_size = cls_inst.batch_size
    img_queue = setup_dataloader(dataset_path, os.path.join(dataset_path, "../val.txt"), batch_size)

    ok = 0
    nok = 0
    i = 0

    while not img_queue.last_batch:
        imgs, lbls = img_queue.get_batch(batch_size, timeout=None)
        imgs = np.array(imgs)
        exp = np.array(lbls)

        ibuf_normal = imgs.reshape(cls_inst.ishape_normal())
        obuf_normal = cls_inst.execute(ibuf_normal)
        obuf_normal = obuf_normal.reshape(batch_size, -1)[:, 0]
        ret = np.bincount(obuf_normal.flatten() == exp.flatten())
        nok += ret[0]
        ok += ret[1]
        i += 1
        print("batch %d : total OK %d NOK %d" % (i, ok, nok))

    total = 50000
    acc = 100.0 * ok / (total)
    print("Final top-1 accuracy: {}%".format(acc))

    # write report to file
    report = {
        "top-1_accuracy": acc,
    }
    reportfile = os.path.join(report_dir, "report_dma_validate.json")
    with open(reportfile, "w") as f:
        json.dump(report, f, indent=2)

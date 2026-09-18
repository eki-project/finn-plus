"""Validation script for the ImageNet (ILSVRC2012) dataset."""

import numpy as np
import os
from dataset_loading import FileQueue, ImgQueue
from finn_plus_driver.validate.common import (
    ValidationDataset,
    run_validation,
    shutdown_loaders,
    validation_kwargs,
)
from PIL import Image


def img_resize(img, size):
    """Resize an image so its shorter side equals the given size."""
    w, h = img.size
    if (w <= h and w == size) or (h <= w and h == size):
        return img
    if w < h:
        ow = size
        oh = int(size * h / w)
        return img.resize((ow, oh), Image.BILINEAR)
    oh = size
    ow = int(size * w / h)
    return img.resize((ow, oh), Image.BILINEAR)


def img_center_crop(img, size):
    """Center-crop an image to a square of the given size."""
    crop_height, crop_width = (size, size)
    image_width, image_height = img.size
    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))


def pre_process(img_np):
    """Resize and center-crop a numpy image array for ImageNet inference."""
    img = Image.fromarray(img_np.astype(np.uint8))
    img = img_resize(img, 256)
    img = img_center_crop(img, 224)
    img = np.array(img, dtype=np.uint8)
    return img


def load_image(path):
    """Load and pre-process a single image exactly like the ImgQueue loader threads do."""
    img = Image.open(path).convert(mode="RGB")
    return pre_process(np.array(img).astype(np.float32))


class ImageNetDataset(ValidationDataset):
    """The ILSVRC2012 validation set, streamed from JPEG files by dataset_loading."""

    def __init__(self, dataset_path, label_file_path, n_images=50000, num_threads=4):
        self.dataset_path = dataset_path
        self.files = [f"ILSVRC2012_val_{i:08d}.JPEG" for i in range(1, n_images + 1)]
        self.labels = np.loadtxt(label_file_path, dtype=int, usecols=1)[:n_images]
        assert len(self.labels) == n_images, "Label file has fewer entries than images"
        self.num_threads = num_threads

    def __len__(self):
        return len(self.files)

    def sample_id(self, index):
        return self.files[index]

    def iter_batches(self, cls_inst):
        batch_size = cls_inst.batch_size
        # The loader threads deliver images in non-deterministic order, so every file is queued
        # together with its (index, label) pair to map the batches back to the samples.
        items = list(zip(self.files, zip(range(len(self.files)), self.labels.tolist())))
        file_queue = FileQueue()
        file_queue.load_epochs(items, shuffle=False, max_epochs=1)
        img_queue = ImgQueue(maxsize=batch_size)
        img_queue.start_loaders(
            file_queue,
            num_threads=self.num_threads,
            img_dir=self.dataset_path,
            transform=pre_process,
        )
        try:
            while not img_queue.last_batch:
                imgs, lbls = img_queue.get_batch(batch_size, timeout=None)
                lbls = np.array(lbls, dtype=np.int64)
                inputs = np.array(imgs).reshape(cls_inst.ishape_normal())
                yield lbls[:, 0], inputs, lbls[:, 1]
        finally:
            shutdown_loaders(img_queue)

    def load(self, cls_inst, indices):
        imgs = [load_image(os.path.join(self.dataset_path, self.files[i])) for i in indices]
        return np.array(imgs).reshape(cls_inst.ishape_normal())

    def predict(self, obuf, batch_size):
        # the accelerator returns the top-1 class index (TopK / LabelSelect layer in hardware)
        return obuf.reshape(batch_size, -1)[:, 0]


def validate(cls_inst, *args, **kwargs):
    """Run ImageNet validation and report top-1 accuracy."""
    report_dir = kwargs.get("report_dir")
    dataset_path = kwargs.get(
        "dataset_path",
        os.path.join(os.environ["DATASET_DIR"], "ImageNet2012", "ILSVRC2012_img_val"),
    )
    dataset = ImageNetDataset(dataset_path, os.path.join(dataset_path, "../val.txt"))
    run_validation(cls_inst, dataset, report_dir, **validation_kwargs(kwargs))

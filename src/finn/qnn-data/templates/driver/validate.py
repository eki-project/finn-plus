# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import os
import numpy as np
from PIL import Image
from dataset_loading import FileQueue, ImgQueue
import json
from pynq import PL
from driver import io_shape_dict
from driver_base import FINNExampleOverlay

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
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))

def pre_process(img_np):
    img = Image.fromarray(img_np.astype(np.uint8))
    img = img_resize(img, 256)
    img = img_center_crop(img, 224)
    img = np.array(img, dtype=np.uint8)
    return img

def setup_dataloader(val_path, label_file_path = None, batch_size=100, n_images = 50000):
    if label_file_path is None:
        val_folders = [ f.name for f in os.scandir(val_path) if f.is_dir() ]
        val_folders = sorted(val_folders)
        assert len(val_folders) == 1000, "Expected 1000 subfolders in ILSVRC2012 val"
        files = []
        labels = []
        for idx, folder in enumerate(val_folders):
            current_files = sorted(os.listdir(os.path.join(val_path, folder)))
            current_files = [os.path.join(folder, file) for file in current_files]
            files.extend(current_files)
            labels.extend([idx]*len(current_files))
        files = files[:n_images]
    else:
        files = ['ILSVRC2012_val_{:08d}.JPEG'.format(i) for i in range(1,n_images+1)]
        labels = np.loadtxt(label_file_path, dtype=int, usecols=1)

    file_queue = FileQueue()
    file_queue.load_epochs(list(zip(files,labels)), shuffle=False)
    img_queue = ImgQueue(maxsize=batch_size)
    img_queue.start_loaders(file_queue, num_threads=1, img_dir=val_path, transform=pre_process)
    return img_queue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate top-1 accuracy for FINN-generated accelerator"
    )
    parser.add_argument(
        "--batchsize", help="number of samples for inference", type=int, default=100
    )
    parser.add_argument("--dataset", help="dataset to use (mnist, cifar10, cifar100, imagenet)", default="")
    parser.add_argument(
        "--platform", help="Target platform: zynq-iodma alveo", default="zynq-iodma"
    )
    parser.add_argument(
        "--bitfile", help='name of bitfile (i.e. "resizer.bit")', default="resizer.bit"
    )
    parser.add_argument(
        "--dataset_root", help="dataset root dir for download/reuse", default="/tmp"
    )
    parser.add_argument(
        "--reportfile",
        help="Name of output .json report file",
        type=str,
        default="validation.json",
    )
    parser.add_argument(
        "--settingsfile", help="Name of optional input .json settings file", type=str, default=""
    )
    # parse arguments
    args = parser.parse_args()
    bsize = args.batchsize
    dataset = args.dataset
    bitfile = args.bitfile
    platform = args.platform
    reportfile = args.reportfile
    settingsfile = args.settingsfile
    dataset_root = args.dataset_root

    # overwrite settings if specified in settings file
    if settingsfile != "":
        with open(settingsfile, "r") as f:
            settings = json.load(f)
            if "validation_dataset" in settings:
                dataset = settings["validation_dataset"]

    # program FPGA and load driver
    PL.reset()  # reset PYNQ cache
    driver = FINNExampleOverlay(
        bitfile_name=bitfile,
        platform=platform,
        io_shape_dict=io_shape_dict,
        batch_size=bsize,
        runtime_weight_dir="runtime_weights/",
    )

    # prepare dataset
    if dataset == "mnist":
        from dataset_loading import mnist

        trainx, trainy, testx, testy, valx, valy = mnist.load_mnist_data(
            dataset_root, download=True, one_hot=False
        )
    elif dataset == "cifar10":
        from dataset_loading import cifar

        trainx, trainy, testx, testy, valx, valy = cifar.load_cifar_data(
            dataset_root, download=True, one_hot=False
        )
    elif dataset == "cifar100":
        from dataset_loading import cifar
        trainx, trainy, testx, testy, valx, valy = cifar.load_cifar_data(
            dataset_root, download=True, one_hot=False, cifar10=False
        )
    elif dataset == "imagenet":
        val_dir = dataset_root + "/ImageNet/2012/val"
        label_file = dataset_root + "/ImageNet/2012/val.txt"
        img_queue = setup_dataloader(val_dir, label_file, bsize)
        total = 50000
    else:
        raise Exception("Unrecognized dataset")

    # run accelerator on dataset
    if dataset in ["mnist", "cifar10", "cifar100"]:
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
            ibuf_normal = test_imgs[i].reshape(driver.ishape_normal())
            exp = test_labels[i]
            obuf_normal = driver.execute(ibuf_normal)
            #obuf_normal = obuf_normal.reshape(bsize, -1)[:,0]
            #TODO: detect automatically if argmax is needed or output is already top-1
            obuf_normal = np.argmax(obuf_normal, axis=1)
            ret = np.bincount(obuf_normal.flatten() == exp.flatten(), minlength=2)
            nok += ret[0]
            ok += ret[1]
            print("batch %d / %d : total OK %d NOK %d" % (i + 1, n_batches, ok, nok))
    elif dataset in ["imagenet"]:
        ok = 0
        nok = 0
        i = 0
        print("Starting validation..")
        while not img_queue.last_batch:
            imgs, lbls = img_queue.get_batch(bsize, timeout=None)
            imgs = np.array(imgs)
            exp = np.array(lbls)
            ibuf_normal = imgs.reshape(driver.ishape_normal())
            obuf_normal = driver.execute(ibuf_normal)
            #obuf_normal = obuf_normal.reshape(bsize, -1)[:,0]
            #TODO: detect automatically if argmax is needed or output is already top-1
            obuf_normal = np.argmax(obuf_normal, axis=1)
            ret = np.bincount(obuf_normal.flatten() == exp.flatten(), minlength=2)
            nok += ret[0]
            ok += ret[1]
            i += 1
            print("batch %d : total OK %d NOK %d" % (i, ok, nok))

    # calculate top-1 accuracy
    acc = 100.0 * ok / (total)
    print("Final accuracy: %f" % acc)

    # write report to file
    report = {
        "top-1_accuracy": acc,
    }
    with open(reportfile, "w") as f:
        json.dump(report, f, indent=2)

"""Validation script for CIFAR-10/100 datasets."""

import functools
import numpy as np
import os

from finn_plus_driver.validate.common import ArrayDataset, run_validation, validation_kwargs

#: Per-channel mean and standard deviation of the CIFAR-10 training set pixels scaled to [0, 1].
#: These are the constants that CIFAR training scripts conventionally normalize with, and the
#: CIFAR-100 ResNet-18 regression model was trained with them as well (it scores 70.9% top-1 with
#: these, 70.4% with the CIFAR-100 set's own statistics and 1% with raw pixel values).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def normalize_batch(batch, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """Scale a channels-last uint8 image batch to [0, 1] and normalize it per channel.

    Returns a float32 array of the same shape with ``(x / 255 - mean) / std`` applied along
    the last (channel) axis, which is what models exported with a float input quantizer expect.
    """
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    x = np.asarray(batch).astype(np.float32) / 255.0
    return (x - mean) / std


def validate(cls_inst, *args, **kwargs):
    """Run CIFAR dataset validation and report top-1 accuracy.

    Keyword arguments (beyond the ones shared by all validators):

    * ``normalize``: apply :func:`normalize_batch` to the inputs. Defaults to ``True`` for
      CIFAR-100 and ``False`` for CIFAR-10, because the CIFAR-10 accelerators in use (the
      BNN-PYNQ CNVs) take raw ``UINT8`` pixels with the scaling folded into the model, while
      the CIFAR-100 ResNet-18 takes normalized float inputs.
    * ``norm_mean`` / ``norm_std``: per-channel constants for the normalization, defaulting to
      the CIFAR-10 statistics :data:`CIFAR10_MEAN` / :data:`CIFAR10_STD`.
    """
    from dataset_loading import cifar

    report_dir = kwargs.get("report_dir")
    dataset_path = kwargs.get("dataset_path", os.path.dirname(os.path.realpath(__file__)))
    # Dataset name "cifar" selects CIFAR-10, "cifar100" selects CIFAR-100 (fine labels)
    cifar10 = kwargs.get("validation_dataset") != "cifar100"
    normalize = kwargs.get("normalize", not cifar10)

    trainx, trainy, testx, testy, valx, valy = cifar.load_cifar_data(
        dataset_path, download=True, one_hot=False, cifar10=cifar10
    )

    transform = None
    if normalize:
        mean = kwargs.get("norm_mean", CIFAR10_MEAN)
        std = kwargs.get("norm_std", CIFAR10_STD)
        print("Normalizing inputs with mean %s and std %s" % (tuple(mean), tuple(std)))
        transform = functools.partial(normalize_batch, mean=mean, std=std)

    print("Starting validation..")
    dataset = ArrayDataset(testx, testy, transform=transform)
    run_validation(cls_inst, dataset, report_dir, **validation_kwargs(kwargs))

"""Validation script for the RadioML 2018.01A dataset.

Serves two dataset names that differ in the samples they validate on, which follows from how
the respective model was trained:

* ``radioml``: the split of the VGG10 model, see :func:`select_test_indices`
* ``radioml_transformer``: the evaluation split of the finn-transformers RadioML model, see
  :func:`select_transformer_eval_indices`
"""

import h5py
import math
import numpy as np
import os

from finn_plus_driver.validate.common import ValidationDataset, run_validation, validation_kwargs


def quantize(data):
    """Quantize float data to int8 in the range [-2, 2]."""
    quant_min = -2.0
    quant_max = 2.0
    quant_range = quant_max - quant_min
    data_quant = (data - quant_min) / quant_range
    data_quant = np.round(data_quant * 256) - 128
    data_quant = np.clip(data_quant, -128, 127)
    data_quant = data_quant.astype(np.int8)
    return data_quant


def select_test_indices():
    """Assemble the (sorted) list of high-SNR test set indices into the HDF5 file."""
    # do not pre-load large dataset into memory
    np.random.seed(2018)
    test_indices = []
    for mod in range(24):  # all modulations (0 to 23)
        for snr_idx in range(26):  # all SNRs (0 to 25 = -20dB to +30dB)
            start_idx = 26 * 4096 * mod + 4096 * snr_idx
            indices_subclass = list(range(start_idx, start_idx + 4096))

            split = int(np.ceil(0.1 * 4096))  # 90%/10% split
            np.random.shuffle(indices_subclass)
            val_indices_subclass = indices_subclass[:split]

            if snr_idx >= 25:  # select which SNRs to test on
                test_indices.extend(val_indices_subclass)
    return sorted(test_indices)


def select_transformer_eval_indices():
    """Assemble the (sorted) evaluation split of the finn-transformers RadioML case study.

    Reproduces ``get_datasets()`` of https://github.com/iksnagreb/finn-transformers
    (radioml/dataset.py with radioml/params.yaml): all samples from -6 dB upwards are split
    into 80% training, 10% validation and 10% evaluation data, by cutting a seeded permutation
    of the samples twice. The model (revision 3a21c22) is published with 70.76% top-1 accuracy
    on this split. Its 8-bit output scores tie for the first place on a tenth of the samples,
    though, and with ties going to the lower class index, as with ``np.argmax``, it reaches
    71.31%.
    """
    seed = 12
    # all modulations (0 to 23), SNRs from index 7 (-6 dB) to 25 (+30 dB), in file order
    indices = np.arange(24 * 26 * 4096).reshape(24, 26, 4096)[:, 7:].flatten()
    # first cut: 80% training data, the rest is cut again below
    indices = np.random.default_rng(seed).permuted(indices)
    indices = indices[int(0.80 * len(indices)) :]
    # second cut: validation and evaluation data are 10% of the whole each, the fraction is
    # computed as in the original to end up with the very same split point
    indices = np.random.default_rng(seed).permuted(indices)
    indices = indices[int(0.10 / (1.0 - 0.80) * len(indices)) :]
    return sorted(indices.tolist())


class RadioMLDataset(ValidationDataset):
    """Test samples of RadioML 2018.01A, read on demand from the HDF5 file."""

    def __init__(self, dataset_path, test_indices, quantize_inputs=True):
        """Open the HDF5 file and store the selection of test samples.

        ``quantize_inputs`` selects between :func:`quantize`-d int8 inputs and the float
        samples as stored in the file.
        """
        self.h5_file = h5py.File(dataset_path, "r", locking=False)
        self.data_h5 = self.h5_file["X"]
        self.label_mod = np.argmax(self.h5_file["Y"], axis=1)  # comes in one-hot encoding
        self.test_indices = np.array(test_indices)
        self.quantize_inputs = quantize_inputs

    def __len__(self):
        """Number of test samples."""
        return len(self.test_indices)

    def sample_id(self, index):
        """Index of the sample in the HDF5 file."""
        return str(int(self.test_indices[index]))

    def iter_batches(self, cls_inst):
        """Yield the test samples in batches, shrinking the driver batch size for the last one."""
        total = len(self)
        batch_size = cls_inst.batch_size
        for i_batch in range(math.ceil(total / batch_size)):
            i_frame = i_batch * batch_size
            if i_frame + batch_size > total:
                # last, partial batch: shrink the driver batch size (restored by run_validation)
                batch_size = total - i_frame
                cls_inst.batch_size = batch_size
            indices = np.arange(i_frame, i_frame + batch_size)
            yield indices, self.load(cls_inst, indices), self.label_mod[self.test_indices[indices]]

    def load(self, cls_inst, indices):
        """Read (and quantize) the given samples into one accelerator input batch."""
        h5_indices = self.test_indices[np.asarray(indices)]
        # h5py fancy indexing requires increasing, unique indices
        unique, inverse = np.unique(h5_indices, return_inverse=True)
        data = self.data_h5[unique.tolist()][inverse]
        if self.quantize_inputs:
            data = quantize(data)
        return data.reshape(cls_inst.ishape_normal(0))

    def predict(self, obuf, batch_size):
        """Predicted modulation class per sample: argmax over scores, or the top-1 class itself
        if the accelerator computes it in hardware."""
        obuf = obuf.reshape(batch_size, -1)
        if obuf.shape[1] > 1:
            return np.argmax(obuf, axis=1)
        return obuf[:, 0].astype(int)


def validate(cls_inst, *args, **kwargs):
    """Run RadioML validation and report top-1 accuracy on the test samples of the dataset.

    ``radioml`` validates on the high-SNR test samples of the VGG10 model,
    ``radioml_transformer`` on the evaluation split of the finn-transformers model. The
    samples are quantized to int8 for accelerators with an integer input (VGG10) and passed
    as they are to accelerators with a float input (transformer), whose input quantizer is
    part of the hardware.
    """
    report_dir = kwargs.get("report_dir")
    dataset_path = kwargs.get("dataset_path") or os.path.join(
        os.environ["DATASET_DIR"], "GOLD_XYZ_OSC.0001_1024.hdf5"
    )
    if kwargs.get("validation_dataset") == "radioml_transformer":
        test_indices = select_transformer_eval_indices()
    else:
        test_indices = select_test_indices()
    dataset = RadioMLDataset(dataset_path, test_indices, cls_inst.idt().is_integer())
    run_validation(cls_inst, dataset, report_dir, **validation_kwargs(kwargs))

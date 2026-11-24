"""Validation script for the TinyStories dataset (masked language modeling)."""

import numpy as np
import os

from finn_plus_driver.validate.common import ValidationDataset, run_validation, validation_kwargs

#: The validation split of TinyStories, tokenized with the tokenizer of the finn-transformers
#: language model and cut into sequences of the model's context length. ``input_ids`` holds
#: the token ids of every sequence with 15% of the tokens masked, ``labels`` the original
#: token at the masked positions and :data:`IGNORE_INDEX` everywhere else.
#: Created by scripts/prepare_validation_datasets.py in the finn-plus repository.
DATASET_FILE = "tinystories_validation_mlm.npz"

#: Label of the tokens that are not to be predicted
IGNORE_INDEX = -100


class MaskedTokenDataset(ValidationDataset):
    """Token sequences in which some tokens are masked and have to be predicted.

    The accelerator processes whole sequences and returns scores over the vocabulary for
    every token of a sequence, but only the masked tokens count. Each masked token is therefore
    one sample of this dataset, so that the reported top-1 accuracy is the fraction of masked
    tokens that were predicted correctly.
    """

    def __init__(self, input_ids, labels):
        """Store the ``(num_sequences, sequence_length)`` token ids and labels."""
        assert input_ids.shape == labels.shape, "Shapes of token ids and labels differ"
        self.input_ids = input_ids
        # sequence and position within the sequence of every sample, in sequence order
        self.sequences, self.positions = np.nonzero(labels != IGNORE_INDEX)
        self.labels = labels[self.sequences, self.positions]
        # index of the first sample of every sequence (and the end of the last one)
        self.first_sample = np.searchsorted(self.sequences, np.arange(len(input_ids) + 1))
        # row and position of the samples within the batch that was handed out last
        self._selected = None

    def __len__(self):
        """Number of masked tokens."""
        return len(self.labels)

    def sample_id(self, index):
        """Sequence and position of the masked token."""
        return "sequence %d token %d" % (self.sequences[index], self.positions[index])

    def iter_batches(self, cls_inst):
        """Yield the sequences in batches, shrinking the driver batch size for the last one."""
        num_sequences = len(self.input_ids)
        batch_size = cls_inst.batch_size
        for start in range(0, num_sequences, batch_size):
            stop = min(start + batch_size, num_sequences)
            if stop - start != batch_size:
                # last, partial batch: shrink the driver batch size (restored by run_validation)
                cls_inst.batch_size = stop - start
            indices = np.arange(self.first_sample[start], self.first_sample[stop])
            self._selected = (self.sequences[indices] - start, self.positions[indices])
            inputs = self._inputs(cls_inst, np.arange(start, stop))
            yield indices, inputs, self.labels[indices]

    def load(self, cls_inst, indices):
        """Return one sequence per given sample (the sequence the masked token is part of)."""
        indices = np.asarray(indices)
        self._selected = (np.arange(len(indices)), self.positions[indices])
        return self._inputs(cls_inst, self.sequences[indices])

    def _inputs(self, cls_inst, sequences):
        """Token ids of the given sequences as one accelerator input batch."""
        # uint64 is the container of the accelerator's token input, which packs fastest
        return self.input_ids[sequences].astype(np.uint64).reshape(cls_inst.ishape_normal())

    def predict(self, obuf, batch_size):
        """Top-1 token at the masked positions of the batch that was handed out last."""
        rows, positions = self._selected
        scores = obuf.reshape(-1, self.input_ids.shape[1], obuf.shape[-1])
        return np.argmax(scores[rows, positions], axis=-1)


def validate(cls_inst, *args, **kwargs):
    """Run TinyStories validation and report the top-1 accuracy of the masked tokens."""
    report_dir = kwargs.get("report_dir")
    dataset_path = kwargs.get("dataset_path") or os.path.join(
        os.environ["DATASET_DIR"], DATASET_FILE
    )
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"{dataset_path} not found, create it with scripts/prepare_validation_datasets.py"
        )
    data = np.load(dataset_path)

    dataset = MaskedTokenDataset(data["input_ids"], data["labels"])
    run_validation(cls_inst, dataset, report_dir, **validation_kwargs(kwargs))

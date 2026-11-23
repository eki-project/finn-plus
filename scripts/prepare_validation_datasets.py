"""Prepare the validation datasets that the PYNQ driver expects to find in ``$DATASET_DIR``.

The driver validates accelerators on the board against the dataset their model was trained on
(see ``driver/finn_plus_driver/validate``). MNIST and CIFAR are downloaded by the driver itself;
the datasets below have to be placed into the dataset directory once, which is what this script
does:

- ``speechcommands``: Google Speech Commands v2 validation split as MFCC features, published
  with the finn-examples keyword spotting model. Stored as downloaded.
- ``gtsrb``: test set of the German Traffic Sign Recognition Benchmark scaled to 32x32 pixels,
  the same data the finn-examples notebook validates the GTSRB model on. The test set is
  extracted from the downloaded archive and stored as ``gtsrb_test_32x32.npz``.
- ``tinystories``: validation split of TinyStories for the finn-transformers language model,
  which predicts masked tokens. The stories are tokenized with the model's tokenizer, cut into
  sequences of the model's context length and masked like in the model's training, then stored
  as ``tinystories_validation_mlm.npz``. Needs the ``tokenizers`` and ``pyarrow`` packages and
  the tokenizer of the model (``dvc pull`` in the repository).

Usage::

    python scripts/prepare_validation_datasets.py <dataset_dir> [--datasets gtsrb ...]
"""

import argparse
import hashlib
import io
import numpy as np
import os
import pickle
import tempfile
import urllib.request
import zipfile

SPEECHCOMMANDS_URL = (
    "https://github.com/Xilinx/finn-examples/releases/download/kws/"
    "python_speech_preprocessing_all_validation_KWS_data.npz"
)
SPEECHCOMMANDS_SHA256 = "2a4dafd2979250e864bbfe34726dbaf6fd1915a7298257f3a7cde8d8dea7fa7b"
SPEECHCOMMANDS_FILE = "python_speech_preprocessing_all_validation_KWS_data.npz"

GTSRB_URL = (
    "https://d17h27t6h515a5.cloudfront.net/topher/2017/February/"
    "5898cd6f_traffic-signs-data/traffic-signs-data.zip"
)
GTSRB_SHA256 = "0ee14ebdd48b07d73ab9f717c902a233f360eb52c8bf68b554747089809f9c75"
GTSRB_FILE = "gtsrb_test_32x32.npz"

TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/"
    "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/data/"
    "validation-00000-of-00001-869c898b519ad725.parquet"
)
TINYSTORIES_SHA256 = "33406a6206554cfc279c29c11f4df51528af487aa1a602b075566fc83c49dcab"
TINYSTORIES_FILE = "tinystories_validation_mlm.npz"
TINYSTORIES_TOKENIZER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models/transformer/finn-transformers/language/tokenizer/tokenizer.json",
)
# As in language/params.yaml of https://github.com/iksnagreb/finn-transformers
TINYSTORIES_CONTEXT_LENGTH = 256
TINYSTORIES_MLM_PROBABILITY = 0.15
TINYSTORIES_SEED = 0


def download(url, path, sha256):
    """Download a file and check it against its SHA-256 checksum."""
    print(f"Downloading {url}")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, open(path, "wb") as f:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            f.write(chunk)
    if digest.hexdigest() != sha256:
        os.remove(path)
        raise SystemExit(f"Checksum mismatch for {url}: got {digest.hexdigest()}")


def prepare_speechcommands(dataset_dir, args):
    """Fetch the pre-processed Speech Commands v2 validation split."""
    path = os.path.join(dataset_dir, SPEECHCOMMANDS_FILE)
    download(SPEECHCOMMANDS_URL, path, SPEECHCOMMANDS_SHA256)
    data = np.load(path)
    print(f"Wrote {path}: {len(data['label_arr'])} samples")


def prepare_gtsrb(dataset_dir, args):
    """Fetch the 32x32 GTSRB archive and convert its test set to an .npz file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        archive = os.path.join(tmp_dir, "traffic-signs-data.zip")
        download(GTSRB_URL, archive, GTSRB_SHA256)
        with zipfile.ZipFile(archive) as z:
            # unpickling is only safe because the archive matched its checksum
            test = pickle.load(io.BytesIO(z.read("test.p")))
    features = np.asarray(test["features"], dtype=np.uint8)
    labels = np.asarray(test["labels"], dtype=np.uint8)
    assert features.shape == (12630, 32, 32, 3) and labels.shape == (12630,)
    path = os.path.join(dataset_dir, GTSRB_FILE)
    np.savez_compressed(path, features=features, labels=labels)
    print(f"Wrote {path}: {len(labels)} samples")


def mask_tokens(ids, special_ids, mask_id, vocab_size, rng):
    """Mask tokens the way the DataCollatorForLanguageModeling of the model's training does.

    Each token that is not a special token is selected with the masking probability. Of the
    selected tokens, 80% are replaced by the mask token, 10% by a random token and 10% are
    kept. Returns the resulting token ids and the labels: the original token where a token
    was selected and -100 elsewhere.
    """
    selected = (rng.random(ids.shape) < TINYSTORIES_MLM_PROBABILITY) & ~np.isin(ids, special_ids)
    labels = np.where(selected, ids, -100)
    action = rng.random(ids.shape)
    input_ids = ids.copy()
    input_ids[selected & (action < 0.8)] = mask_id
    randomized = selected & (action >= 0.9)
    input_ids[randomized] = rng.integers(0, vocab_size, size=np.count_nonzero(randomized))
    return input_ids, labels


def prepare_tinystories(dataset_dir, args):
    """Fetch the TinyStories validation split and turn it into masked token sequences."""
    try:
        import pandas as pd
        from tokenizers import Tokenizer
    except ImportError as e:
        raise SystemExit(f"tinystories needs the pandas, pyarrow and tokenizers packages: {e}")
    if not os.path.isfile(args.tokenizer):
        raise SystemExit(f"Tokenizer {args.tokenizer} not found, run dvc pull or set --tokenizer")
    tokenizer = Tokenizer.from_file(args.tokenizer)

    with tempfile.TemporaryDirectory() as tmp_dir:
        parquet = os.path.join(tmp_dir, "validation.parquet")
        download(TINYSTORIES_URL, parquet, TINYSTORIES_SHA256)
        stories = pd.read_parquet(parquet)["text"].tolist()

    # As in the training: every story is enclosed in [CLS] and [SEP] by the tokenizer, all
    # stories are concatenated and cut into sequences. The incomplete last sequence is dropped.
    ids = np.concatenate(
        [np.asarray(e.ids, dtype=np.int16) for e in tokenizer.encode_batch(stories)]
    )
    num_sequences = len(ids) // TINYSTORIES_CONTEXT_LENGTH
    ids = ids[: num_sequences * TINYSTORIES_CONTEXT_LENGTH].reshape(num_sequences, -1)

    special_ids = [tokenizer.token_to_id(t) for t in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")]
    input_ids, labels = mask_tokens(
        ids,
        special_ids,
        tokenizer.token_to_id("[MASK]"),
        tokenizer.get_vocab_size(),
        np.random.default_rng(TINYSTORIES_SEED),
    )
    path = os.path.join(dataset_dir, TINYSTORIES_FILE)
    np.savez_compressed(path, input_ids=input_ids, labels=labels)
    print(f"Wrote {path}: {num_sequences} sequences, {np.count_nonzero(labels >= 0)} masked tokens")


DATASETS = {
    "speechcommands": prepare_speechcommands,
    "gtsrb": prepare_gtsrb,
    "tinystories": prepare_tinystories,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset_dir", help="directory that the board sees as $DATASET_DIR")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument(
        "--tokenizer",
        default=TINYSTORIES_TOKENIZER,
        help="tokenizer.json of the language model, for tinystories (default: %(default)s)",
    )
    args = parser.parse_args()
    os.makedirs(args.dataset_dir, exist_ok=True)
    for name in args.datasets:
        DATASETS[name](args.dataset_dir, args)

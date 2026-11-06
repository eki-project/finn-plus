"""Image classification data for NAS experiments, held as tensors on the training device.

The candidate networks are tiny and are trained for only a few epochs each, but there are many
of them. Keeping the whole dataset on the device and augmenting in batches there avoids data
loader worker processes entirely, which is what dominates the run time for such small models.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImageDataset:
    """A labelled image dataset split into train, validation and test parts.

    Images are stored as ``uint8`` tensors of shape (N, C, H, W); :func:`to_input` converts a
    batch to the float representation the networks consume. The validation part is split off
    the official training set so that architecture selection never touches the test set.
    """

    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    num_classes: int

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """Shape (C, H, W) of one image."""
        return tuple(self.train_x.shape[1:])  # type: ignore[return-value]

    def to(self, device: torch.device | str) -> ImageDataset:
        """Return a copy with all tensors on ``device``."""
        return ImageDataset(
            *(
                t.to(device)
                for t in (
                    self.train_x,
                    self.train_y,
                    self.val_x,
                    self.val_y,
                    self.test_x,
                    self.test_y,
                )
            ),
            num_classes=self.num_classes,
        )


def to_input(images: torch.Tensor) -> torch.Tensor:
    """Convert ``uint8`` images to the network input: float values in [0, 1].

    No mean/std normalization on purpose: the accelerator then consumes the raw 8 bit pixels
    (scaled by a power-of-two-free constant that the input quantizer absorbs), and the first
    layer's quantizer and batch norm take over the role of normalization.
    """
    return images.float().div_(255.0)


def split_train_val(
    images: torch.Tensor, labels: torch.Tensor, val_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split off ``val_size`` random samples as validation set, deterministically for ``seed``."""
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(images.shape[0], generator=generator)
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    return images[train_idx], labels[train_idx], images[val_idx], labels[val_idx]


def load_cifar10(
    root: str | Path,
    device: torch.device | str = "cpu",
    val_size: int = 5000,
    seed: int = 0,
    download: bool = True,
) -> ImageDataset:
    """Load CIFAR-10 (downloading it to ``root`` on first use) as an :class:`ImageDataset`."""
    from torchvision.datasets import CIFAR10

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train = CIFAR10(str(root), train=True, download=download)
    test = CIFAR10(str(root), train=False, download=download)
    train_x = torch.from_numpy(train.data).permute(0, 3, 1, 2).contiguous()
    train_y = torch.tensor(train.targets, dtype=torch.long)
    test_x = torch.from_numpy(test.data).permute(0, 3, 1, 2).contiguous()
    test_y = torch.tensor(test.targets, dtype=torch.long)
    train_x, train_y, val_x, val_y = split_train_val(train_x, train_y, val_size, seed)
    dataset = ImageDataset(train_x, train_y, val_x, val_y, test_x, test_y, num_classes=10)
    return dataset.to(device)


def augment(images: torch.Tensor, generator: torch.Generator, pad: int = 4) -> torch.Tensor:
    """Random crop (after reflection padding) and random horizontal flip of a ``uint8`` batch.

    Vectorized over the batch so it can run on the device the images already live on.
    """
    n, _, h, w = images.shape
    device = images.device
    padded = torch.nn.functional.pad(images.float(), (pad, pad, pad, pad), mode="reflect")
    dy = torch.randint(0, 2 * pad + 1, (n,), generator=generator, device=device)
    dx = torch.randint(0, 2 * pad + 1, (n,), generator=generator, device=device)
    flip = torch.rand(n, generator=generator, device=device) < 0.5
    rows = (torch.arange(h, device=device)[None, :] + dy[:, None])[:, :, None]
    cols = torch.arange(w, device=device)[None, :].expand(n, w)
    cols = torch.where(flip[:, None], w - 1 - cols, cols) + dx[:, None]
    cols = cols[:, None, :]
    batch = torch.arange(n, device=device)[:, None, None]
    # advanced indexing yields (N, H, W, C); move channels back to position 1
    cropped = padded[batch, :, rows, cols].permute(0, 3, 1, 2)
    return cropped.to(images.dtype)


def proxy_batch(
    dataset: ImageDataset, batch_size: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a fixed random training minibatch (network inputs and labels) for zero-cost proxies."""
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(dataset.train_x.shape[0], generator=generator)[:batch_size]
    idx = idx.to(dataset.train_x.device)
    return to_input(dataset.train_x[idx]), dataset.train_y[idx]

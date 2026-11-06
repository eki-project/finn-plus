"""Short quantization-aware training runs for candidate networks.

Used for the ground truth that zero-cost proxies are compared against and for training the
final candidates of a search. The defaults favour robustness over peak accuracy, because the
same recipe has to work for arbitrary sampled architectures without per-network tuning.
"""

from __future__ import annotations

import math
import time
import torch
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from finn.nas.data import augment, to_input

if TYPE_CHECKING:
    from finn.nas.data import ImageDataset


@dataclass
class TrainConfig:
    """Hyperparameters of one training run."""

    epochs: int = 10
    batch_size: int = 256
    #: Peak learning rate of the one-cycle schedule (linear warm-up, cosine decay to zero).
    lr: float = 2e-3
    warmup_fraction: float = 0.15
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    augment: bool = True
    seed: int = 0
    #: Stop early once this wall-clock budget is exceeded (checked after each epoch).
    max_seconds: float | None = None
    eval_batch_size: int = 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class TrainResult:
    """Outcome of one training run."""

    val_acc: list[float] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    final_val_acc: float = float("nan")
    best_val_acc: float = float("nan")
    test_acc: float | None = None
    epochs_run: int = 0
    seconds: float = 0.0
    #: The loss became non-finite; training was stopped.
    diverged: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


def lr_at(step: int, total_steps: int, cfg: TrainConfig) -> float:
    """Learning rate of the one-cycle schedule at optimizer step ``step``."""
    warmup = max(1, int(cfg.warmup_fraction * total_steps))
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def accuracy(
    model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor, batch_size: int = 1000
) -> float:
    """Top-1 accuracy of ``model`` (put into eval mode) on ``uint8`` images."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    try:
        for start in range(0, images.shape[0], batch_size):
            x = to_input(images[start : start + batch_size].to(device))
            y = labels[start : start + batch_size].to(device)
            correct += int((model(x).argmax(dim=1) == y).sum().item())
    finally:
        model.train(was_training)
    return correct / max(1, images.shape[0])


def train(
    model: torch.nn.Module,
    dataset: ImageDataset,
    cfg: TrainConfig | None = None,
    device: torch.device | str | None = None,
    eval_test: bool = False,
) -> TrainResult:
    """Train ``model`` in place on ``dataset.train_*`` and evaluate on the validation split.

    The test split is only touched if ``eval_test`` is set (for final, already selected
    networks). Failures of the network itself (non-finite loss, runtime errors such as out
    of memory) are reported in the result instead of raised, since a search must survive
    individual bad candidates.
    """
    cfg = cfg or TrainConfig()
    if device is None:
        device = dataset.train_x.device
    device = torch.device(device)
    result = TrainResult()
    t_start = time.perf_counter()
    try:
        model.to(device)
        torch.manual_seed(cfg.seed)
        generator = torch.Generator(device=dataset.train_x.device).manual_seed(cfg.seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        n = dataset.train_x.shape[0]
        steps_per_epoch = max(1, math.ceil(n / cfg.batch_size))
        total_steps = cfg.epochs * steps_per_epoch
        step = 0
        for _epoch in range(cfg.epochs):
            model.train()
            perm = torch.randperm(n, generator=generator, device=dataset.train_x.device)
            running, batches = 0.0, 0
            for start in range(0, n, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]
                if idx.shape[0] < 2:
                    continue  # batch norm cannot handle a single sample
                images = dataset.train_x[idx]
                if cfg.augment:
                    images = augment(images, generator)
                x = to_input(images.to(device))
                y = dataset.train_y[idx].to(device)
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step, total_steps, cfg)
                loss = loss_fn(model(x), y)
                if not torch.isfinite(loss):
                    result.diverged = True
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                step += 1
                running += float(loss.item())
                batches += 1
            if result.diverged:
                break
            result.train_loss.append(running / max(1, batches))
            result.val_acc.append(
                accuracy(model, dataset.val_x, dataset.val_y, cfg.eval_batch_size)
            )
            result.epochs_run += 1
            if cfg.max_seconds is not None and time.perf_counter() - t_start > cfg.max_seconds:
                break
        if result.val_acc:
            result.final_val_acc = result.val_acc[-1]
            result.best_val_acc = max(result.val_acc)
        if eval_test and not result.diverged:
            result.test_acc = accuracy(model, dataset.test_x, dataset.test_y, cfg.eval_batch_size)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"[:500]
    result.seconds = time.perf_counter() - t_start
    return result

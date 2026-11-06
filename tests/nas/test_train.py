"""Tests for the data handling and the short training loop, on synthetic data (no download)."""

import pytest

import torch
from torch import nn

from finn.nas.data import ImageDataset, augment, proxy_batch, split_train_val, to_input
from finn.nas.train import TrainConfig, accuracy, lr_at, train


def _synthetic_dataset(n_train: int = 256, n_eval: int = 64) -> ImageDataset:
    """Two linearly separable classes: dark images versus bright images."""
    generator = torch.Generator().manual_seed(0)

    def make(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.arange(n) % 2
        base = torch.where(labels == 0, 40, 200)[:, None, None, None]
        noise = torch.randint(-30, 31, (n, 3, 8, 8), generator=generator)
        return (base + noise).clamp(0, 255).to(torch.uint8), labels

    train_x, train_y = make(n_train)
    val_x, val_y = make(n_eval)
    test_x, test_y = make(n_eval)
    return ImageDataset(train_x, train_y, val_x, val_y, test_x, test_y, num_classes=2)


def _small_net() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(), nn.Linear(256, 2)
    )


@pytest.mark.util
class TestData:
    """Dataset helpers."""

    def test_to_input_range(self) -> None:
        x = torch.tensor([0, 128, 255], dtype=torch.uint8)
        out = to_input(x)
        assert out.dtype == torch.float32
        assert out.min() == 0.0 and out.max() == 1.0

    def test_split_is_deterministic_and_disjoint(self) -> None:
        images = torch.arange(100, dtype=torch.uint8)[:, None, None, None].expand(100, 1, 2, 2)
        labels = torch.arange(100)
        a = split_train_val(images, labels, 20, seed=1)
        b = split_train_val(images, labels, 20, seed=1)
        assert all(torch.equal(x, y) for x, y in zip(a, b))
        train_y, val_y = a[1], a[3]
        assert len(train_y) == 80 and len(val_y) == 20
        assert set(train_y.tolist()).isdisjoint(val_y.tolist())

    def test_augment_keeps_shape_dtype_and_content(self) -> None:
        generator = torch.Generator().manual_seed(0)
        images = torch.randint(0, 256, (16, 3, 8, 8), dtype=torch.uint8)
        out = augment(images, generator, pad=2)
        assert out.shape == images.shape and out.dtype == torch.uint8
        # a constant image stays constant under crop and flip
        const = torch.full((4, 3, 8, 8), 77, dtype=torch.uint8)
        assert torch.equal(augment(const, generator), const)
        # zero padding offset and no flip is impossible to force, but every output pixel
        # must be a pixel value of the same image
        for i in range(images.shape[0]):
            assert set(out[i].flatten().tolist()) <= set(images[i].flatten().tolist())

    def test_proxy_batch(self) -> None:
        dataset = _synthetic_dataset()
        x1, y1 = proxy_batch(dataset, 32, seed=3)
        x2, y2 = proxy_batch(dataset, 32, seed=3)
        assert x1.shape == (32, 3, 8, 8) and y1.shape == (32,)
        assert torch.equal(x1, x2) and torch.equal(y1, y2)


@pytest.mark.util
class TestTrain:
    """The training loop."""

    def test_lr_schedule(self) -> None:
        cfg = TrainConfig(lr=1.0, warmup_fraction=0.1)
        rates = [lr_at(s, 100, cfg) for s in range(100)]
        assert rates[0] < rates[9] <= 1.0
        assert max(rates) == pytest.approx(1.0)
        assert rates[-1] < 0.01

    def test_learns_separable_task(self) -> None:
        dataset = _synthetic_dataset()
        model = _small_net()
        result = train(
            model, dataset, TrainConfig(epochs=3, batch_size=32, lr=1e-2), eval_test=True
        )
        assert result.error is None and not result.diverged
        assert result.epochs_run == 3 and len(result.val_acc) == 3
        assert result.final_val_acc > 0.95
        assert result.final_val_acc == accuracy(model, dataset.val_x, dataset.val_y)
        assert result.test_acc is not None and result.test_acc > 0.95
        assert result.best_val_acc == max(result.val_acc)

    def test_is_deterministic(self) -> None:
        dataset = _synthetic_dataset()
        cfg = TrainConfig(epochs=1, batch_size=32, seed=5)
        torch.manual_seed(0)
        m1 = _small_net()
        torch.manual_seed(0)
        m2 = _small_net()
        r1, r2 = train(m1, dataset, cfg), train(m2, dataset, cfg)
        assert r1.train_loss == r2.train_loss and r1.val_acc == r2.val_acc

    def test_reports_divergence_and_errors(self) -> None:
        dataset = _synthetic_dataset()

        class Exploding(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(192, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x.flatten(1)) * float("inf")

        result = train(Exploding(), dataset, TrainConfig(epochs=2, batch_size=32))
        assert result.diverged and result.epochs_run == 0

        class Broken(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(5, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x.flatten(1))

        result = train(Broken(), dataset, TrainConfig(epochs=1, batch_size=32))
        assert result.error is not None and "RuntimeError" in result.error

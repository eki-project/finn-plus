"""Tiny Brevitas networks that are each broken in one specific way, plus correct twins.

They exercise the failure classification of the NAS oracle, see ``test_broken_candidates.py``.
All BatchNorm layers get non-trivial per-channel statistics (:func:`randomize_batchnorm`),
because a freshly initialized BatchNorm is a uniform scale that streamlining treats like a
scalar, which hides topologies that stop working once the network is trained.
"""

from __future__ import annotations

import torch
from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear, QuantReLU
from torch import nn

MLP_SHAPE = (1, 16)
CNN_SHAPE = (1, 3, 8, 8)


def randomize_batchnorm(model: nn.Module, seed: int = 0) -> nn.Module:
    """Give every BatchNorm in ``model`` per-channel statistics as a trained network has."""
    gen = torch.Generator().manual_seed(seed)
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            n = module.num_features
            with torch.no_grad():
                module.running_mean.copy_(torch.randn(n, generator=gen) * 0.5)
                module.running_var.copy_(torch.rand(n, generator=gen) + 0.5)
                module.weight.copy_(torch.rand(n, generator=gen) + 0.5)
                module.bias.copy_(torch.randn(n, generator=gen) * 0.5)
    return model


def _bn_act(channels: int, bits: int = 2) -> nn.Sequential:
    """BatchNorm followed by a quantized ReLU: the block streamlining can absorb."""
    return nn.Sequential(nn.BatchNorm2d(channels), QuantReLU(bit_width=bits))


class _Sin(nn.Module):
    """``torch.sin`` as a module."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the sine."""
        return torch.sin(x)


class MLPBase(nn.Module):
    """Quantized two-layer MLP with hooks before (``mid``) and after (``out``) the last layer."""

    input_shape = MLP_SHAPE

    def __init__(
        self,
        mid: nn.Module | None = None,
        out: nn.Module | None = None,
        abits: int = 2,
        quant_input: bool = True,
    ) -> None:
        """Build the network."""
        super().__init__()
        self.inp = QuantIdentity(bit_width=2) if quant_input else nn.Identity()
        self.fc1 = QuantLinear(16, 8, bias=False, weight_bit_width=2)
        self.bn1 = nn.BatchNorm1d(8)
        self.act1 = QuantReLU(bit_width=abits)
        self.mid = mid if mid is not None else nn.Identity()
        self.fc2 = QuantLinear(8, 4, bias=False, weight_bit_width=2)
        self.out = out if out is not None else nn.Identity()
        randomize_batchnorm(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network."""
        return self.out(self.fc2(self.mid(self.act1(self.bn1(self.fc1(self.inp(x)))))))


class GoodMLP(MLPBase):
    """Reference: fully quantized MLP."""


class SinMLP(MLPBase):
    """A float sine between the layers: FINN has no hardware conversion for ``Sin``."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__(mid=_Sin())


class FloatLinearMLP(MLPBase):
    """An unquantized ``nn.Linear`` (float weights) between quantized layers."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__(mid=nn.Sequential(nn.Linear(8, 8, bias=False), QuantReLU(bit_width=2)))


class WideActMLP(MLPBase):
    """Activation bit width (16) above the oracle's multithreshold limit (8)."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__(abits=16)


class SinOutputMLP(MLPBase):
    """A float sine after the last layer: unmapped, but at the graph boundary (host side)."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__(out=_Sin())


class NoInputQuantMLP(MLPBase):
    """Float input fed straight into the first quantized layer (no input quantizer)."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__(quant_input=False)


class CNNBase(nn.Module):
    """Quantized conv - ``mid`` - conv - pool - classifier network with hooks for the defects.

    ``mid`` replaces the BatchNorm + quantized ReLU after the first convolution and ``pool``
    the max pooling; ``fc_in`` must be given if they change the flattened feature size.
    """

    input_shape = CNN_SHAPE

    def __init__(
        self,
        mid: nn.Module | None = None,
        pool: nn.Module | None = None,
        channels: int = 4,
        kernel: int = 3,
        groups: int = 1,
        quant_input: bool = True,
        fc_in: int | None = None,
        trained_bn: bool = True,
    ) -> None:
        """Build the network."""
        super().__init__()
        self.inp = QuantIdentity(bit_width=2) if quant_input else nn.Identity()
        self.conv1 = QuantConv2d(3, channels, kernel, bias=False, weight_bit_width=2)
        self.mid = mid if mid is not None else _bn_act(channels)
        self.conv2 = QuantConv2d(
            channels, channels, 3, groups=groups, bias=False, weight_bit_width=2
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = QuantReLU(bit_width=2)
        self.pool = pool if pool is not None else nn.MaxPool2d(2)
        self.fc = QuantLinear(
            fc_in if fc_in is not None else channels * 2 * 2, 4, bias=False, weight_bit_width=2
        )
        if trained_bn:
            randomize_batchnorm(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network."""
        x = self.mid(self.conv1(self.inp(x)))
        x = self.pool(self.act2(self.bn2(self.conv2(x))))
        return self.fc(x.flatten(1))


class GoodCNN(CNNBase):
    """Reference: fully quantized two-conv network."""


class NoActivationCNN(CNNBase):
    """conv -> conv directly. Legal: the second layer consumes the wider integer accumulator."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__(mid=nn.Identity())


class OverlappingMaxPoolCNN(CNNBase):
    """``MaxPool2d(3, stride=2)``. Legal: FINN builds overlapping windows with an SWG."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__(pool=nn.MaxPool2d(3, stride=2), fc_in=4)


class UnabsorbedBatchNormCNN(CNNBase):
    """conv -> BN -> conv without a quantized activation that could absorb the BatchNorm.

    With ``trained_bn=False`` the BatchNorm keeps its initial identity statistics, and the
    same topology streamlines.
    """

    def __init__(self, trained_bn: bool = True) -> None:
        """Build the network."""
        super().__init__(mid=nn.BatchNorm2d(4), trained_bn=trained_bn)


class FloatReluCNN(CNNBase):
    """A float (unquantized) ``nn.ReLU`` instead of a quantized activation."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__(mid=nn.Sequential(nn.BatchNorm2d(4), nn.ReLU()))


class LeakyReluCNN(CNNBase):
    """A float ``nn.LeakyReLU``, for which FINN has no hardware conversion."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__(mid=nn.Sequential(nn.BatchNorm2d(4), nn.LeakyReLU(0.1)))


class FloatConvCNN(CNNBase):
    """An unquantized 1x1 ``nn.Conv2d`` (float weights) between quantized layers."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__(mid=nn.Sequential(_bn_act(4), nn.Conv2d(4, 4, 1, bias=False), _bn_act(4)))


class OneBitReluCNN(CNNBase):
    """A 1-bit ``QuantReLU``: FINN only accepts bipolar 1-bit activations."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__(mid=_bn_act(4, bits=1))


class GroupedConvCNN(CNNBase):
    """A grouped convolution that is not depthwise (groups=2 of 4 channels)."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__(groups=2)


class UpsampleCNN(CNNBase):
    """Nearest-neighbour ``nn.Upsample`` between the convolutions (exported as ``Resize``)."""

    def __init__(self) -> None:
        """Build the network."""
        up = nn.Upsample(scale_factor=2, mode="nearest")
        super().__init__(mid=nn.Sequential(_bn_act(4), up), fc_in=4 * 5 * 5)


class FloatAvgPoolCNN(CNNBase):
    """A float ``nn.AvgPool2d``; only Brevitas' truncating average pooling is convertible."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__(pool=nn.AvgPool2d(2))


class NoInputQuantCNN(CNNBase):
    """Float input fed straight into a ``QuantConv2d`` (no input quantizer)."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__(quant_input=False)


class KernelTooLargeCNN(CNNBase):
    """First convolution kernel (9) larger than the 8x8 input: PyTorch itself rejects it."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__(kernel=9)


class WideCNN(CNNBase):
    """Structurally fine, but too wide to fit a small device at a high throughput target."""

    input_shape = (1, 3, 16, 16)

    def __init__(self, channels: int = 192) -> None:
        """Build the network."""
        super().__init__(channels=channels, fc_in=channels * 6 * 6)


class ResidualCNN(nn.Module):
    """Residual add of a skip branch and a conv branch, each behind its own 4-bit quantizer.

    FINN's default streamlining does not move scale factors across forks and joins. A
    residual is therefore only accepted if both quantizers have scale 1, so that the
    branches are plain integers: ``max_val=15`` with a constant scale gives 15 / (2^4 - 1).
    ``max_val=None`` uses Brevitas' default learned scale.
    """

    input_shape = CNN_SHAPE

    def __init__(self, skip_max: float | None, branch_max: float | None) -> None:
        """Build the network."""
        super().__init__()
        self.inp = QuantIdentity(bit_width=2)
        self.conv1 = QuantConv2d(3, 4, 3, bias=False, weight_bit_width=2)
        self.bn1 = nn.BatchNorm2d(4)
        self.skip_act = self._act(skip_max)
        self.conv2 = QuantConv2d(4, 4, 3, padding=1, bias=False, weight_bit_width=2)
        self.bn2 = nn.BatchNorm2d(4)
        self.branch_act = self._act(branch_max)
        self.out_act = QuantReLU(bit_width=2)
        self.fc = QuantLinear(4 * 6 * 6, 4, bias=False, weight_bit_width=2)
        randomize_batchnorm(self)

    @staticmethod
    def _act(max_val: float | None) -> QuantReLU:
        """Create a 4-bit quantized ReLU with a constant (``max_val``) or learned scale."""
        if max_val is None:
            return QuantReLU(bit_width=4)
        return QuantReLU(bit_width=4, scaling_impl_type="const", max_val=max_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network."""
        skip = self.skip_act(self.bn1(self.conv1(self.inp(x))))
        branch = self.branch_act(self.bn2(self.conv2(skip)))
        return self.fc(self.out_act(skip + branch).flatten(1))


class IntegerResidualCNN(ResidualCNN):
    """Reference: residual add of two branches with scale 1."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__(skip_max=15.0, branch_max=15.0)


class MismatchedResidualCNN(ResidualCNN):
    """Residual add of two branches with different quantization scales (1 and 0.5)."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__(skip_max=15.0, branch_max=7.5)


class ScaledResidualCNN(ResidualCNN):
    """Residual add of two branches with Brevitas' default (learned, non-unit) scales."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__(skip_max=None, branch_max=None)

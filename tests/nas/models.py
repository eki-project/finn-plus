"""Tiny Brevitas networks used by the NAS oracle tests."""

from __future__ import annotations

import torch
from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear, QuantReLU
from torch import nn


class TinyMLP(nn.Module):
    """Two quantized fully connected layers (TFC-like)."""

    input_shape = (1, 16)

    def __init__(self, hidden: int = 8, wbits: int = 2, abits: int = 2) -> None:
        """Build the network."""
        super().__init__()
        self.inp = QuantIdentity(bit_width=abits)
        self.fc1 = QuantLinear(16, hidden, bias=False, weight_bit_width=wbits)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act1 = QuantReLU(bit_width=abits)
        self.fc2 = QuantLinear(hidden, 4, bias=False, weight_bit_width=wbits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network."""
        return self.fc2(self.act1(self.bn1(self.fc1(self.inp(x)))))


class BadMLP(TinyMLP):
    """TinyMLP with a float Sigmoid between the layers, which leaves the second layer with a
    non-quantized input that FINN cannot map to a dataflow layer."""

    def __init__(self) -> None:
        """Build the network."""
        super().__init__()
        self.bad = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network."""
        return self.fc2(self.bad(self.act1(self.bn1(self.fc1(self.inp(x))))))


class TinyCNN(nn.Module):
    """One quantized conv layer, pooling and a classifier (CNV-like)."""

    input_shape = (1, 3, 8, 8)

    def __init__(self, channels: int = 4, wbits: int = 2, abits: int = 2) -> None:
        """Build the network."""
        super().__init__()
        self.inp = QuantIdentity(bit_width=abits)
        self.conv = QuantConv2d(3, channels, 3, bias=False, weight_bit_width=wbits)
        self.bn = nn.BatchNorm2d(channels)
        self.act = QuantReLU(bit_width=abits)
        self.pool = nn.MaxPool2d(2)
        self.fc = QuantLinear(channels * 3 * 3, 4, bias=False, weight_bit_width=wbits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network."""
        x = self.pool(self.act(self.bn(self.conv(self.inp(x)))))
        return self.fc(x.flatten(1))


class WideInputMLP(nn.Module):
    """MLP whose first layer has 2048 inputs.

    At a low throughput target ``SetFolding`` leaves that layer at SIMD=1, which the HLS
    matrix-vector unit rejects at code generation time (it requires SIMD >= MW / 1024).
    """

    input_shape = (1, 2048)

    def __init__(self) -> None:
        """Build the network."""
        super().__init__()
        self.inp = QuantIdentity(bit_width=2)
        self.fc1 = QuantLinear(2048, 8, bias=False, weight_bit_width=2)
        self.act1 = QuantReLU(bit_width=2)
        self.fc2 = QuantLinear(8, 4, bias=False, weight_bit_width=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network."""
        return self.fc2(self.act1(self.fc1(self.inp(x))))

"""Genotype -> Brevitas network. The only module of this package that imports torch.

Quantizers follow the BNN-PYNQ reference models (``brevitas_examples.bnn_pynq``:
``CommonWeightQuant`` / ``CommonActQuant``), which FINN is known to export and streamline:
constant scales, bipolar (binary) quantization at 1 bit, narrow-range integers otherwise.
Constant scales also mean that the exported graph structure, and with it FINN's verdict,
depends only on the genotype and not on what training does to learned scale factors.
"""

from __future__ import annotations

import torch
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import FloatToIntImplType, RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear, QuantReLU, TruncAvgPool2d
from brevitas.quant.solver import ActQuantSolver, WeightQuantSolver
from brevitas.quant_tensor import IntQuantTensor
from dataclasses import dataclass
from functools import reduce
from math import ceil, log2
from torch import nn
from typing import TYPE_CHECKING

from finn.nas.space.shapes import (
    Shape,
    aggregate,
    apply_terminal,
    infer_shapes,
    is_image,
    same_padding,
)

if TYPE_CHECKING:
    from finn.nas.space.genotype import Genotype, Node


class _ConstQuant(ExtendedInjector):
    """Settings shared by all quantizers (same as BNN-PYNQ's ``CommonQuant``)."""

    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_per_output_channel = False
    narrow_range = True
    signed = True


class WeightQuant(_ConstQuant, WeightQuantSolver):
    """Weight quantizer with constant scale: weights live in [-1, 1]."""

    scaling_const = 1.0


class ActQuant(_ConstQuant, ActQuantSolver):
    """Activation quantizer with constant range [-1, 1]."""

    min_val = -1.0
    max_val = 1.0


@dataclass(frozen=True)
class IntFormat:
    """Integer format of a tensor that comes straight out of a multi-bit quantizer.

    The builder tracks it through the operations that keep a tensor integer (identity, max
    pooling, truncating average pooling) because an average pooling needs it, see
    :class:`TruncAvgPool`. ``None`` stands for "not known to be integer".
    """

    bits: int
    signed: bool
    scale: float


def _quant_type(bits: int) -> QuantType:
    """Return binary (bipolar, values +-1) quantization for 1 bit and integer otherwise."""
    return QuantType.BINARY if bits == 1 else QuantType.INT


def input_quantizer(bits: int) -> tuple[nn.Module, IntFormat | None]:
    """Return the quantizer in front of the body and the format of its output.

    More than one bit: signed fixed point in [-1, 1) with a power-of-two scale (the Q1.7
    input format of BNN-PYNQ's CNV for 8 bits), so inputs are expected to be normalized to
    that range. One bit: the sign of the input.
    """
    if bits == 1:
        return QuantIdentity(act_quant=ActQuant, bit_width=1, quant_type=QuantType.BINARY), None
    scale = 2.0 ** (1 - bits)
    quantizer = QuantIdentity(
        act_quant=ActQuant,
        bit_width=bits,
        quant_type=QuantType.INT,
        min_val=-1.0,
        max_val=1.0 - scale,
        narrow_range=False,
        restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
    )
    return quantizer, IntFormat(bits, True, scale)


def quant_activation(bits: int) -> tuple[nn.Module, IntFormat | None]:
    """Return a quantized activation and the format of its output.

    One bit is a bipolar sign activation. More bits are an unsigned quantized ReLU that
    saturates at ``2**bits - 1``, which makes its constant scale exactly 1: the activations
    are the integers ``0 .. 2**bits - 1`` (as the bipolar ones are +-1, and as BNN-PYNQ's
    2-bit activations are -1, 0, 1). A preceding batch normalization adapts the range. FINN's
    default streamlining cannot move a scale factor across a fork or a join, so with any
    other range no multi-bit activation could feed a branching node.
    """
    if bits == 1:
        return QuantIdentity(act_quant=ActQuant, bit_width=1, quant_type=QuantType.BINARY), None
    activation = QuantReLU(
        act_quant=ActQuant,
        bit_width=bits,
        quant_type=QuantType.INT,
        signed=False,
        narrow_range=False,
        min_val=0.0,
        max_val=float(2**bits - 1),
    )
    return activation, IntFormat(bits, False, 1.0)


class TruncAvgPool(nn.Module):
    """Average pooling of an integer tensor with an integer result of the same bit width.

    Hardware cannot divide by the window size, so FINN only maps Brevitas'
    ``TruncAvgPool2d``: sum the window, then drop the low-order bits. That layer needs the
    integer format of its input as a ``QuantTensor``; the format is known statically here, so
    it is attached locally instead of passing ``QuantTensor`` objects through the whole
    network (plain tensors keep add/concat and the non-Brevitas layers simple).
    """

    def __init__(self, kernel: int | tuple[int, int], fmt: IntFormat) -> None:
        """Create a pooling with stride = ``kernel`` for inputs of format ``fmt``."""
        super().__init__()
        self.fmt = fmt
        self.pool = TruncAvgPool2d(
            kernel_size=kernel,
            stride=kernel,
            bit_width=fmt.bits,
            float_to_int_impl_type="FLOOR",
            return_quant_tensor=False,
        )

    @property
    def out_format(self) -> IntFormat:
        """Format of the result: same bits, scale grown by 2**dropped_bits / window size.

        Mirrors Brevitas: the window sum of unsigned ``bits``-bit values needs
        ``ceil(log2((2**bits - 1) * window))`` bits, of which the top ``bits`` are kept.
        """
        kernel = self.pool.kernel_size
        window = kernel * kernel if isinstance(kernel, int) else kernel[0] * kernel[1]
        dropped = ceil(log2((2**self.fmt.bits - 1) * window)) - self.fmt.bits
        return IntFormat(self.fmt.bits, self.fmt.signed, self.fmt.scale * 2.0**dropped / window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool ``x``, which must hold values of the format given at construction."""
        fmt = self.fmt
        quant_x = IntQuantTensor(x, fmt.scale, 0.0, float(fmt.bits), fmt.signed, self.training)
        return self.pool(quant_x)


def _conv(node: Node, shape: Shape) -> nn.Module:
    """Build a quantized convolution; input channels follow from the current shape."""
    channels = shape[0]
    bits = node["weight_bits"]
    return QuantConv2d(
        in_channels=channels,
        out_channels=channels if node["depthwise"] else node["out_channels"],
        kernel_size=node["kernel"],
        stride=node["stride"],
        padding=same_padding(node["kernel"]) if node["padding"] == "same" else 0,
        groups=channels if node["depthwise"] else 1,
        bias=False,
        weight_quant=WeightQuant,
        weight_bit_width=bits,
        weight_quant_type=_quant_type(bits),
    )


def _linear(in_features: int, out_features: int, bits: int) -> nn.Module:
    """Build a quantized fully connected layer without bias."""
    return QuantLinear(
        in_features,
        out_features,
        bias=False,
        weight_quant=WeightQuant,
        weight_bit_width=bits,
        weight_quant_type=_quant_type(bits),
    )


def _terminal(
    node: Node, shape: Shape, fmt: IntFormat | None
) -> tuple[nn.Module, IntFormat | None]:
    """Build the module of a terminal applied to ``shape``; also return its output format."""
    kind = node["kind"]
    if kind == "quant_act":
        return quant_activation(node["bits"])
    if kind == "identity":
        return nn.Identity(), fmt
    if kind == "maxpool":
        return nn.MaxPool2d(node["kernel"], node["stride"]), fmt
    if kind == "avgpool":
        if fmt is None:
            return nn.AvgPool2d(node["kernel"]), None
        pool = TruncAvgPool(node["kernel"], fmt)
        return pool, pool.out_format
    if kind == "conv":
        return _conv(node, shape), None
    if kind == "linear":
        return _linear(shape[0], node["out_features"], node["weight_bits"]), None
    if kind == "relu_float":
        return nn.ReLU(), None
    if kind == "batchnorm":
        return (nn.BatchNorm2d(shape[0]) if is_image(shape) else nn.BatchNorm1d(shape[0])), None
    return nn.Flatten(1), None


class Branching(nn.Module):
    """Feed the input to all branches and merge their outputs."""

    def __init__(self, branches: list[nn.Module], agg: str) -> None:
        """Store the branches and the aggregation (``"add"`` or ``"concat"``)."""
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.agg = agg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all branches on ``x`` and aggregate along the channel/feature dimension."""
        outputs = [branch(x) for branch in self.branches]
        if self.agg == "concat":
            return torch.cat(outputs, dim=1)
        return reduce(torch.add, outputs)


def _module(
    node: Node, shape: Shape, fmt: IntFormat | None
) -> tuple[nn.Module, Shape, IntFormat | None]:
    """Build the module of a body node; return it with its output shape and format."""
    kind = node["kind"]
    if kind == "sequential":
        layers = []
        for child in node["children"]:
            layer, shape, fmt = _module(child, shape, fmt)
            layers.append(layer)
        return nn.Sequential(*layers), shape, fmt
    if kind == "branching":
        built = [_module(child, shape, fmt) for child in node["children"]]
        out_shape = aggregate(node["agg"], [branch_out for _, branch_out, _ in built])
        # An aggregation changes value range and layout, so the format is not tracked further.
        return Branching([module for module, _, _ in built], node["agg"]), out_shape, None
    module, out_fmt = _terminal(node, shape, fmt)
    return module, apply_terminal(node, shape)[0], out_fmt


class GenotypeNet(nn.Module):
    """Input quantizer -> sampled body -> classification head, returning float logits."""

    def __init__(self, genotype: Genotype) -> None:
        """Build all layers; raises ``InvalidGenotypeError`` for an ill-formed genotype."""
        super().__init__()
        info = infer_shapes(genotype)
        head = genotype["head"]
        self.input_quant, fmt = input_quantizer(genotype["input_bits"])
        self.body, body_out, fmt = _module(genotype["body"], info.input_shape, fmt)
        if not is_image(body_out) or head["kind"] == "flatten" or body_out[1:] == (1, 1):
            # Vectors need no reduction, and pooling a 1x1 map is the same as flattening it.
            self.reduce: nn.Module = nn.Flatten(1)
        elif fmt is None:
            self.reduce = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        else:
            self.reduce = nn.Sequential(TruncAvgPool(body_out[1:], fmt), nn.Flatten(1))
        self.classifier = _linear(info.head_features, genotype["num_classes"], head["weight_bits"])
        # The weight quantizers have a constant scale of 1, so the default (fan-in scaled)
        # initialization would round almost everything to zero at low bit widths.
        for module in self.modules():
            if isinstance(module, (QuantConv2d, QuantLinear)):
                nn.init.uniform_(module.weight, -1.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a ``(N, C, H, W)`` float batch to ``(N, num_classes)`` logits."""
        return self.classifier(self.reduce(self.body(self.input_quant(x))))


def to_module(genotype: Genotype) -> GenotypeNet:
    """Build the Brevitas network described by ``genotype`` (randomly initialized)."""
    return GenotypeNet(genotype)

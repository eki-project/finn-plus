"""Hand-written seed architectures that lie inside the search space.

As in einspace, seeds give a search known-good starting points next to random samples. They
are ordinary genotypes built with the constructors of :mod:`finn.nas.space.genotype`.
"""

from __future__ import annotations

from finn.nas.space.genotype import (
    Genotype,
    Node,
    batchnorm,
    branching,
    conv,
    depthwise_conv,
    flatten,
    identity,
    linear,
    make_genotype,
    maxpool,
    quant_act,
    sequential,
)


def _block(layer: Node, act_bits: int) -> Node:
    """Return ``layer`` followed by batch normalization and a quantized activation."""
    return sequential(layer, batchnorm(), quant_act(act_bits))


def cnv_like(weight_bits: int = 2, act_bits: int = 2) -> Genotype:
    """Scaled-down BNN-PYNQ CNV: unpadded 3x3 conv blocks, max pooling, one hidden FC layer."""

    def conv_block(channels: int) -> Node:
        """Return a conv-batchnorm-activation block."""
        return _block(conv(channels, 3, padding="valid", weight_bits=weight_bits), act_bits)

    body = sequential(
        conv_block(16),
        conv_block(16),
        maxpool(2, 2),
        conv_block(32),
        conv_block(32),
        maxpool(2, 2),
        conv_block(64),
        conv_block(64),
        flatten(),
        _block(linear(64, weight_bits), act_bits),
    )
    return make_genotype(body, head_kind="flatten", head_weight_bits=weight_bits)


def mlp_like(weight_bits: int = 2, act_bits: int = 2) -> Genotype:
    """Fully connected network (TFC/SFC style): flatten, then three hidden layers."""
    body = sequential(
        flatten(),
        _block(linear(64, weight_bits), act_bits),
        _block(linear(64, weight_bits), act_bits),
        _block(linear(64, weight_bits), act_bits),
    )
    return make_genotype(body, head_kind="flatten", head_weight_bits=weight_bits)


def residual(weight_bits: int = 4, act_bits: int = 4) -> Genotype:
    """Stem, one residual block (identity skip + two conv blocks, add), pooling, flatten head.

    The fork sits directly behind a quantized activation and the residual branch ends in one,
    so the add sees two integer tensors.
    """
    body = sequential(
        _block(conv(16, 3, weight_bits=weight_bits), act_bits),
        branching(
            identity(),
            sequential(
                _block(conv(16, 3, weight_bits=weight_bits), act_bits),
                _block(conv(16, 3, weight_bits=weight_bits), act_bits),
            ),
            agg="add",
        ),
        maxpool(2, 2),
    )
    return make_genotype(body, head_kind="flatten", head_weight_bits=weight_bits)


def depthwise_separable(weight_bits: int = 4, act_bits: int = 4) -> Genotype:
    """MobileNet-style: stem, strided depthwise + pointwise conv blocks, global pooling head."""

    def separable(channels: int) -> Node:
        """Return a strided depthwise block followed by a pointwise block."""
        return sequential(
            _block(depthwise_conv(3, stride=2, weight_bits=weight_bits), act_bits),
            _block(conv(channels, 1, weight_bits=weight_bits), act_bits),
        )

    body = sequential(
        _block(conv(16, 3, weight_bits=weight_bits), act_bits),
        separable(32),
        separable(64),
    )
    return make_genotype(body, head_kind="gap", head_weight_bits=weight_bits)


#: Seed constructors by name; call without arguments for the default variant.
SEEDS = {
    "cnv_like": cnv_like,
    "mlp_like": mlp_like,
    "residual": residual,
    "depthwise_separable": depthwise_separable,
}


def seed_genotypes() -> dict[str, Genotype]:
    """Return the default variant of every seed architecture."""
    return {name: make() for name, make in SEEDS.items()}

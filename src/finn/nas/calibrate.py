"""Initialize the activation quantizer scales of a candidate from data, without training.

Brevitas' default activation quantizers derive their scale from statistics of the activations
seen during the first forward passes in train mode and learn it afterwards. A network that is
exported straight after construction has never run in train mode: every activation then
exports the initial scale ``1 / int_max`` (identical for all quantizers of the same bit
width), which misrepresents the network, makes every join trivially match and so yields false
verdicts. A few forward passes under ``no_grad`` fix that: the 99.999th percentile of one real
batch is already within a few percent of the value the full statistics phase converges to.
"""

from __future__ import annotations

import torch
from torch import nn


def synthetic_inputs(input_shape: tuple[int, ...], count: int = 128, seed: int = 0) -> torch.Tensor:
    """Return ``count`` uniform random inputs in [-1, 1) of ``input_shape`` (no data needed)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((count, *input_shape), generator=generator) * 2.0 - 1.0


def calibrate(
    model: nn.Module, inputs: torch.Tensor, steps: int = 20, batch_size: int = 128
) -> nn.Module:
    """Run ``steps`` forward passes of ``model`` in train mode without gradients.

    ``inputs`` are network inputs (already normalized); consecutive passes cycle through
    chunks of ``batch_size`` of them. Besides the quantizer statistics this also fills the
    batch-norm running statistics. The model is left in eval mode. Done in place.
    """
    if steps <= 0:
        return model.eval()
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    chunks = [inputs[start : start + batch_size] for start in range(0, inputs.shape[0], batch_size)]
    chunks = [chunk for chunk in chunks if chunk.shape[0] >= 2]  # batch norm needs 2 samples
    if not chunks:
        raise ValueError("calibrate needs at least two inputs.")
    model.train()
    with torch.no_grad():
        for step in range(steps):
            model(chunks[step % len(chunks)])
    return model.eval()

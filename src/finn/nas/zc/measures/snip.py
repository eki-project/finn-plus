# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
# Modified for FINN+ (finn.nas.zc): Brevitas layers keep their quantized forward, the mask is
# applied to the latent weight entering the weight quantizer (``snip_mask_quant_weight``).
# Imports, type hints and docstrings. See ../README.md, "Local modifications".
# Vendored: the measure signature, hook signatures and names are kept as in NASLib.
# ruff: noqa: N806, N812
"""SNIP measure (connection sensitivity: gradient w.r.t. a multiplicative weight mask)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from typing import TYPE_CHECKING

from . import measure
from .brevitas_compat import is_quant_layer
from .p_utils import get_layer_metric_array

if TYPE_CHECKING:
    from collections.abc import Callable


def snip_forward_conv2d(self: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """Float convolution with masked weight."""
    return F.conv2d(
        x,
        self.weight * self.weight_mask,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )


def snip_forward_linear(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Float linear layer with masked weight."""
    return F.linear(x, self.weight * self.weight_mask, self.bias)


def snip_mask_quant_weight(layer: nn.Module) -> None:
    """Mask the latent weight of a Brevitas layer on its way into the weight quantizer.

    Replacing the forward as for float layers would drop the input, weight, bias and output
    quantizers. With the mask in front of the quantizer the sensitivity is that of the
    quantized network w.r.t. removing the latent weight (straight-through gradient).
    """

    def apply_mask(_module: nn.Module, args: tuple) -> tuple:
        """Multiply the weight handed to the quantizer by the mask."""
        return (args[0] * layer.weight_mask, *args[1:])

    layer.weight_quant.register_forward_pre_hook(apply_mask)


@measure("snip", bn=True, mode="param")
def compute_snip_per_weight(
    net: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    loss_fn: Callable,
    split_data: int = 1,
) -> list[torch.Tensor]:
    """Return the absolute loss gradient w.r.t. the weight mask of every conv/linear layer."""
    for layer in net.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.weight.requires_grad = False

        if is_quant_layer(layer):
            snip_mask_quant_weight(layer)
            continue

        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(snip_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        outputs = net.forward(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

    # select the gradients that we want to use for search/prune
    def snip(layer: nn.Module) -> torch.Tensor:
        """SNIP saliency of one layer."""
        if layer.weight_mask.grad is not None:
            return torch.abs(layer.weight_mask.grad)
        return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, snip, mode)

    return grads_abs

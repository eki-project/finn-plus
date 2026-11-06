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
# Modified for FINN+ (finn.nas.zc): imports, type hints and docstrings only.
# See ../README.md, "Local modifications".
# Vendored: the measure signature, hook signatures and names are kept as in NASLib.
# ruff: noqa: N806
"""Plain measure (sum of weight times loss gradient)."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from . import measure
from .p_utils import get_layer_metric_array

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn


@measure("plain", bn=True, mode="param")
def compute_plain_per_weight(
    net: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    loss_fn: Callable,
    split_data: int = 1,
) -> list[torch.Tensor]:
    """Return weight times loss gradient for every conv/linear layer."""
    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        outputs = net.forward(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

    # select the gradients that we want to use for search/prune
    def plain(layer: nn.Module) -> torch.Tensor:
        """Weight times gradient of one layer."""
        if layer.weight.grad is not None:
            return layer.weight.grad * layer.weight
        return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, plain, mode)
    return grads_abs

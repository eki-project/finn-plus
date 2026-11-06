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
# Modified for FINN+ (finn.nas.zc): the data loader helpers (``get_some_data``,
# ``get_some_data_grasp``) are not vendored, type hints and docstrings added.
# See ../README.md, "Local modifications".
"""Helpers shared by the vendored zero-cost measures."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def get_layer_metric_array(net: nn.Module, metric: Callable, mode: str) -> list:
    """Apply ``metric`` to every conv/linear layer (Brevitas quant layers are subclasses)."""
    metric_array = []

    for layer in net.modules():
        if mode == "channel" and hasattr(layer, "dont_ch_prune"):
            continue
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            metric_array.append(metric(layer))

    return metric_array


def reshape_elements(elements: list, shapes: list, device: torch.device) -> list:
    """Broadcast per-channel values to tensors of the per-channel weight shapes."""

    def broadcast_val(elements: list, shapes: list) -> list[torch.Tensor]:
        """Fill one tensor of shape ``sh`` per channel value and stack them per layer."""
        ret_grads = []
        for e, sh in zip(elements, shapes, strict=False):
            ret_grads.append(torch.stack([torch.Tensor(sh).fill_(v) for v in e], dim=0).to(device))
        return ret_grads

    if isinstance(elements[0], list):
        outer = []
        for e, sh in zip(elements, shapes, strict=False):
            outer.append(broadcast_val(e, sh))
        return outer
    return broadcast_val(elements, shapes)


def count_parameters(model: nn.Module) -> Any:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

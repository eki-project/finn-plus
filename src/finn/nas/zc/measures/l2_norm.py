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
# ruff: noqa: ARG001
"""L2 norm measure (sum of the per-layer weight norms)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import measure
from .p_utils import get_layer_metric_array

if TYPE_CHECKING:
    import torch
    from torch import nn


@measure("l2_norm", copy_net=False, mode="param")
def get_l2_norm_array(
    net: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    split_data: int = 1,
    **kwargs: Any,
) -> list[torch.Tensor]:
    """Return the norm of the (latent float) weight of every conv/linear layer."""
    return get_layer_metric_array(net, lambda layer: layer.weight.norm(), mode=mode)

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
# Modified for FINN+ (finn.nas.zc): reduced to the network copy (``no_op``, ``copynet``) and the
# score reduction (``sum_arr``, lifted out of ``find_measures``); the data loader handling, the
# out-of-memory retry and the tensorwatch based flops/params are not vendored. Type hints and
# docstrings added. See ../README.md, "Local modifications".
"""Network copy and score reduction of the NASLib zero-cost predictor."""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import types


def no_op(self: nn.Module, x: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
    """Identity forward that replaces batch norm for ``bn=False`` measures."""
    return x


def copynet(self: nn.Module, bn: bool) -> nn.Module:
    """Return a deep copy of the network, with batch norm bypassed if ``bn`` is False."""
    net = copy.deepcopy(self)
    if not bn:
        for layer in net.modules():
            if isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
                layer.forward = types.MethodType(no_op, layer)
    return net


def sum_arr(arr: list[torch.Tensor]) -> float:
    """Reduce the per-layer arrays of a measure to its score, the sum over all elements."""
    total = 0.0
    for i in range(len(arr)):
        total += torch.sum(arr[i])
    return total.item()

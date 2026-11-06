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
"""
This contains implementations of jacov based on
https://github.com/BayesWatch/nas-without-training (jacov).
This script was based on
https://github.com/SamsungLabs/zero-cost-nas/blob/main/foresight/pruners/measures/jacob_cov.py
We found this version of jacov tends to perform
better.
Author: Robin Ru @ University of Oxford
"""

# Modified for FINN+ (finn.nas.zc): a failure is raised to the caller instead of being printed
# and turned into NaN. Imports, type hints and docstrings.
# See ../README.md, "Local modifications".
# Vendored: the measure signature, hook signatures and names are kept as in NASLib.
# ruff: noqa: ARG001, D212, D400, D404, D415

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

from . import measure

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn


def get_batch_jacobian(
    net: nn.Module, x: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the gradient of the summed network output w.r.t. the input batch."""
    net.zero_grad()

    x.requires_grad_(True)

    y = net(x)

    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()

    return jacob, target.detach()


def eval_score(jacob: np.ndarray, labels: torch.Tensor | None = None) -> float:
    """Score the eigenvalues of the correlation matrix of the per-sample Jacobians."""
    corrs = np.corrcoef(jacob)
    v, _ = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1.0 / (v + k))


@measure("jacov", bn=True)
def compute_jacob_cov(
    net: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    split_data: int = 1,
    loss_fn: Callable | None = None,
) -> float:
    """Return the Jacobian covariance score of the network on the batch."""
    # Compute gradients (but don't apply them)
    jacobs, labels = get_batch_jacobian(net, inputs, targets)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
    return eval_score(jacobs, labels)

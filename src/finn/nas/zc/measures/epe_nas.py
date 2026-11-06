# MIT License

# Copyright (c) 2021 VascoLopes

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Implementation of EPE-NAS: Efficient Performance Estimation Without Training
# for Neural Architecture Search (https://arxiv.org/abs/2102.08099) taken from the
# authors' Github repository https://github.com/VascoLopes/EPE-NAS/blob/main/search.py

# Modified for FINN+ (finn.nas.zc): a failure is raised to the caller instead of being printed
# and turned into NaN. Imports, type hints and docstrings.
# See ../README.md, "Local modifications".
# Vendored: the measure signature, hook signatures and names are kept as in NASLib.
# ruff: noqa: ARG001
"""EPE-NAS measure (class-wise correlation of the per-sample input Jacobians)."""

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING, Any

from . import measure

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn


def get_batch_jacobian(
    net: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    to: Any,
    device: Any,
    args: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return the gradient of the summed network output w.r.t. the input batch."""
    net.zero_grad()

    x.requires_grad_(True)

    y = net(x)

    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()

    return jacob, target.detach(), y.shape[-1]


def eval_score_perclass(
    jacob: np.ndarray, labels: list[np.ndarray] | None = None, n_classes: int = 10
) -> float:
    """Score the correlation of the Jacobians of the samples within each class."""
    k = 1e-5

    per_class = {}
    for i, label in enumerate(labels[0]):
        if label in per_class:
            per_class[label] = np.vstack((per_class[label], jacob[i]))
        else:
            per_class[label] = jacob[i]

    ind_corr_matrix_score = {}
    for c in per_class.keys():
        s = 0
        try:
            corrs = np.array(np.corrcoef(per_class[c]))

            s = np.sum(np.log(abs(corrs) + k))  # /len(corrs)
            if n_classes > 100:
                s /= len(corrs)
        except Exception:  # defensive programming
            continue
        ind_corr_matrix_score[c] = s

    # per class-corr matrix A and B
    score = 0
    ind_corr_matrix_score_keys = ind_corr_matrix_score.keys()
    if n_classes <= 100:
        for c in ind_corr_matrix_score_keys:
            # B)
            score += np.absolute(ind_corr_matrix_score[c])
    else:
        for c in ind_corr_matrix_score_keys:
            # A)
            for cj in ind_corr_matrix_score_keys:
                score += np.absolute(ind_corr_matrix_score[c] - ind_corr_matrix_score[cj])

        if len(ind_corr_matrix_score_keys) > 0:
            # should divide by number of classes seen
            score /= len(ind_corr_matrix_score_keys)

    return score


@measure("epe_nas")
def compute_epe_score(
    net: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Callable,
    split_data: int = 1,
) -> float:
    """Return the EPE-NAS score of the network on the batch."""
    jacobs = []
    labels = []

    jacobs_batch, target, n_classes = get_batch_jacobian(net, inputs, targets, None, None)
    jacobs.append(jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().numpy())

    if len(target.shape) == 2:  # Hack to handle TNB101 classification tasks
        target = torch.argmax(target, dim=1)

    labels.append(target.cpu().numpy())

    jacobs = np.concatenate(jacobs, axis=0)

    return eval_score_perclass(jacobs, labels, n_classes)

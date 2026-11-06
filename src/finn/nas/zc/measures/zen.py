# Copyright (C) 2010-2021 Alibaba Group Holding Limited.
# =============================================================================
# Modified for FINN+ (finn.nas.zc): networks without ``forward_before_global_avg_pool`` are
# evaluated on the input of their last ``nn.Linear`` (``forward_before_classifier``) and the
# difference is summed over all non-batch dimensions. Imports, type hints and docstrings.
# See ../README.md, "Local modifications".
# Vendored: the measure signature, hook signatures and names are kept as in NASLib.
# ruff: noqa: ARG001, A001
"""Zen-score measure (sensitivity of the features to an input perturbation, Gaussian weights)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import TYPE_CHECKING

from . import measure
from .brevitas_compat import as_tensor

if TYPE_CHECKING:
    from collections.abc import Callable


def network_weight_gaussian_init(net: nn.Module) -> nn.Module:
    """Re-initialize conv/linear weights from a standard normal and reset the norm layers."""
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is None:
                    continue
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net


def forward_before_classifier(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return the features the Zen-score is computed on.

    NASLib/ZenNAS networks provide them as ``forward_before_global_avg_pool``. An arbitrary
    module does not, so the input of its last ``nn.Linear`` (the classifier) is used instead,
    i.e. the features after pooling/flattening rather than before.
    """
    if hasattr(net, "forward_before_global_avg_pool"):
        return net.forward_before_global_avg_pool(x)
    linear_layers = [m for m in net.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        raise ValueError("Zen-score needs a network with a linear classifier.")
    features = []
    handle = linear_layers[-1].register_forward_pre_hook(
        lambda _module, args: features.append(as_tensor(args[0]))
    )
    net(x)
    handle.remove()
    return features[0]


@measure("zen", bn=True)
def compute_zen_score(
    net: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Callable | None = None,
    split_data: int = 1,
    repeat: int = 1,
    mixup_gamma: float = 1e-2,
    fp16: bool = False,
) -> float:
    """Return the Zen-score of the network for random inputs of the shape of ``inputs``."""
    nas_score_list = []

    device = inputs.device
    dtype = torch.half if fp16 else torch.float32

    with torch.no_grad():
        for _repeat_count in range(repeat):
            network_weight_gaussian_init(net)
            input = torch.randn(size=list(inputs.shape), device=device, dtype=dtype)
            input2 = torch.randn(size=list(inputs.shape), device=device, dtype=dtype)
            mixup_input = input + mixup_gamma * input2

            output = forward_before_classifier(net, input)
            mixup_output = forward_before_classifier(net, mixup_input)

            nas_score = torch.sum(
                torch.abs(output - mixup_output), dim=list(range(1, output.dim()))
            )
            nas_score = torch.mean(nas_score)

            # compute BN scaling
            log_bn_scaling_factor = 0.0
            for m in net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)
            nas_score = torch.log(nas_score) + log_bn_scaling_factor
            nas_score_list.append(float(nas_score))

    avg_nas_score = float(np.mean(nas_score_list))

    return avg_nas_score

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
This contains implementations of nwot based on the updated version of
https://github.com/BayesWatch/nas-without-training
to reflect the second version of the paper https://arxiv.org/abs/2006.04647
"""

# Modified for FINN+ (finn.nas.zc): the hooked modules are the activation-like modules of
# ``brevitas_compat.activation_modules`` (each activation once, Brevitas quantized activations
# included) instead of every module whose type name contains "ReLU"; an activation fed
# directly by the network input is skipped; ``QuantTensor`` inputs are unwrapped. Imports,
# type hints and docstrings. See ../README.md, "Local modifications".
# Vendored: the measure signature, hook signatures and names are kept as in NASLib.
# ruff: noqa: ARG001, N806, D212, D400, D404, D415

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

from . import measure
from .brevitas_compat import activation_modules, as_tensor

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn


@measure("nwot", bn=True)
def compute_nwot(
    net: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    split_data: int = 1,
    loss_fn: Callable | None = None,
) -> float:
    """Return the log-determinant of the kernel of binary activation codes of the batch."""
    batch_size = len(targets)

    def counting_forward_hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        """Add the agreement of the sign patterns at this activation's input to the kernel."""
        if inp[0] is x:
            # An input quantizer sees the same data for every architecture.
            return
        inp = as_tensor(inp[0])
        inp = inp.view(inp.size(0), -1)
        x_bin = (inp > 0).float()  # binary indicator
        K = x_bin @ x_bin.t()
        K2 = (1.0 - x_bin) @ (1.0 - x_bin.t())
        net.K = net.K + K.cpu().numpy() + K2.cpu().numpy()  # hamming distance

    net.K = np.zeros((batch_size, batch_size))
    for module in activation_modules(net):
        module.register_forward_hook(counting_forward_hook)

    x = torch.clone(inputs)
    net(x)
    _, jc = np.linalg.slogdet(net.K)

    return jc

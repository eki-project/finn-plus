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
# Modified for FINN+ (finn.nas.zc): the quantizers of a Brevitas network are disabled, i.e. the
# measure is evaluated on the float skeleton of the network. Imports, type hints and
# docstrings. See ../README.md, "Local modifications".
# Vendored: the measure signature, hook signatures and names are kept as in NASLib.
# ruff: noqa: ARG001
"""SynFlow measure (synaptic flow: path norm of the absolute weights, data-free)."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from . import measure
from .brevitas_compat import disable_quantization
from .p_utils import get_layer_metric_array

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn


@measure("synflow", bn=False, mode="param")
@measure("synflow_bn", bn=True, mode="param")
def compute_synflow_per_weight(
    net: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    split_data: int = 1,
    loss_fn: Callable | None = None,
) -> list[torch.Tensor]:
    """Return ``|weight * gradient|`` of the summed output for an all-ones input, with all
    weights replaced by their absolute value."""
    device = inputs.device

    # SynFlow scores the parameterization and assumes a positively homogeneous network.
    # Quantizers are not (rounding, clipping), a bipolar activation even blocks the
    # straight-through gradient of all layers in front of it. Hence the float skeleton of a
    # Brevitas network is scored; the absolute value taken of the then unused quantizer
    # scales below has no effect.
    disable_quantization(net)

    # convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net: nn.Module) -> dict[str, torch.Tensor]:
        """Replace all weights by their absolute value and return their signs."""
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    # convert to orig values
    @torch.no_grad()
    def nonlinearize(net: nn.Module, signs: dict[str, torch.Tensor]) -> None:
        """Restore the signs of the weights."""
        for name, param in net.state_dict().items():
            if "weight_mask" not in name:
                param.mul_(signs[name])

    # keep signs of all params
    signs = linearize(net.double())

    # Compute gradients with input of 1s
    net.zero_grad()
    net.double()
    input_dim = list(inputs[0, :].shape)
    # inputs = torch.ones([1] + input_dim).to(device)
    inputs = torch.ones([1, *input_dim]).double().to(device)
    output = net.forward(inputs)
    torch.sum(output).backward()

    # select the gradients that we want to use for search/prune
    def synflow(layer: nn.Module) -> torch.Tensor:
        """SynFlow saliency of one layer."""
        if layer.weight.grad is not None:
            return torch.abs(layer.weight * layer.weight.grad)
        return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, synflow, mode)

    # apply signs of all params
    nonlinearize(net, signs)

    return grads_abs

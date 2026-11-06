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
# Modified for FINN+ (finn.nas.zc): Brevitas layers keep their quantized forward
# (``fisher_forward_quant``), the warning about the non-full backward hook is silenced.
# Imports, type hints and docstrings. See ../README.md, "Local modifications".
# Vendored: the measure signature, hook signatures and names are kept as in NASLib.
# ruff: noqa: ARG001, N806, N812
"""Fisher measure (per-channel activation times gradient, squared and averaged)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import warnings
from typing import TYPE_CHECKING

from . import measure
from .brevitas_compat import as_tensor, is_quant_layer
from .p_utils import get_layer_metric_array, reshape_elements

if TYPE_CHECKING:
    from brevitas.quant_tensor import QuantTensor
    from collections.abc import Callable


def fisher_forward_conv2d(self: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """Float convolution that records its output activation."""
    x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    # intercept and store the activations after passing through 'hooked' identity op
    self.act = self.dummy(x)
    return self.act


def fisher_forward_linear(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Float linear layer that records its output activation."""
    x = F.linear(x, self.weight, self.bias)
    self.act = self.dummy(x)
    return self.act


def fisher_forward_quant(
    self: nn.Module, x: torch.Tensor | QuantTensor
) -> torch.Tensor | QuantTensor:
    """Unchanged Brevitas forward (with all quantizers) that records its output activation."""
    out = type(self).forward(self, x)
    self.act = self.dummy(as_tensor(out))
    return self.act if out is as_tensor(out) else out.set(value=self.act)


@measure("fisher", bn=True, mode="channel")
def compute_fisher_per_weight(
    net: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Callable,
    mode: str,
    split_data: int = 1,
) -> list[torch.Tensor]:
    """Return the Fisher information of every conv/linear output channel, broadcast to the
    shape of the layer weight."""
    device = inputs.device

    if mode == "param":
        raise ValueError("Fisher pruning does not support parameter pruning.")

    net.train()
    for layer in net.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            # variables/op needed for fisher computation
            layer.fisher = None
            layer.act = 0.0
            layer.dummy = nn.Identity()

            # replace forward method of conv/linear
            if is_quant_layer(layer):
                layer.forward = types.MethodType(fisher_forward_quant, layer)
            elif isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(fisher_forward_conv2d, layer)
            elif isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(fisher_forward_linear, layer)

            # function to call during backward pass (hooked on identity op at output of layer)
            def hook_factory(layer: nn.Module) -> Callable:
                """Create the backward hook that accumulates the Fisher information of ``layer``."""

                def hook(module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
                    """Accumulate the squared activation-gradient product per channel."""
                    act = layer.act.detach()
                    grad = grad_output[0].detach()
                    if len(act.shape) > 2:
                        g_nk = torch.sum((act * grad), list(range(2, len(act.shape))))
                    else:
                        g_nk = act * grad
                    del_k = g_nk.pow(2).mean(0).mul(0.5)
                    if layer.fisher is None:
                        layer.fisher = del_k
                    else:
                        layer.fisher += del_k
                    # without deleting this, a nasty memory leak occurs! related:
                    # https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555
                    del layer.act

                return hook

            # register backward hook on identity fcn to compute fisher info
            layer.dummy.register_backward_hook(hook_factory(layer))

    N = inputs.shape[0]
    with warnings.catch_warnings():
        # The hook only reads grad_output, which the non-full hook reports correctly.
        warnings.filterwarnings("ignore", message="Using a non-full backward hook")
        for sp in range(split_data):
            st = sp * N // split_data
            en = (sp + 1) * N // split_data

            net.zero_grad()
            outputs = net(inputs[st:en])
            loss = loss_fn(outputs, targets[st:en])
            loss.backward()

    # retrieve fisher info
    def fisher(layer: nn.Module) -> torch.Tensor:
        """Fisher information per output channel of one layer."""
        if layer.fisher is not None:
            return torch.abs(layer.fisher.detach())
        return torch.zeros(layer.weight.shape[0])  # size=ch

    grads_abs_ch = get_layer_metric_array(net, fisher, mode)

    # broadcast channel value here to all parameters in that channel
    # to be compatible with stuff downstream (which expects per-parameter metrics)
    # TODO cleanup on the selectors/apply_prune_mask side (?)
    shapes = get_layer_metric_array(net, lambda layer: layer.weight.shape[1:], mode)

    grads_abs = reshape_elements(grads_abs_ch, shapes, device)

    return grads_abs

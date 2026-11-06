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
# Modified for FINN+ (finn.nas.zc): the network copy is made by ``copynet`` directly instead of
# a ``get_prunable_copy`` method attached to the caller's network, the copy is put into
# training mode, the forced garbage collection after a measure is limited to CUDA, type hints
# and docstrings added. See ../README.md, "Local modifications".
"""Registry of the zero-cost proxy measures vendored from NASLib (branch ``zerocost``)."""

from __future__ import annotations

import gc
import importlib
import torch
from typing import TYPE_CHECKING, Any

from .predictive import copynet

if TYPE_CHECKING:
    from collections.abc import Callable
    from torch import nn

available_measures = []
_measure_impls = {}


def measure(
    name: str, bn: bool = True, copy_net: bool = True, force_clean: bool = True, **impl_args: Any
) -> Callable:
    """Register the decorated function as measure ``name``.

    ``bn=False`` bypasses the batch norm layers of the copy, ``copy_net=False`` hands the
    caller's network to the measure (only for measures that do not modify it).
    """

    def make_impl(func: Callable) -> Callable:
        """Wrap ``func`` so that it runs on a private copy of the network."""

        def measure_impl(
            net_orig: nn.Module, device: torch.device, *args: Any, **kwargs: Any
        ) -> Any:
            """Copy the network to ``device`` and evaluate the measure on the copy."""
            # Training mode on the copy: batch norm and the Brevitas activation quantizers of an
            # untrained network take their statistics/scales from the minibatch.
            net = copynet(net_orig, bn).to(device).train() if copy_net else net_orig
            ret = func(net, *args, **kwargs, **impl_args)
            # A full collection takes ~0.2 s in a FINN process. It is needed between measures
            # where the copies (kept alive by hook reference cycles) hold GPU memory; on the
            # CPU the caller collects once after all measures.
            if copy_net and force_clean and torch.device(device).type == "cuda":
                del net
                torch.cuda.empty_cache()
                gc.collect()
            return ret

        if name in _measure_impls:
            raise KeyError(f"Duplicated measure! {name}")
        available_measures.append(name)
        _measure_impls[name] = measure_impl
        return func

    return make_impl


def calc_measure(name: str, net: nn.Module, device: torch.device, *args: Any, **kwargs: Any) -> Any:
    """Evaluate the registered measure ``name`` on ``net``."""
    return _measure_impls[name](net, device, *args, **kwargs)


def load_all() -> None:
    """Import all measure modules so that they register themselves."""
    for module in (
        "grad_norm",
        "snip",
        "grasp",
        "fisher",
        "jacov",
        "plain",
        "synflow",
        "epe_nas",
        "zen",
        "l2_norm",
        "nwot",
    ):
        importlib.import_module(f".{module}", __name__)


load_all()

"""Helpers that let the vendored zero-cost measures handle Brevitas networks.

Not part of NASLib. The measures were written for float networks; these helpers answer the
Brevitas-specific questions they need (which layers quantize their weight, which modules are
activations, how to switch the quantizers off) in one place.
"""

from __future__ import annotations

from brevitas.nn.quant_layer import QuantNonLinearActLayer, QuantWeightBiasInputOutputLayer
from brevitas.proxy.quant_proxy import QuantProxyFromInjector
from brevitas.quant_tensor import QuantTensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import nn


def is_quant_layer(layer: nn.Module) -> bool:
    """Tell whether ``layer`` is a Brevitas conv/linear layer, whose forward quantizes input,
    weight, bias and output and therefore must not be replaced by a plain ``F.conv2d``."""
    return isinstance(layer, QuantWeightBiasInputOutputLayer)


def as_tensor(value: torch.Tensor | QuantTensor) -> torch.Tensor:
    """Return the plain tensor of a value that may be a Brevitas ``QuantTensor``
    (layers built with ``return_quant_tensor=True`` pass those on)."""
    return value.value if isinstance(value, QuantTensor) else value


def activation_modules(net: nn.Module) -> list[nn.Module]:
    """Return the activation-like modules of ``net``, each activation exactly once.

    These are the float ReLU variants NASLib selects (class name contains ``ReLU``) and all
    Brevitas quantized activation layers (``QuantReLU``, ``QuantIdentity``, ``QuantHardTanh``,
    ...). The latter include ``QuantIdentity`` because in low-precision FINN networks the
    quantizer itself is the nonlinearity (a bipolar ``QuantIdentity`` is a sign activation).
    The float ``nn.ReLU`` that a ``QuantReLU`` wraps internally is skipped, it would count the
    same activation a second time.
    """
    found = []
    quant_act_prefixes: list[str] = []
    for name, module in net.named_modules():
        if name.startswith(tuple(quant_act_prefixes)):
            continue
        if isinstance(module, QuantNonLinearActLayer):
            found.append(module)
            quant_act_prefixes.append(name + ".")
        elif "ReLU" in type(module).__name__:
            found.append(module)
    return found


def disable_quantization(net: nn.Module) -> None:
    """Turn ``net`` into its float skeleton, permanently (meant for a throwaway copy).

    Uses the switches Brevitas' own calibration mode uses: every quantizer proxy is disabled
    (activations keep their float nonlinearity, e.g. the ReLU of a ``QuantReLU``) and no layer
    returns a ``QuantTensor``, which cannot be produced without quantizers.
    """
    for module in net.modules():
        if isinstance(module, QuantProxyFromInjector):
            module.disable_quant = True
        if hasattr(module, "return_quant_tensor"):
            module.return_quant_tensor = False

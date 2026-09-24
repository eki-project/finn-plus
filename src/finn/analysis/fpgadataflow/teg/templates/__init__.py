# Copyright (C) 2026, Paderborn University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Operator control-path templates.

A template turns one FINN hardware node into a set of event chains, internal edges and arcs
(``OpModel``). Templates are registered by ``(op_type, backend)`` where ``op_type`` is the
ONNX op type without the backend suffix (e.g. ``Thresholding``) and ``backend`` is ``hls`` or
``rtl``. Every arc and every constant in a template must carry a reference to the RTL/HLS
source it transcribes; constants that are not visible in the source are *measured* and marked
as such.

The environment (source / sink chains) is built by :func:`environment.attach_environment`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from finn.util.exception import FINNUserError

if TYPE_CHECKING:
    from finn.analysis.fpgadataflow.teg.model import Chain, FIFOEdge
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


@dataclass
class OpModel:
    """Chains, internal edges and stream endpoints of one operator instance.

    ``inputs[i]`` / ``outputs[i]`` name the chain that reads input stream ``i`` / writes output
    stream ``i``; the external edges attached to these streams are created by ``from_onnx``.
    """

    chains: list[Chain]
    internal_edges: list[FIFOEdge]
    inputs: list[str]
    outputs: list[str]
    #: latency constants that are measured rather than read from the source
    notes: dict[str, object] = field(default_factory=dict)


#: A template builder: ``(node instance, chain name prefix, input edge names, output edge
#: names) -> OpModel``. ``in_edges[i]`` is the name of the external edge feeding input ``i``,
#: ``out_edges[i]`` the one fed by output ``i``.
if TYPE_CHECKING:
    TemplateBuilder = Callable[[HWCustomOp, str, list[str], list[str]], OpModel]
else:
    TemplateBuilder = Callable[..., OpModel]

_REGISTRY: dict[tuple[str, str], TemplateBuilder] = {}


def register(op_type: str, backend: str) -> Callable[[TemplateBuilder], TemplateBuilder]:
    """Register the decorated builder for ``(op_type, backend)``."""

    def deco(fn: TemplateBuilder) -> TemplateBuilder:
        _REGISTRY[(op_type, backend)] = fn
        return fn

    return deco


def split_op_type(op_type: str) -> tuple[str, str]:
    """Split ``MVAU_hls`` into ``("MVAU", "hls")``."""
    for suffix in ("_hls", "_rtl"):
        if op_type.endswith(suffix):
            return op_type[: -len(suffix)], suffix[1:]
    raise FINNUserError(
        f"Node type {op_type} has no hls/rtl backend suffix; run SpecializeLayers first"
    )


def get_builder(op_type: str) -> TemplateBuilder:
    """Return the builder for a specialised op type such as ``Thresholding_rtl``."""
    _load_all()
    key = split_op_type(op_type)
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise FINNUserError(
            f"No TEG operator template for {op_type} (backend {key[1]}); supported: "
            + ", ".join(f"{o}_{b}" for o, b in sorted(_REGISTRY))
        ) from exc


def supported_op_types() -> list[str]:
    """Specialised op types with a template."""
    _load_all()
    return sorted(f"{o}_{b}" for o, b in _REGISTRY)


def _load_all() -> None:
    """Import all template modules so that their registrations run."""
    import importlib

    for mod in (
        "hls_loop",
        "thresholding_rtl",
        "swg_rtl",
        "passthrough",
        "dwc_rtl",
        "fmpadding_rtl",
        "mvau_rtl",
        "attention_hls",
    ):
        importlib.import_module(f"{__name__}.{mod}")

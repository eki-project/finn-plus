"""Structural hashing of exported candidate models for result caching."""

from __future__ import annotations

import hashlib
import json
import math
import numpy as np
import onnx
from importlib.metadata import PackageNotFoundError, version
from onnx import numpy_helper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from finn.nas.config import OracleConfig

#: Bump when the oracle's semantics change in a way that invalidates cached results.
ORACLE_VERSION = "1"

#: Initializers with at most this many elements are candidates for hashing by value: quantizer
#: constants (bit widths, zero points), reshape targets, axes and the like define the
#: architecture rather than trained weights. Non-integral float values among them (e.g. a
#: per-tensor weight scale, which depends on the random initialization) are excluded.
SMALL_CONST_NUMEL = 16


def finn_version() -> str:
    """Return the installed finn-plus version string, or ``"unknown"``."""
    try:
        return version("finn-plus")
    except PackageNotFoundError:
        return "unknown"


def hash_candidate(
    onnx_path: Path, config: OracleConfig, include_weights: bool | None = None
) -> str:
    """Compute a stable hash of the architecture in ``onnx_path`` plus the oracle settings.

    Tensor and node *names* are excluded (they depend on module naming), whereas op types,
    attributes, tensor shapes/datatypes, small constants and the graph I/O signature are
    included. Large initializers (weights, per-channel scales) are included only if
    ``include_weights`` (default: ``config.cache_include_weights``) is set.
    """
    if include_weights is None:
        include_weights = config.cache_include_weights
    model = onnx.load(str(onnx_path))
    h = hashlib.sha256()

    def upd(text: str) -> None:
        """Feed a NUL-terminated string into the hash."""
        h.update(text.encode())
        h.update(b"\0")

    upd("finn=" + finn_version())
    upd("oracle=" + ORACLE_VERSION)
    upd(json.dumps(config.hash_relevant(), sort_keys=True, default=str))

    initializers = {init.name: init for init in model.graph.initializer}
    for vi in list(model.graph.input) + list(model.graph.output):
        if vi.name in initializers:
            continue
        tt = vi.type.tensor_type
        upd(f"io:{tt.elem_type}:{[d.dim_value for d in tt.shape.dim]}")
    for node in model.graph.node:
        upd(f"n:{node.domain}:{node.op_type}:{len(node.input)}:{len(node.output)}")
        for attr in sorted(node.attribute, key=lambda a: a.name):
            upd("a:" + attr.name)
            h.update(attr.SerializeToString())
        for inp in node.input:
            init = initializers.get(inp)
            if init is None:
                upd("t" if inp else "-")
                continue
            numel = math.prod(init.dims) if init.dims else 1
            upd(f"init:{list(init.dims)}:{init.data_type}")
            if include_weights:
                h.update(numpy_helper.to_array(init).tobytes())
            elif numel <= SMALL_CONST_NUMEL:
                values = numpy_helper.to_array(init)
                if not np.issubdtype(values.dtype, np.floating) or np.all(
                    values == np.round(values)
                ):
                    h.update(values.tobytes())
    return h.hexdigest()

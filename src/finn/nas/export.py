"""Export of candidate networks to QONNX. The only module in ``finn.nas`` that touches torch."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def export_candidate(model: Any, input_shape: Sequence[int] | None, destination: Path) -> Path:
    """Export ``model`` to a QONNX file at ``destination`` and return that path.

    ``model`` is either a path to an existing ONNX/QONNX file (copied) or a Brevitas/PyTorch
    module, in which case ``input_shape`` (including the batch dimension) is required.
    Training mode and parameter device of the module are left untouched.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, (str, Path)):
        src = Path(model)
        if src.resolve() != destination.resolve():
            shutil.copyfile(src, destination)
        return destination

    import torch
    from brevitas.export import export_qonnx

    if input_shape is None:
        raise ValueError("input_shape is required when exporting a torch module.")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            export_qonnx(model, torch.randn(*input_shape, device=device), str(destination))
    finally:
        model.train(was_training)
    return destination

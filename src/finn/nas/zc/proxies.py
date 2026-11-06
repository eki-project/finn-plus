"""Zero-cost (training-free) proxy scores of a candidate network on one minibatch."""

from __future__ import annotations

import gc
import math
import time
import torch
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from torch import nn
from torch.nn.functional import cross_entropy
from typing import TYPE_CHECKING, Any

import finn.nas.zc.measures as zc_measures
from finn.nas.zc.measures.brevitas_compat import as_tensor
from finn.nas.zc.measures.p_utils import count_parameters
from finn.nas.zc.measures.predictive import sum_arr

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

#: Model statistics that are reported next to the vendored proxies.
STATISTICS: tuple[str, ...] = ("params", "flops")

#: What ``compute_proxies`` evaluates by default: the model statistics and the proxies that
#: were found to give finite, architecture-dependent values on Brevitas networks. ``zen`` and
#: ``synflow_bn`` are available but excluded, see the README of this package.
DEFAULT_MEASURES: tuple[str, ...] = (
    "epe_nas",
    "fisher",
    "grad_norm",
    "grasp",
    "jacov",
    "l2_norm",
    "nwot",
    "plain",
    "snip",
    "synflow",
    *STATISTICS,
)


@dataclass
class ProxyResult:
    """Scores of one network. A measure that failed has a NaN score and an entry in ``errors``."""

    scores: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the result as plain, JSON-serializable dictionaries."""
        return asdict(self)


def available_measures() -> list[str]:
    """Return the names of everything ``compute_proxies`` can evaluate."""
    return sorted([*zc_measures.available_measures, *STATISTICS])


def _count_flops(model: nn.Module, inputs: torch.Tensor) -> float:
    """Count the multiply-accumulate operations of all conv/linear layers for one sample.

    Bias additions, normalization, pooling and activations are not counted. Runs on a copy in
    evaluation mode, because a single sample cannot be batch-normalized in training mode.
    """
    net = deepcopy(model).to(inputs.device).eval()
    total = 0

    def count(module: nn.Module, _inp: tuple, out: Any) -> None:
        """Add the multiplications of one layer call."""
        nonlocal total
        if isinstance(module, nn.Linear):
            per_output = module.in_features
        else:
            per_output = module.in_channels // module.groups * math.prod(module.kernel_size)
        total += as_tensor(out).numel() * per_output

    for module in net.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            module.register_forward_hook(count)
    with torch.no_grad():
        net(inputs[:1])
    return float(total)


def _evaluate(
    name: str,
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Callable,
    device: torch.device,
) -> float:
    """Evaluate one measure and reduce it to a float."""
    if name == "params":
        return float(count_parameters(model))
    # Every measure gets its own copy of the batch: the Jacobian based ones turn on
    # ``requires_grad`` on their input.
    inputs = inputs.detach().to(device, copy=True)
    targets = targets.detach().to(device)
    if name == "flops":
        return _count_flops(model, inputs)
    value = zc_measures.calc_measure(
        name, model, device, inputs, targets, loss_fn=loss_fn, split_data=1
    )
    # NASLib: per-layer arrays are summed, scalar measures are taken as they are.
    return sum_arr(value) if isinstance(value, (list, tuple)) else float(value)


def _device_of(model: nn.Module) -> torch.device:
    """Return the device of the first parameter (CPU for a parameter-free model)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@contextmanager
def _preserved_rng(device: torch.device) -> Iterator[None]:
    """Restore torch's global random state (CPU and ``device``) and the cuDNN flags when the
    block is left, and make cuDNN deterministic inside it, so that the seeding in the block is
    invisible outside."""
    cudnn = torch.backends.cudnn
    flags = (cudnn.deterministic, cudnn.benchmark)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    cudnn.deterministic, cudnn.benchmark = True, False
    try:
        yield
    finally:
        cudnn.deterministic, cudnn.benchmark = flags
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)


def _seed(seed: int, device: torch.device) -> None:
    """Seed the CPU generator and, if used, the generator of ``device``.

    Not ``torch.manual_seed``: that also queues a seeding of every CUDA device for the moment
    CUDA is initialized, which would leak out of ``compute_proxies`` on a CPU-only call.
    """
    torch.default_generator.manual_seed(seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def compute_proxies(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    measures: Sequence[str] | None = None,
    *,
    num_classes: int,
    loss_fn: Callable | None = None,
    device: torch.device | str | None = None,
    seed: int = 0,
) -> ProxyResult:
    """Score ``model`` with zero-cost proxies on the minibatch ``inputs``/``targets``.

    ``measures`` defaults to ``DEFAULT_MEASURES``. ``num_classes`` is used to reject class
    index targets outside ``[0, num_classes)`` early, where a measure would otherwise fail
    with a less helpful message. ``loss_fn`` defaults to cross entropy, ``device`` to the
    device of the model.

    The model is never modified: every measure works on a deep copy (in training mode, so
    that batch norm and the Brevitas activation quantizers use the statistics of the batch)
    and on its own copy of the batch. Each measure runs isolated; if it raises or returns a
    non-finite value its score is NaN and the reason is recorded in ``errors``. The torch
    random generators are seeded with ``seed`` before each measure and restored afterwards,
    so the scores are reproducible and the caller's random stream is not disturbed.
    """
    names = list(DEFAULT_MEASURES if measures is None else measures)
    unknown = sorted(set(names) - set(available_measures()))
    if unknown:
        raise ValueError(f"Unknown measures {unknown}, available: {available_measures()}")
    if not targets.is_floating_point() and targets.numel() > 0:
        low, high = int(targets.min()), int(targets.max())
        if low < 0 or high >= num_classes:
            raise ValueError(f"Targets span [{low}, {high}], expected [0, {num_classes}).")
    loss_fn = cross_entropy if loss_fn is None else loss_fn
    device = _device_of(model) if device is None else torch.device(device)

    result = ProxyResult()
    with _preserved_rng(device), torch.enable_grad():
        for name in names:
            _seed(seed, device)
            start = time.perf_counter()
            try:
                score = _evaluate(name, model, inputs, targets, loss_fn, device)
                if not math.isfinite(score):
                    raise ValueError(f"non-finite score {score}")
            except Exception as exc:
                score = math.nan
                result.errors[name] = f"{type(exc).__name__}: {exc}"[:300]
            result.scores[name] = score
            result.seconds[name] = time.perf_counter() - start
    # The copies are kept alive by reference cycles (hooks that close over the network).
    gc.collect()
    return result

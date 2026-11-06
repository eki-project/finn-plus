"""Evaluate genotypes of the search space with the oracle (and proxies) in worker processes.

One evaluation builds the Brevitas module, gives its batch norms trained-looking parameters,
exports it, runs the oracle pipeline in the worker itself and optionally scores the network
with the zero-cost proxies. Workers are spawned once and reused, which amortizes the several
seconds it takes to import the compiler.
"""

from __future__ import annotations

import os
import signal
import time
import torch
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context
from pathlib import Path
from torch import nn
from typing import TYPE_CHECKING, Any, Self

from finn.nas.config import OracleConfig
from finn.nas.result import EvalResult, FailureCategory

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from finn.nas.space.genotype import Genotype


def randomize_batchnorm(model: nn.Module, seed: int = 0) -> nn.Module:
    """Give every batch norm in ``model`` per-channel parameters as a trained network has.

    A freshly constructed batch norm is an identity, which FINN's streamlining moves and
    absorbs like a scalar; the per-channel scale and shift of a trained one cannot always be
    absorbed. Evaluating untrained candidates would therefore accept topologies that fail
    after training. Done in place; deterministic for ``seed``.
    """
    generator = torch.Generator().manual_seed(seed)
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            n = module.num_features
            with torch.no_grad():
                module.running_mean.copy_(torch.randn(n, generator=generator) * 0.5)
                module.running_var.copy_(torch.rand(n, generator=generator) + 0.5)
                module.weight.copy_(torch.rand(n, generator=generator) + 0.5)
                module.bias.copy_(torch.randn(n, generator=generator) * 0.5)
    return model


@dataclass
class CandidateEvaluation:
    """Everything recorded about one evaluated genotype."""

    genotype: dict[str, Any]
    hash: str
    result: EvalResult
    #: Zero-cost proxy scores (``finn.nas.zc.ProxyResult.to_dict()``), if computed.
    proxies: dict[str, Any] | None = None
    params: int = 0
    nodes: int = 0
    seconds: float = 0.0
    #: Free-form bookkeeping of the caller (sample index, generation, ...).
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        d = asdict(self)
        d["result"] = self.result.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateEvaluation:
        """Reconstruct from :meth:`to_dict` output."""
        data = dict(data)
        data["result"] = EvalResult.from_dict(data["result"])
        return cls(**data)


@dataclass
class EvaluatorConfig:
    """Settings of :class:`GenotypeEvaluator` that are shipped to the workers."""

    oracle: OracleConfig
    #: Zero-cost proxy measures to compute (``None``: the defaults, ``[]``: none).
    proxy_measures: list[str] | None = None
    compute_proxies: bool = True
    randomize_batchnorm: bool = True
    #: Hard limit per candidate, enforced inside the worker.
    timeout_s: float = 300.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "oracle": self.oracle.to_dict(),
            "proxy_measures": self.proxy_measures,
            "compute_proxies": self.compute_proxies,
            "randomize_batchnorm": self.randomize_batchnorm,
            "timeout_s": self.timeout_s,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluatorConfig:
        """Reconstruct from :meth:`to_dict` output."""
        data = dict(data)
        data["oracle"] = OracleConfig.from_dict(data["oracle"])
        return cls(**data)


class _TimeoutError(Exception):
    """Raised by the alarm handler when a candidate exceeds its time limit."""


def _alarm(signum: int, frame: Any) -> None:  # noqa: ARG001
    """Signal handler turning ``SIGALRM`` into :class:`_TimeoutError`."""
    raise _TimeoutError


def evaluate_genotype(
    genotype: Genotype,
    config: EvaluatorConfig,
    work_dir: Path,
    seed: int = 0,
    proxy_batch: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> CandidateEvaluation:
    """Evaluate one genotype in the current process (what a worker runs per task)."""
    from finn.nas.export import export_candidate
    from finn.nas.hashing import hash_candidate
    from finn.nas.pipeline import run_pipeline
    from finn.nas.space import count_params, genotype_hash, infer_shapes
    from finn.nas.space.build import to_module

    t_start = time.perf_counter()
    ghash = genotype_hash(genotype)
    info = infer_shapes(genotype)
    evaluation = CandidateEvaluation(
        genotype=genotype,
        hash=ghash,
        result=EvalResult(),
        params=count_params(genotype),
        nodes=len(info.nodes),
    )
    cand_dir = work_dir / ghash[:16]
    cand_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    use_alarm = hasattr(signal, "SIGALRM") and config.timeout_s > 0
    if use_alarm:
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, config.timeout_s)
    try:
        model = to_module(genotype)
        if config.randomize_batchnorm:
            randomize_batchnorm(model, seed)
        input_shape = (1, *genotype["input_shape"])
        try:
            onnx_path = export_candidate(model, input_shape, cand_dir / "model.onnx")
        except (KeyboardInterrupt, SystemExit, _TimeoutError):
            raise
        except Exception as exc:
            evaluation.result = EvalResult.failed(
                "export", FailureCategory.EXPORT_FAILED, f"{type(exc).__name__}: {exc}"[:500]
            )
        else:
            evaluation.result = run_pipeline(onnx_path, config.oracle, cand_dir)
            evaluation.result.candidate_hash = hash_candidate(onnx_path, config.oracle)
        if config.compute_proxies and proxy_batch is not None:
            from finn.nas.zc import compute_proxies

            inputs, targets = proxy_batch
            proxies = compute_proxies(
                model,
                inputs,
                targets,
                config.proxy_measures,
                num_classes=genotype["num_classes"],
                device="cpu",
                seed=seed,
            )
            evaluation.proxies = proxies.to_dict()
    except _TimeoutError:
        evaluation.result = EvalResult.failed(
            "worker",
            FailureCategory.TIMEOUT,
            f"Evaluation exceeded {config.timeout_s:.0f} s.",
        )
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if config.oracle.keep_artifacts == "never":
            import shutil

            shutil.rmtree(cand_dir, ignore_errors=True)
    evaluation.seconds = time.perf_counter() - t_start
    return evaluation


_WORKER_STATE: dict[str, Any] = {}


def _worker_init(build_dir: str, proxy_batch: tuple[torch.Tensor, torch.Tensor] | None) -> None:
    """Prepare a worker process: own FINN build directory, proxy batch kept in memory."""
    os.environ["FINN_BUILD_DIR"] = build_dir
    torch.set_num_threads(1)
    _WORKER_STATE["proxy_batch"] = proxy_batch


def _worker_task(
    genotype: Genotype, config_dict: dict[str, Any], work_dir: str, seed: int
) -> dict[str, Any]:
    """Task function run in a worker; returns a JSON-serializable evaluation."""
    config = EvaluatorConfig.from_dict(config_dict)
    evaluation = evaluate_genotype(
        genotype, config, Path(work_dir), seed, _WORKER_STATE.get("proxy_batch")
    )
    return evaluation.to_dict()


class GenotypeEvaluator:
    """Evaluate many genotypes with a pool of worker processes.

    Use as a context manager; ``n_workers=0`` evaluates in the calling process (useful for
    tests and debugging, no timeout then unless running on a platform with ``SIGALRM``).
    """

    def __init__(
        self,
        config: EvaluatorConfig,
        work_dir: str | Path,
        n_workers: int = 1,
        proxy_batch: tuple[torch.Tensor, torch.Tensor] | None = None,
        max_tasks_per_worker: int = 50,
    ) -> None:
        """Create the evaluator; the pool starts on first use."""
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.n_workers = n_workers
        self.proxy_batch = None
        if proxy_batch is not None:
            self.proxy_batch = (proxy_batch[0].detach().cpu(), proxy_batch[1].detach().cpu())
        self.max_tasks_per_worker = max_tasks_per_worker
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> Self:
        """Start the worker pool."""
        if self.n_workers > 0:
            self._pool = ProcessPoolExecutor(
                max_workers=self.n_workers,
                mp_context=get_context("spawn"),
                initializer=_worker_init,
                initargs=(str(self.work_dir / "finn_tmp"), self.proxy_batch),
                max_tasks_per_child=self.max_tasks_per_worker,
            )
        return self

    def __exit__(self, *exc: object) -> None:
        """Shut the worker pool down."""
        self.close()

    def close(self) -> None:
        """Shut the worker pool down (idempotent)."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def evaluate(
        self, genotypes: Iterable[Genotype], seeds: Sequence[int] | None = None
    ) -> list[CandidateEvaluation]:
        """Evaluate ``genotypes`` (in order) and return one evaluation per genotype."""
        genotypes = list(genotypes)
        if seeds is None:
            seeds = list(range(len(genotypes)))
        if self._pool is None:
            return [
                evaluate_genotype(g, self.config, self.work_dir, s, self.proxy_batch)
                for g, s in zip(genotypes, seeds, strict=True)
            ]
        config_dict = self.config.to_dict()
        futures = [
            self._pool.submit(_worker_task, g, config_dict, str(self.work_dir), s)
            for g, s in zip(genotypes, seeds, strict=True)
        ]
        return [CandidateEvaluation.from_dict(f.result()) for f in futures]

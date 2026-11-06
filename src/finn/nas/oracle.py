"""The FINN+ oracle: fast feasibility check and cost estimates for candidate networks."""

from __future__ import annotations

import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from finn.nas.cache import ResultCache, make_cache
from finn.nas.classify import first_line
from finn.nas.export import export_candidate
from finn.nas.hashing import hash_candidate
from finn.nas.pipeline import run_pipeline
from finn.nas.result import TRANSIENT_FAILURES, EvalResult, FailureCategory
from finn.nas.worker import run_in_subprocess

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from finn.nas.config import OracleConfig


class FinnOracle:
    """Evaluate candidate networks with the analytical part of the FINN+ flow.

    Example::

        oracle = FinnOracle(OracleConfig(fpga_part="xc7z020clg400-1"))
        res = oracle.evaluate(brevitas_module, input_shape=(1, 3, 32, 32))
        if res.feasible:
            print(res.resources_total["LUT"], res.est_fps)
        else:
            print(res.failure_category, res.failure_op_type)
    """

    def __init__(self, config: OracleConfig, cache: ResultCache | None = None) -> None:
        """Create an oracle. ``cache`` overrides ``config.cache_path``."""
        self.config = config
        if config.work_dir is not None:
            self.work_dir = Path(config.work_dir)
        else:
            self.work_dir = Path(tempfile.gettempdir()) / "finn_nas_oracle"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if cache is None and config.cache_path is not None:
            cache = make_cache(config.cache_path)
        self.cache = cache

    def evaluate(
        self, model: Any, input_shape: Sequence[int] | None = None, tag: str | None = None
    ) -> EvalResult:
        """Evaluate one candidate.

        Parameters
        ----------
        model
            A Brevitas/PyTorch module or the path of an exported QONNX file.
        input_shape
            Input shape including batch dimension; required for modules.
        tag
            Optional prefix for the candidate's working directory (for debugging).
        """
        t_start = time.perf_counter()
        cand_dir = Path(tempfile.mkdtemp(prefix=f"{tag or 'cand'}_", dir=self.work_dir))
        result: EvalResult | None = None
        try:
            try:
                onnx_path = export_candidate(model, input_shape, cand_dir / "model.onnx")
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                result = EvalResult.failed("export", FailureCategory.EXPORT_FAILED, first_line(exc))
                return result
            timings = {"export": time.perf_counter() - t_start}

            key = hash_candidate(onnx_path, self.config)
            if self.cache is not None:
                hit = self.cache.get(key)
                if hit is not None:
                    result = replace(hit, from_cache=True)
                    return result

            if self.config.isolation == "process":
                result = run_in_subprocess(onnx_path, self.config, cand_dir, self.config.timeout_s)
            else:
                result = run_pipeline(onnx_path, self.config, cand_dir)
            result.candidate_hash = key
            result.timings_s = {**timings, **result.timings_s}
            if self.cache is not None and result.failure_category not in TRANSIENT_FAILURES:
                self.cache.put(key, result)
            return result
        finally:
            if result is not None:
                result.timings_s["total"] = time.perf_counter() - t_start
            self._cleanup(cand_dir, result)

    def evaluate_many(
        self,
        candidates: Iterable[tuple[Any, Sequence[int] | None]],
        n_workers: int = 1,
    ) -> list[EvalResult]:
        """Evaluate several ``(model, input_shape)`` pairs, ``n_workers`` at a time.

        Parallelism is only used with ``isolation="process"``; the threads merely wait for
        their worker processes.
        """
        items = list(candidates)
        if n_workers <= 1 or self.config.isolation != "process":
            return [self.evaluate(m, s) for m, s in items]
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            return list(pool.map(lambda ms: self.evaluate(ms[0], ms[1]), items))

    def _cleanup(self, cand_dir: Path, result: EvalResult | None) -> None:
        """Remove the candidate's working directory unless it is to be kept."""
        keep = self.config.keep_artifacts
        if keep == "always":
            return
        if keep == "on_failure" and (result is None or not result.feasible):
            return
        shutil.rmtree(cand_dir, ignore_errors=True)

"""Run the evaluation pipeline in an isolated, killable subprocess."""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

from finn.nas.result import EvalResult, FailureCategory

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from finn.nas.config import OracleConfig


def _child_main(conn: Connection, onnx_path: str, oc_dict: dict[str, Any], cand_dir: str) -> None:
    """Entry point of the worker process."""
    import os

    os.environ["FINN_BUILD_DIR"] = str(Path(cand_dir) / "tmp")
    try:
        from finn.nas.config import OracleConfig
        from finn.nas.pipeline import run_pipeline

        result = run_pipeline(
            Path(onnx_path),
            OracleConfig.from_dict(oc_dict),
            Path(cand_dir),
            on_stage=lambda stage: conn.send(("stage", stage)),
        )
        conn.send(("result", result.to_dict()))
    except BaseException as exc:  # report everything to the parent
        conn.send(("error", repr(exc), traceback.format_exc()))
    finally:
        conn.close()


def run_in_subprocess(
    onnx_path: Path, oc: OracleConfig, cand_dir: Path, timeout_s: float
) -> EvalResult:
    """Evaluate in a freshly spawned interpreter, killing it after ``timeout_s`` seconds.

    ``spawn`` is used because the calling NAS process typically has torch/onnxruntime loaded,
    which makes ``fork`` unsafe (see ``finn.util.multiprocessing``).
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_child_main,
        args=(child_conn, str(onnx_path), oc.to_dict(), str(cand_dir)),
        name="finn-nas-worker",
    )
    proc.start()
    child_conn.close()

    deadline = time.monotonic() + timeout_s
    last_stage: str | None = None
    result: EvalResult | None = None
    error: tuple[str, str] | None = None
    timed_out = False
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                if parent_conn.poll(min(remaining, 0.25)):
                    msg = parent_conn.recv()
                    if msg[0] == "stage":
                        last_stage = msg[1]
                    elif msg[0] == "result":
                        result = EvalResult.from_dict(msg[1])
                        break
                    elif msg[0] == "error":
                        error = (msg[1], msg[2])
                        break
            except EOFError:
                break
            if not proc.is_alive() and not parent_conn.poll():
                break
    finally:
        if timed_out or proc.is_alive():
            proc.kill()
        proc.join(timeout=10)
        parent_conn.close()

    if result is not None:
        return result
    if timed_out:
        return EvalResult.failed(
            last_stage or "worker",
            FailureCategory.TIMEOUT,
            f"Evaluation exceeded {timeout_s:.0f} s (last stage: {last_stage}).",
        )
    if error is not None:
        res = EvalResult.failed(last_stage or "worker", FailureCategory.INTERNAL_ERROR, error[0])
        res.warnings = [error[1]]
        return res
    return EvalResult.failed(
        last_stage or "worker",
        FailureCategory.INTERNAL_ERROR,
        f"Worker process died without a result (exit code {proc.exitcode}).",
    )

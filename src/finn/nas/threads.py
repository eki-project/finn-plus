"""Single-threaded numerics for the oracle worker processes.

One candidate evaluation is small; the parallelism of the NAS experiments comes from running
many worker processes. Every worker therefore gets exactly one thread for torch and for
onnxruntime, otherwise each of them spawns a thread pool sized for the whole node and the
workers starve each other. Limiting onnxruntime also avoids its thread-affinity errors on
Slurm-managed nodes with non-contiguous CPU sets (microsoft/onnxruntime#8313), the same
workaround ``finn bench`` and the test suite apply.
"""

from __future__ import annotations

import os
import torch
from typing import Any

#: Environment variables that cap the BLAS/OpenMP thread pools of a process started after
#: they are set (they are read at library load time).
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def single_threaded_environment() -> None:
    """Make processes spawned from here on (and their numeric libraries) single-threaded.

    Call before starting worker processes; the calling process keeps its already loaded
    thread pools.
    """
    for name in THREAD_ENV_VARS:
        os.environ.setdefault(name, "1")


def limit_to_single_thread() -> None:
    """Limit torch and onnxruntime of the current process to one thread each."""
    torch.set_num_threads(1)
    try:
        import onnxruntime as ort

        # Accesses onnxruntime's undocumented internal capi module (as finn bench does)
        pybind_state = ort.capi._pybind_state  # type: ignore[attr-defined]  # noqa: SLF001
        options = pybind_state.get_default_session_options()
    except (ImportError, AttributeError):
        return

    def single_threaded_session_options() -> Any:
        """Return the default session options with one intra- and inter-op thread."""
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        return options

    pybind_state.get_default_session_options = single_threaded_session_options

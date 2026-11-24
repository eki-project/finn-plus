"""Tests for the single-threaded worker setup (``finn.nas.threads``)."""

import pytest

import os
import torch

threads = pytest.importorskip("finn.nas.threads")


@pytest.mark.util
class TestThreads:
    """The workers get one thread for torch and onnxruntime."""

    def test_limit_to_single_thread(self) -> None:
        ort = pytest.importorskip("onnxruntime")
        before = torch.get_num_threads()
        try:
            threads.limit_to_single_thread()
            assert torch.get_num_threads() == 1
            options = ort.capi._pybind_state.get_default_session_options()  # noqa: SLF001
            assert options.intra_op_num_threads == 1
            assert options.inter_op_num_threads == 1
            # a session created from the defaults inherits the limit
            assert ort.SessionOptions().intra_op_num_threads in (0, 1)
        finally:
            torch.set_num_threads(before)

    def test_environment_for_spawned_processes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in threads.THREAD_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        threads.single_threaded_environment()
        assert os.environ["OMP_NUM_THREADS"] == "4"  # an explicit setting is respected
        assert os.environ["MKL_NUM_THREADS"] == "1"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"

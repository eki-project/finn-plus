# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import tempfile

import tests.testing_util.test as test_helpers
from tests.testing_util.test import make_runtime_weight_stream

pytestmark = pytest.mark.util


@pytest.fixture
def scratch_root(tmp_path, monkeypatch):
    """Redirect the helper's build directory into tmp_path (FINN_BUILD_DIR is only read
    once when the settings are created, so patch make_build_dir instead of the env)."""

    def _make_build_dir(prefix="", return_as_path=False):
        return tempfile.mkdtemp(prefix=prefix, dir=tmp_path)

    monkeypatch.setattr(test_helpers, "make_build_dir", _make_build_dir)
    return tmp_path


class RuntimeWeightOp:
    def __init__(self, content):
        self.content = content

    def make_weight_file(self, weights, mode, path):
        assert weights == "weights"
        assert mode == "decoupled_runtime"
        with open(path, "w") as f:
            f.write(self.content)


def test_make_runtime_weight_stream_removes_clean_scratch(scratch_root):
    tmp_path = scratch_root

    stream = make_runtime_weight_stream(RuntimeWeightOp("1\na\nff\n"), "weights")

    assert stream == [1, 10, 255]
    assert list(tmp_path.iterdir()) == []


def test_make_runtime_weight_stream_retains_failed_scratch(scratch_root):
    tmp_path = scratch_root

    with pytest.raises(ValueError):
        make_runtime_weight_stream(RuntimeWeightOp("not-hex\n"), "weights")

    retained = list(tmp_path.iterdir())
    assert len(retained) == 1
    assert (retained[0] / "weights.dat").read_text() == "not-hex\n"

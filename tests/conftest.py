# Copyright (c) 2020, Xilinx
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

# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for finn.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""

import pytest

import onnxruntime as ort
import os
import shutil
import tempfile
from pathlib import Path

import finn.util.settings
from finn.interface.settings import FINNSettings
from finn.util.settings import get_settings


@pytest.fixture(scope="session", autouse=True)
def load_settings(request) -> None:
    """Load settings passed from run_finn.py and make them available globally."""
    # Load settings. run_finn.py in finn test set these file to use in FINN_SETTINGS
    # which has the highest priority when loading settings
    settings = FINNSettings.init(
        flow_config=Path("/tmp/FINN_TEST_BUILD_DIR/dummy.yaml"), auto_set_environmenmt_vars=True
    )
    finn.util.settings._SETTINGS = settings  # noqa


@pytest.fixture(scope="class", autouse=True)
def isolate_build_dir(request):
    # Retrieve settings
    isolate = os.environ.get("FINN_TESTS_ISOLATE_BUILD_DIRS", "1") == "1"
    cleanup = os.environ.get("FINN_TESTS_CLEANUP_BUILD_DIRS", "0") == "1"

    # Create the top test dir if it doesnt exist yet
    top_build_dir = get_settings().finn_build_dir
    if not top_build_dir.exists():
        top_build_dir.mkdir(parents=True)

    # Setup individual FINN_BUILD_DIR for each test class
    if isolate:
        try:
            # use original test name (without [..parameters..] appended) in case of function scope
            name = request.node.originalname
        except AttributeError:
            # fall back to class name in case of class scope
            name = request.node.name
        test_build_dir = Path(tempfile.mkdtemp(prefix=name + "_", dir=top_build_dir))
        get_settings().finn_build_dir = test_build_dir

    # Execute test(s)
    yield

    # Clean up and reset FINN_BUILD_DIR
    if isolate:
        if cleanup:
            shutil.rmtree(test_build_dir)
        get_settings().finn_build_dir = top_build_dir


@pytest.fixture(scope="session", autouse=True)
def setup_onnxruntime(request):
    # Attempt to work around onnxruntime issue on Slurm-managed clusters:
    # See https://github.com/microsoft/onnxruntime/issues/8313
    # This seems to happen only when assigned CPU cores are not contiguous
    _default_session_options = ort.capi._pybind_state.get_default_session_options()

    def get_default_session_options_new():
        _default_session_options.inter_op_num_threads = 1
        _default_session_options.intra_op_num_threads = 1
        return _default_session_options

    ort.capi._pybind_state.get_default_session_options = get_default_session_options_new

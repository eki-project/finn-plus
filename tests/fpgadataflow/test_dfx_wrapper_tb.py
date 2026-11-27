# Copyright (C) 2026, Paderborn University
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the self-checking testbenches of the multi-DNN stream wrappers in xsim.

finn-rtllib/dfx/tb/dfx_wrapper_tb.sv covers the partial-reconfiguration wrapper
(pipelined body reset after a reconfiguration, draining, one frame per OLEN
cycles), sw_wrapper_tb.sv the selectable-weights wrapper and
dfx_tuser_passthrough_tb.sv the passthrough around static segments. All three
check that every output frame carries its own tUSER while several frames are
inside the wrapped body.
"""

import pytest

import shutil
import subprocess
from pathlib import Path

from finn.util.settings import get_settings

_CASES = [
    # partial reconfiguration: mux interval, stalls, single-beat frames, reset latency
    ("dfx_wrapper", "N=100"),
    ("dfx_wrapper", "N=1"),
    ("dfx_wrapper", "N=3 LAT=3"),
    ("dfx_wrapper", "N=2 RANDREADY=1 RANDVALID=1"),
    ("dfx_wrapper", "N=1 OLEN=1 LAT=5"),
    ("dfx_wrapper", "N=100 OLEN=1 LAT=5"),
    ("dfx_wrapper", "N=2 RESET_LATENCY=0"),
    ("dfx_wrapper", "N=2 RESET_LATENCY=6"),
    ("dfx_wrapper", "N=1 TUSER_TRACK_DEPTH=1"),
    # selectable weights
    ("sw_wrapper", "N=100"),
    ("sw_wrapper", "N=1"),
    ("sw_wrapper", "N=2 NUM_SETS=3"),
    ("sw_wrapper", "N=1 RANDREADY=1 RANDVALID=1 SLOWTAP=1"),
    ("sw_wrapper", "N=1 OLEN=1 LAT=5"),
    ("sw_wrapper", "N=1 LAT=100"),
    ("sw_wrapper", "N=1 LAT=100 TUSER_TRACK_DEPTH=2 FULL_RATE=0"),
    # passthrough around a static segment holding several frames
    ("dfx_tuser_passthrough", "N=1"),
    ("dfx_tuser_passthrough", "N=3"),
    ("dfx_tuser_passthrough", "N=1 RANDREADY=1 RANDVALID=1"),
    ("dfx_tuser_passthrough", "N=100 OLEN=1 LAT=50"),
    ("dfx_tuser_passthrough", "N=1 OLEN=1 LAT=20 TUSER_TRACK_DEPTH=4 FULL_RATE=0"),
]


@pytest.mark.vivado
@pytest.mark.parametrize("wrapper,generics", _CASES)
def test_dfx_wrapper_tb(wrapper: str, generics: str, tmp_path: Path) -> None:
    """Compile and run one wrapper testbench and require its PASS line."""
    if shutil.which("xvlog") is None:
        pytest.skip("xsim (Vivado) not found")
    dfx = Path(get_settings().finn_rtllib) / "dfx"
    sources = [
        dfx / "tuser_track" / "tuser_track.sv",
        dfx / wrapper / f"{wrapper}.sv",
        dfx / "tb" / f"{wrapper}_tb.sv",
    ]
    top = f"{wrapper}_tb"
    generic_args = [arg for g in generics.split() for arg in ("-generic_top", g)]

    def run(cmd: list[str]) -> str:
        result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, f"{' '.join(cmd)} failed:\n{result.stdout[-3000:]}"
        return result.stdout

    run(["xvlog", "-sv", *[str(s) for s in sources]])
    run(["xelab", top, "-s", "sim", *generic_args])
    result = subprocess.run(["xsim", "sim", "-R"], cwd=tmp_path, capture_output=True, text=True)
    log = result.stdout
    assert "FAIL" not in log and "PASS" in log, log[-3000:]

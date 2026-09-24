# Copyright (C) 2026, Paderborn University
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

"""MILP size and solve-time scaling with the tokens per frame (paper table, no Vivado).

Prototype experiment 3: the same chain topology with a growing number of tokens per frame; the
solve time grows super-linearly with the constraint count (~n^1.7 with HiGHS), which is the
argument why the exact formulation cannot be applied to FINN-scale instances (10^8-10^9 rows).
"""

import pytest

import time

from finn.analysis.fpgadataflow.teg.milp import fit_power_law, solve_milp
from finn.analysis.fpgadataflow.teg.simulate import simulate
from tests.fpgadataflow.teg.test_teg_core import chain_graph, unit_candidates

pytestmark = [pytest.mark.fpgadataflow, pytest.mark.fifo_model, pytest.mark.slow]

CANDIDATES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]


def test_milp_scaling_table() -> None:
    """Solve the chain instance for growing token counts and fit ``t = a * n^b``."""
    rows = []
    points = []
    for n, burst in [(32, 8), (64, 8), (128, 8), (256, 16)]:
        m = chain_graph(n=n, burst=burst)
        target = m.bottleneck_interval()
        t0 = time.time()
        res = solve_milp(m, target, unit_candidates(m, CANDIDATES), time_limit=300)
        dt = time.time() - t0
        assert res.depths is not None, res.message
        assert simulate(m, res.depths).interval == target
        rows.append(
            (n, res.size.variables, res.size.constraints, res.size.binaries, dt, res.depths)
        )
        points.append((res.size.constraints, dt))
    a, b = fit_power_law(points)
    print("\n  N   vars   cons  binaries  time[s]  depths")
    for n, nv, nc, nb, dt, d in rows:
        print(f"  {n:4d} {nv:6d} {nc:7d} {nb:8d} {dt:8.2f}  {d}")
    print(f"  fit: t = {a:.3g} * cons^{b:.2f}; extrapolated 1e8 rows: {a * 1e8**b:.3g} s")
    # the constraint count grows linearly with the tokens per frame, the time faster than that
    assert rows[-1][2] > 6 * rows[0][2]
    assert b > 1.0

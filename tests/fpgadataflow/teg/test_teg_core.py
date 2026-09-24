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

"""Core tests of the TEG model, simulator, MILP and search on synthetic graphs (no Vivado).

These port the experiments of the formal-model prototype (``model_prototype/test_*.py``) as
assertions: simulator == brute force == MILP on a chain, the fork/join deadlock structure,
the heterogeneous-width cost gap between local and global optimum, rigid vs. flushable
pipeline operators, and the worked SWG -> MVAU example.
"""

import pytest

import itertools
import math
import time

from finn.analysis.fpgadataflow.teg.milp import estimate_size, fit_power_law, solve_milp
from finn.analysis.fpgadataflow.teg.model import Chain, TEGModel
from finn.analysis.fpgadataflow.teg.patterns import StallPattern
from finn.analysis.fpgadataflow.teg.search import (
    measure_bottleneck,
    measure_peak_occupancy,
    minimize_depths,
)
from finn.analysis.fpgadataflow.teg.simulate import simulate
from finn.analysis.fpgadataflow.teg.templates.hls_loop import flp_loop, mvau_iterations
from finn.analysis.fpgadataflow.teg.templates.swg_rtl import swg_default
from finn.transformation.fpgadataflow.fifo_depth_search import MinimizationOrder

pytestmark = [pytest.mark.fpgadataflow, pytest.mark.fifo_model]


# --------------------------------------------------------------------------- helpers
def interval(model: TEGModel, depths: dict[str, int | None], **kw: object) -> float:
    """Steady-state interval (inf on deadlock)."""
    return simulate(model, depths, **kw).interval  # type: ignore[arg-type]


def greedy_local(
    model: TEGModel, target: float, upper: dict[str, int], lower: dict[str, int], order: list[str]
) -> tuple[dict[str, int], int]:
    """LiveFIFO-style per-FIFO binary search over integer depths (prototype ``greedy_local``)."""
    depths = dict(upper)
    evals = 0
    for e in order:
        lo, hi = lower[e], depths[e]
        while lo < hi:
            mid = (lo + hi) // 2
            trial = dict(depths)
            trial[e] = mid
            evals += 1
            if interval(model, trial) <= target:
                hi = mid
            else:
                lo = mid + 1
        depths[e] = lo
    return depths, evals


def brute_force(
    model: TEGModel, target: float, ranges: dict[str, range], cost: dict[str, int] | None = None
) -> tuple[dict[str, int], int]:
    """Global optimum by exhaustive enumeration with cost pruning."""
    best: dict[str, int] | None = None
    bestc = math.inf
    edges = list(ranges)
    for combo in itertools.product(*[ranges[e] for e in edges]):
        d = dict(zip(edges, combo, strict=True))
        c = sum(v * (cost[e] if cost else 1) for e, v in d.items())
        if c >= bestc:
            continue
        if interval(model, d) <= target:
            best, bestc = d, c
    assert best is not None
    return best, int(bestc)


def unit_candidates(model: TEGModel, depths: list[int]) -> dict[str, list[tuple[int, int]]]:
    """Candidate list with nominal depth == capacity for every external edge."""
    return {e: [(d, d) for d in depths] for e in model.external_edges}


def chain_graph(n: int = 32, burst: int = 8) -> TEGModel:
    """Chain src -> A (bursty: read/idle/write ``burst``) -> B (1 token / 2 cycles) -> sink."""
    m = TEGModel()
    src = Chain("src", kind="source")
    src.events(n, 1, writes=["e0"])
    a = Chain("A")
    for _ in range(n // burst):
        a.events(burst, 1, reads=["e0"])
        a.event(burst, writes=["e1"])
        a.events(burst - 1, 1, writes=["e1"])
    b = Chain("B")
    b.events(n, 2, reads=["e1"], writes=["e2"])
    sink = Chain("sink", kind="sink")
    sink.events(n, 1, reads=["e2"])
    for c in (src, a, b, sink):
        m.add_chain(c)
    m.new_edge("e0", "src", "A", external=True)
    m.new_edge("e1", "A", "B", external=True)
    m.new_edge("e2", "B", "sink", external=True)
    m.validate()
    return m


def forkjoin_graph(n: int = 16, skew: int = 6) -> TEGModel:
    """Fork/join: fork -> {A (1:1), B (reads ``skew`` tokens before writing)} -> join."""
    m = TEGModel()
    src = Chain("src", kind="source")
    src.events(n, 1, writes=["e0"])
    fork = Chain("fork")
    fork.events(n, 1, reads=["e0"], writes=["ea", "eb"])
    a = Chain("A")
    a.events(n, 1, reads=["ea"], writes=["ea2"])
    b = Chain("B")
    b.events(skew, 1, reads=["eb"])
    b.events(n - skew, 1, reads=["eb"], writes=["eb2"])
    b.events(skew, 1, writes=["eb2"])
    join = Chain("join")
    join.events(n, 1, reads=["ea2", "eb2"], writes=["e3"])
    sink = Chain("sink", kind="sink")
    sink.events(n, 1, reads=["e3"])
    for c in (src, fork, a, b, join, sink):
        m.add_chain(c)
    for name, s, d in [
        ("e0", "src", "fork"),
        ("ea", "fork", "A"),
        ("eb", "fork", "B"),
        ("ea2", "A", "join"),
        ("eb2", "B", "join"),
        ("e3", "join", "sink"),
    ]:
        m.new_edge(name, s, d, external=True)
    m.validate()
    return m


# --------------------------------------------------------------------------- experiments
def test_chain_simulator_matches_prototype() -> None:
    """Prototype experiment 1: nominal intervals, unbounded occupancy and the optimum."""
    m = chain_graph()
    assert m.nominal_intervals() == {"A": 92, "B": 64}
    assert m.bottleneck_interval() == 92
    inf = {"e0": None, "e1": None, "e2": None}
    res = simulate(m, inf, max_frames=8, stop_when_stable=False)
    assert res.interval == 92
    # end-of-cycle occupancies (the prototype counted before same-cycle reads: e1 = 5)
    assert res.max_occupancy == {"e0": 482, "e1": 4, "e2": 1}
    # the deterministic execution is periodic: stable state detected with bounded FIFOs
    res = simulate(m, {"e0": 2, "e1": 5, "e2": 1})
    assert res.stable and res.interval == 92 and res.latency is not None
    assert interval(m, {"e0": 2, "e1": 4, "e2": 1}) == 96
    assert interval(m, {"e0": 1, "e1": 5, "e2": 1}) == 120


def test_chain_milp_equals_brute_force_and_greedy() -> None:
    """Prototype experiment 1: exact MILP == brute force == greedy on the chain."""
    m = chain_graph()
    target = 92
    bf, cost = brute_force(m, target, {e: range(1, 17) for e in m.external_edges})
    assert bf == {"e0": 2, "e1": 5, "e2": 1} and cost == 8
    res = solve_milp(m, target, unit_candidates(m, list(range(1, 17))))
    assert res.depths == bf, res.message
    assert res.size.constraints == 1795 - 3 + 3  # same rows as the prototype (chain, data, space)
    upper = dict.fromkeys(m.external_edges, 32)
    lower = dict.fromkeys(m.external_edges, 1)
    for order in (["e0", "e1", "e2"], ["e2", "e1", "e0"]):
        gd, _ = greedy_local(m, target, upper, lower, order)
        assert gd == bf


def test_forkjoin_deadlock_structure() -> None:
    """Prototype experiment 2 and ``test_sum``: feasibility depends on the branch sum."""
    m = forkjoin_graph()
    target = 22
    assert m.bottleneck_interval() == target
    expected = {1: math.inf, 2: math.inf, 5: math.inf, 6: 61, 7: 43, 8: 43}
    for da, exp in expected.items():
        d: dict[str, int | None] = dict.fromkeys(m.external_edges, 1)
        d["ea"] = da
        assert interval(m, d) == exp, da
    # feasible iff d_ea + d_ea2 >= 10 (with all other depths 2 and both >= 2)
    for a in range(2, 11):
        for b in range(2, 11):
            d = dict.fromkeys(m.external_edges, 2)
            d["ea"], d["ea2"] = a, b
            feasible = interval(m, d) <= target
            assert feasible == (a + b >= 10), (a, b)
    res = solve_milp(m, target, unit_candidates(m, list(range(1, 12))))
    assert res.depths is not None and sum(res.depths.values()) == 18
    upper = dict.fromkeys(m.external_edges, 12)
    lower = dict.fromkeys(m.external_edges, 1)
    gd, _ = greedy_local(m, target, upper, lower, m.external_edges)
    assert sum(gd.values()) == 18


def test_width_local_vs_global_gap() -> None:
    """Prototype ``test_width``: 2.06x cost gap between topological greedy and the optimum."""
    m = forkjoin_graph()
    m.edges["ea"].width = 8
    m.edges["ea2"].width = 32
    target = 22
    cands = unit_candidates(m, list(range(1, 12)))
    res = solve_milp(m, target, cands)
    assert res.depths is not None
    bits = sum(d * m.edges[e].width for e, d in res.depths.items())
    assert bits == 136
    assert res.depths["ea"] == 8 and res.depths["ea2"] == 2
    upper = dict.fromkeys(m.external_edges, 12)
    lower = dict.fromkeys(m.external_edges, 1)
    topo, _ = greedy_local(m, target, upper, lower, m.external_edges)
    assert sum(d * m.edges[e].width for e, d in topo.items()) == 280
    rev, _ = greedy_local(m, target, upper, lower, list(reversed(m.external_edges)))
    assert sum(d * m.edges[e].width for e, d in rev.items()) == 136


def _flp_system(mode: str, n: int = 20, sf: int = 1, nf: int = 1, depth: int = 3) -> TEGModel:
    """Prototype ``test_flp``: src -> MVAU-like op -> bursty B -> sink; rigid or flp op."""
    m = TEGModel()
    src = Chain("src", kind="source")
    src.events(n, 1, writes=["e0"])
    m.add_chain(src)
    its = mvau_iterations(n // sf, sf, nf, "e0", "e1")
    if mode == "rigid":
        a = Chain("A")
        for r, w in its:
            a.event(1, reads=r, writes=w)
        m.add_chain(a)
        a_in, a_out = "A", "A"
    else:
        op = flp_loop("A", its, latency=depth - 1, capacity=depth, back_latency=1, regslices=False)
        for c in op.chains:
            m.add_chain(c)
        for e in op.internal_edges:
            m.add_edge(e)
        a_in, a_out = op.inputs[0], op.outputs[0]
    nout = n // sf * nf
    b = Chain("B")
    for _ in range(nout // 5):
        b.events(4, 1, reads=["e1"], writes=["e2"])
        b.event(40, reads=["e1"], writes=["e2"])
    sink = Chain("sink", kind="sink")
    sink.events(nout, 1, reads=["e2"])
    m.add_chain(b)
    m.add_chain(sink)
    m.new_edge("e0", "src", a_in, external=True)
    m.new_edge("e1", a_out, "B", external=True)
    m.new_edge("e2", "B", "sink", external=True)
    m.validate()
    return m


def test_flp_vs_rigid_operator() -> None:
    """Prototype ``test_flp``: the flushable pipeline needs less buffering; rigid sizes are safe."""
    rigid = _flp_system("rigid")
    flp = _flp_system("flp")
    target = 176
    assert rigid.bottleneck_interval() == target
    cands = unit_candidates(rigid, list(range(1, 33)))
    r_res = solve_milp(rigid, target, cands)
    assert r_res.depths is not None and sum(r_res.depths.values()) == 6  # e.g. (2, 2, 2)
    assert interval(rigid, r_res.depths) == target
    f_res = solve_milp(flp, target, unit_candidates(flp, list(range(1, 33))))
    assert f_res.depths is not None and sum(f_res.depths.values()) == 5  # e.g. (1, 2, 2)
    assert interval(flp, f_res.depths) == target
    # over-approximating (rigid) model => its sizes remain safe on the flp system
    assert interval(flp, r_res.depths) == target


def _swg_example() -> tuple[TEGModel, dict[str, object]]:
    """Prototype ``test_swg``: src -> RTL SWG (8x8, k=3) -> HLS MVAU (SF=9, NF=2, L=8) -> sink."""
    m = TEGModel()
    swg = swg_default("swg", "e0", "e1", 8, 8, 3, 1, 1)
    n_win = swg.notes["out_dim"][0] * swg.notes["out_dim"][1]  # type: ignore[index]
    sf, nf = 9, 2
    mvau = flp_loop(
        "mvau",
        mvau_iterations(n_win, sf, nf, "e1", "e2"),
        latency=8,
        capacity=9,
        back_latency=1,
        regslices=False,
    )
    src = Chain("src", kind="source")
    src.events(swg.notes["n_in"], 1, writes=["e0"])  # type: ignore[arg-type]
    sink = Chain("sink", kind="sink")
    sink.events(n_win * nf, 1, reads=["e2"])
    m.add_chain(src)
    for c in swg.chains + mvau.chains:
        m.add_chain(c)
    m.add_chain(sink)
    m.new_edge("e0", "src", swg.inputs[0], external=True)
    for e in swg.internal_edges + mvau.internal_edges:
        m.add_edge(e)
    m.new_edge("e1", swg.outputs[0], mvau.inputs[0], external=True)
    m.new_edge("e2", mvau.outputs[0], "sink", external=True)
    m.validate()
    return m, swg.notes


def test_swg_example_interval_and_occupancy() -> None:
    """The SWG -> MVAU model reproduces the MVAU-bound interval and the prototype's numbers."""
    m, notes = _swg_example()
    assert notes["n_in"] == 64 and notes["n_out"] == 324 and notes["BUF"] == 20
    res = simulate(m, {"e0": None, "e1": None, "e2": None}, max_frames=8, stop_when_stable=False)
    assert res.interval == 648
    assert res.max_occupancy["e2"] == 1
    assert res.max_occupancy["e1"] == 2413 and res.max_occupancy["e0"] == 4205
    # the bounded history windows of the SWG chains do not change the result
    for c in ("swg.R", "swg.F"):
        assert m.chains[c].history_window is not None
    assert interval(m, {"e0": 2, "e1": 5, "e2": 1}) == 648
    assert interval(m, {"e0": 1, "e1": 5, "e2": 1}) > 648


@pytest.mark.slow
def test_swg_example_milp_vs_search() -> None:
    """Prototype ``test_swg``: MILP optimum (2, 5, 1); greedy in topological order finds
    (1, 14, 1), the two other orders the optimum (chain coupling, notes Section 6.4)."""
    m, _ = _swg_example()
    target = 648
    cands = unit_candidates(m, [*range(1, 17), 24, 32, 48, 64])
    res = solve_milp(m, target, cands, time_limit=900)
    assert res.depths == {"e0": 2, "e1": 5, "e2": 1}, res.message
    assert interval(m, res.depths) == target
    base = simulate(m, {"e0": None, "e1": None, "e2": None}, max_frames=8, stop_when_stable=False)
    upper = {e: base.max_occupancy[e] for e in m.external_edges}
    lower = dict.fromkeys(m.external_edges, 1)
    expected = {
        ("e0", "e1", "e2"): {"e0": 1, "e1": 14, "e2": 1},
        ("e2", "e1", "e0"): {"e0": 2, "e1": 5, "e2": 1},
        ("e1", "e0", "e2"): {"e0": 2, "e1": 5, "e2": 1},
    }
    for order, exp in expected.items():
        gd, _ = greedy_local(m, target, upper, lower, list(order))
        assert gd == exp, order


# --------------------------------------------------------------------------- machinery
def test_stall_patterns_are_deterministic_and_random_access() -> None:
    """Patterns are pure functions of the cycle; ``next_true`` agrees with ``__call__``."""
    for pat in (
        StallPattern.none(),
        StallPattern.bernoulli(0.3, seed=7),
        StallPattern.bursty(3, 2, phase=1),
        StallPattern.single_stall(10, 5),
        StallPattern.periodic(4, phase=1),
    ):
        seq = [pat(t) for t in range(200)]
        assert seq == [pat(t) for t in range(200)]
        for t in range(150):
            nt = pat.next_true(t)
            assert nt >= t and pat(nt)
            assert not any(seq[t:nt])
    b1 = [StallPattern.bernoulli(0.5, seed=1)(t) for t in range(100)]
    b2 = [StallPattern.bernoulli(0.5, seed=2)(t) for t in range(100)]
    assert b1 != b2 and 20 < sum(b1) < 80
    assert StallPattern.bernoulli(0.5, seed=1) == StallPattern.bernoulli(0.5, seed=1)


def test_environment_patterns_change_timing_but_not_order() -> None:
    """A stalled sink slows the chain down; a paced source sets the interval."""
    m = chain_graph()
    inf: dict[str, int | None] = {"e0": None, "e1": None, "e2": None}
    m.chains["sink"].pattern = StallPattern.periodic(4)
    res = simulate(m, inf, max_frames=16, stop_when_stable=False)
    assert res.interval == 32 * 4
    m.chains["sink"].pattern = None
    m.chains["src"].period = 200
    res = simulate(m, inf)
    assert res.stable and res.interval == 200
    assert res.max_occupancy["e0"] < 482
    m.chains["src"].pattern = StallPattern.bernoulli(0.5, seed=3)
    m.chains["src"].period = 0
    res = simulate(m, inf, max_frames=16, stop_when_stable=False)
    assert res.interval == 92  # source is still faster than the bottleneck on average


def test_initial_tokens_and_frame_restart() -> None:
    """An edge with an initial token implements a frame-lagged dependency."""
    m = TEGModel()
    src = Chain("src", kind="source")
    src.events(4, 1, writes=["e0"])
    a = Chain("A")
    a.event(1, reads=["e0", "restart"])
    a.events(2, 1, reads=["e0"])
    a.event(1, reads=["e0"], writes=["e1", "restart"])
    sink = Chain("sink", kind="sink")
    sink.events(1, 1, reads=["e1"])
    for c in (src, a, sink):
        m.add_chain(c)
    m.new_edge("e0", "src", "A", external=True)
    m.new_edge("restart", "A", "A", depth=None, lf=5, initial_tokens=1)
    m.new_edge("e1", "A", "sink", external=True)
    m.validate()
    res = simulate(m, {"e0": None, "e1": None}, record_events=True)
    assert res.event_times is not None
    ta = res.event_times["A"]
    # the next frame's first read waits lf cycles after the previous frame's last event
    assert ta[4] - ta[3] == 5 and ta[8] - ta[7] == 5
    assert res.interval == 8


def test_deadlock_and_timeout_are_reported() -> None:
    """A zero-token cycle deadlocks; a cycle budget yields a timeout."""
    m = forkjoin_graph()
    res = simulate(m, dict.fromkeys(m.external_edges, 1))
    assert res.deadlock and res.interval == math.inf
    m2 = chain_graph()
    res = simulate(m2, {"e0": 2, "e1": 5, "e2": 1}, max_cycles=50)
    assert res.timeout and res.interval == math.inf


def test_milp_size_estimate_matches_built_instance() -> None:
    """``estimate_size`` counts exactly the rows that ``solve_milp`` builds."""
    m = chain_graph()
    cands = unit_candidates(m, [1, 2, 4, 8])
    size = estimate_size(m, cands)
    res = solve_milp(m, 92, cands, time_limit=60)
    assert res.size == size
    assert size.binaries == 12 and size.variables == m.num_events() + 12
    a, b = fit_power_law([(1000, 0.1), (2000, 0.4), (4000, 1.6)])
    assert abs(b - 2.0) < 1e-6 and abs(a * 1000**2 - 0.1) < 1e-6


def test_block_granular_search_on_synthetic_model() -> None:
    """``minimize_depths`` runs SimFIFO's search on the abstract simulator.

    The chain of prototype experiment 1 scaled to 512 tokens per frame needs a FIFO of more
    than 32 entries on ``e1`` (the bursty producer writes 64 tokens back-to-back into a
    consumer that takes one every two cycles).
    """
    m = chain_graph(n=512, burst=64)
    widths = {"e0": 8, "e1": 8, "e2": 8}
    base = measure_bottleneck(m)
    assert base.interval == m.bottleneck_interval()
    paced = measure_peak_occupancy(m, int(base.interval))
    assert paced.stable
    t0 = time.time()
    res = minimize_depths(m, widths, minimization_orders=[MinimizationOrder.NODE_ORDER])
    dt = time.time() - t0
    assert res.final_interval == res.target_interval
    assert res.depths["e2"] == 32
    # e1 needs 33 entries; like the distributed search, the shared search keeps the raw start
    # depth (peak + 1) when depth 32 fails and the SRL search has no room between 33 and 64
    assert res.depths["e1"] == 33
    # e0 buffers the source's in-frame burst; its paced peak exceeds the SRL range and the
    # shared search keeps such FIFOs at one full BRAM block (needs_minimization heuristic)
    assert res.depths["e0"] == 2048
    assert res.simulations > 3 and dt < 60
    # the search result is a fixed point: no SRL-range FIFO can shrink by a block
    for e in m.external_edges:
        if 32 < res.depths[e] <= 256:
            d = dict(res.depths)
            d[e] = res.depths[e] - 32
            assert interval(m, d) > res.target_interval


@pytest.mark.slow
def test_simulator_throughput() -> None:
    """Record the pure-Python event rate on a ~1e6-event graph (M1 acceptance metric)."""
    m = chain_graph(n=4096, burst=64)
    inf: dict[str, int | None] = dict.fromkeys(m.external_edges, None)
    t0 = time.time()
    res = simulate(m, inf, max_frames=40, stop_when_stable=False)
    dt = time.time() - t0
    events = res.frames * m.num_events()
    rate = events / dt
    print(f"\nsimulator throughput: {events} events in {dt:.2f}s = {rate:.0f} events/s")
    assert rate > 2e4


# --------------------------------------------------------------------------- native backend
def _backend_models() -> list[tuple[str, TEGModel, dict[str, int | None], dict[str, object]]]:
    """Models and simulation options that exercise every feature of the simulator."""
    cases: list[tuple[str, TEGModel, dict[str, int | None], dict[str, object]]] = []
    m = chain_graph()
    cases.append(("chain unbounded", m, dict.fromkeys(m.external_edges, None), {}))
    cases.append(("chain bounded", chain_graph(), {"e0": 2, "e1": 9, "e2": 3}, {}))
    cases.append(("chain timeout", chain_graph(), {"e0": 2, "e1": 5, "e2": 1}, {"max_cycles": 50}))
    m = chain_graph()
    m.chains["sink"].pattern = StallPattern.periodic(4)
    cases.append(("periodic sink", m, dict.fromkeys(m.external_edges, None), {}))
    m = chain_graph()
    m.chains["src"].pattern = StallPattern.bernoulli(0.5, seed=3)
    m.chains["sink"].pattern = StallPattern.bursty(3, 2, phase=1)
    cases.append(
        (
            "random source, bursty sink",
            m,
            dict.fromkeys(m.external_edges, None),
            {"max_frames": 16, "stop_when_stable": False},
        )
    )
    m = chain_graph()
    m.chains["src"].period = 200
    m.chains["sink"].pattern = StallPattern.single_stall(40, 30)
    cases.append(("paced source, single stall", m, dict.fromkeys(m.external_edges, None), {}))
    m = forkjoin_graph()
    cases.append(("forkjoin deadlock", m, dict.fromkeys(m.external_edges, 1), {}))
    cases.append(
        (
            "forkjoin ok",
            forkjoin_graph(),
            {"e0": 2, "ea": 2, "eb": 8, "ea2": 8, "eb2": 2, "e3": 2},
            {},
        )
    )
    for mode in ("flp", "rigid"):
        m = _flp_system(mode)
        cases.append((f"{mode} system", m, dict.fromkeys(m.external_edges, 2), {}))
    m, _info = _swg_example()
    cases.append(
        ("swg example", m, dict.fromkeys(m.external_edges, None), {"stable_occupancy": False})
    )
    m, _info = _swg_example()
    cases.append(("swg example bounded", m, dict.fromkeys(m.external_edges, 32), {}))
    return cases


def _freeze_model(ports: int = 1) -> TEGModel:
    """Two-chain flp pipeline with ``ports`` freezer output ports (the first with a
    freeze-until-empty lock), negative start offsets, initial tokens and direct sink edges
    (the single-node harness)."""
    m = TEGModel()
    src = Chain("src", kind="source", pattern=StallPattern.bernoulli(0.3, seed=11))
    src.events(12, 1, writes=["in"])
    slices = [f"slice{j}" for j in range(ports)]
    r = Chain("R", freeze_group="p", start=-1)
    r.event(1, reads=["in", "restart"], writes=["pipe"])
    r.events(11, 1, reads=["in"], writes=["pipe"])
    w = Chain("W", freeze_group="p", start=2)
    w.events(11, 1, reads=["pipe"], writes=slices)
    w.event(1, reads=["pipe"], writes=[*slices, "restart"])
    for c in (src, r, w):
        m.add_chain(c)
    m.new_edge("in", "src", "R", depth=2, external=False, direct=True)
    m.new_edge("pipe", "R", "W", depth=3, lf=3, lb=0)
    m.new_edge("restart", "W", "R", depth=None, lf=1, lb=1, initial_tokens=1)
    for j, sl in enumerate(slices):
        po = Chain(f"PO{j}", freeze_group="p", freezer=True, freeze_until_empty=j == 0)
        po.events(12, 1, reads=[sl], writes=[f"out{j}"])
        sink = Chain(f"sink{j}", kind="sink", pattern=StallPattern.bursty(2, 3, phase=2 * j))
        sink.events(12, 1, reads=[f"out{j}"])
        m.add_chain(po)
        m.add_chain(sink)
        m.new_edge(sl, "W", po.name, depth=2 if j == 0 else 1, lf=1, lb=1 if j == 0 else 0)
        m.new_edge(f"out{j}", po.name, sink.name, depth=1, external=False, direct=True)
    m.validate()
    return m


def _assert_same_result(a: object, b: object, what: str) -> None:
    for field_name in (
        "interval",
        "latency",
        "frames",
        "cycles",
        "stable",
        "deadlock",
        "timeout",
        "max_occupancy",
        "first_valid",
        "frame_end_times",
        "frame_start_times",
    ):
        va, vb = getattr(a, field_name), getattr(b, field_name)
        assert va == vb, f"{what}: {field_name} differs: python {va} vs native {vb}"


def test_native_backend_matches_python() -> None:
    """The compiled simulator reproduces the Python simulator field by field."""
    from finn.analysis.fpgadataflow.teg import native

    if not native.available():
        pytest.skip("no C++ compiler for the native TEG simulator")
    cases = [
        *_backend_models(),
        ("freeze model", _freeze_model(), {}, {}),
        ("freeze model, three ports", _freeze_model(3), {}, {}),
    ]
    for what, m, depths, kw in cases:
        py = simulate(m, depths, backend="python", **kw)  # type: ignore[arg-type]
        nat = simulate(m, depths, backend="native", **kw)  # type: ignore[arg-type]
        _assert_same_result(py, nat, what)
        # a second run of the native backend reuses the cached flat model
        again = simulate(m, depths, backend="native", **kw)  # type: ignore[arg-type]
        _assert_same_result(py, again, what)


def test_native_backend_speed() -> None:
    """The native backend is at least 5x faster than the Python one on a synthetic graph."""
    from finn.analysis.fpgadataflow.teg import native

    if not native.available():
        pytest.skip("no C++ compiler for the native TEG simulator")
    m = chain_graph(n=20000, burst=8)
    depths = {"e0": 16, "e1": 64, "e2": 4}
    t0 = time.perf_counter()
    py = simulate(m, depths, backend="python", max_frames=6, min_frames=6, stop_when_stable=False)
    t_py = time.perf_counter() - t0
    native.flatten(m)  # exclude the one-off flattening from the timing
    t0 = time.perf_counter()
    nat = simulate(m, depths, backend="native", max_frames=6, min_frames=6, stop_when_stable=False)
    t_nat = time.perf_counter() - t0
    _assert_same_result(py, nat, "speed model")
    print(f"\npython {t_py:.2f} s, native {t_nat:.3f} s, speedup {t_py / max(t_nat, 1e-9):.0f}x")
    assert t_nat * 5 < t_py

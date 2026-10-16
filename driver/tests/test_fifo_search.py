"""Tests of the FIFO depth search strategies against a synthetic oracle (no board required).

Run from the driver directory with: python -m pytest tests
"""

import pytest

import itertools
import random

from finn_plus_driver.overlays.fifo_search import (
    BINARY_ORDERS,
    SEARCH_STRATEGIES,
    UNBOUNDED_DEPTH,
    EvalResult,
    EvaluationCache,
    FeasibilityCheck,
    Lattice,
    PrimitiveCost,
    SearchBudget,
    SearchContext,
    TotalBitsCost,
    cache_to_iterations,
    compute_lower_bounds,
    producer_blocked_from_stall,
    resolve_strategy_name,
    run_search,
    search_result_to_dict,
)

FEASIBLE_INTERVAL = 100
DEGRADED_INTERVAL = 150
FEASIBLE_LATENCY = 1000
DEGRADED_LATENCY = 1200


class MockOracle:
    """Monotone feasibility predicate built from inequalities ``sum(d[e] for e in T) >= b``.

    ``producer_blocked`` marks every FIFO that appears in a violated inequality, which is a sound
    storage-dependency set: any feasible configuration above the evaluated one must enlarge one
    of them. ``peak`` is a feasible configuration below the evaluated one obtained by lowering
    the FIFOs one after the other as far as the inequalities allow (a stand-in for "the smallest
    capacities that reproduce the same execution"). Severe violations are reported as deadlocks.
    """

    def __init__(self, num_fifos, constraints, report_flags=True, report_peaks=True):
        """``constraints`` is a list of ``(fifo_set, bound)`` pairs."""
        self.num_fifos = num_fifos
        self.constraints = [(tuple(sorted(T)), b) for T, b in constraints]
        self.report_flags = report_flags
        self.report_peaks = report_peaks
        self.calls = 0

    def feasible(self, depths):
        """Plain predicate on a depth vector."""
        return all(sum(depths[e] for e in T) >= b for T, b in self.constraints)

    def peak(self, depths):
        """Lower every FIFO in turn to the minimum the constraints allow given the others."""
        p = list(depths)
        for e in range(self.num_fifos):
            need = 1
            for T, b in self.constraints:
                if e in T:
                    need = max(need, b - sum(p[f] for f in T if f != e))
            p[e] = min(p[e], max(1, need))
        assert self.feasible(p)
        return tuple(p)

    def evaluate(self, depths, need_flags=False, need_peaks=False):
        """Evaluate one configuration."""
        self.calls += 1
        depths = tuple(depths)
        feasible = self.feasible(depths)
        violated = [T for T, b in self.constraints if sum(depths[e] for e in T) < b]
        severe = any(sum(depths[e] for e in T) * 2 < b for T, b in self.constraints)
        blocked = tuple(any(e in T for T in violated) for e in range(self.num_fifos))
        if feasible:
            interval, latency = FEASIBLE_INTERVAL, FEASIBLE_LATENCY
        elif severe:
            interval, latency = 0, 0
        else:
            interval, latency = DEGRADED_INTERVAL, DEGRADED_LATENCY
        return EvalResult(
            depths=depths,
            interval=interval,
            latency=latency,
            min_latency=latency,
            deadlock=interval == 0,
            producer_blocked=blocked if (need_flags and self.report_flags) else None,
            stall_flags=None,
            peak=(self.peak(depths) if feasible else depths)
            if (need_peaks and self.report_peaks)
            else None,
        )


def brute_force_optimum(oracle, cost_model, max_depths):
    """Minimum cost over the full box ``[1, max_depths]``."""
    best = None
    for d in itertools.product(*(range(1, m + 1) for m in max_depths)):
        if oracle.feasible(d):
            c = cost_model.cost(d)
            if best is None or c < best:
                best = c
    return best


def random_instance(rng, num_fifos=None, box_limit=30000):
    """Random monotone instance whose all-max configuration is feasible."""
    n = num_fifos or rng.choice([3, 4, 5])
    max_depth = {3: rng.randint(6, 12), 4: rng.randint(5, 10), 5: rng.randint(4, 7)}[n]
    while max_depth**n > box_limit:
        max_depth -= 1
    max_depths = [max_depth] * n
    widths = [rng.choice([8, 16, 32, 64]) for _ in range(n)]
    constraints = []
    for e in range(n):  # a singleton bound for most FIFOs
        if rng.random() < 0.8:
            constraints.append(({e}, rng.randint(1, max_depth)))
    for _ in range(rng.randint(1, n)):  # a few chain/fork couplings
        size = rng.randint(2, min(3, n))
        T = set(rng.sample(range(n), size))
        constraints.append((T, rng.randint(size, size * max_depth)))
    return MockOracle(n, constraints), widths, max_depths


def synthetic_tables(widths, max_depth=64, rng=None):
    """Cost tables in the ``fifo_cost_tables`` format with a non-monotone ladder per FIFO.

    Depths are real FIFO depths (register value + 1). Every few depths the LUTs grow; now and
    then a "memory tier" adds a RAMB18 and drops LUTs, sometimes by more than the tile costs,
    so that a deeper step can be cheaper than a shallower one (as in the real ladder).
    """
    rng = rng or random.Random(0)
    fifos = {}
    for i, w in enumerate(widths):
        steps = []
        d = rng.randint(2, 5)
        lut, bram = w // 2 + 10, 0
        while d < max_depth:
            steps.append([d, [lut, w, bram, 0]])
            d += rng.randint(2, 8)
            if rng.random() < 0.3:
                bram += 1
                lut = max(10, lut - rng.randint(0, 45))
            else:
                lut += rng.randint(1, 10)
        steps.append([max_depth, [lut, w, bram, 0]])
        fifos[str(i)] = {"width": w, "ram_style": "auto", "steps": steps}
    return {
        "resources": ["lut", "ff", "bram18", "uram"],
        "weights": {"lut": 1.0, "ff": 0.5, "bram18": 30.0, "uram": 400.0},
        "weights_source": "test",
        "fpgapart": None,
        "versal": False,
        "max_depth": max_depth,
        "fifos": fifos,
    }


def make_context(
    oracle, widths, start_depths, stop_condition="both", budget=None, cost_model=None, **kwargs
):
    """Search context with the targets measured at ``start_depths``."""
    cost_model = cost_model if cost_model is not None else TotalBitsCost(widths)
    cache = EvaluationCache(oracle, cost_model)
    target = cache.evaluate(start_depths, role="target")
    assert oracle.feasible(start_depths)
    feasibility = FeasibilityCheck.from_result(stop_condition, 0.0, target.result)
    return SearchContext(
        oracle,
        feasibility,
        cost_model,
        start_depths,
        widths,
        cache=cache,
        budget=budget,
        log=lambda *_: None,
        rng=random.Random(0),
        **kwargs,
    )


# ---------------------------------------------------------------------------------------


def test_feasibility_thresholds_match_original_formulas():
    """Thresholds are target*(1+relaxation) and target*(1+0.0005+relaxation)."""
    target = EvalResult((1,), 200, 4000, 3000, False)
    fb = FeasibilityCheck.from_result("both", 0.1, target)
    assert fb.relaxed_interval_threshold == pytest.approx(220.0)
    assert fb.relaxed_latency_threshold == pytest.approx(4000 * 1.1005)
    assert fb.feasible(EvalResult((1,), 220, 4000, 1, False))
    assert not fb.feasible(EvalResult((1,), 221, 4000, 1, False))
    assert not fb.feasible(EvalResult((1,), 200, 4403, 1, False))
    assert not fb.feasible(EvalResult((1,), 200, 4000, 1, True))
    assert FeasibilityCheck("interval", 0.0, 200, 4000).feasible(
        EvalResult((1,), 200, 9999, 1, False)
    )
    assert FeasibilityCheck("latency", 0.0, 200, 4000).feasible(
        EvalResult((1,), 9999, 4000, 1, False)
    )
    with pytest.raises(ValueError):
        FeasibilityCheck("foo", 0.0, 1, 1)


def test_stall_byte_decoding():
    """Bit 1 of the sticky nibble is the producer-blocked flag."""
    assert producer_blocked_from_stall(0b0010_0000)
    assert not producer_blocked_from_stall(0b0000_0010)
    assert producer_blocked_from_stall(0b0000_0010, sticky=False)
    assert not producer_blocked_from_stall(0b1101_1101)


def test_cost_model_and_lattice():
    """kB cost with the depth offset; unbounded is infinite; lattice bounds are respected."""
    cm = TotalBitsCost([8, 16], offset=1)
    assert cm.cost([1, 1]) == pytest.approx((2 * 8 + 2 * 16) / 8 / 1000)
    assert cm.cost([UNBOUNDED_DEPTH, 1]) == float("inf")
    lat = Lattice([1, 2], [4, 2])
    assert lat.next_up(0, 4) is None and lat.next_up(0, 3) == 4
    assert lat.next_down(1, 2) is None
    assert lat.clamp([0, 9]) == (1, 2)
    assert lat.mid(0, 1, 4) == 2
    assert lat.contains((2, 2)) and not lat.contains((5, 2))


def test_resolve_strategy_name():
    """The deprecated fifo_search_order maps onto the binary strategies."""
    assert resolve_strategy_name() == "binary_largest_first"
    assert resolve_strategy_name(fifo_search_order="reverse") == "binary_reverse"
    assert resolve_strategy_name("top_down_bnb", "reverse") == "top_down_bnb"
    assert set(f"binary_{o}" for o in BINARY_ORDERS) <= set(SEARCH_STRATEGIES)


def reference_binary_search(oracle, widths, start_depths, order):
    """Port of the original size_iteratively_binary_search loop for the mock predicate."""
    from finn_plus_driver.overlays.fifo_search import binary_fifo_order

    fifo_depths = list(start_depths)
    sequence = []
    for fifo_id in binary_fifo_order(order, start_depths, widths, random.Random(0)):
        low, high = 1, fifo_depths[fifo_id]
        best = high
        while low <= high:
            mid = (low + high) // 2
            test = fifo_depths.copy()
            test[fifo_id] = mid
            sequence.append(tuple(test))
            if oracle.feasible(test):
                best = mid
                high = mid - 1
            else:
                low = mid + 1
        fifo_depths[fifo_id] = best
    return fifo_depths, sequence


@pytest.mark.parametrize("order", BINARY_ORDERS)
def test_binary_search_matches_reference(order):
    """The re-expressed binary search tests the same configurations in the same order."""
    rng = random.Random(1)
    for _ in range(20):
        oracle, widths, max_depths = random_instance(rng)
        ctx = make_context(oracle, widths, tuple(max_depths))
        res = run_search(f"binary_{order}", ctx)
        ref_depths, ref_sequence = reference_binary_search(oracle, widths, max_depths, order)
        assert list(res.depths) == ref_depths
        sequence = [e.depths for e in ctx.cache.log if e.info.get("phase") == "binary"]
        assert sequence == ref_sequence
        assert oracle.feasible(res.depths)
        assert not res.optimal_proven
        assert res.num_evaluations == len(ref_sequence) + 1  # plus the target measurement
        # all_iterations format of the original driver
        its = cache_to_iterations(ctx.cache)
        assert its["0"]["fifo_depths"] == max_depths and its["0"]["role"] == "target"
        for k, it in its.items():
            if k != "0":
                assert {"tested_fifo", "tested_depth", "status", "search_bounds"} <= set(it)
                assert it["status"] in ("PASS", "FAIL")


def test_exact_strategies_find_the_brute_force_optimum():
    """bottom_up_exact and top_down_bnb return the optimum on 200 random instances."""
    rng = random.Random(1234)
    stats = {"bottom_up_exact": [], "top_down_bnb": [], "top_down_bnb_best_bound": [], "binary": []}
    for _ in range(200):
        oracle, widths, max_depths = random_instance(rng)
        cost_model = TotalBitsCost(widths)
        optimum = brute_force_optimum(oracle, cost_model, max_depths)
        assert optimum is not None

        ctx = make_context(oracle, widths, tuple(max_depths))
        lb = compute_lower_bounds(ctx)
        assert all(1 <= b <= m for b, m in zip(lb, max_depths))
        assert cost_model.cost(lb) <= optimum + 1e-12
        # every feasible configuration lies above the lower bounds
        for d in itertools.product(*(range(1, m + 1) for m in max_depths[:3])):
            full = tuple(d) + tuple(max_depths[3:])
            if oracle.feasible(full):
                assert all(x >= b for x, b in zip(full, lb))

        res = run_search("bottom_up_exact", ctx, upper_bound_factor=1.0)
        assert res.optimal_proven, res.abort_reason
        assert oracle.feasible(res.depths)
        assert res.cost == pytest.approx(optimum)
        assert res.lower_bounds == lb
        stats["bottom_up_exact"].append(res.num_evaluations)

        for node_order in ("dfs", "best_bound"):
            ctx = make_context(oracle, widths, tuple(max_depths))
            res = run_search("top_down_bnb", ctx, upper_bound_factor=1.0, node_order=node_order)
            assert res.optimal_proven, res.abort_reason
            assert res.open_nodes == 0
            assert oracle.feasible(res.depths)
            assert res.cost == pytest.approx(optimum)
            assert res.lower_bound_cost == pytest.approx(optimum)
            assert res.seed_strategy == "binary_largest_first"
            assert res.cost <= res.seed_cost + 1e-12
            key = "top_down_bnb" if node_order == "dfs" else "top_down_bnb_best_bound"
            stats[key].append(res.num_evaluations)

        for order in BINARY_ORDERS:
            ctx = make_context(oracle, widths, tuple(max_depths))
            res = run_search(f"binary_{order}", ctx)
            assert oracle.feasible(res.depths)
            assert res.cost >= optimum - 1e-12
            stats["binary"].append(res.num_evaluations)

    for name, counts in stats.items():
        print(f"{name}: mean evaluations {sum(counts) / len(counts):.1f}, max {max(counts)}")


def test_top_down_bnb_without_peaks_or_seed():
    """The B&B is exact without peak information and with the start configuration as seed."""
    rng = random.Random(7)
    for _ in range(30):
        oracle, widths, max_depths = random_instance(rng)
        oracle.report_peaks = False
        optimum = brute_force_optimum(oracle, TotalBitsCost(widths), max_depths)
        ctx = make_context(oracle, widths, tuple(max_depths))
        res = run_search("top_down_bnb", ctx, upper_bound_factor=1.0, seed_strategy=None)
        assert res.optimal_proven and res.cost == pytest.approx(optimum)
        assert res.seed_depths == tuple(max_depths)


def test_bottom_up_refuses_unreliable_flags():
    """Without producer-blocked flags the bottom-up search returns its seed and says why."""
    # the coupled constraint makes the lower-bound corner (1, 1, 2) infeasible
    oracle = MockOracle(3, [({0, 1}, 6), ({2}, 2)], report_flags=False)
    widths, max_depths = [8, 16, 32], [5, 5, 5]
    ctx = make_context(oracle, widths, tuple(max_depths))
    res = run_search("bottom_up_exact", ctx, upper_bound_factor=1.0)
    assert not res.optimal_proven
    assert res.abort_reason is not None
    assert res.depths == tuple(max_depths) and oracle.feasible(res.depths)
    # expanding every FIFO instead is still exact, only slower
    optimum = brute_force_optimum(oracle, TotalBitsCost(widths), max_depths)
    ctx = make_context(oracle, widths, tuple(max_depths))
    res = run_search("bottom_up_exact", ctx, upper_bound_factor=1.0, allow_unreliable_flags=True)
    assert res.optimal_proven and res.cost == pytest.approx(optimum)


def test_budget_exhaustion_returns_a_feasible_configuration():
    """With a tiny budget both exact searches return a feasible point and a valid bound."""
    rng = random.Random(11)
    for strategy in ("bottom_up_exact", "top_down_bnb"):
        oracle, widths, max_depths = random_instance(rng, num_fifos=4)
        optimum = brute_force_optimum(oracle, TotalBitsCost(widths), max_depths)
        ctx = make_context(
            oracle, widths, tuple(max_depths), budget=SearchBudget(max_evaluations=12)
        )
        res = run_search(strategy, ctx, upper_bound_factor=1.0, seed_strategy=None)
        assert oracle.feasible(res.depths)
        assert res.cost >= optimum - 1e-12
        assert res.budget_exhausted or res.optimal_proven
        assert res.lower_bound_cost <= optimum + 1e-12
        assert ctx.cache.num_evaluations <= 12 + 1  # the confirmation may add one
        d = search_result_to_dict(res)
        assert d["optimality_gap_kB"] >= 0


def test_bottom_up_pareto_records_trade_off_curve():
    """In pareto mode the search records every evaluated point up to the cost limit."""
    rng = random.Random(5)
    oracle, widths, max_depths = random_instance(rng, num_fifos=3)
    optimum = brute_force_optimum(oracle, TotalBitsCost(widths), max_depths)
    ctx = make_context(oracle, widths, tuple(max_depths))
    res = run_search("bottom_up_exact", ctx, upper_bound_factor=1.0, pareto=True)
    assert res.optimal_proven and res.cost == pytest.approx(optimum)
    assert len(res.pareto) >= 1
    costs = [p["cost"] for p in res.pareto]
    assert costs == sorted(costs)
    assert all(p["cost"] <= res.seed_cost + 1e-12 for p in res.pareto)
    feasible_points = [p for p in res.pareto if p["feasible"]]
    assert feasible_points[0]["cost"] == pytest.approx(optimum)


def test_lower_bound_probes_use_unbounded_depths():
    """The lower-bound phase probes one FIFO with all others unbounded."""
    rng = random.Random(9)
    oracle, widths, max_depths = random_instance(rng, num_fifos=3)
    ctx = make_context(oracle, widths, tuple(max_depths))
    compute_lower_bounds(ctx)
    probes = [e for e in ctx.cache.log if e.info.get("phase") == "lower_bound"]
    assert probes
    for e in probes:
        assert sum(1 for d in e.depths if d == UNBOUNDED_DEPTH) == 2
        assert cache_to_iterations(ctx.cache)[str(e.index)]["total_fifo_size_kB"] is None


def test_cache_reuses_evaluations_only_with_requested_data():
    """A cached result without flags is not reused for a request that needs flags."""
    oracle = MockOracle(2, [({0}, 2), ({1}, 2)])
    cache = EvaluationCache(oracle, TotalBitsCost([8, 8]))
    cache.evaluate((3, 3))
    assert oracle.calls == 1
    cache.evaluate((3, 3))
    assert oracle.calls == 1 and cache.num_cache_hits == 1
    cache.evaluate((3, 3), need_flags=True)
    assert oracle.calls == 2
    cache.evaluate((3, 3), need_flags=True)
    assert oracle.calls == 2
    cache.evaluate((3, 3), reuse=False)
    assert oracle.calls == 3 and cache.num_evaluations == 3


# ---------------------------------------------------------------------------------------
# Primitive cost tables and the candidate lattice
# ---------------------------------------------------------------------------------------


def test_primitive_cost_table_lookup():
    """Envelope lookup, realization, breakpoints, vectors and extrapolation on a small table."""
    tables = {
        "resources": ["lut", "ff", "bram18", "uram"],
        "weights": {"lut": 1, "ff": 0, "bram18": 100, "uram": 1000},
        "fifos": {
            "0": {
                "width": 8,
                "ram_style": "block",
                # scalars 10, 20, 150, 130, 140: the third step is dominated by the fourth
                "steps": [
                    [4, [10, 5, 0, 0]],
                    [8, [20, 5, 0, 0]],
                    [12, [150, 5, 0, 0]],
                    [16, [30, 5, 1, 0]],
                    [24, [40, 5, 1, 0]],
                ],
            }
        },
    }
    cm = PrimitiveCost(tables, offset=1)
    assert cm.num_fifos == 1 and cm.unit == "LUTeq" and cm.ram_styles == ["block"]
    # register value + offset is the depth looked up
    assert cm.cost_fifo(0, 0) == 10 and cm.cost_fifo(0, 3) == 10 and cm.cost_fifo(0, 4) == 20
    assert cm.cost_fifo(0, 9) == 130 and cm.cost_fifo(0, 15) == 130 and cm.cost_fifo(0, 16) == 140
    assert cm.vector_fifo(0, 9) == [30, 5, 1, 0]
    assert cm.realize([9]) == (15,) and cm.realize([3]) == (3,) and cm.realize([16]) == (23,)
    assert cm.breakpoints(0) == [3, 7, 15, 23]
    assert cm.cost_fifo(0, UNBOUNDED_DEPTH) == float("inf")
    assert cm.realize([UNBOUNDED_DEPTH]) == (UNBOUNDED_DEPTH,)
    # beyond the table: the last step continued with the slope of the last two
    assert cm.cost_fifo(0, 31) == pytest.approx(50 + 100)
    assert cm.realize([31]) == (31,)
    assert cm.cost([15]) == 130 and cm.vector([15]) == {"lut": 30, "ff": 5, "bram18": 1, "uram": 0}
    assert cm.describe()["weights"]["bram18"] == 100
    # the envelope is non-decreasing in the register value
    costs = [cm.cost_fifo(0, r) for r in range(0, 40)]
    assert all(a <= b for a, b in zip(costs, costs[1:]))
    with pytest.raises(ValueError):
        PrimitiveCost({"resources": ["lut"], "weights": {}, "fifos": {"0": {"steps": []}}})


def test_lattice_with_candidates():
    """Candidates inside the bounds plus the bounds; the last spacing continues beyond."""
    lat = Lattice([1, 3], [20, 10], candidates=[[4, 8, 12], [2, 5, 9, 30]])
    assert lat.cands(0) == [1, 4, 8, 12, 16, 20]
    assert lat.cands(1) == [3, 5, 9, 10]
    assert lat.next_up(0, 1) == 4 and lat.next_up(0, 5) == 8 and lat.next_up(0, 20) is None
    assert lat.next_down(0, 8) == 4 and lat.next_down(0, 5) == 4 and lat.next_down(0, 1) is None
    assert lat.mid(0, 1, 20) == 8 and lat.mid(0, 5, 7) is None and lat.mid(0, 9, 12) == 12
    assert lat.clamp([5, 0]) == (8, 3) and lat.clamp([99, 10]) == (20, 10)
    restricted = lat.restrict([4, 5], [12, 9])
    assert restricted.cands(0) == [4, 8, 12] and restricted.cands(1) == [5, 9]
    assert restricted.contains((8, 9)) and not restricted.contains((13, 9))
    # the integer lattice is unchanged
    plain = Lattice([1], [4])
    assert plain.cands(0) is None and plain.mid(0, 1, 4) == 2 and plain.mid(0, 5, 4) is None
    assert plain.restrict([2], [3]).next_up(0, 3) is None


def test_exact_strategies_find_the_optimum_with_primitive_cost():
    """With a non-monotone primitive ladder the searches minimise the cost envelope exactly."""
    rng = random.Random(99)
    num_candidates, num_total = 0, 0
    for _ in range(60):
        oracle, widths, max_depths = random_instance(rng, box_limit=4000)
        tables = synthetic_tables(widths, max_depth=max(max_depths) + 8, rng=rng)
        cost_model = PrimitiveCost(tables, offset=1)
        optimum = brute_force_optimum(oracle, cost_model, max_depths)
        candidates = [cost_model.breakpoints(i) for i in range(len(widths))]
        lattice = Lattice([1] * len(widths), max_depths, candidates=candidates)
        num_candidates += sum(len(lattice.cands(i)) for i in range(len(widths)))
        num_total += sum(max_depths)

        for strategy, kwargs in (
            ("bottom_up_exact", {}),
            ("top_down_bnb", {"node_order": "dfs"}),
            ("top_down_bnb", {"node_order": "best_bound", "seed_strategy": None}),
        ):
            ctx = make_context(
                oracle, widths, tuple(max_depths), cost_model=cost_model, lattice=lattice
            )
            res = run_search(strategy, ctx, upper_bound_factor=1.0, **kwargs)
            assert res.optimal_proven, res.abort_reason
            assert res.cost_unit == "LUTeq"
            assert oracle.feasible(res.depths)
            assert res.cost == pytest.approx(optimum)
            realized = cost_model.realize(res.depths)
            assert all(r >= d for r, d in zip(realized, res.depths))
            assert oracle.feasible(realized) and cost_model.cost(realized) == pytest.approx(optimum)
            # the true (non-envelope) cost of the realized configuration is the envelope cost
            true_cost = sum(
                cost_model._scalars[i][cost_model._step(i, d)] for i, d in enumerate(realized)
            )
            assert true_cost == pytest.approx(optimum)

        for order in ("largest_first", "random"):
            ctx = make_context(
                oracle, widths, tuple(max_depths), cost_model=cost_model, lattice=lattice
            )
            res = run_search(f"binary_{order}", ctx)
            assert oracle.feasible(res.depths) and res.cost >= optimum - 1e-9
            # the binary search only tests candidate depths
            for entry in ctx.cache.log[1:]:
                i, d = entry.info["tested_fifo"], entry.info["tested_depth"]
                assert d in lattice.cands(i)
    assert num_candidates < num_total / 2

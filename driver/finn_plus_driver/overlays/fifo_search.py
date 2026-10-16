"""Oracle-agnostic search strategies for FIFO depth configurations.

The live FIFO sizing driver (``live_fifo.py``) measures the performance of a dataflow
accelerator for a given vector of FIFO depths on the FPGA. This module contains everything
*around* such a measurement that does not depend on the hardware: the ``Oracle`` protocol a
measurement backend implements, the feasibility predicate, cost models, the depth lattice, an
evaluation cache that doubles as the search log, and the search strategies themselves:

- ``binary_<order>``: the per-FIFO binary search (coordinate descent) of the original driver,
  for every FIFO ordering it supports. Locally optimal only.
- ``bottom_up_exact``: best-first search from per-FIFO lower bounds upwards, expanding only
  FIFOs that blocked their producer (Stuijk, Geilen, Basten: "Throughput-buffering trade-off
  exploration for cyclo-static and synchronous dataflow graphs", IEEE TC 2008, Algorithm 1).
  The first feasible configuration found is cost-optimal.
- ``top_down_bnb``: branch-and-bound over boxes of depth vectors starting from a feasible
  upper corner. Anytime, exact when run to completion.

All strategies only rely on the monotonicity of feasibility in the FIFO depths and on the
information in ``EvalResult``; how an oracle obtains it (hardware, RTL simulation, abstract
simulation) is irrelevant here, so no PYNQ import must ever be added to this module.
"""

import heapq
import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

#: Capacity register value that makes a virtual FIFO effectively unbounded. Bit 31 is kept
#: clear because the gauge reserves it as a sticky overflow bit in its running-max logic.
UNBOUNDED_DEPTH = 0x7FFFFFFF

#: Bit layout of the ``READ_STALL`` status byte of the virtual FIFO gauge (``fifo_gauge_pkg``).
#: Bits [3:0] are latched per frame, bits [7:4] accumulate since the last ``RUN_*`` command.
STALL_IVLD = 1 << 0  #: irdy && !ivld: producer idle
STALL_IRDY = 1 << 1  #: ivld && !irdy: producer blocked by a full FIFO
STALL_OVLD = 1 << 2  #: ordy && !ovld: consumer starved by an empty FIFO
STALL_ORDY = 1 << 3  #: ovld && !ordy: consumer blocked
STALL_STICKY_SHIFT = 4


def producer_blocked_from_stall(flags: int, sticky: bool = True) -> bool:
    """Return whether a ``READ_STALL`` byte reports a producer blocked by a full FIFO."""
    nibble = (flags >> STALL_STICKY_SHIFT) if sticky else flags
    return bool(nibble & STALL_IRDY)


@dataclass(frozen=True)
class EvalResult:
    """Outcome of evaluating one FIFO depth configuration.

    ``depths`` are capacity register values (the real depth adds the driver's offset).
    ``producer_blocked`` and ``peak`` are ``None`` when the oracle cannot provide them.
    ``flags_reliable`` is ``False`` when ``producer_blocked`` was derived from flags that are
    only latched at frame boundaries (stale in a deadlock) instead of run-sticky ones.
    """

    depths: Tuple[int, ...]
    interval: int
    latency: int
    min_latency: int
    deadlock: bool
    producer_blocked: Optional[Tuple[bool, ...]] = None
    stall_flags: Optional[Tuple[int, ...]] = None
    peak: Optional[Tuple[int, ...]] = None
    flags_reliable: bool = True


class Oracle(Protocol):
    """Measurement backend: evaluates a depth configuration and reports the outcome."""

    num_fifos: int

    def evaluate(
        self, depths: Sequence[int], need_flags: bool = False, need_peaks: bool = False
    ) -> EvalResult:
        """Evaluate a configuration; ``need_flags``/``need_peaks`` request the optional data."""
        ...


class FeasibilityCheck:
    """Decides whether an evaluation meets the performance target within tolerances.

    The thresholds are the ones of the original binary search: the interval may degrade by
    ``relaxation``, the latency by ``relaxation`` plus a fixed default tolerance for the
    measurement jitter observed on some models. Deadlocked evaluations are never feasible.
    """

    LATENCY_DEFAULT_RELAXATION = 0.0005

    def __init__(self, stop_condition, relaxation, target_interval, target_latency):
        """Set up the predicate for a stop condition, a relaxation and the measured targets."""
        if stop_condition not in ("interval", "latency", "both"):
            raise ValueError(f"Unknown stop_condition: {stop_condition}")
        self.stop_condition = stop_condition
        self.relaxation = relaxation
        self.target_interval = target_interval
        self.target_latency = target_latency
        self.relaxed_interval_threshold = target_interval * (1 + relaxation)
        self.relaxed_latency_threshold = target_latency * (
            1 + (self.LATENCY_DEFAULT_RELAXATION + relaxation)
        )

    @classmethod
    def from_result(cls, stop_condition, relaxation, target: EvalResult):
        """Derive the targets from the evaluation of the starting configuration."""
        return cls(stop_condition, relaxation, target.interval, target.latency)

    def degraded(self, result: EvalResult) -> bool:
        """Return whether the result violates the selected performance target(s)."""
        interval_degraded = result.interval > self.relaxed_interval_threshold
        latency_degraded = result.latency > self.relaxed_latency_threshold
        if self.stop_condition == "interval":
            return interval_degraded
        if self.stop_condition == "latency":
            return latency_degraded
        return interval_degraded or latency_degraded

    def feasible(self, result: EvalResult) -> bool:
        """Return whether the configuration is acceptable: no deadlock and no degradation."""
        return not result.deadlock and not self.degraded(result)


class CostModel(Protocol):
    """Cost of a depth configuration; ``cost_fifo`` must be non-decreasing in the depth."""

    def cost(self, depths: Sequence[int]) -> float:
        """Total cost of a configuration (to be minimised)."""
        ...

    def cost_fifo(self, i: int, depth: int) -> float:
        """Cost contribution of FIFO ``i`` at the given depth."""
        ...


class TotalBitsCost:
    """Total FIFO storage in kB: ``sum((depth + offset) * width)`` as reported by the driver."""

    def __init__(self, widths: Sequence[int], offset: int = 1):
        """Widths in bits per FIFO (indexed by FIFO ID) and the depth offset of the driver."""
        self.widths = list(widths)
        self.offset = offset

    def cost_fifo(self, i: int, depth: int) -> float:
        """Storage of one FIFO in kB; unbounded FIFOs are infinitely expensive."""
        if depth >= UNBOUNDED_DEPTH:
            return math.inf
        return (depth + self.offset) * self.widths[i] / 8.0 / 1000.0

    def cost(self, depths: Sequence[int]) -> float:
        """Total storage of a configuration in kB."""
        return sum(self.cost_fifo(i, d) for i, d in enumerate(depths))


class PrimitiveCost:
    """Placeholder for a cost model counting FPGA primitives (SRL/BRAM/URAM steps).

    The metric is not defined yet. Once it is, ``breakpoints`` lets the lattice restrict the
    candidate depths of a FIFO to the depths at which its cost changes, which shrinks the search
    space without any loss in that metric.
    """

    def __init__(self, widths: Sequence[int], offset: int = 1):
        """Widths in bits per FIFO and the depth offset of the driver."""
        self.widths = list(widths)
        self.offset = offset

    def cost_fifo(self, i: int, depth: int) -> float:
        """Not defined yet."""
        raise NotImplementedError("The FPGA primitive cost metric is defined later.")

    def cost(self, depths: Sequence[int]) -> float:
        """Not defined yet."""
        raise NotImplementedError("The FPGA primitive cost metric is defined later.")

    def breakpoints(self, i: int) -> List[int]:
        """Depths at which the cost of FIFO ``i`` changes (hook for a lattice restriction)."""
        raise NotImplementedError("The FPGA primitive cost metric is defined later.")


class Lattice:
    """Candidate depths per FIFO: integer ranges ``[lower_i, upper_i]`` with a common step.

    Every access of the search strategies to neighbouring or intermediate depths goes through
    this class, so that a candidate set based on cost breakpoints can replace the ranges later
    without touching the searches.
    """

    def __init__(self, lower: Sequence[int], upper: Sequence[int], step: int = 1):
        """Per-FIFO lower and upper bounds (inclusive) and the step between candidates."""
        if len(lower) != len(upper):
            raise ValueError("lower and upper bounds must have the same length")
        self.lower = tuple(int(x) for x in lower)
        self.upper = tuple(max(int(u), lo) for u, lo in zip(upper, self.lower))
        self.step = step

    @property
    def num_fifos(self) -> int:
        """Number of FIFOs."""
        return len(self.lower)

    def next_up(self, i: int, depth: int) -> Optional[int]:
        """Next larger candidate depth of FIFO ``i``, or ``None`` at the upper bound."""
        nxt = depth + self.step
        return nxt if nxt <= self.upper[i] else None

    def next_down(self, i: int, depth: int) -> Optional[int]:
        """Next smaller candidate depth of FIFO ``i``, or ``None`` at the lower bound."""
        nxt = depth - self.step
        return nxt if nxt >= self.lower[i] else None

    def mid(self, i: int, lo: int, hi: int) -> int:
        """Candidate depth in the middle of ``[lo, hi]`` for FIFO ``i``."""
        return (lo + hi) // 2

    def clamp(self, depths: Sequence[int]) -> Tuple[int, ...]:
        """Clamp a configuration into the lattice bounds."""
        return tuple(min(max(int(d), lo), up) for d, lo, up in zip(depths, self.lower, self.upper))

    def contains(self, depths: Sequence[int]) -> bool:
        """Return whether a configuration lies within the lattice bounds."""
        return all(lo <= d <= up for d, lo, up in zip(depths, self.lower, self.upper))


class SearchBudget:
    """Limits on the number of oracle evaluations and the wall-clock time of a search."""

    def __init__(
        self, max_evaluations: Optional[int] = None, max_runtime_s: Optional[float] = None
    ):
        """Either limit may be ``None`` for "unlimited"."""
        self.max_evaluations = max_evaluations
        self.max_runtime_s = max_runtime_s
        self.start_time = time.time()

    def start(self):
        """(Re)start the wall-clock budget."""
        self.start_time = time.time()

    def exhausted(self, num_evaluations: int) -> bool:
        """Return whether either limit has been reached."""
        if self.max_evaluations is not None and num_evaluations >= self.max_evaluations:
            return True
        if self.max_runtime_s is not None and time.time() - self.start_time >= self.max_runtime_s:
            return True
        return False


@dataclass
class LogEntry:
    """One oracle evaluation in the search log."""

    index: int
    depths: Tuple[int, ...]
    result: EvalResult
    cost: float
    info: Dict = field(default_factory=dict)


class EvaluationCache:
    """Memoises oracle evaluations and records every real evaluation in order.

    The cache only stores evaluations that actually ran; a cache hit does not add a log entry
    (``num_cache_hits`` counts them). ``evaluate`` returns the log entry of the evaluation so a
    strategy can annotate it with its own bookkeeping (``info``).
    """

    def __init__(self, oracle: Oracle, cost_model: CostModel):
        """Wrap an oracle and the cost model used for the logged cost of every evaluation."""
        self.oracle = oracle
        self.cost_model = cost_model
        self.results: Dict[Tuple[int, ...], LogEntry] = {}
        self.log: List[LogEntry] = []
        self.num_cache_hits = 0

    @property
    def num_evaluations(self) -> int:
        """Number of real oracle evaluations so far."""
        return len(self.log)

    def get(self, depths: Sequence[int]) -> Optional[LogEntry]:
        """Return the cached entry of a configuration, if any."""
        return self.results.get(tuple(depths))

    def evaluate(
        self,
        depths: Sequence[int],
        reuse: bool = True,
        need_flags: bool = False,
        need_peaks: bool = False,
        **info,
    ) -> LogEntry:
        """Evaluate a configuration (or reuse a cached result) and return its log entry.

        A cached result is only reused when it carries the requested optional data.
        """
        key = tuple(int(d) for d in depths)
        if reuse:
            hit = self.results.get(key)
            if hit is not None:
                if (not need_flags or hit.result.producer_blocked is not None) and (
                    not need_peaks or hit.result.peak is not None
                ):
                    self.num_cache_hits += 1
                    return hit
        result = self.oracle.evaluate(key, need_flags=need_flags, need_peaks=need_peaks)
        entry = LogEntry(
            index=len(self.log),
            depths=key,
            result=result,
            cost=self.cost_model.cost(key),
            info=dict(info),
        )
        self.log.append(entry)
        self.results[key] = entry
        return entry


@dataclass
class SearchResult:
    """Outcome of a search strategy.

    ``depths`` is always a configuration believed feasible (at worst the seed), so that a
    report based on it is never empty or infeasible. ``optimal_proven`` is relative to the
    feasibility predicate and the lattice box the search was run on. ``lower_bound_cost`` is a
    proven lower bound on the cost of any feasible configuration in that box.
    """

    strategy: str
    depths: Tuple[int, ...]
    cost: float
    result: Optional[EvalResult]
    optimal_proven: bool = False
    lower_bound_cost: Optional[float] = None
    lower_bounds: Optional[Tuple[int, ...]] = None
    lower_bounds_cost: Optional[float] = None
    num_evaluations: int = 0
    duration_s: float = 0.0
    open_nodes: int = 0
    budget_exhausted: bool = False
    seed_strategy: Optional[str] = None
    seed_depths: Optional[Tuple[int, ...]] = None
    seed_cost: Optional[float] = None
    abort_reason: Optional[str] = None
    incumbent_trace: List[Tuple[int, float]] = field(default_factory=list)
    pareto: List[Dict] = field(default_factory=list)
    extra: Dict = field(default_factory=dict)


class SearchContext:
    """Everything a search strategy needs, shared between strategies run back to back.

    ``start_depths`` is the configuration the targets were measured at (the paced-mode peaks
    for the live driver), which is feasible by definition and serves as the default seed.
    """

    def __init__(
        self,
        oracle: Oracle,
        feasibility: FeasibilityCheck,
        cost_model: CostModel,
        start_depths: Sequence[int],
        widths: Sequence[int],
        cache: Optional[EvaluationCache] = None,
        lattice: Optional[Lattice] = None,
        budget: Optional[SearchBudget] = None,
        log: Callable[[str], None] = print,
        rng=random,
        reuse_binary_evaluations: bool = False,
    ):
        """Bundle oracle, predicate, cost model, lattice, cache and budget for the searches."""
        self.oracle = oracle
        self.feasibility = feasibility
        self.cost_model = cost_model
        self.start_depths = tuple(int(d) for d in start_depths)
        self.widths = list(widths)
        self.cache = cache if cache is not None else EvaluationCache(oracle, cost_model)
        self.lattice = (
            lattice
            if lattice is not None
            else Lattice([1] * len(self.start_depths), self.start_depths)
        )
        self.budget = budget if budget is not None else SearchBudget()
        self.log = log
        self.rng = rng
        # The binary strategies re-measure every configuration like the original driver did,
        # so that their evaluation sequence (and jitter exposure) is unchanged.
        self.reuse_binary_evaluations = reuse_binary_evaluations

    @property
    def num_fifos(self) -> int:
        """Number of FIFOs."""
        return len(self.start_depths)

    def feasible(self, entry: LogEntry) -> bool:
        """Shortcut for the feasibility predicate on a log entry."""
        return self.feasibility.feasible(entry.result)

    def cost(self, depths: Sequence[int]) -> float:
        """Shortcut for the cost model."""
        return self.cost_model.cost(depths)

    def start_entry(self) -> Optional[LogEntry]:
        """Log entry of the starting configuration (the target measurement), if evaluated."""
        return self.cache.get(self.start_depths)

    def budget_exhausted(self) -> bool:
        """Return whether the evaluation/time budget is used up."""
        return self.budget.exhausted(self.cache.num_evaluations)


# ---------------------------------------------------------------------------------------
# Binary search (coordinate descent) in all FIFO orders of the original driver
# ---------------------------------------------------------------------------------------

BINARY_ORDERS = ("forward", "reverse", "largest_first", "deepest_first", "alternating", "random")


def binary_fifo_order(order: str, start_depths: Sequence[int], widths: Sequence[int], rng=random):
    """FIFO visiting order of the binary search strategies (identical to the original driver)."""
    n = len(start_depths)
    if order == "forward":
        return list(range(n))
    if order == "reverse":
        return list(range(n - 1, -1, -1))
    if order == "largest_first":
        return sorted(range(n), key=lambda i: -start_depths[i] * widths[i])
    if order == "deepest_first":
        return sorted(range(n), key=lambda i: -start_depths[i])
    if order == "alternating":
        fifo_order = []
        left, right = 0, n - 1
        while left <= right:
            fifo_order.append(left)
            if left != right:
                fifo_order.append(right)
            left += 1
            right -= 1
        return fifo_order
    if order == "random":
        fifo_order = list(range(n))
        rng.shuffle(fifo_order)
        return fifo_order
    raise ValueError(f"Unknown binary search order: {order}")


def search_binary(ctx: SearchContext, order: str = "largest_first", **_ignored) -> SearchResult:
    """Per-FIFO binary search for the minimum depth, visiting the FIFOs in ``order``.

    This is the original ``size_iteratively_binary_search`` of the live driver expressed on the
    oracle: starting from the configuration the targets were measured at, each FIFO in turn is
    reduced by binary search in ``[1, start]`` while all others keep their current depth.
    """
    t0 = time.time()
    fifo_depths = list(ctx.start_depths)
    fifo_order = binary_fifo_order(order, ctx.start_depths, ctx.widths, ctx.rng)
    fb = ctx.feasibility
    iteration = 0
    budget_exhausted = False

    for fifo_id in fifo_order:
        if budget_exhausted:
            break
        ctx.log(f"Binary searching for FIFO {fifo_id}...")
        low = 1
        high = fifo_depths[fifo_id]
        best_working_depth = high

        while low <= high:
            if ctx.budget_exhausted():
                budget_exhausted = True
                break
            mid = ctx.lattice.mid(fifo_id, low, high)
            test_depths = fifo_depths.copy()
            test_depths[fifo_id] = mid
            entry = ctx.cache.evaluate(
                test_depths,
                reuse=ctx.reuse_binary_evaluations,
                phase="binary",
                tested_fifo=fifo_id,
                tested_depth=mid,
            )
            if ctx.feasible(entry):
                best_working_depth = mid
                high = mid - 1
                status = "PASS"
            else:
                low = mid + 1
                status = "FAIL"
            iteration += 1
            entry.info.update(
                status=status,
                search_bounds=[low, high],
                best_working_depth=best_working_depth,
            )
            r = entry.result
            ctx.log(f"  Iteration {iteration}: Testing depth {mid} - {status}")
            ctx.log(f"    Binary search bounds: [{low}, {high}]")
            ctx.log(f"    Best working depth so far: {best_working_depth}")
            if fb.stop_condition in ("interval", "both"):
                ctx.log(
                    f"    Interval: {r.interval}, "
                    f"Threshold: {fb.relaxed_interval_threshold:.1f} "
                    f"(Target: {fb.target_interval})"
                )
            if fb.stop_condition in ("latency", "both"):
                ctx.log(
                    f"    Latency: {r.latency}, "
                    f"Threshold: {fb.relaxed_latency_threshold:.1f} "
                    f"(Target: {fb.target_latency})"
                )

        fifo_depths[fifo_id] = best_working_depth
        ctx.log(f"  FIFO {fifo_id} minimized to depth {best_working_depth}")
        ctx.log(f"  Total FIFO Size (kB): {ctx.cost(fifo_depths)}")

    final = tuple(fifo_depths)
    final_entry = ctx.cache.get(final)
    return SearchResult(
        strategy=f"binary_{order}",
        depths=final,
        cost=ctx.cost(final),
        result=final_entry.result if final_entry is not None else None,
        optimal_proven=False,
        num_evaluations=ctx.cache.num_evaluations,
        duration_s=time.time() - t0,
        budget_exhausted=budget_exhausted,
        seed_strategy=None,
        seed_depths=ctx.start_depths,
        seed_cost=ctx.cost(ctx.start_depths),
    )


# ---------------------------------------------------------------------------------------
# Shared helpers of the exact searches
# ---------------------------------------------------------------------------------------


def compute_lower_bounds(ctx: SearchContext) -> Tuple[int, ...]:
    """Per-FIFO lower bounds: the minimal depth of each FIFO with all others unbounded.

    By monotonicity every feasible configuration satisfies ``d >= lb`` component-wise, so the
    cost of ``lb`` is a global lower bound on the optimum. "Others unbounded" is deliberately not
    "others at their start depth": shrinking one FIFO can raise the peak occupancy of its
    neighbours above the start depth.

    If the budget runs out, the bounds of the unfinished FIFOs fall back to the smallest depth
    not yet proven infeasible, which keeps them valid (if loose) lower bounds.
    """
    lb = []
    exhausted = False
    for fifo_id in range(ctx.num_fifos):
        start = ctx.start_depths[fifo_id]
        low = ctx.lattice.lower[fifo_id]
        high = start
        best = start
        ctx.log(f"Lower bound for FIFO {fifo_id} (all others unbounded)...")
        while low <= high:
            if exhausted or ctx.budget_exhausted():
                exhausted = True
                best = low
                break
            mid = ctx.lattice.mid(fifo_id, low, high)
            probe = [UNBOUNDED_DEPTH] * ctx.num_fifos
            probe[fifo_id] = mid
            entry = ctx.cache.evaluate(
                probe, phase="lower_bound", tested_fifo=fifo_id, tested_depth=mid
            )
            if ctx.feasible(entry):
                best = mid
                high = mid - 1
                entry.info["status"] = "PASS"
            else:
                low = mid + 1
                entry.info["status"] = "FAIL"
            entry.info["search_bounds"] = [low, high]
            ctx.log(f"  depth {mid}: {entry.info['status']}")
        lb.append(best)
        ctx.log(f"  FIFO {fifo_id} lower bound: {best}{' (budget exhausted)' if exhausted else ''}")
    return tuple(lb)


def _upper_bounds(ctx: SearchContext, upper_bound_factor: float) -> Tuple[int, ...]:
    """Upper corner of the search box: ``factor * start`` per FIFO, at least the start depth."""
    return tuple(
        max(int(math.ceil(upper_bound_factor * d)), d, lo)
        for d, lo in zip(ctx.start_depths, ctx.lattice.lower)
    )


def _run_seed(ctx: SearchContext, seed_strategy: Optional[str], **kwargs) -> SearchResult:
    """Run the seed strategy (or take the start configuration) to obtain a feasible incumbent."""
    if seed_strategy is None:
        start_entry = ctx.start_entry()
        return SearchResult(
            strategy="start",
            depths=ctx.start_depths,
            cost=ctx.cost(ctx.start_depths),
            result=start_entry.result if start_entry is not None else None,
        )
    ctx.log(f"Seeding with strategy {seed_strategy}...")
    return run_search(seed_strategy, ctx, **kwargs)


def _lower_bound_entry_cost(ctx: SearchContext, lb: Tuple[int, ...]) -> float:
    """Cost of the lower-bound corner."""
    return ctx.cost(lb)


# ---------------------------------------------------------------------------------------
# bottom_up_exact: best-first search from the lower bounds upwards (Stuijk et al. 2008)
# ---------------------------------------------------------------------------------------


def search_bottom_up_exact(
    ctx: SearchContext,
    upper_bound_factor: float = 2.0,
    seed_strategy: Optional[str] = None,
    pareto: bool = False,
    pareto_cost_limit: Optional[float] = None,
    allow_unreliable_flags: bool = False,
    lower_bounds: Optional[Sequence[int]] = None,
    **_ignored,
) -> SearchResult:
    """Best-first search over depth configurations ordered by cost, from the lower bounds up.

    Every popped configuration is evaluated; if it is infeasible, only the FIFOs that blocked
    their producer (``producer_blocked``) are enlarged by one lattice step. Any FIFO that never
    blocked its producer could be made unbounded without changing a single event of the
    execution, so every feasible configuration above the popped one must enlarge a blocked
    FIFO: the expansion is complete and the first feasible configuration popped from the
    cost-ordered heap is a global optimum.

    ``producer_blocked`` must come from the run-sticky stall flags: the per-frame flags are
    stale in a deadlock, where most of the visited configurations end up. Without reliable
    flags the search refuses to run unless ``allow_unreliable_flags`` is set.

    With ``pareto`` the search does not stop at the first feasible point but keeps popping until
    the cost exceeds ``pareto_cost_limit`` (default: the cost of the seed) and records interval
    and latency of every evaluated configuration, i.e. the cost/performance trade-off curve.
    """
    t0 = time.time()
    seed = _run_seed(ctx, seed_strategy)
    seed_result_cost = seed.cost

    lb = tuple(lower_bounds) if lower_bounds is not None else compute_lower_bounds(ctx)
    ub = _upper_bounds(ctx, upper_bound_factor)
    lattice = Lattice(lb, ub, ctx.lattice.step)
    lb_cost = _lower_bound_entry_cost(ctx, lb)
    ctx.log(f"Lower bounds: {list(lb)} (cost {lb_cost:.3f} kB), box upper corner: {list(ub)}")

    heap: List[Tuple[float, Tuple[int, ...]]] = [(round(lb_cost, 9), lb)]
    visited = {lb}
    cost_model = ctx.cost_model
    cap_hit = False
    num_popped = 0
    abort_reason = None
    budget_exhausted = False
    found: Optional[LogEntry] = None
    pareto_points: List[Dict] = []
    if pareto and pareto_cost_limit is None:
        pareto_cost_limit = seed_result_cost
    incumbent_trace: List[Tuple[int, float]] = []

    while heap:
        if ctx.budget_exhausted():
            budget_exhausted = True
            break
        cost, depths = heapq.heappop(heap)
        if found is not None and pareto and cost > pareto_cost_limit:
            heapq.heappush(heap, (cost, depths))
            break
        num_popped += 1
        entry = ctx.cache.evaluate(depths, need_flags=True, phase="bottom_up", popped=num_popped)
        feasible = ctx.feasible(entry)
        entry.info["status"] = "PASS" if feasible else "FAIL"
        r = entry.result
        ctx.log(
            f"  Pop {num_popped}: cost {cost:.3f} kB depths {list(depths)} -> "
            f"{entry.info['status']} (interval {r.interval}, latency {r.latency}, "
            f"deadlock {r.deadlock}, open {len(heap)})"
        )
        if pareto:
            pareto_points.append(
                {
                    "depths": list(depths),
                    "cost": cost,
                    "interval": r.interval,
                    "latency": r.latency,
                    "min_latency": r.min_latency,
                    "deadlock": r.deadlock,
                    "feasible": feasible,
                }
            )
        if feasible:
            if found is None:
                found = entry
                incumbent_trace.append((entry.index, cost))
                if not pareto:
                    break
            continue

        blocked = r.producer_blocked
        if blocked is None or (r.deadlock and not r.flags_reliable):
            if not allow_unreliable_flags:
                abort_reason = (
                    "the oracle reports no reliable producer-blocked flags for "
                    f"{list(depths)}; refusing to expand (set allow_unreliable_flags "
                    "to expand every FIFO instead)"
                )
                break
            blocked = tuple([True] * ctx.num_fifos)
        dependency_set = [e for e, b in enumerate(blocked) if b]
        if not dependency_set:
            # An infeasible configuration without any blocked producer executes identically
            # to the unbounded one, which meets the target: the measurement is noisy.
            # Re-measure once before giving up.
            retry = ctx.cache.evaluate(
                depths, reuse=False, need_flags=True, phase="bottom_up", popped=num_popped
            )
            retry_feasible = ctx.feasible(retry)
            retry.info["status"] = "PASS" if retry_feasible else "FAIL"
            retry.info["retry"] = True
            if retry_feasible:
                if pareto:
                    pareto_points[-1].update(feasible=True, retried=True)
                if found is None:
                    found = retry
                    incumbent_trace.append((retry.index, cost))
                    if not pareto:
                        break
                continue
            blocked = retry.result.producer_blocked or ()
            dependency_set = [e for e, b in enumerate(blocked) if b]
            if not dependency_set:
                abort_reason = (
                    f"infeasible configuration {list(depths)} reports no blocked producer "
                    "twice: the stall flags are unreliable"
                )
                break

        # A child differs from its parent in one FIFO, so its cost is the parent's cost plus
        # the cost difference of that FIFO; recomputing it over all FIFOs made every pop
        # O(num_fifos * |dependency_set|) in Python, comparable to the hardware time on
        # large DUTs.
        for e in dependency_set:
            cur = depths[e]
            nxt = lattice.next_up(e, cur)
            if nxt is None:
                cap_hit = True
                continue
            child_t = depths[:e] + (nxt,) + depths[e + 1 :]
            if child_t not in visited:
                visited.add(child_t)
                # rounded well below the cost of a single bit so that accumulated rounding
                # cannot reorder configurations of equal cost relative to a direct computation
                child_cost = round(
                    cost + cost_model.cost_fifo(e, nxt) - cost_model.cost_fifo(e, cur), 9
                )
                heapq.heappush(heap, (child_cost, child_t))

    heap_min = heap[0][0] if heap else None
    if found is not None:
        depths = found.depths
        result = SearchResult(
            strategy="bottom_up_exact",
            depths=depths,
            cost=ctx.cost(depths),
            result=found.result,
            optimal_proven=True,
            lower_bound_cost=ctx.cost(depths),
        )
    else:
        # No feasible configuration found: return the seed. Nothing cheaper than the heap
        # minimum is feasible, which is the proven lower bound of this partial search.
        result = SearchResult(
            strategy="bottom_up_exact",
            depths=seed.depths,
            cost=seed.cost,
            result=seed.result,
            optimal_proven=False,
            lower_bound_cost=heap_min if heap_min is not None else lb_cost,
        )
    result.lower_bounds = lb
    result.lower_bounds_cost = lb_cost
    result.num_evaluations = ctx.cache.num_evaluations
    result.duration_s = time.time() - t0
    result.open_nodes = len(heap)
    result.budget_exhausted = budget_exhausted
    result.seed_strategy = seed_strategy
    result.seed_depths = seed.depths
    result.seed_cost = seed.cost
    result.abort_reason = abort_reason
    result.incumbent_trace = incumbent_trace
    result.pareto = pareto_points
    result.extra = {
        "num_popped": num_popped,
        "box_upper": list(ub),
        "box_cap_hit": cap_hit,
        "num_cache_hits": ctx.cache.num_cache_hits,
    }
    return result


# ---------------------------------------------------------------------------------------
# top_down_bnb: branch-and-bound over boxes of depth vectors
# ---------------------------------------------------------------------------------------


def _branch_fifo(ctx: SearchContext, lo: Tuple[int, ...], hi: Tuple[int, ...]) -> int:
    """FIFO to branch on: the one with the largest cost range within the box."""
    best, best_range = -1, -1.0
    for e, (lo_e, hi_e) in enumerate(zip(lo, hi)):
        if hi_e > lo_e:
            rng = ctx.cost_model.cost_fifo(e, hi_e) - ctx.cost_model.cost_fifo(e, lo_e)
            if rng > best_range:
                best, best_range = e, rng
    return best


def search_top_down_bnb(
    ctx: SearchContext,
    upper_bound_factor: float = 2.0,
    seed_strategy: Optional[str] = "binary_largest_first",
    node_order: str = "dfs",
    lower_bounds: Optional[Sequence[int]] = None,
    **_ignored,
) -> SearchResult:
    """Branch-and-bound over boxes ``[lo, hi]`` of depth vectors.

    A box is pruned when its lower corner is not cheaper than the incumbent, or when its upper
    corner is infeasible (by monotonicity no point in the box is feasible then). Otherwise the
    upper corner is a feasible candidate, and the box is split along the FIFO with the largest
    cost range: a "shrink" child (``hi_e = m``) and a "keep large" child (``lo_e = m + 1``).
    With only the shrink children in a fixed order, this is the per-FIFO binary search; the
    keep-large children plus the bound pruning make it complete.

    The peak occupancies of a feasible upper corner give a cheaper configuration with the exact
    same execution (``min(hi, max(lo, peak))``), which improves the incumbent early. It does
    not shrink the box itself: other points of the box with smaller depths elsewhere can need
    more than the observed peak of a FIFO, so the box must be kept for completeness.

    The search is exact relative to the box ``[lb, ub]`` with ``ub = upper_bound_factor *
    start``; on budget exhaustion the incumbent and the proven lower bound are reported.
    """
    t0 = time.time()
    seed = _run_seed(ctx, seed_strategy)
    seed_num_evaluations = ctx.cache.num_evaluations

    lb = tuple(lower_bounds) if lower_bounds is not None else compute_lower_bounds(ctx)
    ub = _upper_bounds(ctx, upper_bound_factor)
    lattice = Lattice(lb, ub, ctx.lattice.step)
    lb_cost = _lower_bound_entry_cost(ctx, lb)
    ctx.log(f"Lower bounds: {list(lb)} (cost {lb_cost:.3f} kB), box upper corner: {list(ub)}")

    # Incumbent: the seed. ``inferred`` marks an incumbent derived from peaks that has not been
    # evaluated directly; it is confirmed at the end.
    incumbent = tuple(seed.depths)
    incumbent_cost = seed.cost
    incumbent_result = seed.result
    incumbent_inferred = False
    best_evaluated = (incumbent, incumbent_cost, incumbent_result)
    incumbent_trace: List[Tuple[int, float]] = [(ctx.cache.num_evaluations, incumbent_cost)]

    def improve(depths, cost, result, inferred):
        """Update the incumbent (and the best directly evaluated point) with a feasible point."""
        nonlocal incumbent, incumbent_cost, incumbent_result, incumbent_inferred, best_evaluated
        if cost < incumbent_cost:
            incumbent, incumbent_cost, incumbent_result = depths, cost, result
            incumbent_inferred = inferred
            incumbent_trace.append((ctx.cache.num_evaluations, cost))
            ctx.log(f"  New incumbent: cost {cost:.3f} kB depths {list(depths)}")
        if not inferred and cost < best_evaluated[1]:
            best_evaluated = (depths, cost, result)

    # Node = (lo, hi); dfs uses a stack, best_bound a heap keyed by the bound cost(lo).
    root = (lb, ub)
    if node_order == "dfs":
        nodes: list = [root]
        push = nodes.append
        pop = nodes.pop
    elif node_order == "best_bound":
        nodes = []
        counter = [0]

        def push(node):
            """Queue a node keyed by its bound."""
            counter[0] += 1
            heapq.heappush(nodes, (ctx.cost(node[0]), counter[0], node))

        def pop():
            """Take the node with the smallest bound."""
            return heapq.heappop(nodes)[2]

        push(root)
    else:
        raise ValueError(f"Unknown node_order: {node_order}")

    num_nodes = 0
    num_pruned_bound = 0
    num_pruned_infeasible = 0
    budget_exhausted = False
    abort_reason = None

    def open_bound():
        """Smallest bound among the open nodes (None when there are none)."""
        if node_order == "dfs":
            return min((ctx.cost(n[0]) for n in nodes), default=None)
        return nodes[0][0] if nodes else None

    while nodes:
        if ctx.budget_exhausted():
            budget_exhausted = True
            break
        lo, hi = pop()
        num_nodes += 1
        if ctx.cost(lo) >= incumbent_cost:
            num_pruned_bound += 1
            continue
        entry = ctx.cache.evaluate(hi, need_peaks=True, phase="bnb", node=num_nodes)
        feasible = ctx.feasible(entry)
        if not feasible and num_nodes == 1:
            # The root corner dominates the feasible start configuration; a failure here is
            # measurement noise. Re-measure once, then give up rather than "prove" anything.
            entry = ctx.cache.evaluate(hi, reuse=False, need_peaks=True, phase="bnb", node=1)
            entry.info["retry"] = True
            feasible = ctx.feasible(entry)
            if not feasible:
                abort_reason = f"the upper corner {list(hi)} of the search box measured infeasible"
                entry.info["status"] = "FAIL"
                break
        entry.info["status"] = "PASS" if feasible else "FAIL"
        entry.info["box"] = [list(lo), list(hi)]
        if not feasible:
            num_pruned_infeasible += 1
            continue
        improve(hi, entry.cost, entry.result, inferred=False)
        peak = entry.result.peak
        if peak is not None:
            tight = tuple(min(h, max(lo_e, int(p))) for lo_e, h, p in zip(lo, hi, peak))
            if tight != hi:
                cached = ctx.cache.get(tight)
                if cached is not None:
                    if ctx.feasible(cached):
                        improve(tight, cached.cost, cached.result, inferred=False)
                else:
                    improve(tight, ctx.cost(tight), entry.result, inferred=True)
        if lo == hi:
            continue
        e = _branch_fifo(ctx, lo, hi)
        m = lattice.mid(e, lo[e], hi[e])
        keep_large = (lo[:e] + (m + 1,) + lo[e + 1 :], hi)
        shrink = (lo, hi[:e] + (m,) + hi[e + 1 :])
        # dfs explores the shrink child first (pushed last)
        push(keep_large)
        push(shrink)

    # Confirm an incumbent that was only inferred from peak occupancies.
    confirmed = True
    if incumbent_inferred and abort_reason is None:
        entry = ctx.cache.evaluate(incumbent, need_peaks=True, phase="bnb_confirm")
        if ctx.feasible(entry):
            entry.info["status"] = "PASS"
            incumbent_result = entry.result
        else:
            entry.info["status"] = "FAIL"
            confirmed = False
            ctx.log(
                f"  Inferred incumbent {list(incumbent)} failed its confirmation; "
                "falling back to the best directly evaluated configuration."
            )
            incumbent, incumbent_cost, incumbent_result = best_evaluated

    remaining = open_bound()
    if abort_reason is not None:
        optimal_proven = False
        lower_bound_cost = lb_cost
    elif nodes:
        optimal_proven = False
        lower_bound_cost = min(incumbent_cost, remaining) if remaining is not None else lb_cost
    else:
        optimal_proven = confirmed
        lower_bound_cost = incumbent_cost if confirmed else lb_cost

    return SearchResult(
        strategy="top_down_bnb",
        depths=tuple(incumbent),
        cost=incumbent_cost,
        result=incumbent_result,
        optimal_proven=optimal_proven,
        lower_bound_cost=lower_bound_cost,
        lower_bounds=lb,
        lower_bounds_cost=lb_cost,
        num_evaluations=ctx.cache.num_evaluations,
        duration_s=time.time() - t0,
        open_nodes=len(nodes),
        budget_exhausted=budget_exhausted,
        seed_strategy=seed_strategy,
        seed_depths=tuple(seed.depths),
        seed_cost=seed.cost,
        abort_reason=abort_reason,
        incumbent_trace=incumbent_trace,
        extra={
            "node_order": node_order,
            "num_nodes": num_nodes,
            "num_pruned_bound": num_pruned_bound,
            "num_pruned_infeasible": num_pruned_infeasible,
            "box_upper": list(ub),
            "seed_num_evaluations": seed_num_evaluations,
            "incumbent_confirmed": confirmed,
            "num_cache_hits": ctx.cache.num_cache_hits,
        },
    )


# ---------------------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------------------


def _binary_strategy(order):
    """Bind a FIFO order to ``search_binary``."""

    def strategy(ctx, **kwargs):
        """Binary search in a fixed FIFO order."""
        return search_binary(ctx, order=order, **kwargs)

    strategy.__name__ = f"binary_{order}"
    strategy.__doc__ = f"Per-FIFO binary search in {order} order."
    return strategy


SEARCH_STRATEGIES: Dict[str, Callable[..., SearchResult]] = {
    **{f"binary_{order}": _binary_strategy(order) for order in BINARY_ORDERS},
    "bottom_up_exact": search_bottom_up_exact,
    "top_down_bnb": search_top_down_bnb,
}


def run_search(strategy: str, ctx: SearchContext, **kwargs) -> SearchResult:
    """Run a strategy from the registry on a search context."""
    if strategy not in SEARCH_STRATEGIES:
        raise ValueError(
            f"Unknown search_strategy: {strategy} (available: {sorted(SEARCH_STRATEGIES)})"
        )
    return SEARCH_STRATEGIES[strategy](ctx, **kwargs)


def resolve_strategy_name(
    search_strategy=None, fifo_search_order=None, default="binary_largest_first"
):
    """Map the kwargs ``search_strategy`` / deprecated ``fifo_search_order`` to a strategy name."""
    if search_strategy is not None:
        return search_strategy
    if fifo_search_order is not None:
        return f"binary_{fifo_search_order}"
    return default


def _finite(x):
    """JSON-safe float: ``None`` for infinite or missing values."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return x


def cache_to_iterations(cache: EvaluationCache) -> Dict[str, Dict]:
    """Export the search log in the ``all_iterations`` format of the live driver report.

    Entry ``"0"`` is the first evaluation (the target measurement at the start configuration);
    the others follow in evaluation order with the bookkeeping their strategy attached.
    """
    iterations = {}
    for entry in cache.log:
        r = entry.result
        record = {
            **entry.info,
            "interval": r.interval,
            "min_latency": r.min_latency,
            "latency": r.latency,
            "deadlock": r.deadlock,
            "total_fifo_size_kB": _finite(entry.cost),
            "fifo_depths": list(entry.depths),
        }
        if r.producer_blocked is not None:
            record["producer_blocked"] = list(r.producer_blocked)
        if r.stall_flags is not None:
            record["stall_flags"] = list(r.stall_flags)
        if r.peak is not None:
            record["peak"] = list(r.peak)
        iterations[str(entry.index)] = record
    return iterations


def search_result_to_dict(result: SearchResult) -> Dict:
    """JSON-serialisable summary of a search result (without the log)."""
    return {
        "search_strategy": result.strategy,
        "fifo_depths": list(result.depths),
        "total_kB": _finite(result.cost),
        "optimal_proven": result.optimal_proven,
        "lower_bound_total_kB": _finite(result.lower_bound_cost),
        "optimality_gap_kB": _finite(
            result.cost - result.lower_bound_cost if result.lower_bound_cost is not None else None
        ),
        "lower_bounds": list(result.lower_bounds) if result.lower_bounds is not None else None,
        "lower_bounds_total_kB": _finite(result.lower_bounds_cost),
        "num_evaluations": result.num_evaluations,
        "duration_s": result.duration_s,
        "open_nodes": result.open_nodes,
        "budget_exhausted": result.budget_exhausted,
        "seed_strategy": result.seed_strategy,
        "seed_fifo_depths": list(result.seed_depths) if result.seed_depths is not None else None,
        "seed_total_kB": _finite(result.seed_cost),
        "abort_reason": result.abort_reason,
        "incumbent_trace": [[i, _finite(c)] for i, c in result.incumbent_trace],
        "pareto": result.pareto,
        **result.extra,
    }

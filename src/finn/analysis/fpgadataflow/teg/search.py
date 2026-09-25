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

"""Per-FIFO monotone depth minimisation on the abstract simulator (SimFIFO's algorithm).

Phases (LiveFIFO Section 4.4 / SimFIFO Section IV):

1. Measure the bottleneck interval ``I*`` with unbounded external FIFOs and free-running
   sources.
2. Pace the sources at ``I*`` and simulate again with unbounded FIFOs; the maximum occupancies
   are a safe (over-)estimate of the depths (formal model notes, Section 5.2) and become the
   starting point, exactly as ``RunLayerParallelSimulation.create_starting_fifo_depths`` does.
3. Minimise every external FIFO in turn with the block-granular search of
   ``finn.transformation.fpgadataflow.fifo_depth_search``; the oracle for a candidate depth is
   one abstract simulation whose interval must not exceed ``I*``.
4. Validate the joint result with one more simulation.

The search in phase 3 is identical to the distributed simulation's; only the oracle differs.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from finn.analysis.fpgadataflow.teg.model import TEGModel
from finn.analysis.fpgadataflow.teg.simulate import Backend, SimResult, simulate
from finn.transformation.fpgadataflow.fifo_depth_search import (
    MinimizationOrder,
    effective_capacity,
    minimize_fifo_depth,
    needs_minimization,
    round_up_to_full_bram_block,
    safe_bram_starting_depth,
)
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass
class SearchResult:
    """Result of :func:`minimize_depths`."""

    #: nominal depths (as configured on the FIFO nodes) per external edge
    depths: dict[str, int]
    #: bottleneck interval measured with unbounded FIFOs
    target_interval: float
    #: interval of the validation run with the final depths
    final_interval: float
    #: maximum occupancy per external edge with unbounded FIFOs and paced sources
    peak_occupancy: dict[str, int]
    #: cycle of the first write per external edge in the paced run
    first_valid: dict[str, int]
    #: number of simulations per edge
    iterations: dict[str, int] = field(default_factory=dict)
    #: wall-clock seconds spent per edge
    times: dict[str, float] = field(default_factory=dict)
    #: total number of simulations
    simulations: int = 0
    #: total wall-clock seconds of the search
    total_time: float = 0.0
    #: unbounded/paced run for reporting
    baseline: SimResult | None = None


def capacities_for(depths: dict[str, int], max_qsrl_depth: int) -> dict[str, int | None]:
    """Translate nominal depths of the external edges into effective token capacities."""
    return {e: effective_capacity(d, max_qsrl_depth) for e, d in depths.items()}


def order_edges(model: TEGModel, order: MinimizationOrder, widths: dict[str, int]) -> list[str]:
    """Order the external edges for the per-FIFO search.

    ``NODE_ORDER`` is the insertion order of the external edges (producer order of the ONNX
    graph), ``REVERSE_NODE_ORDER`` its reverse. The bitwidth-difference orders sort the edges
    by the width difference between the producing chain's inputs and outputs, which for a
    single edge per producer reduces to sorting by the edge width (largest / smallest first).
    """
    ext = model.external_edges
    match order:
        case MinimizationOrder.NODE_ORDER:
            return list(ext)
        case MinimizationOrder.REVERSE_NODE_ORDER:
            return list(reversed(ext))
        case MinimizationOrder.LARGEST_BITWIDTH_DIFF_FIRST:
            return sorted(ext, key=lambda e: -widths[e])
        case MinimizationOrder.SMALLEST_BITWIDTH_DIFF_FIRST:
            return sorted(ext, key=lambda e: widths[e])
        case _:
            raise NotImplementedError(f"Minimization order {order} not supported")


def measure_bottleneck(
    model: TEGModel,
    max_frames: int = 64,
    max_cycles: int | None = None,
    backend: Backend | None = None,
) -> SimResult:
    """Phase 1: interval with unbounded external FIFOs and free-running sources."""
    depths: dict[str, int | None] = dict.fromkeys(model.external_edges, None)
    res = simulate(
        model,
        depths,
        max_frames=max_frames,
        max_cycles=max_cycles,
        stable_occupancy=False,
        backend=backend,
    )
    if res.deadlock:
        raise FINNUserError(
            "The abstract model deadlocks even with unbounded FIFOs; this points at a modelling "
            "error in an operator template (a cycle without tokens)."
        )
    if not res.stable:
        log.warning(
            f"Bottleneck measurement did not reach a stable state within {res.frames} frames; "
            f"using the mean interval {res.interval:.1f}"
        )
    return res


def measure_peak_occupancy(
    model: TEGModel,
    interval: int,
    max_frames: int = 64,
    max_cycles: int | None = None,
    backend: Backend | None = None,
) -> SimResult:
    """Phase 2: sources paced at ``interval``, unbounded FIFOs; occupancy is the safe bound."""
    paced = TEGModel(chains=dict(model.chains), edges=model.edges, meta=model.meta)
    for name in model.sources:
        c = model.chains[name]
        c2 = type(c)(
            name,
            kind=c.kind,
            pattern=c.pattern,
            period=interval,
            history_window=c.history_window,
            freeze_group=c.freeze_group,
            freezer=c.freezer,
            start=c.start,
            freeze_until_empty=c.freeze_until_empty,
        )
        c2.gaps, c2.reads, c2.writes, c2.arcs = c.gaps, c.reads, c.writes, c.arcs
        paced.chains[name] = c2
    depths: dict[str, int | None] = dict.fromkeys(model.external_edges, None)
    res = simulate(paced, depths, max_frames=max_frames, max_cycles=max_cycles, backend=backend)
    if res.deadlock:
        raise FINNInternalError("Paced simulation with unbounded FIFOs deadlocked")
    if not res.stable:
        log.warning(
            f"Peak-occupancy measurement did not settle within {res.frames} frames; the "
            "starting depths may be too small"
        )
    return res


def minimize_depths(
    model: TEGModel,
    widths: dict[str, int],
    *,
    minimization_orders: Sequence[MinimizationOrder] = (MinimizationOrder.NODE_ORDER,),
    max_qsrl_depth: int = 256,
    max_frames: int = 64,
    target_interval: float | None = None,
    starting_depths: dict[str, int] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    backend: Backend | None = None,
) -> SearchResult:
    """Run the full abstract-simulation FIFO sizing and return nominal depths.

    Args:
        model: validated TEG model whose external edges are to be sized.
        widths: token width in bits per external edge (cost model and BRAM rounding).
        minimization_orders: search orders to try; the cheapest result (total bits) wins.
        max_qsrl_depth: largest SRL (LUTRAM) FIFO depth.
        max_frames: frame cap for every simulation.
        target_interval: override the measured bottleneck interval (e.g. a relaxed target).
        starting_depths: override the safe starting depths.
        progress: optional callback ``(edge, done, total)``.
        backend: simulator backend (``simulate.default_backend()`` when None).
    """
    ext = model.external_edges
    if not ext:
        raise FINNUserError("Model has no external FIFO edges to size")
    t_start = time.time()
    simulations = 0

    # ---- phase 1/2
    base = measure_bottleneck(model, max_frames=max_frames, backend=backend)
    simulations += 1
    interval = float(target_interval) if target_interval is not None else base.interval
    if not math.isfinite(interval):
        raise FINNUserError("Could not determine a finite bottleneck interval")
    target = math.ceil(interval)
    paced = measure_peak_occupancy(model, target, max_frames=max_frames, backend=backend)
    simulations += 1
    peak = {e: paced.max_occupancy[e] for e in ext}
    first_valid = {e: paced.first_valid[e] for e in ext}
    if starting_depths is None:
        starting_depths = {e: safe_bram_starting_depth(peak[e], max_qsrl_depth) for e in ext}
    log.info(
        f"abstract_sim: bottleneck interval {target} cycles, "
        f"starting depths {sum(starting_depths.values())} tokens over {len(ext)} FIFOs"
    )
    # cycle budget per oracle run: like SimFIFO, a little more than the baseline run
    sim_cycles = max(base.cycles, paced.cycles)
    max_cycles = math.ceil(sim_cycles * 1.05) + 10 * len(ext)

    def make_oracle(current: dict[str, int], edge: str) -> Callable[[int], tuple[bool, bool]]:
        def test_depth(depth: int) -> tuple[bool, bool]:
            nonlocal simulations
            trial = dict(current)
            trial[edge] = depth
            # interval stability only: with bounded FIFOs and free-running sources the
            # occupancies keep growing until the FIFOs are full, which would defer the
            # stable-state stop and turn the cycle budget into a false failure
            res = simulate(
                model,
                capacities_for(trial, max_qsrl_depth),
                max_frames=max_frames,
                max_cycles=max_cycles,
                stable_occupancy=False,
                backend=backend,
            )
            simulations += 1
            if res.timeout or res.deadlock:
                return False, res.timeout
            return res.interval <= target, False

        return test_depth

    # ---- phase 3
    results: dict[MinimizationOrder, tuple[dict[str, int], dict[str, int], dict[str, float]]] = {}
    for order in minimization_orders:
        current = dict(starting_depths)
        iterations: dict[str, int] = {}
        times: dict[str, float] = {}
        edge_order = order_edges(model, order, widths)
        log.info(f"abstract_sim: minimising in order {order.name}")
        for n, edge in enumerate(edge_order):
            if not needs_minimization(current[edge], widths[edge], max_qsrl_depth):
                iterations[edge] = 0
                times[edge] = 0.0
                if progress is not None:
                    progress(edge, n + 1, len(edge_order))
                continue
            t0 = time.time()
            depth, its = minimize_fifo_depth(
                current[edge], widths[edge], make_oracle(current, edge), max_qsrl_depth
            )
            current[edge] = depth
            iterations[edge] = its
            times[edge] = time.time() - t0
            log.debug(f"abstract_sim: {edge} -> depth {depth} after {its} simulations")
            if progress is not None:
                progress(edge, n + 1, len(edge_order))
        for edge in ext:
            current[edge] = round_up_to_full_bram_block(current[edge], widths[edge], max_qsrl_depth)
        results[order] = (current, iterations, times)

    # ---- pick the cheapest order (total bits, like RunLayerParallelSimulation)
    best_order = min(results, key=lambda o: sum(results[o][0][e] * widths[e] for e in ext))
    depths, iterations, times = results[best_order]

    # ---- phase 4: validation (a generous cycle budget: this run only has to confirm the
    # interval, it is not an oracle whose timeout means "too slow")
    final = simulate(
        model,
        capacities_for(depths, max_qsrl_depth),
        max_frames=max_frames,
        max_cycles=4 * max_cycles + max_frames * target,
        stable_occupancy=False,
        backend=backend,
    )
    simulations += 1
    if final.deadlock or final.timeout or final.interval > target:
        raise FINNUserError(
            "Final validation with the jointly-minimised FIFO depths failed "
            f"(interval {final.interval} vs target {target}, deadlock={final.deadlock}, "
            f"timeout={final.timeout}, frames={final.frames})."
        )
    return SearchResult(
        depths=depths,
        target_interval=interval,
        final_interval=final.interval,
        peak_occupancy=peak,
        first_valid=first_valid,
        iterations=iterations,
        times=times,
        simulations=simulations,
        total_time=time.time() - t_start,
        baseline=paced,
    )

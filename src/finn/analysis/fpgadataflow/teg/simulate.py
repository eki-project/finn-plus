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

"""Event-driven self-timed execution of a :class:`TEGModel`.

The simulator computes the least fixed point of the model's constraints (formal model notes,
Definition 4): every chain fires its next event at the earliest cycle at which its gap, its
data constraints (tokens readable), its space constraints (slots writable), its arcs and its
environment pattern are satisfied.

Implementation: a chain is either scheduled in a time-ordered heap (it knows the earliest cycle
at which it may fire) or registered as the *waiter* of exactly one blocking edge or chain (it
is re-scheduled when that blocker fires, at ``fire time + latency``). All heap entries of the
current cycle are processed before time advances, which yields the per-cycle fixed point that
zero-delay arcs require. Work is therefore proportional to the number of events, not to the
number of cycles.

Occupancies are measured at the end of every cycle (after the cycle's writes and reads), as
the FIFO's count register and SimFIFO's ``fifo_utilization`` do.

Per-edge memory is bounded by the occupancy plus the depth (a deque of unread write times and
a ring of the last ``depth`` read times); per-chain event histories are kept only for chains
that are referenced by arcs, truncated to ``Chain.history_window`` when set.

The stable-state criterion: the simulation stops once the last ``STABLE_INTERVALS`` sink
intervals are identical and no maximum occupancy has changed for ``STABLE_OCC_FRAMES`` frames
(the execution is deterministic and eventually periodic), or after ``max_frames`` frames with
the mean of the second half of the intervals as a fallback estimate.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import pairwise
from typing import TYPE_CHECKING

from finn.util.exception import FINNInternalError

if TYPE_CHECKING:
    from finn.analysis.fpgadataflow.teg.model import TEGModel

_NEG_INF = -(1 << 60)
STABLE_INTERVALS = 3
STABLE_OCC_FRAMES = 2


@dataclass
class SimResult:
    """Outcome of one simulation run."""

    #: steady-state frame interval at the sinks in cycles; ``inf`` on deadlock or timeout
    interval: float
    #: steady-state latency (first source event to last sink event of a frame), or None
    latency: int | None
    #: frames completed at all sinks
    frames: int
    #: last simulated cycle
    cycles: int
    #: True when the stable-state criterion triggered (interval is exact)
    stable: bool
    #: True when no chain could fire anymore before the frames were done
    deadlock: bool
    #: True when ``max_cycles`` was exceeded
    timeout: bool
    #: maximum occupancy per edge (all edges, external and internal)
    max_occupancy: dict[str, int] = field(default_factory=dict)
    #: cycle of the first write per edge (SimFIFO's ``fifo_cycles_until_first_valid``)
    first_valid: dict[str, int] = field(default_factory=dict)
    #: completion cycle of every frame at the sinks
    frame_end_times: list[int] = field(default_factory=list)
    #: start cycle of every frame at the sources
    frame_start_times: list[int] = field(default_factory=list)
    #: per-chain event times when ``record_events`` was requested
    event_times: dict[str, list[int]] | None = None
    #: per-edge ``(write cycles, read cycles)`` for the edges in ``record_edges``
    handshakes: dict[str, tuple[list[int], list[int]]] | None = None

    @property
    def intervals(self) -> list[int]:
        """Frame-to-frame intervals at the sinks."""
        fe = self.frame_end_times
        return [b - a for a, b in pairwise(fe)]

    @property
    def ok(self) -> bool:
        """True when the run neither deadlocked nor timed out."""
        return not (self.deadlock or self.timeout)


class _Simulator:
    """Compiled, mutable simulation state for one model and one depth assignment."""

    def __init__(
        self,
        model: TEGModel,
        depths: dict[str, int | None] | None,
        record_events: bool,
        record_edges: set[str] | None,
    ) -> None:
        """Compile ``model`` into flat per-chain / per-edge lists."""
        self.model = model
        chain_names = list(model.chains)
        edge_names = list(model.edges)
        cid = {n: i for i, n in enumerate(chain_names)}
        eid = {n: i for i, n in enumerate(edge_names)}
        self.chain_names = chain_names
        self.edge_names = edge_names
        nc = len(chain_names)
        ne = len(edge_names)

        # ---- chains
        self.L = [model.chains[n].num_events for n in chain_names]
        self.gaps = [model.chains[n].gaps for n in chain_names]
        self.rd = [[tuple(eid[e] for e in r) for r in model.chains[n].reads] for n in chain_names]
        self.wr = [[tuple(eid[e] for e in w) for w in model.chains[n].writes] for n in chain_names]
        self.arcs: list[list[tuple[tuple[int, int, int, int], ...]]] = []
        referenced: set[int] = set()
        for n in chain_names:
            c = model.chains[n]
            per_event: list[tuple[tuple[int, int, int, int], ...]] = [()] * c.num_events
            for i, arcs in c.arcs.items():
                per_event[i] = tuple((cid[a.src], a.src_event, a.lag, a.delay) for a in arcs)
                for a in arcs:
                    referenced.add(cid[a.src])
            self.arcs.append(per_event)
        self.kind = [{"op": 0, "source": 1, "sink": 2}[model.chains[n].kind] for n in chain_names]
        self.pattern = [model.chains[n].pattern for n in chain_names]
        self.period = [model.chains[n].period for n in chain_names]
        self.keep_hist = [i in referenced for i in range(nc)]
        self.hist_window = [model.chains[n].history_window for n in chain_names]
        self.sources = [i for i in range(nc) if self.kind[i] == 1]
        self.sinks = [i for i in range(nc) if self.kind[i] == 2]
        groups = sorted({c.freeze_group for c in model.chains.values() if c.freeze_group})
        gid = {g: i for i, g in enumerate(groups)}
        self.group = [
            gid[model.chains[n].freeze_group] if model.chains[n].freeze_group else -1
            for n in chain_names
        ]
        self.freezer = [model.chains[n].freezer for n in chain_names]
        self.until_empty = [model.chains[n].freeze_until_empty for n in chain_names]
        self.ngroups = len(groups)
        self.start = [model.chains[n].start for n in chain_names]
        if not self.sinks:
            raise FINNInternalError("Model has no sink chain")

        # ---- edges
        self.src = [cid[model.edges[n].src] for n in edge_names]
        self.dst = [cid[model.edges[n].dst] for n in edge_names]
        self.lf = [model.edges[n].lf for n in edge_names]
        self.lb = [model.edges[n].lb for n in edge_names]
        self.depth: list[int | None] = []
        for n in edge_names:
            d = model.edges[n].depth
            if depths is not None and n in depths:
                d = depths[n]
            if d is not None and d < 1:
                raise FINNInternalError(f"Edge {n}: depth must be >= 1, got {d}")
            self.depth.append(d)
        self.init_tokens = [model.edges[n].initial_tokens for n in edge_names]
        # direct edges out of a source: the source presents the next token at the first
        # pattern-true cycle after the previous token was accepted (single output register)
        self.src_direct = [
            tuple(
                eid[e]
                for e in sorted({e for w in model.chains[n].writes for e in w})
                if model.edges[e].direct
            )
            if model.chains[n].kind == "source"
            else ()
            for n in chain_names
        ]
        # direct edges into a sink: the sink's pattern gates the writer (one handshake event)
        self.gate = [
            model.chains[model.edges[n].dst].pattern
            if model.edges[n].direct and model.chains[model.edges[n].dst].kind == "sink"
            else None
            for n in edge_names
        ]
        self.record_events = record_events
        self.record_edges = {eid[n] for n in (record_edges or ())}
        self.nc = nc
        self.ne = ne

    def run(
        self,
        max_frames: int,
        min_frames: int,
        max_cycles: int | None,
        stop_when_stable: bool,
        stable_occupancy: bool,
    ) -> SimResult:
        """Execute the model and return the result."""
        nc, ne = self.nc, self.ne
        n_ev, gaps, rd, wr, arcs = self.L, self.gaps, self.rd, self.wr, self.arcs
        kind, pattern, period = self.kind, self.pattern, self.period
        keep_hist, hist_window = self.keep_hist, self.hist_window
        lf, lb, depth, gate = self.lf, self.lb, self.depth, self.gate
        src_direct = self.src_direct
        group, freezer, until_empty = self.group, self.freezer, self.until_empty
        frozen_by = [-1] * self.ngroups  # chain currently freezing the group, or -1

        # ---- mutable state
        idx = [0] * nc  # next event index within the frame
        frame = [0] * nc  # frames completed
        # time of the previous event; the first event fires at >= start
        last_t = [self.start[c] - gaps[c][0] if gaps[c] else 0 for c in range(nc)]
        cnt = [0] * nc  # events fired so far
        hist: list[list[int]] = [[] for _ in range(nc)]
        hist_base = [0] * nc
        waiters: list[list[int]] = [[] for _ in range(nc)]
        times: list[list[int]] | None = [[] for _ in range(nc)] if self.record_events else None

        wq: list[deque[int]] = [deque([_NEG_INF] * self.init_tokens[e]) for e in range(ne)]
        rr: list[list[int] | None] = [
            [_NEG_INF] * depth[e] if depth[e] is not None else None for e in range(ne)
        ]
        wc = list(self.init_tokens)
        rc = [0] * ne
        dw = [-1] * ne  # chain waiting for data on this edge
        sw = [-1] * ne  # chain waiting for space on this edge
        maxocc = list(self.init_tokens)
        first_valid = [-1] * ne
        dirty: list[int] = []  # edges written in the current cycle (occupancy measured at its end)
        in_dirty = [False] * ne
        rec_w: list[list[int] | None] = [[] if e in self.record_edges else None for e in range(ne)]
        rec_r: list[list[int] | None] = [[] if e in self.record_edges else None for e in range(ne)]

        sinks = self.sinks
        sources = self.sources
        frames_done = 0
        frame_end: list[int] = []
        frame_start: list[int] = []
        src_starts: list[list[int]] = [[] for _ in range(nc)]
        last_occ_change = 0
        stable = False
        timeout = False
        done = False
        prio = [0 if self.freezer[c] else 1 for c in range(nc)]  # freezers first in a cycle
        heap: list[tuple[int, int, int]] = []
        for c in range(nc):
            heappush(heap, (self.start[c], prio[c], c))
        limit = max_cycles if max_cycles is not None else (1 << 62)
        t = 0
        cur_t = 0

        while heap:
            t, _p, c = heappop(heap)
            if t != cur_t:
                # end of cycle cur_t: record the occupancy of the edges written in it
                for e in dirty:
                    in_dirty[e] = False
                    occ = wc[e] - rc[e]
                    if occ > maxocc[e]:
                        maxocc[e] = occ
                        last_occ_change = frames_done
                dirty.clear()
                cur_t = t
            if t > limit:
                timeout = True
                break
            i = idx[c]
            k = kind[c]
            # ---- gap, pacing and source gate (negative times are allowed for chains whose
            # first event lies before the environment's cycle 0, see Chain.start)
            tmin = last_t[c] + gaps[c][i]
            if k == 1:
                if i == 0 and period[c]:
                    tp = frame[c] * period[c]
                    if tp > tmin:
                        tmin = tp
                blocked = False
                for e in src_direct[c]:
                    # presentation waits for the acceptance of the previous token
                    kk = wc[e]
                    if kk - rc[e] >= 1:
                        sw[e] = c
                        blocked = True
                        break
                    if kk >= 1:
                        ts = rr[e][(kk - 1) % 1] + lb[e]  # type: ignore[index]
                        if ts > tmin:
                            tmin = ts
                if blocked:
                    continue
                pat = pattern[c]
                if pat is not None:
                    tmin = pat.next_true(tmin)
            if tmin > t:
                heappush(heap, (tmin, prio[c], c))
                continue
            grp = group[c]
            if grp >= 0 and frozen_by[grp] >= 0 and frozen_by[grp] != c:
                # the pipeline is frozen: retry when the freezing chain fires
                waiters[frozen_by[grp]].append(c)
                continue
            tneed = t
            blocked = False
            # ---- data constraints
            for e in rd[c][i]:
                if wc[e] <= rc[e]:
                    dw[e] = c
                    blocked = True
                    break
                tr = wq[e][0] + lf[e]
                if tr > tneed:
                    tneed = tr
            if blocked:
                continue
            data_ready = tneed == t
            # ---- space constraints
            for e in wr[c][i]:
                d = depth[e]
                if d is not None:
                    kk = wc[e]
                    if kk - rc[e] >= d:
                        sw[e] = c
                        blocked = True
                        break
                    if kk >= d:
                        ts = rr[e][(kk - d) % d] + lb[e]  # type: ignore[index]
                        if ts > tneed:
                            tneed = ts
            if blocked:
                if data_ready and freezer[c]:
                    frozen_by[grp] = c
                continue
            # ---- index-mapped arcs
            for s, si, q, dly in arcs[c][i]:
                gi = (frame[c] - q) * n_ev[s] + si
                if gi < 0:
                    continue
                if gi >= cnt[s]:
                    waiters[s].append(c)
                    blocked = True
                    break
                hb = gi - hist_base[s]
                if hb < 0:
                    raise FINNInternalError(
                        f"History of chain {self.chain_names[s]} was truncated below event {gi} "
                        f"referenced by {self.chain_names[c]}; increase history_window"
                    )
                ta = hist[s][hb] + dly
                if ta > tneed:
                    tneed = ta
            if blocked:
                continue
            # ---- sink gate (own pattern) and direct-edge gates (patterns of sink readers)
            if k == 2:
                pat = pattern[c]
                if pat is not None:
                    tneed = pat.next_true(tneed)
            for e in wr[c][i]:
                g = gate[e]
                if g is not None:
                    tneed = g.next_true(tneed)
            if tneed > t:
                if data_ready and freezer[c]:
                    frozen_by[grp] = c
                heappush(heap, (tneed, prio[c], c))
                continue
            # a freeze-until-empty freezer keeps the group frozen while more than one token
            # is still queued on its input edges (after this one is taken)
            if (
                grp >= 0
                and freezer[c]
                and frozen_by[grp] == c
                and (not until_empty[c] or all(wc[e] - rc[e] <= 2 for e in rd[c][i]))
            ):
                frozen_by[grp] = -1
            # ---- fire at t
            for e in rd[c][i]:
                wq[e].popleft()
                r = rc[e]
                rc[e] = r + 1
                d = depth[e]
                if d is not None:
                    rr[e][r % d] = t  # type: ignore[index]
                if rec_r[e] is not None:
                    rec_r[e].append(t)  # type: ignore[union-attr]
                w = sw[e]
                if w >= 0:
                    sw[e] = -1
                    heappush(heap, (t + lb[e], prio[w], w))
            for e in wr[c][i]:
                wq[e].append(t)
                wc[e] += 1
                if not in_dirty[e]:
                    in_dirty[e] = True
                    dirty.append(e)
                if first_valid[e] < 0:
                    first_valid[e] = t
                if rec_w[e] is not None:
                    rec_w[e].append(t)  # type: ignore[union-attr]
                w = dw[e]
                if w >= 0:
                    dw[e] = -1
                    heappush(heap, (t + lf[e], prio[w], w))
            last_t[c] = t
            cnt[c] += 1
            if keep_hist[c]:
                h = hist[c]
                h.append(t)
                win = hist_window[c]
                if win is not None and len(h) > 2 * win + 64:
                    drop = len(h) - win
                    del h[:drop]
                    hist_base[c] += drop
            if times is not None:
                times[c].append(t)
            ws = waiters[c]
            if ws:
                for w in ws:
                    heappush(heap, (t, prio[w], w))
                ws.clear()
            if k == 1 and i == 0:
                # frame start = earliest event 0 among all sources, once every source has one
                src_starts[c].append(t)
                fn = len(frame_start)
                if all(len(src_starts[s]) > fn for s in sources):
                    frame_start.append(min(src_starts[s][fn] for s in sources))
            i += 1
            if i == n_ev[c]:
                i = 0
                frame[c] += 1
                if k == 2:
                    fd = min(frame[s] for s in sinks)
                    while frames_done < fd:
                        frames_done += 1
                        frame_end.append(t)
                        n = len(frame_end)
                        if (
                            stop_when_stable
                            and n >= min_frames
                            and n > STABLE_INTERVALS
                            and (
                                not stable_occupancy
                                or frames_done - last_occ_change >= STABLE_OCC_FRAMES
                            )
                        ):
                            iv = frame_end[-1] - frame_end[-2]
                            if all(
                                frame_end[-j] - frame_end[-j - 1] == iv
                                for j in range(2, STABLE_INTERVALS + 1)
                            ):
                                stable = True
                                done = True
                        if n >= max_frames:
                            done = True
                    if done:
                        break
            idx[c] = i
            heappush(heap, (t + gaps[c][i], prio[c], c))

        for e in dirty:
            occ = wc[e] - rc[e]
            if occ > maxocc[e]:
                maxocc[e] = occ
        deadlock = not done and not timeout
        # ---- interval estimate
        if done and stable:
            interval: float = float(frame_end[-1] - frame_end[-2])
        elif done and len(frame_end) >= 2:
            ivs = [b - a for a, b in pairwise(frame_end)]
            half = ivs[len(ivs) // 2 :]
            interval = float(sum(half)) / len(half)
        else:
            interval = math.inf
        latency: int | None = None
        if frame_end:
            n = min(len(frame_end), len(frame_start))
            if n > 0:
                latency = frame_end[n - 1] - frame_start[n - 1]
        en = self.edge_names
        return SimResult(
            interval=interval,
            latency=latency,
            frames=frames_done,
            cycles=t,
            stable=stable,
            deadlock=deadlock,
            timeout=timeout,
            max_occupancy={en[e]: maxocc[e] for e in range(ne)},
            first_valid={en[e]: first_valid[e] for e in range(ne)},
            frame_end_times=frame_end,
            frame_start_times=frame_start,
            event_times=(
                {self.chain_names[c]: times[c] for c in range(nc)} if times is not None else None
            ),
            handshakes=(
                {en[e]: (rec_w[e], rec_r[e]) for e in self.record_edges}  # type: ignore[misc]
                if self.record_edges
                else None
            ),
        )


def simulate(
    model: TEGModel,
    depths: dict[str, int | None] | None = None,
    *,
    max_frames: int = 64,
    min_frames: int = 4,
    max_cycles: int | None = None,
    stop_when_stable: bool = True,
    stable_occupancy: bool = True,
    record_events: bool = False,
    record_edges: set[str] | None = None,
) -> SimResult:
    """Run the self-timed execution of ``model``.

    Args:
        model: the timed event graph; ``model.validate()`` should have been called.
        depths: overrides for edge depths (``None`` = unbounded); edges not listed keep
            ``FIFOEdge.depth``.
        max_frames: stop after this many frames at the latest.
        min_frames: never stop before this many frames.
        max_cycles: give up (``timeout``) after this cycle.
        stop_when_stable: stop as soon as the stable-state criterion triggers.
        stable_occupancy: also require the maximum occupancies to have settled (disable when
            sources run free into unbounded FIFOs, where occupancy grows forever).
        record_events: keep all event times per chain (memory!).
        record_edges: keep write/read handshake cycles for these edges.
    """
    sim = _Simulator(model, depths, record_events, record_edges)
    return sim.run(max_frames, min_frames, max_cycles, stop_when_stable, stable_occupancy)


def evaluate(
    model: TEGModel, depths: dict[str, int | None] | None = None, **kwargs: object
) -> dict[str, object]:
    """Simulate and return ``{"interval", "latency", "occupancies", "stable", "deadlock"}``.

    ``occupancies`` covers the external edges only.
    """
    res = simulate(model, depths, **kwargs)  # type: ignore[arg-type]
    return {
        "interval": res.interval,
        "latency": res.latency,
        "occupancies": {n: res.max_occupancy[n] for n in model.external_edges},
        "stable": res.stable,
        "deadlock": res.deadlock,
        "frames": res.frames,
        "cycles": res.cycles,
    }


def bottleneck_interval(model: TEGModel, **kwargs: object) -> SimResult:
    """Simulate with unbounded external FIFOs and free-running sources to measure ``I*``."""
    depths: dict[str, int | None] = dict.fromkeys(model.external_edges, None)
    kwargs.setdefault("stable_occupancy", False)
    return simulate(model, depths, **kwargs)  # type: ignore[arg-type]

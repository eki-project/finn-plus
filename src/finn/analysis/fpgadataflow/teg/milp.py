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

"""Exact FIFO sizing as a mixed-integer linear program (periodic-schedule certificate).

By the certificate theorem (formal model notes, Section 5.3) a depth vector achieves frame
interval ``P`` iff there is a periodic schedule ``t(v,i,n) = s(v,i) + n*P`` of all events that
respects the chain gaps, the data arcs and the space arcs of the FIFOs. The MILP has one
continuous variable per event of one frame, one binary per (external edge, candidate depth)
and ``Theta(sum_e N_e * |C_e|)`` big-M space constraints, which is why it is exact but does not
scale (Section 6.2). Stall patterns of environment chains are not representable and ignored.

Every solution must be verified with the simulator before it is applied (``search`` does so).
"""

from __future__ import annotations

import math
import numpy as np
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from finn.util.exception import FINNInternalError

if TYPE_CHECKING:
    from finn.analysis.fpgadataflow.teg.model import TEGModel

#: candidate depth: (nominal depth as configured on the FIFO node, effective token capacity)
Candidate = tuple[int, int]
CostFn = Callable[[str, int], float]


@dataclass
class MILPSize:
    """Instance size of the periodic-schedule MILP (computed without building it)."""

    events: int
    external_tokens: int
    binaries: int
    variables: int
    chain_constraints: int
    data_constraints: int
    arc_constraints: int
    space_constraints_fixed: int
    space_constraints_bigm: int
    selection_constraints: int

    @property
    def constraints(self) -> int:
        """Total number of constraint rows."""
        return (
            self.chain_constraints
            + self.data_constraints
            + self.arc_constraints
            + self.space_constraints_fixed
            + self.space_constraints_bigm
            + self.selection_constraints
        )

    def to_dict(self) -> dict[str, int]:
        """Plain dict for JSON reports."""
        d = dict(self.__dict__)
        d["constraints"] = self.constraints
        return d


@dataclass
class MILPResult:
    """Outcome of :func:`solve_milp`."""

    depths: dict[str, int] | None
    status: int
    message: str
    solve_time: float
    objective: float | None
    size: MILPSize
    #: effective capacities of the chosen candidates (what the simulator has to verify)
    capacities: dict[str, int] = field(default_factory=dict)


def _token_maps(model: TEGModel) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Per edge: event index of the ``k``-th write and of the ``k``-th read within one frame."""
    wr: dict[str, list[int]] = {}
    rd: dict[str, list[int]] = {}
    for name, e in model.edges.items():
        wr[name] = [i for i, w in enumerate(model.chains[e.src].writes) if name in w]
        rd[name] = [i for i, r in enumerate(model.chains[e.dst].reads) if name in r]
        if len(wr[name]) != len(rd[name]):
            raise FINNInternalError(f"Rate mismatch on edge {name}")
    return wr, rd


def estimate_size(model: TEGModel, candidates: dict[str, Sequence[Candidate]]) -> MILPSize:
    """Count variables and constraints of the MILP for ``model`` and candidate sets."""
    wr, _rd = _token_maps(model)
    events = model.num_events()
    ext = model.external_edges
    binaries = sum(len(candidates[e]) for e in ext)
    n_tokens_ext = sum(len(wr[e]) for e in ext)
    chain_cons = events  # L-1 in-frame gaps + 1 wrap-around per chain
    data_cons = sum(len(wr[e]) for e in model.edges)
    arc_cons = model.num_arcs()
    space_fixed = 0
    for name, e in model.edges.items():
        if e.external:
            continue
        if e.depth is not None:
            space_fixed += len(wr[name])
    space_bigm = sum(len(wr[e]) * len(candidates[e]) for e in ext)
    return MILPSize(
        events=events,
        external_tokens=n_tokens_ext,
        binaries=binaries,
        variables=events + binaries,
        chain_constraints=chain_cons,
        data_constraints=data_cons,
        arc_constraints=arc_cons,
        space_constraints_fixed=space_fixed,
        space_constraints_bigm=space_bigm,
        selection_constraints=len(ext),
    )


def default_cost(model: TEGModel) -> CostFn:
    """Cost of a candidate: nominal depth times token width in bits (total FIFO bits).

    This is the metric ``RunLayerParallelSimulation`` uses to pick between minimisation
    orders, so MILP and search results are comparable one-to-one.
    """

    def cost(edge: str, depth: int) -> float:
        return float(depth * model.edges[edge].width)

    return cost


def solve_milp(
    model: TEGModel,
    period: int,
    candidates: dict[str, Sequence[Candidate]],
    cost: CostFn | None = None,
    time_limit: float = 600.0,
    verbose: bool = False,
    big_m: int | None = None,
) -> MILPResult:
    """Build and solve the periodic-schedule MILP with HiGHS (``scipy.optimize.milp``).

    Args:
        model: validated TEG model.
        period: target frame interval ``P`` (cycles).
        candidates: per external edge, the candidate ``(nominal depth, effective capacity)``
            pairs; nominal depths are what gets reported, capacities enter the constraints.
        cost: ``cost(edge, nominal_depth)``; defaults to total bits.
        time_limit: solver time limit in seconds.
        verbose: let HiGHS print its log.
        big_m: big-M constant; defaults to ``4 * period + 10``.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    if cost is None:
        cost = default_cost(model)
    wr, rd = _token_maps(model)
    size = estimate_size(model, candidates)
    per = period
    big = big_m if big_m is not None else 4 * per + 10

    # ---- variable indexing
    sidx: dict[tuple[str, int], int] = {}
    n = 0
    for v, c in model.chains.items():
        for i in range(c.num_events):
            sidx[(v, i)] = n
            n += 1
    yidx: dict[tuple[str, int], int] = {}
    for e in model.external_edges:
        for depth, _cap in candidates[e]:
            yidx[(e, depth)] = n
            n += 1
    nvar = n

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    nrow = 0

    def add(coefs: dict[int, float], lb: float, ub: float) -> None:
        nonlocal nrow
        for j, cval in coefs.items():
            rows.append(nrow)
            cols.append(j)
            vals.append(cval)
        lo.append(lb)
        hi.append(ub)
        nrow += 1

    # ---- chain constraints: s_{i+1} - s_i >= g_{i+1};  s_0 + P - s_{L-1} >= g_0
    for v, c in model.chains.items():
        n_ev = c.num_events
        for i in range(n_ev):
            a = sidx[(v, i)]
            b = sidx[(v, (i + 1) % n_ev)]
            g = c.gaps[(i + 1) % n_ev]
            if a == b:  # single-event chain: the wrap-around reduces to P >= g
                if g > per:
                    return MILPResult(
                        None, 2, f"infeasible: gap of chain {v} exceeds period", 0.0, None, size
                    )
                continue
            if i + 1 < n_ev:
                add({b: 1.0, a: -1.0}, g, math.inf)
            else:
                add({b: 1.0, a: -1.0}, g - per, math.inf)

    # ---- data constraints (all edges), with initial marking m: read j of frame n reads the
    # token produced by write (j - m) mod N of frame n - q, q = -((j - m) // N)
    for name, e in model.edges.items():
        n_tok = len(wr[name])
        m = e.initial_tokens
        for j in range(n_tok):
            jp = (j - m) % n_tok
            q = -((j - m) // n_tok)
            a = sidx[(e.src, wr[name][jp])]
            b = sidx[(e.dst, rd[name][j])]
            add({b: 1.0, a: -1.0}, e.lf - q * per, math.inf)

    # ---- index-mapped arcs: s_b - s_a + q P >= delay
    for v, c in model.chains.items():
        for i, arcs in c.arcs.items():
            b = sidx[(v, i)]
            for arc in arcs:
                a = sidx[(arc.src, arc.src_event)]
                if a == b:
                    continue
                add({b: 1.0, a: -1.0}, arc.delay - arc.lag * per, math.inf)

    # ---- space constraints: write j of frame n (token j + m) needs the read of token
    # j + m - D, i.e. read index r = j + m - D -> frame n - q with q = -(r // N), index r mod N
    def space_rows(name: str, cap: int, y: int | None) -> None:
        e = model.edges[name]
        n_tok = len(wr[name])
        m = e.initial_tokens
        for j in range(n_tok):
            r = j + m - cap
            q = -(r // n_tok)
            jr = r % n_tok
            a = sidx[(e.src, wr[name][j])]
            b = sidx[(e.dst, rd[name][jr])]
            if a == b:
                continue
            if y is None:
                add({a: 1.0, b: -1.0}, e.lb - q * per, math.inf)
            else:
                # s_a - s_b + qP >= lb - M(1 - y)  ->  s_a - s_b - M y >= lb - qP - M
                add({a: 1.0, b: -1.0, y: -float(big)}, e.lb - q * per - big, math.inf)

    for name, e in model.edges.items():
        if e.external:
            for depth, cap in candidates[name]:
                space_rows(name, cap, yidx[(name, depth)])
            add({yidx[(name, depth)]: 1.0 for depth, _ in candidates[name]}, 1.0, 1.0)
        elif e.depth is not None:
            space_rows(name, e.depth, None)

    a_mat = coo_matrix((vals, (rows, cols)), shape=(nrow, nvar)).tocsr()
    cvec = np.zeros(nvar)
    for (e, depth), j in yidx.items():
        cvec[j] = cost(e, depth)
    integrality = np.zeros(nvar)
    lbv = np.full(nvar, -np.inf)
    ubv = np.full(nvar, np.inf)
    for j in yidx.values():
        integrality[j] = 1
        lbv[j] = 0.0
        ubv[j] = 1.0
    t0 = time.time()
    res = milp(
        cvec,
        constraints=LinearConstraint(a_mat, np.array(lo), np.array(hi)),
        integrality=integrality,
        bounds=Bounds(lbv, ubv),
        options={"time_limit": time_limit, "disp": verbose},
    )
    dt = time.time() - t0
    if res.x is None:
        return MILPResult(None, int(res.status), str(res.message), dt, None, size)
    depths: dict[str, int] = {}
    caps: dict[str, int] = {}
    for name in model.external_edges:
        for depth, cap in candidates[name]:
            if res.x[yidx[(name, depth)]] > 0.5:
                depths[name] = depth
                caps[name] = cap
    return MILPResult(depths, int(res.status), str(res.message), dt, float(res.fun), size, caps)


def fit_power_law(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares fit of ``t = a * n**b`` in log-log space; returns ``(a, b)``."""
    pts = [(n, t) for n, t in points if n > 0 and t > 0]
    if len(pts) < 2:
        raise FINNInternalError("Need at least two positive points to fit a power law")
    x = np.log([n for n, _ in pts])
    y = np.log([t for _, t in pts])
    b, loga = np.polyfit(x, y, 1)
    return float(math.exp(loga)), float(b)


def extrapolate(a: float, b: float, n: float) -> float:
    """Evaluate ``a * n**b``."""
    return a * n**b

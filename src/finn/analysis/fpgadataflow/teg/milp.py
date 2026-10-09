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

Solvers: HiGHS through ``scipy.optimize.milp`` (bundled, single-threaded, no callbacks) and
Gurobi through ``gurobipy`` (optional extra ``gurobi``; multi-threaded branch and bound,
every improved solution and periodic progress are reported through callbacks).
"""

from __future__ import annotations

import math
import numpy as np
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from finn.util.exception import FINNInternalError, FINNUserError

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
    #: relative optimality gap, best bound and explored nodes when the solver reports them
    gap: float | None = None
    bound: float | None = None
    nodes: int | None = None
    solver: str = "highs"


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
        """Storage bits of ``edge`` at nominal ``depth``."""
        return float(depth * model.edges[edge].width)

    return cost


@dataclass
class MILPInstance:
    """The built MILP ``lo <= A x <= hi``, ``min c x`` with the index maps to decode a solution."""

    a_mat: object  # scipy.sparse.csr_matrix
    lo: np.ndarray
    hi: np.ndarray
    cvec: np.ndarray
    integrality: np.ndarray
    lbv: np.ndarray
    ubv: np.ndarray
    yidx: dict[tuple[str, int], int]
    size: MILPSize
    candidates: dict[str, Sequence[Candidate]]
    external_edges: list[str]
    #: set when the instance is trivially infeasible (a chain gap exceeds the period)
    infeasible_message: str | None = None


#: incumbent callback: (depths, capacities, objective, best bound, elapsed seconds)
IncumbentFn = Callable[[dict[str, int], dict[str, int], float, float | None, float], None]
#: progress callback: a dict with elapsed, nodes, incumbent, bound, gap, rss_gb, cpu_percent
ProgressFn = Callable[[dict[str, object]], None]


def build_instance(
    model: TEGModel,
    period: int,
    candidates: dict[str, Sequence[Candidate]],
    cost: CostFn | None = None,
    big_m: int | None = None,
) -> MILPInstance:
    """Build the periodic-schedule MILP of ``model`` for frame interval ``period``.

    Args:
        model: validated TEG model.
        period: target frame interval ``P`` (cycles).
        candidates: per external edge, the candidate ``(nominal depth, effective capacity)``
            pairs; nominal depths are what gets reported, capacities enter the constraints.
        cost: ``cost(edge, nominal_depth)``; defaults to total bits.
        big_m: big-M constant for every edge; by default each edge gets its own bound
            ``(ceil(max capacity / N_e) + 1) * P``: a deactivated space row for capacity
            ``D`` is violated by at most ``t_w(k + D' - D) - t_w(k)`` under the active row of
            the selected capacity ``D' > D`` (for ``D' <= D`` it is implied), and those
            ``D' - D`` writes span at most ``ceil((D' - D) / N_e)`` periods plus the writes of
            one frame, which take less than a period (frame-wrap gap row). The former global
            ``4P`` was loose for small FIFOs and unsafe for capacities beyond three frames.
    """
    from scipy.sparse import coo_matrix

    if cost is None:
        cost = default_cost(model)
    wr, rd = _token_maps(model)
    size = estimate_size(model, candidates)
    per = period

    def edge_big_m(name: str) -> float:
        """Big-M of the space rows of external edge ``name`` (see the ``big_m`` argument)."""
        if big_m is not None:
            return float(big_m)
        n_tok = len(wr[name])
        max_cap = max(cap for _, cap in candidates[name])
        return float((-(-max_cap // n_tok) + 1) * per)

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
        """Append the row ``lb <= sum(coefs) <= ub`` to the constraint matrix."""
        nonlocal nrow
        for j, cval in coefs.items():
            rows.append(nrow)
            cols.append(j)
            vals.append(cval)
        lo.append(lb)
        hi.append(ub)
        nrow += 1

    def finish(infeasible: str | None = None) -> MILPInstance:
        """Assemble the sparse instance."""
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
        return MILPInstance(
            a_mat,
            np.array(lo),
            np.array(hi),
            cvec,
            integrality,
            lbv,
            ubv,
            yidx,
            size,
            candidates,
            list(model.external_edges),
            infeasible,
        )

    # ---- chain constraints: s_{i+1} - s_i >= g_{i+1};  s_0 + P - s_{L-1} >= g_0
    for v, c in model.chains.items():
        n_ev = c.num_events
        for i in range(n_ev):
            a = sidx[(v, i)]
            b = sidx[(v, (i + 1) % n_ev)]
            g = c.gaps[(i + 1) % n_ev]
            if a == b:  # single-event chain: the wrap-around reduces to P >= g
                if g > per:
                    return finish(f"infeasible: gap of chain {v} exceeds period")
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
        """Space rows of edge ``name`` at capacity ``cap``, conditional on binary ``y``
        (None: unconditional, for internal edges of fixed depth)."""
        e = model.edges[name]
        n_tok = len(wr[name])
        m = e.initial_tokens
        big = edge_big_m(name) if y is not None else 0.0
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
                add({a: 1.0, b: -1.0, y: -big}, e.lb - q * per - big, math.inf)

    for name, e in model.edges.items():
        if e.external:
            for depth, cap in candidates[name]:
                space_rows(name, cap, yidx[(name, depth)])
            add({yidx[(name, depth)]: 1.0 for depth, _ in candidates[name]}, 1.0, 1.0)
        elif e.depth is not None:
            space_rows(name, e.depth, None)
    return finish()


def _decode(inst: MILPInstance, x: np.ndarray) -> tuple[dict[str, int], dict[str, int]]:
    """Selected nominal depths and capacities of a solution vector."""
    depths: dict[str, int] = {}
    caps: dict[str, int] = {}
    for name in inst.external_edges:
        for depth, cap in inst.candidates[name]:
            if x[inst.yidx[(name, depth)]] > 0.5:
                depths[name] = depth
                caps[name] = cap
    return depths, caps


def _process_usage() -> tuple[float, float]:
    """Resident set size (GB) and CPU utilisation (%) of this process."""
    try:
        import psutil

        proc = psutil.Process()
        return proc.memory_info().rss / 2**30, proc.cpu_percent(interval=None)
    except Exception:  # psutil missing or restricted
        return float("nan"), float("nan")


def solve_milp(
    model: TEGModel,
    period: int,
    candidates: dict[str, Sequence[Candidate]],
    cost: CostFn | None = None,
    time_limit: float = 600.0,
    verbose: bool = False,
    big_m: int | None = None,
    solver: str = "highs",
    threads: int = 0,
    mip_gap: float | None = None,
    log_file: str | Path | None = None,
    on_incumbent: IncumbentFn | None = None,
    on_progress: ProgressFn | None = None,
    progress_interval: float = 60.0,
    mem_limit_gb: float | None = None,
) -> MILPResult:
    """Build and solve the periodic-schedule MILP.

    Args:
        model: validated TEG model.
        period: target frame interval ``P`` (cycles).
        candidates: per external edge, the candidate ``(nominal depth, effective capacity)``
            pairs; nominal depths are what gets reported, capacities enter the constraints.
        cost: ``cost(edge, nominal_depth)``; defaults to total bits.
        time_limit: solver time limit in seconds.
        verbose: let the solver print its log to the console.
        big_m: big-M override, see :func:`build_instance`.
        solver: ``highs`` (SciPy's bundled HiGHS, single-threaded, no callbacks) or
            ``gurobi`` (``gurobipy``, multi-threaded, reports every improved solution).
        threads: solver threads (0: the solver's default, all cores for Gurobi).
        mip_gap: relative optimality gap at which the solver stops (None: solver default).
        log_file: path of the solver's own log (Gurobi only).
        on_incumbent: called with every improved solution (Gurobi: as soon as the solver
            finds it; HiGHS: once with the final solution), so that a long solve leaves its
            best depths behind even when it is cut short.
        on_progress: called about every ``progress_interval`` seconds during the branch and
            bound with node count, incumbent, bound, gap and this process's memory and CPU
            (Gurobi only).
        progress_interval: seconds between progress reports.
        mem_limit_gb: soft memory limit in GB after which the solver stops with its best
            solution (Gurobi ``SoftMemLimit``).
    """
    inst = build_instance(model, period, candidates, cost, big_m)
    if inst.infeasible_message is not None:
        return MILPResult(None, 2, inst.infeasible_message, 0.0, None, inst.size, solver=solver)
    if solver == "highs":
        return _solve_highs(inst, time_limit, verbose, on_incumbent)
    if solver == "gurobi":
        return _solve_gurobi(
            inst,
            time_limit,
            verbose,
            threads,
            mip_gap,
            log_file,
            on_incumbent,
            on_progress,
            progress_interval,
            mem_limit_gb,
        )
    raise FINNInternalError(f"Unknown MILP solver {solver!r} (highs or gurobi)")


def _solve_highs(
    inst: MILPInstance, time_limit: float, verbose: bool, on_incumbent: IncumbentFn | None
) -> MILPResult:
    """Solve with HiGHS through ``scipy.optimize.milp``."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    t0 = time.time()
    res = milp(
        inst.cvec,
        constraints=LinearConstraint(inst.a_mat, inst.lo, inst.hi),
        integrality=inst.integrality,
        bounds=Bounds(inst.lbv, inst.ubv),
        options={"time_limit": time_limit, "disp": verbose},
    )
    dt = time.time() - t0
    if res.x is None:
        return MILPResult(None, int(res.status), str(res.message), dt, None, inst.size)
    depths, caps = _decode(inst, res.x)
    bound = float(res.mip_dual_bound) if getattr(res, "mip_dual_bound", None) is not None else None
    gap = float(res.mip_gap) if getattr(res, "mip_gap", None) is not None else None
    nodes = int(res.mip_node_count) if getattr(res, "mip_node_count", None) is not None else None
    if on_incumbent is not None:
        on_incumbent(depths, caps, float(res.fun), bound, dt)
    return MILPResult(
        depths,
        int(res.status),
        str(res.message),
        dt,
        float(res.fun),
        inst.size,
        caps,
        gap=gap,
        bound=bound,
        nodes=nodes,
        solver="highs",
    )


def gurobi_license_hint() -> None:
    """Point gurobipy at the license of a Gurobi installation reachable through ``GUROBI_HOME``.

    The CI loads Gurobi as an environment module that sets ``GUROBI_HOME`` (python-mip finds
    its library there); gurobipy ships its own library and looks for ``gurobi.lic`` through
    ``GRB_LICENSE_FILE`` or its default locations, so the module's token-server license is
    made visible here when nothing else is configured.
    """
    import os

    if os.environ.get("GRB_LICENSE_FILE"):
        return
    home = os.environ.get("GUROBI_HOME")
    if home:
        lic = Path(home) / "gurobi.lic"
        if lic.is_file():
            os.environ["GRB_LICENSE_FILE"] = str(lic)


def _gurobi_env(
    gp: object, attempts: int = 6, delay: float = 5.0, delay_max: float = 60.0
) -> object:
    """Start a Gurobi environment, retrying with backoff when the token server has no token
    or does not answer in time (Gurobi's own recommendation, as in the multi-FPGA
    partitioner)."""
    last: Exception | None = None
    for attempt in range(attempts):
        env = gp.Env(empty=True)  # type: ignore[attr-defined]
        try:
            env.setParam("OutputFlag", 0)
            env.start()
            return env
        except gp.GurobiError as e:  # type: ignore[attr-defined]
            last = e
            env.dispose()
            if attempt + 1 < attempts:
                time.sleep(delay)
                delay = min(delay * 2, delay_max)
    raise FINNUserError(f"Could not start a Gurobi environment: {last}")


def _solve_gurobi(
    inst: MILPInstance,
    time_limit: float,
    verbose: bool,
    threads: int,
    mip_gap: float | None,
    log_file: str | Path | None,
    on_incumbent: IncumbentFn | None,
    on_progress: ProgressFn | None,
    progress_interval: float,
    mem_limit_gb: float | None,
) -> MILPResult:
    """Solve with Gurobi through ``gurobipy``'s matrix API."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as e:  # pragma: no cover - depends on the installation
        raise FINNUserError(
            "The MILP solver 'gurobi' needs the gurobipy package (pip install "
            "'finn-plus[gurobi]') and a Gurobi license"
        ) from e
    gurobi_license_hint()
    t0 = time.time()
    env = _gurobi_env(gp)
    m = gp.Model("teg_fifo_sizing", env=env)
    try:
        m.Params.OutputFlag = 1 if (verbose or log_file) else 0
        m.Params.LogToConsole = 1 if verbose else 0
        if log_file:
            m.Params.LogFile = str(log_file)
        m.Params.TimeLimit = time_limit
        if threads:
            m.Params.Threads = int(threads)
        if mip_gap is not None:
            m.Params.MIPGap = mip_gap
        if mem_limit_gb:
            m.Params.SoftMemLimit = float(mem_limit_gb)
        nvar = len(inst.cvec)
        vtype = np.where(inst.integrality > 0, GRB.BINARY, GRB.CONTINUOUS)
        x = m.addMVar(nvar, lb=inst.lbv, ub=inst.ubv, obj=inst.cvec, vtype=vtype)
        a = inst.a_mat
        lo, hi = inst.lo, inst.hi
        ge = np.isfinite(lo) & ~np.isfinite(hi)
        le = ~np.isfinite(lo) & np.isfinite(hi)
        eq = np.isfinite(lo) & np.isfinite(hi) & (lo == hi)
        rng = np.isfinite(lo) & np.isfinite(hi) & (lo != hi)
        if ge.any():
            m.addMConstr(a[ge], x, ">", lo[ge])
        if le.any():
            m.addMConstr(a[le], x, "<", hi[le])
        if eq.any():
            m.addMConstr(a[eq], x, "=", lo[eq])
        if rng.any():
            m.addMConstr(a[rng], x, ">", lo[rng])
            m.addMConstr(a[rng], x, "<", hi[rng])
        m.ModelSense = GRB.MINIMIZE
        state = {"last_progress": time.time(), "build_s": time.time() - t0}

        def callback(model: object, where: int) -> None:
            """Report improved solutions and periodic progress."""
            if where == GRB.Callback.MIPSOL and on_incumbent is not None:
                xs = np.asarray(model.cbGetSolution(x))  # type: ignore[attr-defined]
                obj = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))  # type: ignore[attr-defined]
                bnd = float(model.cbGet(GRB.Callback.MIPSOL_OBJBND))  # type: ignore[attr-defined]
                depths, caps = _decode(inst, xs)
                on_incumbent(depths, caps, obj, bnd, time.time() - t0)
            elif where == GRB.Callback.MIP and on_progress is not None:
                now = time.time()
                if now - state["last_progress"] >= progress_interval:
                    state["last_progress"] = now
                    best = float(model.cbGet(GRB.Callback.MIP_OBJBST))  # type: ignore[attr-defined]
                    bnd = float(model.cbGet(GRB.Callback.MIP_OBJBND))  # type: ignore[attr-defined]
                    rss, cpu = _process_usage()
                    gap = abs(best - bnd) / max(abs(best), 1e-9) if best < GRB.INFINITY else None
                    on_progress(
                        {
                            "phase": "branch_and_bound",
                            "elapsed_s": now - t0,
                            "nodes": int(model.cbGet(GRB.Callback.MIP_NODCNT)),  # type: ignore
                            "open_nodes": int(model.cbGet(GRB.Callback.MIP_NODLFT)),  # type: ignore
                            "incumbent": best if best < GRB.INFINITY else None,
                            "bound": bnd,
                            "gap": gap,
                            "rss_gb": rss,
                            "cpu_percent": cpu,
                        }
                    )

        if on_progress is not None:
            rss, cpu = _process_usage()
            on_progress(
                {
                    "phase": "model_built",
                    "elapsed_s": time.time() - t0,
                    "rows": int(a.shape[0]),
                    "columns": nvar,
                    "rss_gb": rss,
                    "cpu_percent": cpu,
                }
            )
        m.optimize(callback)
        dt = time.time() - t0
        status_names = {
            GRB.OPTIMAL: (0, "Optimal"),
            GRB.TIME_LIMIT: (1, "Time limit reached"),
            GRB.INFEASIBLE: (2, "Infeasible"),
            GRB.INF_OR_UNBD: (2, "Infeasible or unbounded"),
            GRB.UNBOUNDED: (3, "Unbounded"),
            GRB.INTERRUPTED: (4, "Interrupted"),
            GRB.MEM_LIMIT: (4, "Memory limit reached"),
        }
        status, message = status_names.get(m.Status, (4, f"Gurobi status {m.Status}"))
        if m.SolCount == 0:
            return MILPResult(None, status, message, dt, None, inst.size, solver="gurobi")
        depths, caps = _decode(inst, np.asarray(x.X))
        bound = float(m.ObjBound) if abs(m.ObjBound) < GRB.INFINITY else None
        return MILPResult(
            depths,
            status,
            message,
            dt,
            float(m.ObjVal),
            inst.size,
            caps,
            gap=float(m.MIPGap) if m.MIPGap < GRB.INFINITY else None,
            bound=bound,
            nodes=int(m.NodeCount),
            solver="gurobi",
        )
    finally:
        m.dispose()
        env.dispose()


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

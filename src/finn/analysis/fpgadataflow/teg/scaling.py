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

"""Scaling experiments of the exact MILP FIFO sizing (paper table and figure).

Three kinds of instances share one measurement path:

* ``chain``: a synthetic three-FIFO chain (bursty producer, half-rate consumer) scaled by the
  tokens per frame;
* ``model``: the complete TEG of a regression model as the ``milp`` strategy would build it;
* ``prefix``: the TEG of the first ``n`` nodes of a model (a down-scaled real instance).

``size`` entries only count the instance (equation 6 of the paper). Every solve writes its
improved solutions to ``<label>_incumbent.json`` as they are found (the depths of a solve that
takes days must survive a cut-short job), a progress heartbeat to ``<label>_progress.json``
and the solver log to ``<label>_solver.log``; the finished row goes to ``<label>.json`` and
``results.jsonl``. :func:`summarize` fits the power law and extrapolates to the full models.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from finn.analysis.fpgadataflow.teg.from_onnx import build_model, depths_to_fifo_config, edge_widths
from finn.analysis.fpgadataflow.teg.milp import (
    Candidate,
    estimate_size,
    extrapolate,
    fit_power_law,
    solve_milp,
)
from finn.analysis.fpgadataflow.teg.model import Chain, TEGModel
from finn.analysis.fpgadataflow.teg.search import measure_bottleneck, measure_peak_occupancy
from finn.analysis.fpgadataflow.teg.simulate import simulate
from finn.transformation.fpgadataflow.fifo_depth_search import (
    candidate_depths,
    effective_capacity,
    safe_bram_starting_depth,
)
from finn.util.logging import log

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qonnx.core.modelwrapper import ModelWrapper

#: candidate depths of the synthetic chain (nominal depth == capacity)
CHAIN_DEPTHS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
MAX_QSRL_DEPTH = 256


@dataclass
class ScalingInstance:
    """One instance of the experiment: model, target interval and candidate depths."""

    label: str
    kind: str
    teg: TEGModel
    target: int
    candidates: dict[str, Sequence[Candidate]]
    notes: dict[str, object] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        """File-name stem of the instance."""
        return re.sub(r"[^A-Za-z0-9._-]+", "_", self.label)


def synthetic_chain(n: int, burst: int = 8) -> TEGModel:
    """``src -> A (reads, idles and writes in bursts of ``burst``) -> B (one token per two cycles)
    -> sink`` with ``n`` tokens per frame (prototype experiment 1)."""
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


def chain_instance(n: int, burst: int = 8) -> ScalingInstance:
    """The synthetic chain with ``n`` tokens per frame."""
    teg = synthetic_chain(n, burst)
    cands = {e: [(d, d) for d in CHAIN_DEPTHS] for e in teg.external_edges}
    return ScalingInstance(f"chain n={n}", "chain", teg, teg.bottleneck_interval(), cands)


def model_candidates(
    teg: TEGModel, max_frames: int = 32, fine: bool = True, backend: str = "native"
) -> tuple[int, dict[str, list[Candidate]]]:
    """Target interval and candidate depths exactly as the ``milp`` sizing strategy builds
    them (bottleneck interval, paced peak occupancies, block-granular depths, fine depths)."""
    widths = edge_widths(teg)
    base = measure_bottleneck(teg, max_frames=max_frames, backend=backend)
    target = math.ceil(base.interval)
    paced = measure_peak_occupancy(teg, target, max_frames=max_frames, backend=backend)
    upper = {
        e: safe_bram_starting_depth(paced.max_occupancy[e], MAX_QSRL_DEPTH)
        for e in teg.external_edges
    }
    fine_depths = [2, 4, 8, 16] if fine else []
    cands = {
        e: sorted(
            {
                (d, effective_capacity(d, MAX_QSRL_DEPTH))
                for d in [*fine_depths, *candidate_depths(widths[e], upper[e], MAX_QSRL_DEPTH)]
            }
        )
        for e in teg.external_edges
    }
    return target, cands


def prefix_graph(model: ModelWrapper, n_nodes: int) -> ModelWrapper:
    """The first ``n_nodes`` nodes of a dataflow graph as a model of their own (the tensors
    that fed the dropped nodes become graph outputs)."""
    import copy

    from onnx import helper
    from qonnx.core.modelwrapper import ModelWrapper

    model = ModelWrapper(copy.deepcopy(model.model))
    nodes = list(model.graph.node)
    if n_nodes >= len(nodes):
        return model
    keep, drop = nodes[:n_nodes], nodes[n_nodes:]
    drop_in = {t for n in drop for t in n.input}
    graph_outputs = {o.name for o in model.graph.output}
    new_outputs = [t for n in keep for t in n.output if t in drop_in or t in graph_outputs]
    for n in drop:
        model.graph.node.remove(n)
    for o in list(model.graph.output):
        model.graph.output.remove(o)
    shapes = {t: list(model.get_tensor_shape(t)) for t in new_outputs}
    for vi in list(model.graph.value_info):
        if vi.name in new_outputs:
            model.graph.value_info.remove(vi)
    for t in new_outputs:
        model.graph.output.append(helper.make_tensor_value_info(t, 1, shapes[t]))
    return model


def model_instance(
    label: str, model: ModelWrapper, n_nodes: int | None = None, max_frames: int = 32
) -> ScalingInstance:
    """Instance of a complete model (``n_nodes`` None) or of its first ``n_nodes`` nodes."""
    kind = "model" if n_nodes is None else "prefix"
    if n_nodes is not None:
        model = prefix_graph(model, n_nodes)
    t0 = time.time()
    teg = build_model(model)
    target, cands = model_candidates(teg, max_frames=max_frames)
    return ScalingInstance(
        label,
        kind,
        teg,
        target,
        cands,
        {"nodes": len(model.graph.node), "prepare_s": round(time.time() - t0, 1)},
    )


def size_row(inst: ScalingInstance) -> dict[str, object]:
    """Instance size (equation 6) without solving."""
    size = estimate_size(inst.teg, inst.candidates)
    return {
        "label": inst.label,
        "kind": inst.kind,
        "target_interval": inst.target,
        "chains": len(inst.teg.chains),
        "fifos": len(inst.teg.external_edges),
        "events_per_frame": inst.teg.num_events(),
        "arcs": inst.teg.num_arcs(),
        "candidates_total": sum(len(c) for c in inst.candidates.values()),
        "candidates_max": max(len(c) for c in inst.candidates.values()),
        **{f"size_{k}": v for k, v in size.to_dict().items()},
        **inst.notes,
    }


def run_instance(
    inst: ScalingInstance,
    out_dir: Path,
    solver: str = "highs",
    time_limit: float = 900.0,
    threads: int = 0,
    mip_gap: float | None = None,
    mem_limit_gb: float | None = None,
    progress_interval: float = 60.0,
    verify_frames: int = 16,
) -> dict[str, object]:
    """Solve one instance, keeping every incumbent on disk, and record the result row."""
    out_dir.mkdir(parents=True, exist_ok=True)
    row = size_row(inst)
    slug = inst.slug
    incumbents: list[dict[str, object]] = []

    def on_incumbent(
        depths: dict[str, int],
        caps: dict[str, int],
        objective: float,
        bound: float | None,
        elapsed: float,
    ) -> None:
        """Write the improved solution immediately (the artifact of a long solve)."""
        incumbents.append({"elapsed_s": elapsed, "objective_bits": objective, "bound": bound})
        (out_dir / f"{slug}_incumbent.json").write_text(
            json.dumps(
                {
                    "label": inst.label,
                    "incumbent": len(incumbents),
                    "elapsed_s": elapsed,
                    "objective_bits": objective,
                    "bound_bits": bound,
                    "target_interval_cycles": inst.target,
                    "fifo_depths": depths,
                    "fifo_capacities": caps,
                    "fifo_config": depths_to_fifo_config(inst.teg, depths)
                    if "nodes" in inst.teg.meta
                    else None,
                },
                indent=2,
            )
        )
        log.info(
            f"{inst.label}: incumbent {len(incumbents)} at {elapsed:.0f} s, {objective:.0f} bits"
        )

    def on_progress(info: dict[str, object]) -> None:
        """Heartbeat into the job log and a file next to the results."""
        (out_dir / f"{slug}_progress.json").write_text(json.dumps({"time": time.time(), **info}))
        log.info(f"{inst.label}: " + ", ".join(f"{k} {v}" for k, v in info.items()))

    log.info(
        f"{inst.label}: {row['size_constraints']} rows, {row['size_variables']} columns, "
        f"{row['size_binaries']} binaries; solving with {solver} (limit {time_limit:.0f} s)"
    )
    res = solve_milp(
        inst.teg,
        inst.target,
        inst.candidates,
        time_limit=time_limit,
        solver=solver,
        threads=threads,
        mip_gap=mip_gap,
        log_file=out_dir / f"{slug}_solver.log",
        on_incumbent=on_incumbent,
        on_progress=on_progress,
        progress_interval=progress_interval,
        mem_limit_gb=mem_limit_gb,
    )
    verified = None
    verified_interval = None
    if res.depths is not None and verify_frames > 0:
        check = simulate(inst.teg, res.capacities, max_frames=verify_frames)
        verified_interval = check.interval
        verified = bool(check.ok and check.interval <= inst.target)
    row.update(
        {
            "solver": res.solver,
            "time_limit_s": time_limit,
            "threads": threads,
            "solve_s": round(res.solve_time, 2),
            "status": res.status,
            "message": res.message[:120],
            "solved": res.depths is not None,
            "optimal": res.status == 0,
            "objective_bits": res.objective,
            "bound_bits": res.bound,
            "mip_gap": res.gap,
            "nodes_explored": res.nodes,
            "incumbents": incumbents,
            "verified": verified,
            "verified_interval": verified_interval,
            "fifo_depths": res.depths,
        }
    )
    (out_dir / f"{slug}.json").write_text(json.dumps(row, indent=2))
    with (out_dir / "results.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def record_size(inst: ScalingInstance, out_dir: Path) -> dict[str, object]:
    """Record the size of an instance that is not solved."""
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {**size_row(inst), "solved": None}
    (out_dir / f"{inst.slug}.json").write_text(json.dumps(row, indent=2))
    with (out_dir / "results.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def _human_time(t: float) -> str:
    """Seconds as a short human-readable duration."""
    for unit, div in (("s", 1), ("min", 60), ("h", 3600), ("d", 86400), ("y", 365.25 * 86400)):
        if t < div * (
            120
            if unit == "s"
            else (120 if unit == "min" else 48 if unit == "h" else 400 if unit == "d" else 1e12)
        ):
            return f"{t / div:.3g} {unit}"
    return f"{t / (365.25 * 86400):.3g} y"


def summarize(out_dir: Path, default_fit: tuple[float, float] = (1.2e-7, 1.7)) -> dict[str, object]:
    """Fit the power law on the finished solves and extrapolate to every recorded size.

    Writes ``summary.json`` and ``table.md`` into ``out_dir`` and returns the summary.
    """
    rows = [json.loads(line) for line in (out_dir / "results.jsonl").open()]
    solved = [
        r
        for r in rows
        if r.get("solved")
        and r.get("optimal")
        and r["solve_s"] > 0.05
        and r["solve_s"] < 0.95 * float(r.get("time_limit_s") or math.inf)
    ]
    pts = [(r["size_constraints"], r["solve_s"]) for r in solved]
    a, b = fit_power_law(pts) if len(pts) >= 2 else default_fit
    table = []
    seen: set[str] = set()
    for r in sorted(rows, key=lambda r: r["size_constraints"]):
        if r["label"] in seen:
            continue
        seen.add(r["label"])
        est = extrapolate(a, b, r["size_constraints"])
        table.append(
            {
                "label": r["label"],
                "kind": r["kind"],
                "constraints": r["size_constraints"],
                "variables": r["size_variables"],
                "binaries": r["size_binaries"],
                "events": r["events_per_frame"],
                "solve_s": r.get("solve_s"),
                "status": "optimal"
                if r.get("optimal")
                else (
                    "solved"
                    if r.get("solved")
                    else ("size only" if r.get("solved") is None else r.get("message"))
                ),
                "verified": r.get("verified"),
                "extrapolated_s": est,
                "extrapolated": _human_time(est),
            }
        )
    summary = {
        "fit": {"a": a, "b": b, "points": len(pts), "unit": "constraint rows"},
        "rows": table,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        f"fit on {len(pts)} optimal solves: t = {a:.3g} * n^{b:.2f} (n = constraint rows)",
        "",
        "| instance | kind | constraints | variables | binaries | events/frame | solve | status "
        "| verified | extrapolated |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for r in table:
        solve = f"{r['solve_s']:.2f} s" if r["solve_s"] is not None else ""
        lines.append(
            f"| {r['label']} | {r['kind']} | {r['constraints']:,} | {r['variables']:,} | "
            f"{r['binaries']:,} | "
            f"{r['events']:,} | {solve} | {r['status']} | {r['verified']} | {r['extrapolated']} |"
        )
    (out_dir / "table.md").write_text("\n".join(lines) + "\n")
    return summary

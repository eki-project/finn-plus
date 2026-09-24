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

"""FIFO sizing strategies built on the timed-event-graph model.

* :class:`RunAbstractSimFIFOSizing`: SimFIFO's per-FIFO minimisation with the RTL simulators
  replaced by the abstract simulator (``auto_fifo_strategy = abstract_sim``).
* :class:`RunMILPFIFOSizing`: the exact periodic-schedule MILP; always writes an instance-size
  report, solves only below a configurable size (``auto_fifo_strategy = milp``).

Both write a ``FIFODepthConfig`` and set the ``fifo_data`` metadata property so that
``ApplySimulatedFIFOSizes`` can apply the result exactly like the distributed simulation.
"""

from __future__ import annotations

import json
import math
import time
from qonnx.transformation.base import Transformation
from typing import TYPE_CHECKING, cast

from finn.analysis.fpgadataflow.teg.from_onnx import build_model, depths_to_fifo_config, edge_widths
from finn.analysis.fpgadataflow.teg.milp import MILPResult, estimate_size, extrapolate, solve_milp
from finn.analysis.fpgadataflow.teg.search import (
    capacities_for,
    measure_bottleneck,
    measure_peak_occupancy,
    minimize_depths,
)
from finn.analysis.fpgadataflow.teg.simulate import simulate
from finn.transformation.fpgadataflow.fifo_depth_search import (
    MinimizationOrder,
    candidate_depths,
    effective_capacity,
    safe_bram_starting_depth,
)
from finn.util.basic import make_build_dir
from finn.util.exception import FINNUserError
from finn.util.logging import log

if TYPE_CHECKING:
    from pathlib import Path
    from qonnx.core.modelwrapper import ModelWrapper

    from finn.analysis.fpgadataflow.teg.model import TEGModel
    from finn.builder.build_dataflow_config import DataflowBuildConfig

#: solve-time calibration ``t = a * n^b`` (seconds vs. constraint rows) measured with HiGHS on
#: the prototype's chain instances (formal model notes, Section 7); refined by the calibration
#: routine of the paper scripts.
DEFAULT_MILP_CALIBRATION = (1.2e-7, 1.7)


def _orders(cfg: DataflowBuildConfig) -> list[MinimizationOrder]:
    names = cast("list[str]", cfg.teg_minimization_orders)
    try:
        return [MinimizationOrder[n] for n in names]
    except KeyError as exc:
        raise FINNUserError(
            f"Unknown teg_minimization_orders entry {exc}; valid: "
            + ", ".join(o.name for o in MinimizationOrder)
        ) from exc


def _write_fifo_data(model: ModelWrapper, teg: TEGModel, depths: dict[str, int]) -> Path:
    work_folder = cast("Path", make_build_dir("fifo_results_teg_", True))
    path = work_folder / "fifo_config.json"
    path.write_text(json.dumps(depths_to_fifo_config(teg, depths)))
    model.set_metadata_prop("fifo_data", str(path))
    return path


def _dump_model(cfg: DataflowBuildConfig, teg: TEGModel, name: str) -> None:
    if cfg.teg_dump_model:
        p = cfg.get_report_directory() / f"{name}.json"
        teg.dump_json(p)
        log.info(f"Dumped TEG model to {p}")


class RunAbstractSimFIFOSizing(Transformation):
    """Size the FIFOs with the abstract (TEG) simulation.

    Requires DWCs to be inserted and all nodes to be specialised (``_hls``/``_rtl``).
    """

    def __init__(self, cfg: DataflowBuildConfig, max_qsrl_depth: int = 256) -> None:
        """Store the build configuration."""
        super().__init__()
        self.cfg = cfg
        self.max_qsrl_depth = max_qsrl_depth

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Build the model, run the search, hand the depths over via ``fifo_data``."""
        t0 = time.time()
        teg = build_model(model)
        _dump_model(self.cfg, teg, "teg_model")
        log.info(
            f"abstract_sim: model with {len(teg.chains)} chains, {teg.num_events()} events per "
            f"frame, {len(teg.external_edges)} FIFOs to size, {teg.num_arcs()} arcs"
        )
        res = minimize_depths(
            teg,
            edge_widths(teg),
            minimization_orders=_orders(self.cfg),
            max_qsrl_depth=self.max_qsrl_depth,
            max_frames=int(self.cfg.teg_max_frames),
            target_interval=self.cfg.teg_target_interval,
        )
        _write_fifo_data(model, teg, res.depths)
        report = {
            "strategy": "abstract_sim",
            "target_interval_cycles": res.target_interval,
            "final_interval_cycles": res.final_interval,
            "simulations": res.simulations,
            "sizing_time_s": time.time() - t0,
            "search_time_s": res.total_time,
            "fifos": {
                e: {
                    "depth": res.depths[e],
                    "peak_occupancy": res.peak_occupancy[e],
                    "first_valid_cycle": res.first_valid[e],
                    "iterations": res.iterations.get(e, 0),
                    "time_s": res.times.get(e, 0.0),
                    "width": teg.edges[e].width,
                }
                for e in teg.external_edges
            },
            "model": {
                "chains": len(teg.chains),
                "events_per_frame": teg.num_events(),
                "arcs": teg.num_arcs(),
            },
        }
        report_dir = self.cfg.get_report_directory()
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "fifo_sizing_abstract_sim.json").write_text(json.dumps(report, indent=2))
        log.info(
            f"abstract_sim: sized {len(teg.external_edges)} FIFOs with {res.simulations} "
            f"simulations in {res.total_time:.1f} s (interval {res.final_interval})"
        )
        return model, False


class RunMILPFIFOSizing(Transformation):
    """Size the FIFOs with the exact MILP, or report why that is infeasible.

    The instance-size report (``fifo_sizing_milp.json``) is always written. The MILP is built
    and solved only when its constraint count is below ``cfg.teg_milp_max_constraints``;
    otherwise the step falls back to ``abstract_sim`` when
    ``cfg.teg_milp_fallback_to_abstract_sim`` is set and raises ``FINNUserError`` otherwise.
    Every solution is verified with the simulator before it is applied.
    """

    def __init__(self, cfg: DataflowBuildConfig, max_qsrl_depth: int = 256) -> None:
        """Store the build configuration."""
        super().__init__()
        self.cfg = cfg
        self.max_qsrl_depth = max_qsrl_depth

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Build the model, size the instance, solve if small enough."""
        cfg = self.cfg
        t0 = time.time()
        teg = build_model(model)
        _dump_model(cfg, teg, "teg_model")
        widths = edge_widths(teg)
        max_frames = int(cfg.teg_max_frames)
        base = measure_bottleneck(teg, max_frames=max_frames)
        interval = (
            float(cfg.teg_target_interval) if cfg.teg_target_interval is not None else base.interval
        )
        target = math.ceil(interval)
        paced = measure_peak_occupancy(teg, target, max_frames=max_frames)
        upper = {
            e: safe_bram_starting_depth(paced.max_occupancy[e], self.max_qsrl_depth)
            for e in teg.external_edges
        }
        fine = [2, 4, 8, 16] if cfg.teg_milp_fine_candidates else []
        candidates = {
            e: sorted(
                {
                    (d, effective_capacity(d, self.max_qsrl_depth))
                    for d in [*fine, *candidate_depths(widths[e], upper[e], self.max_qsrl_depth)]
                }
            )
            for e in teg.external_edges
        }
        size = estimate_size(teg, candidates)
        a, b = DEFAULT_MILP_CALIBRATION
        est_time = extrapolate(a, b, size.constraints)
        report: dict[str, object] = {
            "strategy": "milp",
            "target_interval_cycles": target,
            "instance": size.to_dict(),
            "candidates_per_fifo": {e: len(c) for e, c in candidates.items()},
            "estimated_solve_time_s": est_time,
            "calibration": {"a": a, "b": b, "unit": "constraint rows"},
            "max_constraints": int(cfg.teg_milp_max_constraints),
            "solved": False,
        }
        report_dir = cfg.get_report_directory()
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "fifo_sizing_milp.json"

        def finish(extra: dict[str, object]) -> None:
            report.update(extra)
            report["sizing_time_s"] = time.time() - t0
            report_path.write_text(json.dumps(report, indent=2))

        log.info(
            f"milp: instance with {size.constraints} constraint rows, {size.variables} variables "
            f"({size.binaries} binary); extrapolated solve time {est_time:.1f} s"
        )
        if size.constraints > int(cfg.teg_milp_max_constraints):
            msg = (
                f"MILP instance too large to solve ({size.constraints} constraint rows > "
                f"teg_milp_max_constraints = {cfg.teg_milp_max_constraints}); "
                f"extrapolated solve time {est_time:.0f} s"
            )
            if cfg.teg_milp_fallback_to_abstract_sim:
                log.warning(msg + "; falling back to abstract_sim")
                finish({"fallback": "abstract_sim"})
                return RunAbstractSimFIFOSizing(cfg, self.max_qsrl_depth).apply(model)
            finish({"error": msg})
            raise FINNUserError(msg + f"; report written to {report_path}")

        res: MILPResult = solve_milp(
            teg, target, candidates, time_limit=float(cfg.teg_milp_time_limit)
        )
        if res.depths is None:
            finish({"solver_status": res.status, "solver_message": res.message})
            raise FINNUserError(f"MILP solver failed: {res.message} (report at {report_path})")
        check = simulate(
            teg, capacities_for(res.depths, self.max_qsrl_depth), max_frames=max_frames
        )
        verified = check.ok and check.interval <= target
        finish(
            {
                "solved": True,
                "solver_status": res.status,
                "solver_message": res.message,
                "solve_time_s": res.solve_time,
                "objective_bits": res.objective,
                "verified_interval_cycles": check.interval,
                "verified": verified,
                "fifos": {
                    e: {"depth": res.depths[e], "width": widths[e]} for e in teg.external_edges
                },
            }
        )
        if not verified:
            raise FINNUserError(
                f"MILP solution failed simulator verification (interval {check.interval} vs "
                f"target {target}); refusing to apply it"
            )
        _write_fifo_data(model, teg, res.depths)
        log.info(f"milp: solved in {res.solve_time:.1f} s, objective {res.objective} bits")
        return model, False

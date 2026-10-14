#!/usr/bin/env python3
"""Compare the FIFO sizing search strategies of a live FIFO sizing CI run.

Reads the ``fifo_sizing_report.json`` files that ``experiment_fifosizing`` writes per strategy
and stop condition (``reports/experiment_fifosizing/exp_itr_1/<strategy>/<stop_condition>/``)
for every run of a benchmark and prints one Markdown table per stop condition: rows are the DUTs,
columns the strategies, cells the resulting total FIFO size with the number of hardware
evaluations and the search duration.

Usage, with the measurement artifacts of the ``Measurement`` job (and optionally the build
artifacts for the DUT names) extracted or fetched locally::

    python scripts/compare_fifo_search_strategies.py measurement_artifacts/runs_output \\
        --build-runs build_artifacts/runs_output --output comparison.md

Only the report files are needed, e.g. fetched one by one through the GitLab single-file
artifact API; the rest of the (large) artifact archive can be left out.
"""

import argparse
import json
import math
import re
import sys
from collections import OrderedDict
from pathlib import Path

#: Column order; strategies not listed here are appended alphabetically.
STRATEGY_ORDER = [
    "binary (best order)",
    "binary_forward",
    "binary_reverse",
    "binary_largest_first",
    "binary_deepest_first",
    "binary_alternating",
    "bottom_up_exact",
    "top_down_bnb_seed_binary",
    "top_down_bnb_seed_occupancy",
]

#: Shorter column headers
STRATEGY_LABELS = {
    "binary_forward": "bin fwd",
    "binary_reverse": "bin rev",
    "binary_largest_first": "bin largest",
    "binary_deepest_first": "bin deepest",
    "binary_alternating": "bin altern.",
    "bottom_up_exact": "bottom-up exact",
    "top_down_bnb_seed_binary": "B&B (seed bin)",
    "top_down_bnb_seed_occupancy": "B&B (seed occ.)",
}


def run_index(path):
    """Numeric run id of a ``run_<n>`` directory name."""
    m = re.search(r"run_(\d+)", path.name)
    return int(m.group(1)) if m else -1


def run_params(run_dir, build_runs):
    """Benchmark parameters of a run from its metadata_bench.json (measurement or build tree)."""
    candidates = [run_dir / "metadata_bench.json", run_dir / "reports" / "metadata_bench.json"]
    if build_runs is not None:
        b = build_runs / run_dir.name
        candidates += [b / "metadata_bench.json", b / "reports" / "metadata_bench.json"]
    for c in candidates:
        if c.is_file():
            with open(c) as f:
                return json.load(f).get("params", {})
    return {}


def dut_names(reports, build_runs):
    """Readable DUT name per run: the dut, plus the model name where one dut has several runs."""
    params = {run_dir: run_params(run_dir, build_runs) for run_dir in reports}
    counts = {}
    for p in params.values():
        counts[p.get("dut")] = counts.get(p.get("dut"), 0) + 1
    names = {}
    for run_dir, p in params.items():
        dut = p.get("dut")
        if dut is None:
            names[run_dir] = run_dir.name
            continue
        model = p.get("model_path")
        if counts[dut] > 1 and model:
            stem = Path(model).stem.replace("_qonnx", "")
            if stem == "streamlined":  # finn-transformers layout: <model>/streamlined.onnx
                stem = Path(model).parent.name
            names[run_dir] = f"{dut}/{stem}"
        else:
            names[run_dir] = dut
    return names


def load_reports(runs_output):
    """Collect ``{run_dir: {stop_condition: {strategy: report}}}`` from a runs_output tree."""
    reports = OrderedDict()
    for run_dir in sorted(Path(runs_output).glob("run_*"), key=run_index):
        base = run_dir / "reports" / "experiment_fifosizing" / "exp_itr_1"
        if not base.is_dir():
            continue
        per_run = {}
        for strategy_dir in sorted(base.iterdir()):
            if not strategy_dir.is_dir():
                continue
            for stop_dir in sorted(strategy_dir.iterdir()):
                report = stop_dir / "fifo_sizing_report.json"
                if report.is_file():
                    with open(report) as f:
                        per_run.setdefault(stop_dir.name, {})[strategy_dir.name] = json.load(f)
        if per_run:
            reports[run_dir] = per_run
    return reports


def fmt_duration(seconds):
    """Compact duration string."""
    if seconds is None:
        return "?"
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 600:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.0f}m"


def cell(report, best_kb):
    """Table cell: size in kB, evaluations, duration and status markers."""
    search = report.get("search")
    size = report.get("fifo_size_total_kB")
    if size is None:
        return "n/a"
    if search is not None:
        evals = search.get("num_evaluations")
        duration = search.get("duration_s")
        markers = ""
        if search.get("optimal_proven"):
            markers += " ✓"
        if search.get("budget_exhausted"):
            markers += " ⏱"
        if search.get("abort_reason"):
            markers += " ✗"
    else:  # report of the driver before the search refactor
        bs = report.get("binary_search", {})
        evals = len(bs.get("all_iterations", {})) or None
        duration = bs.get("duration")
        markers = ""
    text = f"{size:.2f} ({evals if evals is not None else '?'}, {fmt_duration(duration)}){markers}"
    if "best_binary_order" in report:
        text += f" [{report['best_binary_order']}]"
    if best_kb is not None and math.isclose(size, best_kb, rel_tol=0, abs_tol=1e-6):
        text = f"**{text}**"
    return text


def upper_bound(per_strategy):
    """Cost of the start configuration (paced-mode peak occupancies) across the run's reports.

    Every experiment measures the occupancies anew, so the values can differ slightly between
    strategies; returns ``(min, max)``.
    """
    values = []
    for report in per_strategy.values():
        first = (report.get("search") or {}).get("all_iterations", {}).get("0")
        if first is None:  # report of the driver before the search refactor
            first = report.get("binary_search", {}).get("all_iterations", {}).get("0")
        if first and first.get("total_fifo_size_kB") is not None:
            values.append(first["total_fifo_size_kB"])
    return (min(values), max(values)) if values else None


def fmt_bound(bound):
    """Single value, or a range when the strategies measured different start configurations."""
    if bound is None:
        return "?"
    lo, hi = bound
    if math.isclose(lo, hi, rel_tol=0, abs_tol=5e-3):
        return f"{lo:.2f}"
    return f"{lo:.2f}–{hi:.2f}"


def lower_bound(per_strategy):
    """Cost of the per-FIFO lower bounds, as reported by any exact strategy of the run."""
    for report in per_strategy.values():
        lb = (report.get("search") or {}).get("lower_bounds_total_kB")
        if lb is not None:
            return lb
    return None


BINARY_BEST = "binary (best order)"


def best_binary(per_strategy):
    """Combine the binary_<order> reports of a run into the one with the lowest FIFO size."""
    binary = {s: r for s, r in per_strategy.items() if s.startswith("binary_")}
    if not binary:
        return per_strategy
    order, report = min(binary.items(), key=lambda kv: kv[1].get("fifo_size_total_kB", math.inf))
    best_size = report.get("fifo_size_total_kB", math.inf)
    tied = [
        s
        for s, r in binary.items()
        if math.isclose(r.get("fifo_size_total_kB", math.inf), best_size, rel_tol=0, abs_tol=1e-6)
    ]
    label = "tie" if len(tied) > 1 else order[len("binary_") :]
    combined = {s: r for s, r in per_strategy.items() if not s.startswith("binary_")}
    combined[BINARY_BEST] = dict(report, best_binary_order=label)
    return combined


def table_for(reports, stop_condition, names, all_binary_orders=False):
    """Markdown table for one stop condition."""
    if not all_binary_orders:
        reports = OrderedDict(
            (run_dir, {sc: best_binary(ps) for sc, ps in per_run.items()})
            for run_dir, per_run in reports.items()
        )
    strategies = set()
    for per_run in reports.values():
        strategies |= set(per_run.get(stop_condition, {}))
    ordered = [s for s in STRATEGY_ORDER if s in strategies] + sorted(
        strategies - set(STRATEGY_ORDER)
    )
    if not ordered:
        return f"No reports with stop_condition `{stop_condition}`.\n"
    header = ["DUT", "target (cycles)", "UB kB", "LB kB"] + [
        STRATEGY_LABELS.get(s, s) for s in ordered
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for run_dir, per_run in reports.items():
        per_strategy = per_run.get(stop_condition, {})
        if not per_strategy:
            continue
        sizes = [
            r["fifo_size_total_kB"] for r in per_strategy.values() if "fifo_size_total_kB" in r
        ]
        best = min(sizes) if sizes else None
        any_report = next(iter(per_strategy.values()))
        target = any_report.get("target_interval_cycles")
        lb = lower_bound(per_strategy)
        row = [
            names[run_dir],
            str(target) if target is not None else "?",
            fmt_bound(upper_bound(per_strategy)),
            f"{lb:.2f}" if lb is not None else "?",
        ]
        for s in ordered:
            row.append(cell(per_strategy[s], best) if s in per_strategy else "–")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs_output", help="measurement_artifacts/runs_output directory")
    parser.add_argument(
        "--build-runs", help="build_artifacts/runs_output directory (for metadata_bench.json)"
    )
    parser.add_argument(
        "--stop-condition",
        nargs="+",
        default=["both", "interval"],
        help="stop conditions to tabulate (default: both interval)",
    )
    parser.add_argument(
        "--all-binary-orders",
        action="store_true",
        help="one column per binary search order instead of the best order per DUT",
    )
    parser.add_argument("--output", help="write the Markdown to this file instead of stdout")
    args = parser.parse_args(argv)

    reports = load_reports(args.runs_output)
    if not reports:
        print(f"No fifo_sizing_report.json found below {args.runs_output}", file=sys.stderr)
        return 1
    build_runs = Path(args.build_runs) if args.build_runs else None
    names = dut_names(reports, build_runs)

    out = []
    out.append("# Live FIFO sizing: search strategy comparison\n")
    out.append(
        "Cells: total FIFO size in kB (hardware evaluations, search duration). "
        "The cheapest result per DUT is bold. Markers: ✓ optimality proven, "
        "⏱ budget exhausted, ✗ search aborted (seed returned). The binary column shows the "
        "cheapest of the binary search orders per DUT, with that order in brackets "
        "(tie: several orders reach the same size). "
        "UB is the size of the start configuration, the peak occupancies measured with "
        "unbounded FIFOs (a range if the strategies' own measurements differed). "
        "LB is the cost of the per-FIFO lower bounds (every FIFO at its minimum with all "
        "others unbounded), a lower bound on any feasible configuration.\n"
    )
    for stop_condition in args.stop_condition:
        out.append(f"## stop_condition = {stop_condition}\n")
        out.append(table_for(reports, stop_condition, names, args.all_binary_orders))
    text = "\n".join(out)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

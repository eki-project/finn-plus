#!/usr/bin/env python3
"""Compare the FIFO sizing cost functions of a live FIFO sizing CI run on their built result.

For the ``live_fifosizing_costcmp.yml`` benchmark every DUT is measured once per experiment
config (``fifosizing_cost_bits.json`` and ``fifosizing_cost_primitives.json``): the same search
(branch-and-bound seeded by the largest-first binary search) minimising the storage in bits or
the FPGA primitive cost. Each result gets its own follow-up build, so the actual resources of
the FIFOs can be compared with what the search predicted, in both metrics.

The script joins, per run, the search report of the measurement job, the parameters of the
build job (DUT name, experiment config), the per-node synthesis resources and FIFO depths of
the follow-up build job, and the throughput measured on the follow-up build, and prints one
Markdown row per (DUT, cost model). Fetch only the files it reads (see ``--help`` and the
``REPORT_FILES`` list) through the GitLab single-file artifact API; the archives are large.

Usage::

    python scripts/compare_fifo_cost_models.py measurement_artifacts/runs_output \\
        --build-runs build_artifacts/runs_output \\
        --followup-build-runs build_artifacts_followup/runs_output \\
        --followup-measurement-runs measurement_artifacts_followup/runs_output \\
        --output cost_models.md
"""

import argparse
import json
import math
import re
import sys
from collections import OrderedDict
from pathlib import Path

#: Files read per run (relative to the run directory of the respective artifact tree)
REPORT_FILES = {
    "measurement": (
        "reports/experiment_fifosizing/exp_itr_1/<report_name>/<stop>/fifo_sizing_report.json"
    ),
    "build": "reports/metadata_bench.json",
    "followup_build": (
        "reports/post_synth_resources.json, reports/fifo_sizing.json, reports/metadata_bench.json"
    ),
    "followup_measurement": (
        "reports/instrumentation_experiment/exp_itr_1/report_experiment_instrumentation.json"
    ),
}

#: Resource columns of the synthesized FIFOs (post_synth_resources.json rows of the FIFO nodes)
SYNTH_RESOURCES = ("LUT", "SRL", "FF", "BRAM_18K", "BRAM_36K", "URAM")

#: Cost-model name from the experiment config file name
COST_MODEL_FROM_CONFIG = {
    "fifosizing_cost_bits.json": "bits",
    "fifosizing_cost_primitives.json": "primitives",
}


def run_index(path):
    """Numeric run id of a ``run_<n>`` directory name."""
    m = re.search(r"run_(\d+)", path.name)
    return int(m.group(1)) if m else -1


def load_json(path):
    """Parsed JSON file, or ``None`` if it does not exist."""
    path = Path(path)
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def run_params(runs, run_name):
    """Benchmark parameters of a run from its metadata_bench.json, or ``{}``."""
    if runs is None:
        return {}
    for c in (
        runs / run_name / "metadata_bench.json",
        runs / run_name / "reports" / "metadata_bench.json",
    ):
        data = load_json(c)
        if data is not None:
            return data.get("params", {})
    return {}


def dut_name(params, run_name):
    """Readable DUT name: dut plus the model name for DUTs with several models."""
    dut = params.get("dut")
    if dut is None:
        return run_name
    model = params.get("model_path")
    if dut in ("bnn-pynq", "transformer") and model:
        stem = Path(model).stem.replace("_qonnx", "")
        if stem == "streamlined":
            stem = Path(model).parent.name
        return f"{dut}/{stem}"
    return dut


def find_search_report(run_dir, selected_dir=None):
    """The fifo_sizing_report.json of a measurement run (the one the follow-up used if known)."""
    base = run_dir / "reports" / "experiment_fifosizing" / "exp_itr_1"
    reports = sorted(base.glob("*/*/fifo_sizing_report.json"))
    if not reports:
        return None, None
    if selected_dir is not None:
        for r in reports:
            if r.parent.parent.name == selected_dir:
                return load_json(r), f"{r.parent.parent.name}/{r.parent.name}"
    r = reports[0]
    return load_json(r), f"{r.parent.parent.name}/{r.parent.name}"


def selected_strategy_dir(followup_params):
    """Strategy sub-directory of the measurement report that the follow-up build was made from."""
    cfg = followup_params.get("fifo_config_file") or ""
    m = re.search(r"exp_itr_1/([^/]+)/[^/]+/fifo_config\.json", cfg)
    return m.group(1) if m else None


def synthesized_fifo_resources(post_synth):
    """Sum of the synthesis resources of all FIFO nodes, or ``None``."""
    if post_synth is None:
        return None
    total = {r: 0 for r in SYNTH_RESOURCES}
    found = False
    for name, row in post_synth.items():
        if "StreamingFIFO" in name and isinstance(row, dict):
            found = True
            for r in SYNTH_RESOURCES:
                total[r] += int(row.get(r, 0))
    return total if found else None


def fmt(x, digits=2):
    """Number or ``n/a``."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def fmt_duration(seconds):
    """Compact duration string."""
    if seconds is None:
        return "?"
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 600:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.0f}m"


def collect_rows(args):
    """One record per run with the predicted and the built FIFO cost."""
    rows = []
    for run_dir in sorted(Path(args.runs_output).glob("run_*"), key=run_index):
        run_name = run_dir.name
        params = run_params(Path(args.build_runs), run_name) if args.build_runs else {}
        fparams = (
            run_params(Path(args.followup_build_runs), run_name) if args.followup_build_runs else {}
        )
        report, report_dir = find_search_report(run_dir, selected_strategy_dir(fparams))
        if report is None:
            continue
        search = report.get("search", {})
        cost_model = report.get("cost_model") or COST_MODEL_FROM_CONFIG.get(
            Path(params.get("experiments_config", "")).name, "bits"
        )
        prim = report.get("primitive_cost")
        predicted = {
            "size_kB": sum(report.get("fifo_sizes", {}).values()) / 8000.0
            if report.get("fifo_sizes")
            else None,
            "scalar": prim.get("scalar") if prim else None,
            "vector": prim.get("total") if prim else None,
            "unit": prim.get("unit") if prim else None,
        }
        built = {"size_kB": None, "resources": None}
        if args.followup_build_runs:
            fb = Path(args.followup_build_runs) / run_name / "reports"
            sizing = load_json(fb / "fifo_sizing.json")
            if sizing and sizing.get("fifo_sizes"):
                built["size_kB"] = sum(sizing["fifo_sizes"].values()) / 8000.0
            built["resources"] = synthesized_fifo_resources(
                load_json(fb / "post_synth_resources.json")
            )
        followup = None
        if args.followup_measurement_runs:
            followup = load_json(
                Path(args.followup_measurement_runs)
                / run_name
                / "reports"
                / "instrumentation_experiment"
                / "exp_itr_1"
                / "report_experiment_instrumentation.json"
            )
        rows.append(
            {
                "run": run_name,
                "dut": dut_name(params, run_name),
                "cost_model": cost_model,
                "report_dir": report_dir,
                "num_fifos": len(report.get("fifo_depths", {})),
                "target_interval": report.get("target_interval_cycles"),
                "search": search,
                "predicted": predicted,
                "built": built,
                "followup": followup,
            }
        )
    return rows


def markdown(rows):
    """Markdown table of the collected rows, grouped by DUT."""
    header = [
        "DUT",
        "FIFOs",
        "cost model",
        "search (evals, time)",
        "pred. size kB",
        "pred. LUT-eq",
        "pred. LUT / FF / BRAM18 / URAM",
        "built size kB",
        "built FIFO LUT (SRL) / FF / BRAM18 / BRAM36 / URAM",
        "follow-up interval (target)",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    by_dut = OrderedDict()
    for r in rows:
        by_dut.setdefault(r["dut"], []).append(r)
    for dut, group in by_dut.items():
        for r in sorted(group, key=lambda x: x["cost_model"]):
            s = r["search"]
            markers = (" ✓" if s.get("optimal_proven") else "") + (
                " ⏱" if s.get("budget_exhausted") else ""
            )
            search = (
                f"{s.get('search_strategy', '?')} ({s.get('num_evaluations', '?')}, "
                f"{fmt_duration(s.get('duration_s'))}){markers}"
            )
            vec = r["predicted"]["vector"]
            pred_vec = (
                " / ".join(fmt(vec.get(k, 0.0), 0) for k in ("lut", "ff", "bram18", "uram"))
                if vec
                else "n/a"
            )
            res = r["built"]["resources"]
            built_res = (
                f"{res['LUT']} ({res['SRL']}) / {res['FF']} / {res['BRAM_18K']} / "
                f"{res['BRAM_36K']} / {res['URAM']}"
                if res
                else "n/a"
            )
            fu = r["followup"]
            if fu is None:
                followup = "n/a"
            else:
                interval = fu.get("interval_cycles")
                target = r["target_interval"]
                flag = "" if target is None or interval is None or interval <= target else " ✗"
                followup = f"{interval} ({target}){flag}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        dut,
                        str(r["num_fifos"]),
                        r["cost_model"],
                        search,
                        fmt(r["predicted"]["size_kB"]),
                        fmt(r["predicted"]["scalar"], 0),
                        pred_vec,
                        fmt(r["built"]["size_kB"]),
                        built_res,
                        followup,
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def main(argv=None):
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs_output", help="measurement_artifacts/runs_output directory")
    parser.add_argument("--build-runs", help="build_artifacts/runs_output (metadata_bench.json)")
    parser.add_argument(
        "--followup-build-runs",
        help="build_artifacts_followup/runs_output (post_synth_resources.json, fifo_sizing.json)",
    )
    parser.add_argument(
        "--followup-measurement-runs",
        help="measurement_artifacts_followup/runs_output (report_experiment_instrumentation.json)",
    )
    parser.add_argument("--output", help="write the Markdown to this file instead of stdout")
    args = parser.parse_args(argv)

    rows = collect_rows(args)
    if not rows:
        print(f"No fifo_sizing_report.json found below {args.runs_output}", file=sys.stderr)
        return 1
    text = (
        "# Live FIFO sizing: cost function comparison\n\n"
        "Predicted: the configuration the search delivered, costed by the driver in both "
        "metrics (storage of the exported depths in kB; FPGA primitives from the cost tables "
        "of the bitfile, LUT-eq = capacity-weighted LUT equivalents). Built: the FIFO depths and "
        "the synthesized resources of all FIFO nodes of the follow-up build (SRLs are also "
        "counted in the LUTs by Vivado), and its measured interval against the target. "
        "✓ optimality proven, ⏱ budget exhausted, ✗ target missed.\n\n" + markdown(rows)
    )
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

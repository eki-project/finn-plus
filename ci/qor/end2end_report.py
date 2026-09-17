#!/usr/bin/env python3
"""Produce end2end evaluation figures and tables from benchmark pipeline artifacts.

Reads the build and measurement artifacts of a CI benchmark run (same layout as
ci/collect/collect.py: ``<artifacts>/runs_output/run_<id>/reports/``), summarizes the
per-layer resource reports (post-synthesis, FINN, HLS and empirical estimates) into operator
categories, joins measured performance and power, and writes:

    results.csv                  one row per run with all flattened values
    results_table.md / .tex      measured LUTs, relative error of each estimator, power
    resource_breakdown.png       per model: post-synth resources per operator category
    estimates_vs_measured.png    per model: measured vs. estimated LUT/DSP/BRAM
    estimation_error.png         mean relative error per operator category and estimator

With ``--persist-to``, the artifacts are additionally copied into that directory so that the
experiment can be compared against later runs (``--compare``).

Only needs pandas, numpy, matplotlib and jinja2 (no FINN installation), the repository's src/
directory is added to the path automatically.
"""

import argparse
import json
import logging
import os
import pandas as pd
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from finn.qor.evaluation import (
    RESOURCE_ESTIMATORS,
    breakdown_resources,
    dataframe_to_markdown,
    end2end_results_table,
    mean_relative_error_by_category,
    plot_estimates_vs_measured,
    plot_mean_relative_error,
    plot_resource_breakdown,
)

logger = logging.getLogger("end2end_report")

BUILD_REPORTS = [
    "metadata_bench.json",
    "post_synth_resources.json",
    "estimate_layer_resources.json",
    "estimate_layer_resources_hls.json",
    "estimate_layer_resources_empirical.json",
    "estimate_power_empirical.json",
    "estimate_network_performance.json",
    "time_per_step.json",
]
RESOURCE_REPORTS = [
    "post_synth_resources.json",
    "estimate_layer_resources.json",
    "estimate_layer_resources_hls.json",
    "estimate_layer_resources_empirical.json",
    "estimate_power_empirical.json",
]
POWER_RAILS = ["0V85_power", "3V3_power", "total_power"]
# (report key, aggregation over measurement iterations)
PERFORMANCE_METRICS = [
    ("frequency_mhz", max),
    ("throughput_fps", max),
    ("latency_ms", min),
    ("min_latency_cycles", min),
    ("interval_cycles", min),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--artifacts-dir",
        default=".",
        help="directory containing build_artifacts/ and measurement_artifacts/ (default: cwd)",
    )
    parser.add_argument(
        "--followup", action="store_true", help="use the *_followup artifact directories"
    )
    parser.add_argument("--output-dir", default="qor_e2e_artifacts")
    parser.add_argument(
        "--persist-to",
        default=None,
        help="copy the artifacts into this experiment directory for later comparison",
    )
    parser.add_argument(
        "--compare",
        nargs="*",
        default=[],
        help="additional experiment directories (same layout) to include in the results",
    )
    parser.add_argument(
        "--names-json",
        default=None,
        help="JSON file mapping model names to display names ('skip' excludes a model)",
    )
    return parser.parse_args()


def read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open("r") as f:
        return json.load(f)


def run_ids(build_dir: Path) -> list[int]:
    runs_dir = build_dir / "runs_output"
    if not runs_dir.is_dir():
        return []
    return sorted(int(d[4:]) for d in os.listdir(runs_dir) if d.startswith("run_"))


def model_name(metadata: dict[str, Any]) -> str:
    """Human-readable model name, mirrors collect.py's extract_model_name."""
    params = metadata.get("params", {})
    dut = params.get("dut", "unknown")
    model_path = params.get("model_path", "")
    if dut == "bnn-pynq" and model_path:
        return os.path.basename(model_path).replace("_qonnx.onnx", "").replace(".onnx", "")
    if dut == "transformer" and "finn-transformers/" in model_path:
        return "T_" + model_path.split("finn-transformers/")[1].split("/")[0]
    return dut


def measured_performance(reports_dir: Path) -> dict[str, Any]:
    """Aggregate the instrumentation reports over all measurement iterations."""
    values: dict[str, list[float]] = {key: [] for key, _ in PERFORMANCE_METRICS}
    for report_path in reports_dir.glob("*/exp_itr_*/report_experiment_instrumentation.json"):
        report = read_json(report_path) or {}
        for key, _ in PERFORMANCE_METRICS:
            if isinstance(report.get(key), (int, float)):
                values[key].append(report[key])
    return {key: fn(values[key]) for key, fn in PERFORMANCE_METRICS if values[key]}


def measured_power(reports_dir: Path) -> dict[str, float]:
    """Average power per rail over all measurement files, mirrors collect.py's power report."""
    per_file: list[dict[str, float]] = []
    for experiment_dir in reports_dir.iterdir() if reports_dir.is_dir() else []:
        if not experiment_dir.is_dir():
            continue
        for file in experiment_dir.glob(f"{experiment_dir.name}*.json"):
            rails = (read_json(file) or {}).get("rails", [])
            if rails:
                per_file.append(
                    {
                        key: sum(r[key] for r in rails if key in r) / len(rails)
                        for key in POWER_RAILS
                    }
                )
    if not per_file:
        return {}
    result = {}
    for key in POWER_RAILS:
        samples = [p[key] for p in per_file]
        result[f"avg_{key}"] = sum(samples) / len(samples)
        result[f"min_{key}"] = min(samples)
        result[f"max_{key}"] = max(samples)
    return result


def load_run(build_dir: Path, measurement_dir: Path, run_id: int) -> Optional[dict[str, Any]]:
    """Collect all reports of one run into a nested dict (None if metadata is missing)."""
    reports_dir = build_dir / "runs_output" / f"run_{run_id}" / "reports"
    metadata = read_json(reports_dir / "metadata_bench.json")
    if metadata is None:
        logger.warning("run_%d: no metadata_bench.json, skipping", run_id)
        return None
    row: dict[str, Any] = {"run_id": run_id, "model_name": model_name(metadata)}
    for report_name in BUILD_REPORTS:
        report = read_json(reports_dir / report_name)
        if report is None:
            continue
        if report_name in RESOURCE_REPORTS:
            report = breakdown_resources(report)
        row[report_name.removesuffix(".json")] = report
    measurement_reports = measurement_dir / "runs_output" / f"run_{run_id}" / "reports"
    row["measured_performance"] = measured_performance(measurement_reports)
    row["measured_power"] = measured_power(measurement_reports)
    return row


def load_experiment(artifacts_dir: Path, followup: bool = False) -> pd.DataFrame:
    suffix = "_followup" if followup else ""
    build_dir = artifacts_dir / f"build_artifacts{suffix}"
    measurement_dir = artifacts_dir / f"measurement_artifacts{suffix}"
    rows = [load_run(build_dir, measurement_dir, i) for i in run_ids(build_dir)]
    df = pd.json_normalize([r for r in rows if r is not None])
    df["experiment"] = artifacts_dir.name
    logger.info("Loaded %d runs from %s", len(df), artifacts_dir)
    return df


def unique_model_names(names: pd.Series) -> pd.Series:
    """Suffix repeated model names (variants of the same model) with _1, _2, ..."""
    counts: dict[str, int] = {}
    result = []
    for name in names:
        if name in counts:
            counts[name] += 1
            result.append(f"{name}_{counts[name]}")
        else:
            counts[name] = 0
            result.append(name)
    return pd.Series(result, index=names.index)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    artifacts_dir = Path(args.artifacts_dir).resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [load_experiment(artifacts_dir, args.followup)]
    frames += [load_experiment(Path(d).resolve()) for d in args.compare]
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        logger.error("No runs found in %s", artifacts_dir)
        return 1
    df["model_name"] = unique_model_names(df["model_name"])
    if args.names_json:
        names = json.loads(Path(args.names_json).read_text())
        df["model_name"] = df["model_name"].map(lambda n: names.get(n, n))
        df = df[df["model_name"] != "skip"].reset_index(drop=True)
    df.to_csv(out_dir / "results.csv", index=False)

    has_measured = any(c.startswith("post_synth_resources.") for c in df.columns)
    if not has_measured:
        logger.warning("No post_synth_resources reports found, only results.csv is written")
        return 0

    table = end2end_results_table(df)
    (out_dir / "results_table.md").write_text(
        "# End2end estimation results\n\n" + dataframe_to_markdown(table)
    )
    (out_dir / "results_table.tex").write_text(
        table.to_latex(
            float_format="%.1f",
            na_rep="--",
            multicolumn_format="c",
            escape=True,
            index_names=False,
            caption="End2end estimation results",
            label="tab:qor_end2end",
        )
    )
    for res in ("LUT", "DSP", "BRAM"):
        errors = mean_relative_error_by_category(df, res)
        errors.to_csv(out_dir / f"estimation_error_{res}.csv")
    plot_resource_breakdown(df, str(out_dir / "resource_breakdown.png"))
    plot_estimates_vs_measured(df, str(out_dir / "estimates_vs_measured.png"))
    plot_mean_relative_error(df, str(out_dir / "estimation_error.png"))
    available = [
        k for k, v in RESOURCE_ESTIMATORS.items() if any(c.startswith(v + ".") for c in df.columns)
    ]
    logger.info("Estimators found in reports: %s", available)
    logger.info("Results:\n%s", dataframe_to_markdown(table))

    if args.persist_to:
        target = Path(args.persist_to)
        target.mkdir(parents=True, exist_ok=True)
        suffix = "_followup" if args.followup else ""
        for name in (f"build_artifacts{suffix}", f"measurement_artifacts{suffix}"):
            src = artifacts_dir / name
            if src.is_dir():
                shutil.copytree(src, target / name.removesuffix("_followup"), dirs_exist_ok=True)
        shutil.copytree(out_dir, target / "report", dirs_exist_ok=True)
        logger.info("Persisted artifacts and report to %s", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())

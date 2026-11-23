#!/usr/bin/env python3
"""Evaluate sets of QoR models offline on the hardware nodes of end2end (regression) builds.

Every build writes the feature row of each hardware node (``qor_node_features.json``, see
``generate_empirical_estimate_reports``) next to its resource reports. This script loads one or
more model sets (directories of stored QoR models: the default tree-based models that builds
use, and the transformer / symbolic / TabPFN sets stored by the manual fitting jobs), predicts
the resources and power of every node from its feature row and compares the predictions with
the post-synthesis resources and the measured power, per node (grouped by operator) and per
design (totals per operator category). This decouples the model comparison from what a build
loads: the QoR End2End Report shows what builds get, this report what every model type would
have given on the same designs.

Writes into ``--output-dir``:

    node_estimates.csv           one row per hardware node: measured and estimated resources
    node_errors_<RES>.md / .csv  error of every estimator per operator (node level)
    results.csv                  one row per run with the per-category totals of every set
    results_table.md / .tex      design totals: relative error of every estimator per model
    estimation_error_<RES>.csv   mean relative error per operator category and estimator
    estimation_error.png, estimates_vs_measured.png
    power_table.md / .csv        measured dynamic power vs. the sum of the node estimates
    model_sets.md / .json        provenance of every model set (regressor, fit date, samples)
    summary.json

Only needs pandas, numpy, matplotlib and jinja2 plus the runtime dependencies of the model
sets (torch for the transformer set, tabpfn + torch + the staged checkpoint for TabPFN, sympy
for the symbolic set); models whose dependencies are missing are skipped with a warning. The
repository's src/ directory is added to the path automatically.
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
import shutil
import sys
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from end2end_report import model_name, read_json, run_ids

from finn.analysis.fpgadataflow.empirical_qor_estimation import NODE_FEATURES_REPORT, QoRModelSet
from finn.benchmarking import exchange  # noqa: E402
from finn.qor.database import POWER_TARGET_COL  # noqa: E402
from finn.qor.estimator import QoREstimator, model_basename, resource_target
from finn.qor.evaluation import (
    MEASURED_RESOURCES,
    breakdown_resources,
    dataframe_to_markdown,
    end2end_results_table,
    estimation_error_table,
    mean_relative_error_by_category,
    normalize_resources,
    plot_estimates_vs_measured,
    plot_mean_relative_error,
)

logger = logging.getLogger("end2end_models")

#: Resource type of the reports -> target column of the models (BRAM in 18K equivalents)
RESOURCE_TARGETS = {
    "LUT": resource_target("LUT"),
    "DSP": resource_target("DSP"),
    "BRAM": resource_target("BRAM_18K"),
    "URAM": resource_target("URAM"),
}
RESOURCES = tuple(RESOURCE_TARGETS)
#: Reference estimates that come with every build (column prefix -> report)
REFERENCE_ESTIMATES = {
    "finn": "estimate_layer_resources",
    "hls": "estimate_layer_resources_hls",
    "empirical_build": "estimate_layer_resources_empirical",
}
REFERENCE_LABELS = {"finn": "FINN", "hls": "HLS", "empirical_build": "Empirical (build)"}
POWER_RAIL = "0V85_power"
IDLE_EXPERIMENT = "idle_experiment"
LOAD_EXPERIMENT = "instrumentation_experiment"


class ModelSetLike(Protocol):
    """What this script needs from a model set (QoRModelSet or a test stub)."""

    def get(self, op_type: str, target: str) -> Any:
        """Return the estimator of an op_type and target, or None."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="directory holding build_artifacts/ and measurement_artifacts/ (default: the "
        "exchange directory of this pipeline, see finn.benchmarking.exchange)",
    )
    parser.add_argument("--followup", action="store_true", help="use the follow-up artifacts")
    parser.add_argument(
        "--model-sets",
        required=True,
        help="comma-separated NAME=DIR pairs of model directories to evaluate",
    )
    parser.add_argument("--output-dir", default="qor_e2e_models_artifacts")
    parser.add_argument(
        "--persist-to",
        default=None,
        help="also copy the report into this directory (e.g. on the benchmark data store)",
    )
    return parser.parse_args()


def parse_model_sets(spec: str) -> dict[str, str]:
    """Parse ``NAME=DIR,NAME=DIR`` into an ordered dict."""
    sets = {}
    for item in spec.split(","):
        name, _, path = item.strip().partition("=")
        if not name or not path:
            raise ValueError(f"--model-sets expects NAME=DIR pairs, got {item!r}")
        sets[name] = path
    return sets


def experiment_power(reports_dir: Path, experiment: str, rail: str = POWER_RAIL) -> float:
    """Mean power of one rail over the recordings of one experiment (NaN if absent)."""
    samples: list[float] = []
    experiment_dir = reports_dir / experiment
    if experiment_dir.is_dir():
        for file in experiment_dir.glob(f"{experiment}*.json"):
            rails = (read_json(file) or {}).get("rails", [])
            samples += [float(r[rail]) for r in rails if rail in r]
    return float(np.mean(samples)) if samples else np.nan


def load_runs(artifacts_dir: Path, followup: bool = False) -> tuple[pd.DataFrame, dict[int, dict]]:
    """Load the hardware nodes of every run with their measured and reference resources.

    Returns the node table (one row per node that has a feature row) and a per-run dict with
    the model name, the full resource reports (for the design totals) and the measured power.
    """
    suffix = "_followup" if followup else ""
    build_dir = artifacts_dir / f"build_artifacts{suffix}"
    measurement_dir = artifacts_dir / f"measurement_artifacts{suffix}"
    rows: list[dict[str, Any]] = []
    runs: dict[int, dict[str, Any]] = {}
    for run_id in run_ids(build_dir):
        reports_dir = build_dir / "runs_output" / f"run_{run_id}" / "reports"
        metadata = read_json(reports_dir / "metadata_bench.json")
        features = read_json(reports_dir / f"{NODE_FEATURES_REPORT}.json")
        if metadata is None or features is None:
            logger.warning(
                "run_%d: no metadata or %s report, skipping", run_id, NODE_FEATURES_REPORT
            )
            continue
        measured = read_json(reports_dir / f"{MEASURED_RESOURCES}.json") or {}
        references = {
            k: read_json(reports_dir / f"{v}.json") or {} for k, v in REFERENCE_ESTIMATES.items()
        }
        measurement_reports = measurement_dir / "runs_output" / f"run_{run_id}" / "reports"
        idle = experiment_power(measurement_reports, IDLE_EXPERIMENT)
        load = experiment_power(measurement_reports, LOAD_EXPERIMENT)
        power_build = read_json(reports_dir / "estimate_power_empirical.json") or {}
        runs[run_id] = {
            "model_name": model_name(metadata),
            "measured": measured,
            "references": references,
            "measured_power": {"idle": idle, "load": load, "dyn": load - idle},
            # the build-time empirical power estimate (sum over the nodes), if models were loaded
            "power_build": float(power_build.get("total", {}).get("power", np.nan)),
        }
        for node, entry in features.items():
            row: dict[str, Any] = {
                "run_id": run_id,
                "model_name": runs[run_id]["model_name"],
                "node": node,
                "op_type": entry.get("op_type"),
                "operator": entry.get("operator"),
                "backend": entry.get("backend"),
                "features": entry.get("features"),
                "error": entry.get("error"),
            }
            for res, value in normalize_resources(measured.get(node, {})).items():
                row[f"measured.{res}"] = value
            for key, report in references.items():
                for res, value in normalize_resources(report.get(node, {})).items():
                    row[f"{key}.{res}"] = value
            rows.append(row)
    df = pd.DataFrame(rows)
    logger.info("Loaded %d hardware nodes of %d runs from %s", len(df), len(runs), artifacts_dir)
    return df, runs


def load_model_sets(spec: dict[str, str]) -> dict[str, QoRModelSet]:
    """Load every model set directory (models with missing dependencies are skipped)."""
    return {name: QoRModelSet.load(path) for name, path in spec.items()}


def model_set_provenance(spec: dict[str, str]) -> pd.DataFrame:
    """One row per stored model: set, operator, target, regressor, fit date, samples, score."""
    rows = []
    for name, path in spec.items():
        for operator, target in QoREstimator.available_models(path):
            meta = read_json(Path(path) / (model_basename(operator, target) + ".json")) or {}
            rows.append(
                {
                    "set": name,
                    "operator": operator,
                    "target": target,
                    "regressor": meta.get("regressor"),
                    "fitted_at": meta.get("fitted_at"),
                    "n_samples": meta.get("n_samples"),
                    "cv_scoring": meta.get("cv_scoring"),
                    "cv_score": meta.get("cv_score"),
                }
            )
    return pd.DataFrame(rows)


def predict_nodes(nodes: pd.DataFrame, model_sets: dict[str, ModelSetLike]) -> pd.DataFrame:
    """Add the predictions of every model set as columns ``<set>.<RES>`` and ``<set>.power``.

    A node gets NaN for a target if the set has no model for its operator, the model does not
    cover the node's categorical feature values (see ``QoREstimator.covers``) or the feature
    row does not fit the model (feature spec changed since the model was fitted).
    """
    nodes = nodes.copy()
    targets = {**RESOURCE_TARGETS, "power": POWER_TARGET_COL}
    for name, models in model_sets.items():
        for key in targets:
            nodes[f"{name}.{key}"] = np.nan
        for idx, row in nodes.iterrows():
            if not row["operator"] or not isinstance(row["features"], dict):
                continue
            features = pd.DataFrame([row["features"]])
            for key, target in targets.items():
                est = models.get(row["op_type"], target)
                if est is None or not est.covers(features):
                    continue
                try:
                    nodes.loc[idx, f"{name}.{key}"] = float(est.predict(features))
                except Exception as e:  # report the node, keep evaluating
                    logger.warning("%s: %s/%s on %s: %s", name, row["op_type"], key, row["node"], e)
    return nodes


def node_error_tables(nodes: pd.DataFrame, set_names: list[str]) -> dict[str, pd.DataFrame]:
    """Node-level error of every estimator per resource type, split by operator."""
    estimators = {label: key for key, label in REFERENCE_LABELS.items()}
    estimators.update({name: name for name in set_names})
    subsets = {"All": pd.Series(True, index=nodes.index)}
    for operator in sorted(nodes["operator"].dropna().unique()):
        subsets[operator] = nodes["operator"] == operator
    tables = {}
    for res in RESOURCES:
        if f"measured.{res}" not in nodes.columns:
            continue
        tables[res] = estimation_error_table(
            nodes, f"measured.{res}", {k: f"{v}.{res}" for k, v in estimators.items()}, subsets
        )
    return tables


def run_totals(nodes: pd.DataFrame, runs: dict[int, dict], set_names: list[str]) -> pd.DataFrame:
    """Design totals per operator category for the measured, reference and model-set estimates.

    For a model set, nodes without a prediction fall back to the FINN analytical estimate (as
    a build does), so the totals are comparable; the number of such nodes is reported in
    ``<set>.fallback_nodes``. Power is the sum of the node predictions over the nodes that
    have one (``<set>.power``).
    """
    records = []
    for run_id, run in runs.items():
        record: dict[str, Any] = {"run_id": run_id, "model_name": run["model_name"]}
        reports = {MEASURED_RESOURCES: run["measured"]}
        reports.update({v: run["references"][k] for k, v in REFERENCE_ESTIMATES.items()})
        run_nodes = nodes[nodes["run_id"] == run_id]
        for name in set_names:
            per_node: dict[str, dict[str, float]] = {}
            fallbacks = 0
            for _, row in run_nodes.iterrows():
                entry = {}
                for res in RESOURCES:
                    value = row.get(f"{name}.{res}", np.nan)
                    if pd.isna(value):
                        value = row.get(f"finn.{res}", np.nan)
                        fallbacks += int(pd.notna(value))
                    if pd.notna(value):
                        entry[res] = float(value)
                per_node[row["node"]] = entry
            total: dict[str, float] = {}
            for entry in per_node.values():
                for res, value in entry.items():
                    total[res] = total.get(res, 0.0) + value
            per_node["total"] = total
            reports[f"set_{name}"] = per_node
            record[f"set_{name}.fallback_nodes"] = fallbacks
            power = (
                run_nodes[f"{name}.power"]
                if f"{name}.power" in run_nodes
                else pd.Series(dtype=float)
            )
            record[f"set_{name}.power"] = float(power.sum()) if power.notna().any() else np.nan
            record[f"set_{name}.power_nodes"] = int(power.notna().sum())
        for prefix, report in reports.items():
            breakdown = breakdown_resources(report) if report else None
            for cat, resources in (breakdown or {}).items():
                for res, value in resources.items():
                    record[f"{prefix}.{cat}.{res}"] = value
        record["empirical_build.power"] = run.get("power_build", np.nan)
        for key, value in run["measured_power"].items():
            record[f"measured_power.{key}"] = value
        records.append(record)
    return pd.DataFrame(records)


def power_table(totals: pd.DataFrame, set_names: list[str]) -> pd.DataFrame:
    """Per model: measured dynamic 0V85 power and the summed node estimate of every set."""
    columns = {
        "Measured dyn [mW]": "measured_power.dyn",
        "Measured load [mW]": "measured_power.load",
        "Empirical (build) [mW]": "empirical_build.power",
    }
    columns.update({f"{name} [mW]": f"set_{name}.power" for name in set_names})
    table = pd.DataFrame({label: totals.get(col, np.nan) for label, col in columns.items()})
    table.index = pd.Index(totals["model_name"], name="Model")
    return table


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    artifacts_dir = Path(args.artifacts_dir or exchange.pipeline_exchange_dir() or ".").resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = parse_model_sets(args.model_sets)

    nodes, runs = load_runs(artifacts_dir, args.followup)
    if nodes.empty:
        logger.warning(
            "No hardware nodes with feature rows found in %s, nothing to do", artifacts_dir
        )
        return 0
    model_sets = load_model_sets(spec)
    set_names = list(model_sets)
    provenance = model_set_provenance(spec)
    provenance.to_json(out_dir / "model_sets.json", orient="records", indent=2)
    (out_dir / "model_sets.md").write_text(
        "# Model sets\n\n" + (dataframe_to_markdown(provenance) if not provenance.empty else "none")
    )

    nodes = predict_nodes(nodes, model_sets)
    nodes.drop(columns=["features"]).to_csv(out_dir / "node_estimates.csv", index=False)
    coverage = {
        name: int(nodes[f"{name}.LUT"].notna().sum()) if f"{name}.LUT" in nodes else 0
        for name in set_names
    }
    for res, table in node_error_tables(nodes, set_names).items():
        table.to_csv(out_dir / f"node_errors_{res}.csv")
        (out_dir / f"node_errors_{res}.md").write_text(
            f"# Node-level {res} estimation error\n\n{dataframe_to_markdown(table)}"
        )

    totals = run_totals(nodes, runs, set_names)
    totals.to_csv(out_dir / "results.csv", index=False)
    estimators = {label: REFERENCE_ESTIMATES[key] for key, label in REFERENCE_LABELS.items()}
    estimators.update({name: f"set_{name}" for name in set_names})
    has_measured = any(c.startswith(f"{MEASURED_RESOURCES}.") for c in totals.columns)
    if has_measured:
        table = end2end_results_table(totals, estimators=estimators)
        (out_dir / "results_table.md").write_text(
            "# End2end estimation results per model set\n\n" + dataframe_to_markdown(table)
        )
        (out_dir / "results_table.tex").write_text(
            table.to_latex(float_format="%.1f", na_rep="--", multicolumn_format="c", escape=True)
        )
        for res in ("LUT", "DSP", "BRAM"):
            mean_relative_error_by_category(totals, res, estimators).to_csv(
                out_dir / f"estimation_error_{res}.csv"
            )
        plot_estimates_vs_measured(
            totals, str(out_dir / "estimates_vs_measured.png"), estimators=estimators
        )
        plot_mean_relative_error(
            totals, str(out_dir / "estimation_error.png"), estimators=estimators
        )
        logger.info("Results:\n%s", dataframe_to_markdown(table))
    else:
        logger.warning("No post_synth_resources reports found, design totals are not compared")
    power = power_table(totals, set_names)
    power.to_csv(out_dir / "power_table.csv")
    (out_dir / "power_table.md").write_text("# Power\n\n" + dataframe_to_markdown(power))

    summary = {
        "artifacts_dir": str(artifacts_dir),
        "runs": len(runs),
        "nodes": int(len(nodes)),
        "model_sets": {
            name: {
                "dir": path,
                "models": int((provenance["set"] == name).sum()) if not provenance.empty else 0,
                "nodes_with_lut_prediction": coverage[name],
            }
            for name, path in spec.items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Summary:\n%s", json.dumps(summary, indent=2))

    if args.persist_to:
        target = Path(args.persist_to)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(out_dir, target, dirs_exist_ok=True)
        logger.info("Persisted the report to %s", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())

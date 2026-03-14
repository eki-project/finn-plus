"""Collect and log benchmark results to DVC."""

import argparse
import collect_fn
import json
import matplotlib.pyplot as plt
import os
import shutil
import subprocess
import sys
import yaml
from datetime import date
from dvc.repo import Repo
from dvclive import Live


def delete_dir_contents(dir):
    """Delete all contents of a directory."""
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def open_json_report(id, report_name, is_followup=False):
    """Open JSON report from build or measurement artifacts."""
    # TODO: handle followup setting better
    # look in both, build & measurement, artifacts
    if is_followup:
        path1 = os.path.join(
            "build_artifacts_followup", "runs_output", "run_%d" % (id), "reports", report_name
        )
        path2 = os.path.join(
            "measurement_artifacts_followup", "runs_output", "run_%d" % (id), "reports", report_name
        )
    else:
        path1 = os.path.join(
            "build_artifacts", "runs_output", "run_%d" % (id), "reports", report_name
        )
        path2 = os.path.join(
            "measurement_artifacts", "runs_output", "run_%d" % (id), "reports", report_name
        )
    if os.path.isfile(path1):
        with open(path1, "r") as f:
            report = json.load(f)
        return report
    elif os.path.isfile(path2):
        with open(path2, "r") as f:
            report = json.load(f)
        return report
    else:
        return None


def generate_power_report(rails_names, experiment_reports_path):
    for filename in os.listdir(experiment_reports_path):
        power_measurements = [
            m
            for m in os.listdir(os.path.join(experiment_reports_path, filename))
            if m.startswith(filename) and m.endswith(".json")
        ]
        if power_measurements:
            per_file_averages = []
            for file in power_measurements:
                with open(os.path.join(experiment_reports_path, filename, file), "r") as f:
                    report = json.load(f)
                rails = report.get("rails", [])
                per_file_averages.append(
                    {
                        key: sum(r[key] for r in rails if key in r) / len(rails)
                        for key in rails_names
                    }
                )
                averaged = {
                    "avg_" + key: sum(a[key] for a in per_file_averages) / len(per_file_averages)
                    for key in rails_names
                }
                min_values = {
                    "min_" + key: min(a[key] for a in per_file_averages) for key in rails_names
                }
                max_values = {
                    "max_" + key: max(a[key] for a in per_file_averages) for key in rails_names
                }
                return {**averaged, **min_values, **max_values}


def generate_metric_plots(metric_reports, file):
    _bar_palette = ["steelblue", "orange", "green", "purple", "brown", "pink"]

    def _is_numeric_key(key):
        """Return True if any report has a numeric (int/float) value for this key."""
        for report in metric_reports.values():
            result = report["metrics"].get(key)
            if result is None:
                continue
            for k in ("current", "compare"):
                v = result.get(k)
                if v is not None:
                    return isinstance(v, (int, float))
        return False

    all_plot_keys = sorted(
        {
            key
            for report in metric_reports.values()
            for key, result in report["metrics"].items()
            if result.get("plot", False)
        }
    )

    numeric_plot_keys = [k for k in all_plot_keys if _is_numeric_key(k)]
    table_plot_keys = [k for k in all_plot_keys if not _is_numeric_key(k)]

    has_table = bool(table_plot_keys)

    duts = sorted(metric_reports.keys())
    n_rows = len(duts)
    n_cols = len(numeric_plot_keys) + (1 if has_table else 0)

    if n_rows == 0 or n_cols == 0:
        return

    col_widths = [4] * len(numeric_plot_keys) + ([6] if has_table else [])
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(sum(col_widths), 4 * n_rows),
        squeeze=False,
        gridspec_kw={"width_ratios": col_widths},
    )

    for row_idx, dut in enumerate(duts):
        report = metric_reports[dut]
        meta = report.get("meta", {})
        bar_entries = [(k, f"{v[0]} ({v[1]}) \n {v[2]}") for k, v in meta.items()]

        for col_idx, key in enumerate(numeric_plot_keys):
            ax = axes[row_idx][col_idx]
            result = report["metrics"].get(key)

            ax.set_title(key, fontsize=8, fontweight="bold")

            if result is None:
                ax.axis("off")
                continue

            status = result.get("status", "ok")
            delta = result.get("delta")
            uncertainty = result.get("allowed_uncertainty")

            labels = [label for _, label in bar_entries]
            values = [result.get(k) for k, _ in bar_entries]
            bar_vals = [v if v is not None else 0 for v in values]
            bar_colors = [_bar_palette[i % len(_bar_palette)] for i in range(len(bar_entries))]
            bars = ax.bar(labels, bar_vals, color=bar_colors)
            ax.tick_params(axis="x", labelsize=7)

            for i, (internal_key, _) in enumerate(bar_entries):
                if internal_key == "current" and status == "not ok":
                    bars[i].set_color("red")

            for bar, val in zip(bars, values):
                if val is not None:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val,
                        f"{val:.4g}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        0,
                        "N/A",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="red",
                        style="italic",
                    )

            delta_str = f"Δ={delta:.1%}" if isinstance(delta, float) else "Δ=N/A"
            unc_str = f" (±{uncertainty:.1%})" if uncertainty is not None else ""
            ax.set_xlabel(f"{delta_str}{unc_str}", fontsize=8)
            ax.set_ylabel(dut, fontsize=8)
            ax.grid(axis="y", alpha=0.3)

            valid_vals = [v for v in values if v is not None]
            if valid_vals:
                if max(abs(v) for v in valid_vals) > 0:
                    ax.set_ylim(0, max(valid_vals) * 1.2)

        if has_table:
            ax = axes[row_idx][-1]
            ax.axis("off")
            ax.set_title("Non-numeric Metrics", fontsize=9, fontweight="bold")

            col_labels = ["Metric"] + [label for _, label in bar_entries]
            cell_text = []
            cell_colors = []
            for key in table_plot_keys:
                result = report["metrics"].get(key)
                if result is None:
                    cell_text.append([key] + ["N/A"] * len(bar_entries))
                    cell_colors.append(["#e8e8e8"] + ["#f9f9f9"] * len(bar_entries))
                else:
                    status = result.get("status", "ok")
                    data_color = "#ffcccc" if status == "not ok" else "#f9f9f9"
                    cell_text.append(
                        [key]
                        + [
                            str(result.get(k)) if result.get(k) is not None else "N/A"
                            for k, _ in bar_entries
                        ]
                    )
                    cell_colors.append(["#e8e8e8"] + [data_color] * len(bar_entries))

            table = ax.table(
                cellText=cell_text,
                colLabels=col_labels,
                cellColours=cell_colors,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.auto_set_column_width(list(range(len(col_labels))))
            table.scale(1, 2.2)

            for col_idx in range(len(col_labels)):
                cell = table[0, col_idx]
                cell.set_facecolor("#4472c4")
                cell.set_text_props(color="white", fontweight="bold")

            ax.set_ylabel(dut, fontsize=8)

    fig.suptitle("Metric Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(file, dpi=150, bbox_inches="tight")
    plt.close()


# Wrapper around DVC Live object
class DVCLoggerHelper:
    """Wrapper around DVC Live for logging experiments."""

    def __init__(self, experiment_name, experiment_msg, id, params, is_followup=False):
        """Initialize DVC logger with experiment details."""
        self.id = id
        self.is_followup = is_followup
        self.experiment_name = experiment_name

        # extract logging settings from params
        self.store_as_experiment = params["params"].get("store_results_in_dvc_experiment", True)
        self.store_as_data = params["params"].get("store_results_in_dvc_data", False)

        if self.store_as_experiment:
            # Start DVC Live experiment session
            # TODO: cache images once we switch to a cache provider that works with DVC Studio
            self.live = Live(
                exp_name=experiment_name, exp_message=experiment_msg, cache_images=False
            )
        else:
            self.live = None

        if self.store_as_data:
            self.data_dict = dict()
        else:
            self.data_dict = None

        self.log_params(params)

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context manager and end DVC session."""
        if self.store_as_experiment:
            # End DVC Live experiment session
            self.live.end()

    def log_params(self, params):
        """Log parameters to DVC."""
        if self.store_as_experiment:
            self.live.log_params(params)
        if self.store_as_data:
            self.data_dict.update(params)

    def log_metric(self, prefix, name, value):
        """Log a single metric to DVC."""
        # sanitize '/' in name because DVC uses it to nest metrics (which we do via prefix)
        name = name.replace("/", "-")

        if self.store_as_experiment:
            self.live.log_metric(prefix + name, value, plot=False)
        if self.store_as_data:
            # store in nested dictionary structure based on prefix
            if "metrics" not in self.data_dict:
                self.data_dict["metrics"] = dict()
            _dict = self.data_dict["metrics"]

            for key in prefix.split("/"):
                if key:
                    if key not in _dict:
                        _dict[key] = dict()
                    _dict = _dict[key]
            _dict[name] = value

    def log_image(self, image_name, image_path):
        """Log an image artifact to DVC."""
        if self.store_as_experiment:
            self.live.log_image(image_name, image_path)

    def log_artifact(self, artifact_path):
        """Log an artifact to DVC."""
        if self.store_as_experiment:
            self.live.log_artifact(artifact_path)

    def log_all_metrics_from_report(self, report_name, prefix=""):
        """Log all metrics from a JSON report."""
        report = open_json_report(self.id, report_name, self.is_followup)
        if report:
            for key in report:
                self.log_metric(prefix, key, report[key])

    def log_metrics_from_report(self, report_name, keys, prefix=""):
        """Log specific metrics from a JSON report."""
        report = open_json_report(self.id, report_name, self.is_followup)
        if report:
            for key in keys:
                if key in report:
                    self.log_metric(prefix, key, report[key])

    def log_nested_metrics_from_report(self, report_name, key_top, keys, prefix=""):
        """Log nested metrics from a JSON report."""
        report = open_json_report(self.id, report_name, self.is_followup)
        if report:
            if key_top in report:
                for key in keys:
                    if key in report[key_top]:
                        self.log_metric(prefix, key, report[key_top][key])


class ExperimentMetricsLogger:
    def __init__(self, dvc_logger, collect_cfg_path):
        self.dvc_logger = dvc_logger
        with open(collect_cfg_path, "r") as f:
            self.collect_cfg = yaml.safe_load(f)

    def retrieve_experiment_reports(self, report_path):
        metrics = self.collect_cfg.get("Metrics", {})
        experiment_files = list(metrics.keys())
        exp_reports = {exp: [] for exp in experiment_files}

        for foldername in os.listdir(report_path):
            files = os.listdir(os.path.join(report_path, foldername))
            folders = [
                f
                for f in files
                if os.path.isdir(os.path.join(report_path, foldername, f))
                and f.startswith("exp_itr_")
            ]
            for exp_itr in folders:
                path = os.path.join(report_path, foldername, exp_itr)
                for file in os.listdir(path):
                    if file in experiment_files:
                        exp_reports[file].append(os.path.join(path, file))

        exp_reports = {
            k: list(map(lambda p: json.load(open(p)), v)) for k, v in exp_reports.items()
        }
        return exp_reports

    def _extract_fn_from_cfg(self, metric, path=None):
        if path is None:
            path = []
        cleaned = {}
        functions = []
        for key, value_list in metric.items():
            current_path = path + [key]
            cleaned_sub = []
            for item in value_list:
                if isinstance(item, dict) and list(item.keys()) == ["fn"]:
                    functions.append((current_path, item["fn"]))
                elif isinstance(item, str):
                    cleaned_sub.append(item)
                elif isinstance(item, dict):
                    sub_cleaned, sub_fns = self._extract_fn_from_cfg(item, current_path)
                    functions.extend(sub_fns)
                    for k, v in sub_cleaned.items():
                        cleaned_sub.append({k: v} if v else k)
            cleaned[key] = cleaned_sub
        return cleaned, functions

    def _filter_report(self, report, cleaned_metrics):
        filtered = {}
        for metric in cleaned_metrics:
            if isinstance(metric, str):
                if metric in report:
                    filtered[metric] = report[metric]
            elif isinstance(metric, dict):
                for key, sub_metrics in metric.items():
                    if key in report:
                        if sub_metrics:
                            sub_filtered = self._filter_report(report[key], sub_metrics)
                            if sub_filtered:
                                filtered[key] = sub_filtered
                        else:
                            filtered[key] = report[key]
        return filtered

    def _merge_reports_and_apply_fn(self, report_list, fn_map):
        if not report_list:
            return {}
        merged = {}
        all_keys = {k for r in report_list for k in r}
        for key in all_keys:
            values = [r[key] for r in report_list if key in r]
            fn_name = fn_map.get((key,))
            if fn_name:
                merged[key] = getattr(collect_fn, fn_name)(values)
            elif all(isinstance(v, dict) for v in values):
                sub_fn_map = {
                    path[1:]: fn for path, fn in fn_map.items() if len(path) > 1 and path[0] == key
                }
                merged[key] = self._merge_reports_and_apply_fn(values, sub_fn_map)
            else:
                # If no function, just use first value
                merged[key] = values[0]
        return merged

    def remove_ignored_metrics_and_apply_fn(self, exp_reports):
        metrics = self.collect_cfg.get("Metrics", {})
        result = {}
        for report_file, report_list in exp_reports.items():
            valid_metrics = metrics.get(report_file, [])
            all_functions = []
            cleaned_metrics = []
            for metric in valid_metrics:
                if isinstance(metric, str):
                    cleaned_metrics.append(metric)
                elif isinstance(metric, dict):
                    cleaned, functions = self._extract_fn_from_cfg(metric)
                    all_functions.extend(functions)
                    for k, v in cleaned.items():
                        cleaned_metrics.append({k: v} if v else k)

            filtered_list = [self._filter_report(r, cleaned_metrics) for r in report_list]
            fn_map = {tuple(path): fn_name for path, fn_name in all_functions}
            result[report_file] = self._merge_reports_and_apply_fn(filtered_list, fn_map)
        return result

    def _log_nested(self, data, prefix):
        for key, value in data.items():
            if isinstance(value, dict):
                self._log_nested(value, prefix + key + "/")
            else:
                self.dvc_logger.log_metric(prefix, key, value)

    def log_metrics(self, metrics, prefix="measurement/"):
        for report_file, report_data in metrics.items():
            file_stem = os.path.splitext(report_file)[0]
            self._log_nested(report_data, prefix + file_stem + "/")


class ExperimentComparator:
    def __init__(self, dvc_logger, collect_cfg_path):
        self.dvc_logger = dvc_logger
        with open(collect_cfg_path, "r") as f:
            self.collect_cfg = yaml.safe_load(f)

    def _get_experiment_data(self):
        tag = self.collect_cfg.get("Compare").get("compare_tag")
        git_remote = "git@github.com:eki-project/finn-plus.git"

        subprocess.run(
            ["git", "fetch", git_remote, "+refs/exps/*:refs/exps/*", "--prune"],
            check=True,
        )

        with Repo(".") as repo:
            remote_exp_map = repo.experiments.ls(git_remote=git_remote, rev=tag)
            remote_exps = set()
            for _, exps in remote_exp_map.items():
                remote_exps.update(a for a, b in exps)

            if not remote_exps:
                raise ValueError("No experiments found with tag %s" % tag)

            exp_data = {}
            for exp_state in repo.experiments.show(revs=tag):
                for exp_range in exp_state.experiments or []:
                    for exp_rev in exp_range.revs or []:
                        if exp_rev.name and exp_rev.data:
                            exp_data[exp_rev.name] = {
                                "metrics": exp_rev.data.metrics["dvclive/metrics.json"]["data"],
                                "params": exp_rev.data.params["dvclive/params.yaml"]["data"],
                                "date": exp_rev.data.timestamp,
                            }

            exp_data = {
                k: v
                for k, v in exp_data.items()
                if not (v.get("metrics", {}).get("status") == "failed")
            }

        return exp_data

    def _flatten_metrics_dict(self, d, prefix=""):
        flat = {}
        for k, v in d.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(self._flatten_metrics_dict(v, key))
            else:
                flat[key] = v
        return flat

    def aggregate_metrics_across_reports(self):
        experiment_data = self._get_experiment_data()

        current_metrics = self.dvc_logger.live.summary
        current_params = self.dvc_logger.live._params

        matching_exps = {}
        current_model_name = self.extract_model_name(current_params)

        for exp_name, exp_data in experiment_data.items():
            exp_model_name = self.extract_model_name(exp_data.get("params"))
            if exp_model_name == current_model_name:
                matching_exps[exp_name] = exp_data

        # Get newest matching experiment
        newest_exp = None
        newest_date = None
        if matching_exps:
            for exp_name, exp_data in matching_exps.items():
                exp_date = exp_data.get("date")
                if exp_date and (newest_date is None or exp_date > newest_date):
                    newest_date = exp_date
                    newest_exp = (exp_name, exp_data)
        else:
            raise ValueError(
                "No matching experiments found with model_name %s" % current_model_name
            )
            # TODO do some other error handling? So we dont interrupt every task?

        compare = {
            "name": "Comparison",
            "experiment_name": newest_exp[0],
            "dut": self.extract_model_name(newest_exp[1].get("params")),
            "metrics": self._flatten_metrics_dict(newest_exp[1].get("metrics")),
            "params": newest_exp[1].get("params"),
            "date": newest_exp[1].get("date"),
        }

        current = {
            "name": "Current",
            "experiment_name": self.dvc_logger.experiment_name,
            "dut": current_model_name,
            "metrics": self._flatten_metrics_dict(current_metrics),
            "params": current_params,
            "date": date.today(),
        }

        return [current, compare]

    def compare_metrics_across_reports(self, aggregated_metrics):
        compare_cfg = self.collect_cfg.get("Compare")
        global_uncertainty = compare_cfg.get("allowed_uncertainty", 0.0)
        metrics_cfg = compare_cfg.get("Metrics", {})
        current_entry = next((m for m in aggregated_metrics if m["name"] == "Current"), None)
        compare_entry = next((m for m in aggregated_metrics if m["name"] == "Comparison"), None)
        current = current_entry["metrics"]
        compare = compare_entry["metrics"]

        results = {}
        all_keys = set(current.keys()) | set(compare.keys())

        for key in sorted(all_keys):
            metric_cfg = metrics_cfg.get(key, {})
            if isinstance(metric_cfg, dict) and "allowed_uncertainty" in metric_cfg:
                uncertainty = metric_cfg["allowed_uncertainty"]
            else:
                uncertainty = global_uncertainty

            if uncertainty is None:
                continue

            is_required = isinstance(metric_cfg, dict) and metric_cfg.get("required", False)
            is_plotted = isinstance(metric_cfg, dict) and metric_cfg.get("plot", False)

            current_val = current.get(key)
            compare_val = compare.get(key)

            if current_val is None or compare_val is None:
                if is_required:
                    results[key] = {
                        "current": current_val,
                        "compare": compare_val,
                        "delta": None,
                        "allowed_uncertainty": uncertainty,
                        "status": "not ok",
                        "required": is_required,
                        "plot": is_plotted,
                    }
                continue

            if not isinstance(current_val, (int, float)) or not isinstance(
                compare_val, (int, float)
            ):
                results[key] = {
                    "current": current_val,
                    "compare": compare_val,
                    "delta": None,
                    "allowed_uncertainty": uncertainty,
                    "status": "ok" if current_val == compare_val else "not ok",
                    "required": is_required,
                    "plot": is_plotted,
                }
                continue

            current_num = float(current_val)
            compare_num = float(compare_val)

            if compare_num != 0:
                delta = (current_num - compare_num) / abs(compare_num)
            else:
                delta = 0.0 if current_num == 0 else float("inf")

            if abs(delta) <= uncertainty:
                status = "ok"
            else:
                status = "not ok"

            results[key] = {
                "current": current_num,
                "compare": compare_num,
                "delta": delta,
                "allowed_uncertainty": uncertainty,
                "status": status,
                "required": is_required,
                "plot": is_plotted,
            }

        return {
            "meta": {
                "current": (
                    current_entry["name"],
                    current_entry["date"].strftime("%Y-%m-%d"),
                    current_entry.get("experiment_name"),
                ),
                "compare": (
                    compare_entry["name"],
                    compare_entry["date"].strftime("%Y-%m-%d"),
                    compare_entry.get("experiment_name"),
                ),
            },
            "metrics": results,
        }

    def extract_model_name(self, metadata):
        dut_name = metadata.get("params").get("dut")
        if dut_name == "bnn-pynq":
            model_path = metadata.get("params", {}).get("model_path", "")
            if model_path:
                # Extract filename from path and remove suffix
                model_filename = os.path.basename(model_path)
                # Remove _qonnx.onnx or similar suffixes
                model_name = model_filename.replace("_qonnx.onnx", "").replace(".onnx", "")
                return model_name
        elif dut_name == "transformer":
            model_path = metadata.get("params", {}).get("model_path", "")
            if model_path and "finn-transformers/" in model_path:
                # Extract directory name after "finn-transformers/"
                # e.g., "models/transformer/finn-transformers/benchmark/streamlined.onnx"
                # -> "benchmark"
                parts = model_path.split("finn-transformers/")
                if len(parts) > 1:
                    # Get the next directory name
                    model_name = parts[1].split("/")[0]
                    return "T_" + model_name
        return dut_name


if __name__ == "__main__":
    """Go through all runs found in the artifacts and log their results to DVC."""
    os.environ["CI_PIPELINE_NAME"] = "DevRun"
    os.environ["CI_PIPELINE_ID"] = "99999999999"
    os.environ["CI_COMMIT_SHA"] = "99999999991"
    os.environ["LOCAL_BENCHMARK_DIR_STORE"] = "test"

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Collect and log benchmark results to DVC.")
    parser.add_argument(
        "--followup",
        action="store_true",
        help="Indicate this is a follow-up run (prevents generating new follow-up configs)",
    )
    args = parser.parse_args()

    if args.followup:
        run_dir_list = os.listdir(os.path.join("build_artifacts_followup", "runs_output"))
    else:
        run_dir_list = os.listdir(os.path.join("build_artifacts", "runs_output"))
    print("Looking for runs in build artifacts")
    run_ids = []
    for run_dir in run_dir_list:
        if run_dir.startswith("run_"):
            run_id = int(run_dir[4:])
            run_ids.append(run_id)
    run_ids.sort()
    print("Found %d runs" % len(run_ids))

    follow_up_bench_cfg = list()
    microbench_result_data = dict()
    metric_reports = dict()

    for id in run_ids:
        print("Processing run %d" % id)
        if args.followup:
            experiment_name = "CI_" + os.environ.get("CI_PIPELINE_ID") + "_followup_" + str(id)
        else:
            experiment_name = "CI_" + os.environ.get("CI_PIPELINE_ID") + "_" + str(id)
        experiment_msg = (
            "[CI] "
            + os.environ.get("CI_PIPELINE_NAME")
            + " ("
            + os.environ.get("CI_PIPELINE_ID")
            + "_"
            + str(id)
            + ")"
        )

        # initialize logging wrapper with input parameters logged by benchmarking infrastructure
        metadata_bench = open_json_report(id, "metadata_bench.json", args.followup)
        params = {"params": metadata_bench["params"]}
        with DVCLoggerHelper(
            experiment_name, experiment_msg, id, params, is_followup=args.followup
        ) as dvc_logger:
            # optional metadata logged by builder
            metadata_builder = open_json_report(id, "metadata_builder.json", args.followup)
            if metadata_builder:
                metadata = {
                    "metadata": {
                        "tool_version": metadata_builder["tool_version"],
                    }
                }
                dvc_logger.log_params(metadata)

            # optional dut_info.json (additional information generated during model generation)
            dut_info_report = open_json_report(id, "dut_info.json", args.followup)
            if dut_info_report:
                dut_info = {"dut_info": dut_info_report}
                dvc_logger.log_params(dut_info)

            # METRICS
            # TODO: make all logs consistent (at generation), e.g., BRAM vs BRAM18 vs BRAM36)

            # status
            status = metadata_bench["status"]
            if status == "ok":
                # mark as failed if either bench or builder indicates failure
                if metadata_builder:
                    status_builder = metadata_builder["status"]
                    if status_builder == "failed":
                        status = "failed"
            dvc_logger.log_metric("", "status", status)

            # verification steps
            if "output" in metadata_bench:
                if "builder_verification" in metadata_bench["output"]:
                    dvc_logger.log_metric(
                        "",
                        "verification",
                        metadata_bench["output"]["builder_verification"]["verification"],
                    )

            # estimate_layer_resources.json
            dvc_logger.log_nested_metrics_from_report(
                "estimate_layer_resources.json",
                "total",
                [
                    "LUT",
                    "DSP",
                    "BRAM_18K",
                    "URAM",
                ],
                prefix="estimate/resources/",
            )

            # estimate_layer_resources_hls.json
            dvc_logger.log_nested_metrics_from_report(
                "estimate_layer_resources_hls.json",
                "total",
                [
                    "LUT",
                    "FF",
                    "DSP",
                    "DSP48E",
                    "DSP58E",  # TODO: aggregate/unify DSP reporting
                    "BRAM_18K",
                    "URAM",
                ],
                prefix="hls_estimate/resources/",
            )

            # estimate_network_performance.json
            dvc_logger.log_metrics_from_report(
                "estimate_network_performance.json",
                [
                    "critical_path_cycles",
                    "max_cycles",
                    "max_cycles_node_name",
                    "estimated_throughput_fps",
                    "estimated_latency_ns",
                ],
                prefix="estimate/performance/",
            )

            # rtlsim_performance.json
            dvc_logger.log_metrics_from_report(
                "rtlsim_performance.json",
                [
                    "N",
                    "TIMEOUT",
                    "latency_cycles",
                    "cycles",
                    "fclk[mhz]",
                    "throughput[images/s]",
                    "stable_throughput[images/s]",
                    # add INPUT_DONE, OUTPUT_DONE, number transactions?
                ],
                prefix="rtlsim/performance/",
            )

            # fifo_sizing.json
            dvc_logger.log_metrics_from_report(
                "fifo_sizing.json", ["total_fifo_size_kB"], prefix="fifosizing/"
            )

            # stitched IP DCP synth resource report
            dvc_logger.log_nested_metrics_from_report(
                "post_synth_resources_dcp.json",
                "(top)",
                [
                    "LUT",
                    "FF",
                    "SRL",
                    "DSP",
                    "BRAM_18K",
                    "BRAM_36K",
                    "URAM",
                ],
                prefix="synth(dcp)/resources/",
            )

            # stitched IP DCP synth resource breakdown
            # TODO: generalize to all build flows and bitfile synth
            layer_categories = ["MAC", "Eltwise", "Thresholding", "FIFO", "DWC", "SWG", "Other"]
            for category in layer_categories:
                dvc_logger.log_nested_metrics_from_report(
                    "res_breakdown_build_output.json",
                    category,
                    [
                        "LUT",
                        "FF",
                        "SRL",
                        "DSP",
                        "BRAM_18K",
                        "BRAM_36K",
                        "URAM",
                    ],
                    prefix="synth(dcp)/resources(breakdown)/" + category + "/",
                )

            # ooc_synth_and_timing.json (OOC synth / step_out_of_context_synthesis)
            dvc_logger.log_metrics_from_report(
                "ooc_synth_and_timing.json",
                [
                    "LUT",
                    "LUTRAM",
                    "FF",
                    "DSP",
                    "BRAM",
                    "BRAM_18K",
                    "BRAM_36K",
                    "URAM",
                ],
                prefix="synth(ooc)/resources/",
            )
            dvc_logger.log_metrics_from_report(
                "ooc_synth_and_timing.json",
                [
                    "WNS",
                    "fmax_mhz",
                    # add TNS? what is "delay"?
                ],
                prefix="synth(ooc)/timing/",
            )

            # post_synth_resources.json (shell synth / step_synthesize_bitfile)
            # special handling for microbenchmarks to extract only the relevant layer
            report_hierarchy_level = "(top)"
            if metadata_bench["params"]["dut"] == "mvau":
                resource_report = open_json_report(id, "post_synth_resources.json", args.followup)
                if resource_report:
                    for key in resource_report:
                        if "MVAU" in key:
                            report_hierarchy_level = key
                            break
                    if report_hierarchy_level == "(top)":
                        print("ERROR: No MVAU found in post_synth_resources.json")
                        sys.exit(1)
            # TODO: also do this for other reports or make it optional/configurable

            dvc_logger.log_nested_metrics_from_report(
                "post_synth_resources.json",
                report_hierarchy_level,
                [
                    "LUT",
                    "FF",
                    "SRL",
                    "DSP",
                    "BRAM_18K",
                    "BRAM_36K",
                    "URAM",
                ],
                prefix="synth/resources/",
            )

            # post synth timing report
            # TODO: only exported as post_route_timing.rpt, not .json

            # power estimation
            dvc_logger.log_all_metrics_from_report(
                "power_estimate_summary.json", prefix="vivado_estimate/power/"
            )

            # power measurement
            experiment_reports_path = os.path.join(
                "measurement_artifacts", "runs_output", "run_%d" % (id), "reports"
            )
            power = generate_power_report(
                ["0V85_power", "3V3_power", "total_power"], experiment_reports_path
            )
            for name, value in power.items():
                dvc_logger.log_metric(prefix="measurement/power/", name=name, value=value)

            # measurement metric logging
            collect_cfg_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "collect.yaml"
            )
            exp = ExperimentMetricsLogger(dvc_logger, collect_cfg_path)
            exp_reports = exp.retrieve_experiment_reports(experiment_reports_path)
            exp_metrics = exp.remove_ignored_metrics_and_apply_fn(exp_reports)
            exp.log_metrics(exp_metrics)

            # live fifosizing graph png
            image = os.path.join(
                "measurement_artifacts",
                "runs_output",
                "run_%d" % (id),
                "reports",
                "fifo_sizing_graph.png",
            )
            if os.path.isfile(image):
                dvc_logger.log_image("fifosizing_pass_1", image)

            # time_per_step.json
            dvc_logger.log_all_metrics_from_report("time_per_step.json", prefix="time/")

            # ARTIFACTS
            # Log build reports as they come from GitLab artifacts,
            # but copy them to a central dir first so all runs share the same path
            if args.followup:
                run_report_dir1 = os.path.join(
                    "build_artifacts_followup", "runs_output", "run_%d" % (id), "reports"
                )
                run_report_dir2 = os.path.join(
                    "measurement_artifacts_followup", "runs_output", "run_%d" % (id), "reports"
                )
            else:
                run_report_dir1 = os.path.join(
                    "build_artifacts", "runs_output", "run_%d" % (id), "reports"
                )
                run_report_dir2 = os.path.join(
                    "measurement_artifacts", "runs_output", "run_%d" % (id), "reports"
                )
            dvc_report_dir = "reports"
            os.makedirs(dvc_report_dir, exist_ok=True)
            delete_dir_contents(dvc_report_dir)
            if os.path.isdir(run_report_dir1):
                shutil.copytree(run_report_dir1, dvc_report_dir, dirs_exist_ok=True)
            if os.path.isdir(run_report_dir2):
                shutil.copytree(run_report_dir2, dvc_report_dir, dirs_exist_ok=True)
            dvc_logger.log_artifact(dvc_report_dir)

            # Save microbenchmark results in a list per DUT for later aggregation
            dut = params["params"]["dut"]
            if dut not in microbench_result_data:
                # Initialize data dict for this DUT
                microbench_result_data[dut] = list()
            microbench_result_data[dut].append(dvc_logger.data_dict)

        # Prepare benchmarking config for follow-up runs after live FIFO-sizing
        # Only generate follow-up config if this is not already a follow-up run
        if not args.followup:
            folding_config_lfs_path = os.path.join(
                "measurement_artifacts",
                "runs_output",
                "run_%d" % (id),
                "reports",
                "experiment_fifosizing",
                "exp_itr_1",
                "largest_first",  # TODO: make configurable or choose best available FIFO-sizing
                "both",
                "folding_config_lfs.json",
            )
            if os.path.isfile(folding_config_lfs_path):
                print(
                    "Creating follow-up experiment config based on lfs folding config: %s"
                    % folding_config_lfs_path
                )

                # Create benchmarking config
                metadata_bench = open_json_report(id, "metadata_bench.json", args.followup)
                configuration = dict()
                for key in metadata_bench["params"]:
                    # wrap in list
                    configuration[key] = [metadata_bench["params"][key]]
                # overwrite FIFO-related params
                configuration["live_fifo_sizing"] = [False]
                configuration["auto_fifo_depths"] = [False]
                configuration["target_fps"] = ["None"]
                configuration["folding_config_file"] = [folding_config_lfs_path]

                # Exception for ResNet-50: Final model doesn't fit board used for FIFO-sizing
                if "dut" in metadata_bench["params"]:
                    if metadata_bench["params"]["dut"] == "resnet50":
                        configuration["board"] = ["U250"]
                        configuration["enable_instrumentation"] = [False]
                        configuration["rtlsim_batch_size"] = [3]
                        configuration["generate_outputs"] = [
                            ["stitched_ip", "rtlsim_performance", "bitfile"]
                        ]

                follow_up_bench_cfg.append(configuration)

            collect_cfg_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "collect.yaml"
            )
            comp = ExperimentComparator(dvc_logger, collect_cfg_path)
            metrics = comp.aggregate_metrics_across_reports()
            report = comp.compare_metrics_across_reports(metrics)
            metric_reports[metrics[0].get("dut")] = report

    # Save microbenchmark results as (DVC-tracked? TODO) JSON for each DUT
    for dut in microbench_result_data:
        if None not in microbench_result_data[dut]:
            # dut_dir = os.path.join("ci", "benchmark_data", dut) TODO
            dut_dir = os.path.join(os.environ.get("LOCAL_BENCHMARK_DIR_STORE"), dut)
            os.makedirs(dut_dir, exist_ok=True)
            dut_json_path = os.path.join(
                dut_dir,
                date.today().strftime("%Y-%m-%d")
                + "_"
                + os.environ.get("CI_COMMIT_SHORT_SHA")
                + "_"
                + os.environ.get("CI_PIPELINE_ID")
                + "_"
                + str(len(microbench_result_data[dut]))
                + ".json",
            )
            dut_json = {
                "dut": dut,
                "date": date.today().strftime("%Y-%m-%d"),
                "commit": os.environ.get("CI_COMMIT_SHA"),
                "pipeline_id": os.environ.get("CI_PIPELINE_ID"),
                "pipeline_name": os.environ.get("CI_PIPELINE_NAME"),
                "runs": microbench_result_data[dut],
            }
            print("Saving microbenchmark results for %s to %s" % (dut, dut_json_path))
            with open(dut_json_path, "w") as f:
                json.dump(dut_json, f, indent=2)

    # Save aggregated benchmarking config for follow-up job to working dir
    # It is forwarded to the follow-up job via GitLab CI artifact
    if follow_up_bench_cfg:
        followup_artifact_path = "followup_bench_config.json"
        print("Saving follow-up bench config as artifact: %s" % followup_artifact_path)
        with open(followup_artifact_path, "w") as f:
            json.dump(follow_up_bench_cfg, f, indent=2)

    # Save metric comparison report as JSON
    metric_artifact_path = "metric_report.json"
    print("Saving metric report as artifact: %s" % metric_artifact_path)
    with open(metric_artifact_path, "w") as f:
        json.dump(metric_reports, f, indent=2)

    # Plot comparisons
    generate_metric_plots(metric_reports, "metric_report_plots.png")

    # Fail collect if any required metric is not ok
    fail = False
    for dut, report in metric_reports.items():
        for metric, result in report["metrics"].items():
            if result.get("required") and result.get("status") != "ok":
                fail = True
                print("Required metric %s for DUT %s is not ok: %s" % (metric, dut, result))

    if fail:
        print("One or more required metrics are not ok, failing collect")
        sys.exit(1)

    print("Done")

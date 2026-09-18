"""Plots and tables for evaluating empirical QoR estimators.

Used by the CI scripts in ``ci/qor/``; nothing in here is needed at build time. Figures are
always written to files (never shown), so the functions work headless.

This module must stay importable without FINN/QONNX or scikit-learn (pandas, numpy,
matplotlib and jinja2 for LaTeX export only), because the end2end report CI job does not
install scikit-learn.
"""

import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    # only for annotations, keeps scikit-learn optional for the end2end report script
    from finn.qor.estimator import SelectionResult

# Figures are only ever written to files, never shown
matplotlib.use("Agg")

logger = logging.getLogger(__name__)

#: Operator type categories (category -> substrings matched against node names/op types) used
#: to summarize per-layer resource reports of end2end builds.
OPTYPE_CATEGORIES: dict[str, list[str]] = {
    "MVAU": ["MVAU"],
    "VVAU": ["VVAU"],
    "Eltwise": ["Elementwise"],
    "Thresholding": ["Thresholding"],
    "FIFO": ["StreamingFIFO"],
    "DWC": ["DataWidthConverter"],
    "SWG": ["ConvolutionInputGenerator"],
    "Pool": ["Pool"],
}

#: Report prefixes (column name prefixes in the end2end DataFrame) of the resource estimators
#: that are compared against post-synthesis results.
RESOURCE_ESTIMATORS: dict[str, str] = {
    "Empirical": "estimate_layer_resources_empirical",
    "FINN": "estimate_layer_resources",
    "HLS": "estimate_layer_resources_hls",
}
MEASURED_RESOURCES = "post_synth_resources"


# ---------------------------------------------------------------------------------------------
# Regressor selection / learning curve
# ---------------------------------------------------------------------------------------------


def plot_regressor_comparison(result: "SelectionResult", out_path: str) -> None:
    """Bar chart of the cross-validated scores (best vs. worst parameter set, error bars =
    min/max fold) and box plots of the out-of-fold per-sample absolute percentage and absolute
    errors (the latter is the readable one for zero-inflated targets such as DSP/BRAM/URAM)."""
    scores = result.scores.dropna(subset=["mean_score"])
    names = list(scores.index)
    x = np.arange(len(names))
    with_timing = {"fit_time_s", "single_predict_ms"} <= set(scores.columns)

    fig, axes = plt.subplots(1, 4 if with_timing else 3, figsize=(26 if with_timing else 20, 5))
    ax1, ax2, ax3 = axes[:3]
    best_err = [
        scores["mean_score"] - scores["min_fold_score"],
        scores["max_fold_score"] - scores["mean_score"],
    ]
    worst_err = [
        scores["worst_mean_score"] - scores["worst_min_fold_score"],
        scores["worst_max_fold_score"] - scores["worst_mean_score"],
    ]
    ax1.bar(x - 0.2, scores["mean_score"], 0.4, yerr=best_err, capsize=4, label="Best params")
    ax1.bar(
        x + 0.2, scores["worst_mean_score"], 0.4, yerr=worst_err, capsize=4, label="Worst params"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.set_ylabel(f"Mean CV {result.scoring}")
    ax1.set_title("Cross-validated score (error bars: min/max fold)")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    box_data = [result.sample_errors[n]["abs_pct_error"].dropna().values for n in names]
    ax2.boxplot(box_data, tick_labels=names, showmeans=True, whis=1.5)
    ax2.set_ylabel("Absolute percentage error (%)")
    ax2.set_ylim(-10, 100)
    ax2.set_title("Out-of-fold per-sample error")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, axis="y", alpha=0.3)

    abs_data = [result.sample_errors[n]["abs_error"].dropna().values for n in names]
    ax3.boxplot(abs_data, tick_labels=names, showmeans=True, whis=1.5)
    ax3.set_yscale("symlog")
    ax3.set_ylabel("Absolute error")
    ax3.set_title("Out-of-fold per-sample absolute error")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, axis="y", alpha=0.3)

    if with_timing:
        ax4 = axes[3]
        ax4.bar(x - 0.2, scores["fit_time_s"].clip(lower=1e-4), 0.4, label="Fit time (s)")
        ax4.bar(
            x + 0.2,
            (scores["single_predict_ms"] / 1e3).clip(lower=1e-6),
            0.4,
            label="Single prediction (s)",
        )
        ax4.set_yscale("log")
        ax4.set_xticks(x)
        ax4.set_xticklabels(names, rotation=45, ha="right")
        ax4.set_title("Cost (best parameter set)")
        ax4.legend()
        ax4.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Regressor comparison, best: {result.best_name}")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve(curve: pd.DataFrame, out_path: str) -> None:
    """Line plot of score vs. fraction of training data, one line per regressor."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in curve.columns:
        series = curve[name].dropna()
        if not series.empty:
            ax.plot(series.index, series.values, marker="o", label=name)
    ax.set_xlabel("Fraction of training data")
    scoring = curve.attrs.get("scoring", "score")
    evaluation = curve.attrs.get("evaluation", "")
    ax.set_ylabel(f"{scoring} ({evaluation})")
    ax.set_title(
        f"Regressor performance vs. training data size ({curve.attrs.get('n_train')} samples)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Estimator accuracy tables
# ---------------------------------------------------------------------------------------------


#: Metrics reported by :func:`error_metrics`, in table order
ERROR_METRICS = ("RMSE", "MAE", "MAPE", "n", "n_zero")


def error_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """RMSE, MAE, MAPE (excluding actual == 0), sample count and number of zero targets over
    the jointly valid entries. MAPE is NaN if no target is nonzero (zero-inflated resources
    such as DSP/URAM), so MAE is the metric to look at there."""
    valid = ~(actual.isna() | predicted.isna())
    a, p = actual[valid].astype(float), predicted[valid].astype(float)
    if len(a) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "n": 0, "n_zero": 0}
    nonzero = a != 0
    mape = (
        float(np.mean(np.abs((p[nonzero] - a[nonzero]) / a[nonzero])) * 100)
        if nonzero.any()
        else np.nan
    )
    return {
        "RMSE": float(np.sqrt(np.mean((p - a) ** 2))),
        "MAE": float(np.mean(np.abs(p - a))),
        "MAPE": mape,
        "n": int(len(a)),
        "n_zero": int((~nonzero).sum()),
    }


def estimation_error_table(
    df: pd.DataFrame,
    actual_col: str,
    estimate_cols: dict[str, str],
    subsets: Optional[dict[str, pd.Series]] = None,
) -> pd.DataFrame:
    """Compare estimators (rows) on data subsets (columns) by RMSE, MAE and MAPE.

    Args:
        df: Samples with actual and estimated values as columns.
        actual_col: Column holding the measured/synthesized reference value.
        estimate_cols: ``{estimator label: column}``; missing columns give NaN rows.
        subsets: ``{subset label: boolean mask}``; default is the whole DataFrame plus, if
            present, the HLS/RTL backends and unstructured-sparse samples of the
            microbenchmark database.

    Returns a DataFrame with a two-level column index ``(subset, metric)``.
    """
    if subsets is None:
        subsets = {"All": pd.Series(True, index=df.index)}
        if "params.backend" in df.columns:
            subsets["HLS"] = df["params.backend"] == "hls"
            subsets["RTL"] = df["params.backend"] == "rtl"
        if "params.sparsity_type" in df.columns:
            subsets["Sparse"] = df["params.sparsity_type"] == "unstructured"
    columns = pd.MultiIndex.from_product([subsets.keys(), list(ERROR_METRICS)])
    table = pd.DataFrame(index=list(estimate_cols.keys()), columns=columns, dtype=float)
    for label, col in estimate_cols.items():
        for subset, mask in subsets.items():
            if col not in df.columns:
                table.loc[label, (subset, "n")] = 0
                continue
            sub = df[mask]
            for k, v in error_metrics(sub[actual_col], sub[col]).items():
                table.loc[label, (subset, k)] = v
    for subset in subsets:
        for count in ("n", "n_zero"):
            table[(subset, count)] = table[(subset, count)].fillna(0).astype(int)
    return table


def dataframe_to_markdown(df: pd.DataFrame, float_fmt: str = "{:.1f}") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no tabulate dependency)."""
    flat = df.copy()
    if isinstance(flat.columns, pd.MultiIndex):
        flat.columns = [" ".join(str(x) for x in col) for col in flat.columns]

    def fmt(v: Any) -> str:
        """Format floats with ``float_fmt`` (NaN as '-'), everything else via str()."""
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else float_fmt.format(v)
        return str(v)

    header = [str(flat.index.name or "")] + [str(c) for c in flat.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for idx in flat.index:
        cells = [str(idx)] + [fmt(flat.at[idx, c]) for c in flat.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# End2end evaluation (per-model reports of complete accelerator builds)
# ---------------------------------------------------------------------------------------------


def _aggregate_resources(src: dict[str, Any], dst: dict[str, float]) -> None:
    """Add a per-layer resource dict into a category total, normalizing key variants."""
    for k, v in src.items():
        if not isinstance(v, (int, float)):
            continue
        if k in ("DSP48E", "DSP58E", "DSP58E1", "DSP58E2"):
            k = "DSP"
        if k == "BRAM_18K":
            k = "BRAM"
        elif k == "BRAM_36K":
            k, v = "BRAM", v * 2
        dst[k] = dst.get(k, 0) + v


def breakdown_resources(
    report: dict[str, dict[str, Any]], categories: Optional[dict[str, list[str]]] = None
) -> Optional[dict[str, dict[str, float]]]:
    """Summarize a per-layer resource report into operator categories.

    ``report`` maps node names (plus ``"total"`` or ``"(top)"``) to resource dicts, as in
    ``estimate_layer_resources.json`` or ``post_synth_resources.json``. Nodes whose name
    contains none of the category patterns go to ``"Other"``. BRAM_36K counts as two BRAM_18K,
    DSP variants are merged into ``DSP``.

    Post-synthesis reports carry a ``"(top)"`` entry (whole design incl. shell and
    cross-hierarchy optimizations) which is kept as ``"Top"``, while ``"Total"`` is always the
    sum of the categories. Returns None if the report has neither ``"total"`` nor ``"(top)"``.
    """
    categories = categories or OPTYPE_CATEGORIES
    totals: dict[str, dict[str, float]] = {cat: {} for cat in categories}
    totals["Other"] = {}
    totals["Total"] = {}
    for node_name, resources in report.items():
        if node_name in ("(top)", "total"):
            continue
        for cat, patterns in categories.items():
            if any(p in node_name for p in patterns):
                _aggregate_resources(resources, totals[cat])
                break
        else:
            _aggregate_resources(resources, totals["Other"])

    if "(top)" in report:
        totals["Top"] = {}
        _aggregate_resources(report["(top)"], totals["Top"])
        for cat in list(categories) + ["Other"]:
            _aggregate_resources(totals[cat], totals["Total"])
    elif "total" in report:
        _aggregate_resources(report["total"], totals["Total"])
    else:
        logger.error("Resource report has neither '(top)' nor 'total' entry")
        return None
    return totals


def plot_resource_breakdown(
    df: pd.DataFrame,
    out_path: str,
    prefix: str = MEASURED_RESOURCES,
    resources: tuple[str, ...] = ("LUT", "DSP", "BRAM", "URAM"),
    categories: Optional[dict[str, list[str]]] = None,
) -> None:
    """One row of bar charts per model: resources per operator category, annotated with the
    share of the model total."""
    cats = list(categories or OPTYPE_CATEGORIES) + ["Other"]
    fig, axes = plt.subplots(
        len(df), len(resources), figsize=(4 * len(resources), 3 * len(df)), squeeze=False
    )
    for i, (_, row) in enumerate(df.iterrows()):
        for j, res in enumerate(resources):
            ax = axes[i, j]
            values = [row.get(f"{prefix}.{cat}.{res}", 0) or 0 for cat in cats]
            total = row.get(f"{prefix}.Total.{res}", 0) or 0
            bars = ax.bar(np.arange(len(cats)), values, width=0.5)
            ax.set_xticks(np.arange(len(cats)))
            ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{row.get('model_name', i)} - {res}", fontsize=9)
            ax.grid(True, axis="y", linewidth=0.5)
            if total:
                for bar, val in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{100 * val / total:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
                ax.set_ylim(top=ax.get_ylim()[1] * 1.1)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_estimates_vs_measured(
    df: pd.DataFrame,
    out_path: str,
    category: str = "Total",
    resources: tuple[str, ...] = ("LUT", "DSP", "BRAM"),
    estimators: Optional[dict[str, str]] = None,
) -> None:
    """Per model and resource: bars for the measured value and each estimator's estimate of one
    operator category (default: the whole design)."""
    estimators = {"Measured": MEASURED_RESOURCES, **(estimators or RESOURCE_ESTIMATORS)}
    fig, axes = plt.subplots(
        len(df), len(resources), figsize=(3.5 * len(resources), 2.4 * len(df)), squeeze=False
    )
    labels = list(estimators)
    for i, (_, row) in enumerate(df.iterrows()):
        for j, res in enumerate(resources):
            ax = axes[i, j]
            values = [row.get(f"{p}.{category}.{res}", np.nan) for p in estimators.values()]
            ax.bar(np.arange(len(labels)), values, width=0.5)
            ax.set_title(f"{row.get('model_name', i)} - {category} {res}", fontsize=9)
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(True, axis="y", linewidth=0.5)
    fig.tight_layout(pad=1.0)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def mean_relative_error_by_category(
    df: pd.DataFrame,
    resource: str,
    estimators: Optional[dict[str, str]] = None,
    categories: Optional[dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """Mean absolute relative error of each estimator (columns) per operator category (rows)
    for one resource type, averaged over all models with a non-zero measured value."""
    estimators = estimators or RESOURCE_ESTIMATORS
    cats = list(categories or OPTYPE_CATEGORIES) + ["Other", "Total"]
    table = pd.DataFrame(index=cats, columns=list(estimators), dtype=float)
    for cat in cats:
        measured = df.get(f"{MEASURED_RESOURCES}.{cat}.{resource}")
        if measured is None:
            continue
        measured = measured.astype(float)
        for label, prefix in estimators.items():
            est = df.get(f"{prefix}.{cat}.{resource}")
            if est is None:
                continue
            est = est.astype(float)
            valid = (measured != 0) & measured.notna() & est.notna()
            if valid.any():
                table.loc[cat, label] = float(
                    np.mean(np.abs((est[valid] - measured[valid]) / measured[valid]))
                )
    return table


def plot_mean_relative_error(
    df: pd.DataFrame,
    out_path: str,
    resources: tuple[str, ...] = ("LUT", "DSP", "BRAM"),
    estimators: Optional[dict[str, str]] = None,
) -> None:
    """Grouped bar chart per resource type: mean relative estimation error per operator
    category for each estimator."""
    estimators = estimators or RESOURCE_ESTIMATORS
    fig, axes = plt.subplots(1, len(resources), figsize=(6 * len(resources), 5), squeeze=False)
    for ax, res in zip(axes[0], resources):
        table = mean_relative_error_by_category(df, res, estimators)
        x = np.arange(len(table))
        width = 0.8 / len(estimators)
        for k, label in enumerate(table.columns):
            ax.bar(x + k * width, table[label].values, width=width, label=label)
        ax.set_xticks(x + width * (len(estimators) - 1) / 2)
        ax.set_xticklabels(table.index, rotation=45, ha="right")
        ax.set_ylabel("Mean absolute relative error")
        ax.set_title(f"Estimation error per operator category - {res}")
        ax.legend()
        ax.grid(True, axis="y", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def end2end_results_table(
    df: pd.DataFrame,
    categories: tuple[str, ...] = ("Total", "MVAU"),
    resource: str = "LUT",
    estimators: Optional[dict[str, str]] = None,
    measured_power_col: str = "measured_power.avg_0V85_power",
    estimated_power_col: str = "estimate_power_empirical.Total.power",
    power_unit: str = "mW",
) -> pd.DataFrame:
    """Per-model summary: measured resource and relative error of each estimator for the given
    categories, plus measured and estimated power (in the unit of the measurement reports).
    Two-level columns ``(group, quantity)``."""
    estimators = estimators or RESOURCE_ESTIMATORS
    rows = []
    for _, row in df.iterrows():
        entry: dict[tuple[str, str], Any] = {}
        for cat in categories:
            group = f"{cat} {resource}"
            actual = row.get(f"{MEASURED_RESOURCES}.{cat}.{resource}", np.nan)
            entry[(group, "Actual")] = actual
            for label, prefix in estimators.items():
                est = row.get(f"{prefix}.{cat}.{resource}", np.nan)
                err = (est - actual) / actual * 100 if actual and pd.notna(est) else np.nan
                entry[(group, f"{label} err. %")] = err
        entry[(f"Power [{power_unit}]", "Measured")] = row.get(measured_power_col, np.nan)
        entry[(f"Power [{power_unit}]", "Estimated")] = row.get(estimated_power_col, np.nan)
        rows.append(pd.Series(entry, name=row.get("model_name")))
    table = pd.DataFrame(rows)
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    table.index.name = "Model"
    return table


def symbolic_equation_markdown(equation: dict[str, Any]) -> str:
    """Markdown rendering of a symbolic regression result (as stored in the model sidecar
    under ``equation``): variables, sympy form, LaTeX and the Pareto front."""
    lines = [
        "# Symbolic regression result",
        "",
        f"Variables: {', '.join(equation.get('variables', []))}",
        "",
        "Selected expression (complexity {}, loss {:.4g}):".format(
            equation.get("complexity", "?"), float(equation.get("loss", float("nan")))
        ),
        "",
        "```",
        str(equation.get("sympy", "")),
        "```",
        "",
        "$$" + str(equation.get("latex", "")) + "$$",
        "",
        "PySR form: `" + str(equation.get("pysr", "")) + "`",
        "",
    ]
    front = equation.get("pareto_front") or []
    if front:
        lines += ["## Pareto front", "", dataframe_to_markdown(pd.DataFrame(front), "{:.4g}")]
    return "\n".join(lines) + "\n"


def write_symbolic_equation_tex(equation: dict[str, Any], out_path: str) -> None:
    """Write the LaTeX form of a symbolic regression result (for inclusion in a paper)."""
    with open(out_path, "w") as f:
        f.write(str(equation.get("latex", "")) + "\n")


def plot_pareto_front(equation: dict[str, Any], out_path: str) -> None:
    """Loss vs. complexity of all expressions found by a symbolic regression run, with the
    selected expression highlighted."""
    front = pd.DataFrame(equation.get("pareto_front") or [])
    if front.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(front["complexity"], front["loss"], marker="o", label="Pareto front")
    selected = front[front["complexity"] == equation.get("complexity")]
    if not selected.empty:
        ax.plot(
            selected["complexity"], selected["loss"], marker="*", ms=16, ls="", label="Selected"
        )
    ax.set_yscale("log")
    ax.set_xlabel("Expression complexity")
    ax.set_ylabel("Loss")
    ax.set_title("Symbolic regression Pareto front")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

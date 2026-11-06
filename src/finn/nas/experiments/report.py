"""Figures and tables of the NAS experiments (``finn nas report``).

The experiment drivers write JSON lines and summaries; this module turns them into the
figures and markdown tables of the write-up. It only reads the experiment directories (a
running experiment can be rendered on its partial records; a finished one is re-rendered by
re-running the command) and never needs the compiler, so it stays importable on a plain
Python with matplotlib. Search figures live in :mod:`finn.nas.experiments.report_search`.

Every figure is written as PDF (vector, for the paper) and PNG (200 dpi, for quick viewing)
with a fixed palette and sizes that read at single-column width. All helpers used by both
modules (style, loading, tables) are defined here.
"""

from __future__ import annotations

import json
import math
import matplotlib
from collections import Counter
from dataclasses import dataclass, field
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from pathlib import Path
from typing import TYPE_CHECKING, Any

from finn.nas.evaluate import CandidateEvaluation
from finn.nas.result import FailureCategory

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from matplotlib.axes import Axes

# --- style --------------------------------------------------------------------------------

#: Categorical series colors in their fixed, colorblind-validated order. Slots are assigned to
#: an entity (a variant, a failure category) once and never cycled; more than eight entities
#: fold into the muted "other" color.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BACKGROUND_POINTS = "#c3c2b7"
#: One-hue sequential ramp (light to dark) for a magnitude such as the parameter count.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "nas_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
#: Marker shapes paired with the series colors so identity never rests on color alone.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

#: Single-column (3.4 in) and double-column figure sizes in inches.
SINGLE = (3.4, 2.5)
DOUBLE = (7.0, 2.6)
PNG_DPI = 200

RC = {
    "font.size": 8,
    "font.family": "sans-serif",
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "lines.linewidth": 1.3,
    "lines.markersize": 4,
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "text.color": INK,
}


def figure(
    size: tuple[float, float] = SINGLE, ncols: int = 1, nrows: int = 1
) -> tuple[Figure, Any]:
    """Create a styled figure with ``nrows x ncols`` axes (a single ``Axes`` for 1x1).

    A :class:`~matplotlib.figure.Figure` is built directly rather than through pyplot so that
    rendering many figures (and the tests) never touches global pyplot state or a GUI backend.
    Callers run inside :func:`styled`; the rc context is entered here as well because the
    artists created later read the rc parameters at creation time.
    """
    with matplotlib.rc_context(RC):
        fig = Figure(figsize=size, layout="constrained")
        axes = fig.subplots(nrows=nrows, ncols=ncols)
    return fig, axes


def save_figure(fig: Figure, out_dir: Path, name: str) -> list[Path]:
    """Write ``name``.pdf and ``name``.png (200 dpi) into ``out_dir`` and return both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with matplotlib.rc_context(RC):
        for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": PNG_DPI})):
            path = out_dir / f"{name}{suffix}"
            fig.savefig(path, bbox_inches="tight", **kwargs)
            paths.append(path)
    return paths


def empty_axes(ax: Axes, message: str) -> None:
    """Show ``message`` instead of marks when a figure has no data (zero feasible candidates)."""
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def label_bar_ends(ax: Axes, bars: Any, labels: Iterable[str], color: str = INK_SECONDARY) -> None:
    """Direct labels after the end of horizontal bars (text ink, never the series color)."""
    for bar, label in zip(bars, labels, strict=True):
        ax.annotate(
            label,
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.5,
            color=color,
        )


def legend_below(fig: Figure, ax: Axes, ncols: int = 2) -> None:
    """Put the legend of ``ax`` underneath the axes so it never covers marks.

    Lines that reach both the top and the bottom of a single-column panel (feasibility rates
    between 10 % and 90 %) leave no free corner inside the plot.
    """
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncols=ncols)


#: Readable names of the size statistics and of estimates used as scatter axes.
LABELS = {"estimated_lut": "estimated LUTs", "params": "parameters", "flops": "FLOPs"}


def pretty(name: str) -> str:
    """Human-readable label of a snake_case identifier."""
    return LABELS.get(name, name.replace("_", " "))


def assign_colors(names: Sequence[str], fixed: dict[str, int] | None = None) -> dict[str, str]:
    """Give every name a series color in fixed order; names past the eighth slot get ``MUTED``.

    ``fixed`` pins known names to slots so that, e.g., the search variants keep their color no
    matter which subset of them is rendered (color follows the entity, not its position).
    """
    colors: dict[str, str] = {}
    used: set[int] = set()
    for name in names:
        if fixed and name in fixed and fixed[name] < len(SERIES):
            colors[name] = SERIES[fixed[name]]
            used.add(fixed[name])
    for name in names:
        if name in colors:
            continue
        free = [i for i in range(len(SERIES)) if i not in used]
        if free:
            colors[name] = SERIES[free[0]]
            used.add(free[0])
        else:
            colors[name] = MUTED
    return colors


#: Fixed slot of each failure category (enum order, ``NONE`` excluded) so a category has the
#: same color in every figure of a report.
CATEGORY_SLOTS = {
    c.value: i for i, c in enumerate(c for c in FailureCategory if c != FailureCategory.NONE)
}


def category_colors(categories: Sequence[str]) -> dict[str, str]:
    """Colors of the failure categories present, in enum order (at most eight distinct)."""
    ordered = sorted(set(categories), key=lambda c: CATEGORY_SLOTS.get(c, len(CATEGORY_SLOTS)))
    return assign_colors(ordered)


# --- loading ------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON lines, skipping lines that do not parse.

    A running experiment appends to its records file; a line caught mid-write (only ever the
    last one) must not make the whole report fail, so unparsable lines are dropped.
    """
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file, ``None`` if it does not exist."""
    return json.loads(path.read_text()) if path.exists() else None


@dataclass
class SamplingData:
    """A sampling run: its records and the summary (recomputed if the file is missing/stale)."""

    directory: Path
    records: list[CandidateEvaluation]
    summary: dict[str, Any]
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def feasible(self) -> list[CandidateEvaluation]:
        """The feasible records."""
        return [r for r in self.records if r.result.feasible]


def load_sampling(directory: str | Path) -> SamplingData:
    """Load a sampling directory (``samples.jsonl``, ``summary.json``, ``config.json``)."""
    from finn.nas.experiments.sampling import RECORDS_FILE, SUMMARY_FILE, summarize

    directory = Path(directory)
    records = [CandidateEvaluation.from_dict(d) for d in read_jsonl(directory / RECORDS_FILE)]
    summary = read_json(directory / SUMMARY_FILE)
    if summary is None or summary.get("count") != len(records):
        summary = summarize(records)
    return SamplingData(directory, records, summary, read_json(directory / "config.json") or {})


def finite(value: Any) -> bool:
    """Tell whether ``value`` is a real, finite number (``None``/NaN/inf/strings are not)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# --- markdown -----------------------------------------------------------------------------


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Format rows as a GitHub-flavoured markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines) + "\n"


def fmt(value: Any, digits: int = 3) -> str:
    """Format a number for a table (``n/a`` for missing or non-finite values)."""
    if not finite(value):
        return "n/a"
    if isinstance(value, int) or (float(value).is_integer() and abs(value) >= 1000):
        return f"{int(value):,}"
    # Counts and estimates (LUTs, fps) read best as plain integers with thousands separators;
    # small quantities (seconds, correlations, p-values) keep ``digits`` significant digits.
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:.{digits}g}"


def percent(numerator: float, denominator: float) -> str:
    """``numerator / denominator`` as a percentage string."""
    return f"{100.0 * numerator / denominator:.1f} %" if denominator else "n/a"


# --- sampling report ----------------------------------------------------------------------


def _median(values: Sequence[float]) -> float:
    """Median of a non-empty sequence (upper median, as the sampling summary does)."""
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else float("nan")


def fig_failure_histogram(data: SamplingData, out_dir: Path) -> list[Path]:
    """Horizontal bars of the failure categories, the feasible count as its own bar."""
    counts = Counter(data.summary.get("failure_category", {}))
    n = data.summary.get("count", len(data.records))
    labels = ["feasible"] + [c for c, _ in counts.most_common()]
    values = [data.summary.get("feasible", 0)] + [counts[c] for c in labels[1:]]
    colors = [SERIES[2]] + [SERIES[0]] * len(counts)
    with matplotlib.rc_context(RC):
        fig, ax = figure((SINGLE[0], 0.9 + 0.24 * len(labels)))
        if not n:
            empty_axes(ax, "no samples")
            return save_figure(fig, out_dir, "sampling_failure_histogram")
        y = list(range(len(labels)))[::-1]
        bars = ax.barh(y, values, color=colors, height=0.72)
        ax.set_yticks(y, [pretty(label) for label in labels])
        ax.set_xlabel(f"samples (of {n})")
        ax.grid(False, axis="y")
        ax.set_xlim(0, max(values) * 1.3)
        label_bar_ends(ax, bars, [f"{v} ({100 * v / n:.1f} %)" for v in values])
        ax.set_title("Oracle verdict of random samples", loc="left")
    return save_figure(fig, out_dir, "sampling_failure_histogram")


def _op_type_counts(records: Iterable[CandidateEvaluation]) -> dict[str, Counter[str]]:
    """Failure categories per blamed op type of the infeasible records."""
    counts: dict[str, Counter[str]] = {}
    for record in records:
        result = record.result
        if result.feasible:
            continue
        op_type = result.failure_op_type or "(none)"
        counts.setdefault(op_type, Counter())[result.failure_category.value] += 1
    return counts


def fig_failure_op_types(data: SamplingData, out_dir: Path, top: int = 12) -> list[Path]:
    """Draw the op types blamed most often, stacked by failure category."""
    counts = _op_type_counts(data.records)
    ranked = sorted(counts, key=lambda op: -sum(counts[op].values()))[:top]
    categories = sorted(
        {c for op in ranked for c in counts[op]},
        key=lambda c: CATEGORY_SLOTS.get(c, len(CATEGORY_SLOTS)),
    )
    colors = category_colors(categories)
    with matplotlib.rc_context(RC):
        fig, ax = figure((SINGLE[0], 1.1 + 0.22 * max(1, len(ranked))))
        if not ranked:
            empty_axes(ax, "no infeasible samples")
            return save_figure(fig, out_dir, "sampling_failure_op_types")
        y = list(range(len(ranked)))[::-1]
        left = [0.0] * len(ranked)
        for category in categories:
            widths = [counts[op].get(category, 0) for op in ranked]
            ax.barh(
                y,
                widths,
                left=left,
                color=colors[category],
                height=0.72,
                label=pretty(category),
                edgecolor="white",
                linewidth=0.5,
            )
            left = [a + b for a, b in zip(left, widths, strict=True)]
        ax.set_yticks(y, ranked)
        ax.set_xlabel("infeasible samples")
        ax.grid(False, axis="y")
        ax.set_xlim(0, max(left) * 1.18)
        bars = ax.barh(y, left, color="none", height=0.72)
        label_bar_ends(ax, bars, [str(int(v)) for v in left])
        ax.legend(loc="lower right", title="failure category", title_fontsize=7)
        ax.set_title(f"Blamed op types (top {len(ranked)})", loc="left")
    return save_figure(fig, out_dir, "sampling_failure_op_types")


def binned_rates(
    records: Sequence[CandidateEvaluation], key: str, width: float, start: float = 0.0
) -> list[tuple[float, int, int]]:
    """Feasibility per bin of ``nodes`` or ``log10(params)``: (bin start, samples, feasible)."""
    bins: dict[int, list[int]] = {}
    for record in records:
        value = math.log10(max(1, record.params)) if key == "params" else float(record.nodes)
        index = math.floor((value - start) / width)
        bins.setdefault(index, [0, 0])
        bins[index][0] += 1
        bins[index][1] += int(record.result.feasible)
    return [(start + i * width, n, f) for i, (n, f) in sorted(bins.items())]


def _rate_bars(ax: Axes, rates: list[tuple[float, int, int]], width: float, xlabel: str) -> None:
    """Feasibility-rate bars per bin with the number of samples above each bar."""
    if not rates:
        empty_axes(ax, "no samples")
        return
    centers = [s + width / 2 for s, _, _ in rates]
    values = [100.0 * f / n for _, n, f in rates]
    ax.bar(centers, values, width=width * 0.86, color=SERIES[0])
    for c, v, (_, n, _) in zip(centers, values, rates, strict=True):
        ax.annotate(
            f"n={n}",
            (c, v),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.5,
            color=MUTED,
            rotation=90 if len(rates) > 9 else 0,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("feasible (%)")
    ax.set_ylim(0, max([*values, 1.0]) * (1.45 if len(rates) > 9 else 1.25))
    ax.grid(False, axis="x")


def fig_feasibility_vs_size(data: SamplingData, out_dir: Path) -> list[Path]:
    """Feasibility rate per node-count bin and per log10(parameter) bin."""
    with matplotlib.rc_context(RC):
        fig, (ax_nodes, ax_params) = figure(DOUBLE, ncols=2)
        _rate_bars(ax_nodes, binned_rates(data.records, "nodes", 4.0), 4.0, "nodes in the network")
        _rate_bars(ax_params, binned_rates(data.records, "params", 0.5), 0.5, "log10(parameters)")
        ax_nodes.set_title("Feasibility vs. depth", loc="left")
        ax_params.set_title("Feasibility vs. size", loc="left")
    return save_figure(fig, out_dir, "sampling_feasibility_vs_size")


def fig_estimates(data: SamplingData, out_dir: Path) -> list[Path]:
    """LUT vs. estimated fps (color = parameters) and BRAM vs. LUT of the feasible samples."""
    feasible = [
        r
        for r in data.feasible
        if finite(r.result.resources_total.get("LUT")) and finite(r.result.est_fps)
    ]
    paths = []
    with matplotlib.rc_context(RC):
        fig, ax = figure()
        if not feasible:
            empty_axes(ax, "no feasible samples")
        else:
            lut = [r.result.resources_total["LUT"] for r in feasible]
            fps = [r.result.est_fps for r in feasible]
            params = [math.log10(max(1, r.params)) for r in feasible]
            sc = ax.scatter(
                lut, fps, c=params, cmap=SEQUENTIAL, s=14, edgecolors="white", linewidths=0.4
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("estimated LUTs")
            ax.set_ylabel("estimated throughput (frames/s)")
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label("log10(parameters)")
            cbar.outline.set_visible(False)
        ax.set_title(f"Feasible samples (n={len(feasible)})", loc="left")
        paths += save_figure(fig, out_dir, "sampling_estimates_lut_fps")

        fig, ax = figure()
        if not feasible:
            empty_axes(ax, "no feasible samples")
        else:
            bram = [r.result.resources_total.get("BRAM_18K", 0.0) for r in feasible]
            ax.scatter(lut, bram, color=SERIES[0], s=14, edgecolors="white", linewidths=0.4)
            ax.set_xscale("log")
            ax.set_xlabel("estimated LUTs")
            ax.set_ylabel("estimated BRAM (18K blocks)")
        ax.set_title("Memory vs. logic of the feasible samples", loc="left")
        paths += save_figure(fig, out_dir, "sampling_estimates_bram_lut")
    return paths


def proxy_availability(records: Sequence[CandidateEvaluation]) -> dict[str, tuple[int, int]]:
    """Per proxy: (candidates with a finite score, candidates with proxies computed)."""
    counts: dict[str, list[int]] = {}
    for record in records:
        if not record.proxies:
            continue
        scores = record.proxies.get("scores", {})
        errors = record.proxies.get("errors", {})
        for name in set(scores) | set(errors):
            counts.setdefault(name, [0, 0])
            counts[name][1] += 1
            counts[name][0] += int(name not in errors and finite(scores.get(name)))
    return {k: (v[0], v[1]) for k, v in counts.items()}


def fig_proxy_availability(data: SamplingData, out_dir: Path) -> list[Path]:
    """Fraction of the feasible candidates with a usable value per zero-cost proxy."""
    availability = proxy_availability(data.feasible)
    names = sorted(
        availability, key=lambda k: (-availability[k][0] / max(1, availability[k][1]), k)
    )
    with matplotlib.rc_context(RC):
        fig, ax = figure((SINGLE[0], 0.9 + 0.2 * max(1, len(names))))
        if not names:
            empty_axes(ax, "no proxies computed for feasible samples")
            return save_figure(fig, out_dir, "sampling_proxy_availability")
        y = list(range(len(names)))[::-1]
        fractions = [100.0 * availability[k][0] / availability[k][1] for k in names]
        bars = ax.barh(y, fractions, color=SERIES[0], height=0.7)
        ax.set_yticks(y, names)
        ax.set_xlim(0, 128)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("feasible candidates with a finite score (%)")
        ax.grid(False, axis="y")
        label_bar_ends(ax, bars, [f"{availability[k][0]}/{availability[k][1]}" for k in names])
        ax.set_title("Zero-cost proxy availability", loc="left")
    return save_figure(fig, out_dir, "sampling_proxy_availability")


def sampling_table(data: SamplingData) -> str:
    """Headline numbers of a sampling run as markdown."""
    s = data.summary
    n = s.get("count", 0)
    feasible = data.feasible
    luts = [r.result.resources_total.get("LUT") for r in feasible]
    fps = [r.result.est_fps for r in feasible]
    rows = [
        ("samples", fmt(n)),
        ("feasible", f"{s.get('feasible', 0)} ({percent(s.get('feasible', 0), n)})"),
        ("mean seconds per candidate", fmt(s.get("mean_seconds"))),
        (
            "mean nodes (all / feasible)",
            f"{fmt(s.get('mean_nodes'))} / {fmt(s.get('mean_nodes_feasible'))}",
        ),
        ("median LUTs of feasible", fmt(_median([v for v in luts if finite(v)]))),
        ("median est. fps of feasible", fmt(_median([v for v in fps if finite(v)]))),
    ]
    failures = Counter(s.get("failure_category", {}))
    n_fail = sum(failures.values())
    text = f"## Sampling: {data.directory.name}\n\n" + markdown_table(("quantity", "value"), rows)
    if failures:
        text += "\nDominant failure causes:\n\n" + markdown_table(
            ("failure category", "count", "% of samples", "% of failures"),
            [(pretty(c), v, percent(v, n), percent(v, n_fail)) for c, v in failures.most_common()],
        )
    ops = Counter(s.get("failure_op_type", {}))
    if ops:
        text += (
            "\nMost blamed op types: "
            + ", ".join(f"{op} ({v}, {percent(v, n_fail)})" for op, v in ops.most_common(5))
            + "\n"
        )
    return text


def render_sampling(directory: str | Path, out_dir: str | Path) -> list[Path]:
    """Render all sampling figures and ``sampling_summary.md`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_sampling(directory)
    paths = []
    paths += fig_failure_histogram(data, out_dir)
    paths += fig_failure_op_types(data, out_dir)
    paths += fig_feasibility_vs_size(data, out_dir)
    paths += fig_estimates(data, out_dir)
    paths += fig_proxy_availability(data, out_dir)
    table = out_dir / "sampling_summary.md"
    table.write_text(sampling_table(data))
    return [*paths, table]


# --- ground-truth report ------------------------------------------------------------------


@dataclass
class GroundTruthData:
    """A ground-truth run: the training rows and the correlation summary."""

    directory: Path
    rows: list[dict[str, Any]]
    summary: dict[str, Any]

    @property
    def usable(self) -> list[dict[str, Any]]:
        """Rows whose training finished without error."""
        return [r for r in self.rows if r.get("train", {}).get("error") is None]


def load_groundtruth(directory: str | Path) -> GroundTruthData:
    """Load ``training.jsonl`` and ``summary.json`` (recomputed while the run is in progress)."""
    from finn.nas.experiments.ground_truth import SUMMARY_FILE, TRAINING_FILE, rank_correlations

    directory = Path(directory)
    rows = read_jsonl(directory / TRAINING_FILE)
    usable = [r for r in rows if r.get("train", {}).get("error") is None]
    summary = read_json(directory / SUMMARY_FILE)
    if summary is None or summary.get("usable") != len(usable):
        summary = rank_correlations(usable) if len(usable) >= 3 else {"count": len(usable)}
        summary["trained"] = len(rows)
        summary["usable"] = len(usable)
    summary.setdefault("proxies", {})
    return GroundTruthData(directory, rows, summary)


#: Size statistics that ``rank_correlations`` adds next to the real proxies.
SIZE_STATISTICS = ("params", "estimated_lut")


def ranked_proxies(data: GroundTruthData) -> list[str]:
    """Names with a Spearman value, by |Spearman| descending."""
    proxies = data.summary.get("proxies", {})
    return sorted(
        (k for k, v in proxies.items() if finite(v.get("spearman"))),
        key=lambda k: -abs(proxies[k]["spearman"]),
    )


def fig_correlations(data: GroundTruthData, out_dir: Path) -> list[Path]:
    """Spearman and Kendall correlation of every proxy with the best validation accuracy."""
    proxies = data.summary.get("proxies", {})
    names = sorted(
        (k for k, v in proxies.items() if finite(v.get("spearman"))),
        key=lambda k: proxies[k]["spearman"],
    )
    with matplotlib.rc_context(RC):
        fig, ax = figure((SINGLE[0], 1.0 + 0.24 * max(1, len(names))))
        if not names:
            empty_axes(ax, f"too few trained candidates ({data.summary.get('usable', 0)})")
            return save_figure(fig, out_dir, "groundtruth_correlations")
        y = list(range(len(names)))
        h = 0.36
        ax.barh(
            [v + h / 2 for v in y],
            [proxies[k]["spearman"] for k in names],
            height=h,
            color=SERIES[0],
            label="Spearman \u03c1",
        )
        ax.barh(
            [v - h / 2 for v in y],
            [proxies[k]["kendall"] for k in names],
            height=h,
            color=SERIES[1],
            label="Kendall \u03c4",
        )
        ax.set_yticks(y, [pretty(k) for k in names])
        ax.axvline(0, color=AXIS, linewidth=0.6)
        ax.set_xlim(-1.08, 1.4)
        ax.set_xlabel("rank correlation with best validation accuracy")
        ax.grid(False, axis="y")
        for v, k in zip(y, names, strict=True):
            ax.annotate(
                f"n={proxies[k]['n']}", (1.1, v), ha="left", va="center", fontsize=6, color=MUTED
            )
        legend_below(fig, ax)
        ax.set_title("Proxy vs. trained accuracy", loc="left")
    return save_figure(fig, out_dir, "groundtruth_correlations")


def _series(rows: list[dict[str, Any]], name: str) -> list[tuple[float, float]]:
    """(x, best accuracy) pairs of a proxy or size statistic with finite values."""
    pairs = []
    for row in rows:
        if name == "params":
            x = row.get("params")
        elif name == "estimated_lut":
            x = row.get("estimates", {}).get("LUT")
        else:
            x = row.get("proxies", {}).get(name)
        y = row.get("train", {}).get("best_val_acc")
        if finite(x) and finite(y):
            pairs.append((float(x), float(y)))
    return pairs


def _x_scale(xs: Sequence[float]) -> str:
    """Axis scale for proxy values: log over decades, symlog for heavy tails of either sign.

    Gradient-based proxies (``grasp``, ``synflow``) spread over orders of magnitude with a few
    extreme values; on a linear axis the bulk collapses into one column and the rank
    correlation in the title cannot be seen.
    """
    magnitudes = sorted(abs(x) for x in xs)
    if not magnitudes or magnitudes[-1] == 0:
        return "linear"
    if min(xs) > 0:
        return "log" if magnitudes[-1] / magnitudes[0] > 10 else "linear"
    median = magnitudes[len(magnitudes) // 2]
    return "symlog" if median > 0 and magnitudes[-1] / median > 50 else "linear"


def _scatter_vs_accuracy(
    ax: Axes, pairs: list[tuple[float, float]], name: str, stats: dict[str, Any]
) -> None:
    """Accuracy over a proxy value; log x when the values are positive and span decades."""
    if not pairs:
        empty_axes(ax, "no data")
        ax.set_title(pretty(name), loc="left")
        return
    xs, ys = zip(*pairs, strict=True)
    ax.scatter(xs, [100 * v for v in ys], color=SERIES[0], s=12, edgecolors="white", linewidths=0.4)
    ax.set_xscale(_x_scale(xs))
    ax.set_xlabel(pretty(name))
    ax.set_ylabel("best val. accuracy (%)")
    rho = stats.get("spearman")
    title = (
        f"{pretty(name)} (\u03c1={round(rho, 2) + 0.0:.2f}, n={len(pairs)})"
        if finite(rho)
        else pretty(name)
    )
    ax.set_title(title, loc="left")


def fig_proxy_scatter(data: GroundTruthData, out_dir: Path) -> list[Path]:
    """Draw the three best proxies and the parameter count against the trained accuracy."""
    proxies = data.summary.get("proxies", {})
    best = [k for k in ranked_proxies(data) if k not in SIZE_STATISTICS][:3]
    names = [*best, "params"]
    with matplotlib.rc_context(RC):
        fig, axes = figure((DOUBLE[0], 4.6), ncols=2, nrows=2)
        flat = list(axes.flat)
        for ax, name in zip(flat, names, strict=False):
            _scatter_vs_accuracy(ax, _series(data.usable, name), name, proxies.get(name, {}))
        for ax in flat[len(names) :]:
            empty_axes(ax, "fewer than three proxies correlated")
    return save_figure(fig, out_dir, "groundtruth_proxy_scatter")


def fig_accuracy(data: GroundTruthData, out_dir: Path) -> list[Path]:
    """Distribution of the trained accuracies and accuracy against the estimated LUTs."""
    accuracies = [
        100 * r["train"]["best_val_acc"]
        for r in data.usable
        if finite(r.get("train", {}).get("best_val_acc"))
    ]
    with matplotlib.rc_context(RC):
        fig, (ax_hist, ax_lut) = figure(DOUBLE, ncols=2)
        if not accuracies:
            empty_axes(ax_hist, "no trained candidates")
            empty_axes(ax_lut, "no trained candidates")
        else:
            lo, hi = 5 * math.floor(min(accuracies) / 5), 5 * math.ceil(max(accuracies) / 5)
            edges = [lo + 5 * i for i in range(int((hi - lo) / 5) + 1)] if hi > lo else 5
            ax_hist.hist(accuracies, bins=edges, color=SERIES[0], edgecolor="white", linewidth=0.5)
            ax_hist.set_xlabel("best validation accuracy (%)")
            ax_hist.set_ylabel("candidates")
            ax_hist.grid(False, axis="x")
            _scatter_vs_accuracy(
                ax_lut,
                _series(data.usable, "estimated_lut"),
                "estimated_lut",
                data.summary.get("proxies", {}).get("estimated_lut", {}),
            )
        ax_hist.set_title(f"Trained candidates (n={len(accuracies)})", loc="left")
    return save_figure(fig, out_dir, "groundtruth_accuracy")


def groundtruth_table(data: GroundTruthData) -> str:
    """Correlation table of a ground-truth run as markdown."""
    proxies = data.summary.get("proxies", {})
    names = ranked_proxies(data) + sorted(
        k for k, v in proxies.items() if not finite(v.get("spearman"))
    )
    accuracies = [
        r["train"]["best_val_acc"]
        for r in data.usable
        if finite(r.get("train", {}).get("best_val_acc"))
    ]
    head = markdown_table(
        ("quantity", "value"),
        [
            (
                "trained / usable",
                (
                    f"{data.summary.get('trained', len(data.rows))} / "
                    f"{data.summary.get('usable', len(data.usable))}"
                ),
            ),
            ("target", data.summary.get("target", "best_val_acc")),
            ("median best val. accuracy", percent(_median(accuracies), 1) if accuracies else "n/a"),
        ],
    )
    rows = [
        (
            pretty(k),
            proxies[k].get("n", 0),
            fmt(proxies[k].get("spearman")),
            fmt(proxies[k].get("spearman_p"), 2),
            fmt(proxies[k].get("kendall")),
            fmt(proxies[k].get("kendall_p"), 2),
        )
        for k in names
    ]
    text = f"## Ground truth: {data.directory.name}\n\n" + head
    if rows:
        text += "\n" + markdown_table(
            ("proxy", "n", "Spearman \u03c1", "p", "Kendall \u03c4", "p"), rows
        )
    return text


def render_groundtruth(directory: str | Path, out_dir: str | Path) -> list[Path]:
    """Render all ground-truth figures and ``groundtruth_summary.md`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_groundtruth(directory)
    paths = []
    paths += fig_correlations(data, out_dir)
    paths += fig_proxy_scatter(data, out_dir)
    paths += fig_accuracy(data, out_dir)
    table = out_dir / "groundtruth_summary.md"
    table.write_text(groundtruth_table(data))
    return [*paths, table]


# --- search report and the command --------------------------------------------------------


def render_search(
    directories: Sequence[str | Path], out_dir: str | Path, baseline: str | Path | None = None
) -> list[Path]:
    """Render the search figures of one or more variants (see ``report_search``)."""
    # Imported here: ``report_search`` uses the helpers above, so a module-level import
    # would be circular.
    from finn.nas.experiments.report_search import render_search_dirs

    return render_search_dirs(directories, out_dir, baseline)


def render_all(
    out_dir: str | Path,
    sampling: str | Path | None = None,
    search: Sequence[str | Path] = (),
    groundtruth: str | Path | None = None,
) -> list[Path]:
    """Render every report whose input is given and combine the tables into ``summary.md``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if sampling is not None:
        paths += render_sampling(sampling, out_dir)
    if search:
        paths += render_search(search, out_dir, baseline=sampling)
    if groundtruth is not None:
        paths += render_groundtruth(groundtruth, out_dir)
    tables = [p.read_text() for p in paths if p.suffix == ".md"]
    combined = out_dir / "summary.md"
    combined.write_text("# NAS experiment report\n\n" + "\n".join(tables))
    return [*paths, combined]

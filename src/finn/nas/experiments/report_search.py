"""Figures and the table of the search experiments (the variants of ``finn nas search``).

The variants are compared on one set of axes per figure: feasibility per generation (does
feedback let the search learn FINN's constraints?), distinct feasible candidates over the
evaluation budget, hypervolume history, the final Pareto fronts and the mean violation. A
variant is identified by its directory name (``search_`` stripped); its color and marker are
pinned by name so that a figure with a subset of the variants keeps the same encoding. The
random-sampling baseline, when given, is drawn as the reference every variant must beat.
"""

from __future__ import annotations

import math
import matplotlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from finn.nas.evaluate import CandidateEvaluation
from finn.nas.experiments.report import (
    DOUBLE,
    MARKERS,
    MUTED,
    RC,
    SERIES,
    SINGLE,
    SamplingData,
    assign_colors,
    empty_axes,
    figure,
    finite,
    fmt,
    legend_below,
    load_sampling,
    markdown_table,
    percent,
    pretty,
    read_json,
    read_jsonl,
    save_figure,
)
from finn.nas.search import SearchConfig, is_sentinel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

#: Fixed series slots of the standard variants; others take the next free slot in call order.
VARIANT_SLOTS = {"nsga2_graded": 0, "nsga2_binary": 1, "nsga2_discard": 2, "random": 3}


@dataclass
class SearchData:
    """One search run: its name, configuration, records and summary."""

    name: str
    directory: Path
    config: SearchConfig
    records: list[CandidateEvaluation]
    summary: dict[str, Any]

    @property
    def label(self) -> str:
        """Legend label: the variant name plus how infeasible candidates are handled."""
        if self.config.algorithm == "random":
            return f"{pretty(self.name)} (no feedback)"
        if self.config.feasibility == "discard":
            return f"{pretty(self.name)} (discard)"
        return f"{pretty(self.name)} ({self.config.violation} violation)"


def variant_name(directory: Path) -> str:
    """Variant name of a search directory (``search_nsga2_graded`` -> ``nsga2_graded``)."""
    name = directory.name
    return name[len("search_") :] if name.startswith("search_") else name


def load_search(directory: str | Path) -> SearchData:
    """Load a search directory; the summary is recomputed when missing or behind the records."""
    from finn.nas.experiments.search import RECORDS_FILE, SUMMARY_FILE, summarize_search

    directory = Path(directory)
    raw_config = read_json(directory / "config.json") or {}
    config = (
        SearchConfig.from_dict(raw_config["search"]) if "search" in raw_config else SearchConfig()
    )
    records = [CandidateEvaluation.from_dict(d) for d in read_jsonl(directory / RECORDS_FILE)]
    records = [r for r in records if "generation" in r.meta]
    summary = read_json(directory / SUMMARY_FILE)
    if summary is None or summary.get("count") != len(records):
        summary = summarize_search(config, records)
    return SearchData(variant_name(directory), directory, config, records, summary)


def variant_styles(runs: Sequence[SearchData]) -> dict[str, tuple[str, str]]:
    """(color, marker) per variant name, pinned for the standard variants."""
    colors = assign_colors([r.name for r in runs], VARIANT_SLOTS)
    styles = {}
    for run in runs:
        slot = SERIES.index(colors[run.name]) if colors[run.name] in SERIES else len(MARKERS) - 1
        styles[run.name] = (colors[run.name], MARKERS[slot])
    return styles


def baseline_rate(baseline: SamplingData | None) -> float | None:
    """Feasibility rate of the random-sampling baseline, if given and non-empty."""
    if baseline is None or not baseline.records:
        return None
    return sum(r.result.feasible for r in baseline.records) / len(baseline.records)


def cumulative_distinct_feasible(records: Sequence[CandidateEvaluation]) -> list[int]:
    """Distinct feasible genotypes after each record (records in trial order)."""
    seen: set[str] = set()
    out = []
    for record in sorted(records, key=lambda r: r.meta.get("trial", 0)):
        if record.result.feasible:
            seen.add(record.hash)
        out.append(len(seen))
    return out


def _plot_per_generation(
    ax: Axes,
    runs: Sequence[SearchData],
    key: str,
    styles: dict[str, tuple[str, str]],
    scale: float = 1.0,
) -> bool:
    """One line per variant of a per-generation summary quantity; False if nothing was drawn."""
    drawn = False
    for run in runs:
        generations = run.summary.get("generations", [])
        xs = [g["generation"] for g in generations if finite(g.get(key))]
        ys = [scale * g[key] for g in generations if finite(g.get(key))]
        if not xs:
            continue
        color, marker = styles[run.name]
        ax.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            markersize=3.2,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=run.label,
        )
        drawn = True
    return drawn


def fig_feasibility_per_generation(
    runs: Sequence[SearchData],
    out_dir: Path,
    baseline: SamplingData | None,
    styles: dict[str, tuple[str, str]],
) -> list[Path]:
    """Feasibility rate per generation and variant, with the sampling rate as reference."""
    with matplotlib.rc_context(RC):
        fig, ax = figure((SINGLE[0], 2.7))
        if not _plot_per_generation(ax, runs, "feasibility_rate", styles, 100.0):
            empty_axes(ax, "no generations recorded")
            return save_figure(fig, out_dir, "search_feasibility_per_generation")
        rate = baseline_rate(baseline)
        if rate is not None:
            ax.axhline(
                100 * rate,
                color=MUTED,
                linestyle="--",
                linewidth=0.9,
                label=f"random sampling ({100 * rate:.1f} %)",
            )
        ax.set_xlabel("generation")
        ax.set_ylabel("feasible trials (%)")
        ax.set_ylim(0, 102)
        ax.xaxis.get_major_locator().set_params(integer=True)
        legend_below(fig, ax)
        ax.set_title("Feasibility per generation", loc="left")
    return save_figure(fig, out_dir, "search_feasibility_per_generation")


def fig_cumulative_feasible(
    runs: Sequence[SearchData],
    out_dir: Path,
    baseline: SamplingData | None,
    styles: dict[str, tuple[str, str]],
) -> list[Path]:
    """Distinct feasible genotypes found vs. trials spent (duplicates and failures included)."""
    with matplotlib.rc_context(RC):
        fig, ax = figure((SINGLE[0], 2.7))
        drawn = False
        for run in runs:
            if not run.records:
                continue
            ys = cumulative_distinct_feasible(run.records)
            color, marker = styles[run.name]
            ax.plot(
                range(1, len(ys) + 1),
                ys,
                color=color,
                label=run.label,
                marker=marker,
                markevery=max(1, len(ys) // 8),
                markersize=3.2,
                markeredgecolor="white",
                markeredgewidth=0.4,
            )
            drawn = True
        if baseline is not None and baseline.records:
            ys = cumulative_distinct_feasible(baseline.records)
            ax.plot(
                range(1, len(ys) + 1),
                ys,
                color=MUTED,
                linestyle="--",
                linewidth=0.9,
                label="random sampling",
            )
            drawn = True
        if not drawn:
            empty_axes(ax, "no records")
            return save_figure(fig, out_dir, "search_cumulative_feasible")
        ax.set_xlabel("trials")
        ax.set_ylabel("distinct feasible candidates")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        legend_below(fig, ax)
        ax.set_title("Feasible candidates found", loc="left")
    return save_figure(fig, out_dir, "search_cumulative_feasible")


def _plot_hypervolume(
    ax: Axes, runs: Sequence[SearchData], styles: dict[str, tuple[str, str]]
) -> bool:
    """Draw the hypervolume histories of ``runs`` (one proxy) on ``ax``; False if none."""
    histories = [
        (run, run.summary.get("hypervolume", {}).get("per_generation_2d", []))
        for run in runs
        if run.summary.get("hypervolume", {}).get("per_generation_2d")
    ]
    if not histories:
        empty_axes(ax, "no hypervolume history")
        return False
    # Hypervolumes are products of a proxy score and a LUT count (1e6..1e8): scale them
    # to a readable axis instead of matplotlib's detached "1e7" offset text.
    largest = max(max(ys) for _, ys in histories)
    exponent = math.floor(math.log10(largest)) if largest > 0 else 0
    scale = 10.0**exponent if exponent >= 3 else 1.0
    for run, ys in histories:
        color, marker = styles[run.name]
        ax.plot(
            range(len(ys)),
            [y / scale for y in ys],
            color=color,
            marker=marker,
            markersize=3.2,
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=run.label,
        )
    pair = histories[0][0].summary["hypervolume"].get("objectives_2d", [])
    unit = f" \u00d7 10$^{{{exponent}}}$" if scale != 1.0 else ""
    ax.set_xlabel("generation")
    ax.set_ylabel(f"hypervolume ({' \u00d7 '.join(pair)}{unit})" if pair else f"hypervolume{unit}")
    # Autoscaled on purpose: the figure compares the variants' progress, and the first
    # feasible generation already covers most of the volume.
    ax.xaxis.get_major_locator().set_params(integer=True)
    return True


def fig_hypervolume(
    runs: Sequence[SearchData], out_dir: Path, styles: dict[str, tuple[str, str]]
) -> list[Path]:
    """2-D hypervolume (proxy vs. LUT projection) of the feasible trials up to each generation.

    Hypervolumes of runs with different proxy objectives live on different scales, so every
    proxy gets its own panel.
    """
    groups: dict[str, list[SearchData]] = {}
    for run in runs:
        groups.setdefault(run.config.proxy, []).append(run)
    proxies = list(groups) or ["proxy"]
    with matplotlib.rc_context(RC):
        size = (SINGLE[0], 2.7) if len(proxies) == 1 else (DOUBLE[0], 2.8)
        fig, axes = figure(size, ncols=len(proxies))
        axes = [axes] if len(proxies) == 1 else list(axes)
        drawn = False
        for ax, proxy in zip(axes, proxies, strict=True):
            drawn |= _plot_hypervolume(ax, groups.get(proxy, []), styles)
            ax.set_title(
                f"Hypervolume of the feasible set ({proxy})"
                if len(proxies) > 1
                else "Hypervolume of the feasible set",
                loc="left",
            )
        if drawn:
            handles, labels = [], []
            for ax in axes:
                h, lbl = ax.get_legend_handles_labels()
                handles += h
                labels += lbl
            fig.legend(handles, labels, loc="outside lower center", ncols=2)
    return save_figure(fig, out_dir, "search_hypervolume")


def front_points(run: SearchData, proxy: str | None = None) -> list[tuple[float, float]]:
    """(LUT, proxy score) of the Pareto-front entries with real (non-sentinel) values.

    ``proxy`` selects the score to plot; when it differs from the run's own objective the
    score is looked up in the run's records (every record carries all proxy scores), so
    fronts of runs with different proxy objectives can be compared in one plane.
    """
    proxy = proxy or run.config.proxy
    scores: dict[str, float] = {}
    if proxy != run.config.proxy:
        for record in run.records:
            value = (record.proxies or {}).get("scores", {}).get(proxy)
            if finite(value):
                scores.setdefault(record.hash, float(value))
    points = []
    for entry in run.summary.get("pareto_front", []):
        lut = (entry.get("resources") or {}).get("LUT")
        value = entry.get("proxy") if proxy == run.config.proxy else scores.get(entry.get("hash"))
        if finite(lut) and finite(value) and not is_sentinel(value) and lut > 0:
            points.append((float(lut), float(value)))
    return sorted(points)


def fig_pareto_fronts(
    runs: Sequence[SearchData],
    out_dir: Path,
    baseline: SamplingData | None,
    styles: dict[str, tuple[str, str]],
) -> list[Path]:
    """Draw the final Pareto fronts in the LUT/proxy plane over the feasible baseline samples.

    The plane is that of the proxy most of the runs optimize; fronts of runs with another
    proxy objective are projected onto it (see :func:`front_points`).
    """
    proxy = Counter(run.config.proxy for run in runs).most_common(1)[0][0] if runs else "proxy"
    with matplotlib.rc_context(RC):
        fig, ax = figure((SINGLE[0], 2.7))
        drawn = False
        if baseline is not None:
            background = [
                (r.result.resources_total["LUT"], r.proxies["scores"][proxy])
                for r in baseline.feasible
                if r.proxies
                and finite(r.result.resources_total.get("LUT"))
                and finite(r.proxies.get("scores", {}).get(proxy))
            ]
            if background:
                ax.scatter(
                    *zip(*background, strict=True),
                    color=MUTED,
                    s=6,
                    alpha=0.35,
                    linewidths=0,
                    label="random samples (feasible)",
                )
                drawn = True
        for run in runs:
            points = front_points(run, proxy)
            if not points:
                continue
            color, marker = styles[run.name]
            ax.scatter(
                *zip(*points, strict=True),
                color=color,
                marker=marker,
                s=18,
                edgecolors="white",
                linewidths=0.5,
                label=run.label,
                zorder=3,
            )
            drawn = True
        if not drawn:
            empty_axes(ax, "no feasible candidates on any front")
            return save_figure(fig, out_dir, "search_pareto_fronts")
        ax.set_xscale("log")
        ax.set_xlabel("estimated LUTs")
        ax.set_ylabel(f"proxy score ({proxy})")
        legend_below(fig, ax)
        ax.set_title("Pareto fronts (LUT vs. proxy projection)", loc="left")
    return save_figure(fig, out_dir, "search_pareto_fronts")


def fig_violation(
    runs: Sequence[SearchData], out_dir: Path, styles: dict[str, tuple[str, str]]
) -> list[Path]:
    """Mean constraint violation per generation of the constraint-handling variants.

    Only the graded violation carries information beyond the feasibility rate (binary is
    ``1 - rate``); all ``constraint`` variants are drawn so they can be compared, and every
    variant is drawn when none uses constraints.
    """
    selected = [r for r in runs if r.config.feasibility == "constraint"] or list(runs)
    with matplotlib.rc_context(RC):
        fig, ax = figure((SINGLE[0], 2.7))
        if not _plot_per_generation(ax, selected, "mean_violation", styles):
            empty_axes(ax, "no generations recorded")
            return save_figure(fig, out_dir, "search_violation_per_generation")
        ax.set_xlabel("generation")
        ax.set_ylabel("mean violation")
        ax.set_ylim(0, 1.05)
        ax.xaxis.get_major_locator().set_params(integer=True)
        legend_below(fig, ax)
        ax.set_title("Constraint violation per generation", loc="left")
    return save_figure(fig, out_dir, "search_violation_per_generation")


def _last_generation_rate(run: SearchData) -> float:
    """Feasibility rate of the last generation (NaN without generations)."""
    generations = run.summary.get("generations", [])
    return generations[-1]["feasibility_rate"] if generations else float("nan")


def _final_hypervolume(run: SearchData) -> float:
    """Last 2-D hypervolume value (NaN without history)."""
    history = run.summary.get("hypervolume", {}).get("per_generation_2d", [])
    return history[-1] if history else float("nan")


def search_table(runs: Sequence[SearchData], baseline: SamplingData | None) -> str:
    """Per-variant headline numbers as markdown."""
    rows = []
    for run in runs:
        s = run.summary
        generations = s.get("generations", [])
        rows.append(
            (
                run.label,
                run.config.proxy,
                f"{s.get('feasible', 0)} / {s.get('count', 0)}",
                percent(s.get("feasible", 0), s.get("count", 0)),
                len(generations),
                percent(_last_generation_rate(run), 1) if generations else "n/a",
                s.get("pareto_front_size", 0),
                fmt(_final_hypervolume(run), 4),
                s.get("distinct_feasible", 0),
                s.get("duplicates", 0),
            )
        )
    rate = baseline_rate(baseline)
    if rate is not None and baseline is not None:
        n = len(baseline.records)
        rows.append(
            (
                "random sampling (baseline)",
                "n/a",
                f"{round(rate * n)} / {n}",
                percent(rate, 1),
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                len({r.hash for r in baseline.feasible}),
                "n/a",
            )
        )
    return "## Search variants\n\n" + markdown_table(
        (
            "variant",
            "proxy",
            "feasible / trials",
            "rate",
            "generations",
            "rate (last gen.)",
            "front size",
            "hypervolume (2-D)",
            "distinct feasible",
            "duplicates",
        ),
        rows,
    )


def render_search_dirs(
    directories: Sequence[str | Path], out_dir: str | Path, baseline: str | Path | None = None
) -> list[Path]:
    """Render all search figures and ``search_summary.md`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [load_search(d) for d in directories]
    sampling = load_sampling(baseline) if baseline is not None else None
    styles = variant_styles(runs)
    paths = []
    paths += fig_feasibility_per_generation(runs, out_dir, sampling, styles)
    paths += fig_cumulative_feasible(runs, out_dir, sampling, styles)
    paths += fig_hypervolume(runs, out_dir, styles)
    paths += fig_pareto_fronts(runs, out_dir, sampling, styles)
    paths += fig_violation(runs, out_dir, styles)
    table = out_dir / "search_summary.md"
    table.write_text(search_table(runs, sampling))
    return [*paths, table]


# --- outcome: trained accuracy of the search's picks vs. random feasible candidates --------

#: LUT budgets at which the envelope accuracy is tabulated.
LUT_BUDGETS = (1000, 2000, 5000, 10000)
#: Colors and markers of the two outcome series. Random candidates take the muted color of the
#: background samples in the search figures; the front takes a slot no variant is pinned to.
OUTCOME_STYLES = {"random": (MUTED, "o"), "front": (SERIES[6], "s")}


@dataclass
class OutcomeSeries:
    """Trained candidates of one origin: (LUT, fps, accuracy %) per usable row."""

    name: str
    label: str
    directory: Path
    points: list[tuple[float, float, float]]

    @property
    def accuracies(self) -> list[float]:
        """Accuracies in percent."""
        return [p[2] for p in self.points]

    @property
    def luts(self) -> list[float]:
        """Estimated LUTs."""
        return [p[0] for p in self.points]


def load_outcome(directory: str | Path, name: str, label: str) -> OutcomeSeries:
    """Load the usable rows of a ground-truth directory as an outcome series."""
    from finn.nas.experiments.report import load_groundtruth

    data = load_groundtruth(directory)
    points = []
    for row in data.usable:
        lut = row.get("estimates", {}).get("LUT")
        fps = row.get("est_fps")
        acc = row.get("train", {}).get("best_val_acc")
        if finite(lut) and lut > 0 and finite(acc):
            points.append(
                (float(lut), float(fps) if finite(fps) and fps > 0 else math.nan, 100 * acc)
            )
    return OutcomeSeries(name, label, data.directory, points)


def upper_envelope(
    points: Sequence[tuple[float, float]], cheaper: str = "low"
) -> list[tuple[float, float]]:
    """Accuracy-vs-cost Pareto staircase: the points that improve on everything cheaper.

    ``cheaper="low"`` (LUTs): the best accuracy reachable with at most ``x``; ``"high"``
    (fps): the best accuracy reachable with at least ``x``. Returned sorted by ``x`` in the
    order of increasing cost.
    """
    ordered = sorted(points, key=lambda p: p[0] if cheaper == "low" else -p[0])
    envelope: list[tuple[float, float]] = []
    best = -math.inf
    for x, y in ordered:
        if y > best:
            envelope.append((x, y))
            best = y
    return envelope


def envelope_at(envelope: Sequence[tuple[float, float]], budget: float) -> float:
    """Best accuracy of a LUT envelope within ``budget`` (NaN if nothing fits)."""
    within = [y for x, y in envelope if x <= budget]
    return max(within) if within else math.nan


def accuracy_lut_hypervolume(points: Sequence[tuple[float, float]], reference_lut: float) -> float:
    """Area dominated by the (LUT, accuracy) points w.r.t. the reference (``reference_lut``, 0 %).

    LUTs are minimized and accuracy maximized, so the area is the staircase under the upper
    envelope between each point and the next cheaper-or-equal improvement, up to the
    reference LUT count. Unit: percent points times LUTs.
    """
    envelope = upper_envelope(points, "low")
    area = 0.0
    for i, (x, y) in enumerate(envelope):
        right = envelope[i + 1][0] if i + 1 < len(envelope) else reference_lut
        area += max(0.0, min(right, reference_lut) - x) * y
    return area


def _plot_outcome_panel(ax: Axes, series: Sequence[OutcomeSeries], axis: str) -> bool:
    """Scatter plus envelope staircase of every series over LUTs or fps; False if empty."""
    drawn = False
    index, cheaper, where = (0, "low", "post") if axis == "lut" else (1, "high", "pre")
    for s in series:
        points = [(p[index], p[2]) for p in s.points if finite(p[index])]
        if not points:
            continue
        color, marker = OUTCOME_STYLES.get(s.name, (SERIES[0], "o"))
        ax.scatter(
            *zip(*points, strict=True),
            color=color,
            marker=marker,
            s=12,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            label=s.label,
        )
        envelope = upper_envelope(points, cheaper)
        if cheaper == "high":
            envelope = envelope[::-1]
        ax.step(*zip(*envelope, strict=True), where=where, color=color, linewidth=1.2, zorder=3)
        drawn = True
    if not drawn:
        empty_axes(ax, "no trained candidates")
        return False
    ax.set_xscale("log")
    ax.set_xlabel("estimated LUTs" if axis == "lut" else "estimated throughput (frames/s)")
    ax.set_ylabel("best val. accuracy (%)")
    ax.set_ylim(bottom=0)
    return True


def fig_outcome(series: Sequence[OutcomeSeries], out_dir: Path) -> list[Path]:
    """Trained accuracy vs. LUTs and vs. fps of the random candidates and the search's front."""
    with matplotlib.rc_context(RC):
        fig, (ax_lut, ax_fps) = figure((DOUBLE[0], 2.9), ncols=2)
        drawn = _plot_outcome_panel(ax_lut, series, "lut")
        _plot_outcome_panel(ax_fps, series, "fps")
        ax_lut.set_title("Accuracy per LUT (lines: upper envelope)", loc="left")
        ax_fps.set_title("Accuracy per throughput", loc="left")
        if drawn:
            legend_below(fig, ax_lut)
    return save_figure(fig, out_dir, "outcome_accuracy_vs_lut")


def outcome_table(series: Sequence[OutcomeSeries]) -> str:
    """Per series: n, accuracy statistics, envelope accuracy at LUT budgets, hypervolume."""
    from finn.nas.experiments.report import _median

    reference = max((max(s.luts) for s in series if s.points), default=0.0)
    rows = []
    for s in series:
        envelope = upper_envelope([(p[0], p[2]) for p in s.points], "low")
        hv = (
            accuracy_lut_hypervolume([(p[0], p[2]) for p in s.points], reference)
            if s.points
            else math.nan
        )
        rows.append(
            (
                s.label,
                len(s.points),
                percent(_median(s.accuracies), 100) if s.points else "n/a",
                percent(max(s.accuracies), 100) if s.points else "n/a",
                fmt(_median(s.luts)) if s.points else "n/a",
                *[percent(envelope_at(envelope, b), 100) for b in LUT_BUDGETS],
                fmt(hv),
                fmt(hv / (100.0 * reference), 3) if reference else "n/a",
            )
        )
    return (
        "## Outcome: trained accuracy of the search's picks\n\n"
        + markdown_table(
            (
                "series",
                "n",
                "median acc.",
                "max acc.",
                "median LUTs",
                *[f"best acc. ≤ {b // 1000}k LUTs" for b in LUT_BUDGETS],
                "HV (acc. \u00d7 LUT)",
                "HV normalized",
            ),
            rows,
        )
        + f"\nHypervolume reference point: 0 % accuracy, {fmt(reference)} LUTs (the largest "
        "candidate of both series); normalized by 100 % \u00d7 reference LUTs.\n"
    )


def render_outcome_dirs(
    random_dir: str | Path, front_dir: str | Path, out_dir: str | Path
) -> list[Path]:
    """Render the outcome figure and ``outcome_summary.md`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    series = [
        load_outcome(random_dir, "random", "random feasible"),
        load_outcome(front_dir, "front", "search Pareto front"),
    ]
    paths = fig_outcome(series, out_dir)
    table = out_dir / "outcome_summary.md"
    table.write_text(outcome_table(series))
    return [*paths, table]

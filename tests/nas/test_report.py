"""Tests of the experiment report (figures and tables) on synthetic experiment directories."""

from __future__ import annotations

import pytest

import json
import math
from pathlib import Path
from typing import Any

from finn.nas.evaluate import CandidateEvaluation
from finn.nas.experiments.report import (
    binned_rates,
    groundtruth_table,
    load_groundtruth,
    load_sampling,
    proxy_availability,
    read_jsonl,
    render_all,
    render_groundtruth,
    render_outcome,
    render_sampling,
    render_search,
)
from finn.nas.experiments.report_search import (
    accuracy_lut_hypervolume,
    cumulative_distinct_feasible,
    envelope_at,
    front_points,
    load_outcome,
    load_search,
    upper_envelope,
    variant_name,
    variant_styles,
)
from finn.nas.result import EvalResult, FailureCategory
from finn.nas.search import SearchConfig

SAMPLING_FIGURES = [
    "sampling_failure_histogram",
    "sampling_failure_op_types",
    "sampling_feasibility_vs_size",
    "sampling_estimates_lut_fps",
    "sampling_estimates_bram_lut",
    "sampling_proxy_availability",
]
SEARCH_FIGURES = [
    "search_feasibility_per_generation",
    "search_cumulative_feasible",
    "search_hypervolume",
    "search_pareto_fronts",
    "search_violation_per_generation",
]
GROUNDTRUTH_FIGURES = [
    "groundtruth_correlations",
    "groundtruth_proxy_scatter",
    "groundtruth_accuracy",
]


def _genotype(i: int) -> dict[str, Any]:
    """Make a stand-in genotype; the report never interprets it."""
    return {"version": 1, "input_shape": [3, 8, 8], "num_classes": 2, "body": {"id": i}}


def _record(
    i: int,
    feasible: bool,
    category: FailureCategory = FailureCategory.NONE,
    op_type: str | None = None,
    stage: str | None = None,
    proxies: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One record in the documented ``CandidateEvaluation.to_dict`` format."""
    if feasible:
        result = EvalResult(
            feasible=True,
            resources_total={
                "LUT": 1000.0 + 100 * i,
                "BRAM_18K": float(i % 4),
                "URAM": 0.0,
                "DSP": 0.0,
            },
            max_cycles=1000 + i,
            est_fps=1e5 / (1000 + i),
        )
    else:
        result = EvalResult.failed(
            stage or "step_create_dataflow_partition", category, "msg", op_type=op_type
        )
    scores = {"nwot": 400.0 + i, "synflow": float(10**i), "params": 100.0 * (i + 1)}
    if proxies is None:
        proxies = {"scores": scores, "errors": {}}
    evaluation = CandidateEvaluation(
        genotype=_genotype(i),
        hash=f"hash{i:04d}",
        result=result,
        proxies=proxies,
        params=100 * (i + 1),
        nodes=3 + i % 7,
        seconds=1.5,
        meta=meta or {"index": i, "seed": 0},
    )
    return evaluation.to_dict()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_sampling_dir(root: Path, n_feasible: int = 4, n_infeasible: int = 6) -> Path:
    """Write a sampling directory with ``n_feasible`` feasible and ``n_infeasible`` failed."""
    directory = root / "sampling_synthetic"
    rows = []
    i = 0
    for _ in range(n_feasible):
        proxies = {
            "scores": {"nwot": -math.inf, "synflow": 2.0, "params": 1.0},
            "errors": {"nwot": "x"},
        }
        rows.append(_record(i, True, proxies=proxies if i == 0 else None))
        i += 1
    causes = [
        (FailureCategory.DATATYPE_UNSUPPORTED, "MultiThreshold", "step_convert_to_hw"),
        (FailureCategory.DATATYPE_UNSUPPORTED, "MultiThreshold", "step_convert_to_hw"),
        (FailureCategory.UNSUPPORTED_OP, "AveragePool", "step_create_dataflow_partition"),
        (FailureCategory.STREAMLINE_INCOMPLETE, "ElementwiseMul", "step_create_dataflow_partition"),
        (FailureCategory.FRONTEND_FAILED, "Conv", "step_streamline"),
        (FailureCategory.RESOURCE_BUDGET_EXCEEDED, None, "estimate"),
    ]
    for k in range(n_infeasible):
        category, op_type, stage = causes[k % len(causes)]
        rows.append(_record(i, False, category, op_type, stage))
        i += 1
    _write_jsonl(directory / "samples.jsonl", rows)
    (directory / "config.json").write_text(
        json.dumps({"count": len(rows), "seed": 0, "n_workers": 1})
    )
    return directory


def make_search_dir(
    root: Path,
    name: str,
    config: SearchConfig,
    rates: list[float],
    with_summary: bool = True,
) -> Path:
    """Write a search directory with ``population_size`` trials per generation at ``rates``."""
    directory = root / f"search_{name}"
    rows = []
    trial = 0
    for generation, rate in enumerate(rates):
        for j in range(config.population_size):
            # Trial 3 proposes the genotype of trial 0 again: a duplicate copies its verdict.
            source = 0 if trial == 3 else trial
            feasible = (
                rows[0]["meta"]["feasible"]
                if trial == 3
                else j < round(rate * config.population_size)
            )
            violation = 0.0 if feasible else (1.0 if config.violation == "binary" else 0.5)
            values: list[float] | None
            if feasible:
                values = [400.0 + source, 1000.0 + 100 * source, float(source % 4), 1000.0 + source]
            elif config.feasibility == "constraint":
                values = [-1e9, 1e9, 1e9, 1e9]
            else:
                values = None
            meta = {
                "trial": trial,
                "generation": generation,
                "violation": violation,
                "values": values,
                "feasible": feasible,
            }
            if trial == 3:
                meta["duplicate_of"] = 0
            rows.append(_record(source, feasible, FailureCategory.UNSUPPORTED_OP, "Sin", meta=meta))
            trial += 1
    _write_jsonl(directory / "records.jsonl", rows)
    (directory / "config.json").write_text(json.dumps({"search": config.to_dict(), "n_workers": 1}))
    if with_summary:
        from finn.nas.experiments.search import summarize_search

        records = [CandidateEvaluation.from_dict(r) for r in rows]
        (directory / "summary.json").write_text(json.dumps(summarize_search(config, records)))
    return directory


def make_groundtruth_dir(root: Path, n: int = 8, with_summary: bool = False) -> Path:
    """Write a ground-truth directory: ``nwot`` ranks accuracy perfectly, ``synflow`` inversely."""
    directory = root / "groundtruth_synthetic"
    rows = []
    for i in range(n):
        acc = 0.1 + 0.05 * i
        rows.append(
            {
                "hash": f"hash{i:04d}",
                "genotype": _genotype(i),
                "params": 100 * (i + 1),
                "nodes": 3,
                "proxies": {
                    "nwot": 400.0 + i,
                    "synflow": float(n - i),
                    "grasp": -float(10**i),
                    "jacov": math.nan,
                },
                "estimates": {"LUT": 1000.0 + 100 * i, "BRAM_18K": 1.0, "URAM": 0.0, "DSP": 0.0},
                "est_fps": 1000.0,
                "train": {
                    "val_acc": [acc],
                    "train_loss": [1.0],
                    "final_val_acc": acc,
                    "best_val_acc": acc,
                    "test_acc": None,
                    "epochs_run": 1,
                    "seconds": 1.0,
                    "diverged": False,
                    "error": None if i != n - 1 else "boom",
                },
            }
        )
    _write_jsonl(directory / "training.jsonl", rows)
    (directory / "config.json").write_text(json.dumps({"records": "x", "config": {}}))
    if with_summary:
        from finn.nas.experiments.ground_truth import rank_correlations

        usable = [r for r in rows if r["train"]["error"] is None]
        summary = rank_correlations(usable)
        summary["trained"] = len(rows)
        summary["usable"] = len(usable)
        (directory / "summary.json").write_text(json.dumps(summary))
    return directory


def _assert_figures(out: Path, names: list[str]) -> None:
    for name in names:
        for suffix in (".pdf", ".png"):
            path = out / f"{name}{suffix}"
            assert path.exists() and path.stat().st_size > 0, path


@pytest.mark.util
class TestSampling:
    """Sampling report."""

    def test_figures_and_table(self, tmp_path: Path) -> None:
        """Every sampling figure and the headline numbers of the table."""
        directory = make_sampling_dir(tmp_path, n_feasible=4, n_infeasible=6)
        out = tmp_path / "out"
        written = render_sampling(directory, out)
        _assert_figures(out, SAMPLING_FIGURES)
        assert out / "sampling_summary.md" in written
        table = (out / "sampling_summary.md").read_text()
        assert "| samples | 10 |" in table
        assert "| feasible | 4 (40.0 %) |" in table
        # Two of six failures are datatype_unsupported: 20 % of samples, 33.3 % of failures.
        assert "| datatype unsupported | 2 | 20.0 % | 33.3 % |" in table
        assert "MultiThreshold (2, 33.3 %)" in table
        # Median LUT of the feasible records 1000, 1100, 1200, 1300 (upper median).
        assert "| median LUTs of feasible | 1,200 |" in table

    def test_zero_feasible(self, tmp_path: Path) -> None:
        """No feasible sample does not crash the estimate figures."""
        directory = make_sampling_dir(tmp_path, n_feasible=0, n_infeasible=3)
        out = tmp_path / "out"
        render_sampling(directory, out)
        _assert_figures(out, SAMPLING_FIGURES)
        assert "| feasible | 0 (0.0 %) |" in (out / "sampling_summary.md").read_text()

    def test_empty_directory(self, tmp_path: Path) -> None:
        """A directory without records renders placeholders."""
        directory = tmp_path / "sampling_empty"
        directory.mkdir()
        render_sampling(directory, tmp_path / "out")
        _assert_figures(tmp_path / "out", SAMPLING_FIGURES)

    def test_helpers(self, tmp_path: Path) -> None:
        """Loader, availability, bins and the tolerance for a torn last line."""
        directory = make_sampling_dir(tmp_path, n_feasible=4, n_infeasible=6)
        data = load_sampling(directory)
        assert data.summary["count"] == 10 and data.summary["feasible"] == 4
        availability = proxy_availability(data.feasible)
        assert availability["nwot"] == (3, 4)
        assert availability["synflow"] == (4, 4)
        rates = binned_rates(data.records, "nodes", 4.0)
        assert sum(n for _, n, _ in rates) == 10
        assert sum(f for _, _, f in rates) == 4
        # A torn last line (a running experiment) is skipped, not fatal.
        with (directory / "samples.jsonl").open("a") as f:
            f.write('{"genotype": {"trunc')
        assert len(read_jsonl(directory / "samples.jsonl")) == 10


@pytest.mark.util
class TestSearch:
    """Search report."""

    def _variants(self, root: Path) -> list[Path]:
        pop = 4
        return [
            make_search_dir(
                root, "nsga2_graded", SearchConfig(population_size=pop, budget=12), [0.25, 0.5, 1.0]
            ),
            make_search_dir(
                root,
                "nsga2_binary",
                SearchConfig(population_size=pop, budget=12, violation="binary"),
                [0.25, 0.25, 0.5],
            ),
            make_search_dir(
                root,
                "nsga2_discard",
                SearchConfig(population_size=pop, budget=12, feasibility="discard"),
                [0.25, 0.25],
                with_summary=False,
            ),
            make_search_dir(
                root,
                "random",
                SearchConfig(population_size=pop, budget=12, algorithm="random"),
                [0.25, 0.0, 0.25],
            ),
            make_search_dir(
                root,
                "nsga2_graded_synflow",
                SearchConfig(population_size=pop, budget=8, proxy="synflow"),
                [0.5, 0.5],
            ),
        ]

    def test_figures_and_table(self, tmp_path: Path) -> None:
        """Every search figure and the per-variant table with the baseline row."""
        dirs = self._variants(tmp_path)
        baseline = make_sampling_dir(tmp_path)
        out = tmp_path / "out"
        written = render_search(dirs, out, baseline=baseline)
        _assert_figures(out, SEARCH_FIGURES)
        assert out / "search_summary.md" in written
        table = (out / "search_summary.md").read_text()
        # graded: (1 + duplicate) + 2 + 4 feasible of 12 trials, last generation fully
        # feasible, 7 distinct feasible genotypes, one duplicate; every feasible trial trades
        # proxy against LUT, so all 7 are on the front.
        row = next(line for line in table.splitlines() if "nsga2 graded" in line)
        assert row.startswith(
            "| nsga2 graded (graded violation) | nwot | 8 / 12 | 66.7 % | 3 | 100.0 % | 7 | "
        )
        assert row.endswith("| 7 | 1 |")
        assert (
            "| random sampling (baseline) | n/a | 4 / 10 | 40.0 % | n/a | n/a | n/a | n/a | 4 "
            "| n/a |" in table
        )
        assert "nsga2 discard (discard)" in table
        assert "random (no feedback)" in table
        assert "| nsga2 graded synflow (graded violation) | synflow | 5 / 8 |" in table

    def test_without_baseline_and_single_variant(self, tmp_path: Path) -> None:
        """One variant and no baseline is enough."""
        dirs = self._variants(tmp_path)[:1]
        out = tmp_path / "out"
        render_search(dirs, out)
        _assert_figures(out, SEARCH_FIGURES)

    def test_zero_feasible(self, tmp_path: Path) -> None:
        """A run without a feasible trial renders placeholders."""
        directory = make_search_dir(
            tmp_path, "nsga2_graded", SearchConfig(population_size=3, budget=6), [0.0, 0.0]
        )
        out = tmp_path / "out"
        render_search([directory], out)
        _assert_figures(out, SEARCH_FIGURES)
        assert "| 0 / 6 | 0.0 % |" in (out / "search_summary.md").read_text()

    def test_helpers(self, tmp_path: Path) -> None:
        """Loader (with and without summary), names, cumulative counts and pinned styles."""
        dirs = self._variants(tmp_path)
        runs = [load_search(d) for d in dirs]
        assert [r.name for r in runs] == [
            "nsga2_graded",
            "nsga2_binary",
            "nsga2_discard",
            "random",
            "nsga2_graded_synflow",
        ]
        assert variant_name(Path("search_nsga2_graded")) == "nsga2_graded"
        assert variant_name(Path("other")) == "other"
        # The discard variant has no summary.json: it is computed from the records.
        assert runs[2].summary["count"] == 8
        assert runs[0].summary["count"] == 12
        assert cumulative_distinct_feasible(runs[0].records)[-1] == 7  # trial 3 duplicates trial 0
        styles = variant_styles(runs)
        assert len({s[0] for s in styles.values()}) == 5
        assert len({s[1] for s in styles.values()}) == 5
        assert styles["random"] == variant_styles(runs[3:4])["random"]
        # The synflow run's front is projected onto the nwot plane through its records.
        own = front_points(runs[4])
        projected = front_points(runs[4], "nwot")
        assert len(own) == len(projected) == 4
        assert own != projected
        assert front_points(runs[0], "nwot") == front_points(runs[0])


@pytest.mark.util
class TestGroundTruth:
    """Ground-truth report."""

    def test_figures_and_table(self, tmp_path: Path) -> None:
        """Every ground-truth figure and the correlation table."""
        directory = make_groundtruth_dir(tmp_path, n=8)
        out = tmp_path / "out"
        written = render_groundtruth(directory, out)
        _assert_figures(out, GROUNDTRUTH_FIGURES)
        assert out / "groundtruth_summary.md" in written
        table = (out / "groundtruth_summary.md").read_text()
        assert "| trained / usable | 8 / 7 |" in table
        assert "| nwot | 7 | 1 |" in table
        assert "| synflow | 7 | -1 |" in table
        assert "| jacov | 0 | n/a |" in table
        data = load_groundtruth(directory)
        assert data.summary["usable"] == 7

    def test_with_summary_file(self, tmp_path: Path) -> None:
        """The summary file is used when current and recomputed when stale."""
        directory = make_groundtruth_dir(tmp_path, n=6, with_summary=True)
        summary = json.loads((directory / "summary.json").read_text())
        summary["marker"] = True
        (directory / "summary.json").write_text(json.dumps(summary))
        # A summary matching the records is used as is (the finished run's numbers)...
        assert load_groundtruth(directory).summary.get("marker") is True
        # ... a stale one (the run continued) is recomputed from the rows.
        summary["usable"] = 1
        (directory / "summary.json").write_text(json.dumps(summary))
        data = load_groundtruth(directory)
        assert "marker" not in data.summary and data.summary["usable"] == 5
        assert groundtruth_table(data).count("| 5 |") >= 1

    def test_too_few_rows(self, tmp_path: Path) -> None:
        """Fewer than three usable rows: no correlations, but figures and table."""
        directory = make_groundtruth_dir(tmp_path, n=2)
        render_groundtruth(directory, tmp_path / "out")
        _assert_figures(tmp_path / "out", GROUNDTRUTH_FIGURES)
        assert (
            "| trained / usable | 2 / 1 |"
            in (tmp_path / "out" / "groundtruth_summary.md").read_text()
        )


@pytest.mark.util
class TestOutcome:
    """Outcome report: accuracy per LUT of the search's front vs. random candidates."""

    def test_envelope_and_hypervolume(self) -> None:
        """Staircase, budget lookup and the dominated area on hand-made points."""
        points = [(1000.0, 20.0), (1500.0, 15.0), (2000.0, 30.0), (4000.0, 30.0), (8000.0, 40.0)]
        assert upper_envelope(points, "low") == [(1000.0, 20.0), (2000.0, 30.0), (8000.0, 40.0)]
        # fps: the best accuracy reachable with at least the given throughput.
        assert upper_envelope(points, "high") == [(8000.0, 40.0)]
        envelope = upper_envelope(points, "low")
        assert envelope_at(envelope, 999) != envelope_at(envelope, 999)  # NaN
        assert envelope_at(envelope, 1000) == 20.0
        assert envelope_at(envelope, 5000) == 30.0
        assert envelope_at(envelope, 1e9) == 40.0
        # 1000..2000 at 20 %, 2000..8000 at 30 %, 8000..10000 at 40 %.
        assert accuracy_lut_hypervolume(points, 10000.0) == pytest.approx(
            1000 * 20 + 6000 * 30 + 2000 * 40
        )
        assert accuracy_lut_hypervolume([], 10000.0) == 0.0

    def test_figure_and_table(self, tmp_path: Path) -> None:
        """Figure files and the per-series numbers of the table."""
        random_dir = make_groundtruth_dir(tmp_path / "random", n=8)
        front_dir = make_groundtruth_dir(tmp_path / "front", n=5)
        out = tmp_path / "out"
        written = render_outcome(random_dir, front_dir, out)
        _assert_figures(out, ["outcome_accuracy_vs_lut"])
        assert out / "outcome_summary.md" in written
        table = (out / "outcome_summary.md").read_text()
        # random: 7 usable rows, accuracies 10..40 %, LUTs 1000..1600 (the 1700 row errored);
        # the envelope reaches 40 % within 2k LUTs and nothing fits in 1k... except row 0.
        assert (
            "| random feasible | 7 | 25.0 % | 40.0 % | 1,300 | 10.0 % | 40.0 % | 40.0 % | 40.0 % |"
            in table
        )
        assert (
            "| search Pareto front | 4 | 20.0 % | 25.0 % | 1,200 | 10.0 % | 25.0 % | 25.0 % "
            "| 25.0 % |" in table
        )
        assert "reference point: 0 % accuracy, 1,600 LUTs" in table
        series = load_outcome(random_dir, "random", "random feasible")
        assert len(series.points) == 7 and series.points[0] == (1000.0, 1000.0, pytest.approx(10.0))

    def test_empty(self, tmp_path: Path) -> None:
        """Two directories without usable rows render placeholders."""
        random_dir = tmp_path / "random" / "groundtruth_synthetic"
        random_dir.mkdir(parents=True)
        front_dir = make_groundtruth_dir(tmp_path / "front", n=1)  # the only row errored
        out = tmp_path / "out"
        render_outcome(random_dir, front_dir, out)
        _assert_figures(out, ["outcome_accuracy_vs_lut"])
        assert "| random feasible | 0 | n/a |" in (out / "outcome_summary.md").read_text()


@pytest.mark.util
def test_render_all(tmp_path: Path) -> None:
    """The command's entry point renders everything and combines the tables."""
    sampling = make_sampling_dir(tmp_path)
    search = make_search_dir(
        tmp_path, "nsga2_graded", SearchConfig(population_size=4, budget=8), [0.25, 0.5]
    )
    groundtruth = make_groundtruth_dir(tmp_path, n=5)
    out = tmp_path / "out"
    written = render_all(
        out,
        sampling=sampling,
        search=[search],
        groundtruth=groundtruth,
        outcome=(groundtruth, groundtruth),
    )
    _assert_figures(
        out, SAMPLING_FIGURES + SEARCH_FIGURES + GROUNDTRUTH_FIGURES + ["outcome_accuracy_vs_lut"]
    )
    combined = (out / "summary.md").read_text()
    assert combined.startswith("# NAS experiment report")
    for heading in ("## Sampling", "## Search variants", "## Ground truth", "## Outcome"):
        assert heading in combined
    assert out / "summary.md" in written

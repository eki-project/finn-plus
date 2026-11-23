"""Export the candidates of a run for full FINN+ builds (hardware fidelity of the oracle).

The oracle only estimates. To measure how far its estimates and verdicts hold up in hardware,
the candidates of interest (the Pareto front of a search run, or feasible samples of a sampling
run) are rebuilt, calibrated and exported as QONNX the way the evaluator exported them, together
with a benchmarking config for ``finn bench`` that builds every one of them with the oracle's
flow (``dut: nas_candidate``) through to synthesis, and with their records so that
``finn nas groundtruth`` can train them.
"""

from __future__ import annotations

import json
import torch
import yaml
from pathlib import Path
from typing import TYPE_CHECKING, Any

from finn.builder.build_dataflow_config import default_build_dataflow_steps
from finn.nas.evaluate import EvaluatorConfig, randomize_batchnorm
from finn.nas.experiments.ground_truth import GroundTruthConfig, select_candidates
from finn.nas.experiments.sampling import (
    CONFIG_FILE,
    RECORDS_FILE,
    SUMMARY_FILE,
    append_records,
    read_records,
)
from finn.nas.pipeline import frontend_steps

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from finn.nas.config import OracleConfig
    from finn.nas.evaluate import CandidateEvaluation

#: Benchmarking DUT of the exported candidates (``src/finn/benchmarking/dut/nas_candidate.yml``).
DUT = "nas_candidate"
BENCH_CONFIG_FILE = "bench_config.yml"
EXPORT_SUMMARY_FILE = "export_summary.json"
#: Records file of a search run (a sampling run writes :data:`RECORDS_FILE`).
SEARCH_RECORDS_FILE = "records.jsonl"
#: Build outputs requested by default: the estimates to compare against and a synthesized bitfile
#: (post-synthesis resources, and a throughput measurement if the board is attached to the CI).
DEFAULT_OUTPUTS: tuple[str, ...] = ("estimate_reports", "bitfile")

Selection = str  # "auto" | "front" | "feasible"


def build_steps(oracle: OracleConfig) -> list[str]:
    """Return the oracle's pre-folding flow followed by the rest of the default build flow."""
    rest = default_build_dataflow_steps[
        default_build_dataflow_steps.index("step_target_fps_parallelization") :
    ]
    return [step for _, step in frontend_steps(oracle)] + rest


def read_run(run_dir: Path) -> tuple[list[CandidateEvaluation], dict[str, Any], dict[str, Any]]:
    """Read records, config and summary (empty if not written yet) of a sampling or search run."""
    records = read_records(run_dir / RECORDS_FILE) or read_records(run_dir / SEARCH_RECORDS_FILE)
    config = json.loads((run_dir / CONFIG_FILE).read_text())
    summary_path = run_dir / SUMMARY_FILE
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return records, config, summary


def select_records(
    records: list[CandidateEvaluation],
    summary: dict[str, Any],
    selection: Selection = "auto",
    limit: int = 0,
) -> list[CandidateEvaluation]:
    """Pick the feasible records to export, one per genotype.

    ``"front"`` takes the Pareto front of a search run (from its summary, in trial order),
    ``"feasible"`` the feasible records spread evenly over the parameter-count range (all of
    them without a ``limit``), ``"auto"`` the front when the summary has one and the feasible
    records otherwise. ``limit`` > 0 caps the number in both cases.
    """
    front = summary.get("pareto_front")
    if selection == "auto":
        selection = "front" if front is not None else "feasible"
    if selection == "front":
        if front is None:
            raise ValueError("The run's summary has no Pareto front (not a finished search run?)")
        wanted = {entry["hash"] for entry in front}
        in_front = [r for r in records if r.hash in wanted]
        config = GroundTruthConfig(
            count=limit or len(in_front), selection="first", require_proxies=False
        )
        return select_candidates(in_front, config)
    if selection != "feasible":
        raise ValueError(f"Unknown selection {selection!r} (auto, front or feasible)")
    config = GroundTruthConfig(
        count=limit or len(records), selection="stratified", require_proxies=False
    )
    return select_candidates(records, config)


def export_record(
    record: CandidateEvaluation, evaluator: EvaluatorConfig, destination: Path, seed: int = 0
) -> Path:
    """Rebuild, calibrate and export one record's network as the evaluator did (random weights).

    The weights differ from the evaluation (the proxy minibatch is not available here, so the
    calibration uses random inputs), which only matters for the weight-dependent part of the
    bit-width minimization; the architecture and the quantizers are the same.
    """
    from finn.nas.calibrate import calibrate, synthetic_inputs
    from finn.nas.export import export_candidate
    from finn.nas.space.build import to_module

    genotype = record.genotype
    input_shape = tuple(genotype["input_shape"])
    torch.manual_seed(seed)
    model = to_module(genotype)
    if evaluator.randomize_batchnorm:
        randomize_batchnorm(model, seed)
    inputs = synthetic_inputs(input_shape, evaluator.calibration_batch_size, seed)
    calibrate(model, inputs, evaluator.calibration_steps, evaluator.calibration_batch_size)
    return export_candidate(model, (1, *input_shape), destination)


def bench_entry(
    model_path: Path, oracle: OracleConfig, generate_outputs: Sequence[str] = DEFAULT_OUTPUTS
) -> dict[str, list[Any]]:
    """One ``finn bench`` parameter set building ``model_path`` as the oracle configured it.

    Every value is a one-element list, the grid notation of the configs in ``ci/cfg/``.
    """
    params: dict[str, Any] = {
        "dut": DUT,
        "model_path": str(model_path),
        "steps": build_steps(oracle),
        "generate_outputs": list(generate_outputs),
        "synth_clk_period_ns": oracle.synth_clk_period_ns,
        "mvau_wwidth_max": oracle.mvau_wwidth_max,
        "folding_two_pass_relaxation": oracle.folding_two_pass_relaxation,
        "minimize_bit_width": oracle.minimize_bit_width,
        "standalone_thresholds": oracle.standalone_thresholds,
        "max_multithreshold_bit_width": oracle.max_multithreshold_bit_width,
    }
    if oracle.board is not None:
        params["board"] = oracle.board
    elif oracle.fpga_part is not None:
        params["fpga_part"] = oracle.fpga_part
    if oracle.folding.kind == "target_fps":
        params["target_fps"] = oracle.folding.target_fps
    else:
        params["folding_config_file"] = oracle.folding.folding_config_file
    params.update(oracle.dataflow_cfg_overrides)
    return {key: [value] for key, value in params.items()}


def export_run(
    run_dir: Path,
    out_dir: Path,
    selection: Selection = "auto",
    limit: int = 0,
    seed: int = 0,
    model_path_prefix: str | Path | None = None,
    generate_outputs: Sequence[str] = DEFAULT_OUTPUTS,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Export the selected candidates of ``run_dir`` into ``out_dir``.

    Writes ``<hash>.onnx`` per candidate, their records as ``samples.jsonl`` (the input of
    ``finn nas groundtruth``), ``bench_config.yml`` for ``finn bench`` and a summary. The bench
    config refers to the models under ``model_path_prefix`` when given (the directory the
    exports are moved to before the builds run), otherwise under ``out_dir``.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    records, config, summary = read_run(run_dir)
    evaluator = EvaluatorConfig.from_dict(config["evaluator"])
    chosen = select_records(records, summary, selection, limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = Path(model_path_prefix) if model_path_prefix is not None else out_dir.absolute()
    entries = []
    candidates = []
    for index, record in enumerate(chosen):
        name = f"{record.hash[:16]}.onnx"
        if progress is not None:
            progress(f"[{index + 1}/{len(chosen)}] exporting {name}")
        export_record(record, evaluator, out_dir / name, seed)
        entries.append(bench_entry(prefix / name, evaluator.oracle, generate_outputs))
        candidates.append(
            {
                "hash": record.hash,
                "model": name,
                "params": record.params,
                "nodes": record.nodes,
                "est_fps": record.result.est_fps,
                "max_cycles": record.result.max_cycles,
                "resources_total": record.result.resources_total,
            }
        )
    records_path = out_dir / RECORDS_FILE
    records_path.unlink(missing_ok=True)
    append_records(records_path, chosen)
    header = (
        f"# {len(entries)} NAS candidates exported from {run_dir} (selection: {selection}).\n"
        "# One experiment per candidate; launch with `finn bench --bench_config <this file>` or\n"
        "# as a manual CI bench run (MANUAL_CFG_PATH, see ci/nas/README.md). Keep PARALLEL_JOBS\n"
        "# in sync with the number of experiments to give every candidate its own Slurm task.\n"
    )
    (out_dir / BENCH_CONFIG_FILE).write_text(header + yaml.safe_dump(entries, sort_keys=False))
    result = {
        "run_dir": str(run_dir),
        "selection": selection,
        "count": len(chosen),
        "model_path_prefix": str(prefix),
        "candidates": candidates,
    }
    (out_dir / EXPORT_SUMMARY_FILE).write_text(json.dumps(result, indent=2))
    return result

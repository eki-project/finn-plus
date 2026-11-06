"""Command line front end of the NAS oracle, used by ``finn nas eval``."""

from __future__ import annotations

import json
from rich.console import Console
from rich.table import Table
from typing import TYPE_CHECKING

from finn.nas.oracle import FinnOracle
from finn.nas.result import TRANSIENT_FAILURES, EvalResult

if TYPE_CHECKING:
    from pathlib import Path

    from finn.nas.config import OracleConfig

#: The model is FINN-compatible.
EXIT_FEASIBLE = 0
#: The model was evaluated and is not FINN-compatible (or exceeds the configured limits).
EXIT_INFEASIBLE = 1
#: The evaluation itself failed (timeout, misconfiguration, internal error).
EXIT_ERROR = 3


def exit_code_for(result: EvalResult) -> int:
    """Map an evaluation result to the process exit code of ``finn nas eval``."""
    if result.feasible:
        return EXIT_FEASIBLE
    if result.failure_category in TRANSIENT_FAILURES:
        return EXIT_ERROR
    return EXIT_INFEASIBLE


def _fmt(value: float | None, digits: int = 0) -> str:
    """Format a number with thousands separators, or a dash if it is missing."""
    if value is None:
        return "-"
    return f"{value:,.{digits}f}"


def print_result(result: EvalResult, console: Console) -> None:
    """Print a human-readable summary of ``result``."""
    if result.feasible:
        console.print("[bold green]FINN-compatible[/bold green]")
    else:
        console.print(
            f"[bold red]Not FINN-compatible[/bold red]: {result.failure_category.value} "
            f"(stage {result.failure_stage})"
        )
        if result.failure_node is not None:
            console.print(f"  node: {result.failure_node} ({result.failure_op_type or '?'})")
        if result.failure_message:
            console.print(f"  {result.failure_message}", markup=False, highlight=False)

    if result.resources_total:
        table = Table(title="Estimated resources", title_justify="left")
        table.add_column("Resource")
        table.add_column("Estimate", justify="right")
        if result.device_fit:
            table.add_column("Share of board budget", justify="right")
        for name, value in sorted(result.resources_total.items()):
            row = [name, _fmt(value)]
            if result.device_fit:
                fit = result.device_fit.get(name)
                row.append("-" if fit is None else f"{100 * fit:.1f} %")
            table.add_row(*row)
        console.print(table)

    if result.max_cycles is not None:
        console.print(
            f"Estimated throughput: {_fmt(result.est_fps, 1)} fps "
            f"(slowest node {result.max_cycles_node}: {_fmt(result.max_cycles)} cycles/frame)"
        )
        console.print(
            f"Estimated latency: {_fmt(result.critical_path_cycles)} cycles "
            f"({_fmt(result.est_latency_ns)} ns, pessimistic critical path)"
        )
    if result.throughput_target_missed:
        console.print("[yellow]The throughput target was not reached by the folding.[/yellow]")
    if result.unmapped_boundary_nodes:
        nodes = ", ".join(f"{name} ({op})" for name, op in result.unmapped_boundary_nodes)
        console.print(f"Nodes at the graph boundary that stay on the host: {nodes}")
    total = result.timings_s.get("total")
    cached = " (from cache)" if result.from_cache else ""
    if total is not None:
        console.print(f"Evaluated in {total:.2f} s{cached}", highlight=False)


def evaluate_model(
    model: Path,
    config: OracleConfig,
    output: Path | None = None,
    console: Console | None = None,
) -> int:
    """Evaluate the QONNX model at ``model`` and report the result.

    Prints a summary, writes the complete :class:`EvalResult` as JSON to ``output`` if given,
    and returns the exit code (see the ``EXIT_*`` constants).
    """
    console = console or Console()
    result = FinnOracle(config).evaluate(model, tag=model.stem)
    print_result(result, console)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
        console.print(f"Full result written to {output}", highlight=False)
    return exit_code_for(result)

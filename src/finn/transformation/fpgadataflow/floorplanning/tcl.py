"""Bridge between the Python floorplanner and its Vivado-side shell.

Python decides *what* to place -- which nodes form a placement unit and which
SLR it belongs to -- and, since Milestone 6, also *where*: the packing
arithmetic is `floorplanning.packer`, run by the Vivado hook as a separate
process. Vivado remains the measurement and application layer, because the
device census (available sites, clock regions, shell exclusion zones) and the
post-`opt_design` utilisation only exist inside it.

This module renders the handover: the interpreter and environment the hook
must use, and a `finn_fp_floorplan_units` call for `finn_floorplanning.tcl`.
"""

import os
import sys
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

from finn.util.exception import FINNInternalError

#: Demand fields the Tcl engine expects, in the order it documents them.
TCL_DEMAND_FIELDS: tuple[str, ...] = (
    "luts",
    "ffs",
    "carries",
    "brams",
    "dsps",
    "urams",
    # LUTRAM and SRLs only fit SLICEM sites, so the packer sizes them separately
    # from plain LUTs. FINN has no analytical model for either, so these are only
    # non-zero once measured post-synthesis.
    "lutrams",
    "srls",
    # Distributed-RAM macro count; only the hook's netlist measurement fills it.
    "ram_macros",
)

#: Maps a FINN resource-estimate key onto the demand field it feeds.
#: BRAM_18K needs conversion (see `demands_from_estimate`) and so is absent here.
RESOURCE_TO_DEMAND: dict[str, str] = {
    "LUT": "luts",
    "FF": "ffs",
    "DSP": "dsps",
    "URAM": "urams",
}

#: Characters that would break out of the Tcl list literals rendered here.
_TCL_UNSAFE: frozenset[str] = frozenset("{}[]$\\\"' \t\n")


def floorplan_tcl_path() -> Path:
    """Return the path of the Vivado-side shell script."""
    finn_root = Path(__file__).parent.parent.parent.parent
    return finn_root / "util" / "vivado_scripts" / "finn_floorplanning.tcl"


def floorplan_packer_path() -> Path:
    """Return the path of the packer script the shell runs as a separate process."""
    return Path(__file__).parent / "packer.py"


#: Environment variables handed to the packer process, and nothing else.
#:
#: Vivado rewrites LD_LIBRARY_PATH when it starts, which breaks conda
#: interpreters, so the hook launches the packer with `env -i` and exactly this
#: list, captured from FINN's own environment at render time. A whitelist rather
#: than a dump of os.environ keeps credentials out of a Tcl file that lands in
#: the build directory.
PACKER_ENV_WHITELIST: tuple[str, ...] = (
    "PATH",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "CONDA_PREFIX",
    "VIRTUAL_ENV",
    "HOME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
)


def packer_environment() -> dict[str, str]:
    """Return the environment the packer runs in: the whitelist plus every FINN_* variable."""
    env = {k: os.environ[k] for k in PACKER_ENV_WHITELIST if k in os.environ}
    env.update({k: v for k, v in os.environ.items() if k.startswith("FINN_")})
    env.setdefault("LANG", "C.UTF-8")
    return env


def _tcl_string(value: str) -> str:
    """Quote an arbitrary string as a Tcl word (braces are escaped, not nested)."""
    return "{" + value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}") + "}"


def demands_from_estimate(estimate: dict[str, int | float]) -> dict[str, int]:
    """Convert a FINN resource estimate into the demand dict the Tcl engine takes.

    Args:
        estimate: Per-unit resource sums keyed by FINN resource name (LUT,
            BRAM_18K, URAM, DSP, and optionally FF).

    Returns:
        A demand dict holding every field in `TCL_DEMAND_FIELDS`. Resources FINN
        has no estimate for come out as 0:

        - **FF** has no analytical cost model, so it is only non-zero when the
          estimate was merged with Vitis HLS reports.
        - **CARRY8** is never estimated; the packer's SLICE rectangle is sized by
          LUTs, which in practice covers the carry chains inside it.

        BRAM is converted from FINN's BRAM_18K units to the RAMB36 sites the
        packer counts, rounding up so a lone 18K block still claims a site.
    """
    demands = dict.fromkeys(TCL_DEMAND_FIELDS, 0)
    for resource, field_name in RESOURCE_TO_DEMAND.items():
        demands[field_name] = int(estimate.get(resource, 0))
    demands["brams"] = ceil(float(estimate.get("BRAM_18K", 0)) / 2)
    return demands


@dataclass
class FloorplanUnitSpec:
    """One placement unit, as handed to the Vivado-side packer.

    Attributes:
        name: Pblock name to create. Must be unique across the design.
        cells: Hierarchical cell paths to constrain, resolved against the
            elaborated design (e.g. `top_i/Hier_MVAU_hls_0`).
        slr: SLR index Level-1 assignment chose for this unit.
        demands: Area requirement, as from `demands_from_estimate`.
    """

    name: str
    slr: int
    demands: dict[str, int]
    cells: list[str] = field(default_factory=list)

    def to_tcl(self) -> str:
        """Render this unit as a Tcl dict literal."""
        _check_tcl_safe(self.name)
        for cell in self.cells:
            _check_tcl_safe(cell, allow_slash=True)
        missing = [f for f in TCL_DEMAND_FIELDS if f not in self.demands]
        if missing:
            raise FINNInternalError(
                f"Placement unit {self.name} is missing demand field(s) {missing}; "
                "build the dict with demands_from_estimate."
            )
        demand_str = " ".join(f"{f} {int(self.demands[f])}" for f in TCL_DEMAND_FIELDS)
        return (
            f"{{name {self.name} cells {{{' '.join(self.cells)}}} "
            f"slr {int(self.slr)} demands {{{demand_str}}}}}"
        )


def _check_tcl_safe(value: str, allow_slash: bool = False) -> None:
    """Reject names that would not survive being pasted into a Tcl list literal.

    Node and cell names in FINN are alphanumeric with underscores (and slashes
    for hierarchical paths), so this never fires in practice. It exists so that a
    name that somehow does contain a brace or a space fails here rather than
    producing a Tcl script that silently parses into the wrong thing.

    Raises:
        FINNInternalError: If the value contains a character that is unsafe
            inside a Tcl list literal.
    """
    if not value:
        raise FINNInternalError("Empty name passed to the floorplanner's Tcl renderer.")
    bad = sorted({c for c in value if c in _TCL_UNSAFE})
    if not allow_slash:
        bad += sorted({c for c in value if c == "/"})
    if bad:
        raise FINNInternalError(
            f"Name {value!r} contains character(s) {bad} that cannot be rendered into "
            "a Tcl list literal."
        )


def render_floorplan_tcl(
    units: list[FloorplanUnitSpec],
    soft: bool = True,
    report_file: str | Path | None = None,
    measure: bool = True,
    margin: float = 1.1,
    overrides_file: str | Path | None = None,
    work_dir: str | Path | None = None,
    python_executable: str | None = None,
    pack_order: str = "dataflow",
    slicem_utilization: float = 0.8,
    logic_utilization: float = 0.75,
    closed_sites: list[str] | None = None,
    closed_sites_vivado: str = "",
    closed_sites_source: str = "",
) -> list[str]:
    """Render the Tcl lines that run the geometry engine over `units`.

    Args:
        units: Placement units in the order they should be packed, i.e. ascending
            `floorplan_group_id`, which is dataflow order. The packer walks that
            order serpentine across each SLR, so the order carries the locality.
        soft: Whether to create IS_SOFT pblocks. True is the validated mode; hard
            pblocks are a later iteration.
        report_file: Where the engine should write its JSON report, if anywhere.
        measure: Whether the engine should size every unit from the utilization it
            measures rather than from the demands in `units`. Both shipped flows
            run it against a synthesized design -- Zynq inside `open_run synth_1`,
            Alveo at the pre-place hook -- so this is True for every build, and the
            demands serve only as a fallback for cells the utilization report does
            not name. False exists for callers with no netlist to read, which in
            practice means the tests, where the engine is exercised against a bare
            `link_design -part`.
        margin: Headroom applied on top of whichever numbers the engine uses.
        overrides_file: Where the engine should look for per-unit demand increases
            written after a placer rejection (see `placer_retry`). The file need
            not exist; a first attempt has none. Omit to disable the mechanism.
        work_dir: Where the shell writes the census, unit and packing JSON files.
            Defaults to the report file's directory, then the overrides file's,
            then Vivado's working directory.
        python_executable: Interpreter for the packer process; FINN's own
            `sys.executable` unless overridden, so the packer sees the same
            environment FINN runs in.
        pack_order: ``"dataflow"`` (graph order, the default) or ``"area"``
            (largest footprint first); see `packer.pack`.
        slicem_utilization: Fraction of a pblock's SLICEM LUTs that LUTRAM and
            shift registers may fill (`DataflowBuildConfig.floorplan_slicem_utilization`).
        logic_utilization: Fraction of a pblock's slices plain logic may fill
            (`DataflowBuildConfig.floorplan_logic_utilization`).
        closed_sites: Sites the platform shell closes to the placer, from its
            shell closure map (`shell_closure.load_map`); the hook prohibits them
            before it takes the census. None or empty when the platform has no
            map, or no shell.
        closed_sites_vivado: The Vivado release the map was measured under; the
            hook warns when it runs under another.
        closed_sites_source: Where the map came from (or why there is none), for
            the log and the report.

    Returns:
        Tcl lines to append to a Vivado script, starting with sourcing the
        engine. Returns an empty list when there is nothing to place, so a caller
        can splice the result in unconditionally.

    Raises:
        FINNInternalError: If two units share a pblock name.
    """
    if not units:
        return []
    names = [unit.name for unit in units]
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise FINNInternalError(f"Duplicate floorplan pblock name(s): {duplicates}")

    if work_dir is None:
        if report_file is not None:
            work_dir = Path(report_file).parent
        elif overrides_file is not None:
            work_dir = Path(overrides_file).parent
    env_words = " ".join(
        f"{_tcl_string(k)} {_tcl_string(v)}" for k, v in sorted(packer_environment().items())
    )
    unit_literals = " ".join(unit.to_tcl() for unit in units)
    lines = [
        f"set FINN_FP_PYTHON {_tcl_string(python_executable or sys.executable)}",
        f"set FINN_FP_PACKER {_tcl_string(str(floorplan_packer_path()))}",
        f"set FINN_FP_PYTHON_ENV [list {env_words}]",
        f"set FINN_FP_PACK_ORDER {pack_order}",
        f"set FINN_FP_SLICEM_UTIL {float(slicem_utilization)}",
        f"set FINN_FP_LOGIC_UTIL {float(logic_utilization)}",
        f"set FINN_FP_WORK_DIR {_tcl_string(str(work_dir) if work_dir is not None else '')}",
        f"set FINN_FP_CLOSED_SITES [list {' '.join(closed_sites or [])}]",
        f"set FINN_FP_CLOSED_SITES_VIVADO {_tcl_string(closed_sites_vivado)}",
        f"set FINN_FP_CLOSED_SITES_SOURCE {_tcl_string(closed_sites_source)}",
        f"source {{{floorplan_tcl_path()}}}",
        f"finn_fp_floorplan_units {{{unit_literals}}} {int(soft)} {int(measure)} {margin}"
        + (f" {{{overrides_file}}}" if overrides_file is not None else ""),
    ]
    if report_file is not None:
        lines.append(f"finn_fp_write_floorplan_report {{{report_file}}}")
    return lines

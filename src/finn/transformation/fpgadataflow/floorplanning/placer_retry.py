"""Re-run implementation with the placer's own verdict after a pblock rejection.

The geometry engine sizes every pblock from measured utilization and from the
sites Vivado reports as available, but `place_design` can still close sites it
judges unroutable and reject a region as too small. Nothing visible before
placement predicts that, so this is the fallback that acts only once it has
happened: parse the exact shortfall the placer prints per pblock and resource,
hand it back to the engine as a demand increase, and run implementation again.

It costs a full implementation attempt per retry and nothing at all when the
first attempt succeeds -- which is the point: no area is spent on a problem that
did not occur.
"""

import re
from pathlib import Path

from finn.util.logging import log

#: How many times to re-run implementation on a placer rejection before giving up.
MAX_PLACER_RETRIES: int = 4

#: The placer's resource names, mapped to the engine's demand keys.
_PLACER_KIND_TO_DEMAND: dict[str, str] = {"DSP": "dsps", "BRAM": "brams", "URAM": "urams"}

#: One `[Place 30-859]` block. The message names the pblock and gives the exact
#: required/available counts, which is what makes a measured retry possible.
_REJECTION = re.compile(
    r"\[Place 30-859\].*?"
    r"PBlock name is: (?P<pblock>\S+)\s*"
    r"Number of (?P<kind>\w+) required by this constraint: (?P<required>\d+)\s*"
    r".*?Number of \w+ available in this constraint region: (?P<available>\d+)",
    re.S,
)

#: One `[Place 30-1112]` message: the legaliser could not find a legal SLICEM site
#: for a LUTRAM/SRL. It names no pblock and no count, only the SLICE range the
#: instance is constrained to, which the floorplan report maps back to a pblock.
_LUTRAM_REJECTION = re.compile(
    r"\[Place 30-1112\] Failed to move LUTRAM instance (?P<inst>\S+).*?"
    r"from site SLICE_X(?P<x0>\d+)Y(?P<y0>\d+).*? to site SLICE_X(?P<x1>\d+)Y(?P<y1>\d+)",
    re.S,
)

#: How much more memory-LUT room a unit asks for after a `30-1112`, as a fraction
#: of what its pblock offered. The message carries no count, so this is the step
#: of the repair loop, not a sizing rule; the sizing rule is the SLICEM ceiling
#: (`floorplan_slicem_utilization`), which is what should prevent the error.
LUTRAM_RETRY_GROWTH: float = 0.25

#: One `[Place 30-488]` message: detail placement could not commit plain logic
#: (LUT, FF, CARRY) inside its pblock. It lists the instances with their type and
#: no count, so the unit is found from the instance's hierarchy.
_COMMIT_FAILURE = re.compile(
    r"\[Place 30-488\] Failed to commit \d+ instances:\s*\n"
    r"(?P<body>(?:[^\n]*No site found[^\n]*\n?)+)"
)
_COMMIT_INSTANCE = re.compile(r"^(?P<inst>\S+) with block Id: \d+ \((?P<type>\w+)\)", re.M)

#: One `[Place 30-1153]` message: detail placement could not commit multi-slice
#: shapes (carry chains with the LUTs and flip-flops packed around them). Each
#: shape is named by one instance and the number of instances it holds.
_MACRO_FAILURE = re.compile(
    r"\[Place 30-1153\] Failed to commit \d+ macros:\s*\n"
    r"(?P<body>.*?)(?=\n(?:ERROR|WARNING|INFO):|\Z)",
    re.S,
)
_MACRO_INSTANCE = re.compile(
    r"^(?P<inst>level\S+) with block Id: \d+\s*\nThis block contains (?P<n>\d+) instances", re.M
)

#: The engine's demand key per primitive type in a `30-488` listing.
_COMMIT_TYPE_TO_DEMAND: dict[str, str] = {"LUT": "luts", "FF": "ffs", "CARRY": "carries"}

#: One `[Place 30-484]` block: the LUTRAM/SRL packer could not fit every memory
#: primitive into a SLICEM of its pblock. It reports device-wide counts and then
#: names the primitives that failed with their type; each is attributed to its unit
#: through the instance's hierarchy, like a `30-488`. First seen when the design
#: needed 93 % of the device's SLICEM (mobilenetv1 with register slices, 15 Oct
#: 2026): twelve RAM64M8 of one window buffer did not fit its pblock.
_LUTRAM_PACK_FAILURE = re.compile(
    r"\[Place 30-484\].*?Names of these LUTRAMs/SRLs:\s*\n(?P<body>(?:\S+ type \w+\s*\n?)+)",
    re.S,
)
_LUTRAM_PACK_INSTANCE = re.compile(r"^(?P<inst>\S+) type (?P<type>\w+)", re.M)

#: Memory LUTs one failed primitive occupies, per type: a RAM64M8 or RAM32M16 fills a
#: SLICEM's eight LUTs, the quad forms four, a dual-port bit two, the rest one.
_LUTRAM_TYPE_LUTS: dict[str, int] = {
    "RAM64M8": 8,
    "RAM32M16": 8,
    "RAM64M": 4,
    "RAM32M": 4,
    "RAM256X1S": 4,
    "RAM128X1S": 2,
    "RAM64X1D": 2,
    "RAM32X1D": 2,
    "RAM128X1D": 4,
    "RAM256X1D": 8,
}

#: LUTs per SLICE (UltraScale+), for sizing a retry in whole slices.
SLICE_LUTS: int = 8

#: How much more of a resource a unit asks for after a `30-488`, as a fraction of
#: what its pblock offered: the step of the repair loop, like
#: `LUTRAM_RETRY_GROWTH`. The message carries no count.
COMMIT_RETRY_GROWTH: float = 0.25

#: The relative location of the implementation log under a Vitis link directory.
VPL_LOG: Path = Path("_x/link/vivado/vpl/runme.log")

#: The floorplan report the hook writes next to the link config.
REPORT_FILE: str = "finn_floorplan.json"

#: The measured per-unit demands the hook exported for the packer, same directory.
UNITS_FILE: str = "finn_floorplan_units.json"

#: Report capacity key per demand key, for sizing an override against the region.
_CAPACITY_KEY: dict[str, str] = {
    "dsps": "dsps",
    "brams": "brams",
    "urams": "urams",
    "lutrams": "lutrams",
}

#: The overrides file, next to the link config, that the engine sources if present.
OVERRIDES_FILE: str = "finn_floorplan_overrides.tcl"


#: Printed by the floorplanning hook at the start of every implementation attempt.
ATTEMPT_MARKER: str = "FINN+ automatic pblock floorplanning"


def latest_attempt(log_text: str) -> str:
    """Return the part of an implementation log written by the latest attempt.

    Vitis appends every resumed implementation to the same `runme.log`, so the
    file holds all earlier attempts' rejections too. Parsed whole, a retry counted
    attempt one's rejections again and missed what the latest attempt reported
    (mobilenetv1, job 3577610, 4 Oct 2026). Each attempt starts with the hook's
    banner; a log without one is returned as is.
    """
    i = log_text.rfind(ATTEMPT_MARKER)
    return log_text if i < 0 else log_text[i:]


def parse_placer_rejections(
    log_text: str, report: dict | None = None, units: list[dict] | None = None
) -> dict[str, dict[str, int]]:
    """Extract per-pblock, per-resource shortfalls from the placer's rejections.

    Five messages are understood. `[Place 30-859]` names the pblock and gives
    the exact required/available counts. `[Place 30-1112]` (a LUTRAM/SRL that
    found no legal SLICEM site) names only the SLICE range the instance was
    constrained to; with the floorplan `report` that range is mapped back to its
    pblock and the unit's memory-LUT demand is raised by `LUTRAM_RETRY_GROWTH` of
    the SLICEM LUTs its pblock offered. `[Place 30-488]` (plain logic that could
    not be committed) lists instances and their type; each is attributed to its
    unit through the `units` cells, and the unit's demand for that type is raised
    by `COMMIT_RETRY_GROWTH` of its pblock's capacity, at least by the number of
    instances that failed. `[Place 30-1153]` (multi-slice shapes -- carry chains
    and the logic packed around them -- that could not be committed) is
    attributed the same way and grows the unit's LUT demand by the same step, at
    least one slice per failed shape. `[Place 30-484]` (memory primitives the
    LUTRAM/SRL packer could not fit into the pblock's SLICEMs) names them with
    their type; each is attributed through the `units` cells and the unit's
    memory-LUT demand grows by the same step, at least by the LUTs the failed
    primitives occupy.

    The placer's shortfall is measured against the sites *it* left open, not
    against the region's size: it closes sites next to the static shell
    (`place.closeUnroutableSites`) and reports "14 required, 11 available" for a
    region that holds 24. Raising the demand by three would leave the packer's
    choice unchanged (mobilenetv1, 2 Oct 2026). With the report's capacity and
    the exported demands, an override is sized so the new demand exceeds what
    the rejected region offered, which forces a different selection.

    Args:
        log_text: Contents of the Vivado implementation log.
        report: The parsed `finn_floorplan.json`, needed to attribute `30-1112`
            and to size overrides against the rejected region.
        units: The parsed `finn_floorplan_units.json` (measured demands and the
            cells of every unit, which attribute `30-488`).

    Returns:
        `{pblock_name: {demand_key: shortfall}}`. A pblock rejected on several
        resources gets one entry per resource; repeated messages for one resource
        keep the largest. Empty when the log holds no placer rejection.
    """
    out: dict[str, dict[str, int]] = {}
    for m in _REJECTION.finditer(log_text):
        key = _PLACER_KIND_TO_DEMAND.get(m["kind"])
        if key is None:
            log.warning(f"Placer rejected {m['pblock']} on {m['kind']}, which has no demand key")
            continue
        short = int(m["required"]) - int(m["available"])
        if short <= 0:
            continue
        extra = _beyond_region(report, units, m["pblock"], key, short)
        per = out.setdefault(m["pblock"], {})
        per[key] = max(per.get(key, 0), extra)

    for m in _COMMIT_FAILURE.finditer(log_text):
        failed: dict[str, dict[str, int]] = {}
        for inst in _COMMIT_INSTANCE.finditer(m["body"]):
            key = _COMMIT_TYPE_TO_DEMAND.get(inst["type"])
            name = _unit_of_instance(inst["inst"], units)
            if key is None or name is None:
                log.warning(
                    f"Placer could not commit {inst['inst']} ({inst['type']}), and it belongs "
                    "to no floorplan unit with a demand for it; cannot attribute the rejection"
                )
                continue
            per = failed.setdefault(name, {})
            per[key] = per.get(key, 0) + 1
        for name, per_key in failed.items():
            entry = (report or {}).get("units", {}).get(name, {})
            cap = entry.get("pblock_capacity", {})
            for key, n in per_key.items():
                extra = _past_capacity(units, name, key, int(cap.get(key, 0)), n)
                per = out.setdefault(name, {})
                per[key] = max(per.get(key, 0), extra)

    for m in _MACRO_FAILURE.finditer(log_text):
        shapes: dict[str, int] = {}
        for inst in _MACRO_INSTANCE.finditer(m["body"]):
            name = _unit_of_instance(inst["inst"], units)
            if name is None:
                log.warning(
                    f"Placer could not commit the shape around {inst['inst']}, which belongs "
                    "to no floorplan unit; cannot attribute the rejection"
                )
                continue
            shapes[name] = shapes.get(name, 0) + 1
        for name, n in shapes.items():
            cap = (report or {}).get("units", {}).get(name, {}).get("pblock_capacity", {})
            # A shape spans whole slices; grow the region by the same step as for
            # loose logic, at least one slice per shape that did not fit.
            extra = _past_capacity(units, name, "luts", int(cap.get("luts", 0)), n * SLICE_LUTS)
            per = out.setdefault(name, {})
            per["luts"] = max(per.get("luts", 0), extra)

    for m in _LUTRAM_PACK_FAILURE.finditer(log_text):
        memory: dict[str, int] = {}
        for inst in _LUTRAM_PACK_INSTANCE.finditer(m["body"]):
            name = _unit_of_instance(inst["inst"], units)
            if name is None:
                log.warning(
                    f"Placer could not pack {inst['inst']} ({inst['type']}) into a SLICEM, and "
                    "it belongs to no floorplan unit; cannot attribute the rejection"
                )
                continue
            memory[name] = memory.get(name, 0) + _LUTRAM_TYPE_LUTS.get(inst["type"], 1)
        for name, luts in memory.items():
            cap = (report or {}).get("units", {}).get(name, {}).get("pblock_capacity", {})
            extra = _past_capacity(units, name, "lutrams", int(cap.get("lutrams", 0)), luts)
            per = out.setdefault(name, {})
            per["lutrams"] = max(per.get("lutrams", 0), extra)

    for m in _LUTRAM_REJECTION.finditer(log_text):
        rng = f"SLICE_X{m['x0']}Y{m['y0']}:SLICE_X{m['x1']}Y{m['y1']}"
        hit = _pblock_with_slice_range(report, rng)
        if hit is None:
            log.warning(
                f"Placer found no legal LUTRAM site for {m['inst']} inside {rng}, but no "
                "floorplan pblock carries that range; cannot attribute the rejection"
            )
            continue
        name, entry = hit
        offered = int(entry.get("pblock_capacity", {}).get("lutrams", 0))
        extra = max(1, int(offered * LUTRAM_RETRY_GROWTH))
        per = out.setdefault(name, {})
        per["lutrams"] = max(per.get("lutrams", 0), extra)
    return out


_SLICE_RANGE = re.compile(r"SLICE_X(\d+)Y(\d+):SLICE_X(\d+)Y(\d+)")


def _slice_bbox(text: str) -> tuple[int, int, int, int] | None:
    """Bounding box (x0, y0, x1, y1) over every SLICE range in a ranges string."""
    boxes = [tuple(int(v) for v in m.groups()) for m in _SLICE_RANGE.finditer(text)]
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _past_capacity(
    units: list[dict] | None, name: str, key: str, capacity: int, at_least: int
) -> int:
    """Demand increase after a commit failure: past the rejected region by one step.

    The new demand is `COMMIT_RETRY_GROWTH` beyond what the rejected pblock
    offered, so the packer has to build a larger region. Adding the step to the
    demand instead left a small unit inside its old region: unit 136 of
    mobilenetv1 (46 LUTs in 10 slices) failed, was raised to 66 LUTs, and got the
    same 10 slices again (job 3577610, 4 Oct 2026). Without the unit's exported
    demand the step alone is added.
    """
    step = int(capacity * COMMIT_RETRY_GROWTH)
    unit = next((u for u in units or [] if u.get("name") == name), None)
    if unit is None or key not in unit.get("demands", {}):
        return max(at_least, step)
    demand = int(unit["demands"][key])
    return max(at_least, capacity + step - demand)


def _beyond_region(
    report: dict | None, units: list[dict] | None, pblock: str, key: str, short: int
) -> int:
    """Demand increase that puts a unit past the capacity of its rejected region.

    Falls back to the bare shortfall when the report or the demands are missing.
    """
    if not report or not units:
        return short
    entry = report.get("units", {}).get(pblock)
    unit = next((u for u in units if u.get("name") == pblock), None)
    cap_key = _CAPACITY_KEY.get(key)
    if not entry or not unit or not cap_key:
        return short
    capacity = int(entry.get("pblock_capacity", {}).get(cap_key, 0))
    demand = int(unit.get("demands", {}).get(key, 0))
    return max(short, capacity + short - demand)


def _pblock_with_slice_range(report: dict | None, slice_range: str) -> tuple[str, dict] | None:
    """Find the report entry whose SLICE ranges span exactly the quoted box, if any.

    The placer quotes one box per instance. A pblock may consist of several SLICE
    ranges (the packer claims columns selectively), so the comparison is on the
    bounding box of all of them, which is what Vivado prints.
    """
    if not report:
        return None
    wanted = _slice_bbox(slice_range)
    for name, entry in report.get("units", {}).items():
        ranges = f"{entry.get('grid_ranges', '')} {entry.get('requested_ranges', '')}"
        if slice_range in ranges.split() or (wanted and _slice_bbox(ranges) == wanted):
            return name, entry
    return None


def _unit_of_instance(instance: str, units: list[dict] | None) -> str | None:
    """Find the unit whose cell is an ancestor of a placer instance, by hierarchical name.

    Unit cells are leaf names (``StreamingDataflowPartition_2_ConvolutionInputGenerator_rtl_4``)
    that appear as one component of the instance's full path; the longest match
    wins, so a unit nested inside another's hierarchy is attributed correctly.
    """
    parts = set(instance.split("/"))
    best: tuple[int, str] | None = None
    for unit in units or []:
        for cell in unit.get("cells", []):
            leaf = cell.rsplit("/", 1)[-1]
            if leaf in parts and (best is None or len(leaf) > best[0]):
                best = (len(leaf), unit["name"])
    return best[1] if best else None


def write_overrides(path: Path, rejections: dict[str, dict[str, int]]) -> None:
    """Write the shortfalls as the Tcl dict the geometry engine sources.

    Args:
        path: Where the engine expects `finn_fp_overrides`; see
            `finn_fp_load_overrides` in finn_floorplanning.tcl.
        rejections: As returned by `parse_placer_rejections`.
    """
    entries = " ".join(
        f"{pblock} {{{' '.join(f'{k} {v}' for k, v in sorted(per.items()))}}}"
        for pblock, per in sorted(rejections.items())
    )
    path.write_text(f"set finn_fp_overrides {{{entries}}}\n")


#: Set in the retry script to tell the hook that the attempt resumes at
#: place_design from the optimised checkpoint. The hook then reads the first
#: attempt's census and measured units back instead of measuring the
#: (unchanged) netlist again, and runs the placer in ultrathreads mode: Vitis
#: switches that mode off (`place.ultrathreadsUsed 0` in its run script), and
#: the placer it leaves behind miscounts the DSP and block-RAM sub-cells of a
#: reopened checkpoint against the pblocks (a unit with 64 DSPs is reported at
#: 711 % of a 72-site pblock, 8 sub-cells per DSP) and rejects floorplans it
#: placed in the first attempt. Measured 12 Oct 2026 on the U55C: the same
#: floorplan placed in all six resumed runs with ultrathreads on and in none of
#: the four with it off; in-session (first attempt) both modes place it.
RESUMED_ATTEMPT_ENV: str = "FINN_FP_RESUMED_ATTEMPT"


#: The v++ link step a retry resumes at: the implementation run's place_design,
#: which Vivado restarts from the optimised checkpoint (`reset_runs -from_step`).
#: The checkpoint exists because the link configuration turns
#: project.writeIntermediateCheckpoints on; measured 12 Oct 2026 on the U55C:
#: opening it takes two minutes, against five for re-linking the design plus two
#: to four for opt_design, which `vpl.impl` would repeat.
RETRY_FROM_STEP: str = "vpl.impl.place_design"


def retry_script(run_script: Path) -> Path:
    """Derive a run script that resumes at placement instead of starting over.

    Synthesis, system linking and opt_design are unchanged by a pblock resize,
    and v++ can resume a saved project at the implementation run's place_design
    step (`RETRY_FROM_STEP`), which is where the hook that sources the engine
    runs. The hook is told that the attempt is resumed (`RESUMED_ATTEMPT_ENV`):
    it reuses the first attempt's census and measurements and selects the
    placer mode that handles a reopened checkpoint.

    Args:
        run_script: The original `run_link.sh`.

    Returns:
        Path of the derived script, written next to the original.
    """
    out = run_script.with_name("run_link_retry.sh")
    lines = []
    for line in run_script.read_text().splitlines():
        if line.lstrip().startswith("v++") and "--from_step" not in line:
            lines.append(f"export {RESUMED_ATTEMPT_ENV}=1")
            line = f"{line} --from_step {RETRY_FROM_STEP}"
        lines.append(line)
    out.write_text("\n".join(lines) + "\n")
    out.chmod(0o755)
    return out

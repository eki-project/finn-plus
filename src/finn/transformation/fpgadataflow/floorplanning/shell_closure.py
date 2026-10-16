"""Sites a platform shell closes to the placer: measure once, cache, exclude.

On a shell-based (DFX) platform such as an Alveo card, Vivado's placer closes
some sites inside the user region as unroutable (``place.closeUnroutableSites``):
the locked static routing occupies the wires those sites would need. A pblock the
floorplanner sized over such sites comes up short in global placement
(``[Place 30-859]``). Which sites close depends only on the platform shell and the
Vivado release -- the U55C shell closes 13 of the 24 block RAM sites on SLR0's two
edge columns in every link, whatever the user design -- so FINN measures them
once per platform and keeps them out of the device census.

The measurement (`map_shell_closure`) is empirical and needs nothing but the
platform: it extracts the locked shell checkpoint from the platform's hardware
archive, fills the empty user region with unconnected probe cells confined to
hard pblocks, and reads the placer's rejections. A first pass probes whole site
columns; a second pass probes single sites in the columns that lost any. The
result is a JSON map, one file per platform and Vivado release, kept in this
package's ``shell_closure_maps/`` directory or in any directory on
``FINN_SHELL_CLOSURE_PATH``.

Resolution is one placement tile: a single-site pblock covers its whole tile
(Vivado widens pblock ranges to tiles), so the two DSP48E2 sites of a DSP tile
are reported together, and a tile with one closed site lists both. Pblocks are
built from whole tiles anyway, so that is also the finest grain a floorplan can
use. Block RAM (one RAMB36 per tile) is exact.

Map a new platform (or a new Vivado release) with::

    finn floorplan map-shell --platform xilinx_u55c_gen3x16_xdma_3_202210_1

or, without the FINN command line, ``python -m
finn.transformation.fpgadataflow.floorplanning.shell_closure map --platform ...``.
Vivado must be on PATH and the platform installed (PLATFORM_REPO_PATHS or
/opt/xilinx/platforms).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

#: Site kinds the probe covers, in the order they are probed. RAMB18 is not
#: probed on its own: a closed RAMB36 site takes both RAMB18 halves of its tile,
#: and the floorplanning hook excludes them together.
KINDS: tuple[str, ...] = ("RAMB36", "DSP48E2", "URAM288", "SLICE", "CARRY8")

#: Maps shipped with FINN.
SHIPPED_DIR = Path(__file__).parent / "shell_closure_maps"

#: Extra directories (``os.pathsep``-separated) searched before the shipped maps.
PATH_ENV = "FINN_SHELL_CLOSURE_PATH"

#: Upper bound on probe cells in one place_design run. A run that still trips
#: the placer's utilisation guard (``[Place 30-4]``, which counts a DSP48E2 as
#: eight sub-cells) is split in half and retried, so this only sets where the
#: splitting starts.
DEFAULT_MAX_CELLS: dict[str, int] = {
    "RAMB36": 4000,
    "DSP48E2": 1000,
    "URAM288": 2000,
    "SLICE": 160000,
    "CARRY8": 40000,
}

#: Slices per CARRY8 probe window: one hard pblock over this many vertically
#: adjacent slices of a column, holding one carry chain of the same length. The
#: closure next to the static shell shows on carry *chains*; a lone CARRY8 per
#: slice is a shape the placer rejects before it reports anything else.
CARRY_WINDOW = 2

#: The search for closed carry windows stops at runs of this many windows (30
#: slices): a run that still loses chains is recorded as closed as a whole.
#: Measured on the U55C shell (3 Oct 2026): in the affected edge columns 45-73 %
#: of all windows reject a chain and every 30-slice stretch loses some, while
#: below that size each fully closed piece aborts a whole placement run (the
#: placer stops at the first shape it cannot place) and is found one per run.
CARRY_MIN_RUN = 15

#: Probe cells per run below the first carry level. Every fully closed run there
#: aborts its placement and is found alone, so small batches let `--jobs` find
#: them in parallel instead of one after another.
CARRY_DEEP_MAX_CELLS = 600

#: Probe cells per site in a column pblock. A SLICE probe cell is one LUT1, an
#: eighth of a slice, so a column is filled with eight per site to be
#: over-subscribed by a single closed slice. (CARRY8, one per slice, cannot be
#: used: a CARRY8 with a floating carry input fails to commit on its own.) A
#: one-site pblock needs one cell: it is empty once its site is closed.
COLUMN_FILL: dict[str, int] = {"SLICE": 8}

#: Reported for a site that was isolated by bisection: a one-site pblock that
#: failed placement without a closure rejection.
UNUSABLE = -1

FORMAT_VERSION = 1

_TCL = Path(__file__).resolve().parents[3] / "util" / "vivado_scripts" / "finn_shell_closure.tcl"

#: One ``[Place 30-859]`` rejection: the pblock and the sites the closure took.
_REJECTION = re.compile(
    r"PBlock name is: (?P<pblock>\S+)\s*\n"
    r"(?:(?!PBlock name is:).)*?"
    r"blocked with place\.closeUnroutableSites param: (?P<blocked>\d+)",
    re.S,
)
_UTILISATION_GUARD = re.compile(r"\[Place 30-4\]")

#: A shape the placer finds no legal place for while it initialises; it stops
#: there, so every other pblock of the run is unknown.
_SHAPE_ABORT = re.compile(r"Could not place shape in pblock (?P<pblock>\S+?)\.?\s*$", re.M)

#: A probe cell named in a commit failure (`[Place 30-1153]` macros, `[Place 30-488]`
#: instances): ``.../finn_cs_<pblock>_<index>``.
_PROBE_CELL = re.compile(r"/finn_cs_(?P<pblock>cs_\w+?)_\d+ with block Id")


def shell_closure_tcl_path() -> Path:
    """Path of the Vivado-side probe script."""
    return _TCL


@dataclass
class ShellClosureMap:
    """The sites one platform shell closes, as measured under one Vivado release.

    Attributes:
        platform: Vitis platform name (``xilinx_u55c_gen3x16_xdma_3_202210_1``).
        part: The FPGA part the shell is built for.
        vivado: Vivado release the probe ran under (``2024.2``).
        kinds: Site kinds that were probed; a kind not listed was not measured.
        closed: Closed site names per kind.
        probed_sites: Available sites per kind inside the user region.
        generated: ISO date of the measurement.
        checkpoint: The checkpoint that was probed, for the record.
        column_rule: Per kind, the sites `close_partial_columns` added on top of
            the measured ones (they are included in ``closed``).
    """

    platform: str
    part: str
    vivado: str
    kinds: list[str]
    closed: dict[str, list[str]]
    probed_sites: dict[str, int] = field(default_factory=dict)
    generated: str = ""
    checkpoint: str = ""
    column_rule: dict[str, int] = field(default_factory=dict)

    @property
    def all_closed(self) -> list[str]:
        """Every closed site, kinds concatenated in probe order."""
        return [s for k in KINDS for s in self.closed.get(k, [])]

    def to_dict(self) -> dict:
        """JSON form (see `from_dict`)."""
        return {
            "format": FORMAT_VERSION,
            "platform": self.platform,
            "part": self.part,
            "vivado": self.vivado,
            "kinds": list(self.kinds),
            "probed_sites": dict(self.probed_sites),
            "closed": {k: sorted(v, key=_site_key) for k, v in self.closed.items()},
            "generated": self.generated,
            "checkpoint": self.checkpoint,
            "column_rule": dict(self.column_rule),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ShellClosureMap:
        """Read the JSON form written by `to_dict`."""
        if int(data.get("format", 0)) != FORMAT_VERSION:
            raise ValueError(f"unsupported shell closure map format {data.get('format')!r}")
        return cls(
            platform=str(data["platform"]),
            part=str(data.get("part", "")),
            vivado=str(data.get("vivado", "")),
            kinds=[str(k) for k in data.get("kinds", [])],
            closed={str(k): [str(s) for s in v] for k, v in data.get("closed", {}).items()},
            probed_sites={str(k): int(v) for k, v in data.get("probed_sites", {}).items()},
            generated=str(data.get("generated", "")),
            checkpoint=str(data.get("checkpoint", "")),
            column_rule={str(k): int(v) for k, v in data.get("column_rule", {}).items()},
        )

    def file_name(self) -> str:
        """Cache file name: one per platform and Vivado release."""
        return map_file_name(self.platform, self.vivado)


def map_file_name(platform: str, vivado: str) -> str:
    """Cache file name for a platform measured under a Vivado release (``2024.2``)."""
    return f"{platform}__vivado-{vivado_release(vivado)}.json"


def vivado_release(version: str) -> str:
    """Return the release part of a Vivado version: ``2024.2.2`` -> ``2024.2``.

    Maps are keyed by release; the patch level is recorded in the map but only
    visible inside Vivado, so the hook compares it and warns on a mismatch.
    """
    return ".".join(version.split(".")[:2])


def _site_key(name: str) -> tuple:
    """Sort key for site names: prefix, then X and Y numerically."""
    m = re.match(r"(.*)_X(\d+)Y(\d+)$", name)
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else (name, 0, 0)


# -- finding the platform's shell -------------------------------------------------


def platform_search_dirs() -> list[Path]:
    """Where Vitis looks for platforms: PLATFORM_REPO_PATHS, then /opt/xilinx/platforms."""
    dirs = [Path(p) for p in os.environ.get("PLATFORM_REPO_PATHS", "").split(os.pathsep) if p]
    dirs.append(Path("/opt/xilinx/platforms"))
    return dirs


def find_platform_xsa(platform: str, search: list[Path] | None = None) -> Path:
    """Locate a platform's hardware archive (``hw/hw.xsa`` next to its ``.xpfm``).

    PLATFORM_REPO_PATHS entries may name the platform directory itself or a
    directory of platforms; both are accepted.

    Raises:
        FileNotFoundError: If the platform is not installed in any search path.
    """
    candidates: list[Path] = []
    for d in search if search is not None else platform_search_dirs():
        candidates += [d, d / platform]
    for d in candidates:
        if (d / f"{platform}.xpfm").is_file():
            xsas = sorted((d / "hw").glob("*.xsa")) if (d / "hw").is_dir() else []
            if xsas:
                return xsas[0]
    raise FileNotFoundError(
        f"platform {platform!r} not found (searched {', '.join(str(c) for c in candidates)}); "
        "set PLATFORM_REPO_PATHS or pass --checkpoint"
    )


def extract_locked_checkpoint(xsa: Path, dest_dir: Path) -> Path:
    """Extract the locked shell checkpoint (``*_bb_locked.dcp``) from a hardware archive.

    Raises:
        FileNotFoundError: If the archive holds no locked checkpoint, i.e. the
            platform is not a shell-based (DFX) one.
    """
    with zipfile.ZipFile(xsa) as z:
        names = [n for n in z.namelist() if n.endswith("_bb_locked.dcp")]
        if not names:
            raise FileNotFoundError(f"{xsa} holds no *_bb_locked.dcp; not a shell-based platform")
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / Path(names[0]).name
        if not target.is_file() or target.stat().st_size != z.getinfo(names[0]).file_size:
            with z.open(names[0]) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        return target


# -- parsing ----------------------------------------------------------------------


@dataclass(frozen=True)
class Site:
    """One available site of the user region, as enumerated by the probe script."""

    kind: str
    name: str
    slr: int
    x: int
    y: int


def parse_enumeration(text: str) -> tuple[dict[str, str], list[Site]]:
    """Parse the ``enumerate`` output: INFO key/value pairs and SITE lines."""
    info: dict[str, str] = {}
    sites: list[Site] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "INFO":
            info[parts[1]] = " ".join(parts[2:])
        elif len(parts) == 6 and parts[0] == "SITE":
            sites.append(Site(parts[1], parts[2], int(parts[3]), int(parts[4]), int(parts[5])))
    return info, sites


def parse_rejections(log_text: str) -> dict[str, int]:
    """Pblocks the placer rejected, with the sites the closure took from each.

    Only rejections attributed to ``place.closeUnroutableSites`` count; a pblock
    rejected for another reason (a DSP48E2 counts as eight sub-cells against one
    site, so every DSP probe pblock is "over-subscribed") has no closure line and
    is not reported here.
    """
    out: dict[str, int] = {}
    for block in re.split(r"(?=ERROR: \[Place 30-859\])", log_text):
        if "[Place 30-859]" not in block:
            continue
        m = _REJECTION.search(block)
        if m and int(m["blocked"]) > 0:
            out[m["pblock"]] = max(out.get(m["pblock"], 0), int(m["blocked"]))
    return out


def count_chain_failures(log_text: str, window: int = CARRY_WINDOW) -> dict[str, int]:
    """Carry chains that failed to commit, counted per probe pblock.

    Probe cells are numbered per pblock and grouped into chains of ``window``
    cells, so a named cell's index identifies its chain.
    """
    chains: dict[str, set[int]] = defaultdict(set)
    for block in re.split(r"(?=ERROR: \[Place 30-(?:1153|488)\])", log_text):
        if re.match(r"ERROR: \[Place 30-(?:1153|488)\]", block):
            for m in re.finditer(r"/finn_cs_(?P<pb>cs_\w+?)_(?P<i>\d+) with block Id", block):
                chains[m["pb"]].add(int(m["i"]) // window)
    return {pb: len(ids) for pb, ids in chains.items()}


def parse_shape_failures(log_text: str) -> tuple[set[str], set[str]]:
    """Pblocks whose probe shapes the placer could not place, for the CARRY8 probe.

    Returns:
        The pblocks the placer aborted on while initialising (`[Place 30-642]`;
        the rest of the run is then unknown), and the pblocks whose cells it
        failed to commit in detail placement (`[Place 30-1153]`, `[Place 30-488]`;
        every other pblock of the run placed).
    """
    aborted = {m["pblock"] for m in _SHAPE_ABORT.finditer(log_text)}
    failed: set[str] = set()
    for block in re.split(r"(?=ERROR: \[Place 30-(?:1153|488)\])", log_text):
        if re.match(r"ERROR: \[Place 30-(?:1153|488)\]", block):
            failed |= {m["pblock"] for m in _PROBE_CELL.finditer(block)}
    return aborted, failed


# -- planning probe runs ------------------------------------------------------------


def column_requests(sites: list[Site]) -> list[tuple[str, str, list[str]]]:
    """One pblock per (kind, SLR, column), holding all of the column's sites."""
    groups: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    for s in sites:
        groups[(s.kind, s.slr, s.x)].append(s.name)
    return [
        (kind, f"cs_{kind}_S{slr}_X{x}", sorted(names, key=_site_key))
        for (kind, slr, x), names in sorted(groups.items())
    ]


def carry_runs(sites: list[Site], window: int = CARRY_WINDOW) -> list[list[list[str]]]:
    """Split every column's slices into runs of adjacent rows, and each run into windows.

    A window is ``window`` vertically adjacent slices (the leftover of a run joins
    its last window; a run shorter than a window is one window). Windows never
    span a gap in a column's rows.

    Returns:
        One list of windows (each a list of site names) per run, ordered by SLR,
        column and row.
    """
    groups: dict[tuple[int, int], list[Site]] = defaultdict(list)
    for s in sites:
        groups[(s.slr, s.x)].append(s)
    out: list[list[list[str]]] = []
    for key in sorted(groups):
        col = sorted(groups[key], key=lambda s: s.y)
        runs: list[list[Site]] = []
        for s in col:
            if runs and s.y == runs[-1][-1].y + 1:
                runs[-1].append(s)
            else:
                runs.append([s])
        for run in runs:
            chunks = [[s.name for s in run[i : i + window]] for i in range(0, len(run), window)]
            if len(chunks) > 1 and len(chunks[-1]) < window:
                tail = chunks.pop()
                chunks[-1].extend(tail)
            out.append(chunks)
    return out


def carry_requests(
    sites: list[Site], window: int = CARRY_WINDOW
) -> list[tuple[str, str, list[str]]]:
    """One pblock per carry window (see `carry_runs`)."""
    return [("CARRY8", f"cs_CARRY8_{w[0]}", w) for run in carry_runs(sites, window) for w in run]


def carry_segment_request(windows: list[list[str]]) -> tuple[str, str, list[str]]:
    """One pblock over a run of adjacent carry windows, named by its first site and size."""
    return ("CARRY8", f"cs_CARRY8_{windows[0][0]}_{len(windows)}", [s for w in windows for s in w])


def site_requests(sites: list[Site]) -> list[tuple[str, str, list[str]]]:
    """One pblock per site."""
    ordered = sorted(sites, key=lambda s: _site_key(s.name))
    return [(s.kind, f"cs_{s.name}", [s.name]) for s in ordered]


def probe_cells(phase: str, kind: str, sites: list[str]) -> int:
    """Probe cells a pblock over ``sites`` gets: enough to fill them (`COLUMN_FILL`)."""
    if kind == "CARRY8":
        return len(sites)
    return len(sites) * (COLUMN_FILL.get(kind, 1) if phase == "columns" else 1)


def batch_requests(
    requests: list[tuple[str, str, list[str]]], max_cells: int, phase: str = "sites"
) -> list[list[tuple[str, str, list[str]]]]:
    """Split pblock requests into runs of at most ``max_cells`` probe cells."""
    batches: list[list[tuple[str, str, list[str]]]] = []
    current: list[tuple[str, str, list[str]]] = []
    n = 0
    for req in requests:
        cells = probe_cells(phase, req[0], req[2])
        if current and n + cells > max_cells:
            batches.append(current)
            current, n = [], 0
        current.append(req)
        n += cells
    if current:
        batches.append(current)
    return batches


# -- running Vivado -----------------------------------------------------------------


class _Runner:
    """Runs the probe script in Vivado batch mode, one working directory per run."""

    def __init__(
        self, checkpoint: Path, work_dir: Path, vivado: str, log: Callable[[str], None]
    ) -> None:
        """Remember the checkpoint, the scratch directory, Vivado and the logger."""
        self.checkpoint = checkpoint
        self.work_dir = work_dir
        self.vivado = vivado
        self.log = log
        self._n = 0

    def _run(self, tag: str, args: list[str]) -> str:
        """Run the probe script in ``work_dir/tag`` and return the Vivado log."""
        run_dir = self.work_dir / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.vivado, "-mode", "batch", "-nojournal", "-log", "vivado.log",
            "-source", str(shell_closure_tcl_path()), "-tclargs", *args,
        ]  # fmt: skip
        with (run_dir / "stdout.txt").open("w") as out:
            subprocess.run(cmd, cwd=run_dir, stdout=out, stderr=subprocess.STDOUT, check=False)
        log_file = run_dir / "vivado.log"
        return log_file.read_text(errors="ignore") if log_file.exists() else ""

    def enumerate(self) -> tuple[dict[str, str], list[Site]]:
        """List the user region's available sites per kind."""
        out = self.work_dir / "enumerate" / "sites.txt"
        self.log("enumerating the user region's sites")
        text = self._run("enumerate", ["enumerate", str(self.checkpoint), str(out)])
        if not out.exists():
            raise RuntimeError(
                f"site enumeration failed; see {out.parent / 'vivado.log'}\n{text[-2000:]}"
            )
        return parse_enumeration(out.read_text())

    def run_batch(self, phase: str, batch: list[tuple[str, str, list[str]]]) -> tuple[str, str]:
        """Run one probe batch as is and return its tag and Vivado log.

        The run directory is named after a hash of the request, and a finished run
        is not repeated: a mapping that was interrupted resumes where it stopped
        when started again with the same work directory.
        """
        request = "".join(
            f"{k} {n} {probe_cells(phase, k, sites)} {' '.join(sites)}\n" for k, n, sites in batch
        )
        tag = f"{phase}_{hashlib.sha1(request.encode()).hexdigest()[:12]}"
        req = self.work_dir / tag / "request.txt"
        cells = sum(probe_cells(phase, k, sites) for k, _, sites in batch)
        log_file = req.parent / "vivado.log"
        if req.is_file() and req.read_text() == request and log_file.is_file():
            done = log_file.read_text(errors="ignore")
            if "FINN_CS place_rc" in done:
                self.log(f"  {tag}: {len(batch)} pblock(s), {cells} probe cell(s) (done earlier)")
                return tag, done
        req.parent.mkdir(parents=True, exist_ok=True)
        req.write_text(request)
        self.log(f"  {tag}: {len(batch)} pblock(s), {cells} probe cell(s)")
        text = self._run(
            tag, ["probe", str(self.checkpoint), str(req), str(req.parent / "result.txt")]
        )
        if "FINN_CS place_rc" not in text:
            raise RuntimeError(
                f"probe run {tag} did not reach placement; see {req.parent / 'vivado.log'}"
            )
        return tag, text

    def probe(self, phase: str, batch: list[tuple[str, str, list[str]]]) -> dict[str, int]:
        """Probe one batch; split it in half while the utilisation guard trips.

        A run that fails without naming a closed site (a failed place_design
        unplaces every cell, so it cannot say which pblock was at fault) is
        bisected until the failing pblocks are isolated; each is reported as
        `UNUSABLE`. In the column pass that sends the column to the site pass; in
        the site pass it records the site.
        """
        tag, text = self.run_batch(phase, batch)
        if _UTILISATION_GUARD.search(text):
            if len(batch) == 1:
                raise RuntimeError(f"probe run {tag}: one pblock alone trips the utilisation guard")
            half = len(batch) // 2
            self.log(f"  {tag}: utilisation guard tripped; splitting")
            return {**self.probe(phase, batch[:half]), **self.probe(phase, batch[half:])}
        found = parse_rejections(text)
        failed = "FINN_CS place_rc 1" in text
        # A DSP probe always "fails" (the placer counts a DSP48E2 as eight
        # sub-cells against one site); any other kind failing without a single
        # closure line failed for a reason this probe does not understand.
        if failed and not found and any(k != "DSP48E2" for k, _, _ in batch):
            if len(batch) == 1:
                self.log(f"  {tag}: {batch[0][1]} cannot be placed; recorded as unusable")
                return {batch[0][1]: UNUSABLE}
            half = len(batch) // 2
            self.log(f"  {tag}: placement failed without a closure rejection; bisecting")
            return {**self.probe(phase, batch[:half]), **self.probe(phase, batch[half:])}
        return found


def _probe_carry_batch(runner: _Runner, batch: list[tuple[str, str, list[str]]]) -> dict[str, int]:
    """Probe one batch of carry pblocks; return failed chains per pblock.

    A pblock the placer finds no legal place for while it initialises stops the
    whole run: it is reported as ``-1`` (every window in it is closed) and the
    rest of the batch is probed again without it.
    """
    result: dict[str, int] = {}
    remaining = list(batch)
    while remaining:
        tag, text = runner.run_batch("carry", remaining)
        aborted, _ = parse_shape_failures(text)
        names = {r[1] for r in remaining}
        if aborted & names:
            for name in aborted & names:
                result[name] = -1
            n = len(aborted & names)
            runner.log(f"  {tag}: {n} pblock(s) rejected at initialisation; re-probing")
            remaining = [r for r in remaining if r[1] not in aborted]
            continue
        counts = count_chain_failures(text)
        if "FINN_CS place_rc 1" in text and not any(counts.get(n) for n in names):
            runner.log(f"  WARNING: {tag}: carry probe failed without naming a chain")
        for name in names:
            result[name] = counts.get(name, 0)
        break
    return result


def _map_carry(runner: _Runner, sites: list[Site], max_cells: int, jobs: int) -> list[str]:
    """Find the carry windows the placer refuses, by halving the runs that lose chains.

    Level one probes every column run as one pblock filled with two-slice chains:
    a run that loses none is clean, one the placer rejects outright is closed as a
    whole, and one that loses some is halved for the next level, down to runs of
    `CARRY_MIN_RUN` windows, which are closed as a whole if they still lose chains.
    Clean columns, the bulk of the device, cost one placement.

    Returns:
        The slices of every closed window.
    """
    level = [w for w in carry_runs(sites) if w]
    closed: list[str] = []
    depth = 0
    while level:
        depth += 1
        requests = [carry_segment_request(seg) for seg in level]
        runner.log(f"  carry level {depth}: {len(level)} run(s), {sum(map(len, level))} window(s)")
        found: dict[str, int] = {}
        limit = max_cells if depth == 1 else min(max_cells, CARRY_DEEP_MAX_CELLS)
        batches = batch_requests(requests, limit, "carry")
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            for result in pool.map(lambda b: _probe_carry_batch(runner, b), batches):
                found.update(result)
        nxt: list[list[list[str]]] = []
        for seg, req in zip(level, requests, strict=True):
            fails = found.get(req[1], 0)
            if fails == 0:
                continue
            if fails < 0 or len(seg) <= CARRY_MIN_RUN or fails >= len(seg):
                closed += [s for w in seg for s in w]
                continue
            half = len(seg) // 2
            nxt += [seg[:half], seg[half:]]
        level = nxt
    return closed


def map_shell_closure(
    platform: str,
    checkpoint: Path | None = None,
    kinds: tuple[str, ...] = KINDS,
    work_dir: Path | None = None,
    vivado: str = "vivado",
    jobs: int = 1,
    max_cells: dict[str, int] | None = None,
    log: Callable[[str], None] = print,
) -> ShellClosureMap:
    """Measure which sites of a platform's user region the placer closes.

    Args:
        platform: Vitis platform name.
        checkpoint: The locked shell checkpoint; extracted from the installed
            platform when omitted.
        kinds: Site kinds to probe.
        work_dir: Scratch directory for the Vivado runs (kept for inspection).
        vivado: Vivado executable.
        jobs: Probe runs in parallel (each opens the shell checkpoint; plan on
            about 8 GB per run).
        max_cells: Per-kind starting batch size, see `DEFAULT_MAX_CELLS`.
        log: Progress callback.

    Returns:
        The closure map; `save_map` writes it where the floorplanner finds it.
    """
    unknown = sorted(set(kinds) - set(KINDS))
    if unknown:
        raise ValueError(f"unknown site kind(s) {unknown}; choose from {list(KINDS)}")
    work_dir = Path(work_dir or Path.cwd() / f"shell_closure_{platform}").absolute()
    work_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint is None:
        xsa = find_platform_xsa(platform)
        log(f"extracting the locked shell checkpoint from {xsa}")
        checkpoint = extract_locked_checkpoint(xsa, work_dir)
        source = f"{xsa} ({checkpoint.name})"
    else:
        source = str(Path(checkpoint).absolute())
    checkpoint = Path(checkpoint).absolute()
    limits = {**DEFAULT_MAX_CELLS, **(max_cells or {})}
    runner = _Runner(checkpoint, work_dir, vivado, log)

    info, sites = runner.enumerate()
    sites = [s for s in sites if s.kind in kinds]
    by_kind: dict[str, list[Site]] = defaultdict(list)
    for s in sites:
        by_kind[s.kind].append(s)
    log("user region: " + ", ".join(f"{len(by_kind[k])} {k}" for k in kinds))

    def run_all(phase: str, requests_by_kind: dict[str, list]) -> dict[str, int]:
        """Probe every request of a pass, batched per kind, and merge the rejections."""
        batches = [
            b
            for k, reqs in requests_by_kind.items()
            for b in batch_requests(reqs, limits[k], phase)
        ]
        found: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            for result in pool.map(lambda b: runner.probe(phase, b), batches):
                found.update(result)
        return found

    log("pass 1: one pblock per site column")
    col_reqs = {k: column_requests(by_kind[k]) for k in kinds if by_kind[k] and k != "CARRY8"}
    rejected_cols = run_all("columns", col_reqs)
    col_names = {name for reqs in col_reqs.values() for _, name, _ in reqs}
    hit = {name for name in rejected_cols if name in col_names}
    log(f"  {len(hit)} column(s) lost sites: {sorted(hit)}")

    suspects = [s for s in sites if f"cs_{s.kind}_S{s.slr}_X{s.x}" in hit]
    log(f"pass 2: one pblock per site, {len(suspects)} site(s) in those columns")
    site_reqs: dict[str, list] = defaultdict(list)
    for req in site_requests(suspects):
        site_reqs[req[0]].append(req)
    rejected_sites = run_all("sites", site_reqs)

    closed: dict[str, list[str]] = {k: [] for k in kinds}
    if by_kind.get("CARRY8"):
        log(f"carry pass: {CARRY_WINDOW}-slice carry chains, halving the runs that lose any")
        closed["CARRY8"] = _map_carry(runner, by_kind["CARRY8"], limits["CARRY8"], jobs)
        log(f"  {len(closed['CARRY8'])} slice(s) do not take a carry chain")
    for s in suspects:
        if f"cs_{s.name}" in rejected_sites:
            closed[s.kind].append(s.name)
    # Every column the first pass flagged must account for its loss in the second.
    for kind, reqs in col_reqs.items():
        for _, name, names in reqs:
            if name in hit and not any(n in closed[kind] for n in names):
                log(f"  WARNING: {name} lost sites as a column but no single site was closed")

    rule = close_partial_columns(closed, sites, tuple(k for k in COLUMN_RULE_KINDS if k in kinds))
    if any(rule.values()):
        log("column rule: " + ", ".join(f"+{n} {k}" for k, n in rule.items() if n))
    measured = list(kinds)
    result = ShellClosureMap(
        platform=platform,
        part=info.get("part", ""),
        vivado=info.get("vivado", ""),
        kinds=measured,
        closed={k: closed[k] for k in measured},
        probed_sites={k: len(by_kind[k]) for k in measured},
        generated=datetime.date.today().isoformat(),
        checkpoint=source,
        column_rule={k: n for k, n in rule.items() if n},
    )
    log("closed: " + ", ".join(f"{len(v)} {k}" for k, v in closed.items()))
    return result


#: Kinds whose columns are closed as a whole once the placer closes any of their
#: sites (`close_partial_columns`).
COLUMN_RULE_KINDS: tuple[str, ...] = ("RAMB36", "DSP48E2")


def close_partial_columns(
    closed: dict[str, list[str]], sites: list[Site], kinds: tuple[str, ...] = COLUMN_RULE_KINDS
) -> dict[str, int]:
    """Close the rest of every column (per SLR) the placer closes in part.

    The placer's closure is not the whole story next to the static shell: on the
    U55C it closes 13 of the 24 block RAM sites in SLR0's edge columns X12/X13
    and leaves 11 open -- and a weight memory placed on those 11 had 33 pins the
    router could not reach (`[Route 35-7]`, mobilenetv1, 4 Oct 2026). Decided the
    same day: a block RAM or DSP column that loses any site to the closure is
    excluded within its SLR as a whole (option one; a routing probe would measure
    it instead).

    Args:
        closed: Closed sites per kind; extended in place.
        sites: The enumerated sites of the user region.
        kinds: Kinds the rule applies to.

    Returns:
        Per kind, how many sites the rule added.
    """
    added: dict[str, int] = {}
    for kind in kinds:
        have = set(closed.get(kind, []))
        if not have:
            continue
        of_kind = [s for s in sites if s.kind == kind]
        cols = {(s.slr, s.x) for s in of_kind if s.name in have}
        extra = [s.name for s in of_kind if (s.slr, s.x) in cols and s.name not in have]
        if extra:
            closed[kind] = list(closed.get(kind, [])) + extra
        added[kind] = len(extra)
    return added


# -- the cache ----------------------------------------------------------------------


def search_dirs() -> list[Path]:
    """Directories searched for maps: FINN_SHELL_CLOSURE_PATH, then the shipped ones."""
    extra = [Path(p) for p in os.environ.get(PATH_ENV, "").split(os.pathsep) if p]
    return [*extra, SHIPPED_DIR]


def merge_maps(old: ShellClosureMap, new: ShellClosureMap) -> ShellClosureMap:
    """Add a run that probed some kinds to an existing map of the same platform and release.

    The kinds the new run probed replace the old results; the others are kept.

    Raises:
        ValueError: If the maps belong to different platforms or Vivado releases.
    """
    if old.platform != new.platform or vivado_release(old.vivado) != vivado_release(new.vivado):
        raise ValueError(
            f"cannot merge a {new.platform}/{new.vivado} map into {old.platform}/{old.vivado}"
        )
    kinds = [k for k in KINDS if k in old.kinds or k in new.kinds]
    closed = {k: (new if k in new.kinds else old).closed.get(k, []) for k in kinds}
    probed = {k: (new if k in new.kinds else old).probed_sites.get(k, 0) for k in kinds}
    return ShellClosureMap(
        platform=new.platform,
        part=new.part or old.part,
        vivado=new.vivado,
        kinds=kinds,
        closed=closed,
        probed_sites=probed,
        generated=new.generated,
        checkpoint=new.checkpoint,
        column_rule={
            k: v
            for k, v in {**old.column_rule, **new.column_rule}.items()
            if k in (new.column_rule if k in new.kinds else old.column_rule)
        },
    )


def save_map(closure: ShellClosureMap, out_dir: Path | None = None, merge: bool = True) -> Path:
    """Write a map to ``out_dir`` (default: the shipped directory) and return its path.

    With ``merge``, a run that probed only some kinds is merged into an existing
    map of the same platform and release (`merge_maps`) instead of replacing it.
    """
    out_dir = Path(out_dir or SHIPPED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / closure.file_name()
    if merge and path.is_file():
        existing, _ = _read_map(path)
        if existing is not None:
            closure = merge_maps(existing, closure)
    path.write_text(json.dumps(closure.to_dict(), indent=1) + "\n")
    return path


def _read_map(path: Path) -> tuple[ShellClosureMap | None, str]:
    """Read one map file; on failure return None and the reason."""
    try:
        return ShellClosureMap.from_dict(json.loads(path.read_text())), ""
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        return None, f"unreadable shell closure map {path}: {e}"


def load_map(
    platform: str, vivado: str | None = None, dirs: list[Path] | None = None
) -> tuple[ShellClosureMap | None, str]:
    """Find the closure map for a platform.

    Prefers a map measured under the given Vivado release; falls back to the
    newest map of the platform under another release, which the caller should
    flag, since the placer's closure rule may change between releases.

    Returns:
        The map (or None) and a short note on how it was chosen.
    """
    found: list[tuple[Path, ShellClosureMap]] = []
    for d in dirs if dirs is not None else search_dirs():
        if not d.is_dir():
            continue
        for path in sorted(d.glob(f"{platform}__vivado-*.json")):
            closure, error = _read_map(path)
            if closure is None:
                return None, error
            found.append((path, closure))
    if not found:
        return None, f"no shell closure map for {platform}"
    if vivado:
        for path, m in found:
            if vivado_release(m.vivado) == vivado_release(vivado):
                return m, f"{path}"
    path, m = max(found, key=lambda pm: pm[1].vivado)
    note = f"{path} (measured under Vivado {m.vivado}"
    note += f", running {vivado})" if vivado else ")"
    return m, note


def vivado_version() -> str | None:
    """Return the Vivado release on this machine, from XILINX_VIVADO (``.../Vivado/2024.2``)."""
    root = os.environ.get("XILINX_VIVADO", "")
    m = re.search(r"(\d{4}\.\d+)", Path(root).name or "") if root else None
    return m.group(1) if m else None


# -- command line ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``map`` a platform's closed sites into the cache, or ``show`` a cached map."""
    ap = argparse.ArgumentParser(prog="shell_closure", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map", help="measure a platform shell's closed sites and cache them")
    m.add_argument("--platform", required=True, help="Vitis platform name")
    m.add_argument(
        "--checkpoint", type=Path, help="locked shell checkpoint (default: from the platform)"
    )
    m.add_argument(
        "--kinds", default=",".join(KINDS), help=f"site kinds (default {','.join(KINDS)})"
    )
    m.add_argument("--out-dir", type=Path, help=f"where to write the map (default {SHIPPED_DIR})")
    m.add_argument("--work-dir", type=Path, help="scratch directory for the Vivado runs")
    m.add_argument("--jobs", type=int, default=1, help="parallel Vivado runs (about 8 GB each)")
    m.add_argument("--vivado", default="vivado", help="Vivado executable")
    m.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing map instead of merging the probed kinds into it",
    )
    s = sub.add_parser("show", help="print the cached map of a platform")
    s.add_argument("--platform", required=True)
    s.add_argument("--vivado", default=None, help="Vivado release (default: from XILINX_VIVADO)")
    args = ap.parse_args(argv)

    if args.cmd == "map":
        kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
        closure = map_shell_closure(
            log=lambda msg: print(msg, flush=True),
            platform=args.platform,
            checkpoint=args.checkpoint,
            kinds=kinds,
            work_dir=args.work_dir,
            vivado=args.vivado,
            jobs=args.jobs,
        )
        print(f"wrote {save_map(closure, args.out_dir, merge=not args.replace)}")
        return 0
    closure, note = load_map(args.platform, args.vivado or vivado_version())
    if closure is None:
        print(note)
        return 1
    print(f"{note}\npart {closure.part}, probed {closure.kinds}, generated {closure.generated}")
    for kind in closure.kinds:
        names = closure.closed.get(kind, [])
        print(f"  {kind}: {len(names)} of {closure.probed_sites.get(kind, '?')} closed")
        for name in sorted(names, key=_site_key):
            print(f"    {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

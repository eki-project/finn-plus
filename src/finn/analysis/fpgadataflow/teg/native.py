# Copyright (C) 2026, Paderborn University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Native backend of the TEG simulator.

``native/teg_sim.cpp`` is a transcription of ``simulate.py::_Simulator.run`` to C++ over flat
CSR arrays. It is compiled on first use with the system C++ compiler (like the XSI shim,
``finn/xsi/setup.py``) into ``FINN_BUILD_DIR/teg_native`` and called through ``ctypes``; the
results are bit-identical to the Python simulator (same event order, same stable-state
criterion), about 30-50x faster and with a fraction of the memory. Event and handshake
recording (``record_events``/``record_edges``) is not supported here; ``simulate`` falls back
to the Python backend for those.

The flattened static structure of a model is cached on the model object, so the search's
repeated simulations of one graph pay the flattening once.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import numpy as np
import os
import shutil
import subprocess
import tempfile
from ctypes import POINTER, c_double, c_int64
from dataclasses import dataclass
from itertools import chain as _chain
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log

if TYPE_CHECKING:
    from finn.analysis.fpgadataflow.teg.model import TEGModel
    from finn.analysis.fpgadataflow.teg.patterns import StallPattern
    from finn.analysis.fpgadataflow.teg.simulate import SimResult

_SOURCE = Path(__file__).resolve().parent / "native" / "teg_sim.cpp"
_PAT_KIND = {"none": 0, "bernoulli": 1, "bursty": 2, "single_stall": 3, "periodic": 4}
_MASK64 = (1 << 64) - 1
_CACHE_ATTR = "_teg_native_flat"

_LIB: ctypes.CDLL | None = None
_LIB_ERROR: str | None = None


# ----------------------------------------------------------------------------- compilation
def _compiler() -> str | None:
    return shutil.which("g++") or shutil.which("clang++")


def _cache_dir() -> Path:
    try:
        from finn.util.settings import get_settings

        base = Path(get_settings().finn_build_dir)
    except Exception:  # no FINN settings (e.g. plain scripts)
        base = Path(tempfile.gettempdir()) / "finn_teg_native"
    return base / "teg_native"


def _library_path() -> Path:
    cxx = _compiler()
    tag = hashlib.sha256(_SOURCE.read_bytes() + (cxx or "").encode()).hexdigest()[:16]
    return _cache_dir() / f"teg_sim_{tag}.so"


def _compile(so: Path) -> None:
    cxx = _compiler()
    if cxx is None:
        raise FINNUserError("No C++ compiler (g++/clang++) found for the native TEG simulator")
    so.parent.mkdir(parents=True, exist_ok=True)
    tmp = so.with_name(f".{so.name}.{os.getpid()}.tmp")
    cmd = [cxx, "-O3", "-std=c++17", "-shared", "-fPIC", "-o", str(tmp), str(_SOURCE)]
    log.info(f"Compiling the native TEG simulator: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise FINNUserError(f"Compiling {_SOURCE.name} failed:\n{proc.stderr}")
    tmp.replace(so)  # atomic: concurrent workers may compile at the same time


def library() -> ctypes.CDLL:
    """Return the loaded shared library, compiling it on first use."""
    global _LIB, _LIB_ERROR
    if _LIB is not None:
        return _LIB
    if _LIB_ERROR is not None:
        raise FINNUserError(_LIB_ERROR)
    try:
        so = _library_path()
        if not so.is_file():
            _compile(so)
        lib = ctypes.CDLL(str(so))
        p = POINTER(c_int64)
        lib.teg_sim_run.restype = c_int64
        lib.teg_sim_run.argtypes = (
            [c_int64]
            + [p] * 18
            + [c_int64]  # chains
            + [c_int64]
            + [p] * 7  # edges
            + [c_int64, p, POINTER(c_double), p]  # patterns
            + [c_int64] * 5  # options
            + [p, p, p]  # outputs
        )
        lib.teg_sim_fetch.restype = None
        lib.teg_sim_fetch.argtypes = [p, p]
    except Exception as exc:
        _LIB_ERROR = str(exc)
        raise
    _LIB = lib
    return lib


def available() -> bool:
    """Return True when the native backend can be compiled and loaded on this machine."""
    try:
        library()
    except Exception:
        return False
    return True


# ----------------------------------------------------------------------------- flattening
@dataclass
class _Flat:
    """Static CSR representation of a model (everything except the depths)."""

    chain_names: list[str]
    edge_names: list[str]
    arrays: dict[str, np.ndarray]
    pat_p: np.ndarray
    ngroups: int
    npat: int


def _csr_from_tuples(seqs: list[tuple[str, ...]], eid: dict[str, int]) -> tuple[np.ndarray, ...]:
    """CSR ``(lens, vals)`` of a list of edge-name tuples (many are the same object)."""
    tid: dict[int, int] = {}
    tuples: list[tuple[str, ...]] = []
    ids = np.empty(len(seqs), dtype=np.int64)
    for j, s in enumerate(seqs):
        k = tid.get(id(s))
        if k is None:
            k = len(tuples)
            tid[id(s)] = k
            tuples.append(s)
        ids[j] = k
    if not tuples:
        return np.zeros(len(seqs), dtype=np.int64), np.zeros(0, dtype=np.int64)
    width = max(len(s) for s in tuples)
    table = np.full((len(tuples), max(width, 1)), -1, dtype=np.int64)
    for k, s in enumerate(tuples):
        for j, e in enumerate(s):
            table[k, j] = eid[e]
    lens = np.array([len(s) for s in tuples], dtype=np.int64)[ids]
    padded = table[ids]
    vals = padded[padded >= 0]
    return lens, vals


def _flatten(model: TEGModel) -> _Flat:
    chain_names = list(model.chains)
    edge_names = list(model.edges)
    cid = {n: i for i, n in enumerate(chain_names)}
    eid = {n: i for i, n in enumerate(edge_names)}
    chains = [model.chains[n] for n in chain_names]
    nc = len(chains)
    n_ev = np.array([c.num_events for c in chains], dtype=np.int64)
    ev_off = np.concatenate([[0], np.cumsum(n_ev)]).astype(np.int64)
    nev = int(ev_off[-1])
    gap = np.fromiter(_chain.from_iterable(c.gaps for c in chains), dtype=np.int64, count=nev)
    rd_lens, rd_val = _csr_from_tuples(list(_chain.from_iterable(c.reads for c in chains)), eid)
    wr_lens, wr_val = _csr_from_tuples(list(_chain.from_iterable(c.writes for c in chains)), eid)
    rd_off = np.concatenate([[0], np.cumsum(rd_lens)]).astype(np.int64)
    wr_off = np.concatenate([[0], np.cumsum(wr_lens)]).astype(np.int64)
    # arcs: 4 values per arc in event order
    arc_lens = np.zeros(nev, dtype=np.int64)
    arc_vals: list[int] = []
    referenced: set[int] = set()
    for c_i, c in enumerate(chains):
        if not c.arcs:
            continue
        base = int(ev_off[c_i])
        for i in sorted(c.arcs):
            arcs = c.arcs[i]
            arc_lens[base + i] = 4 * len(arcs)
            for a in arcs:
                s = cid[a.src]
                referenced.add(s)
                arc_vals.extend((s, a.src_event, a.lag, a.delay))
    arc_off = np.concatenate([[0], np.cumsum(arc_lens)]).astype(np.int64)
    arc_val = np.array(arc_vals, dtype=np.int64)
    # patterns (chains and direct sink gates)
    patterns: list[StallPattern] = []
    pat_index: dict[str, int] = {}

    def pat_id(p: StallPattern | None) -> int:
        if p is None:
            return -1
        key = repr(p)
        if key not in pat_index:
            pat_index[key] = len(patterns)
            patterns.append(p)
        return pat_index[key]

    kind = np.array([{"op": 0, "source": 1, "sink": 2}[c.kind] for c in chains], dtype=np.int64)
    pat = np.array([pat_id(c.pattern) for c in chains], dtype=np.int64)
    period = np.array([c.period for c in chains], dtype=np.int64)
    keep_hist = np.array([i in referenced for i in range(nc)], dtype=np.int64)
    hist_window = np.array(
        [-1 if c.history_window is None else c.history_window for c in chains], dtype=np.int64
    )
    groups = sorted({c.freeze_group for c in chains if c.freeze_group})
    gid = {g: i for i, g in enumerate(groups)}
    group = np.array(
        [gid[c.freeze_group] if c.freeze_group else -1 for c in chains], dtype=np.int64
    )
    freezer = np.array([c.freezer for c in chains], dtype=np.int64)
    until_empty = np.array([c.freeze_until_empty for c in chains], dtype=np.int64)
    start = np.array([c.start for c in chains], dtype=np.int64)
    # edges
    edges = [model.edges[n] for n in edge_names]
    lf = np.array([e.lf for e in edges], dtype=np.int64)
    lb = np.array([e.lb for e in edges], dtype=np.int64)
    init_tokens = np.array([e.initial_tokens for e in edges], dtype=np.int64)
    sd_lens = np.zeros(nc, dtype=np.int64)
    sd_vals: list[int] = []
    for c_i, c in enumerate(chains):
        if c.kind != "source":
            continue
        direct = [eid[e] for e in sorted({e for w in c.writes for e in w}) if model.edges[e].direct]
        sd_lens[c_i] = len(direct)
        sd_vals.extend(direct)
    sd_off = np.concatenate([[0], np.cumsum(sd_lens)]).astype(np.int64)
    sd_val = np.array(sd_vals, dtype=np.int64)
    gate = np.array(
        [
            pat_id(model.chains[e.dst].pattern)
            if e.direct and model.chains[e.dst].kind == "sink"
            else -1
            for e in edges
        ],
        dtype=np.int64,
    )
    pat_kind = np.array([_PAT_KIND[p.kind] for p in patterns], dtype=np.int64)
    pat_p = np.array([p.p for p in patterns], dtype=np.float64)
    pat_i = np.array(
        [[p.seed & _MASK64, p.on, p.off, p.phase, p.at, p.length, p.every] for p in patterns],
        dtype=np.uint64,
    ).reshape(-1, 7)
    # seeds above 2^63 are passed as their two's complement bit pattern
    pat_i = pat_i.astype(np.int64, copy=False) if pat_i.size else np.zeros((0, 7), np.int64)
    arrays = {
        "L": n_ev,
        "ev_off": ev_off,
        "gap": gap,
        "rd_off": rd_off,
        "rd_val": rd_val,
        "wr_off": wr_off,
        "wr_val": wr_val,
        "arc_off": arc_off,
        "arc_val": arc_val,
        "kind": kind,
        "pat": pat,
        "period": period,
        "keep_hist": keep_hist,
        "hist_window": hist_window,
        "group": group,
        "freezer": freezer,
        "until_empty": until_empty,
        "start": start,
        "lf": lf,
        "lb": lb,
        "init_tokens": init_tokens,
        "sd_off": sd_off,
        "sd_val": sd_val,
        "gate": gate,
        "pat_kind": pat_kind,
        "pat_i": np.ascontiguousarray(pat_i.reshape(-1)),
    }
    for k, v in arrays.items():
        arrays[k] = np.ascontiguousarray(v, dtype=np.int64)
    return _Flat(chain_names, edge_names, arrays, pat_p, len(groups), len(patterns))


def _signature(model: TEGModel) -> tuple:
    """Fingerprint of everything the flat representation depends on (chain and edge objects
    and the per-chain environment parameters; depths are read per run)."""
    return (
        tuple(
            (id(c), c.num_events, repr(c.pattern), c.period, c.start) for c in model.chains.values()
        ),
        tuple((id(e), e.initial_tokens, e.lf, e.lb, e.direct) for e in model.edges.values()),
    )


def flatten(model: TEGModel) -> _Flat:
    """Return the cached flat representation of ``model`` (rebuilt when the model changed).

    The cache lives on the model instance; a copy that shares chain objects but replaces
    e.g. the paced sources (``search.measure_peak_occupancy``) gets its own entry.
    """
    sig = _signature(model)
    cached = getattr(model, _CACHE_ATTR, None)
    if cached is not None and cached[0] == sig:
        return cached[1]  # type: ignore[return-value]
    flat = _flatten(model)
    object.__setattr__(model, _CACHE_ATTR, (sig, flat))
    return flat


# ----------------------------------------------------------------------------- run
def _ptr(a: np.ndarray) -> ctypes.POINTER:  # type: ignore[type-arg]
    if a.size == 0:
        a = np.zeros(1, dtype=a.dtype)
    return a.ctypes.data_as(POINTER(c_double if a.dtype == np.float64 else c_int64))


def run(
    model: TEGModel,
    depths: dict[str, int | None] | None,
    max_frames: int,
    min_frames: int,
    max_cycles: int | None,
    stop_when_stable: bool,
    stable_occupancy: bool,
) -> SimResult:
    """Simulate with the native core; same contract as ``simulate.simulate``."""
    from finn.analysis.fpgadataflow.teg.simulate import SimResult

    lib = library()
    flat = flatten(model)
    a = flat.arrays
    ne = len(flat.edge_names)
    depth = np.empty(ne, dtype=np.int64)
    for j, n in enumerate(flat.edge_names):
        d = model.edges[n].depth
        if depths is not None and n in depths:
            d = depths[n]
        if d is not None and d < 1:
            raise FINNInternalError(f"Edge {n}: depth must be >= 1, got {d}")
        depth[j] = -1 if d is None else d
    if not any(k == 2 for k in a["kind"]):
        raise FINNInternalError("Model has no sink chain")
    maxocc = np.zeros(ne, dtype=np.int64)
    first_valid = np.zeros(ne, dtype=np.int64)
    scalars = np.zeros(16, dtype=np.int64)
    chain_keys = (
        "L",
        "ev_off",
        "gap",
        "rd_off",
        "rd_val",
        "wr_off",
        "wr_val",
        "arc_off",
        "arc_val",
        "kind",
        "pat",
        "period",
        "keep_hist",
        "hist_window",
        "group",
        "freezer",
        "until_empty",
        "start",
    )
    rc = lib.teg_sim_run(
        len(flat.chain_names),
        *(_ptr(a[k]) for k in chain_keys),
        flat.ngroups,
        ne,
        _ptr(a["lf"]),
        _ptr(a["lb"]),
        _ptr(depth),
        _ptr(a["init_tokens"]),
        _ptr(a["sd_off"]),
        _ptr(a["sd_val"]),
        _ptr(a["gate"]),
        flat.npat,
        _ptr(a["pat_kind"]),
        _ptr(flat.pat_p),
        _ptr(a["pat_i"]),
        max_frames,
        min_frames,
        -1 if max_cycles is None else max_cycles,
        int(stop_when_stable),
        int(stable_occupancy),
        _ptr(maxocc),
        _ptr(first_valid),
        _ptr(scalars),
    )
    if rc == -1:
        raise FINNInternalError(
            f"History of chain {flat.chain_names[scalars[8]]} was truncated below event "
            f"{scalars[9]} referenced by {flat.chain_names[scalars[7]]}; increase history_window"
        )
    if rc != 0:
        raise FINNInternalError(f"Native TEG simulator failed with code {rc}")
    frames_done, cycles, stable, deadlock, timeout, n_end, n_start = (int(x) for x in scalars[:7])
    frame_end = np.zeros(max(n_end, 1), dtype=np.int64)
    frame_start = np.zeros(max(n_start, 1), dtype=np.int64)
    lib.teg_sim_fetch(_ptr(frame_end), _ptr(frame_start))
    ends = [int(x) for x in frame_end[:n_end]]
    starts = [int(x) for x in frame_start[:n_start]]
    done = not deadlock and not timeout
    if done and stable:
        interval: float = float(ends[-1] - ends[-2])
    elif done and len(ends) >= 2:
        ivs = [b - a_ for a_, b in pairwise(ends)]
        half = ivs[len(ivs) // 2 :]
        interval = float(sum(half)) / len(half)
    else:
        interval = math.inf
    latency: int | None = None
    n = min(len(ends), len(starts))
    if n > 0:
        latency = ends[n - 1] - starts[n - 1]
    en = flat.edge_names
    return SimResult(
        interval=interval,
        latency=latency,
        frames=frames_done,
        cycles=cycles,
        stable=bool(stable),
        deadlock=bool(deadlock),
        timeout=bool(timeout),
        max_occupancy={en[e]: int(maxocc[e]) for e in range(ne)},
        first_valid={en[e]: int(first_valid[e]) for e in range(ne)},
        frame_end_times=ends,
        frame_start_times=starts,
        event_times=None,
        handshakes=None,
    )

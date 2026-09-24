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

"""Timed-event-graph model: chains, FIFO edges and index-mapped arcs.

Semantics (cycle level, see the formal model notes, Sections 1-4):

* A **chain** is a named sequence of ``L`` events per frame. Event ``i`` has a minimum gap
  ``g_i >= 1`` to event ``i-1`` (the gap of event 0 closes the frame, i.e. it is measured to
  event ``L-1`` of the previous frame), a set of FIFO edges it reads one token from and a set
  it writes one token to. A chain fires at most one event per cycle. Event ``i`` of frame ``n``
  is global event ``n*L + i``.
* A **FIFO edge** connects a writing chain to a reading chain. Token ``k`` (global count) is
  written by the ``k``-th writing event of the source and read by the ``k``-th reading event of
  the destination. Reading token ``k`` at cycle ``t`` requires ``w[k] + lf <= t``; writing token
  ``k`` requires ``r[k-d] + lb <= t`` (void for ``k < d``). ``initial_tokens`` pre-fills the
  edge with tokens readable from cycle 0 on (the marking of a marked graph), which expresses
  frame-lagged dependencies such as "the next frame starts after the last write".
* An **arc** ``(src_chain, src_event, lag, delay)`` attached to event ``i`` of a chain means
  that event ``i`` of frame ``n`` fires no earlier than ``time(src event src_event of frame
  n-lag) + delay``. Arcs express index-mapped dependencies that are not FIFOs (random-access
  buffer reuse, replay buffers).
* **Freeze groups** express the stall rule of Vitis HLS pipelines: all chains of a group
  (the pipeline's stages) stop as long as a *freezer* chain of the group holds a token it
  cannot pass on (the output register slice with an unaccepted token, ``apdone_blk``). This
  is the one construct that is not a fixed timed event graph; the MILP ignores it and its
  solutions are verified by the simulator.
* **Environment chains** (kind ``source`` / ``sink``) model the test bench: a source presents
  one token per event when its stall pattern allows it and holds the token until it is
  accepted (AXI-Stream compliant); a sink accepts a token at cycle ``t`` iff one is readable and
  its pattern is true at ``t``. A source may additionally be *paced*: event 0 of frame ``n`` is
  not presented before cycle ``n * period``.

External edges are the FIFOs that get sized; internal edges belong to operator templates and
have a fixed capacity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from finn.util.exception import FINNInternalError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from finn.analysis.fpgadataflow.teg.patterns import StallPattern

ChainKind = Literal["op", "source", "sink"]


@dataclass(frozen=True)
class Arc:
    """Index-mapped dependency on an event of another chain.

    The event this arc is attached to (event ``i`` of frame ``n`` of the owning chain) fires no
    earlier than ``time(src event ``src_event`` of frame ``n - lag``) + delay``. Arcs whose
    referenced frame would be negative are void (this makes frame-restart arcs trivially
    satisfied in the first frame).
    """

    src: str
    src_event: int
    lag: int = 0
    delay: int = 0


class Chain:
    """A named sequence of events per frame with gaps, reads, writes and arcs.

    Chains are built incrementally with :meth:`event`, which appends one event and returns
    its index, so that operator templates can attach arcs to events they just created.
    """

    __slots__ = (
        "arcs",
        "freeze_group",
        "freeze_until_empty",
        "freezer",
        "gaps",
        "history_window",
        "kind",
        "name",
        "pattern",
        "period",
        "reads",
        "start",
        "writes",
    )

    def __init__(
        self,
        name: str,
        kind: ChainKind = "op",
        pattern: StallPattern | None = None,
        period: int = 0,
        history_window: int | None = None,
        freeze_group: str | None = None,
        freezer: bool = False,
        start: int = 0,
        freeze_until_empty: bool = False,
    ) -> None:
        """Create an empty chain.

        Args:
            name: unique chain name.
            kind: ``op`` for operator chains, ``source``/``sink`` for environment chains.
            pattern: stall pattern of an environment chain (``None`` = never stalled).
            period: pacing of a source: event 0 of frame ``n`` is presented at ``>= n*period``.
            history_window: number of most recent event times the simulator has to keep for
                this chain because other chains reference them through arcs. ``None`` keeps
                the full history (safe default for small models), ``0`` keeps nothing.
            freeze_group: name of the freeze group the chain belongs to (None: none).
            freezer: True when this chain freezes its group while it holds a token that it
                cannot hand on (blocked on space or on a sink pattern with its data ready).
            start: earliest cycle of the chain's very first event (initial offset, e.g. a
                pipeline stage that is reached only some cycles after reset); may be negative
                for logic that already runs in the cycle in which the reset is released.
            freeze_until_empty: a freezer that keeps its group frozen until at most one token
                is left on its input edges (the RTL MVU output lock, ``mvu_vvu_axi.sv:380-389``).
        """
        self.name = name
        self.start = start
        self.freeze_until_empty = freeze_until_empty
        self.freeze_group = freeze_group
        self.freezer = freezer
        self.kind: ChainKind = kind
        self.pattern = pattern
        self.period = period
        self.history_window = history_window
        self.gaps: list[int] = []
        self.reads: list[tuple[str, ...]] = []
        self.writes: list[tuple[str, ...]] = []
        self.arcs: dict[int, list[Arc]] = {}

    @property
    def num_events(self) -> int:
        """Number of events per frame (``L``)."""
        return len(self.gaps)

    @property
    def nominal_interval(self) -> int:
        """Frame interval of the chain when it is never stalled (sum of gaps)."""
        return sum(self.gaps)

    def event(
        self,
        gap: int = 1,
        reads: Iterable[str] = (),
        writes: Iterable[str] = (),
        arcs: Iterable[Arc] = (),
    ) -> int:
        """Append one event and return its index within the frame."""
        if gap < 1:
            raise FINNInternalError(f"Chain {self.name}: event gap must be >= 1, got {gap}")
        idx = len(self.gaps)
        self.gaps.append(gap)
        self.reads.append(tuple(reads))
        self.writes.append(tuple(writes))
        arcs = list(arcs)
        if arcs:
            self.arcs[idx] = arcs
        return idx

    def events(
        self, n: int, gap: int = 1, reads: Iterable[str] = (), writes: Iterable[str] = ()
    ) -> int:
        """Append ``n`` identical events and return the index of the first one."""
        first = len(self.gaps)
        reads_t = tuple(reads)
        writes_t = tuple(writes)
        self.gaps.extend([gap] * n)
        self.reads.extend([reads_t] * n)
        self.writes.extend([writes_t] * n)
        return first

    def add_arc(self, event: int, arc: Arc) -> None:
        """Attach an arc to an existing event."""
        if not 0 <= event < len(self.gaps):
            raise FINNInternalError(f"Chain {self.name}: no event {event} for arc {arc}")
        self.arcs.setdefault(event, []).append(arc)

    def count_writes(self, edge: str) -> int:
        """Count the tokens written to ``edge`` per frame."""
        return sum(1 for w in self.writes if edge in w)

    def count_reads(self, edge: str) -> int:
        """Count the tokens read from ``edge`` per frame."""
        return sum(1 for r in self.reads if edge in r)

    def __repr__(self) -> str:
        """Short description."""
        return f"Chain({self.name!r}, kind={self.kind}, L={self.num_events})"


@dataclass
class FIFOEdge:
    """A FIFO between two chains.

    ``depth`` is the number of tokens the FIFO holds; ``None`` means unbounded. ``lf`` is the
    forward latency (write -> readable), ``lb`` the backward latency (read -> slot writable).
    FINN's ``Q_srl`` has ``lf = lb = 1`` (``Q_srl.v``: ``o_v_reg`` and ``i_b_reg`` are
    registered, no push into a full queue in the cycle of a pop). ``width`` is the token width
    in bits (cost weight). ``external`` marks the FIFOs that get sized; ``initial_tokens``
    is the initial marking.
    """

    name: str
    src: str
    dst: str
    depth: int | None = None
    lf: int = 1
    lb: int = 1
    width: int = 1
    external: bool = False
    initial_tokens: int = 0
    #: a direct AXI-Stream connection without a FIFO (test benches only): the writer presents
    #: the token and the reader accepts it in the same cycle (``lf = 0``), the next token can
    #: be presented in the cycle after the acceptance (``depth = 1``, ``lb = 1``). When the
    #: reader is a sink, its stall pattern gates the writer (the handshake is one event). When
    #: the writer is a source, it presents the next token at the first pattern-true cycle after
    #: the previous acceptance (the test bench has a single output register).
    direct: bool = False

    def __post_init__(self) -> None:
        """Validate latencies."""
        if self.direct:
            self.depth, self.lf, self.lb = 1, 0, 1
        if self.lf < 0 or self.lb < 0:
            raise FINNInternalError(f"Edge {self.name}: latencies must be >= 0")
        if self.lf + self.lb < 1:
            raise FINNInternalError(f"Edge {self.name}: lf + lb must be >= 1 (zero-delay cycle)")
        if self.depth is not None and self.depth < 1:
            raise FINNInternalError(f"Edge {self.name}: depth must be >= 1 or None")


@dataclass
class TEGModel:
    """A timed event graph: chains plus FIFO edges.

    Chains and edges are kept in insertion order; the order of the external edges defines the
    order of the depth vector used by the search and the MILP.
    """

    chains: dict[str, Chain] = field(default_factory=dict)
    edges: dict[str, FIFOEdge] = field(default_factory=dict)
    #: free-form metadata (e.g. node names of the ONNX graph that produced the model)
    meta: dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------ construction
    def add_chain(self, chain: Chain) -> Chain:
        """Register a chain (names must be unique)."""
        if chain.name in self.chains:
            raise FINNInternalError(f"Duplicate chain name {chain.name}")
        self.chains[chain.name] = chain
        return chain

    def new_chain(self, name: str, **kwargs: object) -> Chain:
        """Create and register a chain."""
        return self.add_chain(Chain(name, **kwargs))  # type: ignore[arg-type]

    def add_edge(self, edge: FIFOEdge) -> FIFOEdge:
        """Register an edge (names must be unique)."""
        if edge.name in self.edges:
            raise FINNInternalError(f"Duplicate edge name {edge.name}")
        self.edges[edge.name] = edge
        return edge

    def new_edge(self, name: str, src: str, dst: str, **kwargs: object) -> FIFOEdge:
        """Create and register an edge."""
        return self.add_edge(FIFOEdge(name, src, dst, **kwargs))  # type: ignore[arg-type]

    def merge(self, other: TEGModel) -> None:
        """Add all chains and edges of ``other`` to this model."""
        for c in other.chains.values():
            self.add_chain(c)
        for e in other.edges.values():
            self.add_edge(e)

    # ------------------------------------------------------------------ queries
    @property
    def external_edges(self) -> list[str]:
        """Names of the external (sizeable) edges in insertion order."""
        return [n for n, e in self.edges.items() if e.external]

    @property
    def internal_edges(self) -> list[str]:
        """Names of the internal (fixed-capacity) edges in insertion order."""
        return [n for n, e in self.edges.items() if not e.external]

    @property
    def sources(self) -> list[str]:
        """Names of the source chains."""
        return [n for n, c in self.chains.items() if c.kind == "source"]

    @property
    def sinks(self) -> list[str]:
        """Names of the sink chains."""
        return [n for n, c in self.chains.items() if c.kind == "sink"]

    def tokens_per_frame(self, edge: str) -> int:
        """Count the tokens transported over ``edge`` per frame."""
        e = self.edges[edge]
        return self.chains[e.src].count_writes(edge)

    def nominal_intervals(self) -> dict[str, int]:
        """Unstalled frame interval of every operator chain."""
        return {n: c.nominal_interval for n, c in self.chains.items() if c.kind == "op"}

    def bottleneck_interval(self) -> int:
        """Largest nominal interval over all operator chains (a lower bound on ``I*``)."""
        ivs = self.nominal_intervals()
        return max(ivs.values()) if ivs else 0

    def num_events(self) -> int:
        """Total number of events per frame over all chains."""
        return sum(c.num_events for c in self.chains.values())

    def num_arcs(self) -> int:
        """Total number of index-mapped arcs."""
        return sum(len(a) for c in self.chains.values() for a in c.arcs.values())

    def depths(self) -> dict[str, int | None]:
        """Return the current depth of every external edge."""
        return {n: self.edges[n].depth for n in self.external_edges}

    def with_depths(self, depths: dict[str, int | None]) -> TEGModel:
        """Return a shallow copy of the model with the given external depths applied.

        Chains are shared (they are not mutated by any analysis), edges are copied.
        """
        new = TEGModel(chains=self.chains, meta=self.meta)
        for n, e in self.edges.items():
            e2 = FIFOEdge(**e.__dict__)
            if n in depths:
                e2.depth = depths[n]
            new.edges[n] = e2
        return new

    # ------------------------------------------------------------------ validation
    def validate(self) -> None:
        """Check structural consistency; raise FINNInternalError on the first problem."""
        for name, e in self.edges.items():
            if e.src not in self.chains:
                raise FINNInternalError(f"Edge {name}: unknown source chain {e.src}")
            if e.dst not in self.chains:
                raise FINNInternalError(f"Edge {name}: unknown destination chain {e.dst}")
            nw = self.chains[e.src].count_writes(name)
            nr = self.chains[e.dst].count_reads(name)
            if nw != nr:
                raise FINNInternalError(
                    f"Edge {name}: {e.src} writes {nw} tokens per frame but {e.dst} reads {nr}"
                )
            if nw == 0:
                raise FINNInternalError(f"Edge {name}: no tokens per frame")
        written: dict[str, str] = {}
        read: dict[str, str] = {}
        for cname, c in self.chains.items():
            if c.num_events == 0:
                raise FINNInternalError(f"Chain {cname} has no events")
            for i in range(c.num_events):
                for e in c.writes[i]:
                    if e not in self.edges:
                        raise FINNInternalError(f"Chain {cname} writes unknown edge {e}")
                    if self.edges[e].src != cname:
                        raise FINNInternalError(
                            f"Chain {cname} writes edge {e} of {self.edges[e].src}"
                        )
                    written[e] = cname
                for e in c.reads[i]:
                    if e not in self.edges:
                        raise FINNInternalError(f"Chain {cname} reads unknown edge {e}")
                    if self.edges[e].dst != cname:
                        raise FINNInternalError(
                            f"Chain {cname} reads edge {e} of {self.edges[e].dst}"
                        )
                    read[e] = cname
            for i, arcs in c.arcs.items():
                if not 0 <= i < c.num_events:
                    raise FINNInternalError(f"Chain {cname}: arc attached to missing event {i}")
                for a in arcs:
                    if a.src not in self.chains:
                        raise FINNInternalError(
                            f"Chain {cname}: arc references unknown chain {a.src}"
                        )
                    if not 0 <= a.src_event < self.chains[a.src].num_events:
                        raise FINNInternalError(
                            f"Chain {cname}: arc references missing event {a.src_event} of {a.src}"
                        )
                    if a.lag < 0 or a.delay < 0:
                        raise FINNInternalError(f"Chain {cname}: negative lag/delay in arc {a}")
            if c.kind == "source" and any(c.reads):
                raise FINNInternalError(f"Source chain {cname} must not read")
            if c.kind == "sink" and any(c.writes):
                raise FINNInternalError(f"Sink chain {cname} must not write")
        if not self.sinks:
            raise FINNInternalError("Model has no sink chain")

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable description (patterns are described by their repr only)."""
        chains = {}
        for name, c in self.chains.items():
            chains[name] = {
                "kind": c.kind,
                "period": c.period,
                "pattern": None if c.pattern is None else repr(c.pattern),
                "history_window": c.history_window,
                "freeze_group": c.freeze_group,
                "freezer": c.freezer,
                "freeze_until_empty": c.freeze_until_empty,
                "start": c.start,
                "gaps": c.gaps,
                "reads": [list(r) for r in c.reads],
                "writes": [list(w) for w in c.writes],
                "arcs": {
                    str(i): [[a.src, a.src_event, a.lag, a.delay] for a in arcs]
                    for i, arcs in c.arcs.items()
                },
            }
        edges = {n: dict(e.__dict__) for n, e in self.edges.items()}
        return {"chains": chains, "edges": edges, "meta": self.meta}

    def dump_json(self, path: Path) -> None:
        """Write the model to a JSON file (for debugging and paper figures)."""
        path.write_text(json.dumps(self.to_dict()))

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> TEGModel:
        """Rebuild a model from :meth:`to_dict` output (patterns are dropped)."""
        m = cls(meta=dict(d.get("meta", {})))  # type: ignore[arg-type]
        for name, cd in d["chains"].items():  # type: ignore[attr-defined]
            c = Chain(
                name,
                kind=cd["kind"],
                period=cd["period"],
                history_window=cd["history_window"],
                freeze_group=cd.get("freeze_group"),
                freezer=bool(cd.get("freezer", False)),
                start=int(cd.get("start", 0)),
                freeze_until_empty=bool(cd.get("freeze_until_empty", False)),
            )
            c.gaps = list(cd["gaps"])
            c.reads = [tuple(r) for r in cd["reads"]]
            c.writes = [tuple(w) for w in cd["writes"]]
            c.arcs = {int(i): [Arc(*a) for a in arcs] for i, arcs in cd["arcs"].items()}
            m.add_chain(c)
        for ed in d["edges"].values():  # type: ignore[attr-defined]
            m.add_edge(FIFOEdge(**ed))
        return m

    @classmethod
    def load_json(cls, path: Path) -> TEGModel:
        """Read a model written by :meth:`dump_json`."""
        return cls.from_dict(json.loads(path.read_text()))


def rigid_chain(
    name: str, trace: Sequence[tuple[int, Iterable[str], Iterable[str]]], kind: ChainKind = "op"
) -> Chain:
    """Build a chain from an explicit ``(gap, reads, writes)`` trace (rigid-operator semantics)."""
    c = Chain(name, kind=kind)
    for gap, reads, writes in trace:
        c.event(gap, reads, writes)
    return c

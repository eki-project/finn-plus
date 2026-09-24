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

"""Shared harness of the per-operator differential tests.

One FINN hardware node is driven twice with identical, deterministic stall patterns: in XSI
through :class:`StallingInputStreamer` / :class:`StallingOutputCollector` (SimEngine tasks that
gate TVALID / TREADY by a :class:`StallPattern`), and in the abstract simulator through the
node's operator template with direct (FIFO-less) environment edges. Both produce the cycle of
every handshake on every stream; ``trace_compare`` diffs them.

Cycle alignment: model cycle 0 is the first XSI cycle in which the test bench can present a
token, i.e. ``t0 = first task tick after reset + 1`` (a task at tick ``T`` reads the ports
before the clock edge and sets the values that are visible in cycle ``T + 1``).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from typing import TYPE_CHECKING

from finn.analysis.fpgadataflow.teg.from_onnx import build_model, edge_name
from finn.analysis.fpgadataflow.teg.patterns import StallPattern
from finn.analysis.fpgadataflow.teg.simulate import simulate
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import getHWCustomOp

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
    from finn.xsi import SimEngine


def stall_kinds(seed: int) -> dict[str, tuple[StallPattern, StallPattern]]:
    """Return the standard (input pattern, output pattern) pairs, seeded from the test seed."""
    return {
        "free": (StallPattern.none(), StallPattern.none()),
        "in_bernoulli": (StallPattern.bernoulli(0.3, seed), StallPattern.none()),
        "out_bernoulli": (StallPattern.none(), StallPattern.bernoulli(0.3, seed + 1)),
        "both_bernoulli": (
            StallPattern.bernoulli(0.3, seed + 2),
            StallPattern.bernoulli(0.4, seed + 3),
        ),
        "both_bursty": (StallPattern.bursty(5, 3, phase=2), StallPattern.bursty(3, 6)),
        "out_single_stall": (StallPattern.none(), StallPattern.single_stall(12, 25)),
    }


# --------------------------------------------------------------------------- XSI tasks
class StallingInputStreamer:
    """Drive an AXI-Stream input: present a token whenever ``pattern`` allows, hold until
    accepted, record the handshake cycles."""

    def __init__(
        self, sim: SimEngine, stream: str, n_tokens: int, pattern: StallPattern, value: str = "0"
    ) -> None:
        """Bind to the stream ports."""
        self.vld = sim.get_bus_port(stream, "TVALID")
        self.rdy = sim.get_bus_port(stream, "TREADY")
        self.dat = sim.get_bus_port(stream, "TDATA")
        self.n = n_tokens
        self.pattern = pattern
        self.value = value
        self.sent = 0
        self.handshakes: list[int] = []
        self.t0: int | None = None

    def __call__(self, sim: SimEngine) -> dict | None:
        """Advance one cycle."""
        if self.t0 is None:
            self.t0 = sim.ticks + 1
        vld = self.vld.as_bool()
        if vld and not self.rdy.read().as_bool():
            return {}  # hold
        if vld:
            self.handshakes.append(sim.ticks)
            self.sent += 1
        if self.sent >= self.n:
            return {self.vld: "0", self.dat: "0"} if vld else None
        if self.pattern(sim.ticks + 1 - self.t0):
            return {self.dat: self.value, self.vld: "1"}
        return {self.vld: "0", self.dat: "0"} if vld else {}


class StallingOutputCollector:
    """Accept an AXI-Stream output with ``TREADY = pattern(cycle)``, record handshake cycles."""

    def __init__(self, sim: SimEngine, stream: str, n_tokens: int, pattern: StallPattern) -> None:
        """Bind to the stream ports."""
        self.vld = sim.get_bus_port(stream, "TVALID")
        self.rdy = sim.get_bus_port(stream, "TREADY")
        self.n = n_tokens
        self.pattern = pattern
        self.got = 0
        self.handshakes: list[int] = []
        self.t0: int | None = None

    def __call__(self, sim: SimEngine) -> dict | None:
        """Advance one cycle."""
        if self.t0 is None:
            self.t0 = sim.ticks + 1
        rdy = self.rdy.as_bool()
        if rdy and self.vld.read().as_bool():
            self.handshakes.append(sim.ticks)
            self.got += 1
        if self.got >= self.n:
            return {self.rdy: "0"} if rdy else None
        return {self.rdy: "1" if self.pattern(sim.ticks + 1 - self.t0) else "0"}


@dataclass
class XsiTrace:
    """Handshake cycles of one XSI run (absolute ticks) and the model-zero tick."""

    t0: int
    inputs: dict[str, list[int]] = field(default_factory=dict)
    outputs: dict[str, list[int]] = field(default_factory=dict)
    cycles: int = 0

    def relative(self) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        """Handshake cycles relative to model cycle 0."""
        return (
            {k: [t - self.t0 for t in v] for k, v in self.inputs.items()},
            {k: [t - self.t0 for t in v] for k, v in self.outputs.items()},
        )


# --------------------------------------------------------------------------- node preparation
def prepare_node_rtlsim(model: ModelWrapper, fpga_part: str, clk_ns: float) -> ModelWrapper:
    """Specialise, generate IP and compile the XSI library of a single-node model."""
    model = model.transform(SpecializeLayers(fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(PrepareIP(fpga_part, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(PrepareRTLSim())
    return model


def tokens_in(node: HWCustomOp, ind: int = 0) -> int:
    """Tokens per frame on input ``ind``."""
    return int(np.prod(node.get_folded_input_shape(ind)[:-1]))


def tokens_out(node: HWCustomOp, ind: int = 0) -> int:
    """Tokens per frame on output ``ind``."""
    return int(np.prod(node.get_folded_output_shape(ind)[:-1]))


# --------------------------------------------------------------------------- runs
def run_xsi(
    model: ModelWrapper,
    frames: int,
    in_pattern: StallPattern,
    out_pattern: StallPattern,
    cycle_limit: int | None = None,
    extra_inputs: dict[str, int] | None = None,
) -> XsiTrace:
    """Simulate the single node of ``model`` in XSI with the given stall patterns.

    ``extra_inputs`` maps further input streams (e.g. ``in1_V``, the weight stream of a
    decoupled MVAU) to the number of tokens per frame they carry; they are driven without
    stalls, like the memstream that feeds them in the stitched design.
    """
    node = model.graph.node[0]
    inst = getHWCustomOp(node)
    n_in = tokens_in(inst) * frames
    n_out = tokens_out(inst) * frames
    sim = inst.get_rtlsim()
    try:
        inst.reset_rtlsim(sim)
        streamer = StallingInputStreamer(sim, "in0_V", n_in, in_pattern)
        collector = StallingOutputCollector(sim, "out0_V", n_out, out_pattern)
        sim.enlist(streamer)
        sim.enlist(collector)
        extra = []
        for name, n_per_frame in (extra_inputs or {}).items():
            extra.append(
                StallingInputStreamer(sim, name, n_per_frame * frames, StallPattern.none())
            )
            sim.enlist(extra[-1])
        limit = cycle_limit if cycle_limit is not None else 20 * (n_in + n_out) + 1000
        start = sim.ticks
        woken = sim.run(cycles=limit)
        if woken:
            extra_state = ", ".join(f"{e.vld.name()}: {e.sent}/{e.n}" for e in extra)
            raise AssertionError(
                f"XSI run of {node.name} timed out after {limit} cycles "
                f"({streamer.sent}/{n_in} inputs, {collector.got}/{n_out} outputs; "
                f"extra streams {extra_state})"
            )
        assert streamer.t0 is not None and streamer.t0 == collector.t0
        return XsiTrace(
            t0=streamer.t0,
            inputs={"in0": streamer.handshakes},
            outputs={"out0": collector.handshakes},
            cycles=sim.ticks - start,
        )
    finally:
        inst.close_rtlsim(sim)


def run_abstract(
    model: ModelWrapper, frames: int, in_pattern: StallPattern, out_pattern: StallPattern
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Simulate the single node's TEG model with direct environment edges.

    Returns the acceptance cycles of the input tokens and the handshake cycles of the output
    tokens (model cycle 0 = first cycle in which the source may present).
    """
    node = model.graph.node[0]
    inst = getHWCustomOp(node)
    inp, out = node.input[0], node.output[0]
    teg = build_model(
        model,
        in_patterns={inp: in_pattern},
        out_patterns={out: out_pattern},
        direct_environment=True,
    )
    in_edge = f"in.{inp}"
    out_edge = edge_name(node, 0)
    res = simulate(
        teg,
        None,
        max_frames=frames,
        min_frames=frames,
        stop_when_stable=False,
        record_edges={in_edge, out_edge},
    )
    assert res.handshakes is not None and not res.deadlock
    n_in = tokens_in(inst) * frames
    n_out = tokens_out(inst) * frames
    return (
        {"in0": res.handshakes[in_edge][1][:n_in]},
        {"out0": res.handshakes[out_edge][0][:n_out]},
    )

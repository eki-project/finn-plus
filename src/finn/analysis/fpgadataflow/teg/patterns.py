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

"""Deterministic stall patterns for environment chains and RTL test benches.

A pattern is a pure function ``cycle -> bool`` ("the environment is willing to transfer in
this cycle"). The same object drives the abstract simulator (through :class:`Chain.pattern`)
and the XSI test bench tasks of the differential operator tests, so both see identical
stall sequences. Patterns are random-access (no state), which lets the event-driven simulator
skip idle cycles with :meth:`StallPattern.next_true`.
"""

from __future__ import annotations

from typing import Literal

from finn.util.exception import FINNInternalError

PatternKind = Literal["none", "bernoulli", "bursty", "single_stall", "periodic"]

_MASK64 = (1 << 64) - 1


def _mix64(seed: int, t: int) -> int:
    """SplitMix64 finaliser over ``(seed, t)``: a stateless, seedable hash to 64 bits."""
    z = (seed * 0x9E3779B97F4A7C15 + (t + 1) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


class StallPattern:
    """Pure function of the cycle index: True when the environment does *not* stall.

    Kinds:

    * ``none``: never stalls.
    * ``bernoulli``: stalls in each cycle independently with probability ``p`` (seeded).
    * ``bursty``: alternates ``on`` active cycles and ``off`` stalled cycles, starting at
      ``phase``.
    * ``single_stall``: active except for ``length`` cycles starting at cycle ``at``.
    * ``periodic``: active in exactly one out of every ``every`` cycles (throttling).
    """

    __slots__ = ("at", "every", "kind", "length", "off", "on", "p", "phase", "seed")

    def __init__(
        self,
        kind: PatternKind = "none",
        p: float = 0.0,
        seed: int = 0,
        on: int = 1,
        off: int = 0,
        phase: int = 0,
        at: int = 0,
        length: int = 0,
        every: int = 1,
    ) -> None:
        """Create a pattern; see the class docstring for the meaning of the parameters."""
        if kind not in ("none", "bernoulli", "bursty", "single_stall", "periodic"):
            raise FINNInternalError(f"Unknown stall pattern kind {kind}")
        if kind == "bernoulli" and not 0.0 <= p < 1.0:
            raise FINNInternalError("bernoulli stall probability must be in [0, 1)")
        if kind == "bursty" and (on < 1 or off < 0):
            raise FINNInternalError("bursty pattern needs on >= 1 and off >= 0")
        if kind == "periodic" and every < 1:
            raise FINNInternalError("periodic pattern needs every >= 1")
        self.kind = kind
        self.p = p
        self.seed = seed
        self.on = on
        self.off = off
        self.phase = phase
        self.at = at
        self.length = length
        self.every = every

    # ---------------------------------------------------------------- constructors
    @classmethod
    def none(cls) -> StallPattern:
        """Never stall."""
        return cls("none")

    @classmethod
    def bernoulli(cls, p: float, seed: int) -> StallPattern:
        """Stall each cycle with probability ``p``."""
        return cls("bernoulli", p=p, seed=seed)

    @classmethod
    def bursty(cls, on: int, off: int, phase: int = 0) -> StallPattern:
        """``on`` active cycles followed by ``off`` stalled cycles, repeated."""
        return cls("bursty", on=on, off=off, phase=phase)

    @classmethod
    def single_stall(cls, at: int, length: int) -> StallPattern:
        """One stall of ``length`` cycles starting at cycle ``at``."""
        return cls("single_stall", at=at, length=length)

    @classmethod
    def periodic(cls, every: int, phase: int = 0) -> StallPattern:
        """Active once every ``every`` cycles (cycle ``phase`` modulo ``every``)."""
        return cls("periodic", every=every, phase=phase)

    # ---------------------------------------------------------------- evaluation
    def __call__(self, t: int) -> bool:
        """Return True iff the environment is active (not stalling) in cycle ``t``."""
        k = self.kind
        if k == "none":
            return True
        if k == "bernoulli":
            # uniform in [0, 1) from the top 53 bits
            u = (_mix64(self.seed, t) >> 11) * (1.0 / (1 << 53))
            return u >= self.p
        if k == "bursty":
            if self.off == 0:
                return True
            return (t - self.phase) % (self.on + self.off) < self.on
        if k == "single_stall":
            return not (self.at <= t < self.at + self.length)
        # periodic
        return (t - self.phase) % self.every == 0

    def next_true(self, t: int) -> int:
        """Smallest cycle ``t' >= t`` in which the pattern is active."""
        k = self.kind
        if k == "none":
            return t
        if k == "bursty":
            if self.off == 0:
                return t
            period = self.on + self.off
            r = (t - self.phase) % period
            return t if r < self.on else t + (period - r)
        if k == "single_stall":
            return self.at + self.length if self.at <= t < self.at + self.length else t
        if k == "periodic":
            r = (t - self.phase) % self.every
            return t if r == 0 else t + (self.every - r)
        # bernoulli: expected 1/(1-p) iterations
        while not self(t):
            t += 1
        return t

    def __repr__(self) -> str:
        """Compact description used in test ids and error messages."""
        k = self.kind
        if k == "none":
            return "none"
        if k == "bernoulli":
            return f"bernoulli(p={self.p},seed={self.seed})"
        if k == "bursty":
            return f"bursty(on={self.on},off={self.off},phase={self.phase})"
        if k == "single_stall":
            return f"single_stall(at={self.at},len={self.length})"
        return f"periodic(every={self.every},phase={self.phase})"

    def __eq__(self, other: object) -> bool:
        """Patterns are equal iff all parameters are equal."""
        return isinstance(other, StallPattern) and repr(self) == repr(other)

    def __hash__(self) -> int:
        """Hash consistent with :meth:`__eq__`."""
        return hash(repr(self))

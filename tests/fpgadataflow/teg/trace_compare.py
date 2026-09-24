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

"""Diff the handshake traces of an XSI run and an abstract-model run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StreamDiff:
    """Comparison result of one stream."""

    stream: str
    ok: bool
    message: str
    xsi: list[int]
    model: list[int]
    first_divergence: int | None = None


def compare_stream(
    stream: str, xsi: list[int], model: list[int], max_cycle_error: int = 0, window: int = 6
) -> StreamDiff:
    """Compare two lists of handshake cycles (both relative to model cycle 0)."""
    n = min(len(xsi), len(model))
    for i in range(n):
        if abs(xsi[i] - model[i]) > max_cycle_error:
            lo, hi = max(0, i - window), i + window
            msg = (
                f"{stream}: token {i} handshakes at cycle {xsi[i]} in XSI but {model[i]} in the "
                f"model (diff {xsi[i] - model[i]:+d}); XSI[{lo}:{hi}]={xsi[lo:hi]} "
                f"model[{lo}:{hi}]={model[lo:hi]}"
            )
            return StreamDiff(stream, False, msg, xsi, model, i)
    if len(xsi) != len(model):
        msg = f"{stream}: {len(xsi)} handshakes in XSI but {len(model)} in the model"
        return StreamDiff(stream, False, msg, xsi, model, n)
    return StreamDiff(stream, True, f"{stream}: {n} handshakes identical", xsi, model)


def compare_traces(
    xsi_in: dict[str, list[int]],
    xsi_out: dict[str, list[int]],
    model_in: dict[str, list[int]],
    model_out: dict[str, list[int]],
    max_cycle_error: int = 0,
) -> list[StreamDiff]:
    """Compare all streams; input streams compare acceptance cycles, outputs handshake cycles."""
    diffs = []
    for name in sorted(set(xsi_in) | set(model_in)):
        diffs.append(
            compare_stream(name, xsi_in.get(name, []), model_in.get(name, []), max_cycle_error)
        )
    for name in sorted(set(xsi_out) | set(model_out)):
        diffs.append(
            compare_stream(name, xsi_out.get(name, []), model_out.get(name, []), max_cycle_error)
        )
    return diffs


def assert_traces_equal(diffs: list[StreamDiff], context: str = "") -> None:
    """Raise AssertionError listing every diverging stream."""
    bad = [d for d in diffs if not d.ok]
    if bad:
        raise AssertionError(context + "\n" + "\n".join(d.message for d in bad))

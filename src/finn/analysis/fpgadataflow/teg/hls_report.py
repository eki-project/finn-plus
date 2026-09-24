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

"""Timing parameters of HLS nodes from the Vitis HLS synthesis outputs.

Two sources are read from ``{code_gen_dir_ipgen}/project_{node}/sol1``:

* ``syn/report/{node}_csynth.xml``: the pipeline type of the top function (``loop
  auto-rewind flp (delay=N cycles)`` when the loop runs across invocations with a restart
  delay of ``N`` cycles, ``no`` when every invocation drains the pipeline), its initiation
  interval, and the loop's ``PipelineDepth``/``TripCount``. When the hlslib function is not
  inlined into the top (e.g. ``Matrix_Vector_Activate_Batch`` with embedded weights), the loop
  lives in the sub-function's report ``{function}_csynth.xml`` in the same directory and the
  top-level interval minus the trip count is the idle gap between frames.
* ``.autopilot/db/*.sched.adb`` (boost-serialised XML of the schedule): the pipeline stage
  of every stream read/write operation (``node_label_latency`` maps operation ids to
  ``(stage, latency)``). Read-to-write latencies of the two-chain loop model are the
  differences of these stages, so they are transcribed from the schedule rather than measured.

``hls_synth_res_estimation`` parses only ``AreaEstimates``; this module complements it.
"""

from __future__ import annotations

import contextlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from finn.util.logging import log

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


@dataclass
class HLSLoopParams:
    """Timing parameters of one HLS node's pipelined loop."""

    #: pipeline depth of the loop (``PipelineDepth``), None when unknown
    depth: int | None = None
    #: True when the top function reports ``loop auto-rewind``: the loop continues across
    #: invocations (frames) with ``rewind_delay`` idle cycles between them
    rewind: bool = False
    rewind_delay: int = 0
    #: pipeline stage of every stream read/write, keyed by port name (``in0_V``, ``out0_V``)
    read_stage: dict[str, int] = field(default_factory=dict)
    write_stage: dict[str, int] = field(default_factory=dict)
    #: top-level initiation interval and loop trip count (per frame), when reported
    interval: int | None = None
    trip_count: int | None = None
    #: True when the loop is in the top function (inlined); a loop in a non-inlined
    #: sub-function enters one cycle later after reset (the call from the top FSM)
    top_level: bool = True
    #: True when the top function is a ``#pragma HLS dataflow`` region: its processes run
    #: continuously (no per-frame invocation, no output drain between frames)
    dataflow: bool = False
    #: where the values came from ("report", "default")
    source: str = "default"


def solution_dir(node: HWCustomOp) -> Path | None:
    """``project_{node}/sol1`` of the node, or None when IP generation has not run."""
    code_gen_dir = cast("str", node.get_nodeattr("code_gen_dir_ipgen"))
    if not code_gen_dir:
        return None
    p = Path(code_gen_dir) / f"project_{node.onnx_node.name}" / "sol1"
    return p if p.is_dir() else None


def csynth_report_path(node: HWCustomOp) -> Path | None:
    """Path of the node's csynth XML report, or None when IP generation has not run."""
    sol = solution_dir(node)
    if sol is None:
        return None
    p = sol / "syn" / "report" / f"{node.onnx_node.name}_csynth.xml"
    return p if p.is_file() else None


def parse_loop_latencies(xml_path: Path) -> dict[str, dict[str, int]]:
    """Return ``{loop name: {TripCount, Latency, PipelineII, PipelineDepth}}`` of the report."""
    root = ET.parse(xml_path).getroot()
    loops: dict[str, dict[str, int]] = {}
    summary = root.find("PerformanceEstimates/SummaryOfLoopLatency")
    if summary is None:
        return loops
    for loop in summary:
        entry: dict[str, int] = {}
        for tag in ("TripCount", "Latency", "PipelineII", "PipelineDepth"):
            el = loop.find(tag)
            if el is not None and el.text is not None:
                with contextlib.suppress(ValueError):
                    entry[tag] = int(el.text)
        loops[loop.tag] = entry
    return loops


def parse_pipeline_type(xml_path: Path) -> tuple[str, int]:
    """``(PipelineType text, rewind delay)``; the delay is 0 unless ``auto-rewind`` reports one."""
    root = ET.parse(xml_path).getroot()
    el = root.find("PerformanceEstimates/PipelineType")
    text = (el.text or "").strip() if el is not None else ""
    m = re.search(r"delay=(\d+)", text)
    return text, int(m.group(1)) if m else 0


def parse_interval(xml_path: Path) -> int | None:
    """Top-level ``Interval-min`` of the report."""
    root = ET.parse(xml_path).getroot()
    el = root.find("PerformanceEstimates/SummaryOfOverallLatency/Interval-min")
    if el is None or el.text is None:
        return None
    with contextlib.suppress(ValueError):
        return int(el.text)
    return None


_LABEL = re.compile(
    r"<item[^>]*>\s*<first>(\d+)</first>\s*<second[^>]*>\s*<first>(-?\d+)</first>"
    r"\s*<second>(-?\d+)</second>\s*</second>\s*</item>"
)
_PORT = re.compile(r"<id>(\d+)</id>\s*<name>([^<]*)</name>")
_NODE = re.compile(
    r"<id>(\d+)</id>\s*<name>([^<]*)</name>.*?<oprand_edges>\s*<count>\d+</count>"
    r"\s*<item_version>0</item_version>((?:\s*<item>\d+</item>)*)\s*</oprand_edges>"
    r"\s*<opcode>([^<]*)</opcode>",
    re.S,
)
_EDGE = re.compile(
    r"<id>(\d+)</id>\s*<edge_type>(-?\d+)</edge_type>\s*<source_obj>(\d+)</source_obj>"
    r"\s*<sink_obj>(\d+)</sink_obj>"
)
_STREAM_OPCODES = {"read": "read", "nbread": "read", "write": "write", "nbwrite": "write"}


def _section(text: str, tag: str) -> str:
    """Text of the first top-level ``<tag ...>...</tag>`` block (empty when absent)."""
    m = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", text, re.S)
    return m.group(1) if m else ""


def parse_stream_op_stages(adb_path: Path) -> dict[str, tuple[str, int, int]]:
    """Return ``{port: (op, stage, latency)}`` of the stream reads/writes of one ``.sched.adb``.

    Stream operations are the nodes with opcode ``read``/``write`` (``nbread``/``nbwrite``);
    the port they access is found through their operand edges, whose source object is the
    port entry of the ``<ports>`` section. The stage is the first component of the node's
    ``node_label_latency`` entry.
    """
    text = adb_path.read_text(errors="replace")
    ports = {int(m.group(1)): m.group(2) for m in _PORT.finditer(_section(text, "ports"))}
    edges = {
        int(m.group(1)): (int(m.group(3)), int(m.group(4)))
        for m in _EDGE.finditer(_section(text, "edges"))
    }
    labels = {
        int(it.group(1)): (int(it.group(2)), int(it.group(3)))
        for it in _LABEL.finditer(_section(text, "node_label_latency"))
    }
    result: dict[str, tuple[str, int, int]] = {}
    for m in _NODE.finditer(_section(text, "nodes")):
        opcode = m.group(4)
        if opcode not in _STREAM_OPCODES:
            continue
        node_id = int(m.group(1))
        for eid in re.findall(r"<item>(\d+)</item>", m.group(3)):
            src, _sink = edges.get(int(eid), (-1, -1))
            if src in ports and node_id in labels:
                stage, lat = labels[node_id]
                result.setdefault(ports[src], (_STREAM_OPCODES[opcode], stage, lat))
                break
    return result


def hls_function_params(node: HWCustomOp) -> dict[str, HLSLoopParams]:
    """Loop parameters of every synthesised function of the node, keyed by function name.

    Sub-functions (e.g. the two converters of an LCM ``StreamingDataWidthConverter_hls``)
    have their own report and schedule; their ``read_stage``/``write_stage`` are keyed by
    the stream names they use (``in0_V``, ``out0_V``, internal ``hls::stream`` names).
    """
    result: dict[str, HLSLoopParams] = {}
    sol = solution_dir(node)
    if sol is None:
        return result
    report_dir = sol / "syn" / "report"
    db = sol / ".autopilot" / "db"
    for xml in sorted(report_dir.glob("*_csynth.xml")):
        fname = xml.name[: -len("_csynth.xml")]
        params = HLSLoopParams(source="report", top_level=fname == node.onnx_node.name)
        ptype, delay = parse_pipeline_type(xml)
        params.rewind = "auto-rewind" in ptype
        params.rewind_delay = delay
        params.interval = parse_interval(xml)
        loops = parse_loop_latencies(xml)
        depths = [d["PipelineDepth"] for d in loops.values() if "PipelineDepth" in d]
        trips = [d["TripCount"] for d in loops.values() if "TripCount" in d]
        params.depth = max(depths) if depths else None
        params.trip_count = max(trips) if trips else None
        adb = db / f"{fname}.sched.adb"
        if adb.is_file():
            for port, (op, stage, _lat) in parse_stream_op_stages(adb).items():
                target = params.read_stage if op == "read" else params.write_stage
                target.setdefault(port, stage)
        result[fname] = params
    return result


def hls_loop_params(node: HWCustomOp) -> HLSLoopParams:
    """Collect the loop parameters of an HLS node from its synthesis outputs.

    Returns defaults (``source == "default"``) when the node has not been synthesised.
    """
    params = HLSLoopParams()
    sol = solution_dir(node)
    xml = csynth_report_path(node)
    if sol is None or xml is None:
        return params
    params.source = "report"
    ptype, delay = parse_pipeline_type(xml)
    params.rewind = "auto-rewind" in ptype
    params.rewind_delay = delay
    params.dataflow = "dataflow" in ptype
    params.interval = parse_interval(xml)
    loops = parse_loop_latencies(xml)
    if not loops:
        # loop inside a non-inlined sub-function: use its report for depth/trips/delay
        params.top_level = False
        for other in sorted(xml.parent.glob("*_csynth.xml")):
            if other == xml or other.name == "csynth.xml":
                continue
            sub_loops = parse_loop_latencies(other)
            if sub_loops:
                loops = sub_loops
                sub_type, sub_delay = parse_pipeline_type(other)
                if "auto-rewind" in sub_type and not params.rewind:
                    params.rewind_delay = sub_delay
                    if params.dataflow:
                        # a dataflow process rewinds its loop by itself: the next frame
                        # follows after the rewind delay, not after the region's interval
                        params.rewind = True
                break
    depths = [d["PipelineDepth"] for d in loops.values() if "PipelineDepth" in d]
    trips = [d["TripCount"] for d in loops.values() if "TripCount" in d]
    if depths:
        params.depth = max(depths)
    if trips:
        params.trip_count = max(trips)
    db = sol / ".autopilot" / "db"
    if db.is_dir():
        for adb in sorted(db.glob("*.sched.adb")):
            for port, (op, stage, _lat) in parse_stream_op_stages(adb).items():
                target = params.read_stage if op == "read" else params.write_stage
                if port not in target:
                    target[port] = stage
    if not params.rewind and not params.dataflow:
        log.warning(
            f"HLS node {node.onnx_node.name}: top-level pipeline type '{ptype}' is not "
            "auto-rewind; the loop model assumes an idle gap of interval - trip count between "
            "frames, which is approximate"
        )
    return params


def hls_loop_pipeline_depth(node: HWCustomOp) -> int | None:
    """``PipelineDepth`` of the node's (deepest) pipelined loop, or None if unavailable."""
    return hls_loop_params(node).depth

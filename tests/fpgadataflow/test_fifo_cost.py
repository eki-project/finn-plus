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

"""Tests of ``finn.util.fifo_cost`` against the StreamingFIFO estimators and fifo.sv.

The module must agree with ``StreamingFIFO.get_fifo_cost()`` (both are views of
``resource_models._fifo_cost``), its style and capacity rules must be the ones of
``finn-rtllib/fifo/hdl/fifo.sv`` (cross-checked by parsing the RTL's localparams), and its
analytic breakpoints must equal a full scan of the cost function. The last test documents where
the new model deviates from the SRL16E/BRAM-block heuristics of the distributed simulation.
"""

import pytest

import math
import os
import re
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

from finn.util.fifo_cost import (
    DEFAULT_WEIGHTS,
    MIN_DEPTH,
    RESOURCES,
    STYLES,
    FIFOCost,
    capacity_weights,
    cost_breakpoints,
    cost_steps,
    cost_table,
    device_weights,
    fifo_cost,
    geometry,
    monotone_steps,
    realize_depth,
    resolve_style,
    scalarize,
)
from finn.util.platforms import ZU7EV_Platform
from finn.util.settings import get_settings

ZU7EV = "xczu7ev-ffvc1156-2-e"
VC1902 = "xcvc1902-vsva2197-2MP-e-S"


# The FIFO cost heuristics of the distributed simulation
# (finn.transformation.fpgadataflow.simulation_connected), restated here so that the
# comparison does not import the simulation stack: Q_srl style SRL16E LUTs, Vivado
# axis_data_fifo style BRAM/URAM blocks.
def calculate_srl16e_luts(depth, bitwidth):
    """``ceil(depth/32) * ceil(bitwidth/2)`` of the distributed simulation."""
    return math.ceil(depth / 32) * math.ceil(bitwidth / 2)


def calculate_bram_blocks(depth, bitwidth):
    """RAMB18 blocks as the distributed simulation counts them."""
    if bitwidth == 1:
        return math.ceil(depth / 16384)
    if bitwidth == 2:
        return math.ceil(depth / 8192)
    if bitwidth <= 4:
        return math.ceil(depth / 4096) * math.ceil(bitwidth / 4)
    if bitwidth <= 9:
        return math.ceil(depth / 2048) * math.ceil(bitwidth / 9)
    if bitwidth <= 18 or depth > 512:
        return math.ceil(depth / 1024) * math.ceil(bitwidth / 18)
    return math.ceil(depth / 512) * math.ceil(bitwidth / 36)


def calculate_uram_blocks(depth, bitwidth):
    """URAM288 blocks as the distributed simulation counts them."""
    return math.ceil(depth / 4096) * math.ceil(bitwidth / 72)


# the thresholds of the style ladder and the geometry, each with a neighbour on either side
DEPTHS = [2, 3, 4, 5, 6, 32, 33, 34, 64, 65, 66, 256, 257, 258, 513, 514, 1025, 2028, 2029]
DEPTHS += [2048, 2049, 4096, 4113, 4114, 5000, 8193, 16385, 40000, 65537]
WIDTHS = [1, 4, 5, 8, 11, 12, 16, 24, 37, 64, 72, 128, 512]


def make_fifo_node(depth, width_bits, ram_style="auto"):
    """A StreamingFIFO node whose (unpadded) stream width is ``width_bits``."""
    fold = 1 if width_bits <= 64 else width_bits // 64
    bits = width_bits // fold
    assert fold * bits == width_bits and bits <= 64
    dtype = DataType[f"UINT{bits}"]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, fold])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, fold])
    node = helper.make_node(
        "StreamingFIFO",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        depth=depth,
        folded_shape=[1, 1, fold],
        normal_shape=[1, fold],
        dataType=dtype.name,
        ram_style=ram_style,
    )
    graph = helper.make_graph(nodes=[node], name="fifo", inputs=[inp], outputs=[outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="fifo-cost-test"))
    model.set_tensor_datatype("inp", dtype)
    model.set_tensor_datatype("outp", dtype)
    return getCustomOp(model.graph.node[0])


@pytest.mark.fpgadataflow
def test_style_and_cost_match_streamingfifo_estimators():
    """resolve_style/fifo_cost agree with StreamingFIFO.resolve_ram_style/get_fifo_cost."""
    for width in [4, 8, 11, 12, 16, 24, 37, 64, 128]:
        for depth in DEPTHS:
            for style in STYLES:
                node = make_fifo_node(depth, width, style)
                padded = node.get_instream_width_padded()
                assert resolve_style(depth, padded, style) == node.resolve_ram_style()
                for part, versal in ((ZU7EV, False), (VC1902, True)):
                    cost = fifo_cost(depth, padded, style, versal)
                    assert cost.lut == node.lut_estimation(part)
                    assert cost.bram18 == node.bram_estimation(part)
                    assert cost.uram == node.uram_estimation(part)
                    assert cost.ff > 0


def _fifo_sv():
    """The text of the FIFO RTL."""
    with open(os.path.join(get_settings().finn_rtllib, "fifo", "hdl", "fifo.sv")) as f:
        return f.read()


def _localparam(sv, name):
    """The right-hand side of ``localparam ... NAME = ...;``."""
    m = re.search(r"localparam[^;]*?\b%s\s*=(.*?);" % name, sv, re.S)
    assert m is not None, f"{name} not found in fifo.sv; update finn.util.fifo_cost"
    return re.sub(r"\s+", " ", m.group(1)).strip()


@pytest.mark.fpgadataflow
def test_geometry_mirrors_fifo_sv_localparams():
    """The capacity rules are the ones of fifo.sv (parsed from the RTL as a cross-check)."""
    sv = _fifo_sv()
    # the ladder thresholds appear verbatim in the RAM_STYLE_EFF selection
    ladder = _localparam(sv, "RAM_STYLE_EFF")
    for threshold in ("DEPTH <= 33", "DEPTH <= 64 && DATA_WIDTH < 12", "DEPTH <= 257"):
        assert threshold in ladder
    assert "DATA_WIDTH < 5" in ladder and "DEPTH <= 2028" in ladder
    # the memory geometry constants
    min_abits = _localparam(sv, "MIN_ABITS")
    assert min_abits.startswith("IS_URAM? 12 : 9")
    assert _localparam(sv, "QDEPTH").startswith("IS_URAM? 16 : 0")
    assert _localparam(sv, "PIPE_DEPTH").startswith("IS_URAM? 3 + (2**ABITS.lo - 1) / 8192 : 2")
    assert "(DEPTH > 4? DEPTH : 5)" in _localparam(sv, "DEPTH_ACTUAL")
    assert "MEM_IMPL + PIPE_DEPTH + QDEPTH + 1" in _localparam(sv, "DEPTH_ACTUAL")

    # shift: four stages at least, one slot in the output register
    assert geometry(2, 8).capacity == 5 and geometry(2, 8).srl_depth == 4
    assert geometry(33, 8).capacity == 33 and geometry(33, 8).srl_depth == 32
    assert geometry(33, 8).srl_stages == 1 and geometry(100, 8, "srl").srl_stages == 4
    # distributed: exactly the requested depth, power-of-two LUTRAM
    assert geometry(257, 64).capacity == 257 and geometry(257, 64).lutram_rows == 256
    assert geometry(66, 64).lutram_rows == 128
    # block: the smallest BRAM space is 512 rows, two pipeline stages
    g = geometry(258, 64)
    assert (g.style, g.abits_lo, g.abits_hi, g.mem_impl, g.pipe_depth) == ("block", 9, 0, 512, 2)
    assert g.capacity == 512 + 2 + 0 + 1
    # a non-power-of-two depth gets a hi space when that saves a primitive
    g = geometry(1100, 64)
    assert (g.abits_lo, g.abits_hi, g.mem_impl) == (10, 7, 1024 + 128)
    # ultra: 16 entries in the output queue, pipeline grows with the cascade
    g = geometry(2029, 64)
    assert (g.style, g.qdepth, g.pipe_depth, g.abits_lo) == ("ultra", 16, 3, 11)
    assert g.capacity == 2048 + 3 + 16 + 1
    assert geometry(16384 + 17, 64).pipe_depth == 4
    # narrow words never leave LUTRAM with auto
    assert geometry(100000, 4).style == "distributed"
    for width in (4, 8):
        assert geometry(100000, width, "block").style == "block"


@pytest.mark.fpgadataflow
def test_breakpoints_match_full_scan():
    """The analytic breakpoints are exactly the depths at which the cost vector changes."""
    max_depth = 2**15 + 5
    for versal in (False, True):
        for style in STYLES:
            for width in (1, 4, 5, 12, 37, 64, 512):
                scan = []
                prev = fifo_cost(MIN_DEPTH, width, style, versal)
                for depth in range(MIN_DEPTH + 1, max_depth + 1):
                    cost = fifo_cost(depth, width, style, versal)
                    if cost != prev:
                        scan.append(depth - 1)
                        prev = cost
                scan.append(max_depth)
                assert cost_breakpoints(width, max_depth, style, versal) == scan, (
                    versal,
                    style,
                    width,
                )
                steps = cost_steps(width, max_depth, style, versal)
                assert [d for d, _ in steps] == scan
                for depth, cost in steps:
                    assert fifo_cost(depth, width, style, versal) == cost
    assert cost_breakpoints(64, 1) == []
    assert cost_breakpoints(64, 2) == [2]
    # the JSON export the driver tables are made of
    table = cost_table(64, 300, "auto", False)
    assert table["width"] == 64 and table["ram_style"] == "auto"
    assert [d for d, _ in table["steps"]] == cost_breakpoints(64, 300)
    assert table["steps"][-1] == [300, list(fifo_cost(300, 64).as_tuple())]


@pytest.mark.fpgadataflow
def test_scalarize_envelope_and_realize():
    """Default weights are the ZU7EV capacity; the envelope is monotone; realize rounds up."""
    lut, ff, bram18, uram, _ = ZU7EV_Platform().compute_resources[0]
    assert DEFAULT_WEIGHTS == capacity_weights(lut, ff, bram18, uram)
    assert DEFAULT_WEIGHTS["lut"] == 1.0 and DEFAULT_WEIGHTS["bram18"] == pytest.approx(
        377.05, 0.01
    )
    assert device_weights(board="ZCU104")["bram18"] == DEFAULT_WEIGHTS["bram18"]
    assert device_weights(board="Pynq-Z1")["uram"] == 16 * device_weights(board="Pynq-Z1")["bram18"]
    assert device_weights(board="nonexistent")["source"].startswith("default")
    # a board without a platform entry is matched by its device (RFSoC2x2 is a ZU28DR)
    rfsoc = device_weights("xczu28dr-ffvg1517-2-e", "RFSoC2x2")
    assert rfsoc["source"].startswith("ZCU111") and rfsoc["bram18"] == pytest.approx(425000 / 2160)

    cost = FIFOCost(lut=10, ff=20, bram18=2, uram=1)
    assert scalarize(cost, {"lut": 1, "bram18": 100}) == 210
    assert scalarize(cost) == pytest.approx(
        10 + 20 * DEFAULT_WEIGHTS["ff"] + 2 * DEFAULT_WEIGHTS["bram18"] + DEFAULT_WEIGHTS["uram"]
    )
    assert cost.as_dict() == dict(zip(RESOURCES, (10, 20, 2, 1)))
    assert (cost + cost).bram36 == 2

    steps = cost_steps(64, 70000)
    scalars = [scalarize(c) for _, c in steps]
    assert any(b < a for a, b in zip(scalars, scalars[1:])), "the raw ladder is not monotone"
    envelope = monotone_steps(steps)
    env_scalars = [scalarize(c) for _, c in envelope]
    assert all(a < b for a, b in zip(env_scalars, env_scalars[1:]))
    assert envelope[-1] == steps[-1]
    # every step is at least as expensive as the first envelope step at or above it
    env_depths = [d for d, _ in envelope]
    for depth, cost in steps:
        above = next(i for i, d in enumerate(env_depths) if d >= depth)
        assert scalarize(cost) >= env_scalars[above]

    # a 64-bit FIFO of depth 2028 takes 8 RAMB18, one URAM above is cheaper on a ZU7EV
    assert fifo_cost(2028, 64).bram18 == 8 and fifo_cost(2029, 64).uram == 1
    realized = realize_depth(2028, 64)
    assert realized > 2028 and fifo_cost(realized, 64).uram == 1
    assert scalarize(fifo_cost(realized, 64)) < scalarize(fifo_cost(2028, 64))
    # within a step the largest depth is returned
    assert realize_depth(100, 64) == cost_breakpoints(64, 128)[-2]
    assert realize_depth(1, 64) == 1


@pytest.mark.fpgadataflow
def test_deviations_from_the_distributed_simulation_heuristics():
    """Where the new model follows fifo.sv instead of the SRL16E/BRAM-block heuristics.

    The distributed simulation (``simulation_connected``) prices a FIFO as ``ceil(depth/32) *
    ceil(width/2)`` SRL16E LUTs up to ``max_qsrl_depth = 256`` and as ``calculate_bram_blocks``
    tiles above it. ``fifo.sv`` does none of that; the new module intentionally disagrees:
    """
    # (1) a word narrower than 5 bits is LUTRAM at any depth: no BRAM at all
    assert calculate_bram_blocks(4096, 4) == 1
    assert fifo_cost(4096, 4).bram18 == 0 and resolve_style(4096, 4) == "distributed"
    # (2) above 2028 the auto style is URAM, which the old model never counted
    assert calculate_bram_blocks(4096, 64) == 16 and calculate_uram_blocks(4096, 64) == 1
    assert fifo_cost(4096, 64) == FIFOCost(lut=148, ff=fifo_cost(4096, 64).ff, bram18=0, uram=1)
    # (3) LUTRAM/BRAM boundary at 257 (not 256) and LUTRAM steps at powers of two of
    #     depth - 1 (not multiples of 32): 130 and 255 cost the same LUTRAM, 129 does not
    #     (256 adds one bit of occupancy counter, 257 is the last LUTRAM depth)
    assert resolve_style(257, 64) == "distributed" and resolve_style(258, 64) == "block"
    assert calculate_srl16e_luts(130, 64) != calculate_srl16e_luts(255, 64)
    assert fifo_cost(130, 64) == fifo_cost(255, 64) and fifo_cost(129, 64) != fifo_cost(130, 64)
    assert [b for b in cost_breakpoints(64, 300) if 128 < b <= 300] == [129, 255, 257, 300]
    # (4) a shift register is only used up to 33 (64 for narrow words), and costs the fitted
    #     control logic on top of the SRL stages, not ceil(width/2) LUTs
    assert resolve_style(34, 8) == "srl" and resolve_style(65, 8) == "distributed"
    assert calculate_srl16e_luts(32, 64) == 32
    assert fifo_cost(32, 64).lut == 64 + 5 * 7 - 7
    # (5) the BRAM tier has sub-steps from the lo/hi decomposition: up to 1025 the FIFO is
    #     one 1024-row space in 4 tiles, 1026 adds a hi space in LUTRAM (more LUTs), 1100 a
    #     hi space in 2 more tiles; the old model prices all of them as 8 blocks
    assert calculate_bram_blocks(1026, 64) == calculate_bram_blocks(2048, 64) == 8
    assert fifo_cost(1025, 64).bram18 == 4 and fifo_cost(1100, 64).bram18 == 6
    assert fifo_cost(1026, 64).lut > fifo_cost(1025, 64).lut
    # a sub-step a deeper step undercuts (578..769: 4 tiles and 103 LUTs, 770..1025: 4 tiles
    # and 61 LUTs) is dominated and vanishes from the search's candidate set
    steps = cost_steps(64, 2028)
    env = [d for d, _ in monotone_steps(steps)]
    assert 769 in [d for d, _ in steps] and 769 not in env and 1025 in env

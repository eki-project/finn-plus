# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the lifecycle of XSI design libraries.

The XSI simulator kernel keeps process-global state and every design library produced by
xelab exports the same symbols. Loading a design while another one is still open used to
corrupt the simulator and crashed the interpreter with a segmentation fault somewhere
inside ``xsi_open``. These tests pin down the invariants that prevent that.
"""

import pytest

from finn_xsi.sim_engine import xsi
from pathlib import Path

from finn import xsi as finnxsi

# Two tiny, structurally different designs. They only need a clock, a reset and some
# state so that xelab emits a real simulation kernel for them.
DESIGN_SOURCES = {
    "xsi_lifecycle_a": """
`timescale 1ns/1ps
module xsi_lifecycle_a (
  input  wire        ap_clk,
  input  wire        ap_rst_n,
  input  wire [31:0] in0_V,
  output reg  [31:0] out0_V
);
  reg [31:0] acc;
  always @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      acc <= 0; out0_V <= 0;
    end else begin
      acc <= acc + in0_V;
      out0_V <= acc;
    end
  end
endmodule
""",
    "xsi_lifecycle_b": """
`timescale 1ns/1ps
module xsi_lifecycle_b (
  input  wire        ap_clk,
  input  wire        ap_rst_n,
  input  wire [47:0] in0_V,
  output reg  [47:0] out0_V
);
  reg [47:0] pipe [0:3];
  integer i;
  always @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      out0_V <= 0;
      for (i = 0; i < 4; i = i + 1) pipe[i] <= 0;
    end else begin
      pipe[0] <= in0_V ^ 48'hA5A5_5A5A_A5A5;
      for (i = 1; i < 4; i = i + 1) pipe[i] <= pipe[i-1];
      out0_V <= pipe[3];
    end
  end
endmodule
""",
}


@pytest.fixture(scope="module")
def compiled_designs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, Path]]:
    """Elaborate the two helper designs once and return their (base, relative) .so paths."""
    compiled = {}
    for top_name, source in DESIGN_SOURCES.items():
        out_dir = tmp_path_factory.mktemp(top_name)
        src_file = out_dir / f"{top_name}.v"
        src_file.write_text(source)
        compiled[top_name] = finnxsi.compile_sim_obj(top_name, [str(src_file)], out_dir)
    return compiled


def _cycle(sim: finnxsi.SimEngine, n: int = 4) -> None:
    """Run a few clock cycles so that the design is actually simulated."""
    for _ in range(n):
        sim.cycle({})


@pytest.mark.vivado
@pytest.mark.util
def test_alternating_designs_can_be_loaded(
    compiled_designs: dict[str, tuple[Path, Path]],
) -> None:
    """Two different designs can be loaded alternately as long as each one is closed."""
    for i in range(4):
        top_name = "xsi_lifecycle_a" if i % 2 == 0 else "xsi_lifecycle_b"
        sim = finnxsi.load_sim_obj(*compiled_designs[top_name])
        try:
            assert sim.is_open()
            _cycle(sim)
        finally:
            finnxsi.close_rtlsim(sim)
        assert not sim.is_open()


@pytest.mark.vivado
@pytest.mark.util
def test_close_rtlsim_is_idempotent(compiled_designs: dict[str, tuple[Path, Path]]) -> None:
    """Closing a simulation twice is harmless."""
    sim = finnxsi.load_sim_obj(*compiled_designs["xsi_lifecycle_a"])
    finnxsi.close_rtlsim(sim)
    finnxsi.close_rtlsim(sim)
    assert not sim.is_open()


@pytest.mark.vivado
@pytest.mark.util
def test_closed_design_raises_instead_of_crashing(
    compiled_designs: dict[str, tuple[Path, Path]],
) -> None:
    """Using a design or one of its ports after closing raises instead of segfaulting."""
    sim = finnxsi.load_sim_obj(*compiled_designs["xsi_lifecycle_a"])
    port = sim.top.getPort("ap_clk")
    finnxsi.close_rtlsim(sim)

    with pytest.raises(RuntimeError, match="closed"):
        sim.top.run(1)
    with pytest.raises(RuntimeError, match="closed"):
        port.write_back()


@pytest.mark.vivado
@pytest.mark.util
def test_leaked_design_is_closed_before_next_load(
    compiled_designs: dict[str, tuple[Path, Path]],
) -> None:
    """A simulation that was never closed must not leak into the next one.

    Only one XSI design can be open per process, so loading a second design while the
    first is still open used to hang or segfault inside ``xsi_open``.
    """
    leaked = finnxsi.load_sim_obj(*compiled_designs["xsi_lifecycle_a"])
    _cycle(leaked)
    assert leaked.is_open()

    # Deliberately do not close `leaked` here.
    sim = finnxsi.load_sim_obj(*compiled_designs["xsi_lifecycle_b"])
    try:
        assert not leaked.is_open(), "the leaked design should have been closed implicitly"
        assert sim.is_open()
        _cycle(sim)
    finally:
        finnxsi.close_rtlsim(sim)


@pytest.mark.vivado
@pytest.mark.util
def test_concurrent_designs_are_rejected(
    compiled_designs: dict[str, tuple[Path, Path]],
) -> None:
    """Bypassing load_sim_obj and opening two designs at once fails loudly."""
    sim = finnxsi.load_sim_obj(*compiled_designs["xsi_lifecycle_a"])
    try:
        base, rel = compiled_designs["xsi_lifecycle_b"]
        kernel = xsi.Kernel(finnxsi.get_simkernel_so())
        with pytest.raises(RuntimeError, match="still open"):
            xsi.Design(kernel, str(base / rel), None, None)
    finally:
        finnxsi.close_rtlsim(sim)

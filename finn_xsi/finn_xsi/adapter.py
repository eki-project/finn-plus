"""Interface adapter for FINN XSI."""

#############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @brief	rtlsim_multi_io interface adapter for FINN XSI
# @author	Yaman Umuroglu <yaman.umuroglu@amd.com>
#############################################################################

import errno
import numpy as np
import os
import weakref

# provided via pybind11
import xsi
from collections.abc import Mapping
from finn_xsi.sim_engine import SimEngine
from finn_xsi.srcutil import order_pkg_first
from pathlib import Path
from typing import Literal

from finn.util.basic import (
    get_rtlsim_timeout_error_message,
    launch_process_helper,
    resolve_xilinx_tool,
    wait_for_file,
)
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log


def locate_glbl() -> Path | None:
    """Try to determine the glbl.v file path from environment variables.
    Returns None if it cannot be found.
    """
    # Get GLBL from the Vitis environment variable
    vivado_path = os.environ.get("XILINX_VIVADO")
    if vivado_path:
        glbl_path = Path(vivado_path) / "data" / "verilog" / "src" / "glbl.v"
        if glbl_path.is_file():
            return glbl_path
    return None


def compile_sim_obj(
    top_module_name: str,
    source_list: list[str],
    sim_out_dir: Path,
    debug: bool = False,
    behav: bool = False,
    fifosim: bool = False,
) -> tuple[Path, Path]:
    """Compile the simulation object (.so) for the given top module and source files."""
    # create a .prj file with the source files
    with (sim_out_dir / "rtlsim.prj").open("w") as f:
        glbl = locate_glbl()
        if glbl is not None:
            f.write(f"verilog work {glbl}\n")

        # Check that every source file is found in the filesystem
        for src in source_list:
            if not wait_for_file(Path(src), timeout=5):
                raise FINNInternalError(
                    f"Source file {src} not found. Check that all codegen ran successfully and "
                    "that the source files are in the expected location."
                )

        # extract (unique, by using a set) verilog headers for inclusion
        verilog_headers = {str(Path(x).parent) for x in source_list if x.endswith((".vh", ".svh"))}
        verilog_header_incl_str = " ".join(["--include " + x for x in verilog_headers])

        # packages must come before the modules that import them
        for src_line in order_pkg_first(source_list):
            if src_line.endswith(".v"):
                f.write(f"verilog work {verilog_header_incl_str} {src_line}\n")
            elif src_line.endswith(".vhd"):
                # note that Verilog header incls are not added for VHDL
                f.write(f"vhdl2008 work {src_line}\n")
            elif src_line.endswith(".sv"):
                f.write(f"sv work {verilog_header_incl_str} {src_line}\n")
            elif src_line.endswith((".vh", ".svh")):
                # skip adding Verilog headers directly (see verilog_header_incl_str)
                continue
            else:
                raise FINNInternalError(f"Unknown extension for .prj file sources: {src_line}")

    # now call xelab to generate the .so for the design to be simulated
    # list of libs for xelab retrieved from Vitis HLS cosim cmdline
    # the particular lib version used depends on the Vivado/Vitis version being used
    # but putting in multiple (nonpresent) versions seems to pose no problem as long
    # as the correct one is also in there. at least this is how Vitis HLS cosim is
    # handling it.
    # TODO make this an optional param instead of hardcoding
    xelab_libs = [
        "smartconnect_v1_0",
        "axi_protocol_checker_v1_1_12",
        "axi_protocol_checker_v1_1_13",
        "axis_protocol_checker_v1_1_11",
        "axis_protocol_checker_v1_1_12",
        "xil_defaultlib",
        "unisims_ver",
        "xpm",
        "floating_point_v7_1_16",
        "floating_point_v7_0_21",
        "floating_point_v7_1_18",
        "floating_point_v7_1_15",
        "floating_point_v7_1_19",
        "floating_point_v7_1_21",
        "floating_point_v7_0_26",
        "work",
    ]

    cmd_xelab = [
        resolve_xilinx_tool("xelab"),
        "work." + top_module_name if not fifosim else "finn_design_wrapper",
        "-relax",
        "-dll",
        "--O3",
        "-s",
        top_module_name,
        "-prj",
        "rtlsim.prj",
        "-incr",
    ]
    # Xelab defaults to "auto" threading, which can expand to hundreds of
    # workers on shared servers. Large stitched FINNLoop designs have shown
    # intermittent elaborator SIGABRTs in that mode, so keep the default
    # bounded while still allowing explicit override.
    xelab_mt = os.environ.get("FINN_XELAB_MT", os.environ.get("NUM_DEFAULT_WORKERS", "8"))
    if xelab_mt == "1":
        xelab_mt = "off"
    cmd_xelab.extend(["--mt", xelab_mt])
    # Add debug flag if debug is enabled
    if debug:
        cmd_xelab.append("-debug")
        cmd_xelab.append("all")
    # Add behavioural simulation flag if behav is enabled
    if behav:
        cmd_xelab.append("-define")
        cmd_xelab.append("FINN_SIMULATION")
    for lib in xelab_libs:
        cmd_xelab.append("-L")
        cmd_xelab.append(lib)

    if locate_glbl() is not None:
        cmd_xelab.insert(1, "work.glbl")

    cmd_xvlog = ["xvlog", "--incr", "--relax", "-prj", "rtlsim.prj"]

    launch_process_helper(
        cmd_xvlog,
        cwd=sim_out_dir,
        print_stdout=False,
        timeout=None,  # xvlog can take a long time for large designs, so no timeout
        start_new_session=True,
        kill_process_group=True,
    )
    launch_process_helper(
        cmd_xelab,
        cwd=sim_out_dir,
        print_stdout=False,
        timeout=None,  # xelab can take a long time for large designs, so no timeout
        start_new_session=True,
        kill_process_group=True,
    )
    out_so_relative_path = Path(f"xsim.dir/{top_module_name}/xsimk.so")
    out_so_full_path = sim_out_dir / out_so_relative_path

    if not wait_for_file(out_so_full_path, timeout=5):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), out_so_full_path)

    return (sim_out_dir, out_so_relative_path)


def get_simkernel_so() -> Literal["libxv_simulator_kernel.so", "librdi_simulator_kernel.so"]:
    """Return the file name of the XSI simulation kernel library of the Vivado
    installation in XILINX_VIVADO. Vivado 2024.2 renamed it from
    librdi_simulator_kernel.so to libxv_simulator_kernel.so.
    """
    vivado_path = os.environ.get("XILINX_VIVADO")
    if not vivado_path:
        raise FINNUserError(
            "XILINX_VIVADO environment variable is not set. "
            "Did you source the Vitis/Vivado settings script?"
        )
    lib_dir = Path(vivado_path) / "lib" / "lnx64.o"
    if (lib_dir / "libxv_simulator_kernel.so").is_file():
        return "libxv_simulator_kernel.so"
    if (lib_dir / "librdi_simulator_kernel.so").is_file():
        return "librdi_simulator_kernel.so"
    raise FINNUserError(f"No XSI simulation kernel library found in {lib_dir}")


# The XSI simulator kernel keeps process-global state, and every design library produced
# by xelab exports the same symbols. Only a single design can therefore be open per
# process. Track the most recently loaded engine so that a simulation whose owner failed
# to close it (e.g. because an exception unwound past close_rtlsim) can be shut down
# before the next design is loaded, instead of corrupting the simulator.
_active_sim: "weakref.ReferenceType[SimEngine] | None" = None


def _register_sim(sim: SimEngine) -> None:
    """Remember sim as the simulation currently owning the XSI simulator kernel."""
    global _active_sim
    _active_sim = weakref.ref(sim)


def _close_stale_sim() -> None:
    """Close a previously loaded simulation that was never closed by its owner."""
    global _active_sim
    ref, _active_sim = _active_sim, None
    stale = None if ref is None else ref()
    if stale is not None and stale.is_open():
        log.warning(
            "An RTL simulation was still open when the next one was loaded and has been "
            "closed implicitly. Only one XSI design can be open per process. This "
            "usually means an exception unwound past the close_rtlsim() call of the "
            "previous simulation."
        )
        stale.close()


def load_sim_obj(
    sim_out_dir: Path,
    out_so_relative_path: Path,
    tracefile: str | None = None,
    simkernel_so: str | None = None,
) -> SimEngine:
    """Load the compiled simulation object (.so) and return a SimEngine instance."""
    if simkernel_so is None:
        simkernel_so = get_simkernel_so()
    # The xsim kernel opens some snapshot files lazily during simulation, e.g.
    # xsim.dir/<top>/xsim.svtype on the first executed $readmemh (XPM memories with
    # a MEMORY_INIT_FILE do so at #10). With a relative design path, these lookups
    # use the process CWD at that later time and abort the process (exit(1)) if it
    # is no longer sim_out_dir. Hence, load by absolute paths and pass the snapshot
    # directory as xsimDir, which does not change the CWD at all.
    sim_out_dir = Path(sim_out_dir).absolute()
    if not sim_out_dir.is_dir() or not wait_for_file(sim_out_dir / out_so_relative_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), sim_out_dir)
    _close_stale_sim()
    transcript = sim_out_dir / "rtlsim.log"
    if tracefile:
        # relative trace files land in sim_out_dir
        tracefile = str(sim_out_dir / tracefile)
    # has_xsim_dir is missing from an xsi module built before xsimDir support was added
    if getattr(xsi, "has_xsim_dir", False):
        sim = SimEngine(
            simkernel_so,
            str(sim_out_dir / out_so_relative_path),
            tracefile,
            str(sim_out_dir / "xsim.dir"),
            transcript,
        )
    else:
        # XSI < 2023.1 lacks xsimDir: fall back to a temporary chdir around the
        # design load. Only covers lazy file accesses during the load itself (e.g.
        # $readmemh at t=0), not ones after the first time advance.
        oldcwd = Path.cwd()
        os.chdir(sim_out_dir)
        try:
            sim = SimEngine(
                simkernel_so, str(out_so_relative_path), tracefile, transcript=transcript
            )
        finally:
            os.chdir(oldcwd)
    _register_sim(sim)
    if tracefile:
        sim.top.trace_all()
    # messages from the design load, incl. t=0 initial blocks
    sim.drain_messages()
    return sim


def reset_rtlsim(
    sim: SimEngine,
    rst_name: str = "ap_rst_n",  # noqa: ARG001
    active_low: bool = True,  # noqa: ARG001
    clk_name: str = "ap_clk",  # noqa: ARG001
    clk2x_name: str = "ap_clk2x",  # noqa: ARG001
    n_cycles: int = 16,  # noqa: ARG001
) -> None:
    """Reset the RTL simulation by toggling the reset signal for a specified number of cycles."""
    sim.do_reset()
    sim.run()


def close_rtlsim(sim: SimEngine) -> None:
    """Close the RTL simulation, ensuring that any pending traces are flushed.

    Raises SimulationError if the design reported an $error/$fatal since the last
    run, e.g. from a `final` block. The design is closed in either case.
    """
    try:
        # Drive sim_finish (if the design exposes it, see finn-rtllib/sim/hdl/sim_ctrl.v) so
        # that simulation-only `final` blocks (e.g. FIFO gauge logs) are flushed before
        # teardown
        try:
            sim_finish = sim.top.getPort("sim_finish")
        except RuntimeError:
            sim_finish = None
        if sim_finish is not None:
            sim_finish.set(1).write_back()
            sim.cycle({})
        # messages since the last run(), incl. final-block output
        sim.drain_messages()
    finally:
        # Explicitly finalize the design (calls xsi_close -> flushes+closes the .wdb)
        # instead of relying on GC, so that the waveform is not left unflushed on a
        # timeout or error exit.
        sim.close()


def rtlsim_multi_io(
    sim: SimEngine,
    io_dict: dict[str, dict[str, list[int]]],
    num_out_values: int | np.integer | Mapping[str, int | np.integer],
    sname: str = "_V_V",
    liveness_threshold: int = 10000,
    liveness_estimate: int | float | None = None,
) -> int:
    """Run the RTL simulation with multiple input and/or output streams.

    ``liveness_threshold`` is the number of no-output cycles after which the watchdogs fire;
    ``liveness_estimate`` is the (optional) derived cycle estimate it was based on, only used
    to produce a more helpful error message on timeout."""
    if len(io_dict["outputs"]) > 1:
        if not isinstance(num_out_values, dict):
            raise FINNInternalError("num_out_values must be dict for multiple output streams")
    else:
        # num_out_values is provided as integer (indicating the expected
        # outputs from the single output stream) - make into dict
        if not isinstance(num_out_values, int) and not (isinstance(num_out_values, np.integer)):
            raise FINNInternalError(
                f"num_out_values must be int for single output stream, "
                f"but got {type(num_out_values)}"
            )
        oname = next(iter(io_dict["outputs"].keys()))
        num_out_values = {oname: num_out_values}

    # FINN XSI expects hex strings, while rtlsim_multi_io uses
    # lists of arbitrary-precision integers, so need to convert
    # inputs and outputs to appropriate format
    # TODO: refactor components&data packing to directly generate and consume
    # hex strings instead of arb-prec Python integers
    for inp in io_dict["inputs"]:
        arbprec_int_input = io_dict["inputs"][inp]
        hexstring_input = (f"{var:0x}" for var in arbprec_int_input)
        stream_name = inp + sname
        sim.stream_input(stream_name, hexstring_input)

    hex_output_streams = {}
    watchdogs = []
    for out in io_dict["outputs"]:
        stream_name = out + sname
        watchdog = sim.create_watchdog(f"{stream_name} timeout", liveness_threshold)
        watchdogs.append(watchdog)
        hex_output_streams[out] = sim.collect_output(
            stream_name,
            int(num_out_values[out]),
            watchdog=watchdog,
        )

    start_ticks = sim.ticks
    try:
        ret = sim.run()
        if len(ret) > 0:
            raise FINNUserError(
                get_rtlsim_timeout_error_message(liveness_threshold, liveness_estimate)
                + f" Triggered watchdogs: {ret!s}. Check rtlsim_trace if any."
            )
        end_ticks = sim.ticks
        for out in io_dict["outputs"]:
            io_dict["outputs"][out] = [int(var, base=16) for var in hex_output_streams[out]]
    finally:
        # Remove the per-output watchdogs so they do not outlive this data pass.
        # sim.watchdogs is persistent; a leftover (already-exhausted) watchdog
        # would keep ticking and prematurely abort any later sim.run(), e.g. a
        # post-hook AXI-Lite weight read-back (see test_fpgadataflow_mvau).
        for watchdog in watchdogs:
            if watchdog in sim.watchdogs:
                sim.remove_watchdog(watchdog)

    return end_ticks - start_ticks

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
import re
from finn_xsi.sim_engine import SimEngine
from pathlib import Path
from typing import Literal

from finn.util.basic import launch_process_helper, wait_for_file
from finn.util.exception import FINNInternalError, FINNUserError


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

        for src_line in source_list:
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
        "work",
    ]

    cmd_xelab = [
        "xelab",
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
        timeout=240,
        start_new_session=True,
        kill_process_group=True,
    )
    launch_process_helper(
        cmd_xelab,
        cwd=sim_out_dir,
        print_stdout=False,
        timeout=300,
        start_new_session=True,
        kill_process_group=True,
    )
    out_so_relative_path = Path(f"xsim.dir/{top_module_name}/xsimk.so")
    out_so_full_path = sim_out_dir / out_so_relative_path

    if not wait_for_file(out_so_full_path, timeout=5):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), out_so_full_path)

    return (sim_out_dir, out_so_relative_path)


def get_simkernel_so() -> Literal["libxv_simulator_kernel.so", "librdi_simulator_kernel.so"]:
    """Determine the correct XSI simulator kernel .so filename based on the Vivado version."""
    vivado_path = os.environ.get("XILINX_VIVADO")
    if vivado_path is None:
        raise OSError(
            "XILINX_VIVADO environment variable is not set. "
            "Did you source the Vitis/Vivado settings script?"
        )
    # xsi kernel lib name depends on Vivado version (renamed in 2024.2)
    match = re.search(r"\b(20\d{2})\.(1|2)\b", vivado_path)
    if match is None:
        raise ValueError(f"Could not parse Vivado version from XILINX_VIVADO path: {vivado_path}")
    year, minor = int(match.group(1)), int(match.group(2))
    if (year, minor) > (2024, 1):
        simkernel_so = "libxv_simulator_kernel.so"
    else:
        simkernel_so = "librdi_simulator_kernel.so"
    return simkernel_so


def load_sim_obj(
    sim_out_dir: Path,
    out_so_relative_path: Path,
    tracefile: str | None = None,
    simkernel_so: str | None = None,
) -> SimEngine:
    """Load the compiled simulation object (.so) and return a SimEngine instance."""
    if simkernel_so is None:
        simkernel_so = get_simkernel_so()
    oldcwd = Path.cwd()
    if not sim_out_dir.is_dir() or not wait_for_file(sim_out_dir / out_so_relative_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), sim_out_dir)
    os.chdir(sim_out_dir)
    sim = SimEngine(simkernel_so, str(out_so_relative_path), "finnxsi_rtlsim.log", tracefile)
    if tracefile:
        sim.top.trace_all()
    os.chdir(oldcwd)
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
    """Close the RTL simulation, ensuring that any pending traces are flushed."""
    del sim


def rtlsim_multi_io(
    sim: SimEngine,
    io_dict: dict[str, dict[str, list[int]]],
    num_out_values: int | np.integer | dict[str, int | np.integer],
    sname: str = "_V_V",
    liveness_threshold: int = 10000,
) -> int:
    """Run the RTL simulation with multiple input and/or output streams."""
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
    for out in io_dict["outputs"]:
        stream_name = out + sname
        hex_output_streams[out] = sim.collect_output(
            stream_name,
            int(num_out_values[out]),
            watchdog=sim.create_watchdog(f"{stream_name} timeout", liveness_threshold),
        )

    start_ticks = sim.ticks
    ret = sim.run()
    if len(ret) > 0:
        raise FINNUserError(
            f"RTL simulation watchdogs {ret!s} timed out with {liveness_threshold} cycles. "
            f"Check rtlsim_trace if any."
        )
    end_ticks = sim.ticks
    for out in io_dict["outputs"]:
        io_dict["outputs"][out] = [int(var, base=16) for var in hex_output_streams[out]]

    return end_ticks - start_ticks

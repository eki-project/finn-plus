# Copyright (c) 2020 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

"""Helpers for RTL simulation: MLO pre-hook setup and performance metrics annotation."""

import numpy as np
from collections.abc import Callable
from numpy._typing._array_like import NDArray
from onnx import NodeProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from typing import TYPE_CHECKING, Any, cast

from finn.util.exception import FINNInternalError

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.rtl.finn_loop import FINNLoop
    from finn.xsi import SimEngine


def annotate_rtlsim_performance(
    rtlsim_stats: dict[str, Any], batch_size: int, clock_period_ns: float
) -> dict[str, Any]:
    """Add latency and throughput metrics to raw XSI simulation statistics.

    Overall throughput includes pipeline fill and is available for any completed
    run. Steady-state throughput requires at least two completed output frames;
    one frame provides latency only and cannot define an output-to-output rate.

    Args:
        rtlsim_stats: Dictionary of raw statistics from XSI simulation
        batch_size: Number of frames simulated
        clock_period_ns: Clock period in nanoseconds

    Returns:
        Updated rtlsim_stats dictionary with computed metrics
    """
    batch_size = int(batch_size)
    clock_period_ns = float(clock_period_ns)
    cycles = int(rtlsim_stats["cycles"])
    latency_cycles = int(rtlsim_stats["latency_cycles"])
    if batch_size <= 0:
        raise FINNInternalError("rtlsim batch size must be >0")
    if cycles <= 0:
        raise FINNInternalError("rtlsim cycle count must be >0")
    if clock_period_ns <= 0.0:
        raise FINNInternalError("rtlsim clock period must be >0")

    runtime_s = cycles * clock_period_ns * 1.0e-9
    rtlsim_stats["runtime[ms]"] = runtime_s * 1000.0
    rtlsim_stats["throughput[images/s]"] = batch_size / runtime_s
    rtlsim_stats["fclk[mhz]"] = 1000.0 / clock_period_ns

    timeout = int(rtlsim_stats.get("TIMEOUT", 1))
    unfinished_inputs = int(rtlsim_stats.get("UNFINISHED_INS", 1))
    unfinished_outputs = int(rtlsim_stats.get("UNFINISHED_OUTS", 1))
    run_complete = timeout == 0 and unfinished_inputs == 0 and unfinished_outputs == 0
    completed_frames = int(
        rtlsim_stats.get("completed_output_frames", batch_size if run_complete else 0)
    )
    run_complete = run_complete and completed_frames >= batch_size

    interval_cycles = int(rtlsim_stats.get("interval_cycles", 0))
    xsi_interval_valid = bool(
        int(rtlsim_stats.get("interval_valid", completed_frames >= 2 and interval_cycles > 0))
    )
    interval_valid = (
        run_complete and completed_frames >= 2 and interval_cycles > 0 and xsi_interval_valid
    )
    rtlsim_stats["interval_is_steady_state"] = interval_valid
    rtlsim_stats["fps_from_interval"] = (
        1.0e9 / (clock_period_ns * interval_cycles) if interval_valid else None
    )

    # New XSI results report the exact span and frame count between the first
    # and last completed outputs. Fall back to legacy results by removing the
    # first (pipeline-fill) frame from both the count and elapsed cycles.
    steady_state_frames = int(rtlsim_stats.get("steady_state_frames", max(0, batch_size - 1)))
    steady_state_cycles = int(
        rtlsim_stats.get("steady_state_cycles", max(0, cycles - latency_cycles))
    )
    stable_valid = (
        run_complete
        and completed_frames >= 2
        and steady_state_frames > 0
        and steady_state_cycles > 0
    )
    rtlsim_stats["stable_throughput_valid"] = stable_valid
    rtlsim_stats["stable_throughput[images/s]"] = (
        steady_state_frames * 1.0e9 / (clock_period_ns * steady_state_cycles)
        if stable_valid
        else None
    )
    return rtlsim_stats


def dat_file_to_numpy_array(file_path: Path | str) -> NDArray[np.uint8]:
    """Load a .dat file of hex strings into a uint8 numpy array."""
    byte_values = []

    with Path(file_path).open() as file:
        for line in file:
            hex_string = line.strip()
            for i in range(len(hex_string) - 2, -1, -2):
                byte = hex_string[i : i + 2]
                byte_values.append(int(byte, 16))
            if len(hex_string) % 2 == 1:  # Dealing when we have a leftover nibble
                byte_values.append(int(hex_string[-1], 16))
    return np.array(byte_values, dtype=np.uint8)


def mlo_prehook_func_factory(node: NodeProto) -> Callable[["SimEngine"], None]:
    """Construct a prehook function to
    setup the axi memory mapped interfaces for MLO validation using a function factory.
    """
    # Get the FINNLoop
    finnloop_op = cast("FINNLoop", getCustomOp(node))
    finnloop_body = cast("ModelWrapper", finnloop_op.get_nodeattr("body"))

    mvau_mlo_weights: dict[int, dict[str, np.ndarray | str | int]] = {}
    extern_idx = 0
    for idx, lb_inp in enumerate(finnloop_body.graph.input):
        downstream = finnloop_body.find_consumer(lb_inp.name)
        if downstream is None:
            raise FINNInternalError(
                f"Input {lb_inp.name} has no consumer in the FINNLoop body graph"
            )
        if downstream.op_type.startswith("MVAU"):
            mvau_mlo_weights[idx] = {}
            mvau_mlo_weights[idx]["name"] = lb_inp.name
            code_gen_dir = finnloop_op.get_nodeattr("code_gen_dir_ipgen")
            datfile = f"{code_gen_dir}/memblock_MVAU_rtl_id_{idx}.dat"
            # memblock.dat already holds the per-layer weights padded to LAYER_OFFS
            mvau_mlo_weights[idx]["value"] = dat_file_to_numpy_array(datfile)
            mvau_mlo_weights[idx]["extern_idx"] = extern_idx
            mvau_mlo_weights[idx]["extern_name"] = f"m_axi_MVAU_id_{idx}"
            mvau_mlo_weights[idx]["offset"] = getCustomOp(downstream).get_nodeattr("address_offset")
            extern_idx += 1

    def mlo_rtlsim_prehook(sim: "SimEngine") -> None:
        """Prehook that queues and populates AXI memory for MLO sims."""
        sim.aximm_queue("m_axi_intermediate_frame")
        for intf in mvau_mlo_weights.values():
            sim.aximm_ro_image(
                cast("str", intf["extern_name"]),
                cast("int", intf["offset"]),
                cast("np.ndarray", intf["value"]).flatten(),
            )

    return mlo_rtlsim_prehook

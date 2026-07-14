# Copyright (C) 2025, Advanced Micro Devices, Inc.
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


"""Module contains helpers for handling the MLO rtlsimulation. It instantiates
aximm simulation tasks for handling the aximm interfaces."""

import numpy as np
from collections.abc import Callable
from numpy._typing._array_like import NDArray
from onnx import NodeProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from typing import TYPE_CHECKING, cast

from finn.util.exception import FINNInternalError
from finn.xsi import SimEngine

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.rtl.finn_loop import FINNLoop


def is_mlo(model: ModelWrapper) -> bool:
    """Return True if the model is an MLO model, false otherwise."""
    return any(node.op_type == "FINNLoop" for node in model.graph.node)


def dat_file_to_numpy_array(file_path: Path) -> NDArray[np.uint8]:
    """Load a .dat file of hex strings into a uint8 numpy array."""
    byte_values = []

    with file_path.open() as file:
        for line in file:
            hex_string = line.strip()
            for i in range(len(hex_string) - 2, -1, -2):
                byte = hex_string[i : i + 2]
                byte_values.append(int(byte, 16))
            if len(hex_string) % 2 == 1:  # Dealing when we have a leftover nibble
                byte_values.append(int(hex_string[-1], 16))
    byte_array = np.array(byte_values, dtype=np.uint8)

    return byte_array


def mlo_prehook_func_factory(node: NodeProto) -> Callable[[SimEngine], None]:
    """Construct a prehook function to
    setup the axi memory mapped interfaces for MLO validation using a function factory.
    """
    # Get the FINNLoop
    finnloop_op = cast("FINNLoop", getCustomOp(node))

    finnloop_body = cast("ModelWrapper", finnloop_op.get_nodeattr("body"))

    mvau_hbm_weights: dict[int, dict[str, np.ndarray | str | int]] = {}
    extern_idx = 0
    for idx, lb_inp in enumerate(finnloop_body.graph.input):
        downstream = finnloop_body.find_consumer(lb_inp.name)
        if downstream is None:
            raise FINNInternalError(
                f"Input {lb_inp.name} has no consumer in the FINNLoop body graph"
            )
        if downstream.op_type.startswith("MVAU"):
            mvau_hbm_weights[idx] = {}
            mvau_hbm_weights[idx]["name"] = lb_inp.name
            datfile = (
                f"{finnloop_op.get_nodeattr('code_gen_dir_ipgen')}/memblock_MVAU_rtl_id_{idx}.dat"
            )
            mvau_hbm_weights[idx]["value"] = dat_file_to_numpy_array(Path(datfile))
            mvau_hbm_weights[idx]["extern_idx"] = extern_idx
            mvau_hbm_weights[idx]["extern_name"] = f"m_axi_MVAU_id_{idx}"
            extern_idx += 1

    def mlo_rtlsim_prehook(sim: SimEngine) -> None:
        """Prehook that queues and populates AXI memory for MLO sims."""
        sim.aximm_queue("m_axi_hbm")
        for intf in mvau_hbm_weights.values():
            sim.aximm_ro_image(
                cast("str", intf["extern_name"]), 0, cast("np.ndarray", intf["value"]).flatten()
            )

    return mlo_rtlsim_prehook

# Copyright (C) 2023, Advanced Micro Devices, Inc.
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

"""Base class for hardware custom operations in FINN dataflow architecture.

This module provides the HWCustomOp base class for custom operations that can be
implemented using HLS or RTL backends in FPGA dataflow architectures.
"""

import numpy as np
import numpy.typing as npt
from abc import abstractmethod
from collections.abc import Sequence
from onnx import NodeProto
from pathlib import Path
from qonnx.core.datatype import BaseDataType
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import roundup_to_integer_multiple
from typing import TYPE_CHECKING, Any, cast

from finn.util.basic import get_watchdog_timeout_cycles
from finn.util.exception import FINNInternalError

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper
    from finn.transformation.fpgadataflow.loop_rolling import LoopBodyInputType
    from finn.xsi import SimEngine

import finn.xsi as finnxsi


class HWCustomOp(CustomOp):
    """HWCustomOp class all custom ops that can be implemented with either
    HLS or RTL backend are based on. Contains different functions every fpgadataflow
    custom node should have. Some as abstract methods, these have to be filled
    when writing a new fpgadataflow custom op node.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize HWCustomOp with an ONNX node.

        Args:
            onnx_node: The ONNX node to wrap.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(onnx_node, **kwargs)
        self.code_gen_dict = {}

    def get_nodeattr_types(
        self,
    ) -> dict[
        str,
        tuple[str, bool, int | float | str | bool | np.ndarray | list]
        | tuple[str, bool, int | float | str | bool | np.ndarray | list, set | None],
    ]:
        """Return node attribute types for HWCustomOp.

        Returns:
            Dictionary mapping attribute names to their type specifications.

        """
        return {
            "backend": ("s", True, "fpgadataflow"),
            "preferred_impl_style": ("s", False, "", {"", "hls", "rtl"}),
            "code_gen_dir_ipgen": ("s", False, ""),
            "ipgen_path": ("s", False, ""),
            "ip_path": ("s", False, ""),
            "ip_vlnv": ("s", False, ""),
            "exec_mode": ("s", False, "", {"", "rtlsim", "cppsim"}),
            "cycles_rtlsim": ("i", False, 0),
            "cycles_estimate": ("i", False, 0),
            "rtlsim_trace": ("s", False, ""),
            "res_estimate": ("s", False, ""),
            "res_synth": ("s", False, ""),
            "rtlsim_so": ("s", False, ""),
            # partitioning info
            # ID of SLR to which the Op is attached in Vitis builds
            # Set to -1 as 'don't care'
            "slr": ("i", False, -1),
            # Vitis memory port to which any AXI-MM interface
            # of this Op should be attached in Vitis builds
            # E.g.: "DDR[0]", "HBM[0]", "PLRAM[0]"
            "mem_port": ("s", False, ""),
            # Partition to which the Op belongs; all Ops with the
            # same partition_id are stitched together
            # Users should avoid setting this attribute manually
            # and instead use the floorplan transform to set
            # partition IDs from Vitis design rules and SLR IDs
            "partition_id": ("i", False, 0),
            # ID of FPGA device to which this Op is allocated, in
            # a multi-FPGA setting
            "device_id": ("i", False, 0),
            # input and output FIFO depths for multi-I/O nodes
            "inFIFODepths": ("ints", False, [2]),
            "outFIFODepths": ("ints", False, [2]),
            "output_hook": ("s", False, ""),
            "bodies": ("i", False, 0),
            "mlo_max_iter": ("i", False, 0),
            # DDR byte offset of this node's (streamed) weights, see AssignMemoryOffset
            "address_offset": ("i", False, 0),
        }

    def make_shape_compatible_op(self, model: "ModelWrapper") -> NodeProto:  # noqa: ARG002
        """Make a shape compatible operation.

        Args:
            model: The model wrapper containing this node.

        Returns:
            The ONNX node for the shape compatible operation.
        """
        oshape = self.get_normal_output_shape()
        if oshape is None:
            raise FINNInternalError(
                f"Cannot make shape compatible op for {self.onnx_node.name} "
                "since normal output shape is not defined."
            )
        # implement tensor with correct shape
        return super().make_const_shape_op(oshape)

    def get_verilog_top_module_name(self) -> str:
        """Return the Verilog top module name for this node."""
        node = self.onnx_node
        prefixed_top_name = node.name

        return prefixed_top_name

    def get_verilog_top_module_intf_names(self) -> dict[str, list[tuple[str, int]] | list[str]]:
        """Return a dict of names of input and output interfaces.
        The keys reflect the protocols each interface implements:
        'clk', 'rst', 'm_axis', 's_axis', 'aximm', 'axilite'.
        Values are lists of tuples (axis, aximm) or names (axilite):
        'axis' tuples correspond to the list of node inputs in order,
        each tuple is (interface_name, interface_width_bits).
        axilite always assumed to be 32 bits and is not tuple (name only).
        Each block must have at most one aximm and one axilite.
        """
        node = self.onnx_node
        intf_names = {}
        intf_names["clk"] = ["ap_clk"]
        intf_names["rst"] = ["ap_rst_n"]
        intf_names["s_axis"] = []
        for i in range(len(node.input)):
            # not every node input will result in an interface of the produced HW
            # filter out inputs that have no stream width associated with them
            width = self.get_instream_width_padded(i)
            if width != 0:
                intf_names["s_axis"].append((f"in{i}_V", self.get_instream_width_padded(i)))
        intf_names["m_axis"] = []
        for i in range(len(node.output)):
            intf_names["m_axis"].append((f"out{i}_V", self.get_outstream_width_padded(i)))
        intf_names["aximm"] = []
        intf_names["axilite"] = []
        intf_names["ap_none"] = []
        return intf_names

    def get_rtlsim(self) -> "SimEngine":
        """Return a xsi wrapper for the emulation library for this node."""
        rtlsim_so = Path(cast("str", self.get_nodeattr("rtlsim_so")))
        if not rtlsim_so.is_file():
            raise FINNInternalError(
                f"rtlsim_so attribute points to non-existent file in {self.onnx_node.name}, "
                "cannot get rtlsim"
            )

        rtlsim_parts = rtlsim_so.parts
        try:
            xsim_idx = rtlsim_parts.index("xsim.dir")
        except ValueError as exc:
            raise FINNInternalError(
                f"rtlsim_so path does not contain xsim.dir for {self.onnx_node.name}"
            ) from exc
        sim_base = Path(*rtlsim_parts[:xsim_idx])
        sim_rel = Path(*rtlsim_parts[xsim_idx:])
        # pass in correct tracefile from attribute
        tracefile = cast("str", self.get_nodeattr("rtlsim_trace"))
        if tracefile == "default":
            tracefile = self.onnx_node.name + ".wdb"
        sim = finnxsi.load_sim_obj(sim_base, sim_rel, tracefile)

        return sim

    def close_rtlsim(self, sim: "SimEngine") -> None:
        """Close and free up resources for rtlsim.

        Args:
            sim: The RTL simulation object to close.

        """
        finnxsi.close_rtlsim(sim)

    def node_res_estimation(self, fpgapart: str) -> dict[str, int | float]:
        """Return summarized resource estimation of BRAMs and LUTs of the node as a dictionary."""
        ret = {}
        ret["BRAM_18K"] = self.bram_estimation(fpgapart)
        ret["BRAM_efficiency"] = self.bram_efficiency_estimation(fpgapart)
        ret["LUT"] = self.lut_estimation(fpgapart)
        ret["URAM"] = self.uram_estimation(fpgapart)
        ret["URAM_efficiency"] = self.uram_efficiency_estimation(fpgapart)
        ret["DSP"] = self.dsp_estimation(fpgapart)
        return ret

    def bram_efficiency_estimation(self, fpgapart: str) -> float:  # noqa: ARG002
        """Estimate BRAM efficiency.

        Returns actual parameter storage needed divided by the allocated BRAM
        storage (from estimation).

        """
        return 1

    def uram_efficiency_estimation(self, fpgapart: str) -> float:  # noqa: ARG002
        """Estimate URAM efficiency.

        Returns actual parameter storage needed divided by the allocated URAM
        storage (from estimation).

        """
        return 1

    def bram_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Estimate BRAM resource usage.

        Member function of HWCustomOp class that must be implemented by every node.

        """
        return 0

    def uram_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Estimate UltraRAM resource usage.

        Member function of HWCustomOp class that must be implemented by every node.

        """
        return 0

    def lut_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Estimate LUT resource usage.

        Member function of HWCustomOp class that must be implemented by every node.

        """
        return 0

    def dsp_estimation(self, fpgapart: str) -> int:  # noqa: ARG002
        """Estimate DSP resource usage.

        Member function of HWCustomOp class that must be implemented by every node.

        Args:
            fpgapart: Target FPGA part string.

        """
        return 0

    def get_exp_cycles(self) -> int:
        """Estimate expected cycles for set folding.

        Member function of HWCustomOp class that must be implemented by every node.

        """
        return 0

    def get_op_and_param_counts(self) -> dict[str, int]:
        """Return a dictionary with number of ops needed per inference.

        Returns number of ops needed per inference for this layer as well as
        parameter count (weights, thresholds, etc.). Entries should be in the
        format: {op_<optype> : <count>, param_<paramtype>: <count>}.

        """
        return {}

    def reset_rtlsim(self, sim: "SimEngine") -> None:
        """Set reset input in finnxsi to zero, toggle the clock and set it back to one."""
        finnxsi.reset_rtlsim(sim)

    def rtlsim_multi_io(
        self, sim: "SimEngine", io_dict: dict[str, dict[str, list[int]]], sname: str = "_V"
    ) -> None:
        """Run rtlsim for this node, supports multiple i/o streams."""
        num_out_values = self.get_number_output_values()
        # LIVENESS_THRESHOLD can only increase the expected cycle count.
        exp_cycles = self.get_exp_cycles()
        effective_threshold = get_watchdog_timeout_cycles(exp_cycles)
        total_cycle_count = finnxsi.rtlsim_multi_io(
            sim,
            io_dict,
            num_out_values,
            sname=sname,
            liveness_threshold=effective_threshold,
            liveness_estimate=exp_cycles,
        )

        self.set_nodeattr("cycles_rtlsim", total_cycle_count)

    def verify_node(self) -> Any:
        """Can be implemented to verify that all attributes the node needs
        are there and that particular attributes are set correctly. Can also
        check if the number of inputs is equal to the expected number."""

    def generate_params(
        self, model: "ModelWrapper", path: str | Path, fpgapart: str | None = None
    ) -> None:
        """Generate parameters (i.e. weights and thresholds). ``fpgapart`` is only
        consumed by nodes whose parameter layout depends on the target device (e.g.
        Requant, which resolves the DSP version from it).

        Member function of HWCustomOp class that must be implemented by every node
        that needs to generate parameters.

        Args:
            model: The model wrapper containing this node.
            path: Path where parameters should be generated.
            fpgapart: Target FPGA part, for nodes whose parameter layout depends on it.

        """

    def get_number_output_values(self) -> int:
        """Get the number of expected output values.

        Member function of HWCustomOp class that must be implemented by every node.

        """
        folded_oshape = self.get_folded_output_shape()
        if folded_oshape is None:
            raise FINNInternalError(
                f"Cannot get number of output values for {self.onnx_node.name} "
                "since folded output shape is not defined."
            )
        return int(np.prod(folded_oshape[:-1]))

    @abstractmethod
    def get_input_datatype(self, ind: int = 0) -> BaseDataType:
        """Return FINN DataType of input stream ind."""

    @abstractmethod
    def get_output_datatype(self, ind: int = 0) -> BaseDataType:
        """Return FINN DataType of output stream ind."""

    @abstractmethod
    def get_normal_input_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:
        """Return normal input shape if implemented."""

    @abstractmethod
    def get_normal_output_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:
        """Return folded output shape if implemented."""

    @abstractmethod
    def get_folded_input_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:
        """Return folded input shape (according to synapse folding), if implemented."""

    @abstractmethod
    def get_folded_output_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:
        """Return folded output shape (according to neuron folding), if implemented."""

    @abstractmethod
    def get_instream_width(self, ind: int = 0) -> int:
        """Return input stream width, if implemented."""

    @abstractmethod
    def get_outstream_width(self, ind: int = 0) -> int:
        """Return output stream width, if implemented."""

    def get_instream_width_padded(self, ind: int = 0) -> int:
        """Return input stream width padded to a multiple of 8.

        This is required by the AXI Stream spec.

        Args:
            ind: Input index (default: 0).

        """
        in_width = self.get_instream_width(ind=ind)
        if in_width != 0:
            return roundup_to_integer_multiple(in_width, 8)
        return 0

    def get_outstream_width_padded(self, ind: int = 0) -> int:
        """Return output stream width padded to a multiple of 8.

        This is required by the AXI Stream spec.

        Args:
            ind: Output index (default: 0).

        """
        out_width = self.get_outstream_width(ind=ind)
        return roundup_to_integer_multiple(out_width, 8)

    def adapt_for_loop_body(self, input_types: "list[LoopBodyInputType]") -> None:
        """Called by LoopRolling transformation to allow operators to adapt their
        attributes when being placed inside a loop body.

        This base implementation does nothing. Operators that need to modify
        their behavior when placed in loops should override this method.

        Args:
            input_types: List of LoopBodyInputType values for each input,
                         indicating whether inputs are ACTIVATION, CONSTANT,
                         PARAMETER, etc.

        Example:
            If an operator has a parameter that becomes a streamed input
            in a loop context (PARAMETER type), it might need to change
            an attribute like `rhs_style` from "const" to "input".
        """  # noqa: D401
        # Default: no adaptation needed

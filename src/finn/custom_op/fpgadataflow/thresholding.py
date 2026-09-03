# Copyright (C) 2024, Advanced Micro Devices, Inc.
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

"""Multi-threshold activation hardware custom operator.

The thresholding operation compares input values against a set of thresholds to
produce quantized outputs.
"""

import numpy as np
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.util.basic import (
    interleave_matrix_outer_dim_from_partitions,
    roundup_to_integer_multiple,
)
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.exception import FINNInternalError
from finn.util.logging import log

if TYPE_CHECKING:
    from onnx import GraphProto

# Type of the dictionary returned by get_nodeattr_types: maps attribute names to
# their (dtype, required, default[, allowed_values]) specification tuples
NodeAttrTypes = dict[
    str,
    tuple[str, bool, int | float | str | bool | np.ndarray | list]
    | tuple[str, bool, int | float | str | bool | np.ndarray | list, set | None],
]


class Thresholding(HWCustomOp):
    """Abstraction layer for HW implementation of Thresholding."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize the Thresholding node."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return a dictionary of attribute names and their types for this node.

        The dictionary describes node attributes including parallelization (PE),
        number of channels, data types, and runtime configuration options.
        """
        my_attrs: NodeAttrTypes = {
            # whether weights (thresholds) will be
            # writable through an AXI-lite interface during runtime
            # 1 for enabled, 0 for disabled.
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
            # parallelization; channels thresholded per cycle
            "PE": ("i", True, 0),
            # number of channels (each may have different thresholds)
            "NumChannels": ("i", True, 0),
            # number of steps in thresholding function. Used only in decoupled mode
            "numSteps": ("i", True, 1),
            # FINN DataTypes for inputs, outputs
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
            # initialization value for the thresholding accumulator
            "ActVal": ("i", False, 0),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    @property
    def pe(self) -> int:
        """Return the configured parallelism (channels thresholded per cycle)."""
        return cast("int", self.get_nodeattr("PE"))

    @property
    def num_channels(self) -> int:
        """Return the number of channels."""
        return cast("int", self.get_nodeattr("NumChannels"))

    @property
    def num_steps(self) -> int:
        """Return the number of threshold steps."""
        return cast("int", self.get_nodeattr("numSteps"))

    @property
    def act_val(self) -> int:
        """Return the thresholding accumulator initialization value."""
        return cast("int", self.get_nodeattr("ActVal"))

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer and set the data types for node inputs and outputs.

        Updates the ``inputDataType`` attribute based on the model's tensor datatype
        and sets the output tensor datatype based on the ``outputDataType`` attribute.
        """
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype(0):
            log.warning(
                f"inputDataType changing for {node.name}: "
                f"{self.get_input_datatype(0).name} -> {idt.name}"
            )
        self.set_nodeattr("inputDataType", idt.name)
        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def verify_node(self) -> list[str]:
        """Verify that the node is configured correctly.

        Checks that the backend attribute is set to ``fpgadataflow`` and that
        all necessary attributes exist.
        """
        info_messages = []
        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify that all necessary attributes exist
        # TODO collect automatically from get_nodeattr_types
        try:
            self.get_nodeattr("code_gen_dir_cppsim")
            self.get_nodeattr("executable_path")
            self.get_nodeattr("NumChannels")
            self.get_nodeattr("PE")
            self.get_nodeattr("inputDataType")
            self.get_nodeattr("outputDataType")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append("""The required Threshold_Batch attributes do not exist.""")

        return info_messages

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:
        """Return FINN DataType of input."""
        if ind == 0:
            return DataType[cast("str", self.get_nodeattr("inputDataType"))]
        if ind == 1:
            return DataType[cast("str", self.get_nodeattr("weightDataType"))]
        raise FINNInternalError(f"{self.onnx_node.name}: input ind {ind} out of range")

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of output."""
        return DataType[cast("str", self.get_nodeattr("outputDataType"))]

    def minimize_weight_bit_width(self, model: ModelWrapper) -> BaseDataType:
        """Minimize threshold datatype bitwidth based on actual threshold values.

        This function should not round or clip the threshold values; that is done
        in RoundAndClipThresholds.
        """
        thresholds = model.get_initializer(self.onnx_node.input[1])
        if self.get_nodeattr("runtime_writeable_weights") or self.get_nodeattr("mlo_max_iter"):
            return DataType[cast("str", self.get_nodeattr("weightDataType"))]
        if not isinstance(thresholds, np.ndarray):
            raise FINNInternalError(f"{self.onnx_node.name}: threshold initializer is missing")
        threshold_tensor = self.get_hw_compatible_threshold_tensor(thresholds)
        # TODO: extend this for fixed point
        if self.get_input_datatype(0).is_integer() and self.get_input_datatype(1).is_integer():
            # minimize threshold width only if input and thresholds are integer
            # Use double precision for intermediate calculations to prevent overflow
            min_threshold = float(thresholds.min())
            max_threshold = float(thresholds.max())
            # Check if input datatype is signed
            input_is_signed = self.get_input_datatype(0).signed()
            # Special case: all thresholds are zero
            # get_smallest_possible(-1) returns BIPOLAR which can't represent 0
            if min_threshold == max_threshold == 0:
                tdt = DataType["INT2"] if input_is_signed else DataType["UINT1"]
            elif min_threshold < 0:
                if abs(min_threshold) > max_threshold:
                    tdt = DataType.get_smallest_possible(min_threshold)
                else:
                    tdt = DataType.get_smallest_possible(-max_threshold - 1)
            elif input_is_signed:
                # If input is signed, use signed threshold datatype even if thresholds are positive
                tdt = DataType.get_smallest_possible(-max_threshold - 1)
            else:
                tdt = DataType.get_smallest_possible(max_threshold)
        else:
            # special case: if input is float, we keep thresholds as is
            tdt = self.get_input_datatype(1)
        if not np.vectorize(tdt.allowed)(threshold_tensor).all():
            raise FINNInternalError(f"Thresholds can't be expressed with type {tdt!s}")
        self.set_nodeattr("weightDataType", tdt.name)
        # Update QONNX DataType of tensor for consistency
        model.set_tensor_datatype(self.onnx_node.input[1], tdt)
        return DataType[cast("str", self.get_nodeattr("weightDataType"))]

    def get_instream_width(self, ind: int = 0) -> int:
        """Return the width of the input stream in bits.

        ``ind`` is the input index (0 for data input, 1 for threshold/weight input).
        """
        if ind == 0:
            i_bits = self.get_input_datatype(0).bitwidth()
            return i_bits * self.pe
        if ind == 1:
            # try to access mem_mode attribute, doesn't exist for RTL Thresholding
            try:
                mem_mode = self.get_nodeattr("mem_mode")
            except AttributeError:
                mem_mode = 0
            if mem_mode == "internal_decoupled":
                wp = self.get_input_datatype(1).bitwidth()
                return self.pe * wp * self.num_steps
            return 0
        raise FINNInternalError(f"{self.onnx_node.name}: input ind {ind} out of range")

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return the width of the output stream in bits."""
        o_bits = self.get_output_datatype().bitwidth()
        return o_bits * self.pe

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the folded input shape for hardware implementation.

        The folded shape accounts for parallelization (PE) and temporal memory
        (TMEM) organization used in the hardware accelerator.
        """
        fold = self.calc_tmem()
        vecs = list(cast("list[int]", self.get_nodeattr("numInputVectors")))
        return (*vecs, fold, self.pe)

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the folded output shape for hardware implementation.

        Same shape as the folded input shape.
        """
        return self.get_folded_input_shape()

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the normal (unfolded) input shape."""
        vecs = list(cast("list[int]", self.get_nodeattr("numInputVectors")))
        return (*vecs, self.num_channels)

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the normal (unfolded) output shape.

        Same shape as the normal input shape.
        """
        return self.get_normal_input_shape()

    def get_exp_cycles(self) -> int:
        """Return the expected number of execution cycles.

        Calculated as: Channels/PE * batch size * feature map dimensions.
        """
        # Channels/PE * batch size * fmdim * fmdim
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    def get_hw_compatible_threshold_tensor(self, orig_thres_matrix: np.ndarray) -> np.ndarray:
        """Convert the original numpy threshold matrix into hlslib-compatible form.

        The steps performed are:
        * ensure MH % PE == 0
        * for unsigned inputs, ensure thresholds are positive
        * interleave rows between PEs
        * reshape into (PE, TMEM, n_thres_steps) and return
        """
        mh = self.num_channels
        pe = self.pe
        tmem = mh // pe
        if mh % pe != 0:
            raise FINNInternalError("Requirement NumChannels divisable by PE is violated.")
        if orig_thres_matrix.ndim != 2:
            raise FINNInternalError("Threshold matrix dimension is not as expected (2).")
        n_thres_steps = orig_thres_matrix.shape[1]
        if n_thres_steps != self.num_steps:
            raise FINNInternalError("Mismatch in threshold steps")
        if not self.get_input_datatype(0).signed() and not (orig_thres_matrix >= 0).all():
            # ensure all thresholds are nonnegative
            raise FINNInternalError("Threshold matrix contains negative values for unsigned input")
        ret = orig_thres_matrix
        # ensure channels = mh , duplicating if necessary
        if ret.shape[0] == 1:
            ret = np.tile(ret, (mh, 1))
        if ret.shape[0] != mh:
            raise FINNInternalError("Channels of threshold matrix are not as expected (mh)")
        # distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        if ret.shape[0] != pe:
            raise FINNInternalError(
                "First dimension after distribution of the rows between PEs "
                "is not as expected (pe)"
            )
        if ret.shape[1] != tmem:
            raise FINNInternalError(
                "Second dimension after distribution of the rows between PEs "
                "is not as expected (tmem)"
            )
        if ret.shape[2] != n_thres_steps:
            raise FINNInternalError(
                "Third dimension after distribution of the rows between PEs "
                "is not as expected (n_thres_steps)"
            )
        return ret.reshape(1, pe, tmem, n_thres_steps)

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"
    ) -> None:  # noqa: ARG002
        """Execute the thresholding operation.

        Performs multi-threshold comparison on input values using the threshold
        tensor. Handles data layout transformations and applies output bias
        (ActVal) if configured. Converts output to bipolar format if the output
        data type is BIPOLAR.
        """
        node = self.onnx_node
        inp_values = context[node.input[0]]
        th_val = context[node.input[1]]
        out_bias = self.act_val

        # Consider the data layout for transposing the input into the format
        # accepted by the multithreshold function above, i.e, the channel
        # dimension is along the axis with index 1.
        # If there is no layout annotation, guess based on rank of the tensor
        # TODO: Currently there is no mechanism here to get the layout
        #  annotation, we always guess, but this matches the previous behavior.
        if len(inp_values.shape) >= 5:
            raise FINNInternalError(
                f"{node.name}: cannot guess a data layout for rank-{len(inp_values.shape)} input"
            )
        # Maps tensor rank to layout annotation
        rank_to_layout = {0: None, 1: "C", 2: "NC", 3: "NWC", 4: "NHWC"}
        # Lookup the layout required by this input shape
        data_layout = rank_to_layout[len(inp_values.shape)]
        # Lookup the index of the channel dimension in the data layout
        # Note: Assumes there is at most one "C" which denotes the channel dimension
        cdim = data_layout.index("C") if data_layout is not None and "C" in data_layout else 1
        # Rearrange the input to the expected (N, C, ...) layout
        inp_values = inp_values.swapaxes(cdim, 1)
        y = multithreshold(inp_values, th_val, out_bias=out_bias)
        # Rearrange the output back to the original layout
        y = y.swapaxes(cdim, 1)

        act = DataType[cast("str", self.get_nodeattr("outputDataType"))]
        if act == DataType["BIPOLAR"]:
            # binary to bipolar
            y = 2 * y - 1
        context[node.output[0]] = y.astype(np.float32)

    def calc_tmem(self) -> int:
        """Calculate and return TMEM."""
        return self.num_channels // self.pe

    def get_verilog_top_module_intf_names(self) -> dict[str, list[tuple[str, int]] | list[str]]:
        """Return the signal names for the Verilog top module."""
        intf_names: dict[str, list[tuple[str, int]] | list[str]] = {}
        intf_names["clk"] = ["ap_clk"]
        intf_names["rst"] = ["ap_rst_n"]
        intf_names["s_axis"] = [("in0_V", self.get_instream_width_padded(0))]
        intf_names["m_axis"] = [("out0_V", self.get_outstream_width_padded(0))]
        intf_names["aximm"] = []
        intf_names["axilite"] = []
        intf_names["ap_none"] = []
        mlo_max_iter = cast("int", self.get_nodeattr("mlo_max_iter"))
        if mlo_max_iter:
            stream_width = DataType.get_smallest_possible(mlo_max_iter).bitwidth()
            stream_width_padded = roundup_to_integer_multiple(stream_width, 8)
            s_axis = cast("list[tuple[str, int]]", intf_names["s_axis"])
            s_axis.append(("in1_V", stream_width_padded))
        else:
            # try to access mem_mode attribute, doesn't exist for RTL Thresholding
            try:
                mem_mode = self.get_nodeattr("mem_mode")
            except AttributeError:
                mem_mode = 0

            if mem_mode == "internal_decoupled":
                # only expose axilite interface if attribute is set
                runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1
                if runtime_writable:
                    intf_names["axilite"] = ["s_axilite"]
        return intf_names

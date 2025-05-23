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

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.util.basic import interleave_matrix_outer_dim_from_partitions

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.logging import log


class Thresholding(HWCustomOp):
    """Abstraction layer for HW implementation of Thresholding."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
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

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype(0):
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype(0).name),
                str(idt.name),
            )
            log.warning(warn_str)
        self.set_nodeattr("inputDataType", idt.name)
        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def verify_node(self):
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

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        if ind == 0:
            dt = DataType[self.get_nodeattr("inputDataType")]
        elif ind == 1:
            dt = DataType[self.get_nodeattr("weightDataType")]
        else:
            raise Exception("Index out of range")
        return dt

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

    def minimize_accumulator_width(self, model):
        "Minimize threshold width ('accumulator width' here due to convention)"
        thresholds = model.get_initializer(self.onnx_node.input[1])
        threshold_tensor = self.get_hw_compatible_threshold_tensor(thresholds)
        min_threshold = thresholds.min()
        max_threshold = thresholds.max()
        min_input = self.get_input_datatype(0).min()
        max_input = self.get_input_datatype(0).max()
        # get range required by threshold values
        tdt_min = min(min_input, min_threshold)
        tdt_max = max(max_input, max_threshold)
        if tdt_min < 0:
            if abs(tdt_min) > tdt_max:
                tdt = DataType.get_smallest_possible(tdt_min)
            else:
                tdt = DataType.get_smallest_possible(-tdt_max - 1)
        else:
            tdt = DataType.get_smallest_possible(tdt_max)
        assert np.vectorize(tdt.allowed)(
            threshold_tensor
        ).all(), "Thresholds can't be expressed with type %s" % str(tdt)
        self.set_nodeattr("weightDataType", tdt.name)
        # Update QONNX DataType of tensor for consistency
        model.set_tensor_datatype(self.onnx_node.input[1], tdt)
        return DataType[self.get_nodeattr("weightDataType")]

    def get_instream_width(self, ind=0):
        if ind == 0:
            i_bits = self.get_input_datatype(0).bitwidth()
            width = i_bits * self.get_nodeattr("PE")
        elif ind == 1:
            # try to access mem_mode attribute, doesn't exist for RTL Thresholding
            try:
                mem_mode = self.get_nodeattr("mem_mode")
            except AttributeError:
                mem_mode = 0
            if mem_mode == "internal_decoupled":
                pe = self.get_nodeattr("PE")
                wp = self.get_input_datatype(1).bitwidth()
                n_thres_steps = self.get_nodeattr("numSteps")
                width = pe * wp * n_thres_steps
            else:
                width = 0
        else:
            raise Exception("Index out of range")
        return width

    def get_outstream_width(self, ind=0):
        o_bits = self.get_output_datatype().bitwidth()
        return o_bits * self.get_nodeattr("PE")

    def get_folded_input_shape(self, ind=0):
        pe = self.get_nodeattr("PE")
        fold = self.calc_tmem()
        vecs = list(self.get_nodeattr("numInputVectors"))
        folded_input_shape = tuple(vecs + [fold, pe])
        return folded_input_shape

    def get_folded_output_shape(self, ind=0):
        # same shape as input
        return self.get_folded_input_shape()

    def get_normal_input_shape(self, ind=0):
        ich = self.get_nodeattr("NumChannels")
        vecs = list(self.get_nodeattr("numInputVectors"))
        normal_input_shape = tuple(vecs + [ich])
        return normal_input_shape

    def get_normal_output_shape(self, ind=0):
        # same shape as input
        return self.get_normal_input_shape()

    def get_number_output_values(self):
        nf = np.prod(self.get_folded_output_shape()[:-1])
        return nf

    def get_exp_cycles(self):
        # Channels/PE * batch size * fmdim * fmdim
        return np.prod(self.get_folded_output_shape()[:-1])

    def get_hw_compatible_threshold_tensor(self, orig_thres_matrix):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure MH % PE == 0
        * for unsigned inputs, ensure thresholds are positive
        * interleave rows between PEs
        * reshape into (PE, TMEM, n_thres_steps) and return
        """
        mh = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        tmem = mh // pe
        assert mh % pe == 0, "Requirement NumChannels divisable by PE is violated."
        assert (
            orig_thres_matrix.ndim == 2
        ), """Threshold matrix dimension is
        not as expected (2)."""
        n_thres_steps = orig_thres_matrix.shape[1]
        assert n_thres_steps == self.get_nodeattr("numSteps"), "Mismatch in threshold steps"
        if not self.get_input_datatype(0).signed():
            # ensure all thresholds are nonnegative
            assert (orig_thres_matrix >= 0).all()
        # ensure all thresholds are integer
        assert np.equal(np.mod(orig_thres_matrix, 1), 0).all(), "Need int threshold tensor"
        ret = orig_thres_matrix
        # ensure channels = mh , duplicating if necessary
        if ret.shape[0] == 1:
            ret = np.tile(ret, (mh, 1))
        assert ret.shape[0] == mh, "Channels of threshold matrix are not as expected (mh)"
        # distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        assert (
            ret.shape[0] == pe
        ), """First dimension after distribution of the
        rows between PEs is not as expected (pe)"""
        assert (
            ret.shape[1] == tmem
        ), """Second dimension after distribution of the
        rows between PEs is not as expected (tmem)"""
        assert (
            ret.shape[2] == n_thres_steps
        ), """Third dimension after distribution of the
        rows between PEs is not as expected (n_thres_steps)"""
        return ret.reshape(1, pe, tmem, n_thres_steps)

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp_values = context[node.input[0]]
        th_val = context[node.input[1]]
        out_bias = self.get_nodeattr("ActVal")

        # Consider the data layout for transposing the input into the format
        # accepted by the multithreshold function above, i.e, the channel
        # dimension is along the axis with index 1.
        data_layout = None
        # If there is no layout annotation, guess based on rank of the tensor
        # TODO: Currently there is no mechanism here to get the layout
        #  annotation, we allways guess, but this matches the previous behavior.
        if len(inp_values.shape) < 5:
            # Maps tensor rank to layout annotation
            rank_to_layout = {0: None, 1: "C", 2: "NC", 3: "NWC", 4: "NHWC"}
            # Lookup the layout required by this input shape
            data_layout = rank_to_layout[len(inp_values.shape)]
        # Lookup the index of the channel dimension in the data layout
        # Note: Assumes there is at most one "C" which denotes the channel
        # dimension
        cdim = data_layout.index("C") if "C" in data_layout else 1
        # Rearrange the input to the expected (N, C, ...) layout
        inp_values = inp_values.swapaxes(cdim, 1)
        y = multithreshold(inp_values, th_val, out_bias=out_bias)
        # Rearrange the output back to the original layout
        y = y.swapaxes(cdim, 1)

        act = DataType[self.get_nodeattr("outputDataType")]
        if act == DataType["BIPOLAR"]:
            # binary to bipolar
            y = 2 * y - 1
        context[node.output[0]] = y.astype(np.float32)

    def calc_tmem(self):
        """Calculates and returns TMEM."""
        num_channels = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        return num_channels // pe

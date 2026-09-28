# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Hardware abstraction layer for selecting one token from a token sequence."""
import numpy as np
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.logging import log


class SelectToken(HWCustomOp):
    """Select one token vector from a sequence of token vectors."""

    def get_nodeattr_types(self):
        """Return node attribute types, adding token, channel,
        index, SIMD and datatype attributes.
        """
        my_attrs = super().get_nodeattr_types()
        my_attrs.update(
            {
                "NumTokens": ("i", True, 0),
                "NumChannels": ("i", True, 0),
                "TokenIndex": ("i", True, 0),
                "SIMD": ("i", False, 1),
                "inputDataType": ("s", True, ""),
                "outputDataType": ("s", False, ""),
            }
        )
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        """Return the unfolded input shape (1, NumTokens, NumChannels)."""
        if ind != 0:
            raise ValueError("SelectToken only has one input")
        return (1, self.get_nodeattr("NumTokens"), self.get_nodeattr("NumChannels"))

    def get_folded_input_shape(self, ind=0):
        """Return the folded input shape."""
        normal_shape = self.get_normal_input_shape(ind)
        simd = self.get_nodeattr("SIMD")
        num_channels = normal_shape[-1]
        assert num_channels % simd == 0, "SIMD must divide NumChannels"
        return normal_shape[:-1] + (num_channels // simd, simd)

    def get_normal_output_shape(self, ind=0):
        """Return the unfolded output shape (1, NumChannels)."""
        return (1, self.get_nodeattr("NumChannels"))

    def get_folded_output_shape(self, ind=0):
        """Return the folded output shape."""
        normal_shape = self.get_normal_output_shape(ind)
        simd = self.get_nodeattr("SIMD")
        num_channels = normal_shape[-1]
        assert num_channels % simd == 0, "SIMD must divide NumChannels"
        return normal_shape[:-1] + (num_channels // simd, simd)

    def make_shape_compatible_op(self, model):
        """Check the input shape and return a constant-shape op for the output."""
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == self.get_normal_input_shape(), "Unexpected input shape for token sequence"
        return super().make_const_shape_op(self.get_normal_output_shape())

    def infer_node_datatype(self, model):
        """Infer and set the input and output datatypes from the model tensors."""
        node = self.onnx_node
        attr_idt = None
        if self.get_nodeattr("inputDataType") != "":
            attr_idt = self.get_input_datatype()
        idt = model.get_tensor_datatype(node.input[0]) or attr_idt
        if idt is None:
            raise ValueError("SelectToken input datatype is not set")
        if attr_idt is not None and attr_idt != idt:
            log.warning("inputDataType changing for %s: %s -> %s" % (node.name, attr_idt, idt))
        self.set_nodeattr("inputDataType", idt.name)
        self.set_nodeattr("outputDataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def get_input_datatype(self, ind=0):
        """Return the FINN DataType of the input."""
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self, ind=0):
        """Return the FINN DataType of the output (defaults to the input datatype)."""
        odt = self.get_nodeattr("outputDataType")
        return self.get_input_datatype(ind) if odt == "" else DataType[odt]

    def get_instream_width(self, ind=0):
        """Return the width of the input stream in bits."""
        return self.get_input_datatype().bitwidth() * self.get_nodeattr("SIMD")

    def get_outstream_width(self, ind=0):
        """Return the width of the output stream in bits."""
        return self.get_output_datatype().bitwidth() * self.get_nodeattr("SIMD")

    def get_exp_cycles(self):
        """Return the expected cycle count (one per input stream word)."""
        return int(np.prod(self.get_folded_input_shape()[:-1]))

    def execute_node(self, context, graph):
        """Execute the node in Python by slicing the selected token from the sequence."""
        node = self.onnx_node
        token_index = self.get_nodeattr("TokenIndex")
        num_tokens = self.get_nodeattr("NumTokens")
        if token_index < 0:
            token_index += num_tokens
        assert 0 <= token_index < num_tokens, "TokenIndex must select an existing token"
        result = context[node.input[0]][:, token_index, :]
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(
            self.get_normal_output_shape()
        )

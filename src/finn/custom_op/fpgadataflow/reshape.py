"""Hardware operator corresponding to the standard ONNX Reshape."""

# Numpy math and arrays
import numpy as np

# QONNX/FINN datatypes
from qonnx.core.datatype import DataType

# QONNX wrapper to ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# Utility for registering HWCustomOp implementations into the module scope
from finn.custom_op.fpgadataflow import register_custom_op

# Derive custom operators form the FINN base custom op
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

# FINN logging
from finn.util.logging import log


@register_custom_op
class Reshape(HWCustomOp):
    """Reshape operator, essentially passthrough with different input/output shape."""

    def get_nodeattr_types(self):
        """Custom node attributes with their types and default values."""
        # Start from parent operator class attributes  # noqa: Duplicate
        attrs = HWCustomOp.get_nodeattr_types(self)
        # Update attributes dictionary for new custom operator
        attrs.update({
            # Shape of the input
            "inp_shape": ("ints", True, [1]),
            # Shape of the output
            "out_shape": ("ints", True, [1]),
            # Datatype of input and output elements
            "dtype": ("s", True, ""),
            # Number of parallel elements in the last dimension of the output
            "PE": ("i", False, 1),
        })
        # Return updated attribute dictionary
        return attrs

    @property
    def inp_shape(self):
        """Input shape attribute."""
        return self.get_nodeattr("inp_shape")

    @property
    def out_shape(self):
        """Output shape attribute."""
        return self.get_nodeattr("out_shape")

    @property
    def dtype(self):
        """Datatype attribute as QONNX DataType."""
        # Note: Converts from string to QONNX data type
        return DataType[self.get_nodeattr("dtype")]

    @property
    def pe(self):
        """Parallel elements in the last dimension of the output."""
        return self.get_nodeattr("PE")

    def get_input_datatype(self, ind=0):
        """Datatype of the input tensor, same as the output."""
        return self.dtype

    def get_output_datatype(self, ind=0):
        """Datatype of the output tensor, same as the input."""
        return self.dtype

    def get_normal_input_shape(self, ind=0):
        """Regular input shape as seen by the ONNX standard."""
        return self.inp_shape

    def get_normal_output_shape(self, ind=0):
        """Regular output shape as seen by the ONNX standard."""
        return self.out_shape

    def get_folded_input_shape(self, ind=0):
        """Shape of the folded (PE) input tensor"""
        *num_inputs, num_elems = self.get_normal_input_shape(ind=ind)
        # Valid folding requires the PE to divide the number of elements
        assert num_elems % self.pe == 0, "PE must divide last axis"
        # Folding along the last dimension
        return *num_inputs, num_elems // self.pe, self.pe

    def get_folded_output_shape(self, ind=0):
        """Shape of the folded (PE) output tensor"""
        *num_outputs, num_elems = self.get_normal_output_shape(ind=ind)
        # Valid folding requires the PE to divide the number of elements
        assert num_elems % self.pe == 0, "PE must divide last axis"
        # Folding along the last dimension
        return *num_outputs, num_elems // self.pe, self.pe

    def get_instream_width(self, ind=0):
        """Widths of the input data stream of the input at index ind"""
        # Get the number of bits used to represent the input
        i_bits = self.get_input_datatype(ind).bitwidth()
        # Parallelism is the number of elements in the last dimension of the
        # folded input
        *_, elems = self.get_folded_input_shape(ind)
        # Width of a stream receiving input elements in parallel
        return elems * i_bits

    def get_outstream_width(self, ind=0):
        """Widths of the output data stream of the output at index ind"""
        # Get the number of bits used to represent the output
        o_bits = self.get_output_datatype(ind).bitwidth()
        # Parallelism is the number of elements in the last dimension of the
        # folded output
        *_, elems = self.get_folded_output_shape(ind)
        # Width of a stream producing output elements in parallel
        return elems * o_bits

    def get_number_output_values(self):
        """Expected output values for the operation given the folding."""
        return np.prod(self.get_folded_output_shape()[:-1])

    def get_exp_cycles(self):
        """Expected cycles for the operation given the folding."""
        return np.prod(self.get_folded_output_shape()[:-1])

    def infer_node_datatype(self, model: ModelWrapper):
        """Infers the datatype of the node output from the model graph."""
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Test for changing input datatype
        if model.get_tensor_datatype(node.input[0]) != self.dtype:
            # Get the new datatype
            new_dtype = model.get_tensor_datatype(node.input[0])
            # Issue a warning message
            log.warning(
                f"{node.name}: inp_dtype changing from"
                f" {self.dtype} to {new_dtype}"
            )
            # Set the new datatype attribute
            self.set_nodeattr("inp_dtype", new_dtype.name)
        # Force the output data type stored as a node attribute
        model.set_tensor_datatype(node.output[0], self.dtype)

    def execute_node(self, context, graph):
        """Execute reshape operation (Python fallback)."""
        # Get the node wrapped by this custom op
        node = self.onnx_node  # noqa: Duplicate
        # Get the input from the execution context
        inp = context[node.input[0]]
        # Squeeze the input along the optionally specified axes
        out = np.reshape(inp, newshape=self.out_shape)
        # Make sure the output has the right type (always use float32 as the
        # container type) and insert into the execution context
        context[node.output[0]] = out.astype(np.float32)

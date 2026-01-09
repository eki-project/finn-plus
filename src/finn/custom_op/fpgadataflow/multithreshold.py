"""Hardware operator corresponding to the standard MultiThreshold operator."""

# FINN hardware custom operator base and registry
from finn.custom_op.fpgadataflow import register_custom_op
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

# FINN logging
from finn.util.logging import log

# QONNX arbitrary precision datatypes
from qonnx.core.datatype import DataType
# QONNX wrapper to ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# Numpy math and arrays, shape calculations
import numpy as np


# Numpy reference implementation of multi-threshold functions
def multithreshold(x, thresholds, weights):  # noqa: Shadows outer scope
    return np.sum(weights * (x.reshape(*x.shape, 1) >= thresholds), axis=-1)


@register_custom_op
class MultiThreshold(HWCustomOp):
    """MultiThreshold custom operator.

    In contrast to the Thresholding operator and the QONNX MultiThreshold, this
    variant offers arbitrary granularity and non-monotonicity.
    """

    def get_nodeattr_types(self):
        """Custom node attributes with their types and default values."""
        # Start from parent operator class attributes  # noqa: Duplicate
        attrs = HWCustomOp.get_nodeattr_types(self)
        # Update attributes dictionary for new custom operator
        attrs.update({
            # Shape of the threshold tensor (excluding the number of thresholds)
            "threshold_shape": ("ints", True, [1]),
            # QONNX data type of the threshold values
            "threshold_type": ("s", True, ""),
            # Number of thresholds per element
            "N": ("i", True, 1),

            # Shape of the weights tensor (excluding the number of thresholds)
            "weights_shape": ("ints", True, [1]),
            # QONNX data type of the weights values
            "weights_type": ("s", True, ""),

            # Shape of the input tensor (also the output shape as this operator
            # only implements unidirectional broadcasting from the thresholds to
            # the input shape)
            "input_shape": ("ints", True, [1]),
            # QONNX data type of the input values
            "input_type": ("s", True, ""),

            # QONNX data type of the output values
            "output_type": ("s", True, ""),

            # Optional global output bias parameter
            "out_bias": ("f", False, 0.0),

            # Number of elements processed in parallel, folding the innermost
            # dimension
            "PE": ("i", False, 1),
        })
        # Return updated attribute dictionary
        return attrs

    @property
    def pe(self):
        """Parallel elements in the last dimension of the output."""
        return self.get_nodeattr("PE")

    def get_input_datatype(self, ind=0):
        """Datatype of the tensor at input index ind."""
        return [
            DataType[self.get_nodeattr("input_type")],
            DataType[self.get_nodeattr("threshold_type")],
            DataType[self.get_nodeattr("weights_type")],
        ][ind]

    def get_output_datatype(self, ind=0):
        """Datatype of the output tensor."""
        return DataType[self.get_nodeattr("output_type")]

    def get_normal_input_shape(self, ind=0):
        """Regular input shape as seen by the ONNX standard."""
        return [
            self.get_nodeattr("input_shape"),
            (*self.get_nodeattr("threshold_shape"), self.get_nodeattr("N")),
            (*self.get_nodeattr("weights_shape"), self.get_nodeattr("N")),
        ][ind]

    def get_normal_output_shape(self, ind=0):
        """Regular output shape as seen by the ONNX standard."""
        return self.get_nodeattr("input_shape")

    def get_folded_input_shape(self, ind=0):
        """Shape of the folded (PE) input tensor"""
        # Decompose leading dimensions and innermost elements
        *num_inputs, num_elems = self.get_normal_input_shape(ind=ind)
        # No folding of threshold and weights supported
        if ind >= 1:
            return *num_inputs, num_elems, 1
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
        if (model.get_tensor_datatype(node.input[0])
                != self.get_input_datatype(0)):
            # Get the new datatype
            new_dtype = model.get_tensor_datatype(node.input[0])
            # Issue a warning message
            log.warning(
                f"{node.name}: input_type changing from"
                f" {self.get_input_datatype(0)} to {new_dtype}"
            )
            # Set the new datatype attribute
            self.set_nodeattr("input_type", new_dtype.name)

        # Test for changing threshold datatype
        if (model.get_tensor_datatype(node.input[1])
                != self.get_input_datatype(1)):
            # Get the new datatype
            new_dtype = model.get_tensor_datatype(node.input[1])
            # Issue a warning message
            log.warning(
                f"{node.name}: threshold_type changing from"
                f" {self.get_input_datatype(1)} to {new_dtype}"
            )
            # Set the new datatype attribute
            self.set_nodeattr("threshold_type", new_dtype.name)

        # Test for changing weights datatype
        if (model.get_tensor_datatype(node.input[2])
                != self.get_input_datatype(2)):
            # Get the new datatype
            new_dtype = model.get_tensor_datatype(node.input[2])
            # Issue a warning message
            log.warning(
                f"{node.name}: weights_type changing from"
                f" {self.get_input_datatype(2)} to {new_dtype}"
            )
            # Set the new datatype attribute
            self.set_nodeattr("weights_type", new_dtype.name)

        # Force the output data type stored as a node attribute
        model.set_tensor_datatype(node.output[0], self.get_output_datatype(0))

    def execute_node(self, context, graph):
        """Execute multithreshold operation (Python fallback)."""
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Get the input, threshold and weights from the execution context
        x = context[node.input[0]]
        thresholds = context[node.input[1]]
        weights = context[node.input[2]]
        # Get the optional output bias
        bias = self.get_nodeattr("out_bias")
        # Apply the multithreshold operator to the inputs from the execution
        # context using the NumPy reference implementation
        out = multithreshold(x, thresholds, weights) + bias
        # Make sure the output has the right type (always use float32 as the
        # container type) and insert into the execution context
        context[node.output[0]] = out.astype(np.float32)

    def minimize_accumulator_width(self, model: ModelWrapper):
        """Minimize the accumulator bit width according to the weight values."""

        # Minimization is only implemented for integer types...
        if not self.get_input_datatype(2).is_integer():
            return

        # Get the parameter tensors from the model wrapper
        weights = model.get_initializer(self.onnx_node.input[2])

        # Number of thresholds
        N =  self.get_nodeattr("N")

        # Broadcasting of weights along the threshold axis must be made explicit
        if len(weights.shape) < 1 or weights.shape[-1] != N:
            weights = np.broadcast_to(weights, (*weights.shape[:-1], N))

        # Get the optional output bias and expand to the weights shape
        bias = np.full((*weights.shape[:-1], 1), self.get_nodeattr("out_bias"))

        # Multi-threshold hardware uses cumulative weights left-padded by the
        # optional bias as output values at/between steps
        weights = np.concatenate((bias, weights), -1)
        values = np.cumsum(weights, axis=-1)

        # Find the minimum and maximum value produced at the output and,
        # depending on the sign and magnitude select the smallest possible data
        # type to represent these values
        if np.min(values) < 0:
            if abs(np.min(values)) > np.max(values):
                dtype = DataType.get_smallest_possible(np.min(values))
            else:
                dtype = DataType.get_smallest_possible(-np.max(values) - 1)
        else:
            dtype = DataType.get_smallest_possible(np.max(values))

        # Update the node attribute and the output tensor type annotation
        self.set_nodeattr("output_type", dtype.name)
        model.set_tensor_datatype(self.onnx_node.output[0], dtype)

    def minimize_weight_bit_width(self, model: ModelWrapper):
        """Minimize the bit width based on the values of the thresholds"""

        # Minimization is only implemented for integer types...
        if not self.get_input_datatype(ind=1).is_integer():
            return

        # Get the parameter tensors from the model wrapper
        thresholds = model.get_initializer(self.onnx_node.input[1])

        # Find the minimum and maximum threshold parameter and, depending on the
        # sign and magnitude select the smallest possible data type to represent
        # these values
        if np.min(thresholds) < 0:
            if abs(np.min(thresholds)) > np.max(thresholds):
                dtype = DataType.get_smallest_possible(np.min(thresholds))
            else:
                dtype = DataType.get_smallest_possible(-np.max(thresholds) - 1)
        else:
            dtype = DataType.get_smallest_possible(np.max(thresholds))

        # Update the node attribute and the threshold tensor type annotation
        self.set_nodeattr("threshold_type", dtype.name)
        model.set_tensor_datatype(self.onnx_node.input[1], dtype)

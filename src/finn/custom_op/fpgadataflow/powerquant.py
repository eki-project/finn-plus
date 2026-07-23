"""Hardware operator for PowerQuant matrix multiplication."""
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


@register_custom_op
class PowerQuantMatMul(HWCustomOp):
    """PowerQuantMatMul custom operator."""

    def get_nodeattr_types(self):
        """Custom node attributes with their types and default values."""

        # Start from parent operator class attributes and update with custom-op
        # specific attributes
        attrs = HWCustomOp.get_nodeattr_types(self)

        attrs.update({
            # Shape and QONNX type of the static weight tensor
            "weights_shape": ("ints", True, [1]),
            "weights_type": ("s", True, ""),

            # Shape and QONNX type of the dynamic input tensor
            "input_shape": ("ints", True, [1]),
            "input_type": ("s", True, ""),

            # QONNX type of the output tensor (shape is derived from the inputs)
            "output_type": ("s", True, ""),

            # The power of the PowerQuant weight/input transformation
            "alpha": ("f", True, 1.0),

            # Number of fixed-point fractional bits used internally to represent
            # the power and the accumulator
            "fractional": ("i", False, 23),

            # Folding: Number of elements processed in parallel on the input and
            # output dimension
            "SIMD": ("i", False, 1),
            "PE": ("i", False, 1)
        })

        return attrs

    @property
    def simd(self):
        """Parallel elements in the last dimension of the input."""
        return self.get_nodeattr("SIMD")

    @property
    def pe(self):
        """Parallel elements in the last dimension of the output."""
        return self.get_nodeattr("PE")

    def get_input_datatype(self, ind=0):
        """Datatype of the tensor at input index ind."""
        return [
            DataType[self.get_nodeattr("input_type")],
            DataType[self.get_nodeattr("weights_type")],
        ][ind]

    def get_output_datatype(self, ind=0):
        """Datatype of the output tensor."""
        return DataType[self.get_nodeattr("output_type")]

    def get_normal_input_shape(self, ind=0):
        """Regular input shape as seen by the ONNX standard."""
        return [
            self.get_nodeattr("input_shape"),
            self.get_nodeattr("weights_shape"),
        ][ind]

    def get_normal_output_shape(self, ind=0):
        """Regular output shape as seen by the ONNX standard."""
        *nx, l, k = self.get_nodeattr("input_shape")
        *nw, k, m = self.get_nodeattr("weights_shape")

        # Leading (batch matrix-matrix multiplication) dimensions must match
        assert nx == nw, "Incompatible shapes"

        # Assemble the output shape of batched x @ weights
        return *nx, l, m

    def get_folded_input_shape(self, ind=0):
        """Shape of the folded (PE) input tensor"""
        # Decompose leading dimensions and innermost elements
        *num_inputs, num_elems = self.get_normal_input_shape(ind=ind)
        # No folding of weights supported (this is handled internally)
        if ind >= 1:
            return *num_inputs, num_elems, 1
        # Valid folding requires the SIMD to divide the number of elements
        assert num_elems % self.simd == 0, "SIMD must divide last axis"
        # Folding along the last dimension
        return *num_inputs, num_elems // self.simd, self.simd

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
        node = self.onnx_node  # noqa: Duplicate...

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

        # Test for changing weights datatype
        if (model.get_tensor_datatype(node.input[1])
                != self.get_input_datatype(1)):
            # Get the new datatype
            new_dtype = model.get_tensor_datatype(node.input[1])
            # Issue a warning message
            log.warning(
                f"{node.name}: weights_type changing from"
                f" {self.get_input_datatype(1)} to {new_dtype}"
            )
            # Set the new datatype attribute
            self.set_nodeattr("weights_type", new_dtype.name)

        # Force the output data type stored as a node attribute
        model.set_tensor_datatype(node.output[0], self.get_output_datatype(0))

    def execute_node(self, context, graph):
        """Execute PowerQuantMatMul operation (Python fallback)."""
        # Get the node wrapped by this custom op
        node = self.onnx_node

        # Get the input, threshold and weights from the execution context
        x = context[node.input[0]]
        y = context[node.input[1]]

        # Node attributes controlling the power and the internal fixed-point
        # representation
        alpha = self.get_nodeattr("alpha")
        fractional = self.get_nodeattr("fractional")

        # Apply the powerquant matmul operator to the inputs from the execution
        # context using the NumPy reference implementation
        out = np.matmul(
            np.round(2 ** fractional * np.sign(x) * np.abs(x) ** alpha),
            np.round(2 ** fractional * np.sign(y) * np.abs(y) ** alpha)
        )

        # Get rid of the extra fractional bits not implemented in the actual
        # implementation
        out = out.astype(int) >> fractional

        # Make sure the output has the right type (always use float32 as the
        # container type) and insert into the execution context
        context[node.output[0]] = out.astype(np.float32)

    def minimize_accumulator_width(self, model: ModelWrapper):
        """Minimize the accumulator and output bitwidth."""

        # Minimization is only implemented for integer types...
        if not (input_type := self.get_input_datatype(ind=0)).is_integer():
            return

        # Minimization is only implemented for integer types...
        if not (weight_type := self.get_input_datatype(ind=1)).is_integer():
            return

        # Get the range of input and weight values according to their datatypes
        min_x = int(input_type.min())
        max_x = int(input_type.max())

        min_w = int(weight_type.min())
        max_w = int(weight_type.max())

        # Maximum overall input expected, used to properly size the table of
        # precomputed powers
        maximum: int = max(abs(min_x), abs(max_x), abs(min_w), abs(max_w))

        # Node attributes controlling the power and the internal fixed-point
        # representation
        alpha: float = self.get_nodeattr("alpha")
        fractional: int = self.get_nodeattr("fractional")

        # Number of integer bits required to represent the power of the largest
        # input expected
        integer: int = int(np.ceil(np.log2(maximum ** alpha)))

        # Innermost dimension to accumulate over, i.e., the accumulator must be
        # large enough to hold k adds of integer^2.fractional muls.
        *_, k = self.get_nodeattr("input_shape")

        # Required accumulator size, mirrors size derivation in powerquant.hpp,
        # without specifying the fractional part here as FINN/streamlining
        # treats this as and integer with implicit scale.
        bits = int(np.ceil(np.log2(k)) + 2 * integer + fractional)
        dtype = DataType[f"INT{bits}"]

        # Update the node attribute and the output tensor type annotation
        self.set_nodeattr("output_type", dtype.name)
        model.set_tensor_datatype(self.onnx_node.output[0], dtype)

    def minimize_weight_bit_width(self, model: ModelWrapper):
        """Minimize the weight bitwidth based on the values of the weights"""

        # Minimization is only implemented for integer types...
        if not self.get_input_datatype(ind=1).is_integer():
            return

        # Get the parameter tensors from the model wrapper and ensure integer
        # values are actual integers to avoid float artifacts for large values
        weights = model.get_initializer(self.onnx_node.input[1])
        weights = weights.astype(np.int64)

        # Find the minimum and maximum weight parameter and select the smallest
        # possible data type to represent these values
        if np.min(weights) < 0:  # noqa: Duplicate...
            if abs(np.min(weights)) > np.max(weights):
                dtype = DataType.get_smallest_possible(np.min(weights))
            else:
                dtype = DataType.get_smallest_possible(-np.max(weights) - 1)
        else:
            dtype = DataType.get_smallest_possible(np.max(weights))

        # Update the node attribute and the weight tensor type annotation
        self.set_nodeattr("weights_type", dtype.name)
        model.set_tensor_datatype(self.onnx_node.input[1], dtype)

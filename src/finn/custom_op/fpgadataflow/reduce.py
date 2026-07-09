# Numpy math and arrays
"""Module for to implement arbitrary reduce operations."""

import numpy as np

# Helper for creating ONNX nodes
from onnx import GraphProto, NodeProto, TensorProto
from onnx import helper as oh

# QONNX/FINN datatypes
from qonnx.core.datatype import BaseDataType, DataType

# QONNX wrapper to ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
from typing import cast

from finn.custom_op.fpgadataflow import register_custom_op

# Derive custom operators form the FINN base custom op
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.exception import FINNInternalError, FINNUserError

# FINN logging
from finn.util.logging import log


@register_custom_op
class Reduce(HWCustomOp):
    """Class for Reduce."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        # Just forward all arguments to the init method of the CustomOp base
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(
        self,
    ) -> dict[
        str,
        tuple[str, bool, int | float | str | bool | np.ndarray | list]
        | tuple[str, bool, int | float | str | bool | np.ndarray | list, set | None],
    ]:
        """Define attributes which must be present on this node.
        Start from parent operator class attributes."""
        attrs = HWCustomOp.get_nodeattr_types(self)
        # Update attributes dictionary for new custom operator
        attrs.update(
            {
                # Number of reductions processed in parallel
                "PE": ("i", True, 1),
                # Index of first axis to reduce over, 0-based
                "index_start_axis": ("i", True, 0),
                # Index of last axis to reduce over, 0-based
                # By default, the Channel dimension is excluded but this can be changed
                "index_stop_axis": ("i", False, -2),
                # Input shape
                "input_shape": ("ints", True, []),
                # Output shape
                "output_shape": ("ints", False, []),
                # FINN DataTypes for inputs/outputs
                "InputDataType": ("s", True, ""),
                "OutputDataType": ("s", True, ""),
                # Reduction operation to perform, one of "sum", "min", "max", "product"
                "op": ("s", True, "", {"sum", "min", "max", "product"}),
                "keepdims": ("i", False, 0),
                "depthwise": ("i", False, 0),
            }
        )
        # Return updated attribute dictionary
        return attrs

    @property
    def start_index(self) -> int:
        """Return the start index of the axis to reduce over."""
        return cast("int", self.get_nodeattr("index_start_axis"))

    @property
    def stop_index(self) -> int:
        """Return the stop index of the axis to reduce over."""
        # Convert negative index to positive index
        stop_index = cast("int", self.get_nodeattr("index_stop_axis"))
        if stop_index < 0:
            stop_index += len(self.input_shape)
        return stop_index

    @property
    def idtype(self) -> BaseDataType:
        """Return input dtype of the node.
        Note: Converts from string to QONNX data type."""
        return DataType[cast("str", self.get_nodeattr("InputDataType"))]

    @property
    def odtype(self) -> BaseDataType:
        """Return output dtype of the node.
        Note: Converts from string to QONNX data type."""
        return DataType[cast("str", self.get_nodeattr("OutputDataType"))]

    @property
    def pe(self) -> int:
        """Return the number of parallel reductions."""
        return cast("int", self.get_nodeattr("PE"))

    @property
    def op(self) -> str:
        """Return the reduction operation."""
        return cast("str", self.get_nodeattr("op"))

    @property
    def input_shape(self) -> list[int]:
        """Return the input shape of the node."""
        return cast("list[int]", self.get_nodeattr("input_shape"))

    @property
    def output_shape(self) -> list[int]:
        """Return the output shape of the node."""
        oshape = cast("list[int]", self.get_nodeattr("output_shape"))
        if len(oshape) == 0:
            oshape_proto: list[int | None] = cast("list[int|None]", self.input_shape)
            for i in range(self.start_index, self.stop_index + 1):
                oshape_proto[i] = 1 if self.keepdims else None
            oshape = [d for d in oshape_proto if d is not None]
        return oshape

    @property
    def keepdims(self) -> bool:
        """Return whether to keep the reduced dimensions."""
        return bool(cast("int", self.get_nodeattr("keepdims")))

    @property
    def depthwise(self) -> bool:
        """Return whether to perform depthwise reduction."""
        return bool(cast("int", self.get_nodeattr("depthwise")))

    # Makes an operation compatible with the output shape for shape inference
    #   Note: Propagates shape forward, i.e., never asks for the shape of the
    #   output, even if it seems easier.
    def make_shape_compatible_op(self, model: ModelWrapper) -> NodeProto:
        """Return an ONNX op used for shape inference with validated shapes."""
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # There must be exactly one input to the reduction operation
        if len(node.input) != 1:
            raise FINNInternalError(f"Reduction operation {node.name} requires exactly one input")
        # Validate input shape matches what is stored as attribute
        if model.get_tensor_shape(node.input[0]) != self.input_shape:
            raise FINNInternalError(
                f"Input shape mismatch: {node.name} {node.input[0]}. "
                f"Expected {self.input_shape}, got {model.get_tensor_shape(node.input[0])}"
            )

        axis_list = list(range(self.start_index, self.stop_index + 1))
        axis_arr = np.array(axis_list, dtype=np.int64)
        axis_val_info = model.make_new_valueinfo_name()
        init_tensor = oh.make_tensor(axis_val_info, TensorProto.INT64, axis_arr.shape, axis_arr)
        val_info = oh.make_tensor_value_info(axis_val_info, TensorProto.INT64, axis_arr.shape)
        model.graph.value_info.append(val_info)
        model.graph.initializer.append(init_tensor)
        # Simulate behavior via the standard ONNX add operation
        inputs = [node.input[0], axis_val_info]

        return oh.make_node(
            "ReduceSum",
            inputs,
            node.output,
            keepdims=int(self.keepdims),
        )

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infers the datatype of the node output and updates the input datatypes of the node."""
        node = self.onnx_node
        # Test for changing input datatype
        if model.get_tensor_datatype(node.input[0]) != self.idtype:
            # Get the new datatype
            new_dtype = model.get_tensor_datatype(node.input[0])
            # Issue a info message
            log.info(f"{node.name}: input dtype changing from {self.idtype} to {new_dtype}")
            # Set the new datatype attribute
            self.set_nodeattr("InputDataType", new_dtype.name)
        # set output datatype from property
        model.set_tensor_datatype(node.output[0], self.odtype)

    def execute_node(
        self,
        context: dict[str, np.ndarray],
        graph: GraphProto,  # noqa: ARG002
    ) -> None:
        """Execute the node with inputs from context writing outputs to context."""
        # simulate behavior with Python functionality
        node = self.onnx_node
        inp_values = context[node.input[0]]
        if inp_values.shape != self.get_folded_input_shape():
            raise FINNUserError(
                f"Input shape mismatch: {node.name} {node.input[0]}. "
                f"Expected {self.get_folded_input_shape()}, got {inp_values.shape}"
            )
        # compute expected output values
        out_values: np.ndarray = np.asarray([])
        ax = tuple(range(self.start_index + 1, inp_values.ndim))
        if self.op == "sum":
            out_values = np.sum(
                inp_values,
                axis=ax,
                keepdims=self.keepdims,
            )
        elif self.op == "min":
            out_values = np.min(
                inp_values,
                axis=ax,
                keepdims=self.keepdims,
            )
        elif self.op == "max":
            out_values = np.max(
                inp_values,
                axis=ax,
                keepdims=self.keepdims,
            )
        elif self.op == "product":
            out_values = np.prod(
                inp_values,
                axis=ax,
                keepdims=self.keepdims,
            )
        # write output values to context
        context[node.output[0]] = out_values

    # Note: End of QONNX CustomOp region, below is FINN HWCustomOp stuff

    # Gets the datatype of input at index ind
    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        # All inputs (there should only be one) have the same type
        """Return input datatype."""
        return self.idtype

    # Gets the datatype of the output at index ind
    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        # All outputs will hae the same type, which is the same as the input
        """Return output datatype."""
        return self.odtype

    # Gets the shape of the input at index ind without folding
    def get_normal_input_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return normal input shape."""
        return self.input_shape

    # Gets the shape of the output at index ind without folding
    def get_normal_output_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return normal output shape."""
        return self.output_shape

    # Gets the shape of the input at index ind with folding
    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        # Valid folding requires the PE to divides the number of elements
        """Return folded input shape."""
        folding_axis = self.input_shape[-1]
        if folding_axis % self.pe != 0:
            raise FINNUserError(
                f"PE {self.pe} must divide input shape "
                f"{self.input_shape} at axis {len(self.output_shape) - 1}"
            )
        # Folding along the last dimension
        return (
            *self.input_shape[:-1],
            folding_axis // self.pe,
            self.pe,
        )

    # Gets the shape of the output at index ind with folding
    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        # Valid folding requires the PE to divides the number of elements
        """Return folded output shape."""
        if self.start_index == 0:
            raise FINNUserError(
                f"Folding is not supported when reducing over all "
                f"axes for node {self.onnx_node.name}."
            )
        if self.start_index > len(self.output_shape):
            raise FINNUserError(
                f"Start index {self.start_index} is larger than the number of output "
                f"dimensions {len(self.output_shape)} for node {self.onnx_node.name}."
            )
        # A depthwise reduction always leaves a single output channel,
        # regardless of PE, since all PE lanes get folded together.
        pe = 1 if self.depthwise else self.pe
        folding_axis = self.output_shape[-1]
        if folding_axis % pe != 0:
            raise FINNUserError(
                f"PE {pe} must divide output shape "
                f"{self.output_shape} at axis {len(self.output_shape) - 1}"
            )
        # Folding along the last dimension
        return (
            *self.output_shape[:-1],
            folding_axis // pe,
            pe,
        )

    # Widths of the input data stream of the input at index ind
    def get_instream_width(self, ind: int = 0) -> int:
        # Get the number of bits used to represent the input
        """Return instream width."""
        i_bits = self.get_input_datatype(ind).bitwidth()
        shape = self.get_folded_input_shape(ind)
        pe = shape[-1]
        # Width of a stream receiving input elements in parallel
        return pe * i_bits

    # Widths of the output data stream of the output at index ind
    def get_outstream_width(self, ind: int = 0) -> int:
        # Get the number of bits used to represent the output
        """Return outstream width."""
        o_bits = self.get_output_datatype(ind).bitwidth()
        shape = self.get_folded_output_shape(ind)
        pe = shape[-1]
        # Width of a stream producing output elements in parallel
        return pe * o_bits

    # Gets the number of expected output values, i.e. how many times read()
    # could/should be called on any output stream of this operator
    def get_number_output_values(self) -> int:
        # Elements over all but the last dimension of the output folded along
        # the embedding dimension.
        """Return number output values."""
        num_outputs_per_stream = np.prod(self.get_folded_output_shape()[:-1])
        return int(num_outputs_per_stream)

    # Derives the expected cycles for the stream replication operation given the
    # folding configuration
    def get_exp_cycles(self) -> int:
        """Return estimation of expected cycles for set folding."""
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    def _derive_out_dtype(self) -> BaseDataType:
        """Derive the output datatype based on the input datatype and the operation for integers."""
        # Use the unfolded shape: folding only splits the channel axis into
        # (channel // PE, PE) but does not change the total element count
        # over the reduced axes, and get_folded_input_shape() would
        # otherwise undercount depthwise reductions with PE > 1.
        reductions = np.prod(self.input_shape[self.start_index : self.stop_index + 1])
        if self.depthwise:
            reductions = np.prod(self.input_shape[self.start_index :])
        dtype = "INT" if self.idtype.signed() else "UINT"
        if self.op == "min" or self.op == "max":
            return self.idtype
        if self.op == "sum":
            return DataType[dtype + str(self.idtype.bitwidth() + int(np.ceil(np.log2(reductions))))]
        if self.op == "product":
            return DataType[dtype + str(self.idtype.bitwidth() * reductions)]
        raise FINNUserError(
            f"Unsupported reduction operation {self.op} for node {self.onnx_node.name}"
        )

    def minimize_accumulator_width(self, model: ModelWrapper) -> BaseDataType:
        """Minimize output bit-width when possible."""
        # If any of the inputs is not an integer, the bit-width cannot be
        # minimized
        if not self.idtype.is_integer():
            # Check the annotated tensor data type corresponds to the stored
            # attribute
            if model.get_tensor_datatype(self.onnx_node.output[0]) != self.odtype:
                raise FINNInternalError(f"Output type mismatch for {self.onnx_node.name}")
            # Exit here, returning the not-minimized data type
            return self.odtype
        # Call the output type derivation specialized by the concrete operator
        # implementation
        out_dtype = self._derive_out_dtype()
        # Set the new output data type as attribute
        self.set_nodeattr("OutputDataType", out_dtype.name)
        # Annotate the output tensor with the new data type
        model.set_tensor_datatype(self.onnx_node.output[0], out_dtype)
        # Return the minimized output data type
        # Note: Probably not required by MinimizeAccumulatorWidth transformation
        return out_dtype

    def get_pe_in(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return PE for the input stream of this node."""
        return self.pe

    def get_pe_out(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return PE for the output stream of this node.

        A depthwise reduction folds all PE lanes into a single output
        channel, so its output stream PE is always 1.
        """
        return 1 if self.depthwise else self.pe

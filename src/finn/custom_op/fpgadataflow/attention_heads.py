"""Multi-head attention split and merge operators."""

# fmt: off
# Disable formatter. This is deliberately formatted to stay within 80 characters
# per line. Black, however, formats some lines going beyond this.

# Numpy math and arrays
import numpy as np

# Operating system stuff, e.g. paths
import os

# Helper for creating ONNX nodes
from onnx import helper as oh

# QONNX/FINN datatypes
from qonnx.core.datatype import DataType

# QONNX wrapper to ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# Derive custom operators form the FINN base custom op
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

# Converts inputs/outputs to/from RTL simulation format
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

# FINN logging
from finn.util.logging import log


class SplitMultiHeads(HWCustomOp):
    """Split input tensor into multiple attention heads.

    This operator splits the input tensor after input projections to create
    separate attention heads for multi-head attention mechanisms. The output
    can be either packed as a single tensor or split into multiple output tensors.
    """

    def __init__(self, onnx_node, **kwargs):
        """Initialize the SplitMultiHeads operator."""
        # Just forward all arguments to the init method of the CustomOp base
        super().__init__(onnx_node, **kwargs)

        # Need to override the default depths of outputs FIFOs here as these
        # depend on the number of heads, which are not known during calls to
        # get_nodeattr_types.
        if not self.get_nodeattr("outFIFODepths"):
            self.set_nodeattr("outFIFODepths", [2 for _ in range(self.heads)])

    def get_nodeattr_types(self):
        """Get node attribute types for the SplitMultiHeads operator.

        Defines the attributes that must be present on this node, including
        the number of attention heads, packing mode, data type, and other
        configuration parameters inherited from the parent HWCustomOp class.

        Returns:
            dict: Dictionary mapping attribute names to their type specifications
        """
        # Start from parent operator class attributes
        attrs = HWCustomOp.get_nodeattr_types(self)
        # Update attributes dictionary for new custom operator
        attrs.update({
            # Number of attention heads
            "heads": ("i", True, 1),
            # Specifies whether the output is packed as a single output tensor
            # or split as multiple output tensors
            "packed": ("i", True, 1),
            # Data type of input and output elements
            "dtype": ("s", True, ""),
            # Number of input elements to be split
            "num_elems": ("i", True, 1),
            # Number of inputs to be processed sequentially
            "num_inputs": ("ints", True, [1]),
            # Possible execution modes for simulating this node
            #   Note: Override to support python mode
            "exec_mode": (
                "s", False, "python", {"", "rtlsim", "cppsim", "python"}
            ),

            # Input and output FIFO depths for multi-I/O nodes
            #   Note: Need to override here as there multiple outputs
            "inFIFODepths": ("ints", False, [2]),
            "outFIFODepths": ("ints", False, []),  # Default will be override
        })
        # Return updated attribute dictionary
        return attrs

    @property
    def heads(self):
        """Get number of attention heads."""
        return self.get_nodeattr("heads")

    @property
    def packed(self):
        """Get packed attribute."""
        # Note: Converts from int to bool
        return bool(self.get_nodeattr("packed"))

    @property
    def dtype(self):
        """Get data type attribute."""
        # Note: Converts from string to QONNX data type
        return DataType[self.get_nodeattr("dtype")]

    @property
    def num_elems(self):
        """Get number of elements attribute."""
        return self.get_nodeattr("num_elems")

    @property
    def num_inputs(self):
        """Get number of inputs attribute."""
        return self.get_nodeattr("num_inputs")


    def make_shape_compatible_op(self, model: ModelWrapper):  # noqa
        """
        Make an operation compatible with the output shape for shape inference
        Note: Propagates shape forward, i.e., never asks for the shape of the output,
        even if it seems easier.
        """
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Determine the rank of the input tensor to support batched and
        # non-batched inputs
        rank = len(self.num_inputs) + 1
        # The input shape determines the sequence length
        (seq, *_), dim = self.num_inputs, self.num_elems
        # Packed outputs a represented by a reshape operation producing one
        # tensor
        if self.packed:
            # Create a new name for the temporary shape tensor
            shape = model.make_new_valueinfo_name()
            # Set the target shape of slices heads
            model.set_initializer(
                shape, np.asarray([self.heads, seq, dim // self.heads])
            )
            # Return a node simulating the shape effect of slicing into
            # multi-heads
            return oh.make_node(
                "Reshape", [node.input[0], shape], [node.output[0]]
            )
        # Prepare a dummy input to simulate reordering of batch/head dimension
        # to the front
        mock_input = model.make_new_valueinfo_name()
        # Set the target shape of slices heads
        model.set_tensor_shape(
            mock_input, [1, seq, dim] if rank == 3 else [seq, dim]
        )
        # If the outputs are not packed, the operation is represented as a split
        # operation producing number of heads outputs along the last axis
        return oh.make_node(
            "Split", [mock_input], node.output, num_outputs=self.heads, axis=-1
        )

    def infer_node_datatype(self, model: ModelWrapper):  # noqa
        """Infer the datatype of the node output."""
        # Get the node wrapped by this custom op  # noqa Duplicate
        node = self.onnx_node
        # Test for changing input datatype
        if model.get_tensor_datatype(node.input[0]) != self.dtype:
            # Get the new datatype
            new_dtype = model.get_tensor_datatype(node.input[0])
            # Issue a warning message
            log.warning(
                f"{node.name}: dtype changing from {self.dtype} to {new_dtype}"
            )
            # Set the new datatype attribute
            self.set_nodeattr("dtype", new_dtype.name)
        # Propagate the type from the input to each output tensor
        for o in node.output:
            # Slicing simply propagates the dtype to the output
            model.set_tensor_datatype(o, self.dtype)

    def _execute_node_python(self, context, graph):  # noqa: graph unused
        """Execute multi-head splitting operation in Python mode.

        Performs the multi-head attention splitting either as a packed operation
        (single input to single output with reshape and transpose) or as a split
        operation (single input to multiple outputs). Input shape must be either
        seq x 1 x dim or seq x dim format.

        Args:
            context: Execution context containing input/output tensors
            graph: ONNX graph (unused but required by interface)
        """
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Get the input out of the execution context
        #   Note: Shape must be either seq x 1 x dim or seq x dim
        inp = context[node.input[0]]
        # Packed execution boils down to a reshape of the single input to a
        # single output
        if self.packed:
            # Reshape to separate the heads out of the embedding dimensions,
            # finally transpose to heads first layout
            out = inp.reshape(inp.shape[0], self.heads, -1).transpose(1, 0, 2)
            # Write the output into the execution context
            context[node.output[0]] = out
        # Split is realized as the split operation of numpy
        else:
            # Produces multiple outputs as a list
            splits = np.split(inp, indices_or_sections=self.heads, axis=-1)
            # Correspondence between outputs and splits in order
            for o, out in zip(node.output, splits):
                # Write the output into the execution context
                context[o] = out

    def _execute_node_cppsim(self, context, graph):  # noqa: graph unused
        """Execute node in C++ simulation mode."""
        # C++ Simulation needs to be implemented in HLS backend specialization
        raise NotImplementedError(
            f"exec_mode cppsim of {self.__class__.__name__} is not implemented!"
        )

    def _execute_node_rtlsim(self, context, graph):  # noqa: graph unused
        """Execute node in RTL simulation mode."""
        # Get the node wrapped by this custom op    # noqa Duplicate
        node = self.onnx_node
        # Input data is stored in numpy files in the code generation dictionary
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        # Get the input out of the execution context
        #   Note: Shape must be either seq x 1 x dim or seq x dim
        inp = context[node.input[0]]
        # Validate the shape of the input
        assert inp.shape == self.get_normal_input_shape(ind=0), \
            f"Input shape mismatch for {node.input[0]}"
        # Reshape the input into folded form
        inp = inp.reshape(self.get_folded_input_shape(ind=0))
        # Path to store the intermediate input in numpy format
        filename = os.path.join(code_gen_dir, "in.npy")
        # Save the folded inputs to file to be used by simulation
        np.save(filename, inp)
        # Start collecting inputs/outputs to the RTL simulation in a dictionary
        #   Note: Prepare one output list per head
        io_dict = {
            "inputs": {}, "outputs": {f"out{i}": [] for i in range(self.heads)}
        }
        # Type and width of the input tensor
        dtype = self.get_input_datatype(ind=0)
        width = self.get_instream_width(ind=0)
        # Convert inputs to RTL simulation format
        io_dict["inputs"]["in"] = npy_to_rtlsim_input(filename, dtype, width)

        # Setup PyVerilator simulation of the node
        sim = self.get_rtlsim()
        # Reset the RTL simulation
        super().reset_rtlsim(sim)
        # Run the RTL Simulation
        self.rtlsim_multi_io(sim, io_dict)

        # Enumerate the node outputs
        for i, name in enumerate(node.output):
            # Collect the output from RTL simulation
            out = io_dict["outputs"][f"out{i}"]
            # Type and sizes of the output tensor
            dtype = self.get_output_datatype(ind=i)
            width = self.get_outstream_width(ind=i)
            shape = self.get_folded_output_shape(ind=i)
            # Path to store the intermediate numpy file
            filename = os.path.join(code_gen_dir, f"out{i}.npy")
            # Convert from RTL simulation format to numpy format
            rtlsim_output_to_npy(
                out, filename, dtype, shape, width, dtype.bitwidth()
            )
            # Load the generated output numpy file
            out = np.load(filename)
            # Reshape the folded output and insert into the execution context
            context[name] = out.reshape(self.get_normal_output_shape(ind=i))

    def execute_node(self, context, graph):
        """Execute the node."""
        # Get the configured execution mode
        mode = self.get_nodeattr("exec_mode")
        # Lookup table mapping execution modes to implementing methods
        exec_fns = {
            "python": self._execute_node_python,
            "cppsim": self._execute_node_cppsim,
            "rtlsim": self._execute_node_rtlsim,
        }
        # Select and execute the function by mode string
        exec_fns[mode](context, graph)

    def verify_node(self):
        """Verify node attribute/input/output correctness."""
        # TODO: Implement
        return []

    # Note: End of QONNX CustomOp region, below is FINN HWCustomOp stuff

    def get_input_datatype(self, ind=0):
        """Get input data type."""
        # All inputs (there should only be one) have the same type
        return self.dtype

    def get_output_datatype(self, ind=0):
        """Get output data type."""
        # All outputs will hae the same type, which is the same as the input
        return self.dtype

    def get_normal_input_shape(self, ind=0):
        """Get normal input shape."""
        # There is only one input with shape configured as attributes
        #   Unpack multi-axis inputs list to yield a flat tuple as shape
        return *self.num_inputs, self.num_elems

    def get_normal_output_shape(self, ind=0):
        """Get normal output shape."""
        # Packed layout is currently not implemented
        assert not self.packed, "Packed multi-heads are not implemented yet"
        # All output have the same shape, which correspond to distributing the
        # number of input elements to the heads specified as attributes
        #   Unpack multi-axis inputs list to yield a flat tuple as shape
        return *self.num_inputs, self.num_elems // self.heads

    def get_folded_input_shape(self, ind=0):
        """Get folded input shape."""
        # No folding for now, normal and folded shape are the same
        return self.get_normal_input_shape(ind=ind)

    def get_folded_output_shape(self, ind=0):
        """Get folded output shape."""
        # No folding for now, normal and folded shape are the same
        return self.get_normal_output_shape(ind=ind)

    def get_instream_width(self, ind=0):
        """Get input stream width."""
        # Get the number of bits used to represent the input
        i_bits = self.get_input_datatype(ind).bitwidth()
        # Parallelism is the number of elements in the last dimension of the
        # folded input
        *_, elems = self.get_folded_input_shape(ind)
        # Width of a stream receiving input elements in parallel
        return elems * i_bits

    def get_outstream_width(self, ind=0):
        """Get output stream width."""
        # Get the number of bits used to represent the output
        o_bits = self.get_output_datatype(ind).bitwidth()
        # Parallelism is the number of elements in the last dimension of the
        # folded output
        *_, elems = self.get_folded_output_shape(ind)
        # Width of a stream producing output elements in parallel
        return elems * o_bits

    # Gets the number of expected output values, i.e. how many times read()
    # could/should be called on any output stream of this operator
    def get_number_output_values(self):
        """Get the number of expected output values, i.e. how many times read()
        could/should be called on any output stream of this operator
        """
        # Elements over all but the last dimension of the output folded along
        # the embedding dimension.
        # In case of multiple outputs, the new FINN XSI simulation back-end requires
        # this to be specified on a per-output basis, in the form of a dict.
        num_outputs_per_stream = np.prod(self.get_folded_output_shape()[:-1])
        if self.heads > 1:
            return {f"out{i}": num_outputs_per_stream for i in range(self.heads)}
        else:
            return num_outputs_per_stream

    def get_exp_cycles(self):
        """Derive the expected cycles of the operator given the folding configuration."""
        # Currently, this implicitly assumes fully parallelized processing
        # along the embedding dimension, i.e., always max PE
        return np.prod(self.num_inputs)


class MergeMultiHeads(HWCustomOp):
    """Merging of attention heads (before output projections) custom operator."""
    # Initializes the operator given an onnx graph node
    def __init__(self, onnx_node, **kwargs):
        """Initialize the operator."""
        # Just forward all arguments to the init method of the CustomOp base
        super().__init__(onnx_node, **kwargs)

        # Need to override the default depths of input FIFOs here as these
        # depend on the number of heads, which are not known during calls to
        # get_nodeattr_types.
        if not self.get_nodeattr("inFIFODepths"):
            self.set_nodeattr("inFIFODepths", [2 for _ in range(self.heads)])

    def get_nodeattr_types(self):
        """Define attributes which must be present on this node"""
        # Start from parent operator class attributes
        attrs = HWCustomOp.get_nodeattr_types(self)
        # Update attributes dictionary for new custom operator
        attrs.update({
            # Number of attention heads
            "heads": ("i", True, 1),
            # Specifies whether the output is packed as a single output tensor
            # or split as multiple output tensors
            "packed": ("i", True, 1),
            # Data type of input and output elements
            "dtype": ("s", True, ""),
            # Number of input elements to be split
            "num_elems": ("i", True, 1),
            # Number of inputs to be processed sequentially
            "num_inputs": ("ints", True, [1]),
            # Output needs to be squeezed
            "squeezed": ("i", True, 0),
            # Possible execution modes for simulating this node
            #   Note: Override to support python mode
            "exec_mode": (
                "s", False, "python", {"", "rtlsim", "cppsim", "python"}
            ),

            # Input and output FIFO depths for multi-I/O nodes
            #   Note: Need to override here as there multiple inputs
            "inFIFODepths": ("ints", False, []),  # Default will be override
            "outFIFODepths": ("ints", False, [2]),
        })
        # Return updated attribute dictionary
        return attrs

    @property
    def heads(self):
        """Get number of attention heads."""
        return self.get_nodeattr("heads")

    @property
    def packed(self):
        """Get packed attribute."""
        # Note: Converts from int to bool
        return bool(self.get_nodeattr("packed"))

    @property
    def dtype(self):
        """Get data type."""
        # Note: Converts from string to QONNX data type
        return DataType[self.get_nodeattr("dtype")]

    @property
    def num_elems(self):
        """Get number of elements."""
        return self.get_nodeattr("num_elems")

    @property
    def num_inputs(self):
        """Get number of inputs."""
        return self.get_nodeattr("num_inputs")

    @property
    def squeezed(self):
        """Get squeezed attribute."""
        # Note: Converts from int to bool
        return bool(self.get_nodeattr("squeezed"))

    def make_shape_compatible_op(self, model: ModelWrapper):  # noqa
        """Makes an operation compatible with the output shape for shape inference
        Note: Propagates shape forward, i.e., never asks for the shape of the
        output, even if it seems easier.
        """
        # Squeeze single-element batch dimension from the output?
        squeezed = self.squeezed
        # Assume unpacked inputs by default, here seq sill be the number of
        # input feature maps
        seq = self.num_inputs
        # Packed inputs a represented by a reshape operation consuming one
        # tensor
        if self.packed:
            # Drop the heads-first dimension from packed inputs
            seq = self.num_inputs[1:]
        # Distribute the heads into the embedding dimension
        dim = self.heads * self.num_elems
        # Constant operation producing output of given shape
        return super().make_const_shape_op(
            [*seq, dim] if squeezed else [*seq, 1, dim]
        )

    def infer_node_datatype(self, model: ModelWrapper):  # noqa
        """Infer the datatype of the node output."""
        # Get the node wrapped by this custom op
        node = self.onnx_node  # noqa Duplicate
        # Test for changing input datatype
        if model.get_tensor_datatype(node.input[0]) != self.dtype:
            # Get the new datatype
            new_dtype = model.get_tensor_datatype(node.input[0])
            # Issue a warning message
            log.warning(
                f"{node.name}: dtype changing from {self.dtype} to {new_dtype}"
            )
            # Set the new datatype attribute
            self.set_nodeattr("dtype", new_dtype.name)
        # All inputs must have the same datatype
        assert all(
            model.get_tensor_datatype(inp) == self.dtype for inp in node.input
        ), f"{node.name}: All inputs must have the same datatype"
        # Merging simply propagates the datatype to the output
        model.set_tensor_datatype(node.output[0], self.dtype)

    def _execute_node_python(self, context, graph):  # noqa: graph unused
        """Execute node in Python mode."""
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Get the input out of the execution context
        #   Note: Shape must be heads x seq x dim
        inp = context[node.input[0]]
        # Packed execution boils down to a reshape of the single input to a
        # single output
        if self.packed:
            # Transpose back into sequence first layout then reintegrate the
            # heads via reshape
            out = inp.transpose(1, 0, 2).reshape(
                inp.shape[1], 1, self.heads * inp.shape[-1]
            )
        # Split is realized as the concat operation of numpy
        else:
            # Collect the list of inputs from the execution context and
            # concatenate along the last axis
            out = np.concatenate([context[i] for i in node.input], axis=-1)
            # Reshape to simulate the batch dimensions if it is not present
            out = out.reshape(out.shape[0], 1, out.shape[-1])
        # Optionally squeeze the output (remove batch dimension of size 1)
        if self.squeezed:
            # Squeeze batch dimension via reshape
            out = out.reshape(out.shape[0], out.shape[-1])
        # Write the output into the execution context. Force output shape
        # which might be squeezed
        context[node.output[0]] = out

    def _execute_node_cppsim(self, context, graph):  # noqa: graph unused
        """Execute node in C++ simulation mode."""
        # C++ Simulation needs to be implemented in HLS backend specialization
        raise NotImplementedError(
            f"exec_mode cppsim of {self.__class__.__name__} is not implemented!"
        )

    def _execute_node_rtlsim(self, context, graph):  # noqa: graph unused
        """Execute node in RTL simulation mode."""
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Input data is stored in numpy files in the code generation dictionary
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        # Start collecting inputs/outputs to the RTL simulation in a dictionary
        #   Note: Prepare one output list per head
        io_dict = {
            "inputs": {}, "outputs": {"out": []}
        }

        # Enumerate the node inputs
        for i, name in enumerate(node.input):
            # Get the input out of the execution context
            #   Note: Shape must be either 1 x seq x dim or seq x dim
            inp = context[name]
            # Validate the shape of the input
            assert inp.shape == self.get_normal_input_shape(ind=i), \
                f"Input shape mismatch for {name}"
            # Reshape the input into folded form
            inp = inp.reshape(self.get_folded_input_shape(ind=i))
            # Path to store the intermediate input in numpy format
            filename = os.path.join(code_gen_dir, f"in{i}.npy")
            # Save the folded inputs to file to be used by simulation
            np.save(filename, inp)
            # Type and width of the input tensor
            dtype = self.get_input_datatype(ind=i)
            width = self.get_instream_width(ind=i)
            # Convert inputs to RTL simulation format
            io_dict["inputs"][f"in{i}"] = npy_to_rtlsim_input(
                filename, dtype, width
            )

        # Setup PyVerilator simulation of the node
        sim = self.get_rtlsim()
        # Reset the RTL simulation
        super().reset_rtlsim(sim)
        # Run the RTL Simulation
        self.rtlsim_multi_io(sim, io_dict)

        # Collect the output from RTL simulation
        out = io_dict["outputs"]["out"]
        # Type and sizes of the output tensor
        dtype = self.get_output_datatype(ind=0)
        width = self.get_outstream_width(ind=0)
        shape = self.get_folded_output_shape(ind=0)
        # Path to store the intermediate numpy file
        filename = os.path.join(code_gen_dir, "out.npy")
        # Convert from RTL simulation format to numpy format
        rtlsim_output_to_npy(
            out, filename, dtype, shape, width, dtype.bitwidth()
        )
        # Load the output numpy file generated by the RTL simulation
        out = np.load(filename)
        # Reshape the folded output and insert into the execution context
        context[node.output[0]] = out.reshape(
            self.get_normal_output_shape(ind=0)
        )

    def execute_node(self, context, graph):
        """Executes multi-head slicing in simulation (either python c++ or rtl sim)."""
        # Get the configured execution mode
        mode = self.get_nodeattr("exec_mode")
        # Lookup table mapping execution modes to implementing methods
        exec_fns = {
            "python": self._execute_node_python,
            "cppsim": self._execute_node_cppsim,
            "rtlsim": self._execute_node_rtlsim,
        }
        # Select and execute the function by mode string
        exec_fns[mode](context, graph)

    def verify_node(self):
        """Verify node attribute/input/output correctness."""
        # TODO: Implement
        return []

    # Note: End of QONNX CustomOp region, below is FINN HWCustomOp stuff

    def get_input_datatype(self, ind=0):
        """Get input data type."""
        # All inputs (there should only be one) have the same type
        return self.dtype

    def get_output_datatype(self, ind=0):
        """Get output data type."""
        # All outputs will have the same type, which is the same as the input
        return self.dtype

    def get_normal_input_shape(self, ind=0):
        """Get normal input shape."""
        # Packed layout is currently not implemented
        assert not self.packed, "Packed multi-heads are not implemented yet"
        # There is only one input with shape configured as attributes
        #   Unpack multi-axis inputs list to yield a flat tuple as shape
        return *self.num_inputs, self.num_elems

    def get_normal_output_shape(self, ind=0):
        """Get normal output shape."""
        # All output have the same shape, which correspond to collecting the
        # number of input elements from the heads specified as attributes
        #   Unpack multi-axis inputs list to yield a flat tuple as shape
        return *self.num_inputs, self.num_elems * self.heads

    def get_folded_input_shape(self, ind=0):
        """Get folded input shape."""
        # No folding for now, normal and folded shape are the same
        return self.get_normal_input_shape(ind=ind)

    def get_folded_output_shape(self, ind=0):
        """Get folded output shape."""
        # No folding for now, normal and folded shape are the same
        return self.get_normal_output_shape(ind=ind)

    def get_instream_width(self, ind=0):
        """Get input stream width."""
        # Get the number of bits used to represent the input
        i_bits = self.get_input_datatype(ind).bitwidth()
        # Parallelism is the number of elements in the last dimension of the
        # folded input
        *_, elems = self.get_folded_input_shape(ind)
        # Width of a stream receiving input elements in parallel
        return elems * i_bits

    def get_outstream_width(self, ind=0):
        """Get output stream width."""
        # Get the number of bits used to represent the output
        o_bits = self.get_output_datatype(ind).bitwidth()
        # Parallelism is the number of elements in the last dimension of the
        # folded output
        *_, elems = self.get_folded_output_shape(ind)
        # Width of a stream producing output elements in parallel
        return elems * o_bits

    def get_number_output_values(self):
        """Gets the number of expected output values, i.e. how many times read()
        could/should be called on any output stream of this operator.
        """
        # Elements over all but the last dimension of the output folded along
        # the embedding dimension
        return np.prod(self.get_folded_output_shape()[:-1])

    def get_exp_cycles(self):
        """Derive the expected cycles of the operator given the folding configuration."""
        # Currently, this implicitly assumes fully parallelized processing
        # along the embedding dimension, i.e., always max PE
        return np.prod(self.num_inputs)

"""HLSBackend specialization of the MultiThreshold operator."""
# FINN HLS custom operator base and registry
from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

# The generic HW custom operator version of the operator as a base class
from finn.custom_op.fpgadataflow.multithreshold import MultiThreshold

# Packing and unpacking utility to feed data to FINN RTL simulation
from finn.util.data_packing import rtlsim_output_to_npy, npy_to_rtlsim_input

# QONNX wrapper to ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# Numpy math and arrays, shape calculations
import numpy as np

# os.path for assembling filenames
import os


@register_custom_op
class MultiThreshold_hls(MultiThreshold, HLSBackend):
    """HLSBackend specialization of the Reshape operator"""

    def get_nodeattr_types(self):
        """Custom node attributes with their types and default values."""
        # Start from parent operator class attributes
        attrs = MultiThreshold.get_nodeattr_types(self)
        # Add the HLSBackend default attributes on top
        attrs.update(HLSBackend.get_nodeattr_types(self))
        # Resource to use for memories (thresholds and values storage)
        attrs["ram_style"] = (
            "s", False, "distributed", {"auto", "distributed", "block", "ultra"}
        )
        # Uses the hls::vector interface
        attrs.update({
            "cpp_interface": ("s", False, "hls_vector", {"hls_vector"})
        })
        # Add/Specialize implementation specific attributes here...
        # Return the updated attributes dictionary
        return attrs

    def global_includes(self):
        """Generate list of C++ includes at the top of the generated code."""
        self.code_gen_dict["$GLOBALS$"] = [
            '#include "multithreshold.hpp"',
            '#include "bind_storage.hpp"',
            '#include "params.hpp"',
        ]

    def defines(self, var):
        """Generate global C++ definitions: types, macros, constants."""
        self.code_gen_dict["$DEFINES$"] = []

    def read_npy_data(self):
        """Generate commands for reading data from .npy file in C++."""

        # Code generation directory for C++ simulation
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        # The HLS operator supports only the hls::vector interface
        assert self.get_nodeattr("cpp_interface") == "hls_vector"

        # Get the shape and type configuration of the operator to generate the
        # HLS C++ types and shape containers
        PE = self.get_nodeattr("PE")
        XType = self.get_input_datatype(ind=0).get_hls_datatype_str()

        # Insert a single npy to stream into the C++ node execution template
        self.code_gen_dict["$READNPYDATA$"] = [
            f'npy2vectorstream<{XType}, float, {PE}>('
            f'  "{os.path.join(code_gen_dir, "input_0.npy")}", in0_V, false'
            f');'
        ]

    def dataoutstrm(self):
        """Generate commands for reading out data from C++ to npy format."""

        # Code generation directory for C++ simulation
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        # The HLS operator supports only the hls::vector interface
        assert self.get_nodeattr("cpp_interface") == "hls_vector"

        # Get the shape and type configuration of the operator to generate the
        # HLS C++ types and shape containers
        *shape, PE = self.get_folded_output_shape()
        OType = self.get_output_datatype(ind=0).get_hls_datatype_str()

        # Generate C++ array representation of the tensor shape
        shape = str((*shape, PE)).replace("(", "{").replace(")", "}")

        # Insert a single stream to npy into the C++ node execution template
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            f'vectorstream2npy<{OType}, float, {PE}>('
            f'  out0_V, {shape}, "{os.path.join(code_gen_dir, "output_0.npy")}"'
            f');'
        ]

    def strm_decl(self):
        """Generate commands for stream declaration in C++."""

        # The HLS operator supports only the hls::vector interface
        assert self.get_nodeattr("cpp_interface") == "hls_vector"

        # Get the shape and type configuration of the operator to generate the
        # HLS C++ types and shape containers
        PE = self.get_nodeattr("PE")

        OType = self.get_output_datatype(ind=0).get_hls_datatype_str()
        XType = self.get_input_datatype(ind=0).get_hls_datatype_str()

        # Generate a single input and a single output stream
        self.code_gen_dict["$STREAMDECLARATIONS$"] = [
            f"Packs<{XType}, {PE}> in0_{self.hls_sname()};",
            f"Packs<{OType}, {PE}> out0_{self.hls_sname()};",
        ]

    def generate_params(self, model: ModelWrapper, path):
        """Generate C++ parameters file including thresholds parameters."""

        # Generate C/C++ array initializer code from Numpy
        def carray(array):
            # Recursion to handle nested dimensions
            if array.ndim > 1:
                return "{" + ",".join((carray(inner) for inner in array)) + "}"
            # Innermost dimension joins values represented as string
            return "{" + ",".join((f"{value:3}" for value in array)) + "}"

        # Get the shape and type configuration of the operator to generate the
        # HLS C++ types and shape containers
        *threshold_shape, N = self.get_normal_input_shape(ind=1)
        *weights_shape, _ = self.get_normal_input_shape(ind=2)

        TShape = f"Shape<{','.join((str(dim) for dim in threshold_shape))}>"
        WShape = f"Shape<{','.join((str(dim) for dim in weights_shape))}>"

        OType = self.get_output_datatype(ind=0).get_hls_datatype_str()
        TType = self.get_input_datatype(ind=1).get_hls_datatype_str()

        # Get the parameter tensors from the model wrapper
        thresholds = model.get_initializer(self.onnx_node.input[1])
        weights = model.get_initializer(self.onnx_node.input[2])

        if self.get_input_datatype(ind=1).is_integer():
            thresholds = thresholds.astype(np.int64)

        # Broadcasting of thresholds and weights along the threshold axis must
        # be made explicit
        if len(weights.shape) < 1 or weights.shape[-1] != N:
            weights = np.broadcast_to(weights, (*weights.shape[:-1], N))

        if len(thresholds.shape) < 1 or thresholds.shape[-1] != N:
            thresholds = np.broadcast_to(
                thresholds, (*thresholds.shape[:-1], N)
            )

        # Get the optional output bias and expand to the weights shape
        bias = np.full((*weights.shape[:-1], 1), self.get_nodeattr("out_bias"))

        # Multi-threshold hardware uses cumulative weights left-padded by the
        # optional bias as output values at/between steps
        weights = np.concatenate((bias, weights), axis=-1)
        values = np.cumsum(weights, axis=-1)

        if self.get_output_datatype(ind=0).is_integer():
            values = values.astype(np.int64)

        # Open a file with int code generation directory to store the thresholds
        # parameters as C++ code
        with open(f"{path}/params.hpp", "w") as file:
            # Write lines of C++ code separated by newlines to the file
            file.write("\n".join([
                f"MultiThreshold<"
                f"{TShape}, {N}, {OType}, {TType}, {WShape}"
                f"> thresholds {{"
                f"{carray(thresholds.reshape(-1, N))},"
                f"{carray(values.reshape(-1, N + 1))}"
                f"}};",
                # Append a newline at the end of the file (to avoid problems
                # when including, required by C standard?)
                "\n"
            ]))

    def docompute(self):
        """Generate C++ code for the computation part of the operator."""

        # Get the shape and type configuration of the operator to generate the
        # HLS C++ types and shape containers
        input_shape = self.get_normal_input_shape(ind=0)
        XShape = f"Shape<{','.join((str(dim) for dim in input_shape))}>"

        # Mat the memory stype attribute to the corresponding HLS selector
        ram_style = {
            "auto": "AUTO",
            "block": "BRAM",
            "distributed": "LUTRAM",
            "ultra": "URAM"
        }[self.get_nodeattr("ram_style")]

        # Write the compute body of the MultiThreshold top-level function
        self.code_gen_dict["$DOCOMPUTE$"] = [
            # Apply bind_storage directives to the threshold parameter arrays to
            # instantiate multi-port ROMs in LUTRAM.
            f"bind_storage<ROM_NP, {ram_style}>(thresholds.thresholds);",
            f"bind_storage<ROM_NP, {ram_style}>(thresholds.values);",
            # Apply multi-threshold operator from input stream to output stream
            f"thresholds.apply<{XShape}>(",
            f"    in0_{self.hls_sname()}, out0_{self.hls_sname()}",
            f");",
        ]

    def blackboxfunction(self):
        """Generate the signature of the C++ top-level function."""

        # Get the shape and type configuration of the operator to generate the
        # HLS C++ types and shape containers
        PE = self.get_nodeattr("PE")

        OType = self.get_output_datatype(ind=0).get_hls_datatype_str()
        XType = self.get_input_datatype(ind=0).get_hls_datatype_str()

        # Write the signature of the MultiThreshold top-level function
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            # Note: Assumes stream type aliases to be set in defines
            f"void {self.onnx_node.name} (",
            f"    Packs<{XType}, {PE}> &in0_{self.hls_sname()},"
            f"    Packs<{OType}, {PE}> &out0_{self.hls_sname()}",
            f")",
        ]

    def pragmas(self):
        """Generate C++ pragmas to be inserted into the main function."""
        # Add HLS interface directives specifying how to create RTL ports for
        # the top-level function arguments
        self.code_gen_dict["$PRAGMAS$"] = [
            f"#pragma HLS INTERFACE axis port=in0_V",
            f"#pragma HLS aggregate variable=in0_V compact=bit",
            f"#pragma HLS INTERFACE axis port=out0_V",
            f"#pragma HLS aggregate variable=out0_V compact=bit",
            # No block-level I/O protocol for the function return value
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        ]

    def execute_node(self, context, graph):
        """Execute multithreshold operation (C++/RTL simulation)."""

        # Execution mode for simulation and wrapped ONNX node
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node

        # Execution mode of simulation must be either C++ or RTL simulation
        assert mode in {"cppsim", "rtlsim"}, f"Invalid exec_mode: {mode}"

        # Load the input tensor from the execution context and reshape into the
        # folded shape expected by the hardware operator
        inp = context[node.input[0]].reshape(self.get_folded_input_shape(0))

        # C++ simulation prepares inputs as numpy .npy files, executes
        # precompiled C++ code and loads results back from .npy files
        if mode == "cppsim":
            # Code generation directory depending on simulation mode
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")

            # Write the input to the node into numpy .npy file to be read by the
            # simulation
            np.save(os.path.join(code_gen_dir, "input_0.npy"), inp)

            # Execute the precompiled node and collect output from .npy into the
            # execution context
            super().exec_precompiled_singlenode_model()
            super().npy_to_dynamic_output(context)

            # Make sure the output has the right type (always use float32 as the
            # container type) and insert into the execution context
            context[node.output[0]] = context[node.output[0]].astype(np.float32)

        # RTL simulation converts the .npy to RTL-simulation compatible inputs,
        # fills the io-dictionary and executes the simulation wrapper
        elif mode == "rtlsim":
            # Convert input to format consumed by the RTL simulation: packing
            # and padding the bits
            inp = npy_to_rtlsim_input(
                inp, self.get_input_datatype(0), self.get_instream_width(0)
            )

            # Prepare inputs and placeholder for the outputs to simulation
            io_dict = {"inputs": {"in0": inp}, "outputs": {"out0": []}}

            # Get the RTL simulator instance for this operator
            sim = self.get_rtlsim()

            # Execute node in RTL simulation
            super().reset_rtlsim(sim)
            self.rtlsim_multi_io(sim, io_dict)
            super().close_rtlsim(sim)

            # Extract the output from the simulation: Remove packing and padding
            output = rtlsim_output_to_npy(
                io_dict["outputs"]["out0"],
                None,  # Do not use indirection via .npy file
                self.get_output_datatype(),
                self.get_folded_output_shape(0),
                self.get_outstream_width(0),
                self.get_output_datatype(0).bitwidth()
            )

            # Reshape the output to remove folding from the last two dimensions
            output = output.reshape(self.get_normal_output_shape())

            # Make sure the output has the right type (always use float32 as the
            # container type) and insert into the execution context
            context[node.output[0]] = output.astype(np.float32)

"""HLSBackend specialization of the PowerQuantMatMul operator."""

# FINN HLS custom operator base and registry
from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

# The generic HW custom operator version of the operator as a base class
from finn.custom_op.fpgadataflow.powerquant import PowerQuantMatMul

# Packing and unpacking utility to feed data to FINN RTL simulation
from finn.util.data_packing import rtlsim_output_to_npy, npy_to_rtlsim_input

# QONNX wrapper to ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# Numpy math and arrays, shape calculations
import numpy as np

# Remove indentation from generated code
from textwrap import fill, dedent
# Template engine for rendering the code
from jinja2 import Template

# os.path for assembling filenames
import os


def _format_str(array: np.ndarray, no_fmt: bool = False) -> str:
    """Get the formatstring for rendering the array."""
    if not no_fmt:
        if np.issubdtype(array.dtype, np.integer):
            n = np.ceil(
                np.log(max(abs(int(np.max(array))), 1) + 1) / np.log(16)
            )
            return f"{{:+#0{int(n + 3)}x}}"

        if np.issubdtype(array.dtype, np.floating):
            n = np.ceil(np.log10(max(abs(int(np.max(array + 1))), 1)))
            p = np.finfo(array.dtype).precision
            return f"{{:+#0{int(n + p + 1)}.{p}f}}"

        if np.issubdtype(array.dtype, np.bool):
            return f"{{:+#03x}}"

    # Default formatting...
    return "{}"


def render_array(
        array: np.ndarray | list | tuple, indent=0, width=80, no_fmt=False
) -> str:
    """Renders an array to a string suitable for C++ code generation."""
    # If the array is a raw python list or tuple (potentially nested), convert
    # to NumPy array for convenient flattening.
    if not isinstance(array, np.ndarray):
        array: np.ndarray = np.asarray(list(array))

    # Figure out the number of digits required to render the longest number from
    # the array.
    fstr = _format_str(array, no_fmt)
    # Join all elements into a single long string separated by comma. Render
    # each element to the same width
    string = ", ".join(fstr.format(x) for x in array.flatten())

    # Convert indent to string representation
    indent = indent * " "

    # Wrap the long string to the configure width and indent all subsequent
    # lines to properly lay out the generated code.
    return str.strip(fill(
        string, subsequent_indent=indent, initial_indent=indent, width=width
    ))


@register_custom_op
class PowerQuantMatMul_hls(PowerQuantMatMul, HLSBackend):
    """HLSBackend specialization of the PowerQuantMatMul operator"""

    def get_nodeattr_types(self):
        """Custom node attributes with their types and default values."""

        # Start from parent operator class attributes and update using backend
        # and operator specific attributes
        attrs = PowerQuantMatMul.get_nodeattr_types(self)
        attrs.update(HLSBackend.get_nodeattr_types(self))

        attrs.update({
            # Only supports the hls::vector interface option
            "cpp_interface": ("s", False, "hls_vector", {"hls_vector"})
        })

        return attrs

    def global_includes(self):
        """Generate list of C++ includes at the top of the generated code."""

        includes = """
        // std::size_t
        #include <cstddef>
        // std::array
        #include <array>
        
        // Vitis HLS vector of parallel elements
        #include <hls_vector.h>
        // Vitis HLS stream interface
        #include <hls_stream.h>
        
        // Vitis HLS arbitrary precision datatypes
        #include <ap_int.h>
        #include <ap_fixed.h>
        
        // PowerQuantMatMul operator template
        #include "powerquant.hpp"
        """

        self.code_gen_dict["$GLOBALS$"] = [
            *dedent(includes).strip().split("\n")
        ]

    def defines(self, var):
        """Generate global C++ definitions: types, macros, constants."""

        defines = """
        // Instantiate the PowerQuant matrix multiplication
        using MatMul = PowerQuantMatMul<{{max}}, {{integer}}, {{fractional}}>;
        
        // Set the input, weight and output types and shapes
        using XType = {{x_type}};
        using WType = {{w_type}};
        using YType = ap_int<MatMul::Accumulator<{{shape_x[-1]}}>::width>;
        
        using XShape = Shape<{{render_array(shape_x, no_fmt=True)}}>;
        using WShape = Shape<{{render_array(shape_w, no_fmt=True)}}>;
        using YShape = Shape<{{render_array(shape_y, no_fmt=True)}}>;
        
        constexpr static std::size_t SIMD = {{SIMD}};
        constexpr static std::size_t PE = {{PE}};
        
        // Include parameters here to have access to all definitions
        #include "params.hpp"
        """

        # Only implemented for integer types...
        if not (input_type := self.get_input_datatype(ind=0)).is_integer():
            return

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

        # The HLS implementation always expects 3-D shapes. If the shapes are
        # shorter or longer, we can expand/flatten as long as they are otherwise
        # compatible.
        shape_x = self.get_normal_input_shape(0)
        shape_w = self.get_normal_input_shape(1)
        shape_y = self.get_normal_output_shape(0)

        while len(shape_x) < 3:
            shape_x = (1, *shape_x)

        if len(shape_x) > 3:
            shape_x = (np.prod(shape_x[:-2], *shape_x[-2:]))

        while len(shape_w) < 3:
            shape_w = (1, *shape_w)

        if len(shape_w) > 3:
            shape_w = (np.prod(shape_w[:-2], *shape_w[-2:]))

        while len(shape_y) < 3:
            shape_y = (1, *shape_y)

        if len(shape_y) > 3:
            shape_y = (np.prod(shape_y[:-2], *shape_y[-2:]))

        defines = Template(dedent(defines)).render(
            max=maximum,
            integer=integer,
            fractional=fractional,
            x_type=input_type.get_hls_datatype_str(),
            w_type=weight_type.get_hls_datatype_str(),
            shape_x=shape_x,
            shape_w=shape_w,
            shape_y=shape_y,
            SIMD=self.simd,
            PE=self.pe,
            render_array=render_array
        )

        self.code_gen_dict["$DEFINES$"] = [
            *dedent(defines).strip().split("\n")
        ]

    def read_npy_data(self):
        """Generate commands for reading data from .npy file in C++."""

        # Code generation directory for C++ simulation
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        # The HLS operator supports only the hls::vector interface
        assert self.get_nodeattr("cpp_interface") == "hls_vector"

        # Get the shape and type configuration of the operator to generate the
        # HLS C++ types and shape containers
        dtype = self.get_input_datatype(ind=0).get_hls_datatype_str()

        # Insert a single npy to stream into the C++ node execution template
        self.code_gen_dict["$READNPYDATA$"] = [
            f'npy2vectorstream<{dtype}, float, {self.simd}>('
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
        *shape, pe = self.get_folded_output_shape()
        dtype = self.get_output_datatype(ind=0).get_hls_datatype_str()

        # Generate C++ array representation of the tensor shape
        shape = str((*shape, pe)).replace("(", "{").replace(")", "}")

        # Insert a single stream to npy into the C++ node execution template
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            f'vectorstream2npy<{dtype}, float, {pe}>('
            f'  out0_V, {shape}, "{os.path.join(code_gen_dir, "output_0.npy")}"'
            f');'
        ]

    def strm_decl(self):
        """Generate commands for stream declaration in C++."""

        # The HLS operator supports only the hls::vector interface
        assert self.get_nodeattr("cpp_interface") == "hls_vector"

        # Get the shape and type configuration of the operator to generate the
        # HLS C++ types and shape containers
        simd = self.get_nodeattr("SIMD")
        pe = self.get_nodeattr("PE")

        otype = self.get_output_datatype(ind=0).get_hls_datatype_str()
        xtype = self.get_input_datatype(ind=0).get_hls_datatype_str()

        # Generate a single input and a single output stream
        self.code_gen_dict["$STREAMDECLARATIONS$"] = [
            f"hls::stream<hls::vector<{xtype},{simd}>> in0_{self.hls_sname()};",
            f"hls::stream<hls::vector<{otype},{pe}>> out0_{self.hls_sname()};",
        ]

    def generate_params(self, model: ModelWrapper, path):
        """Generate C++ parameters file including thresholds parameters."""

        # Get the parameter tensors from the model wrapper and ensure integer
        # values are actual integers to avoid float artifacts for large values
        weights = model.get_initializer(self.onnx_node.input[1])
        weights = weights.astype(np.int64)

        # Rearrange weights into folded shape, packing SIMD x PE tiles in the
        # innermost dimension
        weights = weights.reshape(
            -1, weights.shape[-2] // self.simd,
            self.simd, weights.shape[-1] // self.pe, self.pe
        )
        weights = weights.transpose((0, 1, 3, 2, 4))

        params = """
        // Instantiate the PowerQuant matrix multiplication
        const MatMul matmul(/*alpha=*/{{alpha}});
        
        // Constant weights parameter array
        static const WType weights[WShape::size / SIMD / PE][SIMD][PE] = {
            {{render_array(weights, indent=4)}}
        };
        """

        params = Template(dedent(params).strip()).render(
            alpha=self.get_nodeattr("alpha"),
            weights=weights,
            render_array=render_array
        )

        # Open a file with int code generation directory to store the thresholds
        # parameters as C++ code
        with open(f"{path}/params.hpp", "w") as file:
            # Write lines of C++ code separated by newlines to the file
            file.write(params)

    def docompute(self):
        """Generate C++ code for the computation part of the operator."""

        docompute = f"""
        #pragma HLS dataflow
        #pragma HLS bind_storage variable=weights type=ROM_NP impl=LUTRAM
            matmul.apply<XShape, WShape, YShape, SIMD, PE>(
                in0_{self.hls_sname()}, weights, out0_{self.hls_sname()}
            );
        """

        self.code_gen_dict["$DOCOMPUTE$"] = [
            *dedent(docompute).strip().split("\n")
        ]

    def blackboxfunction(self):
        """Generate the signature of the C++ top-level function."""

        blackboxfunction = f"""
        void powerquant(
            hls::stream<hls::vector<XType, SIMD>>& in0_{self.hls_sname()},
            hls::stream<hls::vector<YType, PE>>& out0_{self.hls_sname()}
        )
        """

        self.code_gen_dict["$DOCOMPUTE$"] = [
            *dedent(blackboxfunction).strip().split("\n")
        ]

    def pragmas(self):
        """Generate C++ pragmas to be inserted into the main function."""
        self.code_gen_dict["$PRAGMAS$"] = [
            f"#pragma HLS INTERFACE axis port=in0_V",
            f"#pragma HLS aggregate variable=in0_V compact=bit",
            f"#pragma HLS INTERFACE axis port=out0_V",
            f"#pragma HLS aggregate variable=out0_V compact=bit",
            # No block-level I/O protocol for the function return value
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        ]

    def execute_node(self, context, graph):
        """Execute PowerQuant matmul operation (C++/RTL simulation)."""

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

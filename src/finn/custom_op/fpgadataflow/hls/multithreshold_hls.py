"""HLSBackend specialization of the MultiThreshold operator."""
# FINN HLS custom operator base and registry
from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

# The generic HW custom operator version of the operator as a base class
from finn.custom_op.fpgadataflow.multithreshold import MultiThreshold

# QONNX wrapper to ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# Numpy math and arrays, shape calculations
import numpy as np


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
        self.code_gen_dict["$DEFINES$"] = []

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

        TShape = f"Shape<{','.join((str(dim) for dim in threshold_shape))}>"

        OType = self.get_output_datatype(ind=0).get_hls_datatype_str()
        TType = self.get_input_datatype(ind=1).get_hls_datatype_str()

        # Get the parameter tensors from the model wrapper
        thresholds = model.get_initializer(self.onnx_node.input[1])
        weights = model.get_initializer(self.onnx_node.input[2])

        if self.get_input_datatype(ind=1).is_integer():
            thresholds = thresholds.astype(np.int64)

        # Get the optional output bias and expand to the weights shape
        bias = np.full((*weights.shape[:-1], 1), self.get_nodeattr("out_bias"))

        # Multi-threshold hardware uses cumulative weights left-padded by the
        # optional bias as output values at/between steps
        weights = np.concatenate((bias, weights), -1)
        values = np.cumsum(weights, axis=-1)

        if self.get_output_datatype(ind=0).is_integer():
            values = values.astype(np.int64)

        # Open a file with int code generation directory to store the thresholds
        # parameters as C++ code
        with open(f"{path}/params.hpp", "w") as file:
            # Write lines of C++ code separated by newlines to the file
            file.write("\n".join([
                f"MultiThreshold<{TShape}, {N}, {OType}, {TType}> thresholds {{"
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
            # f"#pragma HLS bind_storage variable=thresholds.thresholds type=ROM_NP impl={ram_style}",
            f"bind_storage<ROM_NP, {ram_style}>(thresholds.values);",
            # f"#pragma HLS bind_storage variable=thresholds.values type=ROM_NP impl={ram_style}",
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
            f"#pragma HLS INTERFACE axis port=in0_{self.hls_sname()}",
            f"#pragma HLS INTERFACE axis port=out0_{self.hls_sname()}",
            # No block-level I/O protocol for the function return value
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        ]

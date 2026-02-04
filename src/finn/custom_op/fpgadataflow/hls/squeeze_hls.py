"""HLS backend implementation of the Squeeze operator."""
# noqa: Duplicate: The HLS implementation is identical to the Unsqueeze
#  operator, maybe these should be unified...
# fmt: off
# Disable formatter. This is deliberately formatted to stay within 80 characters
# per line. Black, however, formats some lines going beyond this.

# Numpy math and arrays
import numpy as np

# Utility for registering HLSBackend HWCustomOp implementations into the module
# scope
from finn.custom_op.fpgadataflow.hls import register_custom_op

# Base class for specializing HW operators as implemented via HLS
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

# The generic HW custom operator version of the operator as a base class
from finn.custom_op.fpgadataflow.squeeze import Squeeze


# HLS Backend specialization of the squeeze operator
@register_custom_op
class Squeeze_hls(Squeeze, HLSBackend):  # noqa: N801
    """HLS backend implementation of the Squeeze operator.

    Removes single-dimension entries from the shape of a tensor using HLS synthesis.
    """

    # CapWords convention
    # Node attributes matching the HLS operator
    def get_nodeattr_types(self):
        """Return the dictionary of node attributes for the HLS Squeeze operator."""
        # Start from parent operator class attributes
        attrs = Squeeze.get_nodeattr_types(self)
        # Add the HLSBackend default attributes on top
        attrs.update(HLSBackend.get_nodeattr_types(self))
        # Add/Specialize implementation specific attributes here...
        # Return the updated attributes dictionary
        return attrs

    # Generates list of C++ includes to be placed at the top of the generated code
    def global_includes(self) -> None:
        """Generate list of C++ includes for the top of the generated code."""
        # Currently nothing to include
        self.code_gen_dict["$GLOBALS$"] = []

    # Generates C++ code of type alias, global constant and macro definitions
    def defines(self, var) -> None:
        """Generate C++ code for type alias, global constant, and macro definitions."""
        # Currently nothing to define
        self.code_gen_dict["$DEFINES$"] = []

    def execute_node(self, context, graph) -> None:
        """Execute node via generic HLSBackend implementation (cppsim/rtlsim)."""
        HLSBackend.execute_node(self, context, graph)

    # Generates C++ code for calling the computation part of the operator
    def docompute(self) -> None:
        """Generate C++ code for the computation part of the operator."""
        # Number of iterations required to process the whole folded input stream
        #   Note: This is all but the PE (last) dimension
        num_iter = np.prod(self.get_folded_output_shape()[:-1])
        # Write the body of the top-level function
        self.code_gen_dict["$DOCOMPUTE$"] = [
            # Repeat for the number of inputs
            f"for(std::size_t i = 0; i < {num_iter}; ++i) {{",
            # Pipeline the steps of this loop
            "#pragma HLS pipeline II=1 style=flp",
            # Just read from the input and immediately write the same element to
            # the output. Squeezed dimensions, i.e., those with a size of 1 do
            # not contribute to the number and order of elements and thus can
            # simply be ignored.
            "out0_V.write(in0_V.read());",
            "}"  # noqa: f-string symmetry
        ]

    # Generates essentially the head of the C++ function from which the IP block
    # will be generated during ipgen, i.e. actual synthesis
    def blackboxfunction(self) -> None:
        """Generate the C++ function signature for the IP block generation."""
        # Insert function head describing the top level interface of the
        # squeeze operator
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            # Note: Assumes stream type aliases to be set in defines
            f"void {self.onnx_node.name} (",
            f"  hls::stream<ap_uint<{self.get_instream_width()}>> &in0_V,",
            f"  hls::stream<ap_uint<{self.get_outstream_width()}>> &out0_V",
            ")",
        ]

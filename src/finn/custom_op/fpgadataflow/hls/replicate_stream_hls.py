# fmt: off
# Disable formatter. This is deliberately formatted to stay within 80 characters
# per line. Black, however, formats some lines going beyond this.

# Numpy math and arrays
import numpy as np

# Base class for specializing HW operators as implemented via HLS
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
# The generic HW custom operator version of the operator as a base class
from finn.custom_op.fpgadataflow.replicate_stream import ReplicateStream


# HLS Backend specialization of the stream-replication operator
class ReplicateStream_hls(  # noqa: Class name does not follow
    # CapWords convention
    ReplicateStream, HLSBackend
):
    # Node attributes matching the HLS operator
    def get_nodeattr_types(self):
        # Start from parent operator class attributes
        attrs = ReplicateStream.get_nodeattr_types(self)
        # Add the HLSBackend default attributes on top
        attrs.update(HLSBackend.get_nodeattr_types(self))
        # Add/Specialize implementation specific attributes here...
        # Return the updated attributes dictionary
        return attrs

    # Maximum width of any ap_int used in this operator
    def get_ap_int_max_w(self):
        # Find the widths of the widest input
        # Note: There is just one input.
        i_bits_max = self.get_instream_width(ind=0)
        # Find the widths of the widest output
        # Note: there is one output per replica
        o_bits_max = max(
            (self.get_outstream_width(ind) for ind in range(self.num))
        )
        # Find the biggest of the inputs/outputs
        return max([i_bits_max, o_bits_max])

    # Note: End of shape and datatype utilities

    # Generates list of C++ includes to be placed at the top of the generated
    # code
    def global_includes(self):
        # Currently nothing to include
        self.code_gen_dict["$GLOBALS$"] = []

    # Generates C++ code of type alias, global constant and macro definitions
    def defines(self, var):
        # Insert constants and type aliases into the dictionary
        self.code_gen_dict["$DEFINES$"] = [
            # Input and output element datatypes
            f"using IType = {self.dtype.get_hls_datatype_str()};",
            f"using OType = {self.dtype.get_hls_datatype_str()};",
            # Width of single elements to avoid using ::width attribute which is
            # not present for datatype float
            f"static constexpr auto ElemWidth = {self.dtype.bitwidth()};"
            # Datatype of elements packed into the input stream
            f"using IPacked = ap_uint<{self.get_instream_width()}>;",
            # Datatype of elements packed into the output stream
            f"using OPacked = ap_uint<{self.get_outstream_width()}>;",
            # Input and output HLS stream datatypes
            "using IStream = hls::stream<"
            f"  ap_uint<{self.get_instream_width()}>"
            ">;",
            "using OStream = hls::stream<"
            f"  ap_uint<{self.get_outstream_width()}>"
            ">;",
        ]

    # Generates C++ code for calling the computation part of the operator
    def docompute(self):
        # Generates the name of the ith output stream
        def out(i):
            return f"out{i}_{self.hls_sname()}"

        # Number of iterations required to process the whole folded input stream
        #   Note: This is all but the PE (last) dimension
        num_iter = np.prod(self.get_folded_output_shape()[:-1])

        # Write the body of the stream replicating top-level function
        self.code_gen_dict["$DOCOMPUTE$"] = [
            # Repeat for the number of inputs
            # Note: Repeat for all num_inputs dimensions
            f"for(std::size_t i = 0; i < {num_iter}; ++i) {{",
            # Pipeline the steps of this loop
            "#pragma HLS pipeline II=1 style=flp",
            # Read the next input element from the stream
            f"const auto x = in0_{self.hls_sname()}.read();",
            # Write the same input element into each output stream
            *(f"{out(i)}.write(x);" for i in range(self.num)),
            # End of for-loop over repetitions body
            f"}}"  # noqa: f-string symmetry
        ]

    # Generates essentially the head of the C++ function from which the IP block
    # will be generated during ipgen, i.e. actual synthesis
    def blackboxfunction(self):
        # Insert function head describing the top level interface of the stream
        # replicating operator
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            # @formatter:off Prevent Python formatter from messing with C++
            # formatting
            # Note: Assumes stream type aliases to be set in defines
            f"void {self.onnx_node.name} (",
            # Input HLS stream
            f"IStream &in0_{self.hls_sname()}, ", ",".join([
                # One output HLS stream per replica  # noqa: Formatting
                f"OStream &out{i}_{self.hls_sname()}" for i in range(self.num)
            ]),
            ")",
            # @formatter:off
        ]

    # Generates C++ pragmas to be inserted into the main function of the C++
    # simulation and the ipgen-blackboxfunction as well
    def pragmas(self):
        # Add HLS interface directives specifying how to create RTL ports for
        # the top-level function arguments
        self.code_gen_dict["$PRAGMAS$"] = [
            # Connect the input stream with an axi stream interface
            f"#pragma HLS INTERFACE axis port=in0_{self.hls_sname()}"
        ]
        # Connect each output stream with an axi stream interface
        for i in range(self.num):
            # Add new interface directive for the output stream
            self.code_gen_dict["$PRAGMAS$"] += [
                f"#pragma HLS INTERFACE axis port=out{i}_{self.hls_sname()}"
            ]
        # No block-level I/O protocol for the function return value
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )

    # Returns the names of input and output interfaces grouped by protocol
    def get_verilog_top_module_intf_names(self):
        # Start collecting interface names in a dictionary  # noqa Duplicate
        # starting with clock and reset
        intf_names = {"clk": ["ap_clk"], "rst": ["ap_rst_n"]}  # noqa
        # AXI stream input interfaces
        intf_names["s_axis"] = [
            # Just one input stream
            (f"in0_{self.hls_sname()}", self.get_instream_width_padded(ind=0)),
        ]
        # AXI stream output interfaces
        intf_names["m_axis"] = [
            # One output stream per replica
            (f"out{i}_{self.hls_sname()}",
             self.get_outstream_width_padded(ind=i)) for i in range(self.num)
        ]
        # No AXI-MM, AXI-Lite or protocol-less interfaces
        intf_names["aximm"] = []
        intf_names["axilite"] = []
        intf_names["ap_none"] = []
        # Return the interface name dictionary
        return intf_names

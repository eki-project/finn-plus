"""HLS backend implementation of scaled dot-product attention operator."""

# fmt: off
# Disable formatter. This is deliberately formatted to stay within 80 characters
# per line. Black, however, formats some lines going beyond this.

# Numpy math and arrays
import numpy as np

# QONNX/FINN datatypes
from qonnx.core.datatype import DataType
# QONNX wrapper to ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper
# Some utils for working with tensors in qonnx
from qonnx.util.basic import interleave_matrix_outer_dim_from_partitions

# The generic HW custom operator version of the operator as a base class
from finn.custom_op.fpgadataflow.attention import ScaledDotProductAttention
# Base class for specializing HW operators as implemented via HLS
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
# Convert and pack (numpy) data for C++ code generation
from finn.util.data_packing import numpy_to_hls_code

# Mapping of memory resource attributes to the corresponding C++ HLS
# pragma directives
RAM_STYLES = {
    "auto": "AUTO", "block": "BRAM", "distributed": "LUTRAM", "ultra": "URAM"
}


class ScaledDotProductAttention_hls(  # noqa: Class name does not follow
    # CapWords convention
    ScaledDotProductAttention, HLSBackend
):
    """HLS Backend specialization of the Scaled Dot-product Attention Operator."""

    def get_nodeattr_types(self):
        """Return node attributes matching the HLS operator."""
        # Start from parent operator class attributes
        attrs = ScaledDotProductAttention.get_nodeattr_types(self)
        # Add the HLSBackend default attributes on top
        attrs.update(HLSBackend.get_nodeattr_types(self))
        # Add/Specialize implementation specific attributes here...
        # Return the updated attributes dictionary
        return attrs

    def get_ap_int_max_w(self):
        """Return the maximum width of any ap_int used in this operator.

        Calculates the maximum bit width required across all inputs, outputs,
        and optional mask elements for use in determining the widest ap_int
        type needed for HLS synthesis.
        """
        # Find the widths of the widest input
        i_bits_max = max((self.get_instream_width(ind) for ind in range(3)))
        # Find the widths of the widest output
        o_bits_max = max((self.get_outstream_width(ind) for ind in range(1)))
        # Assume no bits to represent the mask, if there is no mask
        m_bits = 0
        # A mask received as input has a bit-width as well
        if self.get_nodeattr("mask_mode") in {"input", "const"}:
            # Parallelism is the number of elements in the last dimension of the
            # folded mask input
            *_, elems = self.get_folded_input_shape(ind=3)
            # Get width of the mask datatype
            m_bits = elems * DataType[self.get_nodeattr("MType")].bitwidth()

        # Elements per folded key input (second input)
        *_, i_elems = self.get_folded_input_shape(ind=1)
        # Elements per folded value input (third input), same as the number of
        # output elements
        *_, o_elems = self.get_folded_input_shape(ind=2)

        # Parallelism is the number of elements in the last dimension of the
        # folded attention weights
        *_, s_elems = self.get_folded_attention_shape()
        # Number of bits used for the attention weights stream
        a_bits = s_elems * DataType[self.get_nodeattr("AType")].bitwidth()

        # Maximum bits per tile of the key and value matrix streams
        tile_bits_max = max([
            i_elems * s_elems * DataType[self.get_nodeattr("KType")].bitwidth(),
            o_elems * s_elems * DataType[self.get_nodeattr("VType")].bitwidth(),
        ])
        # Maximum bits per matmul accumulators
        acc_bits_max = max([
            # These are not streamed, thus single element width is counted
            DataType[self.get_nodeattr("AccQKMatMul")].bitwidth(),
            DataType[self.get_nodeattr("AccAVMatMul")].bitwidth(),
        ])
        # Maximum bits per matmul outputs
        out_bits_max = max([
            # These are the stream widths, which are always >= than individual
            # elements
            s_elems * DataType[self.get_nodeattr("OutQKMatMul")].bitwidth(),
            o_elems * DataType[self.get_nodeattr("OutAVMatMul")].bitwidth(),
        ])
        # Aggregate the maximum bit width in both matmul operators over all
        # inputs, intermediates and outputs
        matmul_bits_max = max([
            tile_bits_max, acc_bits_max, out_bits_max
        ])

        # Find maximum of all (maximal) bit-widths
        return max([i_bits_max, o_bits_max, m_bits, a_bits, matmul_bits_max])

    def global_includes(self):
        """Generate list of C++ includes to be placed at the top of the generated code.

        Adds necessary header files for the attention operator HLS implementation,
        including FINN HLSLIB activation functions and the attention-specific
        HLS implementation header.
        """
        # FINN HLSLIB activation functions: e.g. PassThroughActivation
        self.code_gen_dict["$GLOBALS$"] = ['#include "activations.hpp"']
        # Attention operator HLS code
        self.code_gen_dict["$GLOBALS$"] += ['#include "attention.hpp"']

    def generate_params(self, model: ModelWrapper, path):
        """Generate C++ parameters file including activation function thresholds.

        Creates parameter files including activation function thresholds and
        other configuration parameters needed for HLS synthesis. The code
        generation directory is specified as an argument to work for both
        RTL and C++ simulation modes.
        """
        # The code generation directory is specified as an argument, so this
        # will work for both RTL and C++ simulation
        code_gen_dir = path

        # Note: The attention operator itself has no weights to be generated as
        # a parameter file

        # Start all three activations defaulting to pass-through of the
        # accumulator type.
        #   Note: This might allow type-casts to the output types if they are
        #   not the same as the accumulators.
        act_qk_matmul = "PassThroughActivation<AccQKMatMul>"
        act_av_matmul = "PassThroughActivation<AccAVMatMul>"
        act_a_softmax = "PassThroughActivation<float>"

        # Start all thresholds defaulting to empty default initializer braces
        thresholds_qk_matmul = "{}"
        thresholds_av_matmul = "{}"
        thresholds_a_softmax = "{}"

        def prepare_thresholds(ts, length, fold, dtype):
            """Prepare a threshold tensor as C++ string for code generation.

            Converts threshold tensors into the proper format for HLS code generation.
            Performs broadcasting from per-tensor to per-channel, partitions thresholds
            along the length dimension into parallel folds, and formats as C++ code.
            """
            # Number of thresholds is given as the last dimension of the
            # threshold tensor, first dimension is covering all output elements
            num = ts.shape[-1]  # noqa
            # Explicitly broadcast thresholds from per-tensor to per-channel
            ts = np.broadcast_to(ts, (length, num))
            # Partition the thresholds along the length into folds of parallel
            # elements
            ts = interleave_matrix_outer_dim_from_partitions(ts, length // fold)
            # Reshape folded thresholds adding an outer dimension
            ts = ts.reshape(1, length // fold, fold, num)
            # Format the thresholds as C++ array code
            # Note: no packing, no variable name/type declaration
            return numpy_to_hls_code(ts, dtype, "_", False, True), num

        # Get shape and folding configuration. None of the activations fold
        # along the query-key embedding dimension or the query sequence length
        (_, _, vdim, kvlen), (embfold, seqfold) = self.shapes, self.folds

        # Query-key matmul can have an optional activation function set to
        # thresholding activations via node attribute
        if self.get_nodeattr("ActQKMatMul") == "thresholds":
            # In this case there will be a thresholds parameter initializer
            thresholds = model.get_initializer(
                self.get_input_name_by_name("thresholds_qk_matmul")
            )
            # Get the datatype of the thresholds
            thresholds_dtype = DataType[self.get_nodeattr("AccQKMatMul")]
            # Activation value, i.e., bias applied after thresholding activation
            bias = self.get_nodeattr("BiasActQKMatMul")
            # No support for floating-point bias
            assert int(bias) == bias, "BiasActQKMatMul must be integer"
            # Convert the bias to integer representation, so it can be used as a
            # template argument
            bias = int(bias)
            # Format the thresholds as C++ array code: QK matmul outputs fold
            # along the key-value sequence length dimension
            thresholds_qk_matmul, num = prepare_thresholds(
                thresholds, kvlen, seqfold, thresholds_dtype
            )
            # Get the HLS datatype string corresponding to the thresholds
            # datatype for C++ code generation
            dtype_str = thresholds_dtype.get_hls_datatype_str()
            # Replace default pass-through activation by thresholding activation
            #   Note: Relies on type and shape definitions generated by the
            #   "defines" method
            act_qk_matmul = "\n".join([
                "ThresholdsActivation<",
                " SeqFold,"
                " KVLen/SeqFold,"
                f" {num},"
                " AccQKMatMul,"
                " OutQKMatMul,"
                f" {bias},"
                f" comp::less<{dtype_str}, {dtype_str}>",
                ">"
            ])

        # Softmax can have an optional activation function set to thresholding
        # activations via node attribute
        if self.get_nodeattr("ActASoftmax") == "thresholds":
            # In this case there will be a thresholds parameter initializer
            thresholds = model.get_initializer(
                self.get_input_name_by_name("thresholds_a_softmax")
            )
            # Get the datatype of the thresholds
            thresholds_dtype = DataType[self.get_nodeattr("AccASoftmax")]
            # Activation value, i.e., bias applied after thresholding activation
            bias = self.get_nodeattr("BiasActASoftmax")
            # No support for floating-point bias
            assert int(bias) == bias, "BiasActASoftmax must be integer"
            # Convert the bias to integer representation, so it can be used as a
            # template argument
            bias = int(bias)
            # Format the thresholds as C++ array code: Softmax outputs fold
            # along the key-value sequence length dimension
            thresholds_a_softmax, num = prepare_thresholds(
                thresholds, kvlen, seqfold, thresholds_dtype
            )
            # Get the HLS datatype string corresponding to the thresholds
            # datatype for C++ code generation
            dtype_str = thresholds_dtype.get_hls_datatype_str()
            # Replace default pass-through activation by thresholding activation
            #   Note: Relies on type and shape definitions generated by the
            #   "defines" method
            act_a_softmax = "\n".join([
                "ThresholdsActivation<",
                " SeqFold,"
                " KVLen/SeqFold,"
                f" {num},"
                " AccASoftmax,"
                " AType,"
                f" {bias},"
                f" comp::less<{dtype_str}, {dtype_str}>",
                ">"
            ])

        # Attention-value matmul can have an optional activation function set to
        # thresholding activations via node attribute
        if self.get_nodeattr("ActAVMatMul") == "thresholds":
            # In this case there will be a thresholds parameter initializer
            thresholds = model.get_initializer(
                self.get_input_name_by_name("thresholds_av_matmul")
            )
            # Get the datatype of the thresholds
            thresholds_dtype = DataType[self.get_nodeattr("AccAVMatMul")]
            # Activation value, i.e., bias applied after thresholding activation
            bias = self.get_nodeattr("BiasActAVMatMul")
            # No support for floating-point bias
            assert int(bias) == bias, "BiasActAVMatMul must be integer"
            # Convert the bias to integer representation, so it can be used as a
            # template argument
            bias = int(bias)
            # Format the thresholds as C++ array code: AV matmul outputs fold
            # along the value embedding dimension
            thresholds_av_matmul, num = prepare_thresholds(
                thresholds, vdim, embfold, thresholds_dtype
            )
            # Get the HLS datatype string corresponding to the thresholds
            # datatype for C++ code generation
            dtype_str = thresholds_dtype.get_hls_datatype_str()
            # Replace default pass-through activation by thresholding activation
            #   Note: Relies on type and shape definitions generated by the
            #   "defines" method
            act_av_matmul = "\n".join([
                "ThresholdsActivation<",
                " EmbFold,"
                " VDim/EmbFold,"
                f" {num},"
                " AccAVMatMul,"
                " OutAVMatMul,"
                f" {bias},"
                f" comp::less<{dtype_str}, {dtype_str}>",
                ">"
            ])

        # Assume no attention mask as a default: Generate C++ code of tag
        # instance of "none" mask type
        attention_mask = \
            "static const auto attention_mask = attention::mask::NONE"

        # If a causal mask is specified, set the appropriate tag dispatching
        # instance
        if self.get_nodeattr("mask_mode") == "causal":
            # Generate C++ code of tag instance of causal mask type
            attention_mask = \
                "static const auto attention_mask = attention::mask::CAUSAL"

        # If a constant mask is specified, array code needs to be generated
        if self.get_nodeattr("mask_mode") == "const":
            # Attention mask type of folded constant mask array
            mask_type = "attention::mask::Const<SeqFold, KVLen/SeqFold, QLen>"
            # Get the constant mask values
            mask = model.get_initializer(self.get_input_name_by_name("M"))
            # Num should always be equal to QLen
            num = mask.shape[-1]
            # Partition the mask along the length into folds of parallel
            # elements
            mask = interleave_matrix_outer_dim_from_partitions(
                mask, kvlen // seqfold
            )
            # Reshape folded mask adding an outer dimension
            mask = mask.reshape(num, kvlen // seqfold, seqfold).squeeze()
            # Format the mask as C++ array code
            # Note: no packing, no variable name/type declaration
            mask = numpy_to_hls_code(mask, DataType["BINARY"], "_", False, True)
            # Generate C++ code initializing the constant mask array
            attention_mask = f"static const {mask_type} attention_mask = {mask}"

        # If a mask is provided as input, no object parameters need to be
        # generated here
        if self.get_nodeattr("mask_mode") == "input":
            # Attention mask type of input stream
            mask_type = "attention::mask::Input<SeqFold, KVLen/SeqFold, QLen>"
            # Generate C++ code creating an input stream instance for the mask
            # Note: This is just a dummy, the real input stream will be part
            # of the operator interface
            attention_mask = f"static const {mask_type} attention_mask;"

        # Open a file to store the thresholds parameters as C++ code
        with open(f"{code_gen_dir}/params.hpp", "w") as file:
            # Write lines of C++ code separated by newlines to the file
            file.write("\n".join([
                # Scale factor preceding the softmax activation function to
                # dequantize the input to floating-point representation
                "static const float dequant_softmax ="
                f" {self.get_nodeattr('DequantSoftmax')};",
                # Attention mask parameters if "none", "causal" or "const"
                f"{attention_mask};",
                # Type alias to the generated attention mask for convenience
                "using AttentionMask = decltype(attention_mask);",
                # Add type definition and threshold initialization of the
                # query-key matmul activation
                f"using ActQKMatMul = {act_qk_matmul};",
                f"ActQKMatMul act_qk_matmul = {thresholds_qk_matmul};",
                # Add type definition and threshold initialization of the
                # attention-value matmul activation
                f"using ActAVMatMul = {act_av_matmul};",
                f"ActAVMatMul act_av_matmul = {thresholds_av_matmul};",
                # Add type definition and threshold initialization of the
                # softmax activation
                f"using ActASoftmax = {act_a_softmax};",
                f"ActASoftmax act_a_softmax = {thresholds_a_softmax};",
                # Append a newline at the end of the file (to avoid problems
                # when including, required by C standard?)
                "\n"
            ]))

    def defines(self, var):
        """Generate C++ code of type alias, global constant and macro definitions."""
        def shapedefs(*names):
            """Generate shape definitions from attributes to C++ constant definitions."""
            # C++ qualified type to be used for shape constants
            shape = "static constexpr std::size_t"
            # Generate a C++ constant definition for each of the attributes
            # given by argument list names
            return (
                f"{shape} {name} = {self.get_nodeattr(name)};" for name in names
            )

        def typedefs(*names):
            """Generate datatype definitions mapping from QONNX DataType to HLS type."""
            def hls_type(name):
                """Get the HLS type string for the datatype specified by the named attribute."""
                # Looks up the datatype specified for the attribute and
                # translates from QONNX to HLS type
                return DataType[self.get_nodeattr(name)].get_hls_datatype_str()

            # Generate a C++ type alias definition for each of the attributes
            # given by argument list names
            return (f"using {name} = {hls_type(name)};" for name in names)

        # Attribute specifying the memory to use for internal buffers
        ram_style = self.get_nodeattr("ram_style")
        # Attribute specifying the resources to use for implementing MAC
        # operations
        mac_resource = self.get_nodeattr("mac_resource")

        # Mapping of memory resource attributes to the corresponding C++ tag
        # types
        mem_resources = {
            "auto": "Resource::AUTO",
            "block": "Resource::BRAM",
            "distributed": "Resource::LUTRAM",
            "ultra": "Resource::URAM"
        }
        # Mapping of compute resource attributes to the corresponding C++ tag
        # types
        compute_resources = {
            "auto": "ap_resource_dflt",
            "lut": "ap_resource_lut",
            "dsp": "ap_resource_dsp"
        }

        # Insert constants and type aliases into the dictionary
        self.code_gen_dict["$DEFINES$"] = [
            # Shape constant definitions of attention inputs (query, key and
            # value) and folding configuration
            *shapedefs(
                "QKDim",
                "QLen",
                "VDim",
                "KVLen",
                "EmbFold",
                "SeqFold"
            ),
            # Type alias definitions for all input, output and intermediate
            # datatypes
            *typedefs(
                "QType",
                "KType",
                "VType",
                "MType",
                "AType",
                "OType"
            ),
            # Type alias definitions for the matmul accumulators and output
            # datatypes
            *typedefs(
                "AccQKMatMul",
                "OutQKMatMul",
                "AccAVMatMul",
                "OutAVMatMul",
                "AccASoftmax"
            ),
            # Type alias definitions for the resource type selection tags
            f"using MacResource = {compute_resources[mac_resource]};",
            f"using MemResource = {mem_resources[ram_style]};",
            # Include the activation function type definitions and parameters
            #   Note: The typedefs in this header require the typedefs above,
            #   thus adding this to the global includes is not possible.
            '#include "params.hpp"',
            # Type alias of the properly configured attention operator class
            "using Attention = ScaledDotProductAttention<",
            "    QKDim,",
            "    QLen,",
            "    VDim,",
            "    KVLen,",
            "    EmbFold,",
            "    SeqFold,",
            "    QType,",
            "    KType,",
            "    VType,",
            "    MType,",
            "    AType,",
            "    OType,",  # Note: OType and last MatMul out must match
            "    AccQKMatMul,",
            "    OutQKMatMul,",
            "    ActQKMatMul,",
            "    AccAVMatMul,",
            "    OType,",  # Note: OType and last MatMul out must match
            "    ActAVMatMul,",
            "    ActASoftmax,",
            "    MacResource,",
            "    MemResource"
            ">;",
            # Short type aliases of attention input and output streams
            "using QStream = Attention::QStream;",
            "using KStream = Attention::KStream;",
            "using VStream = Attention::VStream;",
            "using OStream = Attention::OStream;",
            "using MStream = Attention::MStream;",
        ]

    def docompute(self):
        """Generate C++ code for calling the computation part of the operator.

        Creates the main HLS computation call with proper RAM style directives
        for thresholds and mask storage, along with necessary pragmas for
        threshold arrays and storage binding.
        """
        # Convert the thresholds RAM style attribute to HLS directive
        ram_style_thresholds = RAM_STYLES[
            self.get_nodeattr("ram_style_thresholds")
        ]
        # Convert the attention mask RAM style attribute to HLS directive
        ram_style_mask = RAM_STYLES[self.get_nodeattr("ram_style_mask")]

        def bind_threshold_storage(name: str):
            """Generate the BIND_STORAGE pragma for the activations threshold memory."""
            return (f"#pragma HLS BIND_STORAGE variable={name}"
                    f" type=ROM_2P impl={ram_style_thresholds}")

        def partition_thresholds_array(name: str, dim: int):
            """Generate the ARRAY_PARTITION pragma for the activations threshold memory."""
            return (f"#pragma HLS ARRAY_PARTITION variable={name}"
                    f" complete dim={dim}")

        # Collect pragmas which need to be inserted into the DOCOMPUTE code
        pragmas = []

        # If there are thresholds activations following the query-key matmul,
        # these need storage and array partition pragmas
        if self.get_nodeattr("ActQKMatMul") == "thresholds":
            # Add pragma compiler directives to the list of pragmas inserted
            # into the DOCOMPUTE
            pragmas.extend([
                # Partition the thresholds array along the PE (dim=1) and number
                # of thresholds (dim=3) axis for parallel access
                partition_thresholds_array(
                    "attention.qk_matmul.activation.m_thresholds", dim=1
                ),
                partition_thresholds_array(
                    "attention.qk_matmul.activation.m_thresholds", dim=3
                ),
                # Implement the thresholds array as a dual-port ROM with the
                # RAM-Style selected via attribute
                bind_threshold_storage(
                    "attention.qk_matmul.activation.m_thresholds"
                )
            ])

        # If there are thresholds activations following the attention-value
        # matmul, these need storage and array partition pragmas
        if self.get_nodeattr("ActAVMatMul") == "thresholds":
            # Add pragma compiler directives to the list of pragmas inserted
            # into the DOCOMPUTE
            pragmas.extend([
                # Partition the thresholds array along the PE (dim=1) and number
                # of thresholds (dim=3) axis for parallel access
                partition_thresholds_array(
                    "attention.av_matmul.activation.m_thresholds", dim=1
                ),
                partition_thresholds_array(
                    "attention.av_matmul.activation.m_thresholds", dim=3
                ),
                # Implement the thresholds array as a dual-port ROM with the
                # RAM-Style selected via attribute
                bind_threshold_storage(
                    "attention.av_matmul.activation.m_thresholds"
                )
            ])

        # If there are thresholds activations following the softmax
        # normalization, these need storage and array partition pragmas
        if self.get_nodeattr("ActASoftmax") == "thresholds":
            # Add pragma compiler directives to the list of pragmas inserted
            # into the DOCOMPUTE
            pragmas.extend([
                # Partition the thresholds array along the PE (dim=1) and number
                # of thresholds (dim=3) axis for parallel access
                partition_thresholds_array(
                    "attention.softmax.activation.m_thresholds", dim=1
                ),
                partition_thresholds_array(
                    "attention.softmax.activation.m_thresholds", dim=3
                ),
                # Implement the thresholds array as a dual-port ROM with the
                # RAM-Style selected via attribute
                bind_threshold_storage(
                    "attention.softmax.activation.m_thresholds"
                )
            ])

        # If a constant mask is specified, there needs to be storage and array
        # partition pragmas to be inserted
        if self.get_nodeattr("mask_mode") == "const":
            # Note: Probably no need for partitioning this array, as the PE
            # dimension is packed into the datatype (which is a bitvector with
            # one bit per element, i.e., per PE)
            # Implement the attention mask array as a dual-port ROM with the
            # RAM-Style selected via attribute
            pragmas.extend([
                f"#pragma HLS BIND_STORAGE variable=attention_mask"
                f" type=ROM_2P impl={ram_style_mask}"
            ])

        # Write the body of the attention top-level function
        self.code_gen_dict["$DOCOMPUTE$"] = [
            # Instantiate the attention operator and connect to the generated
            # threshold parameters
            # Note: Assumes "Attention" to be aliased and configured in defines
            # Note: Assumes parameters to be generated in 'generate_params' and
            #   made available via include/defines before.
            "Attention attention {",
            "    act_qk_matmul, act_av_matmul, act_a_softmax, dequant_softmax",
            "};",
            # Insert some more pragmas here to be able to configure
            # implementation details of components internal to "attention"
            *pragmas,
            # Connect the attention operator to the input and output streams
            f"for(std::size_t i = 0; i < {self.iterations}; ++i) {{",
            "    attention("
            "    in0_V, "  # q
            "    in1_V, "  # k
            "    in2_V, "  # v
            "    out0_V, "  # output
            # TODO: Does not work for "input" mode mask
            "    attention_mask"
            ");",
            "}"
        ]

    def blackboxfunction(self):
        """Generate the head of the C++ function from which the IP block will be generated.

        Creates the function signature describing the top level interface of the
        attention operator for HLS synthesis (ipgen).
        """
        # Insert function head describing the top level interface of the
        # attention operator
        # TODO: support mask input?
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            # Note: Assumes stream type aliases to be set in defines
            f"void {self.onnx_node.name} (",
            "  QStream &in0_V,"
            "  KStream &in1_V,"
            "  VStream &in2_V,"
            "  OStream &out0_V",
            ")",
        ]

    def pragmas(self):
        """Generate C++ pragmas to be inserted into the main function.

        Creates HLS interface directives specifying how to create RTL ports for
        the top-level function arguments in both C++ simulation and ipgen-blackboxfunction.
        """
        # Add HLS interface directives specifying how to create RTL ports for
        # the top-level function arguments
        self.code_gen_dict["$PRAGMAS$"] = [
            # Connect the query input stream with an axi stream interface
            "#pragma HLS INTERFACE axis port=in0_V",
            # Connect the key input stream with an axi stream interface
            "#pragma HLS INTERFACE axis port=in1_V",
            # Connect the value input stream with an axi stream interface
            "#pragma HLS INTERFACE axis port=in2_V",
            # Connect the output stream with an axi stream interface
            "#pragma HLS INTERFACE axis port=out0_V",
        ]
        # No block-level I/O protocol for the function return value
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )

    def get_verilog_top_module_intf_names(self):
        """Return the names of input and output interfaces grouped by protocol.

        Collects interface names in a dictionary organized by protocol type
        (clock, reset, AXI stream, etc.) for Verilog module generation.
        """
        # Start collecting interface names in a dictionary starting with clock
        # and reset
        intf_names = {"clk": ["ap_clk"], "rst": ["ap_rst_n"]}  # noqa
        # AXI stream input interfaces
        # TODO: support mask input?
        intf_names["s_axis"] = [
            ("in0_V", self.get_instream_width_padded(ind=0)),  # q
            ("in1_V", self.get_instream_width_padded(ind=1)),  # k
            ("in2_V", self.get_instream_width_padded(ind=2))  # v
        ]
        # AXI stream output interfaces
        intf_names["m_axis"] = [
            ("out0_V", self.get_outstream_width_padded(ind=0))
        ]
        # No AXI-MM, AXI-Lite or protocol-less interfaces
        intf_names["aximm"] = []
        intf_names["axilite"] = []
        intf_names["ap_none"] = []
        # Return the interface name dictionary
        return intf_names

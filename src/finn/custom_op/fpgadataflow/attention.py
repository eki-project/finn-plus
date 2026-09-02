"""Scaled dot-product attention hardware custom operator."""

# fmt: off
# Disable formatter. This is deliberately formatted to stay within 80 characters
# per line. Black, however, formats some lines going beyond this.

# Python builtin math functions: math.ceil returns int, while np.ceil returns
# float
import math

# Numpy math and arrays
import numpy as np

# ONNX graph node protobuf type used for annotations and node construction
from onnx import NodeProto

# QONNX/FINN datatypes
from qonnx.core.datatype import BaseDataType, DataType

# QONNX wrapper to ONNX model graphs
from qonnx.core.modelwrapper import ModelWrapper

# Multithreshold activations
from qonnx.custom_op.general.multithreshold import multithreshold

# Some utils for working with tensors in qonnx
from qonnx.util.basic import calculate_matvec_accumulator_range

# Typing helpers
from typing import TYPE_CHECKING, cast

# Derive custom operators form the FINN base custom op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

# FINN errors: internal invariant violations and user-fixable config problems
from finn.util.exception import FINNInternalError, FINNUserError

# FINN logging
from finn.util.logging import log

if TYPE_CHECKING:
    # ONNX graph protobuf type, only needed for type annotations
    from onnx import GraphProto

# Type of the dictionary returned by get_nodeattr_types: maps attribute names to
# their (dtype, required, default[, allowed_values]) specification tuples
NodeAttrTypes = dict[
    str,
    tuple[str, bool, int | float | str | bool | np.ndarray | list]
    | tuple[str, bool, int | float | str | bool | np.ndarray | list, set | None],
]


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    """Compute a softmax with overflow handling matching the HLS operator.

    On overflow of the exponential sum, the probability mass is distributed
    equally over all occurrences of the maximum input value along ``axis``.
    """
    # For overflow handling, find the maximum value along axis and place ones at
    # each occurrence
    max_ones = (x == np.max(x, axis=axis, keepdims=True)).astype(np.float32)
    # Count the occurrences of the maximum along the normalization axis
    max_counts = np.sum(max_ones, axis=axis, keepdims=True)
    # Exponential of the input
    exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
    # Compute the total along axis
    total = np.sum(exp, axis=axis, keepdims=True)
    # Detect overflow of the summation
    overflow = np.isinf(total)
    # Replace overflows by equal weight given to all instances of the maximum
    # input value. For non overflow just compute normal softmax
    return np.where(overflow, max_ones / max_counts, exp / total)


# Scaled Dot-Product Attention Custom Operator
class ScaledDotProductAttention(HWCustomOp):
    """Scaled dot-product attention abstraction layer.

    Computes ``softmax(Q @ Kᵀ + mask) @ V`` with optional quantizing
    thresholding activations after each matmul and the softmax, matching the
    behavior of the attention-hlslib operator.
    """

    # Initializes the operator given an onnx graph node
    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize the operator from an ONNX node."""
        # Just forward all arguments to the init method of the CustomOp base
        super().__init__(onnx_node, **kwargs)

    # Node attributes matching the HLS operator
    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return the dictionary of node attributes for this operator."""
        # Start from parent operator class attributes
        attrs = HWCustomOp.get_nodeattr_types(self)
        # Update attributes dictionary for new custom operator
        attrs.update({
            # Embedding dimension of queries and keys
            "QKDim": ("i", True, 0),
            # Length of the query sequence
            "QLen": ("i", True, 0),
            # Embedding dimension of the values
            "VDim": ("i", True, 0),
            # Length of the key and value sequence
            "KVLen": ("i", True, 0),

            # NUmber of attention heads computed by this operator (packed into
            # first dimension, does not distinguish from batch dimension)
            "Heads": ("i", False, 1),

            # Folding along the embedding dimensions
            "EmbFold": ("i", True, 0),
            # Folding along the sequence dimensions
            "SeqFold": ("i", True, 0),

            # Datatype of query matrix elements
            "QType": ("s", True, ""),
            # Datatype of key matrix elements
            "KType": ("s", True, ""),
            # Datatype of value matrix elements
            "VType": ("s", True, ""),
            # Datatype of mask matrix elements
            "MType": ("s", False, "INT0"),
            # Datatype of attention weights elements
            "AType": ("s", False, "UINT32"),
            # Datatype of output elements
            "OType": ("s", True, ""),

            # Datatype of accumulator elements of the Query x Key multiplication
            "AccQKMatMul": ("s", False, "UINT32"),
            # Datatype of output elements of the Query x Key multiplication
            "OutQKMatMul": ("s", False, "UINT32"),
            # Activation function type of the Query x Key multiplication
            "ActQKMatMul": ("s", False, "none", {"none", "thresholds"}),
            # Output bias to be applied to the thresholding activation following
            # the Query x Key multiplication
            "BiasActQKMatMul": ("f", False, 0.0),

            # Datatype of accumulator elements of the Attention x Value
            # multiplication
            "AccAVMatMul": ("s", False, "UINT32"),
            # Datatype of output elements of the Attention x Value
            # multiplication
            "OutAVMatMul": ("s", False, "UINT32"),
            # Activation function type of the Attention x Value multiplication
            "ActAVMatMul": ("s", False, "none", {"none", "thresholds"}),
            # Output bias to be applied to the thresholding activation following
            # the Attention x Value multiplication
            "BiasActAVMatMul": ("f", False, 0.0),

            # Scale factor preceding the softmax normalization to dequantize the
            # input
            "DequantSoftmax": ("f", False, 1.0),
            # Datatype of softmax normalization before applying activation or
            # type cast. This is called Acc to stick to the naming scheme of the
            # MatMul operators before.
            #   Note: Currently this is ALWAYS floats
            "AccASoftmax": ("s", False, "FLOAT32"),
            # Activation function type of the softmax normalization of the
            # attention weights
            "ActASoftmax": ("s", False, "none", {"none", "thresholds"}),
            # Output bias to be applied to the thresholding activation following
            # the softmax normalization of the attention weights
            "BiasActASoftmax": ("f", False, 0.0),

            # Mode used for providing the attention mask: There can be no mask,
            # a mask sent as the fourth dynamic input, a mask provided as fourth
            # constant input or a causal attention mask which is generated by
            # the operator itself.
            "mask_mode": (
                "s", True, "none", {"none", "input", "const", "causal"}
            ),

            # Possible execution modes for simulating this node
            #   Note: Override to support python mode
            "exec_mode": (
                "s", False, "python", {"", "rtlsim", "cppsim", "python"}
            ),

            # FPGA resource type for memories/internal buffers of the operator
            # Note: Currently only used for StreamTile buffers
            "ram_style": (
                "s", False, "auto", {"auto", "block", "distributed", "ultra"}
            ),
            # FPGA resource type for memories of the thresholds parameters
            # Note: Not yet used...
            "ram_style_thresholds": (
                "s", False, "auto", {"auto", "block", "distributed"}
            ),
            # FPGA resource type for memories of the attention mask if the
            # mask_mode is "const"
            "ram_style_mask": (
                "s", False, "auto", {"auto", "block", "distributed"}
            ),
            # FPGA resource type to implement the MAC operations of the two
            # internal matmul operations
            "mac_resource": ("s", False, "auto", {"auto", "lut", "dsp"}),

            # Input and output FIFO depths for multi-I/O nodes
            #   Note: Need to override here as there are three inputs
            "inFIFODepths": ("ints", False, [2, 2, 2]),
            "outFIFODepths": ("ints", False, [2]),
        })
        # Return updated attribute dictionary
        return attrs

    # Shape configuration of the operator
    @property
    def shapes(self) -> tuple[int, int, int, int]:
        """Return the (QKDim, QLen, VDim, KVLen) shape configuration."""
        # Note: This matches the order of definition above and the order of the
        # HLS lib template as well
        return cast("tuple[int, int, int, int]", (
            self.get_nodeattr("QKDim"), self.get_nodeattr("QLen"),
            self.get_nodeattr("VDim"), self.get_nodeattr("KVLen")
        ))

    # Folding configuration of the operator
    @property
    def folds(self) -> tuple[int, int]:
        """Return the (EmbFold, SeqFold) folding configuration."""
        # Note: This matches the order of definition above and the order of the
        # HLS lib template as well
        return cast("tuple[int, int]", (
            self.get_nodeattr("EmbFold"), self.get_nodeattr("SeqFold")
        ))

    # Tests whether the given folding is a valid configuration with respect to
    # the shape configuration
    @property
    def is_valid_folding(self) -> bool:
        """Return whether the folding evenly divides the folded dimensions."""
        # Get and unpack the shape attributes (except the q matrix length, which
        # is never folded)
        qkdim, _, vdim, kvlen = self.shapes
        # Get and unpack the folding attributes
        embfold, seqfold = self.folds
        # All shapes must be multiples of their corresponding fold
        return not ((qkdim % embfold) or (vdim % embfold) or (kvlen % seqfold))

    # Number of iterations of the attention operator - this can be both, a batch
    # dimension and repetition over packed attention heads.
    @property
    def iterations(self) -> int:
        """Return the number of attention heads / batch iterations."""
        try:
            return cast("int", self.get_nodeattr("Heads"))
        except AttributeError:
            return 1

    # Returns an ONNX node that has the same shape inference behavior
    def make_shape_compatible_op(self, model: ModelWrapper) -> NodeProto:
        """Return an ONNX node with the same shape inference behavior."""
        # Infer the output shape from the input shapes
        _, q_len, v_dim, _ = self.shapes
        o_shape = (q_len, v_dim)
        # Get the node wrapped by this custom op
        node = self.onnx_node
        # Get the shape of the input tensor for inferring the number of
        # heads and correctly propagating shapes
        shape = model.get_tensor_shape(node.input[0])
        if shape is None:
            raise FINNInternalError(
                f"{node.name}: shape of input {node.input[0]} is not known"
            )
        # Determine the rank of the input tensor to support batched and
        # non-batched inputs
        rank = len(shape)
        # Constant operation producing output of given shape
        #   Note: Rank == 3 allows for propagating yet unrolled multi-attention
        #   heads.
        return super().make_const_shape_op(
            (shape[0], *o_shape) if (rank == 3) else o_shape
        )

    # Infers the output data types and updates the input datatypes of the node
    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer the output datatype and update the input datatypes."""
        # ONNX graph node of the operator
        node = self.onnx_node

        # Get input datatypes from model for query, key, value nodes in order
        q_dtype = model.get_tensor_datatype(node.input[0])
        k_dtype = model.get_tensor_datatype(node.input[1])
        v_dtype = model.get_tensor_datatype(node.input[2])

        # Test for changing query input datatype
        if q_dtype != self.get_nodeattr("QType"):
            # Issue a warning message
            log.warning(
                f"QType changing for {node.name}: "
                f"{self.get_nodeattr('QType')} -> {q_dtype} "
            )
        # Test for changing key input datatype
        if k_dtype != self.get_nodeattr("KType"):
            # Issue a warning message
            log.warning(
                f"KType changing for {node.name}: "
                f"{self.get_nodeattr('KType')} -> {k_dtype} "
            )
        # Test for changing value input datatype
        if v_dtype != self.get_nodeattr("VType"):
            # Issue a warning message
            log.warning(
                f"VType changing for {node.name}: "
                f"{self.get_nodeattr('VType')} -> {v_dtype} "
            )

        # Update the node datatype attributes
        self.set_nodeattr("QType", q_dtype.name)
        self.set_nodeattr("KType", k_dtype.name)
        self.set_nodeattr("VType", v_dtype.name)

        # Attention mask might be provided as an input as well
        if self.get_nodeattr("mask_mode") == "input":
            # Get the datatype attribute of the attention mask
            #   Note: Optional mask will be provided as the fourth input
            mask_dtype = model.get_tensor_datatype(node.input[3])
            # Test for changing mask input datatype
            if mask_dtype != self.get_nodeattr("MType"):
                # Issue a warning message
                log.warning(
                    f"MType changing for {node.name}: "
                    f"{self.get_nodeattr('MType')} -> {mask_dtype} "
                )
            # Update the node datatype attribute of the attention mask
            self.set_nodeattr("MType", mask_dtype.name)

        # Set the model output datatype
        model.set_tensor_datatype(node.output[0], self.get_output_datatype(0))

    # Executes the attention operator in python mode simulation
    def _execute_node_python(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute the attention operator in Python mode simulation."""
        # Get the node wrapped by this custom op
        node = self.onnx_node

        # Read the input from the execution context and reshape to match the
        # expected folding
        q = context[node.input[0]].reshape(self.get_normal_input_shape(ind=0))
        k = context[node.input[1]].reshape(self.get_normal_input_shape(ind=1))
        v = context[node.input[2]].reshape(self.get_normal_input_shape(ind=2))

        # Quantization activation function following the query and key
        # multiplication
        def act_qk_matmul(x: np.ndarray) -> np.ndarray:
            """Apply the optional thresholding activation after Q x K."""
            # Only applies if this is specified as a thresholding activation
            if self.get_nodeattr("ActQKMatMul") == "thresholds":
                # Get the thresholds initializer by name from ordered list of
                # optional inputs
                thresholds = context[
                    self.get_input_name_by_name("thresholds_qk_matmul")
                ]
                # Activation value, i.e., bias applied after thresholding
                # activation
                bias = cast("float", self.get_nodeattr("BiasActQKMatMul"))
                # Applies thresholding activation in python to the input
                return multithreshold(x, thresholds, channels_last=True) + bias
            # If not thresholds, assume identity function
            return x

        # Quantization activation function following the softmax normalization
        def act_a_softmax(x: np.ndarray) -> np.ndarray:
            """Apply the optional thresholding activation after the softmax."""
            # Only applies if this is specified as a thresholding activation
            if self.get_nodeattr("ActASoftmax") == "thresholds":
                # Get the thresholds initializer by name from ordered list of
                # optional inputs
                thresholds = context[
                    self.get_input_name_by_name("thresholds_a_softmax")
                ]
                # Activation value, i.e., bias applied after thresholding
                # activation
                bias = cast("float", self.get_nodeattr("BiasActASoftmax"))
                # Applies thresholding activation in python to the input
                return multithreshold(x, thresholds, channels_last=True) + bias
            # If not thresholds, assume identity function
            return x

        # Quantization activation function following the attention and values
        # multiplication
        def act_av_matmul(x: np.ndarray) -> np.ndarray:
            """Apply the optional thresholding activation after A x V."""
            # Only applies if this is specified as a thresholding activation
            if self.get_nodeattr("ActAVMatMul") == "thresholds":
                # Get the thresholds initializer by name from ordered list of
                # optional inputs
                thresholds = context[
                    self.get_input_name_by_name("thresholds_av_matmul")
                ]
                # Activation value, i.e., bias applied after thresholding
                # activation
                bias = cast("float", self.get_nodeattr("BiasActAVMatMul"))
                # Applies thresholding activation in python to the input
                return multithreshold(x, thresholds, channels_last=True) + bias
            # If not thresholds, assume identity function
            return x

        # Scale used to dequantize the qk matrix before computing the softmax in
        # floating point
        dequant = cast("float", self.get_nodeattr("DequantSoftmax"))

        # 1. Queries and keys multiplication followed by quantizing activation
        # function
        qk = act_qk_matmul(np.matmul(q, k.swapaxes(-2, -1)))

        # Load or create the attention mask for mutually exclusive mask modes

        # There might be no attention mask
        if self.get_nodeattr("mask_mode") == "none":
            # No mask can be realized by adding zero, which does not change
            # anything
            mask = 0
        # There might eb a causal attention mask
        elif self.get_nodeattr("mask_mode") == "causal":
            # A causal mask does not need to be stored and can be generated on
            # the fly
            mask = np.triu(-np.inf * np.ones_like(qk), 1)
        # There might be a constant initializer attention mask
        elif self.get_nodeattr("mask_mode") == "const":
            # Load the mask initializer from the execution context
            mask = context[
                self.get_input_name_by_name("M")
            ]
            # The attention operator represents attention masks as binary masks,
            # but the numpy simulation requires floats with 0 and -inf
            mask = np.where(mask, -np.inf * np.ones_like(mask), 0)
        # The attention mask might be streamed in as the third input
        elif self.get_nodeattr("mask_mode") == "input":
            # Load the mask input from the execution context
            mask = context[
                self.get_input_name_by_name("M")
            ]
            # The attention operator represents attention masks as binary masks,
            # but the numpy simulation requires floats with 0 and -inf
            mask = np.where(mask, -np.inf * np.ones_like(mask), 0)
        # All other mask modes are not supported
        else:
            raise NotImplementedError(
                f"Mask Mode {self.get_nodeattr('mask_mode')} is not implemented"
            )
        # Softmax-normalization of the attention weights followed by quantizing
        # activation function
        a = act_a_softmax(
            # Note: Reshape after masking, as the mask might broadcast messing
            # with the shape
            softmax((dequant * qk + mask).reshape(qk.shape), axis=-1)
        )
        # 2. Attention weights and values matmul followed by quantization
        # activation function
        out = act_av_matmul(np.matmul(a, v))

        # Insert the results into the execution context
        context[self.onnx_node.output[0]] = out.reshape(
            self.get_normal_output_shape(ind=0)
        )

    # Executes the attention operator in C++ mode simulation
    def _execute_node_cppsim(
        self, context: dict[str, np.ndarray], graph: "GraphProto"
    ) -> None:
        """Execute the attention operator in C++ mode simulation."""
        # C++ Simulation needs to be implemented in HLS backend specialization
        raise NotImplementedError(
            f"exec_mode cppsim of {self.__class__.__name__} is not implemented!"
        )

    # Executes the attention operator in RTL mode simulation
    def _execute_node_rtlsim(
        self, context: dict[str, np.ndarray], graph: "GraphProto"
    ) -> None:
        """Execute the attention operator in RTL mode simulation."""
        # RTL Simulation needs to be implemented in backend specialization
        raise NotImplementedError(
            f"exec_mode rtlsim of {self.__class__.__name__} is not implemented!"
        )

    # Executes the attention operator in simulation (either python, c++ or rtl)
    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"
    ) -> None:
        """Execute the attention operator in the configured simulation mode."""
        # Get the configured execution mode
        mode = self.get_nodeattr("exec_mode")
        if mode == "python":
            self._execute_node_python(context, graph)
        else:
            # Delegate execution to parent class for cppsim and rtlsim
            #   Note: Only reachable on the _hls/_rtl specializations, which do
            #   inherit HLSBackend
            HLSBackend.execute_node(cast("HLSBackend", self), context, graph)

    # Optional node verification
    def verify_node(self) -> list:
        """Verify node attribute/input/output correctness."""
        # TODO: Implement
        return []

    # Gets the datatype of input at index ind
    def get_input_datatype(self, ind: int = 0) -> BaseDataType:
        """Get the datatype of the input at index ind."""
        # Ordered list of names of allowed inputs
        inputs = ["QType", "KType", "VType"]

        # If the attention mask is provided as input, it has a type as well
        if self.get_nodeattr("mask_mode") == "input":
            # The mask type is an attribute itself
            inputs += ["MType"]

        # TODO: All the following types are probably never requested, they are
        #  implemented for the sake of completeness for now. If they are ever
        #  actually required, check whether the following defaults and dummies
        #  actually still make sense.

        # If there is a thresholding activation for the first matmul, it will
        # have a type as well
        if self.get_nodeattr("ActQKMatMul") == "thresholds":
            # The thresholds will always be of the accumulator type as the
            # activation maps from AccQKMatMul to OutQKMatMul
            inputs += ["AccQKMatMul"]

        # If there is a thresholding activation for the softmax normalization,
        # it will have a type as well
        if self.get_nodeattr("ActASoftmax") == "thresholds":
            # While there is a dummy configurable attribute describing the
            # threshold type of the softmax, these are currently always floats
            inputs += ["AccASoftmax"]

        # If there is a thresholding activation for the second matmul, it will
        # have a type as well
        if self.get_nodeattr("ActAVMatMul") == "thresholds":
            # The thresholds will always be of the accumulator type as the
            # activation maps from AccAVMatMul to OutAVMatMul
            inputs += ["AccAVMatMul"]

        # Look up datatype name in attributes and convert to DataType
        return DataType[cast("str", self.get_nodeattr(f"{inputs[ind]}"))]

    # Gets the datatype of the output (at index ind, but there is just one)
    def get_output_datatype(self, ind: int = 0) -> BaseDataType:
        """Get the datatype of the output at index ind (there is just one)."""
        # Ordered list of names of allowed outputs
        outputs = ["O"]
        # Look up datatype name in attributes and convert to DataType
        return DataType[
            cast("str", self.get_nodeattr(f"{outputs[ind]}Type"))
        ]

    # Gets the shape of the input at index ind without folding
    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Get the shape of the input at index ind without folding."""
        # Unpack the shape configuration
        qk_dim, q_len, v_dim, kv_len = self.shapes
        # List shapes of inputs in order
        inputs_shapes: list[tuple[int, ...]] = [
            # Query input sequence
            (self.iterations, q_len, qk_dim),
            # Key input sequence
            (self.iterations, kv_len, qk_dim),
            # Value input sequence
            (self.iterations, kv_len, v_dim),
        ]

        # If the attention mask is provided as input, it has a shape as well
        if self.get_nodeattr("mask_mode") in {"input", "const"}:
            # Mask shape is inferred from query and key sequence lengths
            inputs_shapes += [(q_len, kv_len)]

        # TODO: All the following shapes are probably never requested, they are
        #  implemented for the sake of completeness for now. If they are ever
        #  actually required, remember to insert meaningful shapes.

        # If there is a thresholding activation for the first matmul, these will
        # be the next input index after the (optional) mask
        if self.get_nodeattr("ActQKMatMul") == "thresholds":
            # TODO: This is just a dummy shape
            inputs_shapes += [(0, 0)]

        # If there is a thresholding activation for the softmax normalization,
        # these will be the next (and last) input index after the (optional)
        # second thresholds
        if self.get_nodeattr("ActASoftmax") == "thresholds":
            # TODO: This is just a dummy shape
            inputs_shapes += [(0, 0)]

        # If there is a thresholding activation for the second matmul, these
        # will be the next input index after the (optional) first thresholds
        if self.get_nodeattr("ActAVMatMul") == "thresholds":
            # TODO: This is just a dummy shape
            inputs_shapes += [(0, 0)]

        # Get the shape by indexing into the ordered list of all inputs
        return inputs_shapes[ind]

    # Gets the shape of the output at index ind (there is just one) without
    # folding
    def get_normal_output_shape(
        self, ind: int = 0  # noqa: ARG002
    ) -> tuple[int, int, int]:
        """Get the shape of the output without folding (there is just one)."""
        # The output shape is inferred from the length of the query sequence and
        # the embedding dimension of the values
        _, q_len, v_dim, _ = self.shapes
        return (self.iterations, q_len, v_dim)

    # Gets the shape of the attention weights at index ind (there is just one)
    # without folding
    def get_normal_attention_shape(
        self, ind: int = 0  # noqa: ARG002
    ) -> tuple[int, int, int]:
        """Get the shape of the attention weights without folding."""
        # The attention weights have shape covering both sequence dimensions
        _, q_len, _, kv_len = self.shapes
        return (self.iterations, q_len, kv_len)

    # Gets the shape of the input at index ind with folding
    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Get the shape of the input at index ind with folding."""
        # Get the unfolded size of the input
        *num, idim = self.get_normal_input_shape(ind)
        # Get the folding configuration specifying the amount of parallelism
        embfold, seqfold = self.folds

        # Queries, keys and values are all folded similarly along the embedding
        # dimension
        if ind in (0, 1, 2):
            # Note: Embedding dimension is always assumed to be the second
            # dimension, any transpose is handled implicitly by the operator
            return *num, embfold, idim // embfold

        # If the mask is provided as input, it is folded along the second
        # sequence dimension
        if ind == 3 and self.get_nodeattr("mask_mode") in {"input", "const"}:
            # Note: Both dimensions are sequence dimension, the second
            # corresponds to the KVLen
            return *num, seqfold, idim // seqfold

        # If this point is reached, probably something went wrong
        # TODO: Requesting the folded shape of thresholds will reach here. Can
        #  this actually happen? Probably it is indeed an error, there should be
        #  no reason to ask for the shape of the thresholds, just ask for the
        #  initializer and get its shape? Folding of the thresholds behaves
        #  differently and would require to actually keep track of mapping
        #  indices to optional inputs to correctly associate the folding
        #  dimensions.
        # TODO: This is just a dummy shape
        return *num, idim, 1

    # Gets the shape of the output at index ind (there is just one) with folding
    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Get the shape of the output at index ind with folding."""
        # Get the unfolded size of the output
        *num, odim = self.get_normal_output_shape(ind)
        # Get the folding configuration specifying the amount of parallelism
        embfold, _seqfold = self.folds
        # The output is always folded along the embedding dimension, which is
        # assumed to be the second dimension
        return *num, embfold, odim // embfold

    # Gets the shape of the attention weights at index ind (there is just one)
    # with folding
    def get_folded_attention_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Get the shape of the attention weights at index ind with folding."""
        # Get the unfolded size of the attention weights
        *num, adim = self.get_normal_attention_shape(ind)
        # Get the folding configuration specifying the amount of parallelism
        _embfold, seqfold = self.folds
        # The attention weights are always folded along the sequence dimension,
        # which is assumed to be the second dimension
        return *num, seqfold, adim // seqfold

    # Widths of the input data stream of the input at index ind
    def get_instream_width(self, ind: int = 0) -> int:
        """Get the width of the input data stream at index ind."""
        if ind in (0, 1, 2):
            # Get the number of bits used to represent the input
            i_bits = self.get_input_datatype(ind).bitwidth()
            # Parallelism is the number of elements in the last dimension of the
            # folded input
            *_, elems = self.get_folded_input_shape(ind)
            # Width of a stream receiving input elements in parallel
            return elems * i_bits

        if ind == 3 and self.get_nodeattr("mask_mode") == "input":
            mask_bits = self.get_input_datatype(ind).bitwidth()
            *_, elems = self.get_folded_input_shape(ind)
            return elems * mask_bits

        # Any further inputs are either a fixed mask or fixed thresholds
        return 0  # 0 = not exposed as stream

    # Widths of the output data stream of the output at index ind
    def get_outstream_width(self, ind: int = 0) -> int:
        """Get the width of the output data stream at index ind."""
        # Get the number of bits used to represent the output
        o_bits = self.get_output_datatype(ind).bitwidth()
        # Parallelism is the number of elements in the last dimension of the
        # folded output
        *_, elems = self.get_folded_output_shape(ind)
        # Width of a stream producing output elements in parallel
        return elems * o_bits

    # Minimize the accumulator bit width
    def minimize_accumulator_width(
        self, model: ModelWrapper  # noqa: ARG002
    ) -> None:
        """Minimize the accumulator bit widths of the internal matmuls."""
        # Get the query, key, value and attention weights type
        QType = DataType[cast("str", self.get_nodeattr("QType"))]  # noqa: N806
        KType = DataType[cast("str", self.get_nodeattr("KType"))]  # noqa: N806
        VType = DataType[cast("str", self.get_nodeattr("VType"))]  # noqa: N806
        AType = DataType[cast("str", self.get_nodeattr("AType"))]  # noqa: N806

        # Compute the worst-case upper and lower bounds of the accumulator range
        lower_worst = QType.min() * np.ones(self.get_normal_input_shape(0)[-2:])
        lower_range = calculate_matvec_accumulator_range(lower_worst, KType)
        upper_worst = QType.max() * np.ones(self.get_normal_input_shape(0)[-2:])
        upper_range = calculate_matvec_accumulator_range(
            upper_worst, KType
        )
        # Minimum and maximum values of the range
        acc_min = min(min(lower_range), min(upper_range))
        acc_max = max(max(upper_range), max(upper_range))
        # Unsigned accumulator range
        if acc_min >= 0:
            # Number of bits necessary to represent the maximum value of the
            # range. Some values between 0 and acc_min might be unused.
            bitwidth = math.ceil(np.log2(acc_max + 1))
            # New unsigned accumulator datatype of this bitwidth
            AccQKMatMul = DataType[f"UINT{bitwidth}"]  # noqa: N806
        # Signed accumulator range
        else:
            # Maximum absolute value which needs to be represented
            acc_max = max(-acc_min, 1 + acc_max)
            # Number of bits necessary to represent the maximum value of the
            # range. Some values on one of the ends might remain unused.
            bitwidth = math.ceil(np.log2(acc_max) + 1)
            # New signed accumulator datatype of this bitwidth
            AccQKMatMul = DataType[f"INT{bitwidth}"]  # noqa: N806
        # Update the accumulator datatype attribute
        self.set_nodeattr("AccQKMatMul", AccQKMatMul.name)
        # If there is no activation function following the accumulator, the
        # output type needs to be adjusted as well
        if self.get_nodeattr("ActQKMatMul") == "none":
            # Update the output datatype attribute to the same type as the
            # accumulator
            self.set_nodeattr("OutQKMatMul", AccQKMatMul.name)

        # Compute the worst-case upper and lower bounds of the accumulator range
        lower_worst = AType.min() * np.ones(
            self.get_normal_attention_shape(0)[-2:])
        lower_range = calculate_matvec_accumulator_range(lower_worst, VType)
        upper_worst = AType.max() * np.ones(
            self.get_normal_attention_shape(0)[-2:])
        upper_range = calculate_matvec_accumulator_range(
            upper_worst, VType
        )
        # Minimum and maximum values of the range
        acc_min = min(min(lower_range), min(upper_range))
        acc_max = max(max(upper_range), max(upper_range))
        # Unsigned accumulator range
        if acc_min >= 0:
            # Number of bits necessary to represent the maximum value of the
            # range. Some values between 0 and acc_min might be unused.
            bitwidth = math.ceil(np.log2(acc_max + 1))
            # New unsigned accumulator datatype of this bitwidth
            AccAVMatMul = DataType[f"UINT{bitwidth}"]  # noqa: N806
        # Signed accumulator range
        else:
            # Maximum absolute value which needs to be represented
            acc_max = max(-acc_min, 1 + acc_max)
            # Number of bits necessary to represent the maximum value of the
            # range. Some values on one of the ends might remain unused.
            bitwidth = math.ceil(np.log2(acc_max) + 1)
            # New signed accumulator datatype of this bitwidth
            AccAVMatMul = DataType[f"INT{bitwidth}"]  # noqa: N806
        # Update the accumulator datatype attribute
        self.set_nodeattr("AccAVMatMul", AccAVMatMul.name)
        # If there is no activation function following the accumulator, the
        # output type needs to be adjusted as well
        if self.get_nodeattr("ActAVMatMul") == "none":
            # Update the output datatype attribute to the same type as the
            # accumulator
            self.set_nodeattr("OutAVMatMul", AccAVMatMul.name)
            # # The output type of the whole operator is the same as the output
            # # type of the last MatMul
            # TODO: This currently breaks MergeMultiHeads via
            #  MinimizeAccumulatorWidth, which re-infers datatypes after
            #  each custom op instead of once after traversing the whole graph.
            # self.set_nodeattr("OType", AccQKMatMul.name)

    # Gets the number of expected input values, i.e. how many times read()
    # could/should be called on the input stream of this operator
    def get_number_input_values(self, ind: int = 0) -> int:
        """Get the number of expected input values for the input at index ind."""
        # Elements over all but the last dimension of the input folded along
        # the embedding dimension
        return int(np.prod(self.get_folded_input_shape(ind=ind)[:-1]))

    # Gets the number of expected output values, i.e. how many times read()
    # could/should be called on the output stream of this operator
    def get_number_output_values(self) -> int:
        """Get the number of expected output values on the output stream."""
        # Elements over all but the last dimension of the output folded along
        # the embedding dimension
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    # Converts names of optional inputs to the node input index and from there
    # to the ONNX node input name if the input is present.
    #   Note: This mapping is required as the ONNX graph/node may provide
    #   different names (in particular automatically generated unique names) and
    #   some of these are optional inputs.
    def get_input_name_by_name(self, name: str) -> str:
        """Resolve an optional input's logical name to its ONNX tensor name."""
        # Ordered names of the (optional) threshold inputs
        thresholds = [
            "thresholds_qk_matmul",
            "thresholds_a_softmax",
            "thresholds_av_matmul",
        ]

        # Ordered names of primary query, key, value inputs and optional mask
        # and threshold inputs.
        inputs = ["Q", "K", "V", "M", *thresholds]

        # Specify for each input whether it is present or not
        inputs_present = [
            # Note: Primary inputs are always present, the mask is present in
            # "input" or "const" mask mode
            True, True, True, self.get_nodeattr("mask_mode") in {
                "input", "const"
            },
        ]

        # Thresholds are present if the activation function is set to
        # thresholds
        inputs_present.extend([
            self.get_nodeattr("ActQKMatMul") == "thresholds",
            self.get_nodeattr("ActASoftmax") == "thresholds",
            self.get_nodeattr("ActAVMatMul") == "thresholds"
        ])

        # Filter the ordered list of input names for those which are actually
        # present
        inputs = [
            x for x, present in zip(inputs, inputs_present, strict=True)
            if present
        ]

        # Find the position of the requested input name and look up the
        # corresponding input name of the ONNX node
        return self.onnx_node.input[inputs.index(name)]

    # Derives the expected cycles for the attention operation given the folding
    # configuration
    def get_exp_cycles(self) -> int:
        """Derive the expected cycles given the folding configuration."""
        # Verify the folding configuration
        if not self.is_valid_folding:
            raise FINNUserError(
                f"Invalid folding configuration for {self.onnx_node.name}"
            )
        # Get the input/output dimensions
        _qk_dim, q_len, _v_dim, kv_len = self.shapes
        # Get folding configuration describing how to parallelize along the
        # dimensions
        emb_fold, seq_fold = self.folds
        # Assume perfect overlap of the constituents of the operator, i.e., of
        # the buffering, both matmul and the softmax, then the expected cycles
        # is the maximum over these operators
        #   Overall worst case cycles without any parallelization: ~ T x T x d
        return self.iterations * max(
            # Transposed keys buffer cycles
            #   Worst case: kv_len * qk_dim, ~ T x d
            kv_len * emb_fold,
            # Queries - keys matmul cycles
            #   Worst case: q_len * qk_dim * kv_len, ~ T x T x d
            q_len * emb_fold * seq_fold,
            # Softmax normalization cycles
            #   Worst case: q_len * kv_len, ~ T x T
            q_len * seq_fold,
            # Values buffer cycles
            #   Worst case: kv_len * v_dim, ~ T x d
            kv_len * emb_fold,
            # Attention weights - values matmul
            #   Worst case: q_len * v_dim * kv_len, ~ T x T x d
            q_len * emb_fold * seq_fold
        )

"""Squeeze hardware custom operator (ONNX ``Squeeze``, folding-aware passthrough)."""

import copy
import numpy as np
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow import register_custom_op
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.exception import FINNInternalError
from finn.util.logging import log

if TYPE_CHECKING:
    from onnx import GraphProto

# Type of the dictionary returned by get_nodeattr_types: maps attribute names to
# their (dtype, required, default[, allowed_values]) specification tuples
NodeAttrTypes = dict[
    str,
    tuple[str, bool, int | float | str | bool | np.ndarray | list]
    | tuple[str, bool, int | float | str | bool | np.ndarray | list, set | None],
]


@register_custom_op
class Squeeze(HWCustomOp):
    """Hardware custom operator for the Squeeze operation.

    Removes single-dimension entries from the shape of a tensor.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize the Squeeze operator from an ONNX node."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return the dictionary of node attributes for the Squeeze operator."""
        # Note: the ``axes`` attribute has a ``None`` default, which the
        # NodeAttrTypes alias does not model; hence the cast on return.
        attrs = {
            # Axes to be squeezed can be given as an attribute for opset < 13
            "axes": ("ints", False, None),
            # Data type of the input elements
            "inp_dtype": ("s", True, ""),
            # Data type of the output elements
            "out_dtype": ("s", True, ""),
            # Shape of the input
            "inp_shape": ("ints", True, [1]),
            # Shape of the output
            "out_shape": ("ints", True, [1]),
            # Number of elements in the last dimensions processed in parallel
            "PE": ("i", False, 1),
            # Possible execution modes for simulating this node
            #   Note: Override to support python mode
            "exec_mode": ("s", False, "python", {"", "rtlsim", "cppsim", "python"}),
        }
        attrs.update(HWCustomOp.get_nodeattr_types(self))
        return cast("NodeAttrTypes", attrs)

    @property
    def inp_dtype(self) -> BaseDataType:
        """Return the input datatype."""
        return DataType[cast("str", self.get_nodeattr("inp_dtype"))]

    @property
    def out_dtype(self) -> BaseDataType:
        """Return the output datatype."""
        return DataType[cast("str", self.get_nodeattr("out_dtype"))]

    @property
    def inp_shape(self) -> list[int]:
        """Return the input shape."""
        return list(cast("list[int]", self.get_nodeattr("inp_shape")))

    @property
    def out_shape(self) -> list[int]:
        """Return the output shape."""
        return list(cast("list[int]", self.get_nodeattr("out_shape")))

    @property
    def pe(self) -> int:
        """Return the number of parallel processing elements (PE)."""
        return cast("int", self.get_nodeattr("PE"))

    def make_shape_compatible_op(self, model: ModelWrapper) -> NodeProto:  # noqa: ARG002
        """Create a shape-compatible operation for ONNX shape inference.

        Returns a standard ONNX Squeeze node for shape inference purposes.
        """
        node = copy.deepcopy(self.onnx_node)
        # Though providing squeezed axes via a second input is supported by the
        # implementation, the inferred shapes might be incorrect if this is
        # truly a dynamic list of axes changing at runtime.
        if len(node.input) > 1:
            log.warning(
                f"{node.name}: providing dimensions to squeeze as an input "
                f"might invalidate shape inference if these are not constant."
            )
        # Transplant this operator back into the standard ONNX domain
        node.domain = ""
        return node

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer and set the datatype of the node output."""
        node = self.onnx_node
        # Test for changing input datatype
        if model.get_tensor_datatype(node.input[0]) != self.inp_dtype:
            new_dtype = model.get_tensor_datatype(node.input[0])
            log.warning(f"{node.name}: inp_dtype changing from {self.inp_dtype} to {new_dtype}")
            self.set_nodeattr("inp_dtype", new_dtype.name)
        # Though providing squeezed axes via a second input is supported by the
        # implementation, the datatype of this input is ignored here
        if len(node.input) > 1:
            log.warning(
                f"{node.name}: providing dimensions to squeeze as an input "
                f"will be ignored by datatype inference."
            )
        # Make sure the output always has the same type as the input
        if self.out_dtype != self.inp_dtype:
            log.warning(
                f"{node.name}: out_dtype changing from {self.out_dtype} to {self.inp_dtype}"
            )
            self.set_nodeattr("out_dtype", self.inp_dtype.name)
        # Force the output data type stored as a node attribute
        model.set_tensor_datatype(node.output[0], self.out_dtype)

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute the squeeze operation (Python fallback)."""
        node = self.onnx_node
        inp = context[node.input[0]]
        # Try with axes specified as attribute first
        axes = self.get_nodeattr("axes")
        # If there are no axes specified via attribute but there is a second
        # input to the operator, this input specifies the axes to be squeezed
        if axes is None and len(node.input) > 1:
            axes = context[node.input[1]]
        axis = tuple(cast("list[int] | np.ndarray", axes)) if axes is not None else None
        out = np.squeeze(inp, axis=axis)
        # Always use float32 as the container type
        context[node.output[0]] = out.astype(np.float32)

    def verify_node(self) -> list[str]:
        """Verify the node attributes, inputs and outputs."""
        # TODO: Implement
        return []

    # Note: End of QONNX CustomOp region, below is FINN HWCustomOp stuff

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return the datatype of the input at the given index."""
        # There is only one proper input (the optional axes input is ignored)
        return self.inp_dtype

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return the datatype of the output at the given index."""
        return self.out_dtype

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...] | list[int]:
        """Return the unfolded input shape at the given index."""
        # Infer shape of axes input
        if ind == 1:
            axes = self.get_nodeattr("axes")
            if axes is None:
                raise FINNInternalError(
                    f"{self.onnx_node.name}: axes input requested but no axes attribute is set"
                )
            return (len(cast("list[int]", axes)),)
        # Data input
        return self.inp_shape

    def get_normal_output_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Return the unfolded output shape at the given index."""
        return self.out_shape

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return the folded input shape at the given index.

        Applies PE-based folding to the last dimension.
        """
        # Axes input
        if ind == 1:
            return tuple(self.get_normal_input_shape(ind=ind))
        *num_inputs, num_elems = self.get_normal_input_shape(ind=ind)
        if num_elems % self.pe != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: PE ({self.pe}) must divide the last input axis "
                f"({num_elems})"
            )
        return (*num_inputs, num_elems // self.pe, self.pe)

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return the folded output shape at the given index.

        Applies PE-based folding to the last dimension.
        """
        *num_outputs, num_elems = self.get_normal_output_shape(ind=ind)
        if num_elems % self.pe != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: PE ({self.pe}) must divide the last output axis "
                f"({num_elems})"
            )
        return (*num_outputs, num_elems // self.pe, self.pe)

    def get_instream_width(self, ind: int = 0) -> int:
        """Return the width of the input stream in bits at the given index."""
        # Axes input is not exposed
        if ind == 1:
            return 0
        i_bits = self.get_input_datatype(ind).bitwidth()
        # Parallelism is the number of elements in the last dimension of the folded input
        *_, elems = self.get_folded_input_shape(ind)
        return elems * i_bits

    def get_outstream_width(self, ind: int = 0) -> int:
        """Return the width of the output stream in bits at the given index."""
        o_bits = self.get_output_datatype(ind).bitwidth()
        # Parallelism is the number of elements in the last dimension of the folded output
        *_, elems = self.get_folded_output_shape(ind)
        return elems * o_bits

    def get_number_output_values(self) -> int:
        """Return the number of expected output values from the operator."""
        # Elements over all but the last (parallelized) dimension of the output
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    def get_exp_cycles(self) -> int:
        """Return the expected number of cycles for the squeeze operation."""
        # Number of iterations required to process the whole folded stream
        # (all but the PE, last, parallelized dimension)
        return int(np.prod(self.get_folded_output_shape()[:-1]))

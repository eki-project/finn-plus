"""Reshape hardware custom operator (ONNX ``Reshape``, folding-aware passthrough)."""

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
class Reshape(HWCustomOp):
    """Reshape operator, essentially a passthrough with different input/output shape."""

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return custom node attributes with their types and default values."""
        attrs: NodeAttrTypes = {
            # Shape of the input
            "inp_shape": ("ints", True, [1]),
            # Shape of the output
            "out_shape": ("ints", True, [1]),
            # Datatype of input and output elements
            "dtype": ("s", True, ""),
            # Number of parallel elements in the last dimension of the output
            "PE": ("i", False, 1),
        }
        attrs.update(HWCustomOp.get_nodeattr_types(self))
        return attrs

    @property
    def inp_shape(self) -> list[int]:
        """Input shape attribute."""
        return list(cast("list[int]", self.get_nodeattr("inp_shape")))

    @property
    def out_shape(self) -> list[int]:
        """Output shape attribute."""
        return list(cast("list[int]", self.get_nodeattr("out_shape")))

    @property
    def dtype(self) -> BaseDataType:
        """Datatype attribute as a QONNX DataType."""
        return DataType[cast("str", self.get_nodeattr("dtype"))]

    @property
    def pe(self) -> int:
        """Parallel elements in the last dimension of the output."""
        return cast("int", self.get_nodeattr("PE"))

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Datatype of the input tensor, same as the output."""
        return self.dtype

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Datatype of the output tensor, same as the input."""
        return self.dtype

    def get_normal_input_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Regular input shape as seen by the ONNX standard."""
        return self.inp_shape

    def get_normal_output_shape(self, ind: int = 0) -> list[int]:  # noqa: ARG002
        """Regular output shape as seen by the ONNX standard."""
        return self.out_shape

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Shape of the folded (PE) input tensor."""
        *num_inputs, num_elems = self.get_normal_input_shape(ind=ind)
        if num_elems % self.pe != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: PE ({self.pe}) must divide the last input axis "
                f"({num_elems})"
            )
        # Folding along the last dimension
        return (*num_inputs, num_elems // self.pe, self.pe)

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Shape of the folded (PE) output tensor."""
        *num_outputs, num_elems = self.get_normal_output_shape(ind=ind)
        if num_elems % self.pe != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: PE ({self.pe}) must divide the last output axis "
                f"({num_elems})"
            )
        # Folding along the last dimension
        return (*num_outputs, num_elems // self.pe, self.pe)

    def get_instream_width(self, ind: int = 0) -> int:
        """Width of the input data stream of the input at index ``ind``."""
        i_bits = self.get_input_datatype(ind).bitwidth()
        # Parallelism is the number of elements in the last dimension of the folded input
        *_, elems = self.get_folded_input_shape(ind)
        return elems * i_bits

    def get_outstream_width(self, ind: int = 0) -> int:
        """Width of the output data stream of the output at index ``ind``."""
        o_bits = self.get_output_datatype(ind).bitwidth()
        # Parallelism is the number of elements in the last dimension of the folded output
        *_, elems = self.get_folded_output_shape(ind)
        return elems * o_bits

    def get_number_output_values(self) -> int:
        """Return the number of expected output values given the folding."""
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    def get_exp_cycles(self) -> int:
        """Return the expected cycle count given the folding."""
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer the datatype of the node output from the model graph."""
        node = self.onnx_node
        # Test for changing input datatype
        if model.get_tensor_datatype(node.input[0]) != self.dtype:
            new_dtype = model.get_tensor_datatype(node.input[0])
            log.warning(f"{node.name}: inp_dtype changing from {self.dtype} to {new_dtype}")
            self.set_nodeattr("dtype", new_dtype.name)
        # Force the output data type stored as a node attribute
        model.set_tensor_datatype(node.output[0], self.dtype)

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute reshape operation (Python fallback)."""
        node = self.onnx_node
        inp = context[node.input[0]]
        out = np.reshape(inp, self.out_shape)
        # Always use float32 as the container type
        context[node.output[0]] = out.astype(np.float32)

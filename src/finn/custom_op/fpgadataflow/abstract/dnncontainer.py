"""Custom op for wrapping a sub-graph (DNN) as a container node."""

import numpy as np
import numpy.typing as npt
from onnx import GraphProto, NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import get_by_name, qonnx_make_model
from typing import cast

from finn.core.onnx_exec import execute_onnx
from finn.custom_op.fpgadataflow import register_custom_op
from finn.util.exception import FINNInternalError, FINNUserError

# Value types accepted/returned by the base ``get_nodeattr``/``set_nodeattr``.
BaseNodeAttrValue = int | float | str | bool | npt.NDArray | list[str | int | float] | None
# ``set_nodeattr`` on this op additionally accepts a graph value for the ``body``
# attribute (as a ``ModelWrapper`` or a raw ``GraphProto``).
SetNodeAttrValue = ModelWrapper | GraphProto | BaseNodeAttrValue
# Shape of the dict returned by ``get_nodeattr_types``: attribute name ->
# (dtype, required, default[, allowed_values]). Must match the base class.
NodeAttrTypes = dict[
    str,
    tuple[str, bool, int | float | str | bool | npt.NDArray | list]
    | tuple[str, bool, int | float | str | bool | npt.NDArray | list, set | None],
]


@register_custom_op
class DNNContainer(CustomOp):
    """ONNX custom op that encapsulates an entire DNN subgraph as one node."""

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return attribute type definitions for DNNContainer."""
        my_attrs: NodeAttrTypes = {
            "body": ("g", True, ""),
            "io_map": ("s", True, "{}"),
        }
        return my_attrs

    def get_nodeattr(self, name: str) -> BaseNodeAttrValue:
        """Get a node attribute by name, handling graph-type attributes.

        Note: the graph-typed ``body`` attribute is returned as a
        ``ModelWrapper`` (use the :attr:`body` property for a typed accessor).
        The return type is kept identical to the base method; callers that need
        the ``ModelWrapper`` cast it themselves.
        """
        # Copied from FinnLoop OP
        try:
            (dtype, req, def_val, _allowed_values) = self.get_nodeattr_def(name)
            attr = get_by_name(self.onnx_node.attribute, name)
            if attr is not None:
                # dtype indicates which ONNX Attribute member to use
                # g : graph
                if dtype == "g":
                    ret = attr.__getattribute__(dtype)
                    return cast("BaseNodeAttrValue", ModelWrapper(qonnx_make_model(ret)))
                return super().get_nodeattr(name)
            if req:
                raise FINNUserError(
                    f"Required attribute {name} unspecified in a {self.onnx_node.op_type} node"
                )
            # not set, return default value
            return def_val
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name) from None

    def set_nodeattr(self, name: str, value: SetNodeAttrValue) -> None:
        """Set a node attribute by name, handling graph-type attributes.

        In addition to the base value types, the graph-typed ``body`` attribute
        may be set from a ``ModelWrapper`` or a raw ``GraphProto``.
        """
        # Copied from FinnLoop OP
        try:
            (dtype, _req, _def_val, _allowed_values) = self.get_nodeattr_def(name)
            attr = get_by_name(self.onnx_node.attribute, name)
            if attr is not None and dtype == "g":
                # dtype indicates which ONNX Attribute member to use
                # g : graph
                if isinstance(value, ModelWrapper):
                    value = value.model.graph
                if not isinstance(value, GraphProto):
                    raise FINNInternalError(
                        "Value for graph attribute must be a GraphProto or ModelWrapper"
                    )
                attr.g.CopyFrom(value)
            else:
                super().set_nodeattr(name, cast("BaseNodeAttrValue", value))
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name) from None

    @property
    def body(self) -> ModelWrapper:
        """Return the contained DNN model (the ``body`` graph attribute)."""
        return cast("ModelWrapper", self.get_nodeattr("body"))

    @property
    def io_map(self) -> str:
        """Return the JSON-encoded mapping between container and subgraph I/O."""
        return cast("str", self.get_nodeattr("io_map"))

    def make_shape_compatible_op(self, model: ModelWrapper) -> NodeProto:  # noqa: ARG002
        """Not supported - DNNContainer is a container node."""
        raise FINNInternalError(
            f"{self.onnx_node.name}: shape inference is not defined for "
            f"DNNContainer container nodes"
        )

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer output datatype (not applicable for DNNContainer)."""

    def execute_node(
        self, context: dict[str, np.ndarray], graph: GraphProto  # noqa: ARG002
    ) -> None:
        """Execute the contained subgraph using the FINN ONNX executor."""
        # Copied from GenericPartition
        # Validate this code
        model = self.body
        return_full_exec_context = True
        node = self.onnx_node
        inp_ctx = dict(filter(lambda x: x[0] in node.input, context.items()))
        # inputs may have been renamed in partition
        for i, old_iname in enumerate(node.input):
            new_iname = model.graph.input[i].name
            if old_iname != new_iname:
                inp_ctx[new_iname] = inp_ctx[old_iname]
                del inp_ctx[old_iname]
        ret = execute_onnx(model, inp_ctx, return_full_exec_context)
        # outputs may have been renamed in partition
        for i, node_oname in enumerate(node.output):
            model_oname = model.graph.output[i].name
            context[node_oname] = ret[model_oname]
        # prefix and insert exec context entries
        if return_full_exec_context:
            for tname in ret:
                if tname not in [x.name for x in model.graph.output]:
                    context[node.name + "_" + tname] = ret[tname]

    def verify_node(self) -> list[str]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Verify that the DNNContainer node has the correct number of attributes.

        Note: the qonnx ``CustomOp.verify_node`` base is annotated ``-> None`` but
        the FINN verification passes expect a list of messages.
        """
        # Copied from GenericPartition
        info_messages = []

        # verify number of attributes
        num_of_attr = 2
        if len(self.onnx_node.attribute) == num_of_attr:
            info_messages.append("The number of attributes is correct")
        else:
            info_messages.append(
                "The number of attributes is incorrect, "
                f"{self.onnx_node.op_type} should have {num_of_attr} attributes"
            )
        # verify that all necessary attributes exist
        try:
            self.get_nodeattr("body")
            self.get_nodeattr("io_map")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                "The necessary attributes do not exist. "
                "DNNContainer needs the following attribute(s): body, io_map"
            )

        return info_messages

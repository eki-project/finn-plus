"""Implementation of the GiveUniqueNodeNamesRecursive transformation,
which assigns unique names to each node in the graph and its subgraphs (e.g., loop bodies)
using enumeration with an optional prefix."""

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.rtl.finn_loop import FINNLoop


class GiveUniqueNodeNamesRecursive(Transformation):
    """Give unique names to each node in the graph using enumeration, starting
    with given prefix (if specified in the constructor)."""

    def __init__(self, prefix: str | None = None) -> None:
        """Initialize the transformation with an optional prefix for node names."""
        super().__init__()
        self.prefix = prefix

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, Literal[False]]:
        """Apply the transformation to the given model and all of its submodels."""
        optype_count = {}
        for n in model.graph.node:
            if n.op_type not in optype_count.keys():
                optype_count[n.op_type] = 0
            n.name = (
                f"{self.prefix}_{n.op_type}_{optype_count[n.op_type]}"
                if self.prefix is not None
                else f"{n.op_type}_{optype_count[n.op_type]}"
            )
            optype_count[n.op_type] += 1
            if n.op_type == "FINNLoop":
                loop_inst = cast("FINNLoop", getCustomOp(n))
                loop_body = cast("ModelWrapper", loop_inst.get_nodeattr("body"))
                loop_body = loop_body.transform(GiveUniqueNodeNamesRecursive(prefix=n.name))
                loop_inst.set_nodeattr("body", loop_body.graph)
        # return model_was_changed = False as single iteration is always enough
        return (model, False)

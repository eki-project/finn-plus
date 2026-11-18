"""Streamlining step of the oracle frontend based on FINN+'s ``StreamlinePlus``.

The builder's default ``step_streamline`` runs the classic ``Streamline`` transformation,
which cannot move a scale factor past a fork or a join and does not lower concat joins. The
candidate networks of the search space have learned per-tensor scales everywhere, so the
exhaustive ``StreamlinePlus`` of the ResNet step library is needed to absorb them. This step
is the default ``step_streamline`` with ``StreamlinePlus`` in place of ``Streamline``; it is
selected with ``OracleConfig.streamline = "plus"`` (the default).
"""

from __future__ import annotations

import hashlib
from qonnx.transformation.base import Transformation
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.composed import ComposedTransformation
from qonnx.transformation.general import RemoveUnusedTensors
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from typing import TYPE_CHECKING, Any

from finn.transformation.streamline import absorb
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC
from finn.transformation.streamline.streamline_plus import StreamlinePlus

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

    from finn.builder.build_dataflow_config import DataflowBuildConfig

#: Upper bound on the number of graph modifications one ``StreamlinePlus`` run may make. The
#: exhaustive loop never terminates on some graphs (two transformations undoing each other, see
#: eki-project/finn-plus#321); a bound turns that into a frontend failure instead of a hang
#: that only the per-candidate timeout ends. Streamlining a mappable candidate of the search
#: space takes a few dozen modifications.
MAX_STREAMLINE_MODIFICATIONS = 500
#: A leaf transformation that produces the same graph from the same graph this many times is
#: cycling (the transformations are deterministic); detected after a few seconds, long before
#: the modification budget, which costs a cleanup pass per modification.
MAX_STATE_REVISITS = 3


class StreamlineBudgetExceededError(RuntimeError):
    """``StreamlinePlus`` cycles between graphs or exceeded its modification budget."""


def _graph_state(model: ModelWrapper) -> str:
    """Return a digest of the graph: op types, tensor wiring, attributes and constants.

    The constants matter: a transformation may legitimately report a modification that only
    changes initializer values (threshold rounding), which is not a revisited state.
    """
    digest = hashlib.sha1()
    for node in model.graph.node:
        digest.update(node.SerializeToString())
    for init in model.graph.initializer:
        digest.update(init.SerializeToString())
    return digest.hexdigest()


class _Counted(Transformation):
    """Wrap a leaf transformation; count modifications and detect revisited graph states."""

    def __init__(self, inner: Transformation, shared: dict[str, Any], budget: int) -> None:
        """Wrap ``inner``; ``shared`` is common to all leaves of one streamlining run."""
        super().__init__()
        self.inner = inner
        self.shared = shared
        self.budget = budget

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the inner transformation; raise on a cycle or once the budget is used up."""
        before = _graph_state(model)
        model, modified = self.inner.apply(model)
        if not modified:
            return model, modified
        name = type(self.inner).__name__
        self.shared["n"] += 1
        key = (name, before, _graph_state(model))
        seen = self.shared["seen"]
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= MAX_STATE_REVISITS:
            raise StreamlineBudgetExceededError(
                f"StreamlinePlus cycles: {name} produced the same graph from the same graph "
                f"{seen[key]} times; the exhaustive streamlining does not converge on this "
                "graph (two transformations undo each other, cf. eki-project/finn-plus#321)."
            )
        if self.shared["n"] > self.budget:
            raise StreamlineBudgetExceededError(
                f"StreamlinePlus made more than {self.budget} graph modifications (last: "
                f"{name}); the exhaustive streamlining does not converge on this graph."
            )
        return model, modified


def bounded_streamline_plus(budget: int = MAX_STREAMLINE_MODIFICATIONS) -> Transformation:
    """Return ``StreamlinePlus`` with every leaf transformation guarded against cycling."""
    shared: dict[str, Any] = {"n": 0, "seen": {}}

    def wrap(transformation: Transformation) -> Transformation:
        """Recursively wrap the leaves of composed transformations."""
        if isinstance(transformation, ComposedTransformation):
            return ComposedTransformation([wrap(t) for t in transformation.transformations])
        return _Counted(transformation, shared, budget)

    return wrap(StreamlinePlus())


def step_streamline_plus(
    model: ModelWrapper, cfg: DataflowBuildConfig  # noqa: ARG001
) -> ModelWrapper:
    """Streamline with ``StreamlinePlus``; otherwise the builder's ``step_streamline``."""
    model = model.transform(absorb.AbsorbScalarBiasIntoMultiThreshold())
    model = model.transform(bounded_streamline_plus())
    if len(model.get_nodes_by_op_type("Conv")) > 0:
        model = model.transform(LowerConvsToMatMul())
        model = model.transform(MakeMaxPoolNHWC())
        model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
        model = model.transform(MakeMaxPoolNHWC())
        model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(bounded_streamline_plus())
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())
    return model

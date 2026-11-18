"""Analysis for detecting illegal folding (parallelization) configurations.

This module runs the folding legality check of every hardware layer, so that a folding
that cannot be implemented is reported before any code is generated.
"""

import onnx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.exception import FINNFoldingError


def folding_violations(model: ModelWrapper, apply_to_subgraphs: bool = False) -> dict[str, str]:
    """Check the folding (parallelization) attributes of all hardware layers, i.e. of all
    nodes whose domain starts with ``finn.custom_op.fpgadataflow``, by calling their
    ``check_folding`` method. The check only reads node attributes: it neither modifies the
    model nor generates any files.

    Returns {node name : message} for every node with an illegal folding, which is empty if
    the folding of all nodes is legal. Ensure that all nodes have unique names (by calling
    the GiveUniqueNodeNames transformation) prior to calling this analysis pass.

    With apply_to_subgraphs=True the pass descends into subgraphs (e.g. FINNLoop bodies).

    Only ``FINNFoldingError`` is collected. Any other error raised by a ``check_folding``
    implementation, e.g. for a node configuration that cannot be implemented with any
    folding, is not a folding violation and propagates to the caller.
    """
    violations = {}
    for node in model.graph.node:
        if node.domain.startswith("finn.custom_op.fpgadataflow"):
            inst = getCustomOp(node)
            if isinstance(inst, HWCustomOp):
                try:
                    inst.check_folding()
                except FINNFoldingError as e:
                    violations[node.name] = e.msg
        if apply_to_subgraphs:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    subgraph = model.make_subgraph_modelwrapper(attr.g)
                    violations.update(folding_violations(subgraph, apply_to_subgraphs=True))
    return violations

# Copyright (C) 2022, Xilinx, Inc.
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Analytical FIFO sizing from token access vectors (TAVs): generating the TAVs.

The flow, as the ``heuristic_analytical`` strategy of ``step_set_fifo_depths`` runs it:

1. ``DeriveTokenAccessVectors`` gives every node that has a tree model
   (``fifo_sizing.tree_models``) its token access vectors ``io_chrc_in`` /
   ``io_chrc_out``: cumulative per-cycle transaction counts over two periods.
   This is the only place a TAV is written, nothing below modifies one.
2. ``DeriveFIFOSizes`` (``fifo_sizing.derive_fifo_sizes``) reads the TAVs, derives
   a wall-clock schedule for every edge with the chained-TAV pass and writes the
   resulting depths to ``inFIFODepths`` / ``outFIFODepths``.

This module holds step 1. It is the FINN+ port of the analytical half of upstream
FINN's TAV-based FIFO sizing (Xilinx/finn#1185). Upstream can also characterize
nodes by rtlsim, either all of them (``heuristic_rtlsim``) or only those without a
tree model (the fallback of ``heuristic_analytical``). FINN+ does not carry that
path: a node without a tree model gets no TAV and is treated as transparent by the
sizing pass, which is what upstream does in estimate-only builds.
"""

from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import NodeLocalTransformation

from finn.transformation.fpgadataflow.fifo_sizing.tree_models.base import get_tree_model
from finn.util.basic import getHWCustomOp
from finn.util.exception import FINNUserError
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

#: Ops that never get a TAV: the fork and the FIFOs themselves. Joins do not
#: get one either, see ``is_stream_join``.
_UNCHARACTERIZED_OPS = [
    "DuplicateStreams_hls",
    "StreamingFIFO_hls",
    "StreamingFIFO_rtl",
]


def is_stream_join(node: NodeProto | None) -> bool:
    """Return True if ``node`` merges two streamed inputs into one output.

    That is an elementwise binary operation (``ElementwiseAdd``,
    ``ElementwiseMul``, ...) whose inputs are *both* streams. Joins get no TAV:
    the sizer treats them as transparent (output token *k* leaves once token *k*
    is in on both inputs) and gives their inputs the join window floor.

    Args:
        node: Node to check, None is not a join.
    """
    if node is None or not node.op_type.startswith("Elementwise"):
        return False
    if len(node.input) < 2:
        return False
    inst = getHWCustomOp(node)
    return inst.get_nodeattr("lhs_style") == "input" and inst.get_nodeattr("rhs_style") == "input"


def expects_token_access_vectors(node: NodeProto) -> bool:
    """Return True if ``node`` is an HLS or RTL node the sizing pass wants a TAV for.

    Forks, FIFOs and joins are transparent by design and never get one.

    Args:
        node: Node to check.
    """
    if not (is_hls_node(node) or is_rtl_node(node)):
        return False
    return not (node.op_type in _UNCHARACTERIZED_OPS or is_stream_join(node))


def nodes_without_tree_model(model: ModelWrapper) -> list[str]:
    """Return the names of the nodes that should have a TAV but have no tree model.

    The analytical sizing pass treats these nodes as transparent: they forward
    their input timeline unchanged, so the FIFOs around them are sized without
    knowledge of their rate change, burstiness or latency.

    Args:
        model: Model to inspect, before or after ``DeriveTokenAccessVectors``.
    """
    return [
        node.name
        for node in model.graph.node
        if expects_token_access_vectors(node) and get_tree_model(getHWCustomOp(node)) is None
    ]


class DeriveTokenAccessVectors(NodeLocalTransformation):
    """Derive the token access vectors of every node that has a tree model.

    The vectors are stored in sidecar files referenced by the ``io_chrc_in`` /
    ``io_chrc_out`` node attributes, the period in ``io_chrc_period``. Nodes
    without a tree model are left untouched (see ``nodes_without_tree_model``).
    No IP generation, synthesis or simulation is involved.
    """

    def __init__(self, strategy: str = "tree_model", num_workers: int | None = None) -> None:
        """Initialize the transformation.

        Args:
            strategy: Source of the TAVs. Only ``"tree_model"`` is supported; the
                argument exists for call compatibility with upstream FINN.
            num_workers: Number of parallel workers, see NodeLocalTransformation.

        Raises:
            FINNUserError: If a strategy other than ``"tree_model"`` is requested.
        """
        super().__init__(num_workers=num_workers)
        if strategy != "tree_model":
            raise FINNUserError(
                f"Token access vectors can only be derived from tree models in FINN+, "
                f"got strategy '{strategy}' (rtlsim characterization is not supported)"
            )
        self.strategy = strategy

    def applyNodeLocal(self, node: NodeProto) -> tuple[NodeProto, bool]:  # noqa: N802
        """Derive and store the token access vectors of a single node.

        Args:
            node: Node to characterize.

        Returns:
            The node and False, since the graph structure is never modified.
        """
        if expects_token_access_vectors(node):
            getHWCustomOp(node).derive_token_access_vectors(self.strategy)
        return (node, False)

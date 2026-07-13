"""Transformations for inserting and managing muxes and demuxes."""
import onnx.helper as oh
from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames, GiveUniqueTensorNames, SortGraph
from typing import TYPE_CHECKING, cast

from finn.util.exception import FINNInternalError
from finn.util.logging import log

if TYPE_CHECKING:
    from collections import Sequence

    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class InsertMuxByName(Transformation):
    """Given a list of node names, connect all of their edges to a single Multiplexer,
    while connecting all their successor edges to a single Demultiplexer.

    When the insertion is done between nodes on different devices, this implicitly
    multiplexes multiple branches across a single network channel.
    """

    def __init__(self, node_edges: list[tuple[str, str]]) -> None:
        """Insert mux by node names. `node_edges` contains edges (source_node, target_node),
        in which a mux/demux pair is inserted.
        """
        self.node_edges = node_edges

    def find_connecting_tensors(
        self, model: ModelWrapper, node_a: str | NodeProto, node_b: str | NodeProto
    ) -> list[str]:
        """If two nodes are connected, get the names of the tensors between them.
        If they are not connected, return an empty list. If one of the nodes does
        not exist, raises an error.
        """
        a = node_a
        b = node_b
        tensors = []
        if type(node_a) is str:
            a = model.get_node_from_name(node_a)
            if a is None:
                raise FINNInternalError(f"No node with the name {node_a} found in the graph!")
        if type(node_b) is str:
            b = model.get_node_from_name(node_b)
            if b is None:
                raise FINNInternalError(f"No node with the name {node_b} found in the graph!")
        a = cast("NodeProto", a)
        b = cast("NodeProto", b)
        for out in a.output:
            consumers = model.find_consumers(out)
            if b in consumers:
                tensors.append(out)
        return tensors

    def get_mux_instream_widths(
        self, model: ModelWrapper, node_name: str, output_tensors: list[str]
    ) -> list[int]:
        """Collect all widths between the given node and the mux."""
        widths = []
        for node in model.graph.node:
            if node.name == node_name:
                for i, out in enumerate(node.output):
                    if out in output_tensors:
                        widths.append(cast("HWCustomOp", getCustomOp(node)).get_outstream_width(i))
        if len(widths) == 0:
            raise FINNInternalError(f"No output stream widths found for node {node_name}.")
        return widths

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        mux_inputs = []
        demux_outputs = []
        mux_instream_widths = []
        mux_in_dtypes = []
        mux_in_normal_shapes = []
        mux_in_folded_shapes = []

        # Find the tensors between the nodes
        for a, b in self.node_edges:
            node_a = model.get_node_from_name(a)
            node_b = model.get_node_from_name(b)
            if node_a is None:
                raise FINNInternalError(f"Could not find node {a}.")
            if node_b is None:
                raise FINNInternalError(f"Could not find node {b}.")
            node_a = cast("HWCustomOp", getCustomOp(node_a))
            node_b = cast("HWCustomOp", getCustomOp(node_b))

            # Find the names of the connecting tensors
            tensors = self.find_connecting_tensors(model, a, b)
            if len(tensors) == 0:
                log.warning(
                    f"Could not insert Mux between nodes {a} and {b}: "
                    f"No connection between these nodes was found."
                )
                continue

            # Add all instream widths. We add them in the order of the node-edges given
            mux_instream_widths += self.get_mux_instream_widths(model, a, tensors)

            # Make connection: A -> old tensor -> mux -> demux -> new tensor -> B
            mux_inputs += tensors
            for i, tens in enumerate(tensors):
                # For every tensor we need to add the type
                mux_in_dtypes.append(node_a.get_output_datatype(i).get_canonical_name())

                # Also add shapes
                mux_in_folded_shapes.append(node_a.get_folded_output_shape(i))
                mux_in_normal_shapes.append(node_a.get_normal_output_shape(i))

                # Create the new tensor between demux and B
                old_vi = model.get_tensor_valueinfo(tens)
                if old_vi is None:
                    raise FINNInternalError(f"ValueInfo for tensor {tens} not found.")

                # Create new VI
                new_vi = oh.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    old_vi.type.tensor_type.elem_type,
                    # Have to cast the sequence, because the normal shape might be an ndarray
                    cast("Sequence[int]", node_a.get_normal_output_shape()),
                )
                model.graph.value_info.append(new_vi)
                model.set_tensor_datatype(new_vi.name, node_a.get_output_datatype())
                demux_outputs.append(new_vi.name)

        # Create value_info for connecting mux and demux
        connector = oh.make_empty_tensor_value_info("connector")
        model.graph.value_info.append(connector)

        model = model.transform(GiveUniqueTensorNames())

        # Create the mux node
        mux = oh.make_node(
            "Multiplexer_hls",
            domain="finn.custom_op.fpgadataflow.hls",
            backend="fpgadataflow",
            name="MuxNode",
            inputs=mux_inputs,
            outputs=[connector.name],
            muxStrategy="static_schedule_round_robin",
            inStreams=[f"in{i}_V" for i in range(len(mux_inputs))],
            inStreamWidths=mux_instream_widths,
            inStreamDataTypes=mux_in_dtypes,
            inStreamFoldedOutputShapes=mux_in_folded_shapes,
            inStreamNormalOutputShapes=mux_in_normal_shapes,
            outStream="out",
        )
        demux = oh.make_node(
            "Demultiplexer_hls",
            domain="finn.custom_op.fpgadataflow.hls",
            backend="fpgadataflow",
            name="DeMuxNode",
            inputs=[connector.name],
            outputs=demux_outputs,
            muxStrategy="static_schedule_round_robin",
            outStreams=[f"out{i}_V" for i in range(len(demux_outputs))],
            # outStreamWidths=,
            # outStreamDataTypes,
            # outStreamFoldedOutputShapes,
            # outStreamNormalOutputShapes,
            inStream="in",
        )

        model.graph.node.append(mux)
        model.graph.node.append(demux)
        model = model.transform(GiveUniqueTensorNames())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(SortGraph())
        return model, False

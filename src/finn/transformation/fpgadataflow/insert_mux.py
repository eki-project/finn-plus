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
    from collections.abc import Sequence

    from finn.custom_op.fpgadataflow.hls.multiplexer_hls import Multiplexer_hls
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class InsertMuxDemuxPairByName(Transformation):
    """Given a list of node names, connect all of their edges to a single Multiplexer,
    while connecting all their successor edges to a single Demultiplexer.

    When the insertion is done between nodes on different devices, this implicitly
    multiplexes multiple branches across a single network channel.

    For multiple mux/demux pairs, the transformation has be called multiple times.
    """

    def __init__(
        self,
        node_edges: list[tuple[str, str]],
        variant: str = "static_schedule",
        subvariant: str = "round_robin",
    ) -> None:
        """Insert mux by node names. `node_edges` contains edges (source_node, target_node),
        in which a mux/demux pair is inserted.
        """
        self.node_edges = node_edges
        self.variant = variant
        self.subvariant = subvariant

    def find_connecting_tensor(
        self, model: ModelWrapper, node_a: str | NodeProto, node_b: str | NodeProto
    ) -> tuple[str, int, int]:
        """If two nodes are connected, get the name of the tensor between them, as
        well as the index from the predecessor.output and successor.input views.
        If they are not connected or if one of the nodes does
        not exist, raises an error.
        """
        a = node_a
        b = node_b
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

        for out_index, out_name in enumerate(a.output):
            for in_index, in_name in enumerate(b.input):
                if out_name == in_name:
                    return out_name, out_index, in_index
        raise FINNInternalError(
            f"It seems that there is no connection between nodes {node_a} and {node_b}."
        )

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
        in_stream_widths = []
        in_stream_dts = []
        in_stream_normal_shapes = []
        in_stream_folded_shapes = []
        out_stream_widths = []
        out_stream_dts = []
        out_stream_normal_shapes = []
        out_stream_folded_shapes = []

        # Find the tensors between the nodes
        for a, b in self.node_edges:
            node_a = model.get_node_from_name(a)
            node_b = model.get_node_from_name(b)
            if node_a is None:
                raise FINNInternalError(f"Could not find node {a}.")
            if node_b is None:
                raise FINNInternalError(f"Could not find node {b}.")
            node_op_a = cast("HWCustomOp", getCustomOp(node_a))
            node_op_b = cast("HWCustomOp", getCustomOp(node_b))

            # Find the names of the connecting tensor
            tensor, out_index, in_index = self.find_connecting_tensor(model, a, b)

            # Check if widths match
            out_width = node_op_a.get_outstream_width(out_index)
            in_width = node_op_b.get_instream_width(in_index)
            if out_width != in_width:
                raise FINNInternalError(
                    f"Cannot mux/demux streams with non matching "
                    f"stream widths. Mux-input has width: {out_width}, "
                    f"Demux-output has width: {in_width}. "
                    f"Consider inserting DWCs first."
                )

            # Add all instream widths. We add them in the order of the node-edges given
            in_stream_widths.append(node_op_a.get_outstream_width(out_index))
            out_stream_widths.append(node_op_b.get_instream_width(in_index))

            # Make connection: A -> old tensor -> mux -> demux -> new tensor -> B
            mux_inputs.append(tensor)

            # For every tensor we need to add the type
            in_stream_dts.append(node_op_a.get_output_datatype(out_index).get_canonical_name())
            out_stream_dts.append(node_op_b.get_input_datatype(in_index).get_canonical_name())

            # Also add shapes
            in_stream_folded_shapes.append(
                ", ".join(str(s) for s in node_op_a.get_folded_output_shape(out_index))
            )
            in_stream_normal_shapes.append(
                ", ".join(str(s) for s in node_op_a.get_normal_output_shape(out_index))
            )
            out_stream_folded_shapes.append(
                ", ".join(str(s) for s in node_op_b.get_folded_output_shape(in_index))
            )
            out_stream_normal_shapes.append(
                ", ".join(str(s) for s in node_op_b.get_normal_output_shape(in_index))
            )

            # Create the new tensor between demux and B
            old_vi = model.get_tensor_valueinfo(tensor)
            if old_vi is None:
                raise FINNInternalError(f"ValueInfo for tensor {tensor} not found.")

            # Create new VI
            new_vi = oh.make_tensor_value_info(
                model.make_new_valueinfo_name(),
                old_vi.type.tensor_type.elem_type,
                # Have to cast the sequence, because the normal shape might be an ndarray
                cast("Sequence[int]", node_op_a.get_normal_output_shape(out_index)),
            )
            model.graph.value_info.append(new_vi)
            model.set_tensor_datatype(new_vi.name, node_op_a.get_output_datatype(out_index))
            demux_outputs.append(new_vi.name)
            node_b.input[in_index] = new_vi.name

        # Create value_info for connecting mux and demux
        connector = oh.make_empty_tensor_value_info("connector")
        model.graph.value_info.append(connector)

        # Create mux/demux nodes
        mux = oh.make_node(
            "Multiplexer_hls",
            domain="finn.custom_op.fpgadataflow.hls",
            backend="fpgadataflow",
            name="MuxNode",
            inputs=mux_inputs,
            outputs=[connector.name],
            muxVariant=self.variant,
            muxVariantSubtype=self.subvariant,
            streamNames=[f"in{i}_V" for i in range(len(mux_inputs))],
            streamWidths=in_stream_widths,
            streamDataTypes=in_stream_dts,
            streamsFoldedShapes=in_stream_folded_shapes,
            streamsNormalShapes=in_stream_normal_shapes,
            connectionStream="out",
        )
        demux = oh.make_node(
            "Demultiplexer_hls",
            domain="finn.custom_op.fpgadataflow.hls",
            backend="fpgadataflow",
            name="DeMuxNode",
            inputs=[connector.name],
            outputs=demux_outputs,
            muxVariant=self.variant,
            muxVariantSubtype=self.subvariant,
            streamNames=[f"out{i}_V" for i in range(len(demux_outputs))],
            streamWidths=out_stream_widths,
            streamDataTypes=out_stream_dts,
            streamsFoldedShapes=out_stream_folded_shapes,
            streamsNormalShapes=out_stream_normal_shapes,
            connectionStream="in",
        )
        model.graph.node.append(mux)
        model.graph.node.append(demux)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(SortGraph())
        model = model.transform(GiveUniqueTensorNames())

        connection_dtype = cast("Multiplexer_hls", getCustomOp(mux)).get_connection_dtype()
        log.info(
            f"Added mux/demux pair between {len(mux_inputs)} pairs of nodes. "
            f"Type: {self.variant} ({self.subvariant}) "
            f"Connection datatype between components: "
            f"{connection_dtype.get_canonical_name()}"
        )

        return model, False

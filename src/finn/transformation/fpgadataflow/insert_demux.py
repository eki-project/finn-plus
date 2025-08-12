from onnx import NodeProto, TensorProto
from onnx import helper as oh
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from typing import TYPE_CHECKING, Any

from finn.custom_op.fpgadataflow.hls import demux_hls
from finn.custom_op.fpgadataflow.hls.demux_hls import MultiplexStrategy
from finn.util.exception import FINNConfigurationError, FINNInternalError

if TYPE_CHECKING:
    from qonnx.core.datatype import DataType

    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


# TODO: Add consideration for multifpga device assignment?


class InsertDeMuxAfterNodes(Transformation):
    """Insert (de)mux nodes after the given node names.
    If we have
    A B C
    | | |
    V V V
    D E F

    afterwards we have
    A B C
    | / /
    | /
    Mux
    |
    Demux
    |
    |\\ \\
    D E F"""

    def __init__(
        self,
        producer_names: list[str],
        network_bitwidth: int,
        multiplex_strategy: MultiplexStrategy,
    ) -> None:
        super().__init__()
        self.prods = producer_names
        self.network_bitwidth = network_bitwidth
        self.multiplex_strategy = multiplex_strategy

    def apply(self, model: ModelWrapper) -> tuple[bool, ModelWrapper]:
        self.consumer_nodes: list[NodeProto] = []
        self.producer_nodes: list[NodeProto] = []

        # Run sanity checks and collect the nodes
        for producer in self.prods:
            producer_node: NodeProto | None = model.get_node_from_name(producer)
            if producer_node is None:
                raise FINNInternalError(
                    f"Cannot create (de)mux after node {producer}, "
                    "since such a producer is not in the graph!"
                )
            succs: list[Any] | None = model.find_direct_successors(producer_node)
            if succs is None:
                raise FINNInternalError(
                    f"Cannot create (de)mux after node {producer}, "
                    "since the node does not have a successor!"
                )
            if len(succs) > 1:
                raise FINNInternalError(
                    f"Cannot place (de)mux after {producer}, " "since it has > 1 successors"
                )
            self.consumer_nodes.append(succs[0])
            self.producer_nodes.append(producer_node)

        # Sanity check
        if not len(self.consumer_nodes) == len(self.producer_nodes):
            raise FINNInternalError(
                f"Cannot input (de)mux between varying number of producers "
                f"and consumers. (Cannot (de)mux {len(self.producer_nodes)} "
                f"onto {len(self.consumer_nodes)} streams!) (Producers are: "
                f"{[s.name for s in self.producer_nodes]}, Consumers are: "
                f"{[s.name for s in self.consumer_nodes]})"
            )

        # Fill the metadata requried by the (de)mux
        names: list[str] = []
        types: list[str] = []
        normals: list[str] = []
        foldeds: list[str] = []
        widths: list[str] = []

        # TODO: Can possibly remove names
        names = [s.name for s in self.producer_nodes]

        # Add types, shapes and widths
        for prod, cons in zip(self.producer_nodes, self.consumer_nodes):
            prodop: HWCustomOp = getCustomOp(prod)
            consop: HWCustomOp = getCustomOp(cons)

            # Datatypes
            prodtype: DataType | None = prodop.get_output_datatype()
            if prodtype is None:
                raise FINNInternalError(f"Cannot determine output datatype of node {prod.name}!")
            constype: DataType | None = consop.get_input_datatype()
            if constype is None:
                raise FINNInternalError(f"Cannot determine input datatype of node {cons.name}!")
            if prodtype != constype:
                raise FINNInternalError(
                    f"InsertDeMuxAfterNodes tried to place a mux-demux "
                    f"pair between to two nodes ({prod.name}, {cons.name})"
                    f" of non-matching datatypes: {prodtype} and "
                    f"{constype}!"
                )
            types.append(str(prodtype))

            # Shapes
            prodshape_norm: None | Any = prodop.get_normal_output_shape()
            consshape_norm: None | Any = consop.get_normal_input_shape()
            if prodshape_norm is None:
                raise FINNConfigurationError(
                    f"Node {prod.name} has no normal output shape. Maybe run shape inference?"
                )
            if consshape_norm is None:
                raise FINNConfigurationError(
                    f"Node {cons.name} has no normal input shape. Maybe run shape inference?"
                )
            if prodshape_norm != consshape_norm:
                raise FINNInternalError(
                    f"Cannot place de-mux pair between two nodes of different shapes "
                    f"({prod.name}: {prodshape_norm}, {cons.name}: {consshape_norm})"
                )
            normals.append(prodshape_norm)

            prodshape_fold: None | Any = prodop.get_folded_output_shape()
            consshape_fold: None | Any = consop.get_folded_input_shape()
            if prodshape_fold is None:
                raise FINNConfigurationError(
                    f"Node {prod.name} has no folded output shape. Maybe run shape inference?"
                )
            if consshape_fold is None:
                raise FINNConfigurationError(
                    f"Node {cons.name} has no folded input shape. Maybe run shape inference?"
                )
            if prodshape_fold != consshape_fold:
                raise FINNInternalError(
                    f"Cannot place de-mux pair between two nodes of different shapes "
                    f"({prod.name}: {prodshape_fold}, {cons.name}: {consshape_fold})"
                )
            foldeds.append(prodshape_fold)

            prodwidth: None | Any = prodop.get_outstream_width()
            conswidth: None | Any = consop.get_instream_width()
            if prodwidth is None:
                raise FINNInternalError(f"Node {prod.name} has no outstream width!")
            if conswidth is None:
                raise FINNInternalError(f"Node {cons.name} has no instream width!")
            if prodwidth != conswidth:
                raise FINNInternalError(
                    f"Cannot place de-mux pair between two nodes with different stream widths:"
                    f"{prod.name} (out): {prodwidth}, {cons.name} (in): {conswidth}"
                )
            widths.append(prodwidth)

        # Already create a name for the connection between mux and demux
        connection_name = model.make_new_valueinfo_name()

        # Insert the new nodes (demux)
        mux_node = oh.make_node(
            "AnnotatedMux_hls",
            [inp.name for inp in self.producer_nodes],
            [connection_name],
            domain="finn.custom_op.fpgadataflow.hls",
            backend="fpgadataflow",
            streamNames=names,
            streamTypes=types,
            streamNormalShapes=normals,
            streamFoldedShapes=foldeds,
            streamWidths=widths,
            muxed_bitwidth=self.network_bitwidth,
            multiplexStrategy=self.multiplex_strategy,
        )
        demux_node = oh.make_node(
            "AnnotatedDemux_hls",
            [connection_name],
            [outp.name for outp in self.consumer_nodes],
            domain="finn.custom_op.fpgadataflow.hls",
            backend="fpgadataflow",
            streamNames=names,
            streamTypes=types,
            streamNormalShapes=normals,
            streamFoldedShapes=foldeds,
            streamWidths=widths,
            muxed_bitwidth=self.network_bitwidth,
            multiplexStrategy=self.multiplex_strategy,
        )

        # TODO: Check that (de)mux shapes are correctly implemented in demux_hls.py
        mux_op: demux_hls.AnnotatedMux_hls = getCustomOp(mux_node)
        demux_op: demux_hls.AnnotatedDemux_hls = getCustomOp(demux_node)
        if mux_op.get_normal_output_shape() != demux_op.get_normal_input_shape():
            raise FINNInternalError("Shape mismatch between mux and demux!")
        if mux_op.get_output_datatype() != demux_op.get_input_datatype():
            raise FINNInternalError("Datatype mismatch between mux and demux!")

        # Create the value info needed to connect the mux and demux
        connection_vi = oh.make_tensor_value_info(
            connection_name, TensorProto.FLOAT, mux_op.get_normal_output_shape()
        )
        model.graph.value_info.append(connection_vi)
        model.set_tensor_datatype(connection_vi, mux_op.get_output_datatype())

        # Insert nodes after last producer
        # TODO: Check if we need to remove old connections in graph
        node_index = 0
        for i, node in enumerate(model.graph.node):
            if node.name == self.producer_nodes[-1].name:
                node_index = i
        model.graph.node.insert(node_index + 1, mux_node)
        model.graph.node.insert(node_index + 2, demux_node)
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        return False, model

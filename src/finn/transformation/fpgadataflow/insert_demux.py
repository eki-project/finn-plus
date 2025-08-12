from onnx import NodeProto
from onnx import helper as oh
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from typing import TYPE_CHECKING, Any

from finn.custom_op.fpgadataflow.hls.demux_hls import MultiplexStrategy
from finn.util.exception import FINNConfigurationError, FINNInternalError

if TYPE_CHECKING:
    from qonnx.core.datatype import DataType

    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


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

        # Can possibly remove names
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
                    f"Node {prod.name} has no normal output shape. " f"Maybe run shape inference?"
                )
            if consshape_norm is None:
                raise FINNConfigurationError(
                    f"Node {cons.name} has no normal input shape. " f"Maybe run shape inference?"
                )
            if prodshape_norm != consshape_norm:
                raise FINNInternalError(
                    f"Cannot place de-mux pair between two nodes of different shapes ({prod.name})"
                )

        # TODO: Change "network" value info, so that not every mux connects to the same one
        #       tensor in the whole network
        # Insert the new nodes (demux)
        mux_node = oh.make_node(
            "AnnotatedMux_hls",
            [inp.name for inp in self.producer_nodes],
            ["network"],
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
            ["network"],
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

        # TODO: Remove. Needed because of pre-commit
        print(mux_node)
        print(demux_node)

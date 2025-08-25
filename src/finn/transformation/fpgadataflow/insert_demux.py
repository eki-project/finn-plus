"""Contains transformations to insert Mux-Demux pairs into a folded and FIFO-sized graph."""

from onnx import NodeProto, TensorProto
from onnx import helper as oh
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from typing import TYPE_CHECKING, Any

from finn.custom_op.fpgadataflow.hls.demux_hls import MultiplexStrategy
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.exception import FINNConfigurationError, FINNInternalError
from finn.util.logging import log

if TYPE_CHECKING:
    from qonnx.core.datatype import DataType

    from finn.custom_op.fpgadataflow.hls import demux_hls
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


# TODO: Add consideration for multifpga device assignment?


class InsertDeMuxAfterNodes(Transformation):
    """Insert Mux-Demux pairs after the given producer nodes.

    After every given producer a cut is being made. All producers lead to a single AnnotatedMux_hls
    operator, with only one output. Following is the matching AnnotatedDemux_hls operator.
    Can automatically insert DWCs in case of mismatching stream widths between the nodes.
    Start with
    ```
    A -> C
    B -> D
    ```
    after this transform we have
    ```
    A                        C
      ==> Mux ---> Demux ==>
    B                        D
    ```
    """

    def __init__(
        self,
        producer_names: list[str],
        network_bitwidth: int,
        multiplex_strategy: MultiplexStrategy,
        automatically_insert_dwc: bool = True,
        fpgapart: str = "",
    ) -> None:
        """Create an insertion transformation for Mux-Demux pairs.

        Args:
            producer_names: Names of the nodes after which to insert the mux op.
            network_bitwidth: The width of the single connection between Mux and Demux.
            multiplex_strategy: Which strategy the Mux should use.
            automatically_insert_dwc: If True, the transformation automatically inserts DWCs,
                                        if the producer and consumer bitwidths don't match.
            fpgapart: The FPGA-part given in the build config.

        """
        super().__init__()
        self.prods = producer_names
        self.network_bitwidth = network_bitwidth
        self.multiplex_strategy = multiplex_strategy
        if automatically_insert_dwc and fpgapart == "":
            raise FINNInternalError(
                "InsertDeMuxAfterNodes is supposed to automatically insert "
                "DWCs on non-matching stream widths. This requires the "
                "fpgapart to be passed to the transformation, which it isn't!"
            )
        self.insert_dwc = automatically_insert_dwc
        self.fpgapart = fpgapart

    def get_common_valueinfotensor(self, producer: NodeProto, consumer: NodeProto) -> list[str]:
        """Return a list of valueinfo tensors common to producer outputs and consumer inputs.

        Returns:
            A list of names of tensors appearing both in the producer outputs and consumer inputs.

        """
        return list(set(producer.output).intersection(set(consumer.input)))

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
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
                    f"Cannot place (de)mux after {producer}, since it has > 1 successors"
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

        # Collect metadata for the (de)mux HW nodes
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

            # NORMAL SHAPES
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
            if list(prodshape_norm) != list(consshape_norm):
                raise FINNInternalError(
                    f"Cannot place de-mux pair between two nodes of different normal shapes "
                    f"({prod.name}: {prodshape_norm}, {cons.name}: {consshape_norm})"
                )
            normals.append(",".join([str(x) for x in prodshape_norm]))

            # FOLDED SHAPES
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
            foldeds.append(",".join([str(x) for x in prodshape_fold]))

            # STREAM WIDTHS
            prodwidth: None | Any = prodop.get_outstream_width()
            conswidth: None | Any = consop.get_instream_width()
            if prodwidth is None:
                raise FINNInternalError(f"Node {prod.name} has no outstream width!")
            if conswidth is None:
                raise FINNInternalError(f"Node {cons.name} has no instream width!")
            if prodwidth != conswidth:
                if not self.insert_dwc:
                    raise FINNInternalError(
                        f"Cannot place de-mux pair between two nodes with different stream widths:"
                        f"{prod.name} (out): {prodwidth}, {cons.name} (in): {conswidth}. Either "
                        f"run InsertDWC() before or let this transformation do it automatically "
                        f"(see constructor)."
                    )
                # If configured, automatically insert DWCs and run again
                for p, c in zip(self.producer_nodes, self.consumer_nodes):
                    pw = getCustomOp(p).get_outstream_width()
                    cw = getCustomOp(c).get_instream_width()
                    if pw is not None and cw is not None and pw != cw:
                        log.warning(
                            f"Mismatching stream widths between {p.name} "
                            f"and {c.name} - Inserting DWCs to fix."
                        )
                node_count_before = len(model.graph.node)
                model = model.transform(InsertDWC())
                model = model.transform(SpecializeLayers(self.fpgapart))
                model = model.transform(GiveUniqueNodeNames())
                model = model.transform(GiveReadableTensorNames())
                node_count_after = len(model.graph.node)
                log.warning(f"Inserted {node_count_after - node_count_before} DWCs into the graph!")

                # Rerun this transformation with the modified graph!
                return model, True
            widths.append(prodwidth)

        # Already create a name for the connection between mux and demux
        connection_name = model.make_new_valueinfo_name()

        # Insert the new valueinfos and nodes
        # Currently: MVAU_0 -> A -> MVAU_1
        # Afterwards: MVAU_0 -> A -> MUX -> B -> DEMUX -> C -> MVAU_1
        # For every pair (ex. MVAU_0, MVAU_1) gather the A inbetween
        mux_inputs: list[str] = []
        for prod, cons in zip(self.producer_nodes, self.consumer_nodes):
            commons = self.get_common_valueinfotensor(prod, cons)
            if len(commons) > 1:
                raise FINNInternalError(
                    f"Nodes {prod.name} and {cons.name} have multiple "
                    f"connections. Cannot determine which one to use! "
                    f"Common to both are: {commons}"
                )
            if len(commons) == 0:
                raise FINNConfigurationError(
                    f"Cannot place de-mux between {prod.name} and "
                    f"{cons.name}: No output of {prod.name} appears "
                    f"in the inputs of {cons.name}!"
                )
            mux_inputs.append(commons[0])

        # Create new VIs to connect the receiving partner of the pair to:
        # MVAU_0 -> A,     B -> MVAU_1
        demux_outputs: list[str] = []
        for mux_input_vi, cons in zip(mux_inputs, self.consumer_nodes):
            vi = oh.make_tensor_value_info(
                model.make_new_valueinfo_name(),
                TensorProto.FLOAT,
                model.get_tensor_shape(mux_input_vi),
            )
            model.graph.value_info.append(vi)
            index = list(cons.input).index(mux_input_vi)
            cons.input.remove(mux_input_vi)
            cons.input.insert(index, vi.name)
            demux_outputs.append(vi.name)
            model.set_tensor_datatype(vi.name, model.get_tensor_datatype(mux_input_vi))

        # Log interesting information
        log.info("Creating Mux-Demux pair with data:")
        log.info(f"Names: {names}")
        log.info(f"Datatypes: {[str(x) for x in types]}")
        log.info(f"Normal shapes: {normals}")
        log.info(f"Folded shapes: {foldeds}")
        log.info(f"Stream widths: {widths}")
        log.info(f"Bitwidth between Mux and Demux: {self.network_bitwidth}")
        log.info(f"Multiplexing strategy: {self.multiplex_strategy.value}")

        # Create mux and demux nodes, connect MUX to the A's
        # Connect the DEMUX nodes to the B's
        mux_node = oh.make_node(
            "AnnotatedMux_hls",
            mux_inputs,
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
            demux_outputs,
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
        model.set_tensor_datatype(connection_vi.name, mux_op.get_output_datatype())

        # Insert nodes after last producer
        node_index = 0
        for i, node in enumerate(model.graph.node):
            if node.name == self.producer_nodes[-1].name:
                node_index = i
        model.graph.node.insert(node_index + 1, mux_node)
        model.graph.node.insert(node_index + 2, demux_node)
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        return model, False

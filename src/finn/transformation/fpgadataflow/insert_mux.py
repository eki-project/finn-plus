"""Transformations for inserting and managing muxes and demuxes."""
# ruff: noqa: D102, D107
import builtins
import contextlib
import json
import onnx.helper as oh
from json import JSONDecodeError
from onnx import NodeProto, ValueInfoProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueTensorNames,
    SortGraph,
)
from typing import TYPE_CHECKING, Any, cast

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.set_fifo_depths import SplitLargeFIFOs
from finn.util.deprecated import deprecated
from finn.util.exception import FINNInternalError
from finn.util.logging import log

if TYPE_CHECKING:
    from collections.abc import Sequence

    from finn.custom_op.fpgadataflow.hls.demultiplexer_hls import Demultiplexer_hls
    from finn.custom_op.fpgadataflow.hls.multiplexer_hls import Multiplexer_hls


def get_io_index(source: NodeProto, target: NodeProto) -> tuple[int, int]:
    """If source and target are connected, return which output number target is for source,
    and which input number source is for target. (2,4) means that target is source's second output
    and that source is target's 4th input.
    """
    for outidx, outvalue in enumerate(source.output):
        for inidx, invalue in enumerate(target.input):
            if outvalue == invalue:
                return (outidx, inidx)
    raise FINNInternalError(
        f"Cannot get IO connection indices for unconnected "
        f"nodes {source.name} and {target.name}."
    )


class InsertMuxDemuxPairByName(Transformation):
    """Given a list of node names, connect all of their edges to a single Multiplexer,
    while connecting all their successor edges to a single Demultiplexer.

    When the insertion is done between nodes on different devices, this implicitly
    multiplexes multiple branches across a single network channel.

    For multiple mux/demux pairs the transformation has be called multiple times.
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

    def dfs(self, model: ModelWrapper) -> list[NodeProto]:
        """Visit nodes in DFS order."""
        visited = []
        current = [
            node for node in model.graph.node if model.find_direct_predecessors(node) is None
        ]
        while len(current) > 0:
            visited.append(current[0])
            n = current.pop(0)
            successors = model.find_direct_successors(n)
            if successors is not None:
                current = successors + current
        return visited

    @deprecated
    def reorder_pairs(
        self, model: ModelWrapper, pairs: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        def _get_partner(source: NodeProto) -> NodeProto:
            for a, b in pairs:
                if a == source.name:
                    return cast("NodeProto", model.get_node_from_name(b))
            raise FINNInternalError()

        reordered = []
        sources = [x[0] for x in pairs]

        # First filter the required nodes in DFS
        nodes = [node for node in self.dfs(model) if node.name in sources]

        # TODO: There is a better algorithm for this

        # Offset nodes which don't have the partner as first output
        idx = 0
        while idx < len(nodes):
            current = nodes[idx]
            partner = _get_partner(current)
            oidx, _ = get_io_index(current, partner)
            nodes.remove(current)
            idx = idx + oidx
            nodes.insert(idx, current)
            idx += 1

        reordered = [(n.name, _get_partner(n).name) for n in nodes]
        return reordered

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Insert a Mux / Demux pair between the given nodes."""
        mux_inputs = []
        demux_outputs = []
        in_stream_widths = []
        in_stream_dts = []
        in_stream_normal_shapes = []
        in_stream_folded_shapes = []
        in_stream_out_fifodepths = []
        out_stream_widths = []
        out_stream_dts = []
        out_stream_normal_shapes = []
        out_stream_folded_shapes = []
        out_stream_in_fifodepths = []

        model = model.transform(SortGraph())

        if len(self.node_edges) == 0:
            raise FINNInternalError(
                "Cannot insert mux/demux pair between zero node pairs. "
                "Please pass at least 2 or more pairs."
            )
        if len(self.node_edges) == 1:
            a, b = self.node_edges[0]
            log.warning(
                f"Skipping insertion of mux/demux pair between "
                f"the only given pair of nodes ({a}, {b})."
            )
            return model, False

        # Find the tensors between the nodes
        self.node_edges = self.reorder_pairs(model, self.node_edges)

        # Store indices and tensors before modifying the graph
        tensor_out_in = [self.find_connecting_tensor(model, a, b) for a, b in self.node_edges]

        combined_data = [
            (*self.node_edges[i], *tensor_out_in[i]) for i in range(len(self.node_edges))
        ]

        for a, b, tensor, source_output_index, target_input_index in combined_data:
            node_a = model.get_node_from_name(a)
            node_b = model.get_node_from_name(b)
            if node_a is None:
                raise FINNInternalError(f"Could not find node {a}.")
            if node_b is None:
                raise FINNInternalError(f"Could not find node {b}.")
            node_op_a = cast("HWCustomOp", getCustomOp(node_a))
            node_op_b = cast("HWCustomOp", getCustomOp(node_b))

            # Check if widths match
            out_width = node_op_a.get_outstream_width(source_output_index)
            in_width = node_op_b.get_instream_width(target_input_index)
            if out_width != in_width:
                raise FINNInternalError(
                    f"Cannot mux/demux streams with non matching "
                    f"stream widths. Mux-input has width: {out_width}, "
                    f"Demux-output has width: {in_width}. "
                    f"Consider inserting DWCs first."
                )

            # Add all instream widths. We add them in the order of the node-edges given
            in_stream_widths.append(node_op_a.get_outstream_width(source_output_index))
            out_stream_widths.append(node_op_b.get_instream_width(target_input_index))

            # Store FIFO depths
            try:
                source_depth = cast("list[int]", node_op_a.get_nodeattr("outFIFODepths"))[
                    source_output_index
                ]
                if source_depth is None:
                    source_depth = 2
                in_stream_out_fifodepths.append(source_depth)
            except IndexError:
                suc = model.find_direct_successors(node_a)
                assert suc is not None
                suc = [s.name for s in suc]
                raise FINNInternalError(
                    f"Connection number {source_output_index} of node {a} has "
                    f"no corresponding outFIFODepth. outFIFODepths: "
                    f"{node_op_a.get_nodeattr('outFIFODepths')}. Node {a} "
                    f"has these successors: {suc}"
                ) from None

            try:
                target_depth = cast("list[int]", node_op_b.get_nodeattr("inFIFODepths"))[
                    target_input_index
                ]
                if target_depth is None:
                    target_depth = 2
                out_stream_in_fifodepths.append(target_depth)
            except IndexError:
                pre = model.find_direct_predecessors(node_b)
                assert pre is not None
                pre = [p.name for p in pre]
                raise FINNInternalError(
                    f"Connection number {target_input_index} of node {b} has "
                    f"no corresponding inFIFODepth. inFIFODepths: "
                    f"{node_op_a.get_nodeattr('inFIFODepths')}. Node {b} has "
                    f"these predecessors: {pre}"
                ) from None

            # Make connection: A -> old tensor -> mux -> demux -> new tensor -> B
            mux_inputs.append(tensor)

            # For every tensor we need to add the type
            in_stream_dts.append(
                node_op_a.get_output_datatype(source_output_index).get_canonical_name()
            )
            out_stream_dts.append(
                node_op_b.get_input_datatype(target_input_index).get_canonical_name()
            )

            # Also add shapes
            in_stream_folded_shapes.append(
                ", ".join(str(s) for s in node_op_a.get_folded_output_shape(source_output_index))
            )
            in_stream_normal_shapes.append(
                ", ".join(str(s) for s in node_op_a.get_normal_output_shape(source_output_index))
            )
            out_stream_folded_shapes.append(
                ", ".join(str(s) for s in node_op_b.get_folded_output_shape(target_input_index))
            )
            out_stream_normal_shapes.append(
                ", ".join(str(s) for s in node_op_b.get_normal_output_shape(target_input_index))
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
                cast("Sequence[int]", node_op_a.get_normal_output_shape(source_output_index)),
            )
            model.graph.value_info.append(new_vi)
            model.set_tensor_datatype(
                new_vi.name, node_op_a.get_output_datatype(source_output_index)
            )
            demux_outputs.append(new_vi.name)
            node_b.input[target_input_index] = new_vi.name

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
            streamWidths=in_stream_widths,
            streamDataTypes=in_stream_dts,
            streamsFoldedShapes=in_stream_folded_shapes,
            streamsNormalShapes=in_stream_normal_shapes,
            inFIFODepths=in_stream_out_fifodepths,
            outFIFODepths=[32],
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
            streamWidths=out_stream_widths,
            streamDataTypes=out_stream_dts,
            streamsFoldedShapes=out_stream_folded_shapes,
            streamsNormalShapes=out_stream_normal_shapes,
            inFIFODepths=[32],
            outFIFODepths=out_stream_in_fifodepths,
        )
        model.graph.node.append(mux)
        model.graph.node.append(demux)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(SortGraph())
        model = model.transform(GiveUniqueTensorNames())
        model = model.transform(GiveReadableTensorNames())

        connection_dtype = cast("Multiplexer_hls", getCustomOp(mux)).get_connection_dtype()
        log.info(
            f"Added mux/demux pair between {len(mux_inputs)} pairs of nodes. "
            f"Type: {self.variant} ({self.subvariant}). "
            f"Connection datatype between components: "
            f"{connection_dtype.get_canonical_name()}"
        )

        return model, False


class MergeMuxDemuxIntoCombinedOperator(Transformation):
    """Merge Mux/Demux pairs into a combined operator. This is useful for FIFO sizing, which
    cannot use the `functional` mode (for faster simulations) if the operators are separated.
    After FIFO-Sizing is done, the combined operator should be split again.
    """

    def __init__(self, part: str, clk: float) -> None:
        super().__init__()
        self.part = part
        self.clk = clk

    def check_same_attributes(self, a: HWCustomOp, b: HWCustomOp, attrs: list[str]) -> None:
        """Check that the given node attributes are the same on both operators."""
        for attr in attrs:
            value_a = a.get_nodeattr(attr)
            value_b = b.get_nodeattr(attr)
            if value_a != value_b:
                raise FINNInternalError(
                    f"Nodes {a.onnx_node.name} and {b.onnx_node.name} have different "
                    f"values for attribute '{attr}': '{value_a}' and '{value_b}'"
                )

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        # Search a pair of mux/demux ops
        mux = None
        demux = None
        for node in model.graph.node:
            if node.op_type == "Multiplexer_hls":
                suc = model.find_direct_successors(node)
                if suc is None:
                    raise FINNInternalError(
                        f"Found multiplexer without matching demultiplexer: {node.name}!"
                    )
                if len(suc) != 1:
                    raise FINNInternalError(f"Multiplexer {node.name} has 0 or >1 outputs!")
                if suc[0].op_type != "Demultiplexer_hls":
                    raise FINNInternalError(
                        f"Successor of multiplexer node is not a demultiplexer: {suc[0].name}"
                    )
                mux = node
                demux = suc[0]
                break
        if mux is None or demux is None:
            return model, False
        log.info(f"Merging operators: {mux.name}, {demux.name}")

        # Get their operators
        muxop = cast("Multiplexer_hls", getCustomOp(mux))
        demuxop = cast("Demultiplexer_hls", getCustomOp(demux))

        # Make sure the ops match
        self.check_same_attributes(
            muxop,
            demuxop,
            [
                "muxVariant",
                "muxVariantSubtype",
                "streamWidths",
                "streamDataTypes",
                "streamsFoldedShapes",
                "streamsNormalShapes",
            ],
        )

        # Construct the new node
        combined = oh.make_node(
            "CombinedMuxDemux_hls",
            domain="finn.custom_op.fpgadataflow.hls",
            backend="fpgadataflow",
            name="CombinedMuxDemux",
            inputs=mux.input,
            outputs=demux.output,
            muxVariant=muxop.get_nodeattr("muxVariant"),
            muxVariantSubtype=muxop.get_nodeattr("muxVariantSubtype"),
            streamNames=[f"stream{i}" for i in range(muxop.get_stream_count())],
            streamWidths=muxop.get_nodeattr("streamWidths"),
            streamDataTypes=muxop.get_nodeattr("streamDataTypes"),
            streamsFoldedShapes=muxop.get_nodeattr("streamsFoldedShapes"),
            streamsNormalShapes=muxop.get_nodeattr("streamsNormalShapes"),
            inFIFODepths=muxop.get_nodeattr("inFIFODepths"),
            outFIFODepths=demuxop.get_nodeattr("outFIFODepths"),
            connectionStream="none",
        )

        # Remove connector value info
        vi = cast("ValueInfoProto", model.get_tensor_valueinfo(mux.output[0]))
        model.graph.value_info.remove(vi)

        # Remove old nodes, add new one
        idx = cast("int", model.get_node_index(mux))
        model.graph.node.remove(mux)
        model.graph.node.remove(demux)
        model.graph.node.insert(idx, combined)

        # Cleanup and rerun synthesis
        model = model.transform(SortGraph())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(PrepareIP(self.part, self.clk))
        model = model.transform(HLSSynthIP())
        return model, True


class SplitCombinedMuxDemuxIntoSeparateNodes(Transformation):
    """Inverse of `MergeMuxDemuxIntoCombinedOperator`: Take a CombinedMuxDemux operator
    (likely after FIFO sizing and FIFO depth adjustment) and split it into two separate nodes.

    Details: Creates two nodes, fills their node attributes, inserts them where the combined mux
    was before, creates a connection between these nodes (FIFO Depth 32), re-sorts the graph,
    inserts new FIFOs, re-runs IP preparation and HLS synthesis.

    To give the nodes the correct FIFO size (blocking mux operators),
    call `AdjustMuxDemuxAdjacentFIFOs` first.
    """

    def __init__(self, fpgapart: str, clk: float) -> None:
        super().__init__()
        self.part = fpgapart
        self.clk = clk

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        combined = None
        for node in model.graph.node:
            if node.op_type == "CombinedMuxDemux_hls":
                combined = node
                break
        if combined is None:
            return model, False

        log.info(f"Splitting node {combined.name} into separate mux/demux nodes...")

        # Create connector value info
        connector = oh.make_empty_tensor_value_info("connector")
        model.graph.value_info.append(connector)

        # Create new nodes
        op = getCustomOp(combined)
        mux = oh.make_node(
            "Multiplexer_hls",
            domain="finn.custom_op.fpgadataflow.hls",
            backend="fpgadataflow",
            name="MuxNode",
            inputs=combined.input,
            outputs=[connector.name],
            muxVariant=op.get_nodeattr("muxVariant"),
            muxVariantSubtype=op.get_nodeattr("muxVariantSubtype"),
            streamWidths=op.get_nodeattr("streamWidths"),
            streamDataTypes=op.get_nodeattr("streamDataTypes"),
            streamsFoldedShapes=op.get_nodeattr("streamsFoldedShapes"),
            streamsNormalShapes=op.get_nodeattr("streamsNormalShapes"),
            inFIFODepths=op.get_nodeattr("inFIFODepths"),
            outFIFODepths=[32],
        )
        demux = oh.make_node(
            "Demultiplexer_hls",
            domain="finn.custom_op.fpgadataflow.hls",
            backend="fpgadataflow",
            name="DeMuxNode",
            inputs=[connector.name],
            outputs=combined.output,
            muxVariant=op.get_nodeattr("muxVariant"),
            muxVariantSubtype=op.get_nodeattr("muxVariantSubtype"),
            streamWidths=op.get_nodeattr("streamWidths"),
            streamDataTypes=op.get_nodeattr("streamDataTypes"),
            streamsFoldedShapes=op.get_nodeattr("streamsFoldedShapes"),
            streamsNormalShapes=op.get_nodeattr("streamsNormalShapes"),
            inFIFODepths=[32],
            outFIFODepths=op.get_nodeattr("outFIFODepths"),
        )

        # Exchange nodes in the graph
        combined_index = model.get_node_index(combined)
        assert combined_index is not None
        model.graph.node.remove(combined)
        model.graph.node.insert(combined_index, mux)
        model.graph.node.insert(combined_index + 1, demux)

        # Run necessary transformations
        model = model.transform(SortGraph())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InsertFIFO())
        model = model.transform(SortGraph())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(PrepareIP(self.part, self.clk))
        model = model.transform(HLSSynthIP())
        return model, True


class MergeSuccessiveFIFOs(Transformation):
    """Merge successive FIFOs into one."""

    def __init__(self) -> None:
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        first = None
        second = None
        for node in model.graph.node:
            if "StreamingFIFO" in node.op_type:
                sucs = model.find_direct_successors(node)
                if sucs is None:
                    continue
                if len(sucs) == 1 and "StreamingFIFO" in sucs[0].op_type:
                    first = node
                    second = sucs[0]
        if first is None or second is None:
            return model, False
        log.info(f"Merging FIFOs: {first.name}, {second.name}")
        a_op = getCustomOp(first)
        b_op = getCustomOp(second)
        a_op.set_nodeattr(
            "depth", a_op.get_nodeattr("depth") + b_op.get_nodeattr("depth")  # type: ignore
        )
        a_op.set_nodeattr("outFIFODepths", b_op.get_nodeattr("outFIFODepths"))
        with contextlib.suppress(builtins.BaseException):
            model.graph.value_info.remove(first.output[0])

        # Cannot directly assign to a nodeproto.output, so we work around it
        to_remove = list(first.output)
        for t in to_remove:
            first.output.remove(t)
        for new in second.output:
            first.output.append(new)

        # Remove second FIFO
        model.graph.node.remove(second)
        return model, True


class AdjustMuxDemuxAdjacentFIFOs(Transformation):
    """In order for blocking mux/demux pairs to work, the demux may never stall,
    and thus the FIFO between the demux and a given layer needs to be as large as measured
    by the initial (infinite FIFO size) node-connected FIFO simulation, as well as all
    downstream FIFOs too.

    NOTE: This is only necessary for blocking Mux/Demux variants to avoid deadlocks,
    which can be caused by a blocking write from a Demux onto a stalled FIFO.
    If the chosen Mux/Demux variant can negotiate packet transfers over the connection,
    the FIFOs can be left as they are.

    Requirements:
        - InsertMuxDemux... was called.
        - step_set_fifo_depths was called.
        - Initial FIFO depth estimation is available.
        - FIFOs are placed between Mux/Demux and their surrounding nodes.

    If successful:
        - All downstream FIFOs from the demux have adjusted depths
        - Node surrounding these FIFOs have adjusted inFIFODepths and outFIFODepths.
        - IP preparation and HLS synthesis has been re-run.
    """

    def __init__(  # noqa
        self, fpgapart: str, clk: float, initial_sizes: None | Path | dict[str, list[int]] = None
    ) -> None:
        """Read the initial FIFO sizes and prepare the change in the graph.

        Parameters
        ----------
            `initial_sizes`:
                If None: Try to read the data from the file "initial_fifo_sizes_sim_connected.json",
                the directory of which is read from the metadata property "fifo_results_dir".
                If Path: Try to read the data from the given JSON file.
                If dict: Read the data directly from the given data. (node name -> depth mapping)
                Notably, the data must not necessarily contain all nodes, only the ones attached
                to Mux/Demux nodes.
        """
        self.part = fpgapart
        self.clk = clk
        self.data = None
        if initial_sizes is None or type(initial_sizes) is Path:
            # Do nothing here, only read when applying the transform.
            pass
        elif type(initial_sizes) is dict:
            self.data = initial_sizes

    def try_read_from_file(self, p: Path) -> dict[str, list[int]]:
        """Try to read from file. Expects either a direct mapping (name -> depth), or the format
        emitted by the FIFO sizing (list of dicts with {name: ..., fifo_utilization: ...}).
        Raises an error of neither format works. Does not check if the contents are correct.
        """
        data = {}
        try:
            data = json.loads(p.read_text())
        except FileNotFoundError:
            raise FINNInternalError(
                f"Could not read initial FIFO depth estimates from non-existing file: {p}"
            ) from None
        except JSONDecodeError as e:
            raise FINNInternalError(
                f"There was an error while parsing the FIFO depth estimate file: {e}"
            ) from None

        # Figure out the format
        result = {}
        if type(data) is list:
            for layer_info in data:
                if "name" not in layer_info:
                    raise FINNInternalError(
                        "Missing 'name' field in node information in FIFO initial depth file. "
                        "Expected format: {node0: <depth0>} or {name: 'node0', "
                        "fifo_utilization: '<depth0>'}"
                    )
                if "fifo_utilization" not in layer_info:
                    raise FINNInternalError(
                        "Missing 'fifo_utilization' field in node information in "
                        "FIFO initial depth file. "
                        "Expected format: {node0: <depth0>} or {name: 'node0', "
                        "fifo_utilization: '<depth0>'}"
                    )
                result[layer_info["name"]] = layer_info["fifo_utilization"]
        elif type(data) is dict:
            result = data
        return result

    def get_neighbors(self, model: ModelWrapper, node: NodeProto) -> list[NodeProto]:
        """Return all predecessors and successors of the given node in one list."""
        pre = model.find_direct_predecessors(node)
        suc = model.find_direct_successors(node)
        return (pre if pre is not None else []) + (suc if suc is not None else [])

    def set_attribute(self, node: NodeProto, key: str, value: Any) -> tuple[Any, Any]:
        """Set the attribute of the node to the int value passed. Returns (old, new) values."""
        try:
            op = getCustomOp(node)
            old_value = op.get_nodeattr(key)
            op.set_nodeattr(key, value)
            return (old_value, value)
        except (ValueError, TypeError) as e:
            raise FINNInternalError(
                f"Could not set '{key}' value of node {node.name} to "
                f"{value}, due to a type/value error."
            ) from e
        except AttributeError as e:
            raise FINNInternalError(f"FIFO {node.name} has no '{key}' attribute!") from e

    def get_attribute(self, node: NodeProto, key: str) -> Any:
        """Get the attribute or emit an error."""
        try:
            val = getCustomOp(node).get_nodeattr(key)
            if val is None:
                raise AttributeError()
            return val
        except AttributeError:
            raise FINNInternalError(
                f"Node {node.name} has no attribute / attribute for key is not set: {key}."
            ) from None

    def get_all_downstream_nodes(self, model: ModelWrapper, origin: NodeProto) -> list[NodeProto]:
        """Get all downstream nodes from the origin node, including it. Does not necessarily cover
        all nodes that come "later" then the origin node, since this does not include nodes
        on parallel branches.
        """
        visited = []
        current = [origin]
        while len(current) != 0:
            suc = model.find_direct_successors(current[0])
            visited.append(current.pop(0))
            if suc is not None:
                current += suc
        return visited

    def replace_in_fifo_depth(self, source: NodeProto, target: NodeProto, value: int) -> None:
        """Replace the inFIFODepth at given index of the target node to value."""
        _, iidx = get_io_index(source, target)
        depths = self.get_attribute(target, "inFIFODepths")
        depths[iidx] = value
        self.set_attribute(target, "inFIFODepths", depths)

    def replace_out_fifo_depth(self, source: NodeProto, target: NodeProto, value: int) -> None:
        """Replace the outFIFODepth at given index of the source node to value."""
        oidx, _ = get_io_index(source, target)
        depths = self.get_attribute(source, "outFIFODepths")
        depths[oidx] = value
        self.set_attribute(source, "outFIFODepths", depths)

    def set_surrounding_depths(self, model: ModelWrapper, node: NodeProto, new_depth: int) -> None:
        """Set the inFIFODepth and outFIFODepth values of surrounding Mux/Demux nodes."""
        pre = model.find_direct_predecessors(node)
        suc = model.find_direct_successors(node)
        if pre is not None:
            for predecessor in pre:
                self.replace_out_fifo_depth(predecessor, node, new_depth)
        if suc is not None:
            for successor in suc:
                self.replace_in_fifo_depth(node, successor, new_depth)

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Change all FIFO sizes surrounding a Mux or Demux node to their initial
        estimate to avoid deadlocks.
        """
        # Try to load the initial FIFO size data from the model metadata if necessary
        if self.data is None:
            fifo_results_dir = model.get_metadata_prop("fifo_results_dir")
            if fifo_results_dir is None:
                raise FINNInternalError(
                    "No metadata property 'fifo_results_dir' found. Make sure "
                    "to run 'step_set_fifo_depths' with the node-connected simulation first."
                )
            filename = Path(fifo_results_dir) / "initial_fifo_sizes_sim_connected.json"
            self.data = self.try_read_from_file(filename)

        # merge fifos first, otherwise this becomes difficult (they are split in the end again)
        model = model.transform(MergeSuccessiveFIFOs())

        # For every mux/demux, we need to collect all following nodes and adjust their fifo sizes
        while True:
            found_node = None
            for node in model.graph.node:
                if node.op_type in ["CombinedMuxDemux_hls", "Demux_hls"]:
                    found_node = node
                    break
            if found_node is None:
                break

            any_changed = False
            nodes = self.get_all_downstream_nodes(model, found_node)
            for node in nodes:
                if node.op_type == "Mux_hls":
                    log.warning(
                        f"Skipping depth adjustment after Mux_hls node '{node.name}', since this "
                        "FIFO depth is likely set outside of FINN!"
                    )
                    continue
                if "StreamingFIFO" in node.op_type:
                    continue

                # Delivers successors in output order
                suc = model.find_direct_successors(node)
                if suc is None:
                    continue

                for i, successor in enumerate(suc):
                    if "StreamingFIFO" not in successor.op_type:
                        raise FINNInternalError(
                            f"Expected FIFO as {i}th successor of node '{node.name}', "
                            f"but found: type '{successor.op_type}', name '{successor.name}'"
                        )
                    if node.name not in self.data:
                        raise FINNInternalError(
                            f"Could not find FIFO utilization information "
                            f"for node '{node.name}' in data!"
                        )

                    # Adjust value
                    # The new value might be smaller due to the Hardware-aware minimization search
                    # phase of the FIFO sizing algorithm. In this case, we simply
                    # keep the old value. We also set the inFIFODepths and outFIFODepths of the
                    # surrounding nodes accordingly
                    new_depth = self.data[node.name][i]
                    old_value, new_value = self.set_attribute(successor, "depth", new_depth)
                    if old_value < new_value:
                        any_changed = True
                        self.set_surrounding_depths(model, successor, new_depth)
                        increase = (new_value / float(old_value)) * 100.0
                        log.info(
                            f"Adjusted depth of {node.name}: {old_value} -> "
                            f"{new_value} ({increase:.2f}% increase)"
                        )
                    elif old_value > new_value:
                        # Adjust back if the previous value was larger
                        _, _ = self.set_attribute(successor, "depth", old_value)

            # No more transformations to be done
            if not any_changed:
                break

        # Split (probably very large) FIFOs, rerun synthesis
        model = model.transform(SplitLargeFIFOs())
        model = model.transform(PrepareIP(fpgapart=self.part, clk=self.clk))
        model = model.transform(HLSSynthIP())
        return model, False


class InsertMuxDemuxPairForMultiFPGA(Transformation):
    """Insert mux/demux pairs between nodes that cross devices (a change of device_id values
    between nodes). To do so, the transformation collects all device-crossings for all paths
    and creates a pair for every device combination and time that this crossing happens.

    In an Example where two streams (0 and 1) cross from A to B and later again from A to B,
    two pairs are created: one for each crossing.

    Explanation:
    ------------
    To enable sending data from multiple data streams in parallel over a single connection,
    a mux/demux pair is needed. Notably this is independent of the topology used - regardless
    of whether the design is a simple chain, or a complicated tree of connections, as soon as
    two or more streams need to cross from device A to device B in parallel, a mux/demux pair
    is required.
    To collect the correct nodes, we search each path and annotate the number of
    crossings, e.g.: `{(A, B): 1, (C, E): 4}` (this path crossed from A to B once, and
    from C to E four times).
    """

    def __init__(self) -> None:  # noqa
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Walk the graph, collect device crossings and insert muxes between them."""
        raise NotImplementedError()
        return model, False

"""Contains DeMuxBase_hls, AnnotatedMux_hls and AnnotatedDemux_hls.

AnnotatedMux and AnnotatedDemux can be inserted one after another into the graph
to bundle several channels in parallel into a single channel. To do so, their configuration
has to be in sync (muxed_bitwidth, MultiplexingStrategy, etc.)
"""

from __future__ import annotations

from enum import Enum
from qonnx.core.datatype import DataType
from typing import TYPE_CHECKING, Any, cast

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.exception import FINNConfigurationError, FINNInternalError

if TYPE_CHECKING:
    from onnx.onnx_ml_pb2 import NodeProto
    from qonnx.core.datatype import BaseDataType
    from qonnx.core.modelwrapper import ModelWrapper


class MultiplexStrategy(str, Enum):
    """Lists all Multiplex strategies that the HLS operator can utilize.

    Useful for checking. The user can pass any string but if we cant
    build an enum variant from it, it's
    invalid and doesn't even have to be passed to HLS IPGen
    """

    ROUND_ROBIN = "ROUND_ROBIN"
    ROUND_ROBIN_BLOCKING = "ROUND_ROBIN_BLOCKING"
    LOAD_BALANCE = "LOAD_BALANCE"
    PRIORITY_LIST = "PRIORITY_LIST"


class DeMuxBase_hls(HWCustomOp, HLSBackend):  # noqa
    """Common parent class for both the Mux and Demux HLS operators.

    Represents a custom op implementing a Muxing / Demuxing operation on several datastreams.
    The Mux timemultiplexes the incoming streams with a certain predefined strategy and attaches
    a header to the data. This header can be used to find out, which stream the data is from.
    The Demux does just that and distributes the data accordingly.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: dict) -> None:
        """Construct a new (de)mux op."""
        super().__init__(onnx_node, **kwargs)
        self.code_gen_dict: dict[str, list[str]] = {}

    def get_nodeattr_types(self) -> dict:
        """Return all node attributes, including the ones needed specifically by the (de)mux ops.

        This includes:
            - streamNames: Names of the connected streams
            - streamTypes: The datatype on the streams, given as a string
            - streamNormalShapes: The shapes of the streams as comma seperated ints
            - streamFoldedShapes: As above but for the folded shapes
            - muxed_bitwidth: Width of the stream between Mux and Demux
            - simulation_number_outputs(_index): Explained in `get_number_output_values()`
        """
        my_attrs = {
            "streamNames": ("strings", True, []),
            "streamTypes": ("strings", True, []),
            "streamNormalShapes": ("strings", True, []),
            "streamFoldedShapes": ("strings", True, []),
            "streamWidths": ("ints", True, []),
            # Incoming/Outgoing bandwith
            "muxed_bitwidth": ("i", True, 0),
            # For simulation
            "simulation_number_outputs": ("ints", False, []),
            "simulation_number_outputs_index": ("i", False, 0),
        }
        my_attrs.update(HWCustomOp.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def set_sim_output_numbers(self, values: list[int], reset_index: bool = True) -> None:
        """Set the simulation_number_outputs field (and it's index).

        Args:
            values: Values to write into the field. Overwrites old values.
            reset_index: If `True` (default), also reset the index of the current value.
        """
        if any(type(val) is not int for val in values):
            raise FINNInternalError(
                f"Could not set simulation_number_outputs for node "
                f"{self.onnx_node.name}, since not all values are "
                f"integers: {values}"
            )
        self.set_nodeattr("simulation_number_outputs", values)
        if reset_index:
            self.set_nodeattr("simulation_number_outputs_index", 0)

    def has_sim_output_numbers(self) -> bool:
        """Return whether this operator has simulation specific output number values."""
        try:
            nums = self.get_nodeattr("simulation_number_outputs")
            index = self.get_nodeattr("simulation_number_outputs_index")
        except Exception:
            return False
        return not (nums in [None, []] or index is None)

    def get_number_output_values(self) -> int:
        """Get the number of outputs required for a full sample.

        This method works differently for this operator. Because we only know which channel sends
        when at runtime, for simulation we need to give this information explictly so that
        the simulation runs as expected. Use `set_sim_output_numbers(...)` to pass these numbers.
        When `get_number_output_values()` is called in simulation, and numbers were provided, it
        automatically cycles through them. If none are given, the fallback value 1 is returned.

        >>> import onnx.helper as oh
        >>> mux = AnnotatedMux_hls(oh.make_node("", [], []))
        >>> issubclass(AnnotatedMux_hls, DeMuxBase_hls)
        True
        >>> isinstance(mux, DeMuxBase_hls)
        True
        >>> mux.set_sim_output_numbers([2,3,4])
        >>> mux.get_number_output_values()
        2
        >>> mux.get_number_output_values()
        3

        Returns:
            int: The number of stream transactions needed for the current sample.

        Raises:
            FINNInternalError: Raised if the method is called more often than numbers were given.
        """
        if self.has_sim_output_numbers:
            index = 0
            try:
                index = self.get_nodeattr("simulation_number_outputs_index")
            except Exception as e:
                raise FINNInternalError(
                    f"Node {self.onnx_node.name}: Current sim output number "
                    "cannot be retrieved: Index nodeattr missing!"
                ) from e
            assert type(index) is int
            try:
                values = self.get_nodeattr("simulation_number_outputs")
            except Exception as e:
                raise FINNInternalError(
                    f"Node {self.onnx_node.name}: Cannot retrieve sim output "
                    "number, since no numbers were given!"
                ) from e
            if index >= len(values):
                raise FINNInternalError(
                    f"Node {self.onnx_node.name}: Cannot retrieve simulation "
                    f"output number at index {index}, since only "
                    f"{len(values)} numbers were given!"
                )
            self.set_nodeattr("simulation_number_outputs_index", index + 1)
            return values[index]
        return 1

    # CODE GEN FUNCTIONS
    # Rest are overriden in subclass
    def global_includes(self) -> None:
        """Include the annotated_mux HLS header."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "annotated_mux.hpp"']

    def defines(self, _: str) -> None:
        """Set defines."""
        self.code_gen_dict["$DEFINES$"] = []

    def _blackboxfunction(self, param_prefix: str) -> None:
        """Define header of the blackboxfunction.

        Args:
            param_prefix: Can be "in" or "out", and is used to set
                            the types of the function arguments
        Raises:
            FINNInternalError: Raised if param_prefix is neither in nor out

        """
        # TODO: Check: To make the channel_name and actual input/producer of the node match,
        # both must be in the same order
        channel_data = self.get_channel_data()
        header = f"void {self.onnx_node.name}(\n"
        for i, (_, _, bitwidth) in enumerate(channel_data):
            # TODO: What about int instead of uint?
            # TODO: Do we need to keep the in0_V, in1_V, ... name scheme?
            header += f"hls::stream<ap_uint<{bitwidth}>> &{param_prefix}{i}_V"
            header += ",\n"
        network_prefix = None
        if param_prefix == "in":
            network_prefix = "out"
        elif param_prefix == "out":
            network_prefix = "in"
        else:
            raise FINNInternalError(
                "Annotated(De)Mux_hls' blackboxfunction can only receive in or out as "
                "param_prefixes, for Mux and Demux respectively!"
            )
        header += (
            "hls::stream<ap_uint<"
            + str(self.get_nodeattr("muxed_bitwidth"))
            + f">> &{network_prefix}0_V"
        )
        header += ")\n"
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [header]

    def _pragmas(self, param_prefix: str) -> None:
        """Set pragmas for the blackboxfunction. This mainly refers to the HLS INTERFACE pragma.

        Args:
            param_prefix: Set to in/out to match the arguments of the function declared in
                            _blackboxfunction(...)

        Raises:
            FINNInternalError: Raised if param_prefix is neither in nor out

        """
        network_prefix = None
        if param_prefix == "in":
            network_prefix = "out"
        elif param_prefix == "out":
            network_prefix = "in"
        else:
            raise FINNInternalError(
                "Annotated(De)Mux_hls' blackboxfunction can only receive in or out as "
                "param_prefixes, for Mux and Demux respectively!"
            )
        pragmas = []
        channel_data = self.get_channel_data()
        for i in range(len(channel_data)):
            pragmas.append(f"#pragma HLS INTERFACE axis port={param_prefix}{i}_V")
        pragmas.append(f"#pragma HLS INTERFACE axis port={network_prefix}0_V")
        pragmas.append("#pragma HLS INTERFACE ap_ctrl_none port=return")
        self.code_gen_dict["$PRAGMAS$"] = pragmas

    def _docompute(
        self, function_name: str, add_strategy_template_param: bool, param_prefix: str
    ) -> None:
        """Set the DOCOMPUTE of the code gen dict to the correct string.

        Args:
            function_name: The function to be called as compute. Must be one of the Mux/Demux
                            functions from annotated_mux.hls
            add_strategy_template_param: Whether to add the chosen multiplex strategy as a template
                                            parameter to the function call. Usually True for the Mux
                                            which needs to know the strategy, and False for the
                                            Demux, which only needs to read the header of the
                                            incoming data.
            param_prefix: Can be in/out. Similarly to _blackboxfunction and _pragmas determines
                            the function calls argument types.

        Raises:
            FINNInternalError: Raised if param_prefix is neither in nor out
            FINNConfigurationError: Raised if the MultiplexStrategy read from the node
                                    attributes does not exist.

        """
        strategy: MultiplexStrategy | None = None
        try:
            strat_string = self.get_nodeattr("multiplexStrategy")
        except AttributeError:
            strat_string = ""
        if add_strategy_template_param:
            try:
                strategy = MultiplexStrategy(strat_string)
            except ValueError:
                raise FINNConfigurationError from ValueError(
                    f"Cannot create multiplexer with strategy unknown "
                    f"strategy {strat_string}. Available strategies are: "
                    f"{[s.value for s in MultiplexStrategy]}"
                )
        if add_strategy_template_param:
            assert strategy is not None

        network_prefix = None
        if param_prefix == "in":
            network_prefix = "out"
        elif param_prefix == "out":
            network_prefix = "in"
        else:
            raise FINNInternalError(
                "Annotated(De)Mux_hls' blackboxfunction can only receive in or out as "
                "param_prefixes, for Mux and Demux respectively!"
            )
        channel_data = self.get_channel_data()
        body = f"{function_name}<"
        if add_strategy_template_param:
            body += "MultiplexStrategy::" + strategy.value + ", "
        body += str(self.get_nodeattr("muxed_bitwidth"))
        body += ", "
        # TODO: ap_int
        body += ", ".join([f"ap_uint<{bitwidth}>" for _, _, bitwidth in channel_data])
        body += f">({network_prefix}0_V, "
        body += ", ".join(
            [f"{param_prefix}{i}_V" for i, (_, channel_name, _) in enumerate(channel_data)]
        )
        body += ");"
        self.code_gen_dict["$DOCOMPUTE$"] = [body]

    # END: CODE GEN FUNCTIONS

    def get_largest_stream_width(self) -> int:
        """Return the largest bitwidth of any incoming/outgoing streams."""
        return max(map(int, self.get_nodeattr("streamWidths")))

    def _get_stream_normal_shape(self, index: int) -> tuple:
        """Return the normal shape of one of the communicating streams."""
        normal_shapes = [
            self._parse_shape_string(shape) for shape in self.get_nodeattr("streamNormalShapes")
        ]
        if index >= len(normal_shapes):
            raise FINNInternalError(
                f"Tried accessing the normal shape of stream nr. {index}, "
                "but this (de)mux knows only {len(normal_shapes)} shapes!"
            )
        return tuple(normal_shapes[index])

    def _get_stream_folded_shape(self, index: int) -> tuple:
        """Return the folded shape of one of the communicating streams."""
        folded_shapes = [
            self._parse_shape_string(shape) for shape in self.get_nodeattr("streamFoldedShapes")
        ]
        if index >= len(folded_shapes):
            raise FINNInternalError(
                f"Tried accessing the normal shape of stream nr. {index}, "
                "but this (de)mux knows only {len(folded_shapes)} shapes!!"
            )
        return tuple(folded_shapes[index])

    def _get_stream_datatype(self, index: int) -> BaseDataType:
        """Get the qonnx datatype of the given stream."""
        dtypes = self.get_nodeattr("streamTypes")
        if index >= len(dtypes):
            raise FINNInternalError(
                f"Tried accessing the datatype of stream nr. {index}, "
                "but this (de)mux knows only {len(dtypes)} dtypes!"
            )
        return DataType[dtypes[index]]

    def _parse_shape_string(self, shape: str) -> list[int]:
        """Convert the node attribute string "1,2,5" into the shape [1, 2, 5] as an integer list."""
        return [int(shape_split) for shape_split in shape.split(",")]

    def _get_stream_width(self, index: int) -> int:
        """Get the width of the given stream."""
        widths = list(map(int, self.get_nodeattr("streamWidths")))
        if index >= len(widths):
            raise FINNInternalError(
                f"Cannot get stream width of stream index {index}, "
                f"we only have {len(widths)} streams!"
            )
        return widths[index]

    def get_channel_data(self) -> list[tuple[int, str, int]]:
        """Return a list of tuples that describe all signals.

        Runs some asserts and sanity checks on the attribute data that describes the
        connected streams. If no issues are found, collects the metadata into tuples.
        >>> import onnx.helper as oh
        >>> mux = AnnotatedMux_hls(oh.make_node("", [], [], multiplexStrategy="ROUND_ROBIN", streamNames=["s0"], streamTypes=["UINT2"], streamNormalShapes=["1,2,10"], streamFoldedShapes=["1,2,10"], streamWidths=[40], muxed_bitwidth=512))
        >>> mux.get_channel_data()[0]
        (0, 's0', 40)

        Returns:
            (index, channel_name, bitwidth): This tuple is returned for all streams.

        Raises:
            AssertionError: If any sanity check fails.

        """  # noqa: E501
        streamNames = self.get_nodeattr("streamNames")
        streamWidths = [int(w) for w in self.get_nodeattr("streamWidths")]
        streamTypes = self.get_nodeattr("streamTypes")
        streamNormalShapes = [shape.split(",") for shape in self.get_nodeattr("streamNormalShapes")]
        streamFoldedShapes = [shape.split(",") for shape in self.get_nodeattr("streamFoldedShapes")]
        muxed = self.get_nodeattr("muxed_bitwidth")
        assert muxed is not None
        muxedBitwidth = int(cast(int, self.get_nodeattr("muxed_bitwidth")))
        assert len(streamNames) == len(streamWidths)
        assert len(streamWidths) == len(streamTypes)
        assert len(streamTypes) == len(streamNormalShapes)
        assert len(streamNormalShapes) == len(streamFoldedShapes)
        # TODO: Add DWC automatically in the transition
        # that inserts the (De)mux custom ops
        assert max(streamWidths) <= muxedBitwidth, (
            f"The communication stream for "
            f"this Mux can only accept widths up to {muxedBitwidth}, "
            f"while the largest input stream has a width of {max(streamWidths)}. "
            f"Consider addings a datawidth converter!"
        )
        return [(i, streamNames[i], streamWidths[i]) for i in range(len(streamNames))]


class AnnotatedMux_hls(DeMuxBase_hls):  # noqa
    """A (time)-multiplex HLS operator.

    Simulation: In case of simulations, set sim_output_numbers to the number of stream transactions
    that each sample in the simulation will take. If we get 2 samples from StreamA, which both
    require 2 network transactions, and 1 sample that requires 3, this value should be set
    to [2, 2, 3]. If this is set, get_number_output_values() will automatically cycle to the next
    number after being read.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: Any) -> None:  # noqa: D107
        super().__init__(onnx_node, **kwargs)

    def get_folded_input_shape(self, ind: int = 0) -> list[int] | tuple:  # noqa: D102
        return self._get_stream_folded_shape(ind)

    def get_folded_output_shape(self, ind: int = 0) -> list[int] | tuple:  # noqa: D102, ARG002
        return self.get_normal_output_shape()

    def get_normal_input_shape(self, ind: int = 0) -> list[int] | tuple:  # noqa: D102
        return self._get_stream_normal_shape(ind)

    def get_normal_output_shape(self, ind: int = 0) -> list[int] | tuple:  # noqa: ARG002
        """Return the normal output shape. For the mux this is (1, muxed_bitwidth).

        >>> import onnx.helper as oh
        >>> mux = AnnotatedMux_hls(oh.make_node("AnnotatedMux", [], [], muxed_bitwidth=120))
        >>> mux.get_normal_output_shape()
        (1, 120)
        """
        return (1, self.get_outstream_width())

    def get_instream_width(self, ind: int = 0) -> int:  # noqa: D102
        return self._get_stream_width(ind)

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: D102, ARG002
        muxed = self.get_nodeattr("muxed_bitwidth")
        if muxed is None:
            raise FINNInternalError(
                "Tried accessing required attribute muxed_bitwidth of an "
                "AnnotatedMux node, but such an attribute does not exist!"
            )
        return int(cast(int, muxed))  # noqa: TC006

    def get_input_datatype(self, ind: int) -> BaseDataType:  # noqa: D102
        return self._get_stream_datatype(ind)

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: D102, ARG002
        # TODO
        # Needs to be unsigned, but has no real usage here since we only
        # do low-level operations on this data without actual meaning for
        # the datas contents
        return DataType["UINT1"]

    def infer_node_datatype(self, model: ModelWrapper) -> None:  # noqa: D102
        model.set_tensor_datatype(self.onnx_node.output[0], self.get_output_datatype())

    def execute_node(self, context: Any, graph: Any) -> None:  # noqa: D102
        HLSBackend.execute_node(self, context, graph)

    def get_nodeattr_types(self) -> dict:  # noqa: D102
        my_attrs = {
            # What strategy the multiplexer should use to distribute the data
            "multiplexStrategy": ("s", True, "ROUND_ROBIN"),
        }
        my_attrs.update(DeMuxBase_hls.get_nodeattr_types(self))
        return my_attrs

    def pragmas(self) -> None:  # noqa: D102
        return self._pragmas(param_prefix="in")

    def blackboxfunction(self) -> None:  # noqa: D102
        self._blackboxfunction(param_prefix="in")

    def docompute(self) -> None:  # noqa: D102
        self._docompute(
            "AnnotatedMultiplex::StreamingNetworkMultiplex",
            add_strategy_template_param=True,
            param_prefix="in",
        )


class AnnotatedDemux_hls(DeMuxBase_hls):  # noqa
    """Demux HLS operator. Usage requires a preceeding Mux HLS Operator.

    Simulation: In case of simulations, set sim_output_numbers to the number of stream transactions
    that each sample in the simulation will take. If we get 2 samples from StreamA, which both
    require 2 network transactions, and 1 sample that requires 3, this value should be set
    to [2, 2, 3]. If this is set, get_number_output_values() will automatically cycle to the next
    number after being read
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: Any) -> None:  # noqa: D107
        super().__init__(onnx_node, **kwargs)

    def get_folded_input_shape(self, ind: int = 0) -> tuple:  # noqa: D102, ARG002
        return self.get_normal_input_shape()

    def get_folded_output_shape(self, ind: int = 0) -> tuple:  # noqa: D102
        return self._get_stream_folded_shape(ind)

    def get_normal_input_shape(self, ind: int = 0) -> tuple:  # noqa: ARG002
        """Return the normal input shape. For the demux this is (1, muxed_bitwidth).

        >>> import onnx.helper as oh
        >>> mux = AnnotatedDemux_hls(oh.make_node("AnnotatedDemux", [], [], muxed_bitwidth=120))
        >>> mux.get_normal_input_shape()
        (1, 120)
        """
        return (1, self.get_instream_width())

    def get_normal_output_shape(self, ind: int = 0) -> tuple:  # noqa: D102
        return self._get_stream_normal_shape(ind)

    def get_instream_width(self, ind: int = 0) -> int:  # noqa: D102, ARG002
        bitwidth = self.get_nodeattr("muxed_bitwidth")
        if bitwidth is None:
            raise FINNInternalError(
                "AnnotatedDemux_hls operator does not have it's muxed_bitwidth "
                "attributes set, which is required!"
            )
        return int(cast(int, bitwidth))  # noqa: TC006

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: D102
        return int(self.get_nodeattr("streamWidths")[ind])

    def get_output_datatype(self, ind: int) -> BaseDataType:  # noqa: D102
        return self._get_stream_datatype(ind)

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: D102, ARG002
        # TODO
        # Needs to be unsigned, but has no real usage here since we only
        # do low-level operations on this data without actual meaning for
        # the datas contents
        return DataType["UINT1"]

    def infer_node_datatype(self, model: ModelWrapper) -> None:  # noqa: D102
        model.set_tensor_datatype(self.onnx_node.input[0], self.get_input_datatype())

    def execute_node(self, context: Any, graph: Any) -> None:  # noqa: D102
        HLSBackend.execute_node(self, context, graph)

    def pragmas(self) -> None:  # noqa: D102
        return self._pragmas(param_prefix="out")

    def blackboxfunction(self) -> None:  # noqa: D102
        self._blackboxfunction(param_prefix="out")

    def docompute(self) -> None:  # noqa: D102
        self._docompute(
            "AnnotatedDemultiplex::StreamingNetworkDemultiplex",
            add_strategy_template_param=False,
            param_prefix="out",
        )

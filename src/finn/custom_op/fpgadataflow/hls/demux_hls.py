from __future__ import annotations

from enum import Enum
from math import prod
from onnx.onnx_ml_pb2 import NodeProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import Any

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.exception import FINNConfigurationError, FINNInternalError


class MultiplexStrategy(str, Enum):
    """Useful for checking. The user can pass any string but if we cant
    build an enum variant from it, it's
    invalid and doesn't even have to be passed to HLS IPGen"""

    ROUND_ROBIN = "ROUND_ROBIN"
    ROUND_ROBIN_BLOCKING = "ROUND_ROBIN_BLOCKING"
    LOAD_BALANCE = "LOAD_BALANCE"
    PRIORITY_LIST = "PRIORITY_LIST"


class DeMuxBase_hls(HWCustomOp, HLSBackend):  # noqa:
    """Represents a custom op implementing a Muxing / Demuxing operation on several datastreams.
    The Mux timemultiplexes the incoming streams with a certain predefined strategy and attaches
    a header to the data. This header can be used to find out, which stream the data is from.
    The Demux does just that and distributes the data accordingly.
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: dict) -> None:
        super().__init__(onnx_node, **kwargs)
        self.code_gen_dict: dict[str, list[str]] = {}

    def update_network_bitwidth(self, bits: int) -> None:
        self.set_nodeattr("muxed_bitwidth", bits)

    def get_nodeattr_types(self) -> dict:
        my_attrs = {
            "streamNames": ("strings", True, []),
            "streamTypes": ("strings", True, []),
            "streamNormalShapes": ("strings", True, []),
            "streamFoldedShapes": ("strings", True, []),
            "streamWidths": ("ints", True, []),
            # Incoming/Outgoing bandwith
            "muxed_bitwidth": ("i", True, 0),
        }
        my_attrs.update(HWCustomOp.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    # CODE GEN FUNCTIONS
    # Rest are overriden in subclass

    def global_includes(self) -> None:
        """Include the concat library, originally meant for only performing
        channel concatenation"""
        self.code_gen_dict["$GLOBALS$"] = ['#include "annotated_mux.hpp"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = ""

    def _blackboxfunction(self, param_prefix: str):
        # TODO: Check: To make the channel_name and actual input/producer of the node match,
        # both must be in the same order
        channel_data = self.get_channel_data()
        header = f"void {self.onnx_node.name}(\n"
        for i, (index, channel_name, bitwidth) in enumerate(channel_data):
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

    def _pragmas(self, param_prefix: str):
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
        for i, (index, channel_name, bitwidth) in enumerate(channel_data):
            pragmas.append(f"#pragma HLS INTERFACE axis port={param_prefix}{i}_V")
        pragmas.append(f"#pragma HLS INTERFACE axis port={network_prefix}0_V")
        pragmas.append("#pragma HLS INTERFACE ap_ctrl_none port=return")
        self.code_gen_dict["$PRAGMAS$"] = pragmas

    def _docompute(self, function_name: str, add_strategy_template_param: bool, param_prefix: str):
        """Will be called by the respective subclasses"""
        strategy = None
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
            body += "MultiplexStrategy::" + strategy.value + ","
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
        """Return the largest bitwidth of any incoming/outgoing streams"""
        return max(map(int, self.get_nodeattr("streamWidths")))

    def _get_stream_normal_shape(self, index: int) -> tuple:
        """Return the normal shape of one of the communicating streams"""
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
        """Return the folded shape of one of the communicating streams"""
        folded_shapes = [
            self._parse_shape_string(shape) for shape in self.get_nodeattr("streamFoldedShapes")
        ]
        if index >= len(folded_shapes):
            raise FINNInternalError(
                f"Tried accessing the normal shape of stream nr. {index}, "
                "but this (de)mux knows only {len(folded_shapes)} shapes!!"
            )
        return tuple(folded_shapes[index])

    def _get_stream_datatype(self, index: int) -> DataType:
        """Get the qonnx datatype of the given stream"""
        dtypes = self.get_nodeattr("streamTypes")
        if index >= len(dtypes):
            raise FINNInternalError(
                f"Tried accessing the datatype of stream nr. {index}, "
                "but this (de)mux knows only {len(dtypes)} dtypes!"
            )
        return DataType[dtypes[index]]

    def _parse_shape_string(self, shape: str) -> list[int]:
        """Convert the node attribute string "1,2,5" into the shape [1, 2, 5] as an integer list"""
        return [int(shape_split) for shape_split in shape.split(",")]

    def _get_largest_normal_shape_prod(self) -> list[int]:
        """Search the shape that has the biggest product of all its elements,
        and return the product. (1, 10, 5, 20) => 1*10*5*20 = 1000"""
        prods = [
            prod(self._parse_shape_string(shape))
            for shape in self.get_nodeattr("streamNormalShapes")
        ]
        return max(prods)

    def _get_stream_width(self, index: int) -> int:
        """Get the width of the given stream"""
        widths = list(map(int, self.get_nodeattr("streamWidths")))
        if index >= len(widths):
            raise FINNInternalError(
                f"Cannot get stream width of stream index {index}, "
                f"we only have {len(widths)} streams!"
            )
        return widths[index]

    def get_channel_data(self) -> list[tuple[int, str, int]]:
        """Return a list of tuples that describe all signals: (index, channel_name, bitwidth)"""
        streamNames = self.get_nodeattr("streamNames")
        streamWidths = [int(w) for w in self.get_nodeattr("streamWidths")]
        streamTypes = self.get_nodeattr("streamTypes")
        streamNormalShapes = [shape.split(",") for shape in self.get_nodeattr("streamNormalShapes")]
        streamFoldedShapes = [shape.split(",") for shape in self.get_nodeattr("streamFoldedShapes")]
        muxedBitwidth = int(self.get_nodeattr("muxed_bitwidth"))
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
    def __init__(self, onnx_node: NodeProto, **kwargs: Any) -> None:
        super().__init__(onnx_node, **kwargs)

    def get_folded_input_shape(self, ind: int) -> list[int] | tuple:
        return self._get_stream_folded_shape(ind)

    def get_folded_output_shape(self, _: int = 0) -> list[int] | tuple:
        return self.get_normal_output_shape()

    def get_normal_input_shape(self, ind: int) -> list[int] | tuple:
        return self._get_stream_normal_shape(ind)

    def get_normal_output_shape(self, _: int = 0) -> list[int] | tuple:
        return (1, self.get_outstream_width())

    def get_instream_width(self, ind: int = 0) -> int:
        return self._get_stream_width(ind)

    def get_outstream_width(self, _: int = 0) -> int:
        return int(self.get_nodeattr("muxed_bitwidth"))

    def get_input_datatype(self, ind) -> DataType:
        return self._get_stream_datatype(ind)

    def get_output_datatype(self, _: int = 0) -> DataType:
        # Needs to be unsigned, but has no real usage here since we only
        # do low-level operations on this data without actual meaning for
        # the datas contents
        return DataType["UINT1"]

    def get_number_output_values(self) -> int:
        return 1  # self._get_largest_normal_shape_prod()

    def infer_node_datatype(self, model: ModelWrapper):
        model.set_tensor_datatype(self.onnx_node.output[0], self.get_output_datatype())

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)

    def get_nodeattr_types(self) -> dict:
        my_attrs = {
            # What strategy the multiplexer should use to distribute the data
            "multiplexStrategy": ("s", True, "ROUND_ROBIN"),
        }
        my_attrs.update(DeMuxBase_hls.get_nodeattr_types(self))
        return my_attrs

    def pragmas(self) -> None:
        return self._pragmas(param_prefix="in")

    def blackboxfunction(self) -> None:
        self._blackboxfunction(param_prefix="in")

    def docompute(self) -> None:
        self._docompute(
            "AnnotatedMultiplex::StreamingNetworkMultiplex",
            add_strategy_template_param=True,
            param_prefix="in",
        )


class AnnotatedDemux_hls(DeMuxBase_hls):  # noqa
    def __init__(self, onnx_node: NodeProto, **kwargs: Any) -> None:
        super().__init__(onnx_node, **kwargs)

    def get_folded_input_shape(self, ind):
        return self.get_normal_input_shape()

    def get_folded_output_shape(self, ind=0):
        return self._get_stream_folded_shape(ind)

    def get_normal_input_shape(self, _: int = 0):
        return (1, self.get_instream_width())

    def get_normal_output_shape(self, ind=0):
        return self._get_stream_normal_shape(ind)

    def get_instream_width(self, _: int = 0):
        return int(self.get_nodeattr("muxed_bitwidth"))

    def get_outstream_width(self, i: int = 0):
        return int(self.get_nodeattr("streamWidths")[i])

    def get_output_datatype(self, ind):
        return self._get_stream_datatype(ind)

    def get_input_datatype(self, _: int = 0):
        # Needs to be unsigned, but has no real usage here since we only
        # do low-level operations on this data without actual meaning for
        # the datas contents
        return DataType["UINT1"]

    def get_number_output_values(self) -> int:
        return 1  # self._get_largest_normal_shape_prod()

    def infer_node_datatype(self, model: ModelWrapper):
        model.set_tensor_datatype(self.onnx_node.input[0], self.get_input_datatype())

    def execute_node(self, context, graph):
        HLSBackend.execute_node(self, context, graph)

    def update_channel_data(self, model: ModelWrapper) -> None:
        channels = []
        for i, onode in enumerate(model.get_direct_successors(self)):
            channels.append((i, onode.name, onode.get_output_datatype().bitwidth()))

    def pragmas(self) -> None:
        return self._pragmas(param_prefix="out")

    def blackboxfunction(self) -> None:
        self._blackboxfunction(param_prefix="out")

    def docompute(self) -> None:
        self._docompute(
            "AnnotatedDemultiplex::StreamingNetworkDemultiplex",
            add_strategy_template_param=False,
            param_prefix="out",
        )

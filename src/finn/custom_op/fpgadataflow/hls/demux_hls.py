from enum import Enum
from onnx.onnx_ml_pb2 import NodeProto
from qonnx.core.modelwrapper import ModelWrapper

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.exception import FINNInternalError


class MultiplexStrategy(Enum):
    # Round robin strategy. Always advance to next. Good if high load on all channels
    ROUND_ROBIN = 0

    # Round robin, but only advance if data was sent this cycle. If the chosen source has no
    # available data, go in a circle until the first source with data available is found
    ROUND_ROBIN_FLEXIBLE = 1

    # Always try to use the source from the top of the list. If not available, try the next in order
    PRIORITY_LIST = 2

    # Record the last few transactions and observe, which stream sends the most data.
    # Use this stream with the highest priority
    LOAD_BALANCE = 3


class DeMuxBase_hls(HWCustomOp, HLSBackend):  # noqa
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
            # A space seperated list of names for the incoming/outgoing streams
            "streams": ("s", True, ""),
            # List of bitwidths of the channels
            # TODO: Get from node.get_input/output_datatype().bitwidth
            "bitwidths": ("ints", True, []),
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
        self.code_gen_dict["$GLOBALS$"] = ['#include "concat.hpp"']

    def defines(self):
        pass

    def blackboxfunction(self):
        # TODO: Check: To make the channel_name and actual input/producer of the node match,
        # both must be in the same order
        channel_data = self.get_channel_data()
        header = f"void {self.onnx_node.name}(\n"
        for index, channel_name, bitwidth in channel_data:
            # TODO: What about int instead of uint?
            # TODO: Do we need to keep the in0_V, in1_V, ... name scheme?
            header += f"hls::stream<ap_uint<{bitwidth}>> &{channel_name}"
            header += ",\n"
        header += "hls::stream<ap_uint<" + self.get_nodeattr("muxed_bitwidth") + ">> &network"
        header += ")\n"
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [header]

    def pragmas(self):
        pragmas = []
        channel_data = self.get_channel_data()
        for index, channel_name, bitwidth in channel_data:
            pragmas.append(f"#pragma HLS INTERFACE axis port={channel_name}")
        pragmas.append("#pragma HLS INTERFACE axis port=network")
        pragmas.append("#pragma HLS INTERFACE ap_ctrl_none port=return")
        self.code_gen_dict["$PRAGMAS$"] = pragmas

    def _docompute(self, function_name: str):
        """Will be called by the respective subclasses"""
        channel_data = self.get_channel_data()
        body = f"{function_name}<MultiplexStrategy::ROUND_ROBIN,"
        body += str(self.get_nodeattr("muxed_bitwidth"))
        body += ", "
        # TODO: ap_int
        body += ", ".join([f"ap_uint<{bitwidth}>" for _, _, bitwidth in channel_data])
        body += ">(network, "
        body += ", ".join([f"{channel_name}" for _, channel_name, _ in channel_data])
        body += ");"
        self.code_gen_dict["$DOCOMPUTE$"] = [body]

    # END: CODE GEN FUNCTIONS

    def get_channel_data(self) -> list[tuple[int, str, int]]:
        """Return a list of tuples that describe all signals: (index, channel_name, bitwidth)"""
        names = str(self.get_nodeattr("streams")).split(" ")
        bitwidths = self.get_nodeattr("bitwidths")
        if len(names) != len(bitwidths):
            raise FINNInternalError(
                f"Cannot use (De)Mux Arbiter custom_op: The number of "
                f"channels ({len(names)}) does not match the number of "
                f"bitwidths given ({len(bitwidths)})!"
            )
        return [(i, names[i], bitwidths[i]) for i in range(len(names))]

    def code_generation_ipgen(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:
        # TODO: Implement
        pass


class AnnotatedMux_hls(DeMuxBase_hls):  # noqa
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_folded_input_shape():
        pass

    def get_folded_output_shape():
        pass

    def get_normal_input_shape():
        pass

    def get_normal_output_shape():
        pass

    def get_instream_width():
        pass

    def get_outstream_width():
        pass

    def get_input_datatype():
        pass

    def get_output_datatype():
        pass

    def get_number_output_values():
        pass

    def infer_node_datatype():
        pass

    def execute_node():
        pass

    def update_channel_data(self, model: ModelWrapper) -> None:
        channels = []
        for i, inode in enumerate(model.get_direct_predecessors(self)):
            channels.append((i, inode.name, inode.get_output_datatype().bitwidth()))

    def get_nodeattr_types(self) -> dict:
        my_attrs = {
            # What strategy the multiplexer should use to distribute the data
            "strategy": ("i", True, MultiplexStrategy.ROUND_ROBIN_FLEXIBLE),
        }
        my_attrs.update(DeMuxBase_hls.get_nodeattr_types(self))
        return my_attrs

    def docompute(self) -> None:
        self._docompute("AnnotatedMultiplex::StreamingNetworkMultiplex")


class AnnotatedDemux_hls(DeMuxBase_hls):  # noqa
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_folded_input_shape():
        pass

    def get_folded_output_shape():
        pass

    def get_normal_input_shape():
        pass

    def get_normal_output_shape():
        pass

    def get_instream_width():
        pass

    def get_outstream_width():
        pass

    def get_input_datatype():
        pass

    def get_output_datatype():
        pass

    def get_number_output_values():
        pass

    def infer_node_datatype():
        pass

    def execute_node():
        pass

    def update_channel_data(self, model: ModelWrapper) -> None:
        channels = []
        for i, onode in enumerate(model.get_direct_successors(self)):
            channels.append((i, onode.name, onode.get_output_datatype().bitwidth()))

    def docompute(self) -> None:
        self._docompute("AnnotatedDemultiplex::StreamingNetworkDemultiplex")

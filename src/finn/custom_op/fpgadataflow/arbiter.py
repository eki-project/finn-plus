from abc import abstractmethod
from enum import Enum
from math import log2
from onnx.onnx_ml_pb2 import NodeProto
from qonnx.core.modelwrapper import ModelWrapper
from typing import Any

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.exception import FINNInternalError, FINNUserError


class ArbiterStrategy(Enum):
    # Round robin strategy. Always advance to next. Good if high load on all channels
    ROUND_ROBIN = 0

    # Round robin, but only advance if data was sent this cycle. If the chosen source has no
    # available data, go in a circle until the first source with data available is found
    ROUND_ROBIN_FLEXIBLE = 1

    # Always try to use the source from the top of the list. If not available, try the next in order
    PRIORITY_LIST = 2


class ArbiterMode(Enum):
    MUX = 0
    DEMUX = 1


class Arbiter(HWCustomOp):
    """Base class for a (De)Mux arbiter custom op."""

    def __init__(self, onnx_node: NodeProto, **kwargs: Any) -> None:
        super().__init__(onnx_node, **kwargs)

    @abstractmethod
    def get_mode(self) -> ArbiterMode:
        """Return whether this custom op is a muxing arbiter or a demuxer"""

    def get_nodeattr_types(self) -> dict:
        my_attrs = {
            # A space seperated list of names for the incoming/outgoing streams
            "channels": ("s", True, ""),
            # List of bitwidths of the channels
            "bitwidths": ("ints", True, []),
            # Incoming/Outgoing bandwith
            "muxed_bitwidth": ("i", True, 0),
        }
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def set_channel_data(self, model: ModelWrapper) -> None:
        """Configure the attributes for the stream names and bitwidths automatically based on
        the passed model"""
        # TODO: Implement here or in the to-Hardware-conversion

    def get_channel_data(self) -> list[tuple[int, str, int]]:
        """Return a list of tuples that describe all signals: (index, channel_name, bitwidth)"""
        names = str(self.get_nodeattr("channels")).split(" ")
        bitwidths = self.get_nodeattr("bitwidths")
        if len(names) != len(bitwidths):
            raise FINNInternalError(
                f"Cannot use (De)Mux Arbiter custom_op: The number of "
                f"channels ({len(names)}) does not match the number of "
                f"bitwidths given ({len(bitwidths)})!"
            )
        return [(i, names[i], bitwidths[i]) for i in range(len(names))]

    def get_header_bitwidth(self) -> int:
        """Return the number of bits required for the header to identify the data source"""
        return int(log2(len(self.get_channel_data())))

    def get_out_bitwidth(self) -> int:
        return int(self.get_nodeattr("muxed_bitwidth"))

    def check_out_bitwidth(self) -> None:
        """Check that the outgoing bitwidth is large enough for the header + the largest data
        from any channel. If not, raise an error"""
        longest_bitwidth = max([data[2] for data in self.get_channel_data()])
        if longest_bitwidth + self.get_header_bitwidth() > self.get_out_bitwidth():
            raise FINNUserError("TODO")

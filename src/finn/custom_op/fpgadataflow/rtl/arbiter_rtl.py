from enum import Enum
from onnx.onnx_ml_pb2 import NodeProto
from typing import Any

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend


class ArbiterStrategy(Enum):
    # Round robin strategy. Always advance to next. Good if high load on all channels
    ROUND_ROBIN = 0

    # Round robin, but only advance if data was sent this cycle. If the chosen source has no
    # available data, go in a circle until the first source with data available is found
    ROUND_ROBIN_FLEXIBLE = 1

    # Always try to use the source from the top of the list. If not available, try the next in order
    PRIORITY_LIST = 2


class Arbiter(HWCustomOp, RTLBackend):
    """Base class for a (De)Mux arbiter custom op."""

    def __init__(self, onnx_node: NodeProto, **kwargs: Any) -> None:
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> dict:
        my_attrs = {
            # A space seperated list of names for the incoming/outgoing streams
            "channels": ("s", True, ""),
            # List of bitwidths of the channels
            "bitwidths": ("ints", True, []),
        }
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs


class MuxArbiter(Arbiter):
    def __init__(self, onnx_node: NodeProto, **kwargs: Any) -> None:
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> dict:
        my_attrs = {"arbiter_strategy": ("int", True, ArbiterStrategy.ROUND_ROBIN_FLEXIBLE)}
        my_attrs.update(Arbiter.get_nodeattr_types(self))
        return super().get_nodeattr_types()


class DeMuxArbiter(Arbiter):
    pass

import os
import shlex
import subprocess
import sys
from abc import abstractmethod
from enum import Enum
from math import log2
from onnx.onnx_ml_pb2 import NodeProto
from pathlib import Path
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


class Arbiter(HWCustomOp, RTLBackend):
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

    def generate_hdl(self, model: ModelWrapper, fpgapart: str, clk: int) -> None:
        # Locate the generator script, python executable and HDL destination
        self.check_out_bitwidth()
        indices, names, widths = self.get_channel_data()
        comm_width = self.get_out_bitwidth()
        script_name = ""
        if self.get_mode() == ArbiterMode.MUX:
            script_name = "generate_mux.py"
        elif self.get_mode() == ArbiterMode.DEMUX:
            script_name = "generate_demux.py"
        else:
            raise FINNInternalError(f"Invalid ArbiterMode given: {self.get_mode()}")
        generator_location = Path(os.environ["FINN_RTLLIB"]) / "arbiter" / script_name
        if not generator_location.exists():
            raise FINNInternalError(
                f"Could not find the Arbiter custom op HDL generator "
                f"script. Searched at: {generator_location}. Please make sure "
                f"FINN_RTLLIB is set correctly."
            )
        if sys.executable in ["", None]:
            raise FINNInternalError(
                "Could not find the python executable via sys.executable. "
                "Is your environment properly configured?"
            )

        # Prepare the generator script call
        call = f'{sys.executable} {generator_location} "'

        # Pass the list of streams for the arbiter
        for i in range(len(indices)):
            call += f"{names[i]} {widths[i]} "
        call += f'" {comm_width} '

        # Where to save the generated code
        call += self.get_rtl_file_list()[0]

        # Execute the script
        subprocess.run(shlex.split(call), stdout=subprocess.DEVNULL)

    def get_rtl_file_list(self, abspath: bool = False) -> list[str]:
        # Where to save the generated files
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        gen_top_module = self.get_nodeattr("gen_top_module")
        if code_gen_dir is None:
            raise FINNInternalError("Arbiter MUX is missing code_gen_dir")
        if gen_top_module is None:
            raise FINNInternalError("Arbiter MUX is missing a gen_top_module name")
        code_gen_dir = Path(str(code_gen_dir))
        if not code_gen_dir.exists():
            raise FINNInternalError(
                f"code_gen_ip_dir was set to {code_gen_dir}, but " f"this directory does not exist!"
            )
        fname = Path()
        if self.get_mode() == ArbiterMode.MUX:
            fname = Path(str(gen_top_module) + "_mux.v")
        elif self.get_mode() == ArbiterMode.DEMUX:
            fname = Path(str(gen_top_module) + "_demux.v")
        p = Path(code_gen_dir / fname)
        if abspath:
            return [str(p.absolute())]
        return [str(p)]


class MuxArbiter(Arbiter):
    def __init__(self, onnx_node: NodeProto, **kwargs: Any) -> None:
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> dict:
        my_attrs = {"arbiter_strategy": ("i", True, ArbiterStrategy.ROUND_ROBIN_FLEXIBLE)}
        my_attrs.update(Arbiter.get_nodeattr_types(self))
        return super().get_nodeattr_types()

    def get_mode(self) -> ArbiterMode:
        return ArbiterMode.MUX


class DeMuxArbiter(Arbiter):
    def __init__(self, onnx_node: NodeProto, **kwargs: Any) -> None:
        super().__init__(onnx_node, **kwargs)

    def get_mode(self) -> ArbiterMode:
        return ArbiterMode.DEMUX

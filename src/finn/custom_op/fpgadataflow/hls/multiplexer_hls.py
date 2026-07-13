"""General purpose Multiplexer."""
import numpy as np
from collections.abc import Sequence
from numpy import typing as npt
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import cast

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.exception import FINNInternalError


def mux_required_width(count: int) -> int:
    """Return number of bits required to count to this number."""
    return count if count < 2 else 1 + mux_required_width(count // 2)


class Multiplexer_hls(HWCustomOp, HLSBackend):
    """Multiplexer for transmitting multiple branches on one stream."""

    def __init__(self, onnx_node: NodeProto, **kwargs) -> None:  # noqa
        """Create a mux node."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> dict:
        """Node attribute defs."""
        attrs = HWCustomOp.get_nodeattr_types(self)
        attrs.update(
            {
                # Describes the general way in which the mux workws
                "muxVariant": ("s", True, "static_schedule"),
                # Describes the subtype. Round-robin is for example a
                # statically scheduled variant.
                "muxVariantSubtype": ("s", True, "round_robin"),
                "inStreams": ("strings", True, []),
                "inStreamWidths": ("ints", True, []),
                "inStreamDataTypes": ("strings", True, []),
                # A shape is stored as a string with "," separating the tuple elements
                "inStreamFoldedOutputShapes": ("strings", True, []),
                "inStreamNormalOutputShapes": ("strings", True, []),
                "outStream": ("s", True, ""),
            }
        )
        return attrs

    def global_includes(self) -> None:
        """Add the global includes for all mux variants."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "static_mux.hpp"']

    def docompute(self) -> None:
        """Render the mux from a template and insert into the code gen dict."""
        variant = str(self.get_nodeattr("muxVariant"))
        subtype = str(self.get_nodeattr("muxVariantSubtype"))
        in_streams = cast("list[str]", self.get_nodeattr("inStreams"))
        out_stream = str(self.get_nodeattr("outStream"))
        call = ""
        match variant:
            case "static_schedule":
                match subtype:
                    case "round_robin":
                        sequence = ", ".join([str(x) for x in range(len(in_streams))])
                        inputs = ", ".join(in_streams)
                        call = (
                            f"static_mux(std::index_sequence<{sequence}>{{}}, "
                            f"{out_stream}, {inputs});"
                        )
                    case "random":
                        raise NotImplementedError()
                    case _:
                        raise FINNInternalError(
                            f"Unknown subvariant: " f"{subtype} (of mux variant {variant})."
                        )
            case _:
                raise FINNInternalError(f"Unknown mux variant: {variant}")

        self.code_gen_dict["$DOCOMPUTE$"] = [call]

    def pragmas(self) -> None:
        """Add pragmas."""
        self.code_gen_dict["$PRAGMAS$"] = []
        instream_names = cast("list[str]", self.get_nodeattr("inStreams"))
        for i, _ in enumerate(instream_names):
            self.code_gen_dict["$PRAGMAS$"].append(f"  #pragma HLS INTERFACE axis port=in{i}_V")
        self.code_gen_dict["$PRAGMAS$"].append("  #pragma HLS INTERFACE axis port=out0_V")
        self.code_gen_dict["$PRAGMAS$"].append("  #pragma HLS INTERFACE ap_ctrl_none port=return")

    def blackboxfunction(self) -> None:
        """Create the function definition."""
        instream_names = cast("list[str]", self.get_nodeattr("inStreams"))
        instreams = [
            f"hls::stream<{self.get_input_datatype(i).get_hls_datatype_str()}> &in{i}_V"
            for i in range(len(instream_names))
        ]
        outstream = str(self.get_output_datatype().get_hls_datatype_str())
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"void {self.onnx_node.name}({', '.join(instreams)}, "
            f"hls::stream<{outstream}> &out0_V)"
        ]

    def global_includes(self) -> None:  # noqa
        self.code_gen_dict["$GLOBALS$"] = []

    def execute_node(self, context, graph) -> None:  # noqa
        HLSBackend.execute_node(self, context, graph)

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        model.set_tensor_datatype(self.onnx_node.output[0], self.get_output_datatype())

    def get_folded_input_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:
        """Folded input shape is the folded output shape of the incoming streams."""
        return [
            int(i)
            for i in cast("list[str]", self.get_nodeattr("inStreamFoldedOutputShapes"))[ind].split(
                ","
            )
        ]

    def get_folded_output_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:  # noqa
        """The folded output shape is the largest incoming shape
        (since we don't change any shapes)."""  # noqa
        shape_count = len(cast("list[str]", self.get_nodeattr("inStreamFoldedOutputShapes")))
        prods = [np.prod(self.get_folded_input_shape(i)) for i in range(shape_count)]
        return self.get_folded_input_shape(int(np.argmax(prods)))

    def get_normal_output_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:  # noqa
        """The normal output shape is the largest incoming shape
        (since we don't change any shapes)."""  # noqa
        shape_count = len(cast("list[str]", self.get_nodeattr("inStreamNormalOutputShapes")))
        prods = [np.prod(self.get_normal_input_shape(i)) for i in range(shape_count)]
        return self.get_normal_input_shape(int(np.argmax(prods)))

    def get_normal_input_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:
        """Normal input shape is the normal output shape of the incoming streams."""
        return [
            int(i)
            for i in cast("list[str]", self.get_nodeattr("inStreamNormalOutputShapes"))[ind].split(
                ","
            )
        ]

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:
        """Simply return the datatype from the incoming stream without change."""
        return DataType[cast("list[str]", self.get_nodeattr("inStreamDataTypes"))[ind]]

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa
        """Return the output datatype. For integers, select the largest bitwidth + 1, signed, to
        guarantee that all input datatype values can be contained.
        """
        all_input_dts: list[BaseDataType] = [
            DataType[dt] for dt in cast("list[str]", self.get_nodeattr("inStreamDataTypes"))
        ]
        if not all(dt.is_integer() for dt in all_input_dts):
            raise FINNInternalError("Non-integer datatypes in Mux/Demux not supported yet.")
        largest_dt = sorted(all_input_dts, key=lambda x: x.bitwidth(), reverse=True)[0]
        return DataType[f"UINT{largest_dt.bitwidth()+mux_required_width(len(all_input_dts))}"]

    def get_instream_width(self, ind: int = 0) -> int:
        """Simply return the datatype from the incoming stream without change."""
        return cast("list[int]", self.get_nodeattr("inStreamWidths"))[ind]

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa
        """Get the largest input stream width."""
        return max(cast("list[int]", self.get_nodeattr("inStreamWidths")))

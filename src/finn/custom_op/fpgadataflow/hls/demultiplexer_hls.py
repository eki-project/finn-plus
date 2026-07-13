"""General purpose Demultiplexer."""
# ruff: noqa: D102
import numpy as np
from collections.abc import Sequence
from numpy import typing as npt
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.mux_demux import MuxDemux
from finn.util.exception import FINNInternalError


class Demultiplexer_hls(MuxDemux, HLSBackend):
    """Demultiplexer for splitting one into multiple streams."""

    def __init__(self, onnx_node: NodeProto, **kwargs) -> None:  # noqa
        """Create a mux node."""
        super().__init__(onnx_node, **kwargs)

    def global_includes(self) -> None:
        """Add the global includes for all mux variants."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "static_mux.hpp"']

    def docompute(self) -> None:
        """Render the mux from a template and insert into the code gen dict."""
        variant = str(self.get_nodeattr("muxVariant"))
        subtype = str(self.get_nodeattr("muxVariantSubtype"))
        call = ""
        match variant:
            case "static_schedule":
                match subtype:
                    case "round_robin":
                        outputs = ", ".join([f"out{i}_V" for i in range(self.get_stream_count())])
                        call = f"static_demux(in0_V, {outputs});"
                    case "random":
                        raise NotImplementedError()
                    case _:
                        raise FINNInternalError(
                            f"Unknown subvariant: {subtype} (of mux variant {variant})."
                        )
            case _:
                raise FINNInternalError(f"Unknown mux variant: {variant}")

        self.code_gen_dict["$DOCOMPUTE$"] = [call]

    def pragmas(self) -> None:
        """Add pragmas."""
        self.code_gen_dict["$PRAGMAS$"] = []
        for i in range(self.get_stream_count()):
            self.code_gen_dict["$PRAGMAS$"].append(f"  #pragma HLS INTERFACE axis port=out{i}_V")
        self.code_gen_dict["$PRAGMAS$"].append("  #pragma HLS INTERFACE axis port=in0_V")
        self.code_gen_dict["$PRAGMAS$"].append("  #pragma HLS INTERFACE ap_ctrl_none port=return")

    def blackboxfunction(self) -> None:
        """Create the function definition."""
        outstream_parameters = ", ".join(
            [
                f"hls::stream<{self.get_output_datatype(i).get_hls_datatype_str()}> &out{i}_V"
                for i in range(self.get_stream_count())
            ]
        )
        outstream_dt = str(self.get_input_datatype().get_hls_datatype_str())
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"void {self.onnx_node.name}(hls::stream<{outstream_dt}> &in0_V, "
            f"{outstream_parameters})"
        ]

    def global_includes(self) -> None:  # noqa
        self.code_gen_dict["$GLOBALS$"] = []

    def execute_node(self, context, graph) -> None:  # noqa
        HLSBackend.execute_node(self, context, graph)

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        for i in range(len(self.onnx_node.output)):
            model.set_tensor_datatype(self.onnx_node.output[i], self.get_output_datatype(i))

    def get_folded_input_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:  # noqa
        return [1]

    def get_normal_input_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:  # noqa
        return [1]

    def get_folded_output_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:
        return self.get_stream_folded_shape(ind)

    def get_normal_output_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:
        return self.get_stream_normal_shape(ind)

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa
        return self.get_connection_dtype()

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:
        return self.get_stream_dts()[ind]

    def get_instream_width(self, ind: int = 0) -> int:  # noqa
        return self.get_input_datatype().bitwidth()

    def get_outstream_width(self, ind: int = 0) -> int:
        return self.get_stream_widths()[ind]

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


class Demultiplexer_hls(MuxDemux, HLSBackend):
    """Demultiplexer for splitting one into multiple streams."""

    def __init__(self, onnx_node: NodeProto, **kwargs) -> None:  # noqa
        """Create a mux node."""
        super().__init__(onnx_node, **kwargs)

    def get_op_type(self) -> str:
        return "demux"

    def docompute(self) -> None:
        """Render the mux from a template and insert into the code gen dict."""
        outputs = ", ".join([f"out{i}_V" for i in range(self.get_stream_count())])
        self.code_gen_dict["$DOCOMPUTE$"] = [self.render_compute_template(outputs=outputs)]

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

"""Combined mux and demux operator for simulation purposes."""
# ruff: noqa: D102
import numpy as np
from numpy import typing as npt
from onnx import NodeProto
from qonnx.core.datatype import BaseDataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import Sequence

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.mux_demux import MuxDemux


class CombinedMuxDemux(MuxDemux, HLSBackend):
    """Operator that combines a Mux and Demux operators (connected) into
    a single operator. This is done so that functional FIFO sizing can be used.
    If the operators were separate, the constant propagation would disable the
    proper demultiplexing of the network data, causing a model deadlock.

    After FIFO sizing is done, this operator should be separated into different
    components again.
    """

    def __init(self, onnx_node: NodeProto, **kwargs) -> None:  # noqa
        super().__init__(onnx_node, **kwargs)

    def get_op_type(self) -> str:
        return "combined"

    def docompute(self) -> None:
        inputs = ", ".join([f"in{i}_V" for i in range(self.get_stream_count())])
        outputs = ", ".join([f"out{i}_V" for i in range(self.get_stream_count())])
        sequence = ", ".join([str(x) for x in range(self.get_stream_count())])
        self.code_gen_dict["$DOCOMPUTE$"] = [
            self.render_compute_template(
                connection_type=self.get_connection_dtype().get_hls_datatype_str(),
                inputs=inputs,
                outputs=outputs,
                sequence=sequence,
            )
        ]

    def execute_node(self, context, graph) -> None:  # noqa
        HLSBackend.execute_node(self, context, graph)

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        model.set_tensor_datatype(self.onnx_node.output[0], self.get_output_datatype())

    def get_folded_input_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:
        return self.get_stream_folded_shape(ind)

    def get_normal_input_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:
        return self.get_stream_normal_shape(ind)

    def get_folded_output_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:
        return self.get_stream_folded_shape(ind)

    def get_normal_output_shape(self, ind: int = 0) -> Sequence[int] | npt.NDArray[np.int_]:
        return self.get_stream_normal_shape(ind)

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:
        return self.get_stream_dts()[ind]

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:
        return self.get_stream_dts()[ind]

    def get_instream_width(self, ind: int = 0) -> int:
        return self.get_stream_widths()[ind]

    def get_outstream_width(self, ind: int = 0) -> int:
        return self.get_stream_widths()[ind]

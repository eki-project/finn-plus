"""HLSBackend specialization for generic reduce operators: Min, Max, Sum, Product."""

import numpy as np
from onnx import GraphProto

from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.reduce import Reduce


@register_custom_op
class Reduce_hls(Reduce, HLSBackend):
    """Class that corresponds to custom_hls reduce function."""

    def get_nodeattr_types(
        self,
    ) -> dict[
        str,
        tuple[str, bool, int | float | str | bool | np.ndarray | list]
        | tuple[str, bool, int | float | str | bool | np.ndarray | list, set | None],
    ]:
        """Get dictionary of custom node attributes with their types and default values."""
        my_attrs = {}
        my_attrs.update(Reduce.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self) -> None:
        """List include directives for generated HLS code."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "reduce.hpp"']

    def defines(self, var) -> None:  # noqa: ANN001, ARG002
        """Constant and type definitions for generated HLS code."""
        self.code_gen_dict["$DEFINES$"] = []

    def docompute(self) -> None:
        """Generate the computational part of the HLS C++ code."""
        op = self.op.capitalize()
        start_index = self.start_index
        folded_shape = self.get_folded_input_shape()[:-2]  # last dim is lanes and is never reduced
        reduction_mode = "ReductionMode::Depthwise" if self.depthwise else "ReductionMode::Spatial"
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"OuterReduce<{op}, {self.get_folded_input_shape()[-2]}, {start_index}, "
            f"{reduction_mode}, {', '.join(str(dim) for dim in folded_shape)}>(in0_V, out0_V);"
        ]

    def pragmas(self) -> None:
        """Generate HLS pragmas to apply to the HLS C++ code."""
        super().pragmas()
        self.code_gen_dict["$PRAGMAS$"].extend(
            [
                "#pragma HLS dataflow disable_start_propagation",
                "#pragma HLS aggregate variable=in0_V compact=bit",
                "#pragma HLS aggregate variable=out0_V compact=bit",
            ]
        )

    def blackboxfunction(self) -> None:
        """Blackbox function interface from which the IP will be generated."""
        idt_str = self.get_input_datatype().get_hls_datatype_str()
        odt_str = self.get_output_datatype().get_hls_datatype_str()
        i_hls_dt = f"hls::vector<{idt_str}, {self.get_pe_in()}>"
        o_hls_dt = f"hls::vector<{odt_str}, {self.get_pe_out()}>"

        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"void {self.onnx_node.name}"
            f"(hls::stream<{i_hls_dt}> &in0_V, hls::stream<{o_hls_dt}> &out0_V)"
        ]

    def execute_node(self, context: dict[str, np.ndarray], graph: GraphProto) -> None:
        """Execute the node in HLS C++ simulation."""
        HLSBackend.execute_node(self, context, graph)

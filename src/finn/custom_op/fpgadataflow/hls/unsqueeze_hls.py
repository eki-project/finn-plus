"""HLS backend implementation of the Unsqueeze operator.

Note: the HLS implementation is identical to the Squeeze operator; these
could potentially be unified.
"""

import numpy as np
from onnx import GraphProto

from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.unsqueeze import NodeAttrTypes, Unsqueeze


@register_custom_op
class Unsqueeze_hls(Unsqueeze, HLSBackend):
    """HLS backend implementation of the Unsqueeze operator.

    Inserts single-dimension entries into the shape of a tensor using HLS synthesis.
    """

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return the dictionary of node attributes for the HLS Unsqueeze operator."""
        attrs: NodeAttrTypes = {}
        attrs.update(Unsqueeze.get_nodeattr_types(self))
        attrs.update(HLSBackend.get_nodeattr_types(self))
        return attrs

    def global_includes(self) -> None:
        """Generate the list of C++ includes for the top of the generated code."""
        # Currently nothing to include
        self.code_gen_dict["$GLOBALS$"] = []

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Generate C++ code for type alias, global constant, and macro definitions."""
        # Currently nothing to define
        self.code_gen_dict["$DEFINES$"] = []

    def execute_node(self, context: dict[str, np.ndarray], graph: GraphProto) -> None:
        """Execute node via the generic HLSBackend implementation (cppsim/rtlsim)."""
        HLSBackend.execute_node(self, context, graph)

    def docompute(self) -> None:
        """Generate the C++ code for the computation part of the operator."""
        # Number of iterations required to process the whole folded input stream
        # (all but the PE, last, dimension)
        num_iter = int(np.prod(self.get_folded_output_shape()[:-1]))
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"for(std::size_t i = 0; i < {num_iter}; ++i) {{",
            "#pragma HLS pipeline II=1 style=flp",
            # Read from the input and immediately write the same element to the
            # output. Unsqueezed dimensions (size 1) do not contribute to the
            # number or order of elements and can simply be ignored.
            "out0_V.write(in0_V.read());",
            "}",
        ]

    def blackboxfunction(self) -> None:
        """Generate the C++ function signature for IP block generation."""
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            # Note: assumes stream type aliases to be set in defines
            f"void {self.onnx_node.name} (",
            f"  hls::stream<ap_uint<{self.get_instream_width()}>> &in0_V,",
            f"  hls::stream<ap_uint<{self.get_outstream_width()}>> &out0_V",
            ")",
        ]

# Copyright (c) 2020-2022, Xilinx, Inc.
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Module for iodma hls."""

import math
import numpy as np
import numpy.typing as npt
from qonnx.core.datatype import BaseDataType, DataType
from typing import TYPE_CHECKING, Any, cast

from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log

if TYPE_CHECKING:
    from onnx import GraphProto, NodeProto
    from qonnx.core.modelwrapper import ModelWrapper

# Type of the dictionary returned by get_nodeattr_types: maps attribute names to
# their (dtype, required, default[, allowed_values]) specification tuples
NodeAttrTypes = dict[
    str,
    tuple[str, bool, int | float | str | bool | npt.NDArray | list]
    | tuple[str, bool, int | float | str | bool | npt.NDArray | list, set | None],
]

# the IODMA inerfaces a memory-mapped AXI interface and an AXI stream
# direction "in": pulls data from AXI-MM to AXI stream
# direction "out": pushes data from AXI stream to AXI-MM

# DMA Addressing
# - burst mode can be "wrap" or "increment"
# - "increment" bursts will increment the address when moving to the next image
# - "wrap" bursts will reinitialize the address to the start address,
#   and are useful for e.g. streaming weights, where the same buffer is
#   repeatedly read into the FPGA
# - no additional alignment restrictions beyond anything specified in the AXI spec

# Interfaces
# - AXI-MM name specified by intfName unless this is set to "" (empty, the default)
#   in which case output AXI-MM are named "out0_V" and input AXI-MM are named "in0_V"
# - AXI-MM interface width (in bits) is specified by intfWidth
# - AXI-Stream interface width (in bits) is specified by streamWidth
# - If inftWidth and streamWidth are not equal, the DMA core performs
#   width conversion by going up to the least common multiple of bitwidths
#   e.g. intfWidth=32b -> 96b -> sreamWidth=24b
# - transfers occur in multiples of the AXI-MM interface width, therefore
#   the total number of bits in the tensor must be a multiple of intfWidth
# - transfers occur in multiples of the AXI-Stream interface width, therefore
#   the total number of bits in the tensor must be a multiple of streamWidth
# - both interface widths must be a multiple of 8b (AXI protocol requirement)
# - in most systems, intfWidth is also restricted to a power of 2 (e.g. Vitis)
#   but this is not universal so we don't check here explicitly

# Input/output tensor sizes shapes
# - The data being moved is a tensor of shape numInputVectors+[NumChannels]
# - The data type of the tensor elements is specified by dataType
# - on the stream side
#       -the normal shape is the same as the ONNX tensor attached to it
#       -the folded shape is computed from the stream width and normal shape
# - on the AXI-MM side
#       -the normal shape is the same as the one on the stream side
#       -the folded shape is not defined


@register_custom_op
class IODMA_hls(HLSBackend, HWCustomOp):
    """Class that corresponds to finn-hlslib DMA function(s)."""

    def __init__(self, onnx_node: "NodeProto", **kwargs: Any) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return the dictionary of node attributes for the IODMA operator."""
        my_attrs: NodeAttrTypes = {
            "NumChannels": ("i", True, 0),
            # FINN input datatype
            "dataType": ("s", True, ""),
            # Width of input or output stream
            "streamWidth": ("i", False, 32),
            # DMA-specific parameters
            # width of axi-mm interface
            "intfWidth": ("i", False, 32),
            # burst mode for axi-mm interface (wrap used for DRAM weights)
            "burstMode": ("s", False, "increment", {"wrap", "increment"}),
            # IODMA direction: in = read from DRAM, out = write to DRAM
            "direction": ("s", False, "in", {"in", "out"}),
            # shape describing input vecs per execution
            "numInputVectors": ("ints", False, [1]),
            # name of axi-mm interface
            "intfName": ("s", False, ""),
        }
        my_attrs.update(HWCustomOp.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    @property
    def num_channels(self) -> int:
        """Return the number of channels moved by the DMA."""
        return cast("int", self.get_nodeattr("NumChannels"))

    @property
    def stream_width(self) -> int:
        """Return the AXI-Stream interface width in bits."""
        return cast("int", self.get_nodeattr("streamWidth"))

    @property
    def intf_width(self) -> int:
        """Return the AXI-MM interface width in bits."""
        return cast("int", self.get_nodeattr("intfWidth"))

    @property
    def direction(self) -> str:
        """Return the DMA direction ("in" or "out")."""
        return cast("str", self.get_nodeattr("direction"))

    @property
    def burst_mode(self) -> str:
        """Return the AXI-MM burst mode ("increment" or "wrap")."""
        return cast("str", self.get_nodeattr("burstMode"))

    @property
    def intf_name(self) -> str:
        """Return the configured AXI-MM interface name (empty for the default)."""
        return cast("str", self.get_nodeattr("intfName"))

    @property
    def num_input_vectors(self) -> list[int]:
        """Return the shape describing the number of input vectors per execution."""
        return list(cast("list[int]", self.get_nodeattr("numInputVectors")))

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the normal (unfolded) input shape."""
        return (*self.num_input_vectors, self.num_channels)

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the normal (unfolded) output shape."""
        return self.get_normal_input_shape()

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the folded input shape (stream side only)."""
        if self.direction == "in":
            raise FINNInternalError("Folded input shape not defined for input IODMA")
        shape = list(self.get_normal_input_shape())
        itype_bits = self.get_input_datatype().bitwidth()
        intfw = self.stream_width
        if intfw % itype_bits != 0:
            raise FINNUserError("Input stream width must be a multiple of datatype bits")
        elems_per_word = intfw // itype_bits
        if shape[-1] % elems_per_word != 0:
            raise FINNUserError("Fold depth must be integer")
        fold_depth = shape[-1] // elems_per_word
        shape[-1] = fold_depth
        shape.append(elems_per_word)
        return tuple(shape)

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return the folded output shape (stream side only)."""
        if self.direction == "out":
            raise FINNInternalError("Folded output shape not defined for output IODMA")
        shape = list(self.get_normal_output_shape())
        itype_bits = self.get_output_datatype().bitwidth()
        intfw = self.stream_width
        if intfw % itype_bits != 0:
            raise FINNUserError("Input stream width must be a multiple of datatype bits")
        elems_per_word = intfw // itype_bits
        if shape[-1] % elems_per_word != 0:
            raise FINNUserError("Fold depth must be integer")
        fold_depth = shape[-1] // elems_per_word
        shape[-1] = fold_depth
        shape.append(elems_per_word)
        return tuple(shape)

    def infer_node_datatype(self, model: "ModelWrapper") -> None:
        """Infer the node output datatype from the input datatype."""
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = (
                f"inputDataType changing for {node.name}: "
                f"{self.get_input_datatype()!s} -> {idt!s} "
            )
            log.warning(warn_str)
        self.set_nodeattr("dataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return the FINN DataType of the input."""
        return DataType[cast("str", self.get_nodeattr("dataType"))]

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return the FINN DataType of the output (same as the input datatype)."""
        return self.get_input_datatype()

    def get_instream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return the input stream width in bits."""
        if self.direction == "in":
            return self.intf_width
        if self.direction == "out":
            return self.stream_width
        raise FINNUserError("Invalid IODMA direction, please set to in or out")

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return the output stream width in bits."""
        if self.direction == "out":
            return self.intf_width
        if self.direction == "in":
            return self.stream_width
        raise FINNUserError("Invalid IODMA direction, please set to in or out")

    def get_number_output_values(self) -> int:
        """Return the number of expected output values from the operator."""
        oshape = self.get_normal_output_shape()
        itype_bits = self.get_input_datatype().bitwidth()
        stream_width = self.stream_width
        nelems = int(np.prod(oshape))
        nbits = nelems * itype_bits
        if nbits % stream_width != 0:
            raise FINNUserError("DMA: total transfer size must be word multiple")
        return nbits // stream_width

    def global_includes(self) -> None:
        """Return global includes."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "dma.h"']
        self.code_gen_dict["$GLOBALS$"].append('#include "streamtools.h"')

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Return defines."""
        itype_bits = self.get_input_datatype().bitwidth()
        total_bits = itype_bits * int(np.prod(self.get_normal_input_shape()))
        if total_bits % 8 != 0:
            raise FINNUserError("DMA input not a multiple of 1 Byte")
        total_bytes = total_bits // 8
        self.code_gen_dict["$DEFINES$"] = [
            f"#define NumBytes1 {total_bytes}\n#define DataWidth1 {self.intf_width}\n"
        ]

    def get_ap_int_max_w(self) -> int:
        """Return the maximum width of any ap_int used in this module."""
        instream = self.get_instream_width()
        outstream = self.get_outstream_width()
        return (instream * outstream) // math.gcd(instream, outstream)

    def docompute(self) -> None:
        """Return docompute."""
        direction = self.direction
        mode = self.burst_mode
        dwc_func = "StreamingDataWidthConverter_Batch"
        if direction == "in":
            func = "Mem2Stream_Batch_external_wmem" if mode == "wrap" else "Mem2Stream_Batch"
        elif direction == "out":
            func = "Stream2Mem_Batch"
        else:
            raise FINNUserError("Invalid IODMA direction, please set to in or out")

        # templates for instantiation
        def dma_inst(src: str, dst: str) -> str:
            """Render a DMA function instantiation line."""
            return f"{func}<DataWidth1, NumBytes1>({src}, {dst}, numReps);"

        def dwc_inst(w_in: int, w_out: int, n_reps: int, src: str, dst: str) -> str:
            """Render a data-width-converter instantiation line."""
            return f"{dwc_func}<{w_in}, {w_out}, {n_reps}>({src}, {dst}, numReps);"

        # do stream infrastructure and instantiations
        intfw = self.intf_width
        strmw = self.stream_width
        width_lcm = (strmw * intfw) // math.gcd(strmw, intfw)
        # we always need two streams: one of width_lcm, and one of intfw width
        # because we use WidthAdjustedInputStream,
        dtype_bits = self.get_input_datatype().bitwidth()
        total_bits = dtype_bits * int(np.prod(self.get_normal_input_shape()))

        if direction == "in":
            # AXI MM -> IODMA -> (DWCs) -> out
            # DWCs depend on AXI MM and out interface width
            if strmw == intfw:
                # case 0: AXI MM width = out width, no DWCs needed
                self.code_gen_dict["$DOCOMPUTE$"] = [dma_inst("in0_V", "out0_V")]
            elif (strmw % intfw == 0) or (intfw % strmw == 0):
                # case 1: AXI MM width divisible by out width or vice versa
                # single DWC + single extra stream needed
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    f"hls::stream<ap_uint<{intfw}> > dma2dwc;",
                    dma_inst("in0_V", "dma2dwc"),
                    dwc_inst(intfw, strmw, total_bits // intfw, "dma2dwc", "out0_V"),
                ]
            else:
                # case 2: AXI MM width not divisible by out width or vice versa
                # need 2 DWCs (going through the least common multiple width)
                # and 2 streams
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    f"hls::stream<ap_uint<{intfw}> > dma2lcm;",
                    f"hls::stream<ap_uint<{width_lcm}> > lcm2out;",
                    dma_inst("in0_V", "dma2lcm"),
                    dwc_inst(intfw, width_lcm, total_bits // intfw, "dma2lcm", "lcm2out"),
                    dwc_inst(width_lcm, strmw, total_bits // width_lcm, "lcm2out", "out0_V"),
                ]
        elif direction == "out":
            # in0 -> (DWCs) -> IODMA -> AXI MM
            # DWCs depend on AXI MM and out interface width
            if strmw == intfw:
                # case 0: in width = AXI MM width, no DWCs needed
                self.code_gen_dict["$DOCOMPUTE$"] = [dma_inst("in0_V", "out0_V")]
            elif (strmw % intfw == 0) or (intfw % strmw == 0):
                # case 1: AXI MM width divisible by in width or vice versa
                # single DWC + single extra stream needed
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    f"hls::stream<ap_uint<{intfw}> > dwc2dma;",
                    dwc_inst(strmw, intfw, total_bits // strmw, "in0_V", "dwc2dma"),
                    dma_inst("dwc2dma", "out0_V"),
                ]
            else:
                # case 2: AXI MM width not divisible by out width or vice versa
                # need 2 DWCs (going through the least common multiple width)
                # and 2 streams
                self.code_gen_dict["$DOCOMPUTE$"] = [
                    f"hls::stream<ap_uint<{width_lcm}> > in2lcm;",
                    f"hls::stream<ap_uint<{intfw}> > lcm2dma;",
                    dwc_inst(strmw, width_lcm, total_bits // strmw, "in0_V", "in2lcm"),
                    dwc_inst(width_lcm, intfw, total_bits // width_lcm, "in2lcm", "lcm2dma"),
                    dma_inst("lcm2dma", "out0_V"),
                ]
        else:
            raise FINNUserError(f"Unknown IODMA direction: {direction}")

    def blackboxfunction(self) -> None:
        """Return blackboxfunction."""
        packed_ibits = self.get_instream_width()
        packed_hls_type_in = f"ap_uint<{packed_ibits}>"
        packed_obits = self.get_outstream_width()
        packed_hls_type_out = f"ap_uint<{packed_obits}>"
        direction = self.direction
        if direction == "in":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                f"void {self.onnx_node.name}({packed_hls_type_in} *in0_V, "
                f"hls::stream<{packed_hls_type_out} > &out0_V, unsigned int numReps)"
            ]
        elif direction == "out":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                f"void {self.onnx_node.name}(hls::stream<{packed_hls_type_in} > &in0_V, "
                f"{packed_hls_type_out} *out0_V, unsigned int numReps)"
            ]
        else:
            raise FINNUserError("Invalid IODMA direction, please set to in or out")

    def pragmas(self) -> None:
        """Return pragmas."""
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE s_axilite port=numReps bundle=control"
        ]
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE s_axilite port=return bundle=control"
        )
        direction = self.direction
        intfname = self.intf_name
        if direction == "in":
            if intfname == "":
                self.code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS INTERFACE m_axi offset=slave port=in0_V"
                )
            else:
                self.code_gen_dict["$PRAGMAS$"].append(
                    f"#pragma HLS INTERFACE m_axi offset=slave port={intfname}"
                )
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE s_axilite port=in0_V bundle=control"
            )
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        elif direction == "out":
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=in0_V")
            if intfname == "":
                self.code_gen_dict["$PRAGMAS$"].append(
                    "#pragma HLS INTERFACE m_axi offset=slave port=out0_V"
                )
            else:
                self.code_gen_dict["$PRAGMAS$"].append(
                    f"#pragma HLS INTERFACE m_axi offset=slave port={intfname}"
                )
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE s_axilite port=out0_V bundle=control"
            )
        else:
            raise FINNUserError("Invalid IODMA direction, please set to in or out")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS DATAFLOW")

    def execute_node(self, context: dict[str, np.ndarray], graph: "GraphProto") -> None:
        """Execute node (no-op for the IODMA)."""

    def get_verilog_top_module_intf_names(self) -> dict:
        """Return verilog top module intf names."""
        intf_names = super().get_verilog_top_module_intf_names()
        if self.direction == "out":
            intf_names["m_axis"] = []
        else:
            intf_names["s_axis"] = []
        intf_names["axilite"] = ["s_axi_control"]
        intf_names["aximm"] = [("m_axi_gmem", self.intf_width)]
        return intf_names

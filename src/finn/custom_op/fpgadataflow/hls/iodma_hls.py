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

from __future__ import annotations

import math
import numpy as np
import re
from pathlib import Path
from qonnx.core.datatype import BaseDataType, DataType
from typing import TYPE_CHECKING, Any, cast

from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp, NodeAttrTypes
from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log

if TYPE_CHECKING:
    from onnx import GraphProto, NodeProto
    from qonnx.core.modelwrapper import ModelWrapper


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

# Replication and control (burstMode "increment" only, see custom_hls/iodma.hpp)
# - numReplicas N: the DMA serves N copies of the compute pipeline. The input DMA sends frame g of
#   the frame sequence to stream port g % N, the output DMA collects frame g from port g % N, so
#   the host sees one network. Ports are in<p>_V / out<p>_V, p = 0..N-1.
# - ctrlMode "register": ap_ctrl_hs kernel with the buffer address at 0x10 and numReps at 0x1C,
#   the interface the PYNQ driver drives (Zynq and Alveo).
# - ctrlMode "ring": free-running (ap_ctrl_none) kernel fed through a descriptor ring of
#   ringSlots entries in AXI-lite space (address lo/hi, frames), a doorbell counter "prod" written
#   by the host and a completion counter "cons" written by the kernel. Used by the C++ driver;
#   the register offsets are read from the HLS-generated driver header
#   (get_ring_register_offsets).

#: AXI-lite registers and register files of a ring DMA, as named in its HLS top function
RING_REGISTERS = ("addr_lo", "addr_hi", "frames", "prod", "cons")


@register_custom_op
class IODMA_hls(HLSBackend, HWCustomOp):
    """Class that corresponds to finn-hlslib DMA function(s)."""

    def __init__(self, onnx_node: NodeProto, **kwargs: Any) -> None:
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
            # number of replicated compute pipelines served by this DMA
            "numReplicas": ("i", False, 1),
            # control interface: register (ap_ctrl_hs, PYNQ) or ring (free-running, C++ driver)
            "ctrlMode": ("s", False, "register", {"register", "ring"}),
            # number of descriptor slots of a ring DMA
            "ringSlots": ("i", False, 16),
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
    def num_replicas(self) -> int:
        """Return the numReplicas attribute (see _replicas for the checked value)."""
        return cast("int", self.get_nodeattr("numReplicas"))

    @property
    def ctrl_mode(self) -> str:
        """Return the control interface ("register" or "ring")."""
        return cast("str", self.get_nodeattr("ctrlMode"))

    @property
    def ring_slots(self) -> int:
        """Return the number of descriptor slots of a ring DMA."""
        return cast("int", self.get_nodeattr("ringSlots"))

    @property
    def num_input_vectors(self) -> list[int]:
        """Return the shape describing the number of input vectors per execution."""
        return list(cast("list[int]", self.get_nodeattr("numInputVectors")))

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return normal input shape."""
        return (*self.num_input_vectors, self.num_channels)

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return normal output shape."""
        return self.get_normal_input_shape()

    def _folded_stream_shape(self, shape: list[int], dtype_bits: int) -> tuple[int, ...]:
        """Fold the innermost dimension of shape into words of the stream width."""
        stream_width = self.stream_width
        if stream_width % dtype_bits != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: stream width must be a multiple of the datatype bits"
            )
        elems_per_word = stream_width // dtype_bits
        if shape[-1] % elems_per_word != 0:
            raise FINNInternalError(f"{self.onnx_node.name}: fold depth must be integer")
        return (*shape[:-1], shape[-1] // elems_per_word, elems_per_word)

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded input shape."""
        if self.direction == "in":
            raise FINNInternalError("Folded input shape not defined for input IODMA")
        return self._folded_stream_shape(
            list(self.get_normal_input_shape()), self.get_input_datatype().bitwidth()
        )

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded output shape."""
        if self.direction == "out":
            raise FINNInternalError("Folded output shape not defined for output IODMA")
        return self._folded_stream_shape(
            list(self.get_normal_output_shape()), self.get_output_datatype().bitwidth()
        )

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer node datatype."""
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            log.warning(
                f"inputDataType changing for {node.name}: {self.get_input_datatype()} -> {idt}"
            )
        self.set_nodeattr("dataType", idt.name)
        model.set_tensor_datatype(node.output[0], idt)

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of input."""
        return DataType[cast("str", self.get_nodeattr("dataType"))]

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return FINN DataType of output (the same as the input datatype)."""
        return self.get_input_datatype()

    def get_instream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return instream width."""
        if self.direction == "in":
            return self.intf_width
        return self.stream_width

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return outstream width."""
        if self.direction == "out":
            return self.intf_width
        return self.stream_width

    def get_number_output_values(self) -> int:
        """Return number output values."""
        stream_width = self.stream_width
        nbits = self._frame_bits()
        if nbits % stream_width != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: total transfer size must be a stream word multiple"
            )
        return nbits // stream_width

    def _frame_bits(self) -> int:
        """Bits of one frame (one execution of the network)."""
        return int(self.get_input_datatype().bitwidth() * np.prod(self.get_normal_input_shape()))

    def _replicas(self) -> int:
        """Return the number of replicated pipelines (increment-mode DMAs only)."""
        replicas = self.num_replicas
        if replicas < 1:
            raise FINNUserError(
                f"{self.onnx_node.name}: numReplicas must be at least 1, got {replicas}"
            )
        if self.burst_mode == "wrap" and (replicas != 1 or self._is_ring()):
            raise FINNUserError(
                f"{self.onnx_node.name}: weight-streaming (wrap) IODMAs support neither "
                f"replication nor ring control"
            )
        return replicas

    def _is_ring(self) -> bool:
        """Whether this is a free-running ring DMA (C++ driver) rather than a register DMA."""
        return self.ctrl_mode == "ring"

    def _stream_port(self, p: int) -> str:
        """Name of stream port p."""
        return f"out{p}_V" if self.direction == "in" else f"in{p}_V"

    def _mm_port(self) -> str:
        """Name of the memory-mapped port."""
        intfname = self.intf_name
        if intfname != "":
            return intfname
        return "in0_V" if self.direction == "in" else "out0_V"

    def global_includes(self) -> None:
        """Return global includes."""
        self.code_gen_dict["$GLOBALS$"] = [
            '#include "dma.h"',
            '#include "streamtools.h"',
            '#include "iodma.hpp"',
        ]

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Return defines."""
        total_bits = self._frame_bits()
        if total_bits % 8 != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: DMA transfer is not a multiple of 1 byte"
            )
        intfw = self.intf_width
        self.code_gen_dict["$DEFINES$"] = [
            f"#define NumBytes1 {total_bits // 8}\n#define DataWidth1 {intfw}\n"
        ]
        if self._is_ring():
            # free-running in hardware; a C simulation testbench defines these to run a fixed
            # number of descriptors (IODMA_TEST_DESC) and frames (IODMA_TEST_FRAMES) instead
            self.code_gen_dict["$DEFINES$"].append(
                "#ifndef IODMA_FOREVER\n#define IODMA_FOREVER true\n#define IODMA_TEST_FRAMES 0\n"
                "#define IODMA_TEST_DESC 0\n#endif\n"
            )

    def get_ap_int_max_w(self) -> int:
        """Return the maximum width of any ap_int used in this module."""
        return math.lcm(self.get_instream_width(), self.get_outstream_width())

    def _convert(
        self, src: str, src_w: int, dst: str, dst_w: int, port: int, frames: str
    ) -> tuple[list[str], list[str]]:
        """Return stream declarations and calls converting the frames of one port to dst_w bits."""
        n = self._replicas()
        forever = "IODMA_FOREVER" if self._is_ring() else "false"
        frame_bits = self._frame_bits()

        def call(a: str, a_w: int, b: str, b_w: int) -> str:
            tmpl = f"{a_w}, {b_w}, {port}, {n}, {forever}"
            return f"iodma::convert<{tmpl}>({a}, {b}, {frame_bits // a_w}, {frames});"

        if src_w % dst_w == 0 or dst_w % src_w == 0:
            return [], [call(src, src_w, dst, dst_w)]
        lcm = math.lcm(src_w, dst_w)
        mid = f"lcm{port}"
        return [f"hls::stream<ap_uint<{lcm}>> {mid};"], [
            call(src, src_w, mid, lcm),
            call(mid, lcm, dst, dst_w),
        ]

    def docompute(self) -> None:
        """Return docompute."""
        if self.burst_mode == "wrap":
            self._docompute_wrap()
            return
        n = self._replicas()
        ring = self._is_ring()
        intfw = self.intf_width
        strmw = self.stream_width
        frame_bits = self._frame_bits()
        if frame_bits % intfw != 0 or frame_bits % strmw != 0:
            raise FINNInternalError(
                f"{self.onnx_node.name}: a frame ({frame_bits} bits) must be a multiple of the "
                f"memory word ({intfw} bits) and of the stream word ({strmw} bits)"
            )
        frame_words = frame_bits // intfw
        frames = "IODMA_TEST_FRAMES" if ring else "numReps"
        forever = "IODMA_FOREVER" if ring else "false"
        mm = self._mm_port()
        slots = self.ring_slots
        ring_args = f"addr_lo, addr_hi, frames, prod, cons, {frame_words}, IODMA_TEST_DESC"
        decls: list[str] = []
        calls: list[str] = []
        if self.direction == "in":
            # memory -> [split over N ports] -> width conversion per port -> out<p>_V
            direct = n == 1 and intfw == strmw
            rd = self._stream_port(0) if direct else "mem2s"
            if not direct:
                decls.append(f"hls::stream<ap_uint<{intfw}>> mem2s;")
            if ring:
                calls.append(f"iodma::ring_read<{intfw}, {slots}>({mm}, {rd}, {ring_args});")
            else:
                calls.append(f"iodma::read_words<{intfw}>({mm}, {rd}, {frame_words}, numReps);")
            if n > 1:
                decls += [
                    f"hls::stream<ap_uint<{intfw}>> port[{n}];",
                    f"#pragma HLS STREAM variable=port depth={frame_words}",
                ]
                calls.append(
                    f"iodma::split<{intfw}, {n}, {forever}>(mem2s, port, {frame_words}, {frames});"
                )
            if not direct:
                for p in range(n):
                    src = f"port[{p}]" if n > 1 else "mem2s"
                    d, c = self._convert(src, intfw, self._stream_port(p), strmw, p, frames)
                    decls += d
                    calls += c
        else:
            # in<p>_V -> width conversion per port -> [gather from N ports] -> memory
            direct = n == 1 and intfw == strmw
            wr = self._stream_port(0) if direct else "s2mem"
            if not direct:
                decls.append(f"hls::stream<ap_uint<{intfw}>> s2mem;")
            if n > 1:
                decls += [
                    f"hls::stream<ap_uint<{intfw}>> port[{n}];",
                    f"#pragma HLS STREAM variable=port depth={frame_words}",
                ]
            if not direct:
                for p in range(n):
                    dst = f"port[{p}]" if n > 1 else "s2mem"
                    d, c = self._convert(self._stream_port(p), strmw, dst, intfw, p, frames)
                    decls += d
                    calls += c
            if n > 1:
                calls.append(
                    f"iodma::gather<{intfw}, {n}, {forever}>(port, s2mem, {frame_words}, {frames});"
                )
            if ring:
                calls.append(f"iodma::ring_write<{intfw}, {slots}>({wr}, {mm}, {ring_args});")
            else:
                calls.append(f"iodma::write_words<{intfw}>({wr}, {mm}, {frame_words}, numReps);")
        self.code_gen_dict["$DOCOMPUTE$"] = decls + calls

    def _docompute_wrap(self) -> None:
        """Weight-streaming DMA (finn-hlslib, register control, one pipeline)."""
        dwc_inst_template = "StreamingDataWidthConverter_Batch<%d, %d, %d>(%s, %s, numReps);"
        dma_inst = "Mem2Stream_Batch_external_wmem<DataWidth1, NumBytes1>(%s, %s, numReps);"
        intfw = self.intf_width
        strmw = self.stream_width
        total_bits = self._frame_bits()
        mm = self._mm_port()
        if strmw == intfw:
            self.code_gen_dict["$DOCOMPUTE$"] = [dma_inst % (mm, "out0_V")]
        elif strmw % intfw == 0 or intfw % strmw == 0:
            self.code_gen_dict["$DOCOMPUTE$"] = [
                f"hls::stream<ap_uint<{intfw}> > dma2dwc;",
                dma_inst % (mm, "dma2dwc"),
                dwc_inst_template % (intfw, strmw, total_bits // intfw, "dma2dwc", "out0_V"),
            ]
        else:
            width_lcm = math.lcm(strmw, intfw)
            self.code_gen_dict["$DOCOMPUTE$"] = [
                f"hls::stream<ap_uint<{intfw}> > dma2lcm;",
                f"hls::stream<ap_uint<{width_lcm}> > lcm2out;",
                dma_inst % (mm, "dma2lcm"),
                dwc_inst_template % (intfw, width_lcm, total_bits // intfw, "dma2lcm", "lcm2out"),
                dwc_inst_template
                % (width_lcm, strmw, total_bits // width_lcm, "lcm2out", "out0_V"),
            ]

    def blackboxfunction(self) -> None:
        """Return blackboxfunction."""
        n = self._replicas()
        mem_type = f"ap_uint<{self.intf_width}>"
        stream_type = f"hls::stream<ap_uint<{self.stream_width}>>"
        mm = self._mm_port()
        mem_arg = f"hls::burst_maxi<{mem_type}> {mm}" if self._is_ring() else f"{mem_type} *{mm}"
        streams = [f"{stream_type} &{self._stream_port(p)}" for p in range(n)]
        args = [mem_arg, *streams] if self.direction == "in" else [*streams, mem_arg]
        if self._is_ring():
            slots = self.ring_slots
            args += [
                f"const unsigned addr_lo[{slots}]",
                f"const unsigned addr_hi[{slots}]",
                f"const unsigned frames[{slots}]",
                "volatile unsigned &prod",
                "volatile unsigned &cons",
            ]
        else:
            args.append("unsigned int numReps")
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"void {self.onnx_node.name}({', '.join(args)})"
        ]

    def pragmas(self) -> None:
        """Return pragmas."""
        mm = self._mm_port()
        burst = "max_read_burst_length" if self.direction == "in" else "max_write_burst_length"
        pragmas = []
        if self._is_ring():
            pragmas.append("#pragma HLS INTERFACE mode=ap_ctrl_none port=return")
            pragmas += [
                f"#pragma HLS INTERFACE mode=s_axilite port={r} bundle=control"
                for r in RING_REGISTERS
            ]
        else:
            pragmas += [
                "#pragma HLS INTERFACE s_axilite port=numReps bundle=control",
                "#pragma HLS INTERFACE s_axilite port=return bundle=control",
            ]
        pragmas += [
            f"#pragma HLS INTERFACE m_axi offset=slave port={mm} {burst}=64",
            f"#pragma HLS INTERFACE s_axilite port={mm} bundle=control",
        ]
        pragmas += [
            f"#pragma HLS INTERFACE axis port={self._stream_port(p)}"
            for p in range(self._replicas())
        ]
        pragmas.append("#pragma HLS DATAFLOW")
        self.code_gen_dict["$PRAGMAS$"] = pragmas

    def execute_node(self, context: dict[str, np.ndarray], graph: GraphProto) -> None:
        """Execute node (no-op for the IODMA)."""

    def get_verilog_top_module_intf_names(self) -> dict[str, Any]:
        """Return verilog top module intf names."""
        intf_names = super().get_verilog_top_module_intf_names()
        width = self.stream_width
        ports: list[tuple[str, int]] = [
            (self._stream_port(p), width) for p in range(self._replicas())
        ]
        if self.direction == "out":
            intf_names["s_axis"] = ports
            intf_names["m_axis"] = []
        else:
            intf_names["s_axis"] = []
            intf_names["m_axis"] = ports
        intf_names["axilite"] = ["s_axi_control"]
        intf_names["aximm"] = [("m_axi_gmem", self.intf_width)]
        return intf_names

    def get_ring_register_offsets(self) -> dict[str, int]:
        """AXI-lite byte offsets of the ring registers (RING_REGISTERS), read from the driver
        header that HLS generates with the IP. Register files (addr_lo, addr_hi, frames) map to
        the offset of their first slot; slot s is at offset + 4 * s.
        """
        if not self._is_ring():
            raise FINNInternalError(f"{self.onnx_node.name} is not a ring DMA")
        # the HLS project keeps the node name of code generation; BuildAllXOs renames nodes later
        code_gen_dir = Path(cast("str", self.get_nodeattr("code_gen_dir_ipgen")))
        name = self.onnx_node.name
        headers = list(code_gen_dir.glob("project_*/sol1/impl/misc/drivers/*/src/x*_hw.h"))
        if len(headers) != 1:
            raise FINNInternalError(
                f"{name}: expected one HLS driver header, found {headers}; run HLSSynthIP first"
            )
        text = headers[0].read_text()
        offsets = {}
        for reg in RING_REGISTERS:
            # scalars: ..._ADDR_<REG>_DATA, register files: ..._ADDR_<REG>_BASE
            m = re.search(
                rf"#define\s+X\w*_CONTROL_ADDR_{reg.upper()}_(?:DATA|BASE)\s+(0x[0-9a-fA-F]+)", text
            )
            if m is None:
                raise FINNInternalError(f"{name}: register {reg} not found in {headers[0]}")
            offsets[reg] = int(m.group(1), 16)
        return offsets

# Copyright (C) 2023, Advanced Micro Devices, Inc.
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

"""HLS backend implementation of the embedding-lookup operator."""

import numpy as np
from math import ceil, log2
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING

from finn.custom_op.fpgadataflow.base.lookup import Lookup, NodeAttrTypes
from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.data_packing import numpy_to_hls_code, pack_innermost_dim_as_hex_string
from finn.util.exception import FINNInternalError, FINNUserError

if TYPE_CHECKING:
    from onnx import GraphProto, NodeProto


@register_custom_op
class Lookup_hls(Lookup, HLSBackend):
    """Streaming elementwise HLS lookup, mapping indices to values."""

    def __init__(self, onnx_node: "NodeProto", **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {}
        my_attrs.update(Lookup.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self) -> None:
        """Return global includes."""
        global_incls = ['#include "lookup.hpp"']
        if self.mem_mode == "internal_embedded":
            global_incls.append('#include "embeddings.hpp"')
        self.code_gen_dict["$GLOBALS$"] = global_incls

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Return defines."""
        n_inputs = int(np.prod(self.get_folded_input_shape()[:-1]))
        elem_hls_type = self.get_input_datatype().get_hls_datatype_str()
        emb_hls_type = self.embedding_type.get_hls_datatype_str()
        my_defines = [f"#define NumInputs {n_inputs}"]
        if self.mem_mode == "external":
            ext_mem_emb_size = self.get_folded_output_shape()[-2]
            ext_mem_emb_align = ceil(log2(ext_mem_emb_size))
            my_defines += [
                f"#define MemBits {self.ext_mem_width}",
                f"#define EmbeddingSize {ext_mem_emb_size}",
                f"#define EmbeddingAlign {ext_mem_emb_align}",
                f"#define T_SRC {elem_hls_type}",
                "#define T_DST ap_uint<MemBits>",
            ]
        elif self.mem_mode == "internal_embedded":
            my_defines += [
                f"#define NumEmbeddings {self.num_embeddings}",
                f"#define EmbeddingDim {self.embedding_dim}",
                f"#define InputType {elem_hls_type}",
                f"#define EmbeddingType {emb_hls_type}",
            ]
        self.code_gen_dict["$DEFINES$"] = my_defines

    def dataoutstrm(self) -> None:
        """Return dataoutstrm."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
        elem_bits = dtype.bitwidth()
        packed_hls_type = f"ap_uint<{self.get_outstream_width()}>"
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_out = f"{code_gen_dir}/output_0.npy"
        oshape_cpp_str = str(self.get_folded_output_shape()).replace("(", "{").replace(")", "}")

        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            f"apintstream2npy<{packed_hls_type}, {elem_hls_type}, {elem_bits}, float>"
            f'(out0_V, {oshape_cpp_str}, "{npy_out}", false);'
        ]

    def docompute(self) -> None:
        """Return docompute."""
        if self.mem_mode == "internal_embedded":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """StreamingLookup<NumEmbeddings,  EmbeddingDim, NumInputs,
                InputType, EmbeddingType >(in0_V, out0_V, embeddings);"""
            ]
        elif self.mem_mode == "external":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """StreamingLookup_ext<EmbeddingSize>(in0_V, out0_V, mem, size, oob_count,
                oob_irq);"""
            ]

    def blackboxfunction(self) -> None:
        """Return blackboxfunction."""
        if self.mem_mode == "internal_embedded":
            packed_input_hls_type = f"ap_uint<{self.get_instream_width()}>"
            packed_output_hls_type = f"ap_uint<{self.get_outstream_width()}>"
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                f"void {self.onnx_node.name}(hls::stream<{packed_input_hls_type} > &in0_V, "
                f"hls::stream<{packed_output_hls_type} > &out0_V)"
            ]
        elif self.mem_mode == "external":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                f"void {self.onnx_node.name}(hls::stream<T_SRC> &in0_V, "
                f"hls::stream<T_DST> &out0_V, "
                f"T_DST const *const  mem, unsigned const size, "
                f"unsigned &oob_count, bool &oob_irq)"
            ]

    def pragmas(self) -> None:
        """Return pragmas."""
        my_pragmas = [
            "#pragma HLS INTERFACE axis port=in0_V",
            "#pragma HLS INTERFACE axis port=out0_V",
            "#pragma HLS INTERFACE ap_ctrl_none port=return",
        ]
        if self.mem_mode == "internal_embedded":
            my_pragmas.append("#pragma HLS BIND_STORAGE variable=embeddings type=ROM_2P impl=BRAM")
        elif self.mem_mode == "external":
            my_pragmas += [
                "#pragma HLS INTERFACE m_axi offset=slave port=mem",
                "#pragma HLS INTERFACE s_axilite port=mem bundle=control",
                "#pragma HLS INTERFACE s_axilite port=size bundle=control",
                "#pragma HLS INTERFACE s_axilite port=oob_count bundle=control",
                "#pragma HLS INTERFACE ap_none port=oob_irq",
            ]
        else:
            raise FINNInternalError(f"{self.onnx_node.name}: unrecognized mem_mode {self.mem_mode}")
        self.code_gen_dict["$PRAGMAS$"] = my_pragmas

    def generate_params(self, model: ModelWrapper, path: str | Path) -> None:
        """Generate params."""
        code_gen_dir = Path(path)
        embeddings = model.get_initializer(self.onnx_node.input[1])
        if not isinstance(embeddings, np.ndarray):
            raise FINNInternalError(
                f"{self.onnx_node.name}: expected a constant embedding table on input 1"
            )
        edt = self.embedding_type
        if self.mem_mode == "internal_embedded":
            if not np.vectorize(edt.allowed)(embeddings).all():
                raise FINNUserError(
                    f"{self.onnx_node.name}: embeddings cannot be expressed with type {edt}"
                )
            # reverse innermost dim in embeddings to remain compatible with
            # how we normally encode the data in FINN
            embeddings_rev = np.flip(embeddings, -1)
            embeddings_hls_code = numpy_to_hls_code(embeddings_rev, edt, "embeddings", True, False)
            (code_gen_dir / "embeddings.hpp").write_text(embeddings_hls_code)
        elif self.mem_mode == "external":
            if edt.bitwidth() != 8:
                raise FINNUserError(
                    f"{self.onnx_node.name}: Lookup with mem_mode=external only works with "
                    f"8-bit embeddings but found {edt}"
                )
            emb_dim = self.embedding_dim
            # need to zero-pad embeddings in external mode for burst alignment
            # compute how much padding we need
            emb_elems_per_ext_mem_width = self.get_folded_output_shape()[-1]
            ext_mem_emb_size = self.get_folded_output_shape()[-2]
            ext_mem_emb_align = ceil(log2(ext_mem_emb_size))
            align_factor = int((self.ext_mem_width / 8) * 2**ext_mem_emb_align)
            pad_amount = align_factor - emb_dim
            embeddings_padded = np.pad(embeddings, [(0, 0), (0, pad_amount)])
            # reshape for packing the innermost dim
            embeddings_padded = embeddings_padded.reshape(-1, emb_elems_per_ext_mem_width)
            ret = pack_innermost_dim_as_hex_string(
                embeddings_padded, edt, self.ext_mem_width, True, prefix=""
            )
            weight_filename = code_gen_dir / f"{self.onnx_node.name}.dat"
            weight_filename.write_text("".join(f"{line}\n" for line in ret))
        else:
            raise FINNInternalError(f"{self.onnx_node.name}: unrecognized mem_mode {self.mem_mode}")

    def execute_node(self, context: dict[str, np.ndarray], graph: "GraphProto") -> None:
        """Execute node."""
        if self.mem_mode != "internal_embedded":
            raise FINNUserError(
                f"{self.onnx_node.name}: only mem_mode=internal_embedded is supported for "
                f"simulation of the Lookup layer"
            )
        HLSBackend.execute_node(self, context, graph)

    def get_ap_int_max_w(self) -> int:
        """Return ap int max w."""
        parent_max = super().get_ap_int_max_w()
        if self.mem_mode == "external":
            return max(self.ext_mem_width, parent_max)
        return parent_max

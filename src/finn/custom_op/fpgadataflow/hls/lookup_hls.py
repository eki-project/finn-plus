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
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING

from finn.custom_op.fpgadataflow.hls import register_custom_op
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.hwcustomop import NodeAttrTypes
from finn.custom_op.fpgadataflow.lookup import Lookup
from finn.util.basic import get_vivado_version, is_versal
from finn.util.data_packing import (
    hls_datatype_str,
    numpy_to_hls_code,
    pack_innermost_dim_as_hex_string,
)
from finn.util.exception import FINNInternalError, FINNUserError

if TYPE_CHECKING:
    from onnx import GraphProto, NodeProto

RAM_STYLES = {"auto": "AUTO", "block": "BRAM", "distributed": "LUTRAM", "ultra": "URAM"}


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

    def _check_uram_codegen_support(self, fpgapart: str) -> None:
        """Check that internal_embedded URAM tables are supported.

        Both the target part and the HLS version must support URAM-backed
        embedding tables.

        Args:
            fpgapart: The target FPGA part.

        Raises:
            FINNUserError: If URAM storage is requested for an unsupported part or
                Vivado/Vitis HLS version.
        """
        if self.mem_mode != "internal_embedded" or self.ram_style != "ultra":
            return
        if not is_versal(fpgapart):
            raise FINNUserError(
                f"{self.onnx_node.name}: Lookup_hls with internal_embedded URAM requires a "
                "Versal target. Use a non-URAM ram_style or an external/runtime-writeable "
                "memory implementation for non-Versal targets."
            )
        vivado_version = get_vivado_version()
        if vivado_version is not None and vivado_version < (2024, 2):
            raise FINNUserError(
                f"{self.onnx_node.name}: Lookup_hls with internal_embedded URAM requires "
                "Vivado/Vitis HLS 2024.2 or newer because older HLS versions cannot "
                "initialize URAM-backed global/static embedding tables."
            )

    def code_generation_ipgen(self, model: ModelWrapper, fpgapart: str, clk: float) -> None:
        """Generate C++ code and TCL script for IP generation."""
        self._check_uram_codegen_support(fpgapart)
        super().code_generation_ipgen(model, fpgapart, clk)

    def global_includes(self) -> None:
        """Return global includes."""
        global_incls = ['#include "lookup.hpp"']
        if self.mem_mode == "internal_embedded":
            global_incls.append('#include "embeddings.hpp"')
        self.code_gen_dict["$GLOBALS$"] = global_incls

    def defines(self, var: str) -> None:  # noqa: ARG002
        """Return defines."""
        elem_hls_type = hls_datatype_str(self.get_input_datatype())
        emb_hls_type = hls_datatype_str(self.embedding_type)
        my_defines = []
        if self.mem_mode == "external":
            ext_mem_emb_size = self.get_folded_output_shape()[-2]
            ext_mem_emb_align = ceil(log2(ext_mem_emb_size))
            my_defines += [
                f"static constexpr unsigned MemBits = {self.ext_mem_width};",
                f"static constexpr unsigned EmbeddingSize = {ext_mem_emb_size};",
                f"static constexpr unsigned EmbeddingAlign = {ext_mem_emb_align};",
                f"using T_SRC = {elem_hls_type};",
                "using T_DST = ap_uint<MemBits>;",
            ]
        elif self.mem_mode == "internal_embedded":
            my_defines += [
                f"static constexpr unsigned NumEmbeddings = {self.num_embeddings};",
                f"static constexpr unsigned EmbeddingDim = {self.embedding_dim};",
                f"using InputType = {elem_hls_type};",
                f"using EmbeddingType = {emb_hls_type};",
            ]
        self.code_gen_dict["$DEFINES$"] = my_defines

    def dataoutstrm(self) -> None:
        """Return dataoutstrm."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        npy_type = "float"
        npy_out = f"{code_gen_dir}/output_0.npy"
        oshape_cpp_str = str(self.get_folded_output_shape()).replace("(", "{").replace(")", "}")

        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            f'tensorstream2npy<{npy_type}>(strm0, {oshape_cpp_str}, "{npy_out}");'
        ]

    def docompute(self) -> None:
        """Return docompute."""
        if self.mem_mode == "internal_embedded":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """StreamingLookup<NumEmbeddings, EmbeddingDim,
                InputType, EmbeddingType>(in0_V, out0_V, embeddings);"""
            ]
        elif self.mem_mode == "external":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """StreamingLookup_ext<EmbeddingSize>(in0_V, out0_V, mem, size, oob_count,
                oob_irq);"""
            ]

    def blackboxfunction(self) -> None:
        """Return blackboxfunction."""
        if self.mem_mode == "internal_embedded":
            i_hls_type = hls_datatype_str(self.get_input_datatype())
            i_fold = self.get_folded_input_shape()[-1]
            emb_hls_type = hls_datatype_str(self.embedding_type)
            emb_dim = self.embedding_dim
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                (
                    f"void {self.onnx_node.name}("
                    f"hls::stream<tensor<{i_hls_type},{i_fold}>> &in0_V, "
                    f"hls::stream<tensor<{emb_hls_type},{emb_dim}>> &out0_V)"
                )
            ]
        elif self.mem_mode == "external":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                (
                    f"void {self.onnx_node.name}(hls::stream<T_SRC> &in0_V, "
                    f"hls::stream<T_DST> &out0_V, "
                    f"T_DST const *const  mem, unsigned const size, "
                    f"unsigned &oob_count, bool &oob_irq)"
                )
            ]

    def pragmas(self) -> None:
        """Return pragmas."""
        my_pragmas = [
            "#pragma HLS INTERFACE axis port=in0_V",
            "#pragma HLS INTERFACE axis port=out0_V",
            "#pragma HLS INTERFACE ap_ctrl_none port=return",
            "#pragma HLS DATAFLOW disable_start_propagation",
        ]
        if self.mem_mode == "internal_embedded":
            ram_style = RAM_STYLES[self.ram_style]
            storage_type = "RAM_S2P" if ram_style == "URAM" else "ROM_2P"
            my_pragmas.append(
                f"#pragma HLS BIND_STORAGE variable=embeddings type={storage_type} impl={ram_style}"
            )
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

    def generate_params(
        self,
        model: ModelWrapper,
        path: str | Path,
        fpgapart: str | None = None,  # noqa: ARG002
    ) -> None:
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
            # aggregate initialization: tensor<EmbeddingType, EmbeddingDim>
            # handles element-to-bit mapping internally, no reversal needed
            emb_hls_type = hls_datatype_str(edt)
            init_code = numpy_to_hls_code(embeddings, edt, "embeddings", False, True)
            decl = (
                f"tensor<{emb_hls_type}, {self.embedding_dim}> "
                f"embeddings[{self.num_embeddings}] = \n"
            )
            (code_gen_dir / "embeddings.hpp").write_text(decl + init_code)
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

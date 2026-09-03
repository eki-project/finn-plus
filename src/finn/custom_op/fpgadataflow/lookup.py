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

"""Embedding-lookup hardware custom operator (index-to-value gather)."""

import numpy as np
import onnxruntime as rt
from math import ceil
from onnx import NodeProto, TensorProto, helper
from qonnx.core.datatype import BaseDataType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model
from typing import TYPE_CHECKING, cast

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.exception import FINNInternalError
from finn.util.logging import log

if TYPE_CHECKING:
    from onnx import GraphProto

# Type of the dictionary returned by get_nodeattr_types: maps attribute names to
# their (dtype, required, default[, allowed_values]) specification tuples
NodeAttrTypes = dict[
    str,
    tuple[str, bool, int | float | str | bool | np.ndarray | list]
    | tuple[str, bool, int | float | str | bool | np.ndarray | list, set | None],
]


class Lookup(HWCustomOp):
    """Abstraction layer for HW implementation of a streaming embedding lookup.

    Maps a stream of integer indices to the corresponding rows of an embedding
    table (equivalent to an ONNX ``Gather`` with a constant data operand). The
    table can be baked into the bitstream (``mem_mode="internal_embedded"``,
    BRAM) or fetched from external memory over AXI-MM (``mem_mode="external"``).
    """

    def __init__(self, onnx_node: NodeProto, **kwargs: int) -> None:
        """Initialize instance."""
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self) -> NodeAttrTypes:
        """Return nodeattr types."""
        my_attrs: NodeAttrTypes = {
            # Number of embeddings ("memory depth")
            "NumEmbeddings": ("i", True, 0),
            # Dimensionality of each embedding (part of "memory width")
            "EmbeddingDim": ("i", True, 0),
            # Datatype for embeddings (part of "memory width")
            "EmbeddingType": ("s", True, ""),
            # Datatype for inputs
            "InputType": ("s", True, ""),
            # Input shape
            "InputShape": ("ints", False, [1]),
            # Memory mode
            # internal_embedded : parameters baked into bitfile (BRAM)
            # external : lookup performed in external memory over AXI MM
            "mem_mode": ("s", False, "internal_embedded", {"internal_embedded", "external"}),
            # Width for AXI-MM interface
            # only relevant when mem_mode="external"
            "ext_mem_width": ("i", False, 32),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    @property
    def num_embeddings(self) -> int:
        """Get the number of embeddings (memory depth)."""
        return cast("int", self.get_nodeattr("NumEmbeddings"))

    @property
    def embedding_dim(self) -> int:
        """Get the dimensionality of a single embedding."""
        return cast("int", self.get_nodeattr("EmbeddingDim"))

    @property
    def embedding_type(self) -> BaseDataType:
        """Get the FINN DataType of the embedding values."""
        return DataType[cast("str", self.get_nodeattr("EmbeddingType"))]

    @property
    def input_type(self) -> BaseDataType:
        """Get the FINN DataType of the index inputs."""
        return DataType[cast("str", self.get_nodeattr("InputType"))]

    @property
    def input_shape(self) -> list[int]:
        """Get the shape of the index input stream."""
        return list(cast("list[int]", self.get_nodeattr("InputShape")))

    @property
    def mem_mode(self) -> str:
        """Get the memory mode (``internal_embedded`` or ``external``)."""
        return cast("str", self.get_nodeattr("mem_mode"))

    @property
    def ext_mem_width(self) -> int:
        """Get the AXI-MM interface width (only used when ``mem_mode="external"``)."""
        return cast("int", self.get_nodeattr("ext_mem_width"))

    def get_exp_cycles(self) -> int:
        """Return exp cycles."""
        return int(np.prod(self.input_shape))

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return normal input shape."""
        if ind == 0:
            return tuple(self.input_shape)
        if ind == 1:
            return (self.num_embeddings, self.embedding_dim)
        raise FINNInternalError(f"{self.onnx_node.name}: undefined input ind {ind} for Lookup")

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return normal output shape."""
        return (*self.get_normal_input_shape(), self.embedding_dim)

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        """Return folded input shape."""
        if ind == 0:
            return (*self.get_normal_input_shape(), 1)
        return tuple(self.get_normal_input_shape(ind))

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:  # noqa: ARG002
        """Return folded output shape."""
        ishape = self.get_normal_input_shape()
        if self.mem_mode == "internal_embedded":
            return (*ishape, self.embedding_dim)
        if self.mem_mode == "external":
            bits_per_emb_elem = self.get_output_datatype().bitwidth()
            if self.ext_mem_width % bits_per_emb_elem != 0:
                raise FINNInternalError(
                    f"{self.onnx_node.name}: ext_mem_width ({self.ext_mem_width}) must be a "
                    f"multiple of the embedding element width ({bits_per_emb_elem})"
                )
            emb_elems_per_ext_mem_width = self.ext_mem_width // bits_per_emb_elem
            return (
                *ishape,
                self.embedding_dim // emb_elems_per_ext_mem_width,
                emb_elems_per_ext_mem_width,
            )
        raise FINNInternalError(f"{self.onnx_node.name}: unrecognized mem_mode {self.mem_mode}")

    def infer_node_datatype(self, model: ModelWrapper) -> None:
        """Infer node datatype."""
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            log.warning(
                f"InputType changing for {node.name}: {self.get_input_datatype()!s} -> {idt!s}"
            )
        self.set_nodeattr("InputType", idt.name)
        model.set_tensor_datatype(node.output[0], self.embedding_type)

    def get_input_datatype(self, ind: int = 0) -> BaseDataType:
        """Return input datatype."""
        if ind == 0:
            return self.input_type
        if ind == 1:
            return self.embedding_type
        raise FINNInternalError(f"{self.onnx_node.name}: undefined input ind {ind} for Lookup")

    def get_output_datatype(self, ind: int = 0) -> BaseDataType:  # noqa: ARG002
        """Return output datatype."""
        return self.embedding_type

    def get_instream_width(self, ind: int = 0) -> int:
        """Return instream width."""
        if ind == 0:
            return self.get_input_datatype().bitwidth()
        if ind == 1:
            if self.mem_mode == "internal_embedded":
                return 0
            return self.ext_mem_width
        raise FINNInternalError(f"{self.onnx_node.name}: undefined input ind {ind} for Lookup")

    def get_outstream_width(self, ind: int = 0) -> int:  # noqa: ARG002
        """Return outstream width."""
        return self.get_output_datatype().bitwidth() * self.get_folded_output_shape()[-1]

    def execute_node(
        self, context: dict[str, np.ndarray], graph: "GraphProto"  # noqa: ARG002
    ) -> None:
        """Execute node.

        Uses an ONNX Runtime ``Gather`` node to compute the result.
        """
        node = self.onnx_node
        inp_values = context[node.input[0]]
        data_values = context[node.input[1]]
        oshape = context[node.output[0]].shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.INT64, inp_values.shape)
        data = helper.make_tensor_value_info(node.input[1], TensorProto.FLOAT, data_values.shape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)
        node_gather = helper.make_node(
            "Gather",
            inputs=[node.input[1], node.input[0]],
            outputs=[node.output[0]],
        )
        graph_gather = helper.make_graph(
            nodes=[node_gather],
            name="single-gather-exec",
            inputs=[data, inp],
            outputs=[outp],
        )

        opset_imports = [helper.make_opsetid("", 13)]
        model_gather = qonnx_make_model(graph_gather, opset_imports=opset_imports)
        idict = {node.input[0]: inp_values, node.input[1]: data_values}
        sess = rt.InferenceSession(model_gather.SerializeToString())
        result = sess.run(None, idict)
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(oshape)

    def bram_estimation(self) -> int:
        """Return bram estimation."""
        if self.mem_mode == "internal_embedded":
            # current calculation assumes embeddings always stored in BRAM_18Ks
            # when mem_mode is internal_embedded
            width_factor = ceil(self.get_outstream_width() / 16)
            depth_factor = ceil(self.num_embeddings / 1024)
            return width_factor * depth_factor
        # TODO can we estimate BRAMs for the DMA engine?
        return 0

    def bram_efficiency_estimation(self) -> float:
        """Return bram efficiency estimation."""
        bram16_est = self.bram_estimation()
        if bram16_est == 0:
            return 1.0
        ebits = self.get_outstream_width() * self.num_embeddings
        bram16_est_capacity = bram16_est * 18 * 1024
        return ebits / bram16_est_capacity

    def get_verilog_top_module_intf_names(self) -> dict[str, list[tuple[str, int]] | list[str]]:
        """Return the names of the interface signals for the verilog top module."""
        intf_names = super().get_verilog_top_module_intf_names()
        if self.mem_mode == "external":
            intf_names["axilite"] = ["s_axi_control"]
            intf_names["aximm"] = [("m_axi_gmem", self.ext_mem_width)]
            intf_names["ap_none"] = ["oob_irq"]
        return intf_names

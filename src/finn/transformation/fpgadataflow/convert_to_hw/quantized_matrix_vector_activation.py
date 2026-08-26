# Copyright (C) 2023-2024, Advanced Micro Devices, Inc.
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

"""Convert MatMul layers with quantized inputs and weights to MVAU HW layers."""

from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from typing import TYPE_CHECKING, cast

from finn.util.exception import FINNInternalError, FINNUserError

if TYPE_CHECKING:
    import numpy as np


class InferQuantizedMatrixVectorActivation(Transformation):
    """Convert MatMul layers with quantized inputs and weights to
    MatrixVectorActivation layers."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply transformation to convert MatMul to MVAU nodes."""
        graph = model.graph
        graph_modified = False
        for node_ind, n in enumerate(graph.node, start=1):
            if n.op_type == "MatMul" and model.get_tensor_sparsity(n.input[1]) is None:
                mm_input = n.input[0]
                mm_weight = n.input[1]
                mm_output = n.output[0]
                if (mm_in_shape := model.get_tensor_shape(mm_input)) is None or (
                    mm_out_shape := model.get_tensor_shape(mm_output)
                ) is None:
                    raise FINNUserError(f"{n.name}: Expected shape information to be available.")
                idt = model.get_tensor_datatype(mm_input)
                wdt = model.get_tensor_datatype(mm_weight)
                w = cast("np.ndarray | None", model.get_initializer(mm_weight))
                if idt.is_integer() and wdt.is_integer():
                    # extract weight shape, note that ONNX and finn-hlslib
                    # make different assumptions about dim order here
                    # ONNX assumes W has (in, out) shape
                    # finn-hlslib assumes W has (out, in) shape
                    if w is None:
                        # dynamic
                        if (mm_dyn_shape := model.get_tensor_shape(mm_weight)) is None:
                            raise FINNUserError(
                                f"{n.name}: Expected shape information to be available."
                            )
                        mh = int(mm_dyn_shape[-1])
                        mw = int(mm_dyn_shape[-2])
                    else:
                        # static
                        mh = int(w.shape[1])
                        mw = int(w.shape[0])
                    # create node with no parallelization first
                    pe = 1
                    simd = 1
                    wmem = mw * mh // (pe * simd)
                    if mw * mh != wmem * pe * simd:
                        raise FINNInternalError(
                            f"{n.name}: Requirement (MW * MH) divisible by "
                            "(WMEM * PE * SIMD) is violated."
                        )
                    # see if we have any following thresholds
                    consumers = model.find_consumers(mm_output)
                    # Only a single consumer node can be absorbed. Absorbing one
                    # branch of a forking matmul would lead to detached nodes
                    # breaking the graph.
                    consumer = consumers[0] if len(consumers) == 1 else None
                    if consumer is not None and consumer.op_type == "MultiThreshold":
                        # TODO ensure integer thresholds?
                        # create MVTU (i.e. including activation)
                        mt_output = consumer.output[0]
                        if (mt_out_shape := model.get_tensor_shape(mt_output)) is None:
                            raise FINNUserError(
                                f"{consumer.name}: Expected shape information to be available."
                            )
                        mt_thres = consumer.input[1]
                        if (
                            t := cast("np.ndarray | None", model.get_initializer(mt_thres))
                        ) is None:
                            raise FINNUserError(
                                f"{consumer.name}: Cannot infer MVAU, thresholds are not static."
                            )
                        if t.shape[0] != 1 and t.shape[0] != mh:
                            raise FINNUserError(
                                f"{consumer.name}: First dimension of thresholds neither 1 nor MH."
                            )
                        odt = model.get_tensor_datatype(mt_output)
                        scale = getCustomOp(consumer).get_nodeattr("out_scale")
                        actval = cast("float", getCustomOp(consumer).get_nodeattr("out_bias"))
                        if int(actval) != actval:
                            raise FINNUserError(
                                f"{consumer.name}: out_bias must be integer for HLS conversion."
                            )
                        actval = int(actval)
                        odt_is_bipolar = odt == DataType["BIPOLAR"]
                        bipolar_ok = odt_is_bipolar and (scale == 2.0) and (actval == -1)
                        if scale != 1.0 and not bipolar_ok:
                            raise FINNUserError(
                                f"{consumer.name}: out_scale=1 or bipolar output needed for "
                                "conversion."
                            )
                        if odt.signed() and actval >= 0:
                            raise FINNUserError(
                                f"{consumer.name}: Signed output requires actval < 0"
                            )
                        model.set_tensor_shape(mm_input, mm_in_shape)
                        model.set_tensor_shape(mt_output, mt_out_shape)
                        if bipolar_ok:
                            # remove bias for bipolar, since
                            # binary->bipolar is achieved by reinterpretation
                            actval = 0
                        # create and insert new MatrixVectorActivation node
                        new_node = helper.make_node(
                            "MVAU",
                            [mm_input, mm_weight, mt_thres],
                            [mt_output],
                            domain="finn.custom_op.fpgadataflow",
                            backend="fpgadataflow",
                            MW=mw,
                            MH=mh,
                            SIMD=simd,
                            PE=pe,
                            inputDataType=idt.name,
                            weightDataType=wdt.name,
                            outputDataType=odt.name,
                            ActVal=actval,
                            binaryXnorMode=0,
                            noActivation=0,
                            numInputVectors=list(mm_in_shape[:-1]),
                            name="MVAU_" + n.name,
                            dynamic_input=w is None,
                            inFIFODepths=[2, 2] if w is None else [2],
                        )
                        graph.node.insert(node_ind, new_node)
                        # remove old nodes
                        graph.node.remove(n)
                        graph.node.remove(consumer)
                        graph_modified = True
                    else:
                        # no activation, matmul only
                        odt = model.get_tensor_datatype(mm_output)
                        model.set_tensor_shape(mm_input, mm_in_shape)
                        model.set_tensor_shape(mm_output, mm_out_shape)
                        # create and insert new MatrixVectorActivation node
                        new_node = helper.make_node(
                            "MVAU",
                            [mm_input, mm_weight],
                            [mm_output],
                            domain="finn.custom_op.fpgadataflow",
                            backend="fpgadataflow",
                            MW=mw,
                            MH=mh,
                            SIMD=simd,
                            PE=pe,
                            inputDataType=idt.name,
                            weightDataType=wdt.name,
                            outputDataType=odt.name,
                            ActVal=0,
                            binaryXnorMode=0,
                            noActivation=1,
                            numInputVectors=list(mm_in_shape[:-1]),
                            name="MVAU_" + n.name,
                            dynamic_input=w is None,
                            inFIFODepths=[2, 2] if w is None else [2],
                        )
                        graph.node.insert(node_ind, new_node)
                        # remove old node
                        graph.node.remove(n)
                        graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

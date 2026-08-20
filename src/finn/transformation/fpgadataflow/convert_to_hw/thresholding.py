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

"""Convert MultiThreshold layers to standalone Thresholding HW layers."""

import qonnx.core.data_layout as dl
from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.onnx import nchw_to_nhwc
from typing import cast

from finn.util.exception import FINNUserError
from finn.util.logging import log


class InferThresholdingLayer(Transformation):
    """Convert any MultiThreshold into a standalone thresholding layer."""

    def __init__(self) -> None:
        """Initialize the transformation."""
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transformation to infer standalone thresholding layers."""
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "MultiThreshold":
                thl_input = node.input[0]
                thl_threshold = node.input[1]
                thl_output = node.output[0]
                thl_in_shape = model.get_tensor_shape(thl_input)
                thl_thres_shape = model.get_tensor_shape(thl_threshold)
                if thl_in_shape is None or thl_thres_shape is None:
                    raise FINNUserError(
                        f"{node.name}: Expected shape information to exist for infer step."
                    )
                idt = model.get_tensor_datatype(thl_input)
                tdt = model.get_tensor_datatype(thl_threshold)

                # only infer layers where input and thresholds are integers, floats, or fixed-point
                idt_int = idt.is_integer()
                tdt_int = tdt.is_integer()
                idt_fp = idt in ["FLOAT32", "FLOAT16"]
                tdt_fp = tdt in ["FLOAT32", "FLOAT16"]
                idt_fxp = idt.is_fixed_point()
                tdt_fxp = tdt.is_fixed_point()
                if not (idt_int or idt_fp or idt_fxp):
                    continue
                if not (tdt_int or tdt_fp or tdt_fxp):
                    continue

                # Ad-hoc conversion of NCHW MT to NHWC MT by wrapping it in Transpose nodes
                # TODO: this should be removed in favor of proper layout handling in the frontend
                #  this workaround is currently still needed to handle standalone NCHW MTs at the
                #  input of the graph, e.g., for cnv (bnn-pynq) models
                node_inst = getCustomOp(node)
                try:
                    mt_layout = cast("str", node_inst.get_nodeattr("data_layout"))
                    string_to_layout_map = {
                        "NHWC": dl.NHWC,
                        "NCHW": dl.NCHW,
                        "NCW": dl.NCW,
                        "NWC": dl.NWC,
                        "NC": dl.NC,
                    }
                    if mt_layout in string_to_layout_map:
                        mt_layout = string_to_layout_map[mt_layout]
                except AttributeError:
                    log.warning(f"MultiThreshold ({node.name}) is missing a layout annotation.")
                    mt_layout = "missing"
                input_tensor_layout = model.get_tensor_layout(thl_input)
                output_tensor_layout = model.get_tensor_layout(thl_output)

                if input_tensor_layout != mt_layout:
                    log.warning(
                        f"MultiThreshold ({node.name}) layout ({mt_layout}) does not match "
                        f"input tensor layout ({input_tensor_layout})."
                    )
                if output_tensor_layout != mt_layout:
                    log.warning(
                        f"MultiThreshold ({node.name}) layout ({mt_layout}) does not match "
                        f"output tensor layout ({output_tensor_layout})."
                    )

                if input_tensor_layout == dl.NCHW and output_tensor_layout == dl.NHWC:
                    raise FINNUserError(
                        f"MultiThreshold ({node.name}) input (NCHW) and output (NHWC) "
                        "layout mismatch."
                    )
                if input_tensor_layout == dl.NHWC and output_tensor_layout == dl.NCHW:
                    raise FINNUserError(
                        f"MultiThreshold ({node.name}) input (NHWC) and output (NCHW) "
                        "layout mismatch."
                    )

                # Perform conversion only if both, input & output, are annotated as NCHW
                convert = False
                if input_tensor_layout == dl.NCHW and output_tensor_layout == dl.NCHW:
                    convert = True

                if convert:
                    thl_input = nchw_to_nhwc(thl_input, model, node_ind)
                    node_ind += 1
                    if (thl_in_shape := model.get_tensor_shape(thl_input)) is None:
                        raise FINNUserError(
                            f"{node.name}: Expected shape information to exist for infer step."
                        )

                # keep track of where we need to insert the HLS Op
                # it has to be ahead of the output transform
                insert_point = node_ind

                if convert:
                    thl_output = nchw_to_nhwc(thl_output, model, node_ind, reverse=True)
                    node_ind += 1

                # now safe to assume number of channels is in last dimension
                ifc = int(thl_in_shape[-1])
                # create node with no parallelization first
                pe = 1

                odt = model.get_tensor_datatype(thl_output)
                scale = getCustomOp(node).get_nodeattr("out_scale")
                if scale != 1.0:
                    raise FINNUserError(
                        f"{node.name}: MultiThreshold out_scale must be 1 for HLS conversion."
                    )
                actval = cast("float", getCustomOp(node).get_nodeattr("out_bias"))
                if int(actval) != actval:
                    raise FINNUserError(
                        f"{node.name}: MultiThreshold out_bias must be integer for HLS conversion."
                    )
                actval = int(actval)

                # a signed activation should always have a negative bias,
                # but BIPOLAR uses the -1 as 0 encoding so the check does not apply
                if odt != DataType["BIPOLAR"] and odt.signed() and actval >= 0:
                    raise FINNUserError(f"{node.name}: Signed output requires actval < 0")

                new_node = helper.make_node(
                    "Thresholding",
                    [thl_input, thl_threshold],
                    [thl_output],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    NumChannels=ifc,
                    PE=pe,
                    numSteps=thl_thres_shape[1],
                    inputDataType=idt.name,
                    weightDataType=tdt.name,
                    outputDataType=odt.name,
                    numInputVectors=list(thl_in_shape[:-1]),
                    ActVal=actval,
                    name="Thresholding_" + node.name,
                )
                graph.node.insert(insert_point, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        return (model, graph_modified)

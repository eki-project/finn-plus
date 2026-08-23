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

"""Convert MultiThreshold or Quant nodes to Requant HW layers."""

import numpy as np
import qonnx.core.data_layout as data_layout
from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.onnx import nchw_to_nhwc
from typing import cast

from finn.util.logging import log


def _check_uniform_thresholds(
    thresholds: "np.ndarray", rtol: float = 1e-2
) -> "tuple[bool, np.ndarray, np.ndarray]":
    """Check if thresholds have uniform (equal) step sizes per channel.

    For requant conversion, thresholds must be uniform (equal step sizes)
    within each channel. Different channels may have different step sizes.

    Args:
        thresholds: numpy array of shape (num_channels, num_thresholds)
        rtol: relative tolerance for comparing step sizes (default 1%)

    Returns:
        tuple: (is_uniform, step_sizes, first_thresholds) where:
            - is_uniform: True if all channels have uniform steps
            - step_sizes: array of step size per channel
            - first_thresholds: array of first threshold per channel
    """
    num_channels = thresholds.shape[0]
    num_thresholds = thresholds.shape[1]

    if num_thresholds < 2:
        # Single threshold, trivially uniform with step=1
        return True, np.ones(num_channels), thresholds[:, 0]

    step_sizes = []
    first_thresholds = []
    is_uniform = True

    for ch in range(num_channels):
        ch_thresholds = np.sort(thresholds[ch])
        diffs = np.diff(ch_thresholds)

        if len(diffs) > 0:
            step_size = diffs[0]
            step_sizes.append(step_size)
            first_thresholds.append(ch_thresholds[0])

            # Check if all steps are equal (within tolerance)
            if not np.allclose(diffs, step_size, rtol=rtol):
                is_uniform = False
        else:
            step_sizes.append(1.0)
            first_thresholds.append(ch_thresholds[0])

    return is_uniform, np.array(step_sizes), np.array(first_thresholds)


class InferRequantLayer(Transformation):
    """Convert MultiThreshold or Quant nodes to Requant.

    For MultiThreshold nodes where all channels have uniform (equal-step) thresholds,
    the comparison-based threshold lookup can be replaced with a simpler
    requantization operation:

        output = clip(round(input * scale + bias), min, max)

    where:
        scale = 1.0 / step_size
        bias = 0.5 - first_threshold / step_size

    For Quant nodes with scale=1 and zeropt=0 (after ExtractQuantScaleZeroPt),
    the operation simplifies to:

        output = clip(round(input), min, max)

    which is Requant with scale=1 and bias=0.

    This transformation is optional and provides an alternative implementation
    to InferThresholdingLayer. The Requant node can then be specialized to
    either HLS or RTL backend.
    """

    def __init__(self) -> None:
        """Initialize instance."""
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply transformation."""
        graph = model.graph
        node_ind = 0
        graph_modified = False

        for node in graph.node:
            node_ind += 1

            # Variables to be set by either MultiThreshold or Quant handling
            inp_name = None
            out_name = None
            in_shape = None
            idt = None
            odt = None
            scales = None
            biases = None
            narrow = None

            if node.op_type == "MultiThreshold":
                inp_name = node.input[0]
                mt_threshold = node.input[1]
                out_name = node.output[0]
                in_shape = model.get_tensor_shape(inp_name)
                mt_thres_shape = model.get_tensor_shape(mt_threshold)
                if in_shape is None or mt_thres_shape is None:
                    log.warning(f"{node.name}: Missing shape information. Cannot infer Requant.")
                    continue

                idt = model.get_tensor_datatype(inp_name)
                odt = model.get_tensor_datatype(out_name)

                # Only infer layers where input is integer, fixed-point, or float
                idt_ok = (
                    idt.is_integer()
                    or idt.is_fixed_point()
                    or idt in [DataType["FLOAT32"], DataType["FLOAT16"]]
                )
                if not idt_ok:
                    continue

                # Get threshold values
                thresholds = cast("np.ndarray | None", model.get_initializer(mt_threshold))
                if thresholds is None:
                    log.warning(
                        f"{node.name}: Thresholds not found as initializer. "
                        "Cannot infer RequantLayer."
                    )
                    continue

                # Check if thresholds are uniform (per channel)
                is_uniform, step_sizes, first_thresholds = _check_uniform_thresholds(thresholds)

                if not is_uniform:
                    # Thresholds are not uniform, cannot use requant
                    continue

                # Check MultiThreshold out_scale and out_bias
                mt_inst = getCustomOp(node)
                out_scale = mt_inst.get_nodeattr("out_scale")
                out_bias = cast("float", mt_inst.get_nodeattr("out_bias"))

                if out_scale != 1.0:
                    log.warning(
                        f"{node.name}: MultiThreshold out_scale must be 1 for "
                        "RequantLayer conversion."
                    )
                    continue

                # Compute requant scale and bias per channel
                # For uniform thresholds: output = floor((input - T0) / step) + 1
                # which is equivalent to: round(input * (1/step) + (0.5 - T0/step))
                scales = 1.0 / step_sizes
                biases = 0.5 - first_thresholds / step_sizes

                # Adjust for out_bias (ActVal in Thresholding)
                biases = biases + out_bias

                # Determine narrow range from number of thresholds
                # Full range: num_thresholds = 2^bitwidth - 1
                # Narrow range: num_thresholds = 2^bitwidth - 2
                num_thresholds = mt_thres_shape[1]
                bitwidth = odt.bitwidth()
                expected_full_range = 2**bitwidth - 1
                narrow = 1 if num_thresholds < expected_full_range else 0

            elif node.op_type == "Quant":
                # Handle Quant nodes with scale=1 and zeropt=0
                # (typically after ExtractQuantScaleZeroPt transformation)
                node_inst = getCustomOp(node)

                # Check rounding mode
                rmode = cast("str", node_inst.get_nodeattr("rounding_mode"))
                if rmode.upper() != "ROUND":
                    continue

                # Get scale, zeropt, bitwidth from initializers
                scale = cast("np.ndarray | None", model.get_initializer(node.input[1]))
                if scale is None:
                    continue
                zeropt = cast("np.ndarray | None", model.get_initializer(node.input[2]))
                if zeropt is None:
                    continue
                bitwidth_init = cast("np.ndarray | None", model.get_initializer(node.input[3]))
                if bitwidth_init is None:
                    continue

                # Check scale=1 and zeropt=0
                if not (np.all(scale == 1.0) and np.all(zeropt == 0.0)):
                    # Need ExtractQuantScaleZeroPt first
                    continue

                # Extract bitwidth
                if bitwidth_init.size != 1:
                    continue
                bitwidth = int(bitwidth_init.item())

                inp_name = node.input[0]
                out_name = node.output[0]
                if (in_shape := model.get_tensor_shape(inp_name)) is None:
                    log.warning(f"{node.name}: Missing shape information. Cannot infer Requant.")
                    continue

                idt = model.get_tensor_datatype(inp_name)
                odt = model.get_tensor_datatype(out_name)

                # For Quant with scale=1, zeropt=0: output = clip(round(input), min, max)
                # This is Requant with scale=1 and bias=0
                num_channels = int(in_shape[-1])
                scales = np.ones(num_channels, dtype=np.float32)
                biases = np.zeros(num_channels, dtype=np.float32)

                # Get narrow from Quant node attribute
                narrow = node_inst.get_nodeattr("narrow")

            else:
                # Not a supported node type
                continue

            # Common code for both MultiThreshold and Quant

            # Check layout and convert if necessary
            in_layout = model.get_tensor_layout(inp_name)
            if in_layout == data_layout.NCHW:
                inp_name = nchw_to_nhwc(inp_name, model, node_ind)
                node_ind += 1
                if (in_shape := model.get_tensor_shape(inp_name)) is None:
                    log.warning(f"{node.name}: Missing shape information. Cannot infer Requant.")
                    continue

            # Keep track of where we need to insert the HW Op
            insert_point = node_ind
            out_layout = model.get_tensor_layout(out_name)
            if out_layout == data_layout.NCHW:
                out_name = nchw_to_nhwc(out_name, model, node_ind, reverse=True)
                node_ind += 1

            # Now safe to assume number of channels is in last dimension
            num_channels = int(in_shape[-1])

            # Create scale and bias tensors as initializers
            scale_tensor = scales.astype(np.float32)
            bias_tensor = biases.astype(np.float32)

            scale_name = f"{node.name}_scale"
            bias_name = f"{node.name}_bias"

            model.set_initializer(scale_name, scale_tensor)
            model.set_initializer(bias_name, bias_tensor)

            # Create the Requant node
            new_node = helper.make_node(
                "Requant",
                [inp_name, scale_name, bias_name],
                [out_name],
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                NumChannels=num_channels,
                PE=1,
                inputDataType=idt.name,
                outputDataType=odt.name,
                numInputVectors=list(in_shape[:-1]),
                narrow=narrow,
                name="Requant_" + node.name,
            )

            graph.node.insert(insert_point, new_node)
            # Remove old node
            graph.node.remove(node)
            graph_modified = True

        return (model, graph_modified)

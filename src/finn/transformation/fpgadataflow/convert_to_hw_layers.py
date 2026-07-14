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

"""Transformations to map ONNX operators to FINN hardware layers."""

import numpy as np
import qonnx.core.data_layout as DataLayout
from collections.abc import Callable
from onnx import NodeProto, TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name, remove_by_name
from qonnx.util.onnx import nchw_to_nhwc
from typing import TYPE_CHECKING, Literal, cast

# Module containing specializations of elementwise binary operations
import finn.custom_op.fpgadataflow.elementwise_binary as elementwise_binary

# Base class for all FINN custom ops, here just used for type-hinting
from finn.util.basic import getHWCustomOp
from finn.util.exception import FINNUserError
from finn.util.logging import log

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class InferConvInpGen(Transformation):
    """Convert Im2Col layers to ConvolutionInputGenerator layers."""

    def __init__(self) -> None:
        """Initialize the transformation."""
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transformation to infer ConvolutionInputGenerator layers."""
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Im2Col":
                i2c_input = n.input[0]
                i2c_output = n.output[0]
                if (i2c_in_shape := model.get_tensor_shape(i2c_input)) is None:
                    log.warning(f"{n.name} : Input shape is None. Can't infer ConvInpGen.")
                    continue
                if (i2c_out_shape := model.get_tensor_shape(i2c_output)) is None:
                    log.warning(f"{n.name} : Output shape is None. Can't infer ConvInpGen.")
                    continue
                dt = model.get_tensor_datatype(i2c_input)
                if not dt.is_integer():
                    log.warning(f"{n.name} : Input is not int. Can't infer ConvInpGen.")
                    continue
                i2c_inst = getCustomOp(n)
                stride_h, stride_w = cast("list[int]", i2c_inst.get_nodeattr("stride"))
                k_h, k_w = cast("list[int]", i2c_inst.get_nodeattr("kernel_size"))
                pad_attr = cast("list[int]", i2c_inst.get_nodeattr("pad_amount"))
                pad_h = pad_attr[0] + pad_attr[2]
                pad_w = pad_attr[1] + pad_attr[3]
                dilation_h, dilation_w = cast("list[int]", i2c_inst.get_nodeattr("dilations"))
                pad_val = i2c_inst.get_nodeattr("pad_value")
                depthwise = i2c_inst.get_nodeattr("depthwise")
                ifm_ch = i2c_in_shape[-1]
                ifm_dim_h = i2c_in_shape[1]
                ifm_dim_w = i2c_in_shape[2]
                ofm_dim_h = i2c_out_shape[1]
                ofm_dim_w = i2c_out_shape[2]

                # default params for ConvolutionInputGenerator
                conv_inp_gen_node_idx = node_ind
                conv_inp_gen_input = i2c_input
                conv_inp_gen_idim_h = ifm_dim_h
                conv_inp_gen_idim_w = ifm_dim_w

                if pad_h > 0 or pad_w > 0:
                    if pad_val != 0:
                        raise FINNUserError(
                            f"{n.name}: FMPadding_Batch currently supports only pad_value=0, "
                            f"got pad_value={pad_val}."
                        )

                    odim_padding_h = ifm_dim_h + pad_h
                    odim_padding_w = ifm_dim_w + pad_w

                    padding_out = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        (1, odim_padding_h, odim_padding_w, ifm_ch),
                    )
                    graph.value_info.append(padding_out)
                    padding_out = padding_out.name
                    model.set_tensor_datatype(padding_out, dt)

                    conv_inp_gen_node_idx += 1
                    conv_inp_gen_input = padding_out
                    conv_inp_gen_idim_h = odim_padding_h
                    conv_inp_gen_idim_w = odim_padding_w

                    padding_node = helper.make_node(
                        "FMPadding",
                        [i2c_input],
                        [padding_out],
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        ImgDim=[ifm_dim_h, ifm_dim_w],
                        Padding=pad_attr,
                        NumChannels=ifm_ch,
                        inputDataType=dt.name,
                        SIMD=ifm_ch,
                        name="FMPadding_Batch_" + n.name,
                    )
                    graph.node.insert(node_ind, padding_node)

                is_1d = (ifm_dim_h == 1) or (ifm_dim_w == 1)
                conv_inp_gen_node = helper.make_node(
                    "ConvolutionInputGenerator",
                    [conv_inp_gen_input],
                    [i2c_output],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    ConvKernelDim=[k_h, k_w],
                    IFMChannels=ifm_ch,
                    IFMDim=[conv_inp_gen_idim_h, conv_inp_gen_idim_w],
                    OFMDim=[ofm_dim_h, ofm_dim_w],
                    SIMD=ifm_ch,
                    Stride=[stride_h, stride_w],
                    Dilation=[dilation_h, dilation_w],
                    inputDataType=dt.name,
                    outputDataType=dt.name,
                    depthwise=depthwise,
                    is1D=is_1d,
                    name="ConvolutionInputGenerator_" + n.name,
                )
                graph.node.insert(conv_inp_gen_node_idx, conv_inp_gen_node)
                # remove old nodes
                graph.node.remove(n)
                graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferFMPadding(Transformation):
    """Convert Pad layers to FMPadding layers."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transformation to the entire model graph."""
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False

        # Enumerate all node in the graph and check for standalone standard ONNX
        # padding operators
        for index, node in enumerate(graph.node):
            if node.op_type == "Pad":
                # FMPadding only implements constant padding
                if (mode := get_by_name(node.attribute, "mode")) is not None and mode.s.decode(
                    "ascii"
                ) != "constant":
                    continue

                # Input shape must describe 4d image layout to be compatible
                # with the FMPadding operator
                if (inp := model.get_tensor_shape(node.input[0])) is None:
                    continue
                if len(inp) != 4:
                    continue

                # Padding axes must be constant initializer tensors, we cannot
                # do runtime dynamic behavior
                if (axes := model.get_initializer(node.input[3])) is None:
                    continue

                # Assuming NHWC layout as expected by FMPadding, the axes must
                # be the first two (HW) following the batch dimension
                if list(axes) != [1, 2]:
                    continue

                # FMPadding only implements constant zero padding at the moment
                if (pad_value := model.get_initializer(node.input[2])) is None or pad_value != 0:
                    continue

                # Padding amount for each dimension must be constant and match
                # the HW dimensions
                if (pads := model.get_initializer(node.input[1])) is None:
                    continue

                if len(pads) != 4:
                    continue

                # Configure the FINN CustomOp replacement of the pad operator
                padding = helper.make_node(
                    "FMPadding",
                    [node.input[0]],
                    [*node.output],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    ImgDim=inp[1:3],
                    Padding=list(pads),
                    NumChannels=inp[-1],
                    inputDataType=model.get_tensor_datatype(node.input[0]).name,
                    SIMD=inp[-1],
                    name="FMPadding_" + node.name,
                )

                graph.node.insert(index, padding)
                graph.node.remove(node)

                # Consider the graph to be modified, triggering exhaustive
                # re-application of this transformation
                graph_modified = True
                # Exiting here triggers type and shape inference and cleanup
                # after each transformed node. This helps QONNX to behave
                # better/more consistent in certain cases...
                break

        # Re-do shape and data type annotations after potential changes to the
        # model graph
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


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
                        "NHWC": DataLayout.NHWC,
                        "NCHW": DataLayout.NCHW,
                        "NCW": DataLayout.NCW,
                        "NWC": DataLayout.NWC,
                        "NC": DataLayout.NC,
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

                if (
                    input_tensor_layout == DataLayout.NCHW
                    and output_tensor_layout == DataLayout.NHWC
                ):
                    raise FINNUserError(
                        f"MultiThreshold ({node.name}) input (NCHW) and output (NHWC) "
                        "layout mismatch."
                    )
                if (
                    input_tensor_layout == DataLayout.NHWC
                    and output_tensor_layout == DataLayout.NCHW
                ):
                    raise FINNUserError(
                        f"MultiThreshold ({node.name}) input (NHWC) and output (NCHW) "
                        "layout mismatch."
                    )

                # Perform conversion only if both, input & output, are annotated as NCHW
                convert = False
                if (
                    input_tensor_layout == DataLayout.NCHW
                    and output_tensor_layout == DataLayout.NCHW
                ):
                    convert = True

                if convert:
                    thl_input = nchw_to_nhwc(thl_input, model, node_ind)
                    node_ind += 1
                    thl_in_shape = model.get_tensor_shape(thl_input)

                # keep track of where we need to insert the HLS Op
                # it has to be ahead of the output transform
                insert_point = node_ind

                if convert:
                    thl_output = nchw_to_nhwc(thl_output, model, node_ind, reverse=True)
                    node_ind += 1

                if thl_in_shape is None or thl_thres_shape is None:
                    raise FINNUserError(
                        f"MultiThreshold ({node.name}) input or threshold shape is None."
                    )

                # now safe to assume number of channels is in last dimension
                ifc = int(thl_in_shape[-1])
                # create node with no parallelization first
                pe = 1

                odt = model.get_tensor_datatype(thl_output)
                scale = getCustomOp(node).get_nodeattr("out_scale")
                if scale != 1.0:
                    raise FINNUserError(
                        f"{node.name}: MultiThreshold out_scale must be 1.0 for HLS conversion, "
                        f"got {scale}."
                    )
                actval = cast("float", getCustomOp(node).get_nodeattr("out_bias"))
                if int(actval) != actval:
                    raise FINNUserError(
                        f"MultiThreshold ({node.name}) out_bias must be integer for HLS conversion."
                    )
                actval = int(actval)

                # a signed activation should always have a negative bias,
                # but BIPOLAR uses the -1 as 0 encoding so the assert does not apply
                if odt != DataType["BIPOLAR"] and odt.signed() and actval >= 0:
                    raise FINNUserError(
                        f"{node.name}: Signed output requires out_bias (ActVal) < 0, got {actval}."
                    )

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


def _check_uniform_thresholds(
    thresholds: np.ndarray, rtol: float = 1e-2
) -> tuple[bool, np.ndarray, np.ndarray]:
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
                if (in_shape := model.get_tensor_shape(inp_name)) is None:
                    raise FINNUserError(f"{node.name}: MultiThreshold input shape is None.")
                if (mt_thres_shape := model.get_tensor_shape(mt_threshold)) is None:
                    raise FINNUserError(f"{node.name}: MultiThreshold threshold shape is None.")

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
                thresholds = model.get_initializer(mt_threshold)
                if thresholds is None:
                    log.warning(
                        f"{node.name}: Thresholds not found as initializer. "
                        "Cannot infer RequantLayer."
                    )
                    continue

                # Check if thresholds are uniform (per channel)
                is_uniform, step_sizes, first_thresholds = _check_uniform_thresholds(
                    cast("np.ndarray", thresholds)
                )

                if not is_uniform:
                    # Thresholds are not uniform, cannot use requant
                    continue

                # Check MultiThreshold out_scale and out_bias
                mt_inst = getCustomOp(node)
                out_scale = cast("float", mt_inst.get_nodeattr("out_scale"))
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
                scale = model.get_initializer(node.input[1])
                if scale is None:
                    continue
                zeropt = model.get_initializer(node.input[2])
                if zeropt is None:
                    continue
                bitwidth = model.get_initializer(node.input[3])
                if bitwidth is None:
                    continue

                # Check scale=1 and zeropt=0
                if not (np.all(scale == 1.0) and np.all(zeropt == 0.0)):
                    # Need ExtractQuantScaleZeroPt first
                    continue

                # Extract bitwidth
                bw = cast("np.ndarray", bitwidth)
                if bw.size != 1:
                    continue
                bitwidth = int(bw.item())

                inp_name = node.input[0]
                out_name = node.output[0]
                if (in_shape := model.get_tensor_shape(inp_name)) is None:
                    raise FINNUserError(f"{node.name}: Quant input shape is None.")

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
            if (
                inp_name is None
                or out_name is None
                or in_shape is None
                or idt is None
                or odt is None
                or scales is None
                or biases is None
                or narrow is None
            ):
                raise FINNUserError(
                    f"{node.name}: Internal conversion state incomplete while inferring Requant. "
                    "Expected non-None input/output names, "
                    "shapes, datatypes, scales, biases and narrow."
                )

            # Check layout and convert if necessary
            in_layout = model.get_tensor_layout(inp_name)
            if in_layout == DataLayout.NCHW:
                inp_name = nchw_to_nhwc(inp_name, model, node_ind)
                node_ind += 1
                if (in_shape := model.get_tensor_shape(inp_name)) is None:
                    raise FINNUserError(
                        f"{node.name}: Input shape is None after layout conversion."
                    )

            # Keep track of where we need to insert the HW Op
            insert_point = node_ind
            out_layout = model.get_tensor_layout(out_name)
            if out_layout == DataLayout.NCHW:
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


class InferUpsample(Transformation):
    """Convert Upsample and Resize nodes to UpsampleNearestNeighbour nodes."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transformation to infer UpsampleNearestNeighbour nodes."""
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Upsample" or n.op_type == "Resize":
                # Extract mode and scales and input shape
                if (m := get_by_name(n.attribute, "mode")) is None:
                    raise FINNUserError(
                        f"{n.name}: Upsample/Resize mode attribute is missing. "
                        "Ensure 'mode' attribute is set."
                    )
                mode = m.s.decode("ascii")
                scales = None
                if n.op_type == "Upsample":
                    if (scales := model.get_initializer(n.input[1])) is None:
                        raise FINNUserError(
                            f"{n.name}: Upsample scales are None. "
                            "Ensure scale input is a static initializer."
                        )
                    scales = cast("np.ndarray", scales)
                else:
                    if len(n.input) == 2:
                        # Resize version 10
                        if (scales := model.get_initializer(n.input[1])) is None:
                            raise FINNUserError(
                                f"{n.name}: Resize scales are None. "
                                "Ensure scale input is a static initializer."
                            )
                        scales = cast("np.ndarray", scales)
                    elif len(n.input) == 3:
                        # Resize version 11 and up (no size input)
                        if (scales := model.get_initializer(n.input[2])) is None:
                            raise FINNUserError(
                                f"{n.name}: Resize scales are None. "
                                "Ensure scale input is a static initializer."
                            )
                        scales = cast("np.ndarray", scales)
                    elif len(n.input) == 4:
                        # Resize version 11 and up
                        resize_scales = model.get_initializer(n.input[2])
                        resize_sizes = model.get_initializer(n.input[3])
                        scales_exists = (resize_scales is not None) and (len(resize_scales) != 0)
                        sizes_exists = (resize_sizes is not None) and (len(resize_sizes) != 0)
                        if not (scales_exists ^ sizes_exists):
                            raise FINNUserError(
                                f"{n.name}: Exactly one of 'scales' or 'sizes' must be specified "
                                "for Resize (both absent or both present is invalid)."
                            )
                        resize_scales = cast("np.ndarray", resize_scales)
                        resize_sizes = cast("np.ndarray", resize_sizes)
                        if scales_exists:
                            # Scales input
                            scales = resize_scales
                        else:
                            # Convert sizes to scales
                            sizes = resize_sizes
                            data_input_size = model.get_tensor_shape(n.input[0])
                            if data_input_size is None:
                                raise FINNUserError(
                                    f"{n.name}: "
                                    "Input shape is None, cannot derive Resize scales from sizes."
                                )
                            scales = sizes / data_input_size
                in_shape = model.get_tensor_shape(n.input[0])

                if in_shape is None:
                    raise FINNUserError(
                        f"{n.name}: "
                        "Input shape is None. Please run InferShapes before InferUpsample."
                    )
                if scales is None:
                    raise FINNUserError(
                        f"{n.name}: Upsample/Resize scales are None. "
                        "Ensure scale/size inputs are static initializers."
                    )

                dt = model.get_tensor_datatype(n.input[0])
                if not dt.is_integer():
                    log.warning(f"{n.name}: Input not int. Can't infer UpsampleNearestNeighbour.")
                    continue

                if model.get_tensor_layout(n.input[0]) != DataLayout.NHWC:
                    log.warning(f"{n.name}: Input not NHWC. Can't infer UpsampleNearestNeighbour.")
                    continue

                # Check that the parameters are okay
                if mode != "nearest":
                    raise FINNUserError(
                        f"{n.name}: "
                        f"Upsampling mode '{mode}' is not supported; only 'nearest' is supported."
                    )
                if len(in_shape) != 4:
                    raise FINNUserError(
                        f"{n.name}: "
                        f"Upsampling is only supported for 4D inputs in NHWC, "
                        f"got shape {in_shape}."
                    )
                if scales.shape != (4,):
                    raise FINNUserError(
                        f"{n.name}: "
                        f"Upsampling requires 4D scales for NHWC, got shape {scales.shape}."
                    )
                if not (scales >= 1).all():
                    raise FINNUserError(
                        f"{n.name}: "
                        f"Upsampling supports only scales >= 1 in every dimension, "
                        f"got {scales}."
                    )

                # Assumes nhwc layout for scales and input
                if not (scales[0] == scales[3] == 1):
                    raise FINNUserError(
                        f"{n.name}: "
                        f"Upsampling in NHWC requires scales[0] == scales[3] == 1, "
                        f"got scales={scales}."
                    )

                # Extract information for HW node
                hi = in_shape[1]
                wi = in_shape[2]
                ho = round(hi * scales[1])
                wo = round(wi * scales[2])
                num_channels = in_shape[-1]
                batch_size = in_shape[0]
                idt = dt.name

                # Insert the HWCustomOp node
                upsample_hw_node = helper.make_node(
                    "UpsampleNearestNeighbour",
                    [n.input[0]],
                    [n.output[0]],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    SIMD=1,
                    HO=ho,
                    WO=wo,
                    HI=hi,
                    WI=wi,
                    NumChannels=num_channels,
                    inputDataType=idt,
                    batchSize=batch_size,
                    name="UpsampleNearestNeighbour_" + n.name,
                    cpp_interface="hls_vector",
                    hls_style="freerunning",
                )

                # Remove the old node
                graph.node.insert(node_ind, upsample_hw_node)
                # remove old nodes
                graph.node.remove(n)
                graph_modified = True
        return (model, graph_modified)


class InferDuplicateStreamsLayer(Transformation):
    """Insert a DuplicateStreams HW layer for any tensor with fanout >= 2."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transformation to insert DuplicateStreams HW layers where needed."""
        graph = model.graph
        graph_modified = False
        # check first if global input is split
        successors = model.find_consumers(graph.input[0].name)
        dt = model.get_tensor_datatype(graph.input[0].name)
        if successors is not None and len(successors) >= 2:
            output_tensor = graph.input[0].name
            n_outputs = len(successors)

            # create clone tensors
            out_shape = model.get_tensor_shape(output_tensor)
            if out_shape is None:
                raise FINNUserError(
                    f"{output_tensor}: Tensor shape is None while inserting DuplicateStreams. "
                    "Please run InferShapes first."
                )
            out_tensor_clones = []
            for _i in range(n_outputs):
                clone = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, out_shape
                )
                model.graph.value_info.append(clone)
                out_tensor_clones += [clone.name]

            num_ch = int(out_shape[-1])
            vecs = out_shape[:-1]

            # create node with no parallelization first
            pe = 1

            dup_node = helper.make_node(
                "DuplicateStreams",
                [output_tensor],
                out_tensor_clones,
                domain="finn.custom_op.fpgadataflow",
                backend="fpgadataflow",
                NumChannels=num_ch,
                PE=pe,
                inputDataType=dt.name,
                numInputVectors=vecs,
                NumOutputStreams=n_outputs,
                outFIFODepths=[2] * n_outputs,
                name="DuplicateStreams_" + output_tensor,
                cpp_interface="hls_vector",
                hls_style="freerunning",
            )

            graph.node.insert(0, dup_node)

            # connect successors to out tensor clone
            clone_idx = 0
            for successor in successors:
                for i, succ_input in enumerate(successor.input):
                    if succ_input == output_tensor:
                        successor.input[i] = out_tensor_clones[clone_idx]
                        clone_idx += 1
                        # if one node has multiple connections to the same output
                        # find_direct_successors will return one node per input
                        # so break the inner loop will result in correct behaviour
                        break
            graph_modified = True

        for node_ind, node in enumerate(graph.node):
            for output_tensor in node.output:
                successors = model.find_consumers(output_tensor)
                # check if this tensor is also a global output
                is_global_output = any(out.name == output_tensor for out in graph.output)
                # determine total number of consumers (successors + global output)
                num_successors = len(successors) if successors is not None else 0
                total_consumers = num_successors + (1 if is_global_output else 0)

                if total_consumers >= 2:
                    n_outputs = total_consumers

                    dt = model.get_tensor_datatype(output_tensor)

                    # create clone tensors
                    out_shape = model.get_tensor_shape(output_tensor)
                    new_global_output_tensor = None
                    if out_shape is None:
                        raise FINNUserError(
                            f"{node.name}: Output tensor {output_tensor} shape is None while "
                            "inserting DuplicateStreams."
                        )
                    out_tensor_clones = []
                    for i in range(n_outputs):
                        clone = helper.make_tensor_value_info(
                            model.make_new_valueinfo_name(), TensorProto.FLOAT, out_shape
                        )
                        # if one is a global output reserve
                        # the last out tensor clone for that connection
                        if i == (n_outputs - 1) and is_global_output:
                            new_global_output_tensor = clone
                        # else add it to the value info container
                        else:
                            model.graph.value_info.append(clone)
                        out_tensor_clones += [clone.name]

                    num_ch = int(out_shape[-1])
                    vecs = out_shape[:-1]

                    # create node with no parallelization first
                    pe = 1

                    dup_node = helper.make_node(
                        "DuplicateStreams",
                        [output_tensor],
                        out_tensor_clones,
                        domain="finn.custom_op.fpgadataflow",
                        backend="fpgadataflow",
                        NumChannels=num_ch,
                        PE=pe,
                        inputDataType=dt.name,
                        numInputVectors=vecs,
                        NumOutputStreams=n_outputs,
                        outFIFODepths=[2] * n_outputs,
                        name="DuplicateStreams_" + node.name,
                        cpp_interface="hls_vector",
                        hls_style="freerunning",
                    )

                    graph.node.insert(node_ind, dup_node)

                    # connect successors to out tensor clone
                    clone_idx = 0
                    for successor in successors:
                        for i, succ_input in enumerate(successor.input):
                            if succ_input == output_tensor:
                                successor.input[i] = out_tensor_clones[clone_idx]
                                clone_idx += 1
                                # if one node has multiple connections to the same output
                                # find_direct_successors will return one node per input
                                # so break the inner loop will result in correct behaviour
                                break

                    # if the tensor is a global output, connect the last clone to it
                    if is_global_output:
                        for i, graph_out in enumerate(graph.output):
                            if graph_out.name == output_tensor:
                                if new_global_output_tensor is None:
                                    raise FINNUserError(
                                        f"{node.name}: "
                                        "Global output tensor clone is None while inserting "
                                        "DuplicateStreams."
                                    )
                                graph.output[i].CopyFrom(new_global_output_tensor)
                                break

                    graph_modified = True

        if graph_modified:
            model = model.transform(SortGraph())
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferLabelSelectLayer(Transformation):
    """Convert any TopK into a LabelSelect HW layer."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply transformation to convert TopK nodes to LabelSelect hardware layers.

        This transformation identifies TopK operations and converts them to FINN's
        custom LabelSelect nodes for hardware acceleration.
        """
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "TopK":
                fc_input = node.input[0]
                k_input = node.input[1]
                val_output = node.output[0]
                idx_output = node.output[1]
                fc_in_shape = model.get_tensor_shape(fc_input)
                if fc_in_shape is None:
                    raise FINNUserError(
                        f"{node.name}: TopK input shape is None. Please run InferShapes first."
                    )

                idt = model.get_tensor_datatype(fc_input)

                # skip conversion for layers with float input
                if not idt.is_integer():
                    continue

                # skip conversion for if value output is connected (not supported)
                if model.find_consumer(val_output) is not None:
                    continue

                num_labels = int(fc_in_shape[-1])
                num_inp_vecs = list(fc_in_shape[:-1])
                # create node with no parallelization first
                pe = 1

                k_init = model.get_initializer(k_input)
                if k_init is None or cast("np.ndarray", k_init).size == 0:
                    raise FINNUserError(
                        f"{node.name}: TopK K input must be a non-empty constant initializer."
                    )
                k = cast("np.ndarray", k_init)[0]

                # create and insert new LabelSelect node
                new_node = helper.make_node(
                    "LabelSelect",
                    [fc_input],
                    [idx_output],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    Labels=num_labels,
                    PE=pe,
                    K=k,
                    inputDataType=idt.name,
                    numInputVectors=num_inp_vecs,
                    name="LabelSelect_" + node.name,
                )
                graph.node.insert(node_ind, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferGlobalAccPoolLayer(Transformation):
    """Convert any GlobalAveragePool into a GlobalAccPool HW layer and a scalar Mul."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply transformation to infer GlobalAccPool hardware layers."""
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "GlobalAveragePool":
                in0 = node.input[0]
                result = node.output[0]
                in0_shape = model.get_tensor_shape(in0)
                if in0_shape is None:
                    raise FINNUserError(
                        f"{node.name}: GlobalAveragePool input shape is None. "
                        "Please run InferShapes first."
                    )

                idt = model.get_tensor_datatype(in0)

                # skip conversion for layers with float input
                if not idt.is_integer():
                    continue

                # check layout and convert if necessary
                in0_layout = model.get_tensor_layout(in0)
                result_layout = model.get_tensor_layout(result)

                if in0_layout == DataLayout.NCHW:
                    in0 = nchw_to_nhwc(in0, model, node_ind)
                    node_ind += 1
                    in0_shape = model.get_tensor_shape(in0)
                    if in0_shape is None:
                        raise FINNUserError(
                            f"{node.name}: Input shape is None after NCHW->NHWC conversion."
                        )

                # keep track of where we need to insert the HW Op
                # it has to be ahead of the output transform
                insert_point = node_ind

                if result_layout == DataLayout.NCHW:
                    result = nchw_to_nhwc(result, model, node_ind, reverse=True)
                    node_ind += 1

                num_ch = int(in0_shape[-1])
                vecs = in0_shape[:-1]
                if len(vecs) < 3:
                    raise FINNUserError(
                        f"{node.name}: "
                        f"GlobalAccPool expects NHWC input rank 4, got shape {in0_shape}."
                    )
                # create node with no parallelization first
                pe = 1

                # create an additional tensor of the same shape and layout as result
                out_shape = model.get_tensor_shape(result)
                if out_shape is None:
                    raise FINNUserError(
                        f"{node.name}: Output shape is None while inferring GlobalAccPool."
                    )
                pool_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, out_shape
                )
                model.graph.value_info.append(pool_out)
                pool_out = pool_out.name
                if (tl := model.get_tensor_layout(result)) is None:
                    raise FINNUserError(
                        f"{node.name}: Output tensor {result} already has no layout set. "
                        "Cannot infer GlobalAccPool."
                    )
                model.set_tensor_layout(pool_out, tl)

                new_pool = helper.make_node(
                    "GlobalAccPool",
                    [in0],
                    [pool_out],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    NumChannels=num_ch,
                    PE=pe,
                    inputDataType=idt.name,
                    numInputVectors=vecs,
                    name="GlobalAccPool_" + node.name,
                )

                mul_value = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(), TensorProto.FLOAT, [1]
                )
                model.graph.value_info.append(mul_value)
                model.set_initializer(
                    mul_value.name, np.array(1 / (vecs[1] * vecs[2]), dtype=np.float32)
                )
                new_mul = helper.make_node(
                    "Mul",
                    [pool_out, mul_value.name],
                    [result],
                )
                graph.node.insert(insert_point, new_pool)
                graph.node.insert(insert_point + 1, new_mul)
                node_ind += 1
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


# class InferPool(Transformation):
#     """If kernel_shape > strides, replace Pool layer with Im2col + pool combination.

#     When kernel_shape > strides, replaces Pool layer with Im2col followed by
#     pool (with kernel_shape == strides), plus Transpose layers to keep the
#     original data layout.
#     """

#     def apply(self, model):
#         """Apply transformation to convert Pool operations with kernel_shape > strides."""
#         graph = model.graph
#         node_ind = 0
#         graph_modified = False
#         for node in graph.node:
#             node_ind += 1
#             if node.op_type in ["MaxPool", "QuantAvgPool2d", "MaxPoolNHWC"]:
#                 node_input = node.input[0]
#                 ishape = model.get_tensor_shape(node_input)
#                 node_output = node.output[0]
#                 idt = model.get_tensor_datatype(node_input)
#                 oshape = model.get_tensor_shape(node_output)
#                 # only support 4D input tensors (1D convs need extra dummy dim)
#                 if len(ishape) != 4:
#                     continue

#                 # extract pool parameters
#                 if node.op_type == "MaxPool":
#                     kh, kw = list(get_by_name(node.attribute, "kernel_shape").ints)
#                     sh, sw = list(get_by_name(node.attribute, "strides").ints)
#                     dlayout = "NCHW"
#                 elif node.op_type == "QuantAvgPool2d":
#                     inst = getCustomOp(node)
#                     # QuantAvgPool2d has a single scalar attribute
#                     # for kernel size and stride (implicit square)
#                     kh = kw = inst.get_nodeattr("kernel")
#                     sh = sw = inst.get_nodeattr("stride")
#                     dlayout = inst.get_nodeattr("data_layout")
#                 elif node.op_type == "MaxPoolNHWC":
#                     inst = getCustomOp(node)
#                     kh, kw = inst.get_nodeattr("kernel_shape")
#                     sh, sw = inst.get_nodeattr("strides")
#                     dlayout = "NHWC"
#                 try:
#                     pad = list(get_by_name(node.attribute, "pads").ints)
#                 except AttributeError:
#                     pad = [0, 0, 0, 0]

#                 if not idt.is_integer():
#                     continue

#                 if (kh < sh) or (kw < sw):
#                     # TODO check/implement swg support
#                     continue

#                 odt = model.get_tensor_datatype(node_output)

#                 if dlayout == "NCHW":
#                     _, ifm_ch, ifm_h, ifm_w = ishape
#                     _, ofm_ch, ofm_h, ofm_w = oshape
#                 elif dlayout == "NHWC":
#                     _, ifm_h, ifm_w, ifm_ch = ishape
#                     _, ofm_h, ofm_w, ofm_ch = oshape
#                 else:
#                     raise Exception("Unknown dlayout: " + str(dlayout))

#                 # if data layout NCHW, we need transpose nodes surrounding
#                 # the hw layer
#                 if dlayout == "NCHW":
#                     # create new intermediate values
#                     inp_trans_out = helper.make_tensor_value_info(
#                         model.make_new_valueinfo_name(),
#                         TensorProto.FLOAT,
#                         (1, ifm_h, ifm_w, ifm_ch),  # NHWC
#                     )
#                     graph.value_info.append(inp_trans_out)
#                     inp_trans_out = inp_trans_out.name
#                     model.set_tensor_datatype(inp_trans_out, idt)

#                     pool_output = helper.make_tensor_value_info(
#                         model.make_new_valueinfo_name(),
#                         TensorProto.FLOAT,
#                         (1, ofm_h, ofm_w, ofm_ch),
#                     )
#                     graph.value_info.append(pool_output)
#                     pool_output = pool_output.name

#                 im2col_out = helper.make_tensor_value_info(
#                     model.make_new_valueinfo_name(),
#                     TensorProto.FLOAT,
#                     (1, ofm_h, ofm_w, ifm_ch * kh * kw),
#                 )
#                 graph.value_info.append(im2col_out)
#                 im2col_out = im2col_out.name
#                 model.set_tensor_datatype(im2col_out, idt)

#                 # create new nodes
#                 if dlayout == "NCHW":
#                     # NCHW -> NHWC
#                     inp_trans_node = helper.make_node(
#                         "Transpose", [node_input], [inp_trans_out], perm=[0, 2, 3, 1]
#                     )
#                     im2col_in = inp_trans_out
#                 else:
#                     im2col_in = node_input
#                     pool_output = node_output

#                 accum_bits = 0
#                 pool_size_param = 0  # will be overridden if neededs
#                 pad_value = 0
#                 if node.op_type in ["MaxPool", "MaxPoolNHWC"]:
#                     pool_fxn = "MaxPool"
#                     odt = idt
#                     pad_value = idt.min()
#                 elif node.op_type == "QuantAvgPool2d":
#                     assert odt.is_integer(), """Output data type for QuantAvgPool2d
#                     needs to be integer"""
#                     assert all(x == 0 for x in pad), "Padding is not supported for QuantAvgPool2d"
#                     inst = getCustomOp(node)
#                     pool_fxn = "QuantAvgPool"
#                     pool_size_param = inst.get_shifts()
#                     accum_bits = inst.get_accum_size()

#                 else:
#                     raise Exception(f"pad_value and pool_fxn not configured for {node.op_type}")

#                 # format input tensor
#                 im2col_node = helper.make_node(
#                     "Im2Col",
#                     [im2col_in],
#                     [im2col_out],
#                     domain="qonnx.custom_op.general",
#                     stride=[sh, sw],
#                     kernel_size=[kh, kw],
#                     pad_amount=pad,
#                     pad_value=pad_value,
#                     depthwise=1,
#                     input_shape=f"(1,{ifm_h},{ifm_w},{ifm_ch})",
#                     name="Im2Col_" + node.name,
#                 )

#                 # Warning PE has to be equal to ifm_ch until Im2Col is replaced by
#                 # ConvolutionInputGenerator with depthwise=1.
#                 # For other settings the output will be incorrect due to incorrect input
#                 # data layout
#                 pool_node = helper.make_node(
#                     "Pool",
#                     [im2col_out],
#                     [pool_output],
#                     domain="finn.custom_op.fpgadataflow",
#                     backend="fpgadataflow",
#                     InputDataType=idt.name,
#                     OutputDataType=odt.name,
#                     Channels=ifm_ch,
#                     PE=ifm_ch,
#                     KernelSize=[kh, kw],
#                     Function=pool_fxn,
#                     OutImgDims=[ofm_h, ofm_w],
#                     AccumBits=accum_bits,
#                     Size=pool_size_param,
#                     BatchSize=1,
#                     cpp_interface="hls_vector",
#                     name="Pool_" + node.name,
#                 )

#                 if dlayout == "NCHW":
#                     # NHWC -> NCHW
#                     out_trans_node = helper.make_node(
#                         "Transpose", [pool_output], [node_output], perm=[0, 3, 1, 2]
#                     )

#                 # insert nodes where the conv is to preserve topological ordering
#                 if dlayout == "NCHW":
#                     graph.node.insert(node_ind, inp_trans_node)
#                     graph.node.insert(node_ind + 1, im2col_node)
#                     graph.node.insert(node_ind + 2, pool_node)
#                     graph.node.insert(node_ind + 3, out_trans_node)
#                 else:
#                     graph.node.insert(node_ind, im2col_node)
#                     graph.node.insert(node_ind + 1, pool_node)
#                 # remove old node
#                 graph.node.remove(node)
#                 graph_modified = True

#         if graph_modified:
#             model = model.transform(InferShapes())
#             model = model.transform(InferDataTypes())
#         return (model, graph_modified)


# class InferPoolFromReduce(Transformation):
#     """Infer pooling hardware from lowered pooling, i.e., Im2Col+Reduce."""

#     def apply(self, model: ModelWrapper):
#         """Apply transformation to convert lowered pooling to hardware."""
#         # Get the model graph out of the model wrapper object
#         graph = model.graph
#         # Keep track of whether the graph has been modified
#         graph_modified = False

#         # Enumerate all node in the graph and check for standalone standard ONNX
#         # padding operators
#         for index, node in enumerate(graph.node):
#             if node.op_type in {"ReduceMax", "ReduceSum", "ReduceMean"}:
#                 # Reduction axes must be constants to turn this into hardware
#                 if (axes := model.get_initializer(node.input[1])) is None:
#                     continue

#                 # The input to the reduction must be produced by a Reshape
#                 # operator unpacking the channel axis from the kernel shape
#                 if (reshape := model.find_producer(node.input[0])) is None:
#                     continue

#                 if reshape.op_type != "Reshape":
#                     continue

#                 # The reshape must be static, i.e., the shape parameter is
#                 # constant
#                 if (shape := model.get_initializer(reshape.input[1])) is None:
#                     continue

#                 # Reduction must operate on the second to last axis, which is
#                 # the (spatial) extent of the pooling window
#                 if list(axes) != [-2] and list(axes) != [len(shape) - 2]:
#                     continue

#                 # The overall input must be produced from a sliding window input
#                 # generators, i.e., Im2Col operator
#                 if (im2col := model.find_producer(reshape.input[0])) is None:
#                     continue

#                 if im2col.op_type != "Im2Col":
#                     continue

#                 # Get the current input datatype annotation (Im2Col and Reshape
#                 # do not modify the datatype)
#                 idt = model.get_tensor_datatype(im2col.input[0])

#                 # Fallback: Assume output to be the same as the input and the
#                 # accumulator to be zero-sized
#                 odt, accum_bits = idt, 0

#                 # Simple type inference depending on the matched reduction
#                 # operator: Could be refined by minimize_accumulator_width
#                 if node.op_type == "ReduceMax":
#                     accum_bits = 0
#                     odt = idt

#                 if node.op_type == "ReduceSum":
#                     # Minimum and maximum accumulated value to expect from
#                     # reducing the input type over the reduction axis
#                     minimum = shape[-2] * idt.min()
#                     maximum = shape[-2] * idt.max()
#                     # The output datatype must be able to fit the larger
#                     # magnitude of the two
#                     if abs(minimum) > abs(maximum):
#                         odt = DataType.get_smallest_possible(minimum)
#                     else:
#                         odt = DataType.get_smallest_possible(maximum)
#                     # Accumulator size is the same as the output size
#                     accum_bits = odt.bitwidth()

#                 if node.op_type == "ReduceMean":
#                     # Minimum and maximum accumulated value to expect from
#                     # reducing the input type over the reduction axis
#                     minimum = shape[-2] * idt.min()
#                     maximum = shape[-2] * idt.max()
#                     # The accumulator datatype must be able to fit the larger
#                     # magnitude of the two
#                     if abs(minimum) > abs(maximum):
#                         acc = DataType.get_smallest_possible(minimum)
#                     else:
#                         acc = DataType.get_smallest_possible(maximum)
#                     # Accumulator size if the bitwidth of this accumulator type
#                     accum_bits = acc.bitwidth()
#                     # The output type is the same as the input, as it is
#                     # averaged over, i.e., divided by, the kernel size
#                     odt = idt

#                 # Annotate the output to use the inferred type instead of the
#                 # current type annotation
#                 model.set_tensor_datatype(node.output[0], odt)

#                 # Lookup the pooling backend function corresponding to this
#                 # reduction operator
#                 pool_fxn = {
#                     "ReduceMax": "MaxPool",
#                     "ReduceSum": "AccPool",
#                     "ReduceMean": "AvgPool",
#                 }[node.op_type]

#                 # This is indeed a supported pooling operator in its lowered
#                 # form: Keep Im2Col but replace Reshape+Reduce by HW operator
#                 pool = helper.make_node(
#                     # This is the pooling custom hardware operator from the FINN
#                     # custom domain
#                     op_type="Pool",
#                     domain="finn.custom_op.fpgadataflow",
#                     backend="fpgadataflow",
#                     # Connect the new operator to make use of the old inputs and
#                     # outputs
#                     inputs=[im2col.output[0]],
#                     outputs=[node.output[0]],
#                     # Pooling needs to know the input/output dimensions and the
#                     # size of the pooling window
#                     Channels=shape[-1],
#                     KernelSize=get_by_name(im2col.attribute, "kernel_size").ints,
#                     OutImgDims=shape[1:-2],
#                     # Select the pooling backend implementation
#                     Function=pool_fxn,
#                     # Set parallelism to cover all input (also output) channels
#                     PE=shape[-1],
#                     # Configure the size of the internal accumulator if
#                     # applicable (MaxPool ignores this)
#                     AccumBits=accum_bits,
#                     # Set the names of the input/output datatype
#                     InputDataType=idt.name,
#                     OutputDataType=odt.name,
#                     # Pooling backend already uses the new hls::vector interface
#                     cpp_interface="hls_vector",
#                 )

#                 # The input generator needs to be switched into depthwise mode,
#                 # as pooling does not reduce/expand along the channels
#                 getCustomOp(im2col).set_nodeattr("depthwise", 1)

#                 # Insert the pooling node into the graph, but do not remove the
#                 # old nodes as they might still have other consumers
#                 graph.node.insert(index, pool)

#                 # The reduction can always be removed, all consumers are rewired
#                 # to use the pooling output
#                 graph.node.remove(node)

#                 # If the reshape has only a single consumer, we can remove this
#                 # from the graph
#                 if len(model.find_consumers(reshape.output[0])) <= 1:
#                     graph.node.remove(reshape)

#         # Re-do shape and data type annotations after potential changes to the
#         # model graph
#         model = model.transform(InferShapes())
#         model = model.transform(InferDataTypes())

#         # Return the transformed model and indicate whether the graph actually
#         # has been transformed
#         return model, graph_modified


class InferLookupLayer(Transformation):
    """Convert Gather nodes with constant op0 into Lookup HW layers."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply transformation to convert Gather operations to Lookup hardware layers.

        This transformation identifies Gather operations with constant first operand
        and converts them to FINN's custom Lookup nodes for hardware acceleration.
        """
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "Gather":
                emb_name = node.input[0]
                embs = model.get_initializer(emb_name)
                axis = get_by_name(node.attribute, "axis")
                # skip conversion if input0 is not constant
                if embs is None:
                    continue
                # skip conversion if axis != 0
                if axis is not None and axis.i != 0:
                    continue
                ind_name = node.input[1]
                ind_dtype = model.get_tensor_datatype(ind_name)
                emb_dtype = model.get_tensor_datatype(emb_name)
                # skip conversion if inputs are not unsigned integers
                if (not ind_dtype.is_integer()) or ind_dtype.signed():
                    continue
                embs = cast("np.ndarray", embs)
                num_embs, emb_dim = embs.shape
                out_name = node.output[0]
                ishape = model.get_tensor_shape(node.input[1])
                if ishape is None:
                    raise FINNUserError(
                        f"{node.name}: Gather index input shape is None. "
                        "Please run InferShapes first."
                    )
                # create and insert new Lookup node
                new_node = helper.make_node(
                    "Lookup",
                    [ind_name, emb_name],
                    [out_name],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    name="Lookup_" + node.name,
                    NumEmbeddings=num_embs,
                    EmbeddingDim=emb_dim,
                    EmbeddingType=emb_dtype.name,
                    InputType=ind_dtype.name,
                    InputShape=list(ishape),
                )
                graph.node.insert(node_ind, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferConcatLayer(Transformation):
    """Convert suitable Concat nodes (operating on last/-1 axis) into StreamingConcat HW layers."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply transformation to convert Concat operations to StreamingConcat hardware layers.

        This transformation identifies Concat operations operating on the last axis
        and converts them to FINN's custom StreamingConcat nodes.
        """
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "Concat":
                ishape = model.get_tensor_shape(node.input[0])
                axis = get_by_name(node.attribute, "axis")
                if (axis is None) or (ishape is None):
                    continue
                axis = axis.i
                last_axis = len(ishape) - 1
                # skip conversion if not using last axis
                if (axis != -1) and (axis != last_axis):
                    continue
                # check datatype coherence
                if any(model.get_tensor_datatype(x) is None for x in node.input):
                    log.warning(
                        "Inputs with undefined datatype detected, skipping InferConcatLayer()"
                    )
                    continue
                # skip conversion if any inputs are static
                any_static = any(model.get_initializer(x) is not None for x in node.input)
                if any_static:
                    continue
                # skip conversion if inputs are not integers
                all_integer = all(model.get_tensor_datatype(x).is_integer() for x in node.input)
                if not all_integer:
                    log.warning(
                        "Inputs with non-integer datatype detected, skipping InferConcatLayer()"
                    )
                    continue
                # ready for conversion
                input_shapes = []
                for inp in node.input:
                    inp_shape = model.get_tensor_shape(inp)
                    if inp_shape is None:
                        raise FINNUserError(
                            f"{node.name}: "
                            f"Input shape for tensor {inp} is None in Concat conversion."
                        )
                    input_shapes.append(inp_shape)
                channels_per_stream = [inp_shape[-1] for inp_shape in input_shapes]
                inp_vec = list(input_shapes[0][:-1])
                new_node = helper.make_node(
                    "StreamingConcat",
                    node.input,
                    node.output,
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    name="StreamingConcat_" + node.name,
                    SIMD=1,
                    ChannelsPerStream=channels_per_stream,
                    inputDataTypes=[model.get_tensor_datatype(x).name for x in node.input],
                    numInputVectors=inp_vec,
                    inFIFODepths=[2] * len(node.input),
                    cpp_interface="hls_vector",
                    hls_style="freerunning",
                )
                graph.node.insert(node_ind, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferSplitLayer(Transformation):
    """Convert suitable Split nodes (operating on last/-1 axis) into StreamingSplit HW layers."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply transformation to convert Split operations to StreamingSplit hardware layers.

        This transformation identifies Split operations operating on the last axis
        and converts them to FINN's custom StreamingSplit nodes.
        """
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "Split":
                split_param = node.input[1]
                if model.get_initializer(split_param) is None:
                    log.warning("Split param not constant, skipping InferSplitLayer()")
                    continue
                ishape = model.get_tensor_shape(node.input[0])
                axis = get_by_name(node.attribute, "axis")
                if (axis is None) or (ishape is None):
                    continue
                axis = axis.i
                last_axis = len(ishape) - 1
                # skip conversion if not using last axis
                if (axis != -1) and (axis != last_axis):
                    log.warning(
                        "StreamingSplit supports only last axis, skipping InferSplitLayer()"
                    )
                    continue
                # only one input allowed (two including split_param)
                if len(node.input) != 2:
                    log.warning("Only one input allowed, skipping InferSplitLayer()")
                    continue
                # skip conversion if the input is static
                if model.get_initializer(node.input[0]) is not None:
                    log.warning("Static input detected, skipping InferSplitLayer()")
                    continue
                # skip conversion if inputs are not integers
                if not model.get_tensor_datatype(node.input[0]).is_integer():
                    log.warning("Non-integer input detected, skipping InferSplitLayer()")
                    continue
                # ready for conversion
                output_shapes = []
                for out in node.output:
                    out_shape = model.get_tensor_shape(out)
                    if out_shape is None:
                        raise FINNUserError(
                            f"{node.name}: "
                            "Output shape for tensor {out} is None in Split conversion."
                        )
                    output_shapes.append(out_shape)
                channels_per_stream = [out_shape[-1] for out_shape in output_shapes]
                inp_vec = list(ishape[:-1])
                # when creating the fpgadataflow node we remove the second parameter input
                new_node = helper.make_node(
                    "StreamingSplit",
                    [node.input[0]],
                    node.output,
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    name="StreamingSplit_" + node.name,
                    SIMD=1,
                    cpp_interface="hls_vector",
                    hls_style="freerunning",
                    ChannelsPerStream=channels_per_stream,
                    inputDataType=model.get_tensor_datatype(node.input[0]).name,
                    numInputVectors=inp_vec,
                    outFIFODepths=[2] * len(node.output),
                )
                graph.node.insert(node_ind, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferBinaryMatrixVectorActivation(Transformation):
    """Convert XnorPopcountMatMul layers to MatrixVectorActivation layers.

    Any immediately following MultiThreshold layers will also be absorbed into the MVTU.
    """

    def __init__(self) -> None:
        """Initialize the transformation."""
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply transformation to convert XnorPopcountMatMul to MVAU nodes.

        This transformation identifies XnorPopcountMatMul operations and converts them
        to FINN's custom MVAU (Matrix Vector Activation Unit) nodes, potentially
        absorbing following MultiThreshold layers.
        """
        graph = model.graph
        graph_modified = False
        for node_ind, n in enumerate(graph.node):
            if n.op_type == "XnorPopcountMatMul":
                mm_input = n.input[0]
                mm_weight = n.input[1]
                mm_output = n.output[0]
                mm_in_shape = model.get_tensor_shape(mm_input)
                mm_out_shape = model.get_tensor_shape(mm_output)
                if mm_in_shape is None:
                    raise FINNUserError(
                        f"{n.name}: Input tensor shape is None for XnorPopcountMatMul. "
                        "Please run InferShapes first."
                    )
                if mm_out_shape is None:
                    raise FINNUserError(
                        f"{n.name}: Output tensor shape is None for XnorPopcountMatMul. "
                        "Please run InferShapes first."
                    )
                if model.get_tensor_datatype(mm_input) != DataType["BINARY"]:
                    raise FINNUserError(
                        f"{n.name}: First input for XnorPopcountMatMul must have FINN datatype "
                        f"BINARY, got {model.get_tensor_datatype(mm_input)}."
                    )
                if model.get_tensor_datatype(mm_weight) != DataType["BINARY"]:
                    raise FINNUserError(
                        f"{n.name}: Weight input for XnorPopcountMatMul must have FINN datatype "
                        f"BINARY, got {model.get_tensor_datatype(mm_weight)}."
                    )
                idt = DataType["BINARY"]
                wdt = DataType["BINARY"]
                mm_output = n.output[0]
                w = model.get_initializer(mm_weight)
                if w is None:
                    raise FINNUserError(
                        f"{n.name}: "
                        f"XnorPopcountMatMul requires static weights initializer, got None."
                    )
                w = cast("np.ndarray", w)
                # extract weight shape, note that ONNX and finn-hlslib
                # make different assumptions about dim order here
                # ONNX assumes W has (in, out) shape
                # finn-hlslib assumes W has (out, in) shape
                mh = int(w.shape[1])
                mw = int(w.shape[0])
                # create node with no parallelization first
                pe = 1
                simd = 1
                wmem = mw * mh // (pe * simd)
                if mw * mh != wmem * pe * simd:
                    raise FINNUserError(
                        f"{n.name}: Requirement (MW * MH) divisible by (WMEM * PE * SIMD) is "
                        f"violated: MW={mw}, MH={mh}, WMEM={wmem}, PE={pe}, SIMD={simd}."
                    )
                # see if we have any following thresholds
                consumers = model.find_consumers(mm_output) or []
                # Only a single consumer node can be absorbed. Absorbing one
                # branch of a forking matmul would lead to detached nodes
                # breaking the graph.
                consumer = consumers[0] if len(consumers) == 1 else None
                if consumer is not None and consumer.op_type == "MultiThreshold":
                    # TODO ensure integer thresholds?
                    # create MVTU (i.e. including activation)
                    mt_output = consumer.output[0]
                    mt_out_shape = model.get_tensor_shape(mt_output)
                    if mt_out_shape is None:
                        raise FINNUserError(
                            f"{consumer.name}: MultiThreshold output shape is None. "
                            "Please run InferShapes first."
                        )
                    mt_thres = consumer.input[1]
                    t = model.get_initializer(mt_thres)
                    if t is None:
                        raise FINNUserError(
                            f"{consumer.name}: Threshold tensor '{mt_thres}' initializer is None."
                        )
                    t = cast("np.ndarray", t)
                    if t.shape[0] != 1 and t.shape[0] != mh:
                        raise FINNUserError(
                            f"{consumer.name}: First threshold dimension must be 1 or MH={mh}, "
                            f"got {t.shape[0]}."
                        )
                    odt = model.get_tensor_datatype(mt_output)
                    # covers both bipolar and binary
                    actval = 0 if odt.bitwidth() == 1 else odt.min()
                    model.set_tensor_shape(mm_input, mm_in_shape)
                    model.set_tensor_shape(mt_output, mt_out_shape)
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
                        binaryXnorMode=1,
                        noActivation=0,
                        numInputVectors=list(mm_in_shape[:-1]),
                        name=n.name,
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
                        binaryXnorMode=1,
                        noActivation=1,
                        numInputVectors=list(mm_in_shape[:-1]),
                        name=n.name,
                    )
                    graph.node.insert(node_ind, new_node)
                    # remove old node
                    graph.node.remove(n)
                    graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferQuantizedMatrixVectorActivation(Transformation):
    """Convert MatMul layers with quantized inputs and weights to
    MatrixVectorActivation layers."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply transformation to convert MatMul to MVAU nodes."""
        graph = model.graph
        graph_modified = False
        for node_ind, n in enumerate(graph.node):
            if n.op_type == "MatMul" and model.get_tensor_sparsity(n.input[1]) is None:
                mm_input = n.input[0]
                mm_weight = n.input[1]
                mm_output = n.output[0]
                mm_in_shape = model.get_tensor_shape(mm_input)
                mm_out_shape = model.get_tensor_shape(mm_output)
                if mm_in_shape is None:
                    raise FINNUserError(
                        f"{n.name}: MatMul input shape is None. Please run InferShapes first."
                    )
                if mm_out_shape is None:
                    raise FINNUserError(
                        f"{n.name}: MatMul output shape is None. Please run InferShapes first."
                    )
                idt = model.get_tensor_datatype(mm_input)
                wdt = model.get_tensor_datatype(mm_weight)
                w = model.get_initializer(mm_weight)
                if idt.is_integer() and wdt.is_integer():
                    # extract weight shape, note that ONNX and finn-hlslib
                    # make different assumptions about dim order here
                    # ONNX assumes W has (in, out) shape
                    # finn-hlslib assumes W has (out, in) shape
                    if w is None:
                        # dynamic
                        mm_dyn_shape = model.get_tensor_shape(mm_weight)
                        if mm_dyn_shape is None or len(mm_dyn_shape) < 2:
                            raise FINNUserError(
                                f"{n.name}: "
                                "Dynamic weight shape for MatMul is invalid: {mm_dyn_shape}."
                            )
                        mh = int(mm_dyn_shape[-1])
                        mw = int(mm_dyn_shape[-2])
                    else:
                        # static
                        w = cast("np.ndarray", w)
                        mh = int(w.shape[1])
                        mw = int(w.shape[0])
                    # create node with no parallelization first
                    pe = 1
                    simd = 1
                    wmem = mw * mh // (pe * simd)
                    if mw * mh != wmem * pe * simd:
                        raise FINNUserError(
                            f"{n.name}: Requirement (MW * MH) divisible by (WMEM * PE * SIMD) is "
                            f"violated: MW={mw}, MH={mh}, WMEM={wmem}, PE={pe}, SIMD={simd}."
                        )
                    # see if we have any following thresholds
                    consumers = model.find_consumers(mm_output) or []
                    # Only a single consumer node can be absorbed. Absorbing one
                    # branch of a forking matmul would lead to detached nodes
                    # breaking the graph.
                    consumer = consumers[0] if len(consumers) == 1 else None
                    if consumer is not None and consumer.op_type == "MultiThreshold":
                        # TODO ensure integer thresholds?
                        # create MVTU (i.e. including activation)
                        mt_output = consumer.output[0]
                        mt_out_shape = model.get_tensor_shape(mt_output)
                        if mt_out_shape is None:
                            raise FINNUserError(
                                f"{consumer.name}: MultiThreshold output shape is None. "
                                "Please run InferShapes first."
                            )
                        mt_thres = consumer.input[1]
                        t = model.get_initializer(mt_thres)
                        if t is None:
                            raise FINNUserError(
                                f"{consumer.name}: "
                                f"Threshold tensor '{mt_thres}' initializer is None."
                            )
                        t = cast("np.ndarray", t)
                        if t.shape[0] != 1 and t.shape[0] != mh:
                            raise FINNUserError(
                                f"{consumer.name}: First threshold dimension must be 1 or MH={mh}, "
                                f"got {t.shape[0]}."
                            )
                        odt = model.get_tensor_datatype(mt_output)
                        scale = cast("float", getCustomOp(consumer).get_nodeattr("out_scale"))
                        actval = cast("float", getCustomOp(consumer).get_nodeattr("out_bias"))
                        if int(actval) != actval:
                            raise FINNUserError(
                                f"{consumer.name}: out_bias must be integer for HLS conversion, "
                                f"got {actval}."
                            )
                        actval = int(actval)
                        odt_is_bipolar = odt == DataType["BIPOLAR"]
                        bipolar_ok = odt_is_bipolar and (scale == 2.0) and (actval == -1)
                        if not (scale == 1.0 or bipolar_ok):
                            raise FINNUserError(
                                f"{consumer.name}: out_scale must be 1.0 for standard conversion "
                                "or "
                                f"(out_scale=2.0, out_bias=-1, BIPOLAR output) for bipolar "
                                "conversion. "
                                f"Got out_scale={scale}, out_bias={actval}, "
                                f"outputDataType={odt.name}."
                            )
                        if odt.signed() and actval >= 0:
                            raise FINNUserError(
                                f"{consumer.name}: Signed output requires actval < 0, got {actval}."
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


class InferVectorVectorActivation(Transformation):
    """Convert MatMul layers to VectorVectorActivation layers for depthwise convolutions.

    Converts MatMul layers with quantized inputs and weights to VectorVectorActivation
    layers, if the sparsity annotation of the weight matrix indicates that the MatMul
    layer belongs to a depthwise convolution. Any immediately following MultiThreshold
    layers will also be absorbed into the VVAU.
    """

    def __init__(self) -> None:
        """Initialize the transformation."""
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply transformation to convert MatMul to VVAU nodes for depthwise convolutions."""
        graph = model.graph
        graph_modified = False
        for node_ind, n in enumerate(graph.node):
            if n.op_type == "MatMul" and model.get_tensor_sparsity(n.input[1]) is not None:
                sparsity = model.get_tensor_sparsity(n.input[1])
                if sparsity is None:
                    raise FINNUserError(
                        f"{n.name}: MatMul weight tensor sparsity annotation is None."
                    )
                try:
                    k_h, k_w = sparsity["dw"]["kernel_shape"]
                except KeyError as err:
                    raise FINNUserError(
                        f"{n.name}: sparsity annotation doesn't indicate that MatMul"
                        f" belongs to a depthwise convolution."
                    ) from err

                mm_input = n.input[0]
                mm_weight = n.input[1]
                mm_output = n.output[0]
                if (mm_in_shape := model.get_tensor_shape(mm_input)) is None:
                    raise FINNUserError(f"{n.name}: Input tensor shape is not defined.")
                if (mm_out_shape := model.get_tensor_shape(mm_output)) is None:
                    raise FINNUserError(f"{n.name}: Output tensor shape is not defined.")
                idt = model.get_tensor_datatype(mm_input)
                wdt = model.get_tensor_datatype(mm_weight)
                if idt.is_integer() and wdt.is_integer():
                    mm_output = n.output[0]
                    w = model.get_initializer(mm_weight)
                    if w is None:
                        raise FINNUserError(
                            f"{n.name}: "
                            f"Depthwise MatMul conversion requires static weights initializer, "
                            "got None."
                        )
                    w = cast("np.ndarray", w)
                    # infer dense weight tensor from sparse weight matrix
                    # kernel size (k_h, k_w) which was extracted above and the value of
                    # the channels is used.
                    # the weight matrix has a shape of (k_h * k_w * Channels, Channels)
                    # we need to reverse the creation of the sparse weight matrix
                    # to achieve a weight tensor of shape (Channels, 1, k_h, k_w)
                    channels = int(w.shape[1])
                    # transpose to achieve a shape of (k_h * k_w * Channels, Channels)
                    w = w.T
                    # reshape to (Channels, k_h, k_w, Channels) to transpose afterwards
                    # to (Channels, Channels, k_h, k_w)
                    w = w.reshape(channels, k_h, k_w, channels)
                    w = w.transpose(0, 3, 1, 2)
                    # now we can extract the values using a for loop over the channels
                    # and fill a zero numpy array in the correct shape
                    w_tensor = np.zeros((channels, 1, k_h, k_w), dtype=np.float32)
                    for ch in range(channels):
                        w_tensor[ch][0] = w[ch][ch]
                    model.set_initializer(mm_weight, w_tensor)
                    model.set_tensor_shape(mm_weight, (channels, 1, k_h, k_w))
                    # create node with pe=channels as default
                    pe = channels
                    # see if we have any following thresholds
                    consumers = model.find_consumers(mm_output)
                    if consumers is None:
                        consumers = []
                    # Only a single consumer node can be absorbed. Absorbing one
                    # branch of a forking matmul would lead to detached nodes
                    # breaking the graph.
                    consumer = consumers[0] if len(consumers) == 1 else None
                    if consumer is not None and consumer.op_type == "MultiThreshold":
                        # create VVAU (i.e. including activation)
                        mt_output = consumer.output[0]
                        mt_out_shape = model.get_tensor_shape(mt_output)
                        if mt_out_shape is None:
                            raise FINNUserError(
                                f"{consumer.name}: MultiThreshold output shape is None. "
                                "Please run InferShapes first."
                            )
                        mt_thres = consumer.input[1]
                        t = model.get_initializer(mt_thres)
                        if t is None:
                            raise FINNUserError(
                                f"{consumer.name}: "
                                f"Threshold tensor '{mt_thres}' initializer is None."
                            )
                        t = cast("np.ndarray", t)
                        if t.shape[0] != 1 and t.shape[0] != channels:
                            raise FINNUserError(
                                f"{consumer.name}: First threshold dimension must be 1 or "
                                f"Channels={channels}, got {t.shape[0]}."
                            )
                        odt = model.get_tensor_datatype(mt_output)
                        scale = cast("float", getCustomOp(consumer).get_nodeattr("out_scale"))
                        if scale != 1.0:
                            raise FINNUserError(
                                f"{consumer.name}: out_scale must be 1.0 for VVAU HLS conversion, "
                                f"got {scale}."
                            )
                        actval = cast("float", getCustomOp(consumer).get_nodeattr("out_bias"))
                        if int(actval) != actval:
                            raise FINNUserError(
                                f"{consumer.name}: "
                                "out_bias must be integer for VVAU HLS conversion, "
                                f"got {actval}."
                            )
                        actval = int(actval)
                        if odt.signed() and actval >= 0:
                            raise FINNUserError(
                                f"{consumer.name}: Signed output requires actval < 0, got {actval}."
                            )
                        model.set_tensor_shape(mm_input, mm_in_shape)
                        model.set_tensor_shape(mt_output, mt_out_shape)
                        # create and insert new VectorVectorActivation node
                        new_node = helper.make_node(
                            "VVAU",
                            [mm_input, mm_weight, mt_thres],
                            [mt_output],
                            domain="finn.custom_op.fpgadataflow",
                            backend="fpgadataflow",
                            PE=pe,
                            Dim=[mm_in_shape[1], mm_in_shape[2]],
                            Channels=channels,
                            Kernel=[k_h, k_w],
                            inputDataType=idt.name,
                            weightDataType=wdt.name,
                            outputDataType=odt.name,
                            ActVal=actval,
                            noActivation=0,
                            name="VVAU_" + n.name,
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
                        # create and insert new VVAU node
                        new_node = helper.make_node(
                            "VVAU",
                            [mm_input, mm_weight],
                            [mm_output],
                            domain="finn.custom_op.fpgadataflow",
                            backend="fpgadataflow",
                            PE=pe,
                            Dim=[mm_in_shape[1], mm_in_shape[2]],
                            Channels=channels,
                            Kernel=[k_h, k_w],
                            inputDataType=idt.name,
                            weightDataType=wdt.name,
                            outputDataType=odt.name,
                            ActVal=0,
                            noActivation=1,
                            name="VVAU_" + n.name,
                        )
                        graph.node.insert(node_ind, new_node)
                        # remove old node
                        graph.node.remove(n)
                        graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferHWSoftmax(Transformation):
    """Infers a regular softmax node without merging the multithreshold
    and setting the softmax to perform the quantisation.
    """

    def __init__(self) -> None:
        """Infers a regular softmax node without merging the multithreshold
        and setting the softmax to perform the quantisation.
        """
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, Literal[False]]:
        """Apply the transformation."""
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            if n.op_type == "Softmax":
                input_shape = model.get_tensor_shape(n.input[0])
                if input_shape is None:
                    raise FINNUserError(
                        f"{n.name}: Softmax input shape is None. Please run InferShapes first."
                    )
                idt0 = model.get_tensor_datatype(n.input[0])
                odt0 = model.get_tensor_datatype(n.output[0])
                new_node = helper.make_node(
                    "HWSoftmax",
                    [n.input[0]],  # input tensor(s)
                    [n.output[0]],  # output tensor(s)
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    ifm_dim=input_shape,
                    input_data_type=idt0.name,
                    output_data_type=odt0.name,
                    name=n.name,
                    SIMD=1,
                    NumChannels=input_shape[-1],
                    cpp_interface="hls_vector",
                    hls_style="freerunning",
                )
                graph.node.insert(node_ind, new_node)
                graph.node.remove(n)
        return (model, graph_modified)


def skip_first_node_transpose(model: ModelWrapper, node: NodeProto) -> bool:
    """Default filter for InferShuffle: skip Transpose if it's the first node in the graph.
    This is useful for image classification networks where the first transpose converts
    NCHW to NHWC layout for data preprocessing."""
    return node != model.graph.node[0]


class InferShuffle(Transformation):
    """Find transpose layers with (optionally) reshape layers around them
    and convert them into a shuffle operator.
    """

    def __init__(
        self, _filter: Callable[[ModelWrapper, NodeProto], bool] = skip_first_node_transpose
    ) -> None:
        """Initialize instance."""
        super().__init__()
        # Register the filter function as attribute
        self._filter = _filter

    def _is_streaming_ptranspose(self, perm: list, shape: list) -> bool:
        """Check if the permutation represents a streaming InnerShuffle case.
        A streaming InnerShuffle works when the last two dimensions are swapped,
        regardless of how many outer dimensions there are.
        """
        if len(perm) < 2 or len(shape) < 2:
            return False

        # Check if last two dimensions are swapped while others stay in order
        expected_perm = [*list(range(len(perm) - 2)), len(perm) - 1, len(perm) - 2]
        return perm == expected_perm

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transformation."""
        graph = model.graph
        graph_modified = False
        for node_ind, n in enumerate(graph.node, start=1):
            if n.op_type == "Transpose":
                # Apply filter function to decide whether to convert this node
                if not self._filter(model, n):
                    continue
                to_remove = [n]

                new_in_tensor = None
                new_out_tensor = None

                perm = n.attribute[0]

                new_in_tensor = n.input[0]
                in_shape = model.get_tensor_shape(n.input[0])
                in_reshaped = in_shape

                # Detect a reshape at the input and capture it
                producer = model.find_producer(n.input[0])
                if producer is not None and producer.op_type == "Reshape":
                    new_in_tensor = producer.input[0]
                    in_shape = model.get_tensor_shape(new_in_tensor)
                    in_reshaped = model.get_tensor_shape(n.input[0])
                    to_remove.append(producer)

                new_out_tensor = n.output[0]
                if (out_shape := model.get_tensor_shape(new_out_tensor)) is None:
                    raise FINNUserError(
                        f"Could not infer shape for tensor {new_out_tensor}. "
                        f"Please run InferShapes first."
                    )
                out_reshaped = out_shape

                # Detect a reshape at the output and capture it
                consumer = model.find_consumer(n.output[0])
                if consumer is not None and consumer.op_type == "Reshape":
                    new_out_tensor = consumer.output[0]
                    out_shape = model.get_tensor_shape(n.output[0])
                    out_reshaped = model.get_tensor_shape(new_out_tensor)
                    to_remove.append(consumer)

                # Handle None shapes (shape inference might have failed)
                if in_shape is None:
                    raise FINNUserError(
                        f"Could not infer shape for tensor {new_in_tensor}. "
                        "Please run InferShapes first."
                    )
                if in_reshaped is None:
                    raise FINNUserError(
                        f"Could not infer reshaped input shape for tensor {n.input[0]}. "
                        "Please run InferShapes first."
                    )
                if out_shape is None:
                    raise FINNUserError(
                        f"Could not infer transpose output shape for tensor {n.output[0]}. "
                        "Please run InferShapes first."
                    )
                if out_reshaped is None:
                    raise FINNUserError(
                        f"Could not infer final output shape for tensor {new_out_tensor}. "
                        "Please run InferShapes first."
                    )

                idt = model.get_tensor_datatype(new_in_tensor)
                odt = model.get_tensor_datatype(new_out_tensor)

                # Some sanity checks for the transformation
                if idt != odt:
                    raise FINNUserError(
                        """
                    Input datatype and output datatype of the shuffle must be the same,
                    did something go wrong during transformation?
                    """
                    )

                if len(perm.ints) != len(in_reshaped):
                    raise FINNUserError(
                        f"""
                    Permutation list {perm.ints=} does not match the reshaped input dimension
                    {in_reshaped=}
                    """
                    )

                if len(perm.ints) != len(out_shape):
                    raise FINNUserError(
                        f"""
                    Permutation list {perm.ints=} does not match the reshaped out dimension
                    {out_reshaped=}
                    """
                    )

                simd = 1

                new_node = helper.make_node(
                    "Shuffle",
                    [new_in_tensor],
                    [new_out_tensor],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    in_shape=in_shape,
                    transpose_in_shape=in_reshaped,
                    out_shape=out_reshaped,
                    transpose_out_shape=out_shape,
                    data_type=idt.name,
                    name=f"Shuffle_{n.name}",
                    SIMD=simd,
                    NumChannels=in_reshaped[-1],
                )
                new_node.attribute.extend([perm])
                graph.node.insert(node_ind, new_node)

                for i in to_remove:
                    graph.node.remove(i)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())

        return (model, graph_modified)


def lift_to_rank1(name: str, model: ModelWrapper) -> None:
    """Lift scalar to rank-1 tensor.

    Converts scalar tensors (shape []) to rank-1 tensors with a single element (shape [1]).
    """
    shape = model.get_tensor_shape(name)
    if shape is None:
        raise FINNUserError(
            f"{name}: Tensor shape is None while lifting scalar to rank-1. "
            "Please run InferShapes first."
        )
    # Scalars have a shape of lengths zero
    if len(shape) == 0:
        # Lift shape to rank-1 tensor with single element
        model.set_tensor_shape(name, [1])
        # Check whether this tensor has an initializer
        if (tensor := model.get_initializer(name)) is not None:
            tensor = cast("np.ndarray", tensor)
            # Set new initializer tensor of shape [1]
            model.set_initializer(name, tensor.reshape(1))


class InferElementwiseBinaryOperation(Transformation):
    """Convert supported elementwise binary operations to their FINN custom operation."""

    @staticmethod
    def reject_output_dequant(model: ModelWrapper, node: NodeProto) -> bool:
        """Filter function to filter out the last elementwise Mul operation.

        Typically filters output de-quantization operations which should happen off-chip.
        """
        # The operator must be a Mul and have no successor nodes
        # False to reject, True to accept
        return not (
            node.op_type == "Mul"
            and not model.find_direct_successors(node)
            and model.get_tensor_datatype(node.output[0]) == "FLOAT32"
        )

    def __init__(self, _filter: Callable | None = None) -> None:
        """Initialize the transformation method with an optional filter function."""
        # Initialize the base class Transformation object
        super().__init__()
        # Register the filter function as attribute
        self._filter = _filter if _filter is not None else lambda *_: True

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transform to convert elementwise binary operations to FINN custom ops."""
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Skip transforming nodes rejected by the filter
            if not self._filter(model, node):
                continue
            # If a custom operation with corresponding name is implemented in
            # the module, this operator is supported for conversion
            if f"Elementwise{node.op_type}" in dir(elementwise_binary):
                in0 = node.input[0]
                in1 = node.input[1]
                # if both inputs are constant, throw an error and
                # ask user to run FoldConstants transform first
                if (
                    model.get_initializer(in0) is not None
                    and model.get_initializer(in1) is not None
                ):
                    raise FINNUserError(
                        f"{node.name}: Both inputs to {node.op_type} are constant. "
                        "Please run FoldConstants from "
                        "qonnx.transformation.fold_constants first."
                    )
                lhs_style = "input" if model.get_initializer(in0) is None else "const"
                rhs_style = "input" if model.get_initializer(in1) is None else "const"
                result = node.output[0]

                # Need to "lift" potential scalar inputs to rank-1 tensors
                lift_to_rank1(in0, model)
                lift_to_rank1(in1, model)

                in0_shape = model.get_tensor_shape(in0)
                in1_shape = model.get_tensor_shape(in1)
                out_shape = model.get_tensor_shape(result)
                if in0_shape is None:
                    raise FINNUserError(
                        f"{node.name}: "
                        f"Shape of input tensor {in0} is None in elementwise conversion."
                    )
                if in1_shape is None:
                    raise FINNUserError(
                        f"{node.name}: "
                        f"Shape of input tensor {in1} is None in elementwise conversion."
                    )
                if out_shape is None:
                    raise FINNUserError(
                        f"{node.name}: "
                        f"Shape of output tensor {result} is None in elementwise conversion."
                    )

                idt0 = model.get_tensor_datatype(in0)
                idt1 = model.get_tensor_datatype(in1)
                odt0 = model.get_tensor_datatype(result)

                # For constant inputs with FLOAT32 type, check if values are
                # actually integers and infer the smallest FINN datatype.
                if lhs_style == "const":
                    lhs_init = model.get_initializer(in0)
                    if lhs_init is None:
                        raise FINNUserError(
                            f"{node.name}: "
                            f"Expected constant initializer for lhs tensor {in0}, got None."
                        )
                    lhs_init = cast("np.ndarray", lhs_init)
                    if (
                        idt0 == DataType["FLOAT32"]
                        and (lhs_init == lhs_init.astype(np.int64)).all()
                    ):
                        # Values are integers, find smallest datatype
                        _min, _max = lhs_init.min(), lhs_init.max()
                        _mag = _max if _min >= 0 else _min if (abs(_min) > _max) else (-_max - 1)
                        idt0 = DataType.get_smallest_possible(_mag)
                        model.set_tensor_datatype(in0, idt0)

                if rhs_style == "const":
                    rhs_init = model.get_initializer(in1)
                    if rhs_init is None:
                        raise FINNUserError(
                            f"{node.name}: "
                            f"Expected constant initializer for rhs tensor {in1}, got None."
                        )
                    rhs_init = cast("np.ndarray", rhs_init)
                    if (
                        idt1 == DataType["FLOAT32"]
                        and (rhs_init == rhs_init.astype(np.int64)).all()
                    ):
                        # Values are integers, find smallest datatype
                        _min, _max = rhs_init.min(), rhs_init.max()
                        _mag = _max if _min >= 0 else _min if (abs(_min) > _max) else (-_max - 1)
                        idt1 = DataType.get_smallest_possible(_mag)
                        model.set_tensor_datatype(in1, idt1)

                # If both inputs are integers, set output to INT32 as default.
                # MinimizeAccumulatorWidth will optimize this later.
                if idt0.is_integer() and idt1.is_integer():
                    odt0 = DataType["INT32"]
                    model.set_tensor_datatype(result, odt0)

                # Determine the operation type - check for Sub->Abs pattern (AbsDiff)
                op_type = node.op_type
                nodes_to_remove = [node]
                if node.op_type == "Sub":
                    # Look for a downstream Abs node to fuse into AbsDiff
                    res_consumer = model.find_consumer(result)
                    if (res_consumer is not None) and (res_consumer.op_type == "Abs"):
                        op_type = "AbsDiff"
                        result = res_consumer.output[0]
                        out_shape = model.get_tensor_shape(result)
                        if out_shape is None:
                            raise FINNUserError(
                                f"{node.name}: Shape of AbsDiff output tensor {result} is None."
                            )
                        # Update output datatype - AbsDiff result is unsigned
                        if idt0.is_integer() and idt1.is_integer():
                            odt0 = DataType["UINT32"]
                            model.set_tensor_datatype(result, odt0)
                        nodes_to_remove.append(res_consumer)

                new_node = helper.make_node(
                    f"Elementwise{op_type}",
                    [in0, in1],
                    [result],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    lhs_shape=in0_shape,
                    rhs_shape=in1_shape,
                    out_shape=out_shape,
                    lhs_dtype=str(idt0),
                    rhs_dtype=str(idt1),
                    out_dtype=str(odt0),
                    lhs_style=lhs_style,
                    rhs_style=rhs_style,
                )
                graph.node.insert(index + 1, new_node)
                for n in nodes_to_remove:
                    graph.node.remove(n)

                # Consider the graph to be modified, triggering exhaustive
                # re-application of this transformation
                graph_modified = True
                # Exiting here triggers type and shape inference and cleanup
                # after each transformed node. This helps QONNX to behave
                # better / more consistent in certain cases...
                break
        # Re-do shape and data type annotations after potential changes to the
        # model graph
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


class InferReLUAsElementwiseMax(Transformation):
    """Converts ReLU into ElementwiseMaximum(in, 0)."""

    @staticmethod
    def reject_unsupported_dtypes(model: ModelWrapper, node: NodeProto) -> bool:
        """Filter function to filter out any operation involving any floating-point tensor."""

        def dtype_ok(tname: str) -> bool:
            """Check if a datatype is okay."""
            dt = model.get_tensor_datatype(tname)
            if dt is None:
                return False
            return (
                dt.is_integer()
                or dt.is_fixed_point()
                or dt in [DataType["FLOAT32"], DataType["FLOAT16"]]
            )

        return all(dtype_ok(tname) for tname in list(node.input) + list(node.output))

    def __init__(
        self, _filter: Callable[[ModelWrapper, NodeProto], bool] = reject_unsupported_dtypes
    ) -> None:
        """Initialize the transformation method with an optional filter function."""
        # Initialize the base class Transformation object
        super().__init__()
        # Register the filter function as attribute
        self._filter = _filter if _filter is not None else lambda *_: True

    def apply(self, model: ModelWrapper):  # noqa
        """Apply the transformation."""
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for index, node in enumerate(graph.node):
            # Skip transforming nodes rejected by the filter
            if not self._filter(model, node):
                continue
            if node.op_type == "Relu":
                inp = node.input[0]
                # add a second 0-valued input for ReLU
                new_tname = model.make_new_valueinfo_name()
                model.set_initializer(new_tname, np.asarray(0.0, dtype=np.float32))
                # comparison of fp16 and uint2 is not possible in HLS
                new_tdtype = (
                    "FLOAT16"
                    if model.get_tensor_datatype(inp).get_canonical_name() == "FLOAT16"
                    else "UINT2"
                )
                # for the constant 0 input, use a small-width datatype
                # (to avoid unnecessarily promoting output type to something larger)
                model.set_tensor_datatype(new_tname, DataType[new_tdtype])
                result = node.output[0]

                # Need to "lift" potential scalar inputs to rank-1 tensors
                lift_to_rank1(inp, model)
                lift_to_rank1(new_tname, model)

                new_node = helper.make_node(
                    "ElementwiseMax",
                    [inp, new_tname],
                    [result],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    lhs_shape=model.get_tensor_shape(inp),
                    rhs_shape=model.get_tensor_shape(new_tname),
                    out_shape=model.get_tensor_shape(result),
                    lhs_dtype=str(model.get_tensor_datatype(inp)),
                    rhs_dtype=str(model.get_tensor_datatype(new_tname)),
                    out_dtype=str(model.get_tensor_datatype(result)),
                )
                graph.node.insert(index + 1, new_node)
                graph.node.remove(node)

                # Consider the graph to be modified, triggering exhaustive
                # re-application of this transformation
                graph_modified = True
                # Exiting here triggers type and shape inference and cleanup
                # after each transformed node. This helps QONNX to behave
                # better / more consistent in certain cases...
                break
        # Re-do shape and data type annotations after potential changes to the
        # model graph
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


class InferLayerNorm(Transformation):
    """Convert LayerNorm into HW, only norming over channel dim.
    This transform is adapted from Brainsmith InferLayerNorm."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, Literal[False]]:
        """Apply the transformation."""
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "LayerNormalization":
                scale = model.get_initializer(node.input[1])
                bias = model.get_initializer(node.input[2]) if len(node.input) > 2 else None
                if scale is None or bias is None:
                    raise FINNUserError(
                        f"{node.name}: LayerNormalization scale or bias initializer is None. "
                        "Please run ExtractNormScaleBias first."
                    )
                scale = cast("np.ndarray", scale)
                bias = cast("np.ndarray", bias)
                scale_is_one = (scale == 1).all()
                bias_is_zero = bias is None or (not np.any(bias))
                if not (scale_is_one and bias_is_zero):
                    log.warning(
                        f"""{node.name}: Scale is not one or bias is not zero.
                        Can't be converted to HWCustomOp. Please run ExtractNormScaleBias first."""
                    )
                    continue
                act_in = node.input[0]
                act_out = node.output[0]
                # Get any shape info that needs reuse
                shape_in = model.get_tensor_shape(act_in)
                if shape_in is None:
                    raise FINNUserError(
                        f"{node.name}: LayerNormalization input shape is None. "
                        "Please run InferShapes first."
                    )
                # Get datatypes
                idt = model.get_tensor_datatype(act_in)
                odt = model.get_tensor_datatype(act_out)

                norm_axis = helper.get_node_attr_value(node, "axis")
                if model.get_tensor_layout(act_in) == DataLayout.NCHW:
                    act_in = nchw_to_nhwc(act_in, model, node_ind)
                    node_ind += 1
                    shape_in = model.get_tensor_shape(act_in)
                    if shape_in is None:
                        raise FINNUserError(
                            f"{node.name}: Input shape is None after NCHW->NHWC conversion."
                        )
                    # shift axis for norm appropriately
                    norm_axis = (norm_axis + 2) % 4
                ch = shape_in[-1]

                # keep track of where we need to insert the HLS Op
                # it has to be ahead of the output transform
                insert_point = node_ind
                if model.get_tensor_layout(act_out) == DataLayout.NCHW:
                    act_out = nchw_to_nhwc(act_out, model, node_ind, reverse=True)
                    node_ind += 1

                # Check if 1D, norming on channel axis
                if not (norm_axis == -1 or norm_axis == len(shape_in) - 1):
                    continue

                # create node with no parallelization first
                simd = 1
                if ch % simd != 0:
                    raise FINNUserError(
                        f"{node.name}: Requirement IFC divisible by SIMD is violated: "
                        f"Channels={ch}, SIMD={simd}."
                    )
                # create and insert nodes
                new_node = helper.make_node(
                    "LayerNorm",
                    [act_in],
                    [act_out],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    SIMD=simd,
                    ifm_dim=shape_in,
                    epsilon=helper.get_node_attr_value(node, "epsilon"),
                    inputDataType=idt.name,
                    outputDataType=odt.name,
                    name="LayerNorm_" + node.name,
                )
                graph.node.insert(insert_point, new_node)
                # remove old node
                graph.node.remove(node)

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


def elements_are_consecutive(indices: np.ndarray) -> np.bool_ | Literal[True]:
    """Are elements consecutive (max diff. 1 between all adjacent elements)."""
    if indices.size == 1:
        return True
    indices.sort()
    return np.all(np.diff(indices) == 1)


class InferCrop(Transformation):
    """Find gather layers that can be converted into a Crop layer
    and replace them with a Crop layer.
    """

    def __init__(self) -> None:
        """Find gather layers that can be converted into a Crop layer
        and replace them with a Crop layer.
        """
        super().__init__()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transformation."""
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Gather":
                # ensure that the indices input is an initializer
                if model.get_initializer(n.input[1]) is None:
                    continue

                # ensure that the axis is among the two innermost dimensions
                if (input_shape := model.get_tensor_shape(n.input[0])) is None:
                    raise FINNUserError(
                        f"{n.name}: Input tensor shape is not defined. "
                        f"Please run InferShapes first."
                    )
                if len(input_shape) <= 1:
                    raise FINNUserError(
                        f"{n.name}: Input shape needs to be at least 2D to be converted to Crop."
                    )

                max_index = len(input_shape) - 1
                axis_attr = get_by_name(n.attribute, "axis")
                if axis_attr is None:
                    raise FINNUserError(
                        f"{n.name}: Gather axis attribute is missing; cannot infer Crop operator."
                    )
                axis = axis_attr.i
                if len(input_shape) >= 3:
                    if axis not in [max_index - 1, max_index - 2]:
                        raise FINNUserError(
                            f"{n.name}: Crop conversion supports only H/W axes for (N)HWC layout. "
                            f"Got axis={axis} for input shape {input_shape}."
                        )
                else:
                    if axis != max_index - 1:
                        raise FINNUserError(
                            f"{n.name}: "
                            f"For 2D input (WC), Crop conversion supports only width axis "
                            f"{max_index - 1}, got axis={axis}."
                        )
                is_vertical = axis == max_index  # otherwise horizontal
                if is_vertical:
                    raise FINNUserError(f"{n.name}: Vertical crops are currently not supported.")

                # assume that the indices input is an int64 scalar or array
                indices = model.get_initializer(n.input[1])
                if indices is None:
                    raise FINNUserError(
                        f"{n.name}: Gather indices initializer is None, cannot infer Crop."
                    )
                indices = cast("np.ndarray", indices)
                if indices.dtype != np.int64:
                    raise FINNUserError(
                        f"{n.name}: Gather indices must have dtype int64 for Crop conversion, "
                        f"got {indices.dtype}."
                    )
                # Handle both scalar (0-d) and array cases
                # Single scalar index - always consecutive
                indices_to_check = np.array([indices.item()]) if indices.ndim == 0 else indices
                if not elements_are_consecutive(indices_to_check):
                    raise FINNUserError(
                        f"{n.name}: Gather indices for Crop conversion must be consecutive, "
                        f"got {indices_to_check}."
                    )

                idt0 = model.get_tensor_datatype(n.input[0])

                crop_north = 0
                crop_east = 0
                crop_west = 0
                crop_south = 0
                num_inp_vec = [0]

                if len(input_shape) >= 3:
                    height_ind = len(input_shape) - 3
                    width_ind = len(input_shape) - 2
                    channels_ind = len(input_shape) - 1

                    height = input_shape[height_ind]
                    width = input_shape[width_ind]
                    channels = input_shape[channels_ind]
                    # save other dimensions in numInpVectors
                    if len(input_shape) > 3:
                        num_inp_vec = list(input_shape[:height_ind])

                    crop_min = int(np.min(indices_to_check))
                    crop_max = input_shape[axis] - int(np.max(indices_to_check)) - 1

                    if axis == height_ind:
                        crop_north = crop_min
                        crop_south = crop_max
                    elif axis == width_ind:
                        crop_west = crop_min
                        crop_east = crop_max

                elif len(input_shape) == 2:
                    # if there are only two dimensions, assume
                    height = 0
                    width_ind = len(input_shape) - 2
                    channels_ind = len(input_shape) - 1
                    width = input_shape[width_ind]
                    channels = input_shape[channels_ind]

                    # axis is on width dimension
                    crop_west = int(np.min(indices_to_check))
                    crop_east = input_shape[axis] - int(np.max(indices_to_check)) - 1
                else:
                    raise FINNUserError(
                        f"{n.name}: Input shape must be at least 2D for Crop conversion, "
                        f"got {input_shape}."
                    )

                # create and insert new node
                new_node = helper.make_node(
                    "Crop",
                    [n.input[0]],  # input tensor(s)
                    [n.output[0]],  # output tensor(s)
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    DataType=idt0.name,
                    name="Crop" + n.name,
                    SIMD=1,
                    ImgDim=[height, width],
                    NumChannels=channels,
                    CropNorth=crop_north,
                    CropEast=crop_east,
                    CropWest=crop_west,
                    CropSouth=crop_south,
                    numInputVectors=num_inp_vec,
                    cpp_interface="hls_vector",
                    hls_style="freerunning",
                )
                graph.node.insert(node_ind, new_node)
                graph.node.remove(n)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)


class InferSqueeze(Transformation):
    """Converts the Squeeze operation to the corresponding FINN custom operation."""

    # Applies the transform to a whole model graph
    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transform to convert Squeeze operations to FINN custom ops."""
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for _index, node in enumerate(graph.node):
            # Handles Squeeze ONNX operations
            if node.op_type == "Squeeze":
                # Skip already converted nodes
                if node.domain == "finn.custom_op.fpgadataflow":
                    # Skip without warning
                    continue
                # Transplant this operator into our FINN domain
                node.domain = "finn.custom_op.fpgadataflow"
                # Now we can get the CustomOp wrapper instance providing easier
                # attribute access
                inst: HWCustomOp = getHWCustomOp(node)
                # Set the backend attribute to mark this an operation supported
                # to be implemented on an FPGA by FINN
                inst.set_nodeattr("backend", "fpgadataflow")
                # Ge the input and output tensor names
                inp, out = node.input[0], node.output[0]
                inp_shape = model.get_tensor_shape(inp)
                out_shape = model.get_tensor_shape(out)
                if inp_shape is None or out_shape is None:
                    raise FINNUserError(
                        f"{node.name}: Squeeze input/output shapes are undefined. "
                        "Please run InferShapes first."
                    )
                # Set input/output shape and datatype node attributes required
                # by FINN custom op
                inst.set_nodeattr("inp_dtype", str(model.get_tensor_datatype(inp)))
                inst.set_nodeattr("inp_shape", cast("list[str|int|float]", inp_shape))
                inst.set_nodeattr("out_dtype", str(model.get_tensor_datatype(out)))
                inst.set_nodeattr("out_shape", cast("list[str|int|float]", out_shape))
                if len(node.input) > 1:
                    axes = model.get_initializer(node.input[1])
                    if axes is None:
                        raise FINNUserError(
                            f"{node.name}: "
                            "Squeeze axes input must be a constant initializer, got None."
                        )
                    axes = cast("np.ndarray", axes)
                    if np.ndim(axes) == 0:
                        # Fix axes input initializer by converting from scalar (0D) to 1D array
                        axes = np.array([axes])
                        model.set_initializer(node.input[1], axes)
                        model.set_tensor_shape(node.input[1], axes.shape)
                    # Set axes attribute (used by older opsets) even if axes is provided as input
                    inst.set_nodeattr("axes", list(axes))
                # Consider the graph to be modified, triggering exhaustive
                # re-application of this transformation
                graph_modified = True
                # Exiting here triggers type and shape inference and cleanup
                # after each transformed node. This helps QONNX to behave
                # better/more consistent in certain cases...
                break
        # Re-do shape and data type annotations after potential changes to the
        # model graph
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


class InferUnsqueeze(Transformation):
    """Convert the Unsqueeze operation to the corresponding FINN custom operation."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transform to convert Unsqueeze operations to FINN custom ops."""
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False
        # Iterate all nodes in the graph keeping track of the index
        for _index, node in enumerate(graph.node):
            # Handles Squeeze ONNX operations
            if node.op_type == "Unsqueeze":
                # Skip already converted node
                if node.domain == "finn.custom_op.fpgadataflow":
                    # Skip without warning
                    continue
                # Transplant this operator into our FINN domain
                node.domain = "finn.custom_op.fpgadataflow"
                # Now we can get the CustomOp wrapper instance providing easier
                # attribute access
                inst: HWCustomOp = getHWCustomOp(node)
                # Set the backend attribute to mark this an operation supported
                # to be implemented on an FPGA by FINN
                inst.set_nodeattr("backend", "fpgadataflow")
                # Get the input and output tensor names
                inp, out = node.input[0], node.output[0]
                inp_shape = model.get_tensor_shape(inp)
                out_shape = model.get_tensor_shape(out)
                if inp_shape is None or out_shape is None:
                    raise FINNUserError(
                        f"{node.name}: Unsqueeze input/output shapes are undefined. "
                        "Please run InferShapes first."
                    )
                # Set input/output shape and datatype node attributes required
                # by FINN custom op
                inst.set_nodeattr("inp_dtype", str(model.get_tensor_datatype(inp)))
                inst.set_nodeattr("inp_shape", cast("list[str|int|float]", inp_shape))
                inst.set_nodeattr("out_dtype", str(model.get_tensor_datatype(out)))
                inst.set_nodeattr("out_shape", cast("list[str|int|float]", out_shape))
                if len(node.input) > 1:
                    axes = model.get_initializer(node.input[1])
                    if axes is None:
                        raise FINNUserError(
                            f"{node.name}: "
                            "Unsqueeze axes input must be a constant initializer, got None."
                        )
                    axes = cast("np.ndarray", axes)
                    if np.ndim(axes) == 0:
                        # Fix axes input initializer by converting from scalar (0D) to 1D array
                        axes = np.array([axes])
                        model.set_initializer(node.input[1], axes)
                        model.set_tensor_shape(node.input[1], axes.shape)
                    # Set axes attribute (used by older opsets) even if axes is provided as input
                    inst.set_nodeattr("axes", [int(x) for x in axes])  # type: ignore
                # Consider the graph to be modified, triggering exhaustive
                # re-application of this transformation
                graph_modified = True
                # Exiting here triggers type and shape inference and cleanup
                # after each transformed node. This helps QONNX to behave
                # better/more consistent in certain cases...
                break
        # Re-do shape and data type annotations after potential changes to the
        # model graph
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


class InferReshape(Transformation):
    """Converts ONNX Reshape operator to the corresponding HWCustomOp."""

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transform to convert Reshape operations hardware."""
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False

        # Transplant each Reshape operator into the custom domain
        for _index, node in enumerate(graph.node):
            if node.op_type == "Reshape":
                # Skip already converted nodes
                if node.domain == "finn.custom_op.fpgadataflow":
                    continue

                # Skip if this is not a static-shape reshape operation
                if (out_shape := model.get_initializer(node.input[1])) is None:
                    continue

                # Skip if any of the axes is set to 0 to avoid handling
                # differences between allowzero=0 and allowzero=1...
                # TODO: This could be supported by implementing simple shape
                #  inference logic here or by normalizing shapes via streamline
                if np.any(out_shape == 0):
                    continue

                # Skip if the input shape is not statically known
                if (inp_shape := model.get_tensor_shape(node.input[0])) is None:
                    continue

                # Skip if there are more than one axes set to -1
                if list(out_shape).count(-1) > 1:
                    continue

                # The shape is allowed to contain at most one -1, in which case
                # the value is inferred from the size of the tensor and the
                # remaining dimensions
                if list(out_shape).count(-1) == 1:
                    # The following expression assumes target to be a list which
                    # allows item assignment (might be a tuple depending on the
                    # origin of target)
                    out_shape = list(out_shape)
                    # Replace the inferred dimension by the number of elements
                    # missing from the target dimension compared to the input
                    # dimension
                    out_shape[out_shape.index(-1)] = int(
                        np.prod(inp_shape) / np.abs(np.prod(out_shape))
                    )

                # Transplant from the standard ONNX domain into the FINN domain
                node.domain = "finn.custom_op.fpgadataflow"
                # Now we can get the CustomOp wrapper instance providing easier
                # attribute access
                reshape: HWCustomOp = getHWCustomOp(node)
                # Set the backend attribute to mark this an operation supported
                # to be implemented on an FPGA by FINN
                reshape.set_nodeattr("backend", "fpgadataflow")

                # Remove the old allowzero attribute which is not handled by the
                # hardware operator
                remove_by_name(node.attribute, "allowzero")

                # Cast to python-int to match QONNX's type checking
                # This might fail, since technically, out_shape can be of type "Any"
                out_shape = [int(x) for x in out_shape]  # type: ignore

                # The reshape hardware operator statically annotates both, the
                # input and output shape, as node attributes
                reshape.set_nodeattr("inp_shape", list(inp_shape))
                reshape.set_nodeattr("out_shape", list(out_shape))

                # The input and output type of the reshaped tensor is the same
                # and needs to be statically annotated as an attribute
                reshape.set_nodeattr("dtype", model.get_tensor_datatype(node.input[0]).name)

                # Finally, there is no second shape input to the hardware
                # operator, thus delete this from the node input list
                node.input.pop(1)

        # Re-do shape and data type annotations after potential changes to the
        # model graph
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Return the transformed model and indicate whether the graph actually
        # has been transformed
        return model, graph_modified


class InferReduce(Transformation):
    """Converts ONNX ReduceSum/ReduceMin/ReduceMax/ReduceProd operator to
    the corresponding FINN+ Reduce hardware operator."""

    def _normalize_axes(self, axes: np.ndarray, ndim: int) -> tuple[int, ...]:
        """Convert axis spec to sorted tuple of unique positive axes."""
        out = []
        for a in axes:
            if a < 0:
                a += ndim
            if not (0 <= a < ndim):
                raise FINNUserError(f"Axis {a} out of range for ndim={ndim}")
            out.append(a)

        if len(set(out)) != len(out):
            raise FINNUserError(f"Duplicate axes in {axes}")

        return tuple(sorted(out))

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply the transform to convert Reduce operations hardware."""
        # Get the model graph out of the model wrapper object
        graph = model.graph
        # Keep track of whether the graph has been modified
        graph_modified = False

        for index, node in enumerate(graph.node):
            if (
                node.op_type == "ReduceSum"
                or node.op_type == "ReduceMin"
                or node.op_type == "ReduceMax"
                or node.op_type == "ReduceProd"
            ):
                # Skip if this is not a static-shape reduce operation
                if (axes := model.get_initializer(node.input[1])) is None:
                    continue

                if (inputs := model.get_tensor_shape(node.input[0])) is None:
                    continue

                # If input is integer, set output to INT32 as default.
                # MinimizeAccumulatorWidth will optimize this later.
                idt0 = model.get_tensor_datatype(node.input[0])
                odt0 = idt0
                model.set_tensor_datatype(node.output[0], idt0)
                if idt0.is_integer():
                    odt0 = DataType["INT32"]
                    model.set_tensor_datatype(node.output[0], odt0)

                # Get keepdims
                keepdims: int = helper.get_node_attr_value(node, "keepdims")

                # Operation to perform
                op = node.op_type[6:].lower()

                reduced_axes = self._normalize_axes(cast("np.ndarray", axes), len(inputs))

                # Test that the reduction axis are continous and at the end of the input dimensions
                channelreduction = False
                old_reduced_axes = reduced_axes
                if reduced_axes[-1] == len(inputs) - 1:
                    channelreduction = True
                    reduced_axes = reduced_axes[:-1]
                for i, raxis in enumerate(reversed(reduced_axes)):
                    if raxis != len(inputs) - 2 - i:
                        raise FINNUserError(
                            f"{node.name}: Reduction axes {old_reduced_axes} are not continuous."
                        )
                reduced_axes = old_reduced_axes

                shape = model.get_tensor_shape(node.input[0])
                if shape is None:
                    raise FINNUserError(
                        f"{node.name}: Input tensor shape is not defined. "
                        "Please run InferShapes first."
                    )

                # create and insert new node
                new_node = helper.make_node(
                    "Reduce",
                    [node.input[0]],  # input tensor
                    [node.output[0]],  # output tensor
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    name="Reduce" + node.name,
                    keepdims=keepdims,
                    op=op if op != "prod" else "product",
                    index_start_axis=reduced_axes[0],
                    index_stop_axis=reduced_axes[-1] if channelreduction else len(inputs) - 2,
                    PE=1,
                    input_shape=inputs,
                    InputDataType=str(idt0),
                    OutputDataType=str(odt0),
                    cpp_interface="hls_vector",
                    depthwise=int(channelreduction),
                )
                graph.node.insert(index, new_node)
                graph.node.remove(node)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

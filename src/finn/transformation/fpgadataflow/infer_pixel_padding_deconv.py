"""Infer pixel-padding lowering for ConvTranspose nodes."""

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.util.basic import auto_pad_to_explicit_padding, get_by_name
from typing import TYPE_CHECKING, Any, cast

from finn.util.exception import FINNInternalError, FINNUserError
from finn.util.logging import log

if TYPE_CHECKING:
    import numpy.typing as npt


class InferPixelPaddingDeconv(Transformation):
    """Lowering and conversion of ConvTranspose (NCHW) nodes to
    InputDilation + Im2Col + MatMul (NHWC) surrounded by Transpose nodes
    note: this transformation produces a mix of hw layers and non hw layers
    to implement this on an FPGA the Im2Col and MatMul nodes need to be converted to hw layers
    after applying this transformation and the resulting transpose nodes need to be streamlined.
    See deconv test case under tests/fpgadataflow for an example.
    """

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Apply ConvTranspose lowering into pixel padding and matmul."""
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "ConvTranspose":
                # conversion currently only supported for group=1
                group_attr = get_by_name(n.attribute, "group")
                if group_attr is None:
                    raise FINNInternalError(f"{n.name} is missing the group attribute.")
                group = group_attr.i
                if group != 1:
                    log.warning(
                        f"{n.name} : Only group=1 is currently supported.\
                            Can't infer input-dilation deconv lowering."
                    )
                    continue
                deconv_input = n.input[0]
                deconv_output = n.output[0]
                idt = model.get_tensor_datatype(deconv_input)
                odt = model.get_tensor_datatype(deconv_output)
                kernel_shape_attr = get_by_name(n.attribute, "kernel_shape")
                strides_attr = get_by_name(n.attribute, "strides")
                if kernel_shape_attr is None or strides_attr is None:
                    raise FINNInternalError(
                        f"{n.name} is missing the kernel_shape or strides attribute."
                    )
                k_h = kernel_shape_attr.ints[0]
                k_w = kernel_shape_attr.ints[1]
                stride_h = strides_attr.ints[0]
                stride_w = strides_attr.ints[1]
                weight_name = n.input[1]
                w_conv = model.get_initializer(weight_name)
                if w_conv is None:
                    raise FINNInternalError(f"Tensor {weight_name} has no initializer.")
                w_conv = cast("npt.NDArray[Any]", w_conv)
                in_shape = model.get_tensor_shape(n.input[0])
                out_shape = model.get_tensor_shape(n.output[0])
                if in_shape is None or out_shape is None:
                    raise FINNInternalError(
                        f"Could not determine shape of {n.input[0]} or {n.output[0]}."
                    )
                ifm_ch = in_shape[1]  # assume NCHW
                ofm_ch = out_shape[1]  # assume NCHW
                ifm_dim_h = in_shape[2]  # assume NCHW
                ifm_dim_w = in_shape[3]
                ofm_dim_h = out_shape[2]  # assume NCHW
                ofm_dim_w = out_shape[3]
                dilation_attr = get_by_name(n.attribute, "dilations")
                dilation = dilation_attr.ints if dilation_attr is not None else [1, 1]
                # handle both auto_pad and explicit padding
                auto_pad = get_by_name(n.attribute, "auto_pad")
                if auto_pad is not None:
                    # find equivalent specified padding
                    auto_pad_str = auto_pad.s.decode("utf-8")
                    if auto_pad_str == "NOTSET":
                        # use specified padding
                        pads_attr = get_by_name(n.attribute, "pads")
                        if pads_attr is None:
                            raise FINNInternalError(f"{n.name} is missing the pads attribute.")
                        pad = pads_attr.ints
                    else:
                        pad = auto_pad_to_explicit_padding(
                            auto_pad_str,
                            ifm_dim_h,
                            ifm_dim_w,
                            k_h,
                            k_w,
                            stride_h,
                            stride_w,
                            len(in_shape) - 2,
                        )
                else:
                    # use specified padding
                    pads_attr = get_by_name(n.attribute, "pads")
                    if pads_attr is None:
                        raise FINNInternalError(f"{n.name} is missing the pads attribute.")
                    pad = pads_attr.ints

                # If len(pad) == 2, assume no padding for other dimension
                if len(pad) == 2 and not (  # only one dimension should be padded
                    ifm_dim_h == 1 or ifm_dim_w == 1
                ):
                    raise FINNUserError("Padding is assumed to be 1D, image is 2D")
                # reuse ConvTranspose weights for new matmul weights
                # conv weights are [IFM][OFM][k][k]
                # We need to rotate the weights and make them [OFM][IFM][k][k]
                # for pixel padding deconv to remain mathematically equivalent
                # and then convert to [OFM][k][k][IFM] (to remain compatible
                # with finn-hlslib and how it does im2col/sliding window)
                w_conv = np.rot90(w_conv, 2, (2, 3))
                w_conv = np.moveaxis(w_conv, 0, 1)
                w_matmul = w_conv.transpose(0, 2, 3, 1)  # w_conv = [OFM, IFM, k_H, k_W]
                # reshape into [OFM][k*k*IFM] matrix
                w_matmul = w_matmul.reshape(ofm_ch, ifm_ch * k_h * k_w)
                # transpose to get ONNX-compatible [k*k*IFM][OFM] matrix
                w_matmul = w_matmul.T
                model.set_initializer(weight_name, w_matmul)

                # Compute intermediate parameters
                padded_odim_h = ifm_dim_h + (ifm_dim_h - 1) * (stride_h - 1)
                padded_odim_w = ifm_dim_w + (ifm_dim_w - 1) * (stride_w - 1)
                conv_padding = [dilation[0] * (k_h - 1) - pad[0]] * 4

                # create new intermediate values
                inp_trans_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    (1, ifm_dim_h, ifm_dim_w, ifm_ch),  # NHWC
                )
                dilation_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    (1, padded_odim_h, padded_odim_w, ifm_ch),  # NHWC
                )
                graph.value_info.append(inp_trans_out)
                graph.value_info.append(dilation_out)
                inp_trans_out = inp_trans_out.name
                dilation_out = dilation_out.name
                model.set_tensor_datatype(inp_trans_out, idt)
                model.set_tensor_datatype(dilation_out, idt)

                need_im2col = True
                padding = None
                if all(p == 0 for p in conv_padding):
                    padding = 0

                # k_h=k_w==1: pointwise convolution, thus no im2col needed
                if k_h == 1 and k_w == 1 and padding == 0 and stride_h == 1 and stride_w == 1:
                    need_im2col = False

                im2col_out_name = ""
                if need_im2col:
                    im2col_out = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        (1, ofm_dim_h, ofm_dim_w, ifm_ch * k_h * k_w),
                    )
                    graph.value_info.append(im2col_out)
                    im2col_out_name = im2col_out.name
                    model.set_tensor_datatype(im2col_out_name, idt)

                matmul_out = helper.make_tensor_value_info(
                    model.make_new_valueinfo_name(),
                    TensorProto.FLOAT,
                    (1, ofm_dim_h, ofm_dim_w, ofm_ch),
                )
                graph.value_info.append(matmul_out)
                matmul_out = matmul_out.name
                model.set_tensor_datatype(matmul_out, odt)

                # create new nodes

                # NCHW -> NHWC
                inp_trans_node = helper.make_node(
                    "Transpose", [deconv_input], [inp_trans_out], perm=[0, 2, 3, 1]
                )
                # Input dilation (interior zero-padding)
                input_dilation_node = helper.make_node(
                    "InputDilation",
                    [inp_trans_out],
                    [dilation_out],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    ImgDim=(ifm_dim_h, ifm_dim_w),
                    Stride=[stride_h, stride_w],
                    NumChannels=ifm_ch,
                    inputDataType=str(idt.name),
                    numInputVectors=1,
                    SIMD=1,
                )
                # lower input tensor
                matmul_input = dilation_out
                im2col_node = None
                if need_im2col:
                    matmul_input = im2col_out_name
                    im2col_node = helper.make_node(
                        "Im2Col",
                        [dilation_out],
                        [im2col_out_name],
                        domain="qonnx.custom_op.general",
                        stride=[1, 1],
                        kernel_size=[k_h, k_w],
                        pad_amount=conv_padding,
                        input_shape=f"(1,{padded_odim_h},{padded_odim_w},{ifm_ch})",
                        depthwise=False,
                        dilations=dilation,
                    )

                # do matmul
                matmul_node = helper.make_node("MatMul", [matmul_input, weight_name], [matmul_out])
                # NHWC -> NCHW
                out_trans_node = helper.make_node(
                    "Transpose", [matmul_out], [deconv_output], perm=[0, 3, 1, 2]
                )
                # insert nodes where the conv is to preserve topological ordering
                graph.node.insert(node_ind, inp_trans_node)
                if need_im2col:
                    if im2col_node is None:
                        raise FINNInternalError("im2col_node was unexpectedly not created.")
                    graph.node.insert(node_ind + 1, input_dilation_node)
                    graph.node.insert(node_ind + 2, im2col_node)
                    graph.node.insert(node_ind + 3, matmul_node)
                    graph.node.insert(node_ind + 4, out_trans_node)
                else:
                    graph.node.insert(node_ind + 1, input_dilation_node)
                    graph.node.insert(node_ind + 2, matmul_node)
                    graph.node.insert(node_ind + 3, out_trans_node)
                # remove old nodes
                graph.node.remove(n)

        return (model, graph_modified)

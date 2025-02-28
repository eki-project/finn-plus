import json
import numpy as np
import os
import shutil
import torch
import copy
from brevitas.export import export_qonnx
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import (
    GiveRandomTensorNames,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir
from util import summarize_table, summarize_section, power_xml_to_dict, prepare_inputs, delete_dir_contents
from finn.util.test import get_trained_network_and_ishape
from finn.util.basic import alveo_default_platform



from bench_base import bench

def generate_random_threshold_values(
    data_type, num_input_channels, num_steps, narrow=False, per_tensor=False
):
    if per_tensor:
        num_input_channels = 1
    if narrow:
        num_steps -= 1

    return np.random.randint(
        data_type.min(),
        data_type.max() + 1,
        (num_input_channels, num_steps),
    ).astype(np.float32)


def sort_thresholds_increasing(thresholds):
    return np.sort(thresholds, axis=1)

def make_conv_building_block(ifm_dim, ch, kernel_size, simd, pe, parallel_window=0):
    # hardcoded parameters
    idt = DataType["UINT4"]
    wdt = DataType["UINT4"]
    odt = DataType["UINT4"]
    tdt = DataType["UINT32"]
    stride = 1
    in_ch = out_ch = ch  # input channel = output channel for stacking
    # pad so that input dim = output dim for stacking (only supports odd kernel_size for now)
    pad = int(np.floor(kernel_size / 2))

    total_pad = 2 * pad
    out_feature_dim = compute_conv_output_dim(ifm_dim, kernel_size, stride, total_pad)
    weights_shape = [in_ch * kernel_size * kernel_size, out_ch]
    thresholds_shape = [1, odt.get_num_possible_values() - 1]
    input_shape = [1, ifm_dim, ifm_dim, in_ch]
    padding_out_shape = [1, ifm_dim + total_pad, ifm_dim + total_pad, in_ch]
    inpgen_out_shape = [1, out_feature_dim, out_feature_dim, in_ch * kernel_size * kernel_size]
    output_shape = [1, out_feature_dim, out_feature_dim, out_ch]

    assert input_shape == output_shape, "ERROR: Conv layer dimensions not stackable"

    padding_config = {}
    padding_config["domain"] = "finn.custom_op.fpgadataflow.rtl"
    padding_config["backend"] = "fpgadataflow"
    padding_config["ImgDim"] = [ifm_dim, ifm_dim]
    padding_config["NumChannels"] = in_ch
    padding_config["SIMD"] = simd
    padding_config["Padding"] = [pad, pad, pad, pad]
    padding_config["inputDataType"] = idt.name

    inpgen_config = {}
    inpgen_config["domain"] = "finn.custom_op.fpgadataflow.rtl"
    inpgen_config["backend"] = "fpgadataflow"
    inpgen_config["ConvKernelDim"] = [kernel_size, kernel_size]
    inpgen_config["IFMChannels"] = in_ch
    inpgen_config["IFMDim"] = [ifm_dim + total_pad, ifm_dim + total_pad]
    inpgen_config["OFMDim"] = [ifm_dim, ifm_dim]
    inpgen_config["inputDataType"] = idt.name
    inpgen_config["outputDataType"] = idt.name
    inpgen_config["SIMD"] = simd
    inpgen_config["parallel_window"] = parallel_window
    inpgen_config["Stride"] = [stride, stride]
    inpgen_config["Dilation"] = [1, 1]

    mvau_config = {}
    mvau_config["domain"] = "finn.custom_op.fpgadataflow.hls"
    mvau_config["backend"] = "fpgadataflow"
    mvau_config["numInputVectors"] = [1, ifm_dim, ifm_dim]
    mvau_config["MW"] = in_ch * kernel_size * kernel_size
    mvau_config["MH"] = in_ch
    mvau_config["SIMD"] = simd if parallel_window == 0 else simd * kernel_size * kernel_size
    mvau_config["PE"] = pe
    mvau_config["resType"] = "lut"
    mvau_config["mem_mode"] = "internal_embedded"  # internal_decoupled
    mvau_config["inputDataType"] = idt.name
    mvau_config["weightDataType"] = wdt.name
    mvau_config["outputDataType"] = odt.name

    top_in = helper.make_tensor_value_info("top_in", TensorProto.FLOAT, input_shape)
    top_out = helper.make_tensor_value_info("top_out", TensorProto.FLOAT, output_shape)
    value_info = [
        helper.make_tensor_value_info("weights", TensorProto.FLOAT, weights_shape),
        helper.make_tensor_value_info("thresholds", TensorProto.FLOAT, thresholds_shape),
        helper.make_tensor_value_info("padding_out", TensorProto.FLOAT, padding_out_shape),
        helper.make_tensor_value_info("inpgen_out", TensorProto.FLOAT, inpgen_out_shape),
    ]

    modelproto = qonnx_make_model(
        helper.make_graph(
            name="building_block",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info,
            nodes=[
                helper.make_node("FMPadding_rtl", ["top_in"], ["padding_out"], **padding_config),
                helper.make_node(
                    "ConvolutionInputGenerator_rtl",
                    ["padding_out"],
                    ["inpgen_out"],
                    **inpgen_config,
                ),
                helper.make_node(
                    "MVAU_hls", ["inpgen_out", "weights", "thresholds"], ["top_out"], **mvau_config
                ),
            ],
        )
    )

    model = ModelWrapper(modelproto)
    model.set_tensor_datatype("top_in", idt)
    model.set_tensor_layout("top_in", ["N", "H", "W", "C"])
    model.set_tensor_datatype("top_out", odt)
    model.set_tensor_datatype("weights", wdt)
    model.set_tensor_datatype("thresholds", tdt)

    weights = gen_finn_dt_tensor(wdt, weights_shape)
    # TODO: thresholds are all the same
    thresholds = generate_random_threshold_values(
        tdt, out_ch, odt.get_num_possible_values() - 1, False, True
    )
    thresholds = sort_thresholds_increasing(thresholds)

    model.set_initializer("weights", weights)
    model.set_initializer("thresholds", thresholds)

    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    return model


def combine_blocks(lb, rb, ifm_dim, ch, pe):
    # assumes left branch (lb) and right branch (rb) each have a single (dynamic) input/output with the same shape
    # to avoid mix-ups, start by giving all tensors random names
    lb = lb.transform(GiveRandomTensorNames())
    rb = rb.transform(GiveRandomTensorNames())
    # erase all node names to avoid conflict
    for n in lb.graph.node:
        n.name = ""
    for n in rb.graph.node:
        n.name = ""

    lb_input = lb.graph.input[0]
    lb_output = lb.graph.output[0]
    rb_input = rb.graph.input[0]
    rb_output = rb.graph.output[0]

    top_in = helper.make_tensor_value_info("top_in", TensorProto.FLOAT, [1, ifm_dim, ifm_dim, ch])
    top_out = helper.make_tensor_value_info("top_out", TensorProto.FLOAT, [1, ifm_dim, ifm_dim, ch])

    dup_config = {}
    dup_config["domain"] = "finn.custom_op.fpgadataflow.hls"
    dup_config["backend"] = "fpgadataflow"
    dup_config["numInputVectors"] = [1, ifm_dim, ifm_dim]
    dup_config["NumChannels"] = ch
    dup_config["PE"] = pe
    dup_config["NumOutputStreams"] = 2
    dup_config["inputDataType"] = lb.get_tensor_datatype(lb_input.name).name

    add_config = {}
    add_config["domain"] = "finn.custom_op.fpgadataflow.hls"
    add_config["backend"] = "fpgadataflow"
    add_config["numInputVectors"] = [1, ifm_dim, ifm_dim]
    add_config["NumChannels"] = ch
    add_config["PE"] = pe
    add_config["inputDataType"] = lb.get_tensor_datatype(lb_output.name).name

    nodes_lb = [node for node in lb.graph.node]
    nodes_rb = [node for node in rb.graph.node]
    nodes_new = (
        nodes_lb
        + nodes_rb
        + [
            helper.make_node(
                "DuplicateStreams_hls", ["top_in"], [lb_input.name, rb_input.name], **dup_config
            ),
            helper.make_node(
                "AddStreams_hls", [lb_output.name, rb_output.name], ["top_out"], **add_config
            ),
        ]
    )

    value_info_lb = [x for x in lb.graph.value_info]
    value_info_rb = [x for x in rb.graph.value_info]
    value_info_new = value_info_lb + value_info_rb + [lb_input, lb_output, rb_input, rb_output]

    initializer_lb = [x for x in lb.graph.initializer]
    initializer_rb = [x for x in rb.graph.initializer]
    initializer_new = initializer_lb + initializer_rb
    modelproto = qonnx_make_model(
        helper.make_graph(
            name="branching_model",
            inputs=[top_in],
            outputs=[top_out],
            value_info=value_info_new,
            nodes=nodes_new,
        )
    )

    model = ModelWrapper(modelproto)
    model.set_tensor_datatype("top_in", lb.get_tensor_datatype(lb_input.name))
    model.set_tensor_layout("top_in", lb.get_tensor_layout(lb_input.name))
    for i in initializer_new:
        model.graph.initializer.append(i)

    # tidy-up
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(GiveReadableTensorNames())
    return model

class bench_synthetic_nonlinear(bench):
    def step_export_onnx(self, onnx_export_path):
        np.random.seed(0)
        tmp_output_dir = make_build_dir("test_fifosizing")

        #TODO: allow manual folding/fifo config as input

        #TODO: is a scenario possible where reducing depth of a single FIFO at a time is not sufficient for testing tightness?
        #      e.g. reducing > 1 FIFOs simultaneously does not cause a throughput drop while reducing a single FIFO does?

        #TODO: how to determine rtlsim_n automatically?

        # conv parameters
        dim = self.params["dim"]
        kernel_size = self.params["kernel_size"]
        ch = self.params["ch"]
        simd = self.params["simd"]
        pe = self.params["pe"]
        parallel_window = self.params["parallel_window"]

        lb = None
        for i in range(self.params["lb_num_layers"]):
            new_block = make_conv_building_block(
                dim, ch, kernel_size=kernel_size, simd=simd, pe=pe, parallel_window=parallel_window
            )
            lb = new_block if lb is None else lb.transform(MergeONNXModels(new_block))
        lb.save(tmp_output_dir + "/lb.onnx")

        rb = None
        for i in range(self.params["rb_num_layers"]):
            new_block = make_conv_building_block(
                dim, ch, kernel_size=kernel_size, simd=simd, pe=pe, parallel_window=parallel_window
            )
            rb = new_block if rb is None else rb.transform(MergeONNXModels(new_block))
        rb.save(tmp_output_dir + "/rb.onnx")

        model = combine_blocks(lb, rb, dim, ch, pe=4)
        model.save(onnx_export_path)

    def step_build_setup(self):
        # create build config for synthetic test models

        cfg = build_cfg.DataflowBuildConfig(
            # manual folding
            target_fps=None,
        )

        return cfg

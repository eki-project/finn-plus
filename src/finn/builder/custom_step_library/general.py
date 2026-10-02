import numpy as np
import qonnx.core.data_layout as dl
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import RemoveUnusedTensors
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK

from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.util.logging import log


# Insert Div node to divide input by 255
# This is used when raw uint8 pixel data is divided by 255 prior to training (e.g., GTSRB example).
# We want to reflect this in the model, so inference can be performed directly on raw uint8 data.
def add_preproc_divide_by_255(model: ModelWrapper, cfg: DataflowBuildConfig):
    in_name = model.graph.input[0].name
    new_in_name = model.make_new_valueinfo_name()
    new_param_name = model.make_new_valueinfo_name()
    div_param = np.asarray(255.0, dtype=np.float32)
    new_div = oh.make_node(
        "Div",
        [in_name, new_param_name],
        [new_in_name],
        name="PreprocDiv",
    )
    model.set_initializer(new_param_name, div_param)
    model.graph.node.insert(0, new_div)
    model.graph.node[1].input[0] = new_in_name
    # set input dtype to uint8
    model.set_tensor_datatype(in_name, DataType["UINT8"])

    return model


# Insert TopK node to get predicted Top-1 class
def add_postproc_top1(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(InsertTopK(k=1))
    return model


# Remove a Reshape or Flatten that directly consumes the graph input and give the graph the
# flattened [N, C] input instead.
# Reshaping the input is free in software (the driver reshapes its input to the accelerator's
# input shape anyway), while the hardware Reshape folds along the last axis of its input and
# so limits a [N, 1, H, W] image to W elements per cycle. Without it the first layer sees all
# H*W elements as channels and can be parallelized up to one frame per cycle (e.g. the MLPs of
# the BNN-PYNQ examples). To be run after streamlining and before the conversion to HW layers.
def flatten_global_input(model: ModelWrapper, cfg: DataflowBuildConfig):
    in_name = model.graph.input[0].name
    consumers = model.find_consumers(in_name)
    if len(consumers) != 1 or consumers[0].op_type not in ["Reshape", "Flatten"]:
        log.warning("flatten_global_input: the graph input is not consumed by a single Reshape")
        return model
    node = consumers[0]
    out_name = node.output[0]
    in_shape = model.get_tensor_shape(in_name)
    out_shape = model.get_tensor_shape(out_name)
    if (
        out_shape is None
        or len(out_shape) != 2
        or out_shape[0] != in_shape[0]
        or out_name in [o.name for o in model.graph.output]
    ):
        log.warning("flatten_global_input: the input Reshape does not flatten to [N, C]")
        return model
    idt = model.get_tensor_datatype(in_name)
    for consumer in model.find_consumers(out_name):
        for i, name in enumerate(consumer.input):
            if name == out_name:
                consumer.input[i] = in_name
    model.graph.node.remove(node)
    model.set_tensor_shape(in_name, out_shape)
    model.set_tensor_datatype(in_name, idt)
    model.set_tensor_layout(in_name, dl.NC)
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(InferShapes())
    return model

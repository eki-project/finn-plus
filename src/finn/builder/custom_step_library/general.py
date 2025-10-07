import numpy as np
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.insert_topk import InsertTopK

from finn.builder.build_dataflow_config import DataflowBuildConfig


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

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors
from qonnx.transformation.general import GiveUniqueNodeNames

import finn.transformation.streamline.absorb as absorb
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.fpgadataflow.convert_to_hw.channelwise_linear import (
    InferChannelwiseLinearLayer,
)
from finn.transformation.fpgadataflow.convert_to_hw.label_select import InferLabelSelectLayer


def step_pre_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(Change3DTo4DTensors())
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    return model


def step_convert_final_layers(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(InferChannelwiseLinearLayer())
    model = model.transform(InferLabelSelectLayer())
    model = model.transform(GiveUniqueNodeNames())
    return model

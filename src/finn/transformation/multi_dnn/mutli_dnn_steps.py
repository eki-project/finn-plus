import json
from qonnx.core.modelwrapper import ModelWrapper

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.multi_dnn.multi_dnn_wrapper_transformations import (
    CollapseModels,
    CombineInputsChannelwise,
    CombineOutputsChannelwise,
    MultiDNNWrapperExposeIO,
)

from finn.transformation.multi_dnn.mutli_dnn_selectable import ExtractSelectableWeights


def _resolve_multi_dnn_mode(cfg: DataflowBuildConfig):
    with open(cfg.multi_dnn_config_path, "r") as fp_json:
        multi_dnn_config = json.load(fp_json)
    gen = multi_dnn_config.get("Generation")
    return gen["mode"], gen.get("kwargs", None)


def step_apply_multi_dnn(model: ModelWrapper, cfg: DataflowBuildConfig):
    mode, kwargs = _resolve_multi_dnn_mode(cfg)
    if mode == "Parallel":
        if kwargs is not None:
            combine_inputs_channelwise = kwargs.get("combine_inputs_channelwise", None)
            combine_outputs_channelwise = kwargs.get("combine_outputs_channelwise", None)
        model = model.transform(MultiDNNWrapperExposeIO())
        model = model.transform(CombineInputsChannelwise()) if combine_inputs_channelwise else model
        model = (
            model.transform(CombineOutputsChannelwise()) if combine_outputs_channelwise else model
        )
    elif mode == "SelectableWeights":
        model = model.transform(ExtractSelectableWeights(kwargs.get("models")))
        model = model.transform(MultiDNNWrapperExposeIO())
    else:
        raise Exception("This Mode is not implemented")

    return model


def step_collapse_multi_dnn(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(CollapseModels())
    model = model.transform(to_hw.InferSplitLayer())
    model = model.transform(to_hw.InferConcatLayer())
    model = model.transform(SpecializeLayers(cfg._resolve_fpga_part()))  # For Concat and Split
    return model

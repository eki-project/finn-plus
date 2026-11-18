"""Custom build steps for 1D convolutional model processing."""

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors
from qonnx.transformation.general import GiveUniqueNodeNames

import finn.transformation.streamline.absorb as absorb
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.fpgadataflow.convert_to_hw.elementwise_binary_operation import (
    InferElementwiseBinaryOperation,
)
from finn.transformation.fpgadataflow.convert_to_hw.label_select import InferLabelSelectLayer


def step_pre_streamline(
    model: ModelWrapper,
    cfg: DataflowBuildConfig,  # noqa: ARG001
) -> ModelWrapper:
    """Prepare a 1D convolutional model for streamlining.

    Converts 3D tensors to 4D and absorbs scalar mul/add operations into TopK.
    """
    model = model.transform(Change3DTo4DTensors())
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    return model


def step_convert_final_layers(
    model: ModelWrapper,
    cfg: DataflowBuildConfig,  # noqa: ARG001
) -> ModelWrapper:
    """Convert the final elementwise-binary and label-select layers to hardware operations."""
    model = model.transform(InferElementwiseBinaryOperation())
    model = model.transform(InferLabelSelectLayer())
    model = model.transform(GiveUniqueNodeNames())
    return model

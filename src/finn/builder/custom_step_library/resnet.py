# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2022-2024, Advanced Micro Devices, Inc.
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

"""Custom build steps for ResNet model processing.

This module provides specialized transformation steps for converting quantized
ResNet models from QONNX format through various stages of optimization and
hardware conversion.
"""

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.composed import ComposedTransformation
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    ConvertDivToMul,
    ConvertSubToAdd,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
    SortGraph,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.remove import RemoveIdentityOps

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.fpgadataflow.replicate_stream import InferReplicateStream
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.streamline.absorb import (
    Absorb1BitMulIntoConv,
    Absorb1BitMulIntoMatMul,
    AbsorbAddIntoMultiThreshold,
    AbsorbConsecutiveTransposes,
    AbsorbMulIntoMultiThreshold,
    AbsorbScalarMulAddIntoTopK,
    AbsorbSignBiasIntoMultiThreshold,
    AbsorbTransposeIntoMultiThreshold,
    FactorOutMulSignMagnitude,
)
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedAdd,
    CollapseRepeatedMul,
)
from finn.transformation.streamline.remove import RemoveIdentityReshape, RemoveIdentityTranspose

# just for not linear
# just for not linear
from finn.transformation.streamline.reorder import (
    MoveAddPastConv,
    MoveAddPastMul,
    MoveLinearPastEltwiseAdd,
    MoveLinearPastFork,
    MoveMaxPoolPastMultiThreshold,
    MoveMulPastAdd,
    MoveScalarAddPastMatMul,
    MoveScalarLinearPastInvariants,
    MoveScalarMulPastConv,
    MoveScalarMulPastMatMul,
    MoveTransposePastEltwise,
    MoveTransposePastFork,
    MoveTransposePastJoinAdd,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.sign_to_thres import ConvertSignToThres
from finn.transformation.streamline.streamline_plus import StreamlinePlus as Streamline


def step_resnet_tidy(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:  # noqa: ARG001
    """Tidy up ResNet models."""
    model = model.transform(
        ComposedTransformation(
            [
                # Adds shape and datatype annotations to all tensors in this graph
                InferDataTypes(),
                InferShapes(),
                # Cleanup the graph by removing redundant, unnecessary and constant
                # nodes and tensors and give unique names to everything remaining
                GiveUniqueNodeNames(),
                GiveReadableTensorNames(),
                RemoveUnusedTensors(),
                GiveUniqueParameterTensors(),
                FoldConstants(),
                # Remove unnecessary shape and layout transformations
                RemoveIdentityReshape(),
                RemoveIdentityTranspose(),
                # Redo shape and datatype annotations after removing nodes and
                # tensors
                InferShapes(),
                InferDataTypes(),
            ]
        )
    )
    return model


def step_resnet_streamline(
    model: ModelWrapper, cfg: DataflowBuildConfig
) -> ModelWrapper:  # noqa: ARG001
    """Streamline ResNet models."""
    transform = ComposedTransformation(
        [
            MoveMulPastAdd(),
            AbsorbSignBiasIntoMultiThreshold(),
        ]
    )
    model = model.transform(transform)
    model = model.transform(Streamline())
    transform2 = ComposedTransformation(
        [LowerConvsToMatMul(), AbsorbAddIntoMultiThreshold(), AbsorbTransposeIntoMultiThreshold()]
    )
    model = model.transform(transform2)
    model = model.transform(Streamline())
    # model = model.transform(InsertTopK())
    # model = model.transform(AbsorbScalarMulAddIntoTopK())

    return model


def step_resnet_convert_to_hw(
    model: ModelWrapper, cfg: DataflowBuildConfig
) -> ModelWrapper:  # noqa: ARG001
    """Convert ResNet models to hardware-specific operations."""
    # Convert Squeeze and Unsqueeze operators to hardware operations
    model = model.transform(InferDataLayouts())
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferDataTypes())
    model = model.transform(SortGraph())

    to_hw_transformations = [
        to_hw.InferChannelwiseLinearLayer,
        InferReplicateStream,
        to_hw.InferLabelSelectLayer,
        to_hw.InferElementwiseBinaryOperation,
    ]
    for trn in to_hw_transformations:
        model = model.transform(trn())
        model = model.transform(InferDataLayouts())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataTypes())

    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(SortGraph())
    return model


# For backwards compatibility


def step_resnet50_tidy(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Tidy up ResNet-50 models (backwards-compatible legacy step).

    Applies shape and datatype inference, constant folding, unique naming, and
    inserts a TopK layer at the output.
    """
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(InsertTopK())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    return model


def step_resnet50_streamline_linear(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Apply linear streamlining transformations to a ResNet-50 model.

    Moves and absorbs scalar linear operations (mul, add) past convolutions and
    matrix multiplications, collapses repeated operations, converts sign nodes
    to thresholds, and absorbs values into multithreshold nodes.
    """
    streamline_transformations = [
        AbsorbScalarMulAddIntoTopK(),  # before MoveAddPastMul to avoid int->float
        ConvertSubToAdd(),
        ConvertDivToMul(),
        RemoveIdentityOps(),
        CollapseRepeatedMul(),
        BatchNormToAffine(),
        ConvertSignToThres(),
        MoveAddPastMul(),
        MoveScalarAddPastMatMul(),
        MoveAddPastConv(),
        MoveScalarMulPastMatMul(),
        MoveScalarMulPastConv(),
        MoveScalarLinearPastInvariants(),
        MoveAddPastMul(),
        CollapseRepeatedAdd(),
        CollapseRepeatedMul(),
        AbsorbAddIntoMultiThreshold(),
        FactorOutMulSignMagnitude(),
        MoveMaxPoolPastMultiThreshold(),
        AbsorbMulIntoMultiThreshold(),
        Absorb1BitMulIntoMatMul(),
        Absorb1BitMulIntoConv(),
        RoundAndClipThresholds(),
    ]
    for trn in streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
    return model


def step_resnet50_streamline_nonlinear(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Apply non-linear streamlining transformations to a ResNet-50 model.

    Moves linear operations past elementwise-add nodes and fork points to
    enable further fusion in subsequent linear streamlining passes.
    """
    streamline_transformations = [
        MoveLinearPastEltwiseAdd(),
        MoveLinearPastFork(),
    ]
    for trn in streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
    return model


def step_resnet50_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Streamline a ResNet-50 model (backwards-compatible legacy step).

    Iterates linear and non-linear streamlining passes, then lowers convolutions
    to matrix multiplications and absorbs the resulting transpose operations.
    """
    for iter_id in range(4):
        model = step_resnet50_streamline_linear(model, cfg)
        model = step_resnet50_streamline_nonlinear(model, cfg)

        # big loop tidy up
        model = model.transform(RemoveUnusedTensors())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        model = model.transform(SortGraph())

    model = model.transform(DoubleToSingleFloat())

    # Lower convolutions and streamline resulting transposes
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(
        ComposedTransformation(
            [
                MoveTransposePastJoinAdd(),
                MoveTransposePastFork(),
                MoveTransposePastEltwise(),
                AbsorbConsecutiveTransposes(),
                AbsorbTransposeIntoMultiThreshold(),
            ]
        )
    )
    return model


def step_resnet50_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Convert a ResNet-50 model to hardware-specific operations (backwards-compatible legacy step).

    Sets the input datatype to UINT8, then sequentially converts channelwise
    linear layers, pooling, matrix-vector activations, thresholding, convolution
    input generators, stream duplication/addition, and label selection to their
    corresponding HLS hardware layer variants.
    """
    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])
    model = model.transform(InferDataLayouts())
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferDataTypes())
    model = model.transform(SortGraph())

    to_hw_transformations = [
        to_hw.InferChannelwiseLinearLayer,
        to_hw.InferPool,
        AbsorbConsecutiveTransposes,
        RoundAndClipThresholds,
        to_hw.InferQuantizedMatrixVectorActivation,
        to_hw.InferThresholdingLayer,
        to_hw.InferConvInpGen,
        to_hw.InferDuplicateStreamsLayer,
        to_hw.InferAddStreamsLayer,
        to_hw.InferLabelSelectLayer,
    ]
    for trn in to_hw_transformations:
        model = model.transform(trn())
        model = model.transform(InferDataLayouts())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataTypes())

    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(SortGraph())

    return model

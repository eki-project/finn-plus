
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

from finn.transformation.qonnx.fold_quant_weights import FoldQuantWeights
from finn.transformation.qonnx.infer_quant_avg_pool_2d import (
    AvgPoolAndTruncv2ToQuantAvgPool,
)
from finn.transformation.qonnx.quant_act_to_multithreshold import (
    ConvertQuantActToMultiThreshold,
    default_filter_function_generator,
)
from finn.transformation.streamline.streamline_plus import StreamlinePlus as Streamline
from finn.transformation.streamline.remove import RemoveIdentityReshape, RemoveIdentityTranspose
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.composed import ComposedTransformation
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.extract_conv_bias import ExtractBiasFromConv
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    RemoveUnusedTensors,
    SortGraph,
)
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.util.cleanup import cleanup_model

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.fpgadataflow.replicate_stream import InferReplicateStream
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbSignBiasIntoMultiThreshold,
    AbsorbTransposeIntoMultiThreshold,
)

# just for not linear
from finn.transformation.streamline.reorder import (
    MoveMulPastAdd,
)


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


# Temporary step function to replace ConvertQONNXtoFINN class, because qonnx version is to old to
# handle avgpool version parameter
def step_temp_qonnx_to_finn(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:  # noqa: ARG001
    """Convert QONNX dialect to FINN ONNX dialect."""
    model = cleanup_model(model)

    model = model.transform(ExtractBiasFromConv())
    # Gemm operations are not supported by FINN, so we convert them to MatMul
    model = model.transform(GemmToMatMul())
    model = model.transform(FoldTransposeIntoQuantInit())
    # Make sure the datatypes exist, these are required for folding the weights
    model = model.transform(InferDataTypes())
    # Fold weights
    model = model.transform(FoldQuantWeights())
    # Convert activations

    # Perform layout inference so that QuantActBaseHandler can set data_layout
    # attribute of MT for use in later layout inference and NCHW->NHWC conversion
    # in the InferThresholding transformation.
    model = model.transform(InferDataLayouts())
    model = model.transform(InferShapes())
    model = model.transform(
        ConvertQuantActToMultiThreshold(
            filter_function=default_filter_function_generator(max_multithreshold_bit_width=8),
        )
    )
    # Recompute datatypes
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    model = model.transform(InferShapes())
    # Convert AvgPool -> Mul -> Trunc structure to QuantAvgPool2d
    model = model.transform(AvgPoolAndTruncv2ToQuantAvgPool())
    # Remove empty padding if it exists
    model = model.transform(RemoveIdentityOps())
    return model

def step_resnet_streamline(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:  # noqa: ARG001
    """Streamline ResNet models."""
    transform = ComposedTransformation([
        MoveMulPastAdd(),
        AbsorbSignBiasIntoMultiThreshold(),
    ])
    model = model.transform(transform)
    model = model.transform(Streamline())
    transform2 = ComposedTransformation([
        LowerConvsToMatMul(),
        AbsorbAddIntoMultiThreshold(),
        AbsorbTransposeIntoMultiThreshold()
    ])
    model = model.transform(transform2)
    model = model.transform(Streamline())
    #model = model.transform(InsertTopK())
    #model = model.transform(AbsorbScalarMulAddIntoTopK())

    return model


def step_resnet_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:  # noqa: ARG001
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
        to_hw.InferElementwiseBinaryOperation
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

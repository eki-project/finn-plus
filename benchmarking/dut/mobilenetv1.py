from bench_base import bench
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.change_datalayout import ChangeDataLayoutQuantAvgPool2d
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.general import (
    ApplyConfig,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.remove import RemoveIdentityOps

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.transformation.streamline.absorb as absorb
import finn.transformation.streamline.reorder as reorder
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    ShellFlowType,
    VerificationStepType,
)
from finn.builder.build_dataflow_steps import verify_step
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.collapse_repeated import CollapseRepeatedMul
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds


def step_mobilenet_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(Streamline())
    additional_streamline_transformations = [
        DoubleToSingleFloat(),
        reorder.MoveMulPastDWConv(),
        absorb.AbsorbMulIntoMultiThreshold(),
        ChangeDataLayoutQuantAvgPool2d(),
        InferDataLayouts(),
        reorder.MoveTransposePastScalarMul(),
        absorb.AbsorbTransposeIntoFlatten(),
        reorder.MoveFlattenPastAffine(),
        reorder.MoveFlattenPastTopK(),
        reorder.MoveScalarMulPastMatMul(),
        CollapseRepeatedMul(),
        RemoveIdentityOps(),
        RoundAndClipThresholds(),
    ]
    for trn in additional_streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())

    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "streamlined_python", need_parent=False)

    return model


def step_mobilenet_lower_convs(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(InferDataLayouts())
    return model


def step_mobilenet_convert_to_hw_layers(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(to_hw.InferPool())
    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(to_hw.InferVectorVectorActivation())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferChannelwiseLinearLayer())
    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model


def step_mobilenet_slr_floorplan(model: ModelWrapper, cfg: DataflowBuildConfig):
    if cfg.shell_flow_type == ShellFlowType.VITIS_ALVEO:
        try:
            from finnexperimental.analysis.partitioning import partition

            # apply partitioning of the model, restricting the first and last layers
            # to SLR0
            default_slr = 0
            abs_anchors = [(0, [default_slr]), (-1, [default_slr])]
            floorplan = partition(
                model,
                cfg.synth_clk_period_ns,
                cfg.board,
                abs_anchors=abs_anchors,
                multivariant=False,
            )[0]
            # apply floorplan to model
            model = model.transform(ApplyConfig(floorplan))
            print("SLR floorplanning applied")
        except Exception:
            print("No SLR floorplanning applied")
    return model


def step_mobilenet_convert_to_hw_layers_separate_th(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(to_hw.InferPool())
    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferVectorVectorActivation())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferChannelwiseLinearLayer())
    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model


class bench_mobilenetv1(bench):
    def step_build_setup(self):
        # create build config for MobileNetV1 (based on finn-examples)
        mobilenet_build_steps = [
            step_mobilenet_streamline,
            step_mobilenet_lower_convs,
            step_mobilenet_convert_to_hw_layers_separate_th,
            "step_create_dataflow_partition",
            "step_specialize_layers",
            "step_apply_folding_config",
            "step_minimize_bit_width",
            "step_generate_estimate_reports",
            "step_set_fifo_depths",
            "step_hw_codegen",
            "step_hw_ipgen",
            "step_create_stitched_ip",
            "step_synthesize_bitfile",
            "step_make_pynq_driver",
            "step_deployment_package",
        ]
        # mobilenet_build_steps_alveo = [
        #     step_mobilenet_streamline,
        #     step_mobilenet_lower_convs,
        #     step_mobilenet_convert_to_hw_layers,
        #     "step_create_dataflow_partition",
        #     "step_specialize_layers",
        #     "step_apply_folding_config",
        #     "step_minimize_bit_width",
        #     "step_generate_estimate_reports",
        #     "step_hw_codegen",
        #     "step_hw_ipgen",
        #     "step_set_fifo_depths",
        #     "step_create_stitched_ip",
        #     step_mobilenet_slr_floorplan,
        #     "step_synthesize_bitfile",
        #     "step_make_pynq_driver",
        #     "step_deployment_package",
        # ]

        cfg = DataflowBuildConfig(
            steps=mobilenet_build_steps,
        )

        return cfg

# Copyright (C) 2020-2022 Xilinx, Inc.
# Copyright (C) 2022-2025, Advanced Micro Devices, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

import json
import numpy as np
import os
import shutil
from copy import deepcopy
from functools import partial
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    ApplyConfig,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.util.cleanup import cleanup_model
from qonnx.util.config import extract_model_config_to_json
from shutil import copy

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.transformation.streamline.absorb as absorb
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.analysis.fpgadataflow.op_and_param_counts import aggregate_dict_keys, op_and_param_counts
from finn.analysis.fpgadataflow.post_synth_res import post_synth_res
from finn.analysis.fpgadataflow.res_estimation import res_estimation, res_estimation_complete
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType,
    VerificationStepType,
)
from finn.core.onnx_exec import execute_onnx
from finn.core.rtlsim_exec import rtlsim_exec
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.derive_characteristic import (
    DeriveCharacteristic,
    DeriveFIFOSizes,
)
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.make_driver import MakeCPPDriver, MakePYNQDriver
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.replace_verilog_relpaths import ReplaceVerilogRelPaths
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import (
    InsertAndSetFIFODepths,
    RemoveShallowFIFOs,
    SplitLargeFIFOs,
    xsi_fifosim,
)
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext
from finn.transformation.fpgadataflow.vitis_build import VitisBuild
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.qonnx.quant_act_to_multithreshold import default_filter_function_generator
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.util.basic import get_liveness_threshold_cycles, get_rtlsim_trace_depth
from finn.util.logging import log
from finn.util.test import execute_parent


def verify_step(
    model: ModelWrapper,
    cfg: DataflowBuildConfig,
    step_name: str,
    need_parent: bool,
    rtlsim_pre_hook=None,
):
    log.info(f"Running verification for {step_name}")
    verify_out_dir = cfg.output_dir + "/verification_output"
    intermediate_models_dir = cfg.output_dir + "/intermediate_models"
    os.makedirs(verify_out_dir, exist_ok=True)
    (in_npy_all, exp_out_npy_all) = cfg._resolve_verification_io_pair()
    bsize_in = in_npy_all.shape[0]
    bsize_out = exp_out_npy_all.shape[0]
    assert bsize_in == bsize_out, "Batch sizes don't match for verification IO pair"
    all_res = True
    for b in range(bsize_in):
        in_npy = np.expand_dims(in_npy_all[b], axis=0)
        exp_out_npy = np.expand_dims(exp_out_npy_all[b], axis=0)
        if need_parent:
            assert cfg.save_intermediate_models, "Enable save_intermediate_models for verification"
            parent_model_fn = intermediate_models_dir + "/dataflow_parent.onnx"
            child_model_fn = intermediate_models_dir + "/verify_%s.onnx" % step_name
            model.save(child_model_fn)
            parent_model = ModelWrapper(parent_model_fn)
            out_tensor_name = parent_model.graph.output[0].name
            exp_ishape = parent_model.get_tensor_shape(parent_model.graph.input[0].name)
            if in_npy.shape != exp_ishape:
                log.warning(
                    f"Verification input has shape {in_npy.shape} while model expects {exp_ishape}"
                )
                log.info("Attempting to force model shape on verification input")
                in_npy = in_npy.reshape(exp_ishape)
            out_dict = execute_parent(parent_model_fn, child_model_fn, in_npy, return_full_ctx=True)
            out_npy = out_dict[out_tensor_name]
        else:
            inp_tensor_name = model.graph.input[0].name
            out_tensor_name = model.graph.output[0].name
            exp_ishape = model.get_tensor_shape(inp_tensor_name)
            if in_npy.shape != exp_ishape:
                log.warning(
                    f"Verification input has shape {in_npy.shape} while model expects {exp_ishape}"
                )
                log.info("Attempting to force model shape on verification input")
                in_npy = in_npy.reshape(exp_ishape)
            inp_dict = {inp_tensor_name: in_npy}
            if rtlsim_pre_hook is not None:
                out_dict = rtlsim_exec(model, inp_dict, pre_hook=rtlsim_pre_hook)
            else:
                out_dict = execute_onnx(model, inp_dict, True)
            out_npy = out_dict[out_tensor_name]
        exp_oshape = exp_out_npy.shape
        if out_npy.shape != exp_oshape:
            log.warning(
                f"Verification input has shape {exp_oshape} while model expects {out_npy.shape}"
            )
            log.info("Attempting to force model shape on verification input")
            out_npy = out_npy.reshape(exp_oshape)

        res = np.isclose(exp_out_npy, out_npy, atol=1e-3).all()
        all_res = all_res and res
        res_to_str = {True: "SUCCESS", False: "FAIL"}
        res_str = res_to_str[res]
        if cfg.verify_save_full_context:
            verification_output_fn = os.path.join(
                verify_out_dir, f"verify_{step_name}_{b}_{res_str}.npz"
            )
            np.savez(verification_output_fn, **out_dict)
        else:
            verification_output_fn = os.path.join(
                verify_out_dir, f"verify_{step_name}_{b}_{res_str}.npy"
            )
            np.save(verification_output_fn, out_npy)

        if cfg.verify_save_rtlsim_waveforms:
            wdb_path = model.get_metadata_prop("rtlsim_trace")
            if wdb_path is not None and os.path.isfile(wdb_path):
                new_wdb_path = wdb_path.replace(".wdb", "_%d.wdb" % b)
                shutil.move(wdb_path, new_wdb_path)

    log.info(f"Verification for {step_name} : {res_to_str[all_res]}")


def prepare_for_stitched_ip_rtlsim(verify_model, cfg):
    if not cfg.rtlsim_use_vivado_comps:
        need_restitch = False
        # switch impl_style=vivado components to rtl
        # StreamingFIFO must have impl_style=rtl
        for fifo_layer in verify_model.get_nodes_by_op_type("StreamingFIFO_rtl"):
            inst = getCustomOp(fifo_layer)
            if inst.get_nodeattr("impl_style") != "rtl":
                inst.set_nodeattr("impl_style", "rtl")
                inst.set_nodeattr("code_gen_dir_ipgen", "")
                inst.set_nodeattr("ipgen_path", "")
                need_restitch = True
        # if we've made alterations to the model, need to do some re-prep
        if need_restitch:
            log.info("Need to regen/re-stitch some IP for STITCHED_IP_RTLSIM")
            verify_model = verify_model.transform(
                PrepareIP(cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period())
            )
            verify_model = verify_model.transform(HLSSynthIP())
            verify_model = verify_model.transform(
                CreateStitchedIP(
                    cfg._resolve_fpga_part(),
                    cfg.synth_clk_period_ns,
                    vitis=False,
                )
            )
    else:
        log.info("rtlsim_use_vivado_comps is enabled, may yield incorrect results")

    # set top-level prop for stitched-ip rtlsim and launch
    verify_model.set_metadata_prop("exec_mode", "rtlsim")
    # TODO make configurable
    # verify_model.set_metadata_prop("rtlsim_trace", "trace.vcd")
    return verify_model


def step_qonnx_to_finn(model: ModelWrapper, cfg: DataflowBuildConfig):
    """
    This step will only execute if QONNX nodes are found.
    These include the following op_types: "Quant" , "Trunc" and "BinaryQuant".
    If such nodes are found the step will run the tidy-up step from QONNX
    and then convert the QONNX model to the FINN-ONNX dialect.
    """
    # Check if any QONNX nodes exist, i.e. BinaryQuant, Quant or Trunc
    q_count = 0
    for op_type in ["BinaryQuant", "Quant", "Trunc"]:
        q_count += len(model.get_nodes_by_op_type(op_type))
    if q_count == 0:
        return model

    # QONNX cleanup
    model = cleanup_model(model)
    # QONNX to FINN-ONNX
    model = model.transform(
        ConvertQONNXtoFINN(
            filter_function=default_filter_function_generator(
                max_multithreshold_bit_width=cfg.max_multithreshold_bit_width
            )
        )
    )

    if VerificationStepType.QONNX_TO_FINN_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "finn_onnx_python", need_parent=False)

    return model


def step_tidy_up(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Run the tidy-up step on given model. This includes shape and datatype
    inference, constant folding, and giving nodes and tensors better names.
    """

    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())

    if VerificationStepType.TIDY_UP_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "initial_python", need_parent=False)

    return model


def step_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Run streamlining on given model. Streamlining involves moving floating point
    scale/shift parameters around, collapsing adjacent ones into a single parameter,
    then absorbing the scale/shift into the following `MultiThreshold` node.
    Streamlining requires careful topology design and cannot be applied to all
    topologies.
    """

    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(Streamline())
    need_lowering = len(model.get_nodes_by_op_type("Conv")) > 0
    if need_lowering:
        model = model.transform(LowerConvsToMatMul())
        model = model.transform(MakeMaxPoolNHWC())
        model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
        model = model.transform(MakeMaxPoolNHWC())
        model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(Streamline())
    # absorb final add-mul nodes into TopK
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())

    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "streamlined_python", need_parent=False)

    return model


def step_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Convert eligible nodes to `HWCustomOp` subclasses that represent HW
    layers. Which nodes and particular configurations can be converted to HW
    is limited, see the source code of the `convert_to_hw` module for more.
    In the end am empty json file is created which can be used to set user specific
    preferred implementation styles for each node."""

    if cfg.standalone_thresholds:
        # doing this first causes all threshold layers to be standalone
        model = model.transform(to_hw.InferThresholdingLayer())
    # needed for bipolar MatMul layers
    model = model.transform(to_hw.InferBinaryMatrixVectorActivation())
    # needed for non-bipolar MatMul layers
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    # TopK to LabelSelect
    model = model.transform(to_hw.InferLabelSelectLayer())
    # input quantization (if any) as standalone threshold
    model = model.transform(to_hw.InferThresholdingLayer())
    # needed for convolutions -- TODO always exec?
    need_conv = len(model.get_nodes_by_op_type("Im2Col")) > 0
    if need_conv:
        model = model.transform(to_hw.InferConvInpGen())
        model = model.transform(to_hw.InferStreamingMaxPool())
        model = model.transform(RemoveCNVtoFCFlatten())
    # get rid of Tranpose -> Tranpose identity seq
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferDataLayouts())

    return model


def step_create_dataflow_partition(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Separate consecutive groups of HWCustomOp nodes into StreamingDataflowPartition
    nodes, which point to a separate ONNX file. Dataflow accelerator synthesis
    can only be performed on those HWCustomOp sub-graphs."""

    parent_model = model.transform(
        CreateDataflowPartition(
            partition_model_dir=cfg.output_dir + "/intermediate_models/supported_op_partitions"
        )
    )
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 1, "Only a single StreamingDataflowPartition supported."
    sdp_node = sdp_nodes[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    if cfg.save_intermediate_models:
        parent_model.save(cfg.output_dir + "/intermediate_models/dataflow_parent.onnx")
    model = ModelWrapper(dataflow_model_filename)

    # create a configuration json file that can be used to set the specialize layer config
    attrs = [
        "preferred_impl_style",
    ]
    extract_model_config_to_json(
        model, cfg.output_dir + "/template_specialize_layers_config.json", attrs
    )

    return model


def step_specialize_layers(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Convert HW nodes to either an HLS or RTL variant of the node. HW nodes
    get converted either based on pre-determined rules (details can be found
    in `specialize_layers` source code) or the user provides a configuration file
    which contains the desired setting. If the user preference cannot be fulfilled,
    a warning will be printed and the implementation style will be set to a default."""

    if cfg.specialize_layers_config_file is not None:
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(ApplyConfig(cfg.specialize_layers_config_file))
    model = model.transform(SpecializeLayers(cfg._resolve_fpga_part()))
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


def step_target_fps_parallelization(model: ModelWrapper, cfg: DataflowBuildConfig):
    """If target_fps was specified, use the SetFolding transformation to determine
    parallelization attributes. The auto-generated config will be saved under
    auto_folding_config.json under the outputs, which can serve as a basis for
    customizing the folding factors further."""

    target_cycles_per_frame = cfg._resolve_cycles_per_frame()
    if target_cycles_per_frame is not None:
        model = model.transform(
            SetFolding(
                target_cycles_per_frame,
                mvau_wwidth_max=cfg.mvau_wwidth_max,
                two_pass_relaxation=cfg.folding_two_pass_relaxation,
            )
        )
        # extract the suggested configuration and save it as json
        hw_attrs = [
            "PE",
            "SIMD",
            "parallel_window",
            "ram_style",
            "resType",
            "mem_mode",
            "runtime_writeable_weights",
            "depth_trigger_uram",
            "depth_trigger_bram",
        ]
        extract_model_config_to_json(model, cfg.output_dir + "/auto_folding_config.json", hw_attrs)

    return model


def step_apply_folding_config(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Apply the folding configuration file onto the model to set folding (parallelization)
    and other attributes, if config file is specified."""

    if cfg.folding_config_file is not None:
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(ApplyConfig(cfg.folding_config_file))

    if VerificationStepType.FOLDED_HLS_CPPSIM in cfg._resolve_verification_steps():
        # prepare cppsim
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
        verify_step(model, cfg, "folded_hls_cppsim", need_parent=True)
    return model


def step_generate_estimate_reports(model: ModelWrapper, cfg: DataflowBuildConfig):
    "Generate per-layer resource and cycle estimates using analytical models."

    if DataflowOutputType.ESTIMATE_REPORTS in cfg.generate_outputs:
        report_dir = cfg.output_dir + "/report"
        os.makedirs(report_dir, exist_ok=True)
        ops_and_params = model.analysis(op_and_param_counts)
        with open(report_dir + "/op_and_param_counts.json", "w") as f:
            json.dump(ops_and_params, f, indent=2)
        estimate_layer_cycles = model.analysis(exp_cycles_per_layer)
        with open(report_dir + "/estimate_layer_cycles.json", "w") as f:
            json.dump(estimate_layer_cycles, f, indent=2)
        estimate_layer_resources = model.analysis(
            partial(res_estimation, fpgapart=cfg._resolve_fpga_part())
        )
        estimate_layer_resources["total"] = aggregate_dict_keys(estimate_layer_resources)
        with open(report_dir + "/estimate_layer_resources.json", "w") as f:
            json.dump(estimate_layer_resources, f, indent=2)
        estimate_layer_resources_complete = model.analysis(
            partial(res_estimation_complete, fpgapart=cfg._resolve_fpga_part())
        )
        with open(report_dir + "/estimate_layer_config_alternatives.json", "w") as f:
            json.dump(estimate_layer_resources_complete, f, indent=2)
        # need to call AnnotateCycles before dataflow_performance
        model = model.transform(AnnotateCycles())
        estimate_network_performance = model.analysis(dataflow_performance)
        # add some more metrics to estimated performance
        n_clock_cycles_per_sec = (10**9) / cfg.synth_clk_period_ns
        est_fps = n_clock_cycles_per_sec / estimate_network_performance["max_cycles"]
        estimate_network_performance["estimated_throughput_fps"] = est_fps
        est_latency_ns = (
            estimate_network_performance["critical_path_cycles"] * cfg.synth_clk_period_ns
        )
        estimate_network_performance["estimated_latency_ns"] = est_latency_ns
        with open(report_dir + "/estimate_network_performance.json", "w") as f:
            json.dump(estimate_network_performance, f, indent=2)
    return model


def step_minimize_bit_width(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Tighten the weight and accumulator bit widths for each layer."""
    if cfg.minimize_bit_width:
        model = model.transform(MinimizeWeightBitWidth())
        model = model.transform(MinimizeAccumulatorWidth())
        model = model.transform(RoundAndClipThresholds())
        # make sure the changed datatypes are propagated through the network
        model = model.transform(InferDataTypes())
    return model


def step_hw_codegen(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Generate Vitis HLS code to prepare HLSBackend nodes for IP generation.
    And fills RTL templates for RTLBackend nodes."""

    model = model.transform(PrepareIP(cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period()))
    return model


def step_hw_ipgen(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Run Vitis HLS synthesis on generated code for HLSBackend nodes,
    in order to generate IP blocks. For RTL nodes this step does not do anything."""

    model = model.transform(HLSSynthIP())
    model = model.transform(ReplaceVerilogRelPaths())
    report_dir = cfg.output_dir + "/report"
    os.makedirs(report_dir, exist_ok=True)
    estimate_layer_resources_hls = model.analysis(hls_synth_res_estimation)
    with open(report_dir + "/estimate_layer_resources_hls.json", "w") as f:
        json.dump(estimate_layer_resources_hls, f, indent=2)

    if VerificationStepType.NODE_BY_NODE_RTLSIM in cfg._resolve_verification_steps():
        model = model.transform(PrepareRTLSim())
        model = model.transform(SetExecMode("rtlsim"))
        verify_step(model, cfg, "node_by_node_rtlsim", need_parent=True)
    return model


def step_set_fifo_depths(model: ModelWrapper, cfg: DataflowBuildConfig):
    """
    Depending on the auto_fifo_depths setting, do one of the following:
    * if auto_fifo_depths=True:  Run the appropriate auto-sizing transformation
    to attempt to determine the FIFO sizes that provide full throughput.
    May take a long time.
    * if auto_fifo_depths=False:  Assume the folding config file contains FIFO
    sizes as well. Runs the `InsertFIFO` transformation, then
    `ApplyConfig(cfg.folding_config_file)`, and finally `RemoveShallowFIFOs`.
    Coherency with config file node naming is ensured by calling
    `GiveUniqueNodeNames`.
    """

    if cfg.auto_fifo_depths:
        if cfg.auto_fifo_strategy == "characterize":
            model = model.transform(InsertDWC())
            model = model.transform(SpecializeLayers(cfg._resolve_fpga_part()))
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(
                PrepareIP(cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period())
            )
            model = model.transform(HLSSynthIP())
            model = model.transform(PrepareRTLSim())
            model = model.transform(AnnotateCycles())
            period = model.analysis(dataflow_performance)["max_cycles"] + 10
            model = model.transform(DeriveCharacteristic(period))
            model = model.transform(DeriveFIFOSizes())
            model = model.transform(
                InsertFIFO(
                    vivado_ram_style=cfg.large_fifo_mem_style,
                    max_qsrl_depth=256,
                    create_shallow_fifos=True,
                )
            )
            model = model.transform(SpecializeLayers(cfg._resolve_fpga_part()))
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())
        elif cfg.auto_fifo_strategy == "largefifo_rtlsim":
            if cfg.fifosim_save_waveform:
                report_dir = cfg.output_dir + "/report"
                os.makedirs(report_dir, exist_ok=True)
                model.set_metadata_prop(
                    "rtlsim_trace", os.path.abspath(report_dir) + "/fifosim_trace.wdb"
                )
            model = model.transform(
                InsertAndSetFIFODepths(
                    cfg._resolve_fpga_part(),
                    cfg._resolve_hls_clk_period(),
                    swg_exception=cfg.default_swg_exception,
                    vivado_ram_style=cfg.large_fifo_mem_style,
                    fifosim_input_throttle=cfg.fifosim_input_throttle,
                )
            )
            # InsertAndSetFIFODepths internally removes any shallow FIFOs
            # so no need to call RemoveShallowFIFOs here
        else:
            assert "Unsupported auto_fifo_strategy: " + cfg.auto_fifo_strategy
    else:
        # assume folding cfg json contains FIFO sizes too
        # insert DWCs, FIFOs and run ApplyConfig once more
        model = model.transform(InsertDWC())
        # need to make sure all FIFOs are created so that their depth can be
        # set by ApplyConfig, so create_shallow_fifos=True
        model = model.transform(InsertFIFO(create_shallow_fifos=True))
        model = model.transform(SpecializeLayers(cfg._resolve_fpga_part()))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        if cfg.folding_config_file is not None:
            model = model.transform(ApplyConfig(cfg.folding_config_file))

    # extract the final configuration and save it as json
    hw_attrs = [
        "PE",
        "SIMD",
        "parallel_window",
        "ram_style",
        "depth",
        "impl_style",
        "resType",
        "mem_mode",
        "runtime_writeable_weights",
        "inFIFODepths",
        "outFIFODepths",
        "depth_trigger_uram",
        "depth_trigger_bram",
    ]
    extract_model_config_to_json(model, cfg.output_dir + "/final_hw_config.json", hw_attrs)

    # perform FIFO splitting and shallow FIFO removal only after the final config
    # json file has been written. otherwise, since these transforms may add/remove
    # FIFOs, we get name mismatch problems when trying to reuse the final config.
    if cfg.split_large_fifos:
        model = model.transform(SplitLargeFIFOs())
    model = model.transform(RemoveShallowFIFOs())

    # after FIFOs are ready to go, call PrepareIP and HLSSynthIP again
    # this will only run for the new nodes (e.g. FIFOs and DWCs)
    model = model.transform(PrepareIP(cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period()))
    model = model.transform(HLSSynthIP())
    return model


def step_create_stitched_ip(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Create stitched IP for a graph after all HLS IP blocks have been generated.
    Depends on the DataflowOutputType.STITCHED_IP output product."""

    if DataflowOutputType.STITCHED_IP in cfg.generate_outputs:
        stitched_ip_dir = cfg.output_dir + "/stitched_ip"
        model = model.transform(
            CreateStitchedIP(
                cfg._resolve_fpga_part(),
                cfg.synth_clk_period_ns,
                vitis=cfg.stitched_ip_gen_dcp,
                signature=cfg.signature,
            )
        )
        # TODO copy all ip sources into output dir? as zip?
        shutil.copytree(
            model.get_metadata_prop("vivado_stitch_proj"), stitched_ip_dir, dirs_exist_ok=True
        )
        log.info(f"Vivado stitched IP written into {stitched_ip_dir}")
    if VerificationStepType.STITCHED_IP_RTLSIM in cfg._resolve_verification_steps():
        # prepare ip-stitched rtlsim
        verify_model = deepcopy(model)
        verify_model = prepare_for_stitched_ip_rtlsim(verify_model, cfg)
        # use critical path estimate to set rtlsim liveness threshold
        # (very conservative)
        verify_model = verify_model.transform(AnnotateCycles())
        estimate_network_performance = verify_model.analysis(dataflow_performance)
        prev_liveness = get_liveness_threshold_cycles()
        os.environ["LIVENESS_THRESHOLD"] = str(
            int(estimate_network_performance["critical_path_cycles"] * 1.1)
        )
        if cfg.verify_save_rtlsim_waveforms:
            verify_out_dir = cfg.output_dir + "/verification_output"
            os.makedirs(verify_out_dir, exist_ok=True)
            abspath = os.path.abspath(verify_out_dir)
            verify_model.set_metadata_prop("rtlsim_trace", abspath + "/verify_rtlsim.wdb")
        verify_step(verify_model, cfg, "stitched_ip_rtlsim", need_parent=True)
        os.environ["LIVENESS_THRESHOLD"] = str(prev_liveness)
    return model


def step_measure_rtlsim_performance(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Measure performance + latency of stitched-IP model in rtlsim (xsi).
    Depends on the DataflowOutputType.STITCHED_IP output product.
    """

    if DataflowOutputType.RTLSIM_PERFORMANCE in cfg.generate_outputs:
        assert (
            DataflowOutputType.STITCHED_IP in cfg.generate_outputs
        ), "rtlsim_perf needs stitched IP"
        report_dir = cfg.output_dir + "/report"
        os.makedirs(report_dir, exist_ok=True)
        rtlsim_bs = int(cfg.rtlsim_batch_size)
        orig_rtlsim_trace_depth = get_rtlsim_trace_depth()
        assert rtlsim_bs > 0, "rtlsim batch size must be >0"
        if cfg.verify_save_rtlsim_waveforms:
            # set depth to 3 for layer-by-layer visibility
            os.environ["RTLSIM_TRACE_DEPTH"] = "3"
            model.set_metadata_prop(
                "rtlsim_trace",
                "%s/rtlsim_perf_batch_%d.wdb" % (os.path.abspath(report_dir), rtlsim_bs),
            )
        rtlsim_perf_dict = xsi_fifosim(model, rtlsim_bs)
        # keep keys consistent between the Python and C++-styles
        cycles = rtlsim_perf_dict["cycles"]
        clk_ns = cfg.synth_clk_period_ns
        fclk_mhz = 1 / (clk_ns * 0.001)
        runtime_s = (cycles * clk_ns) * (10**-9)
        rtlsim_perf_dict["runtime[ms]"] = runtime_s * 1000
        rtlsim_perf_dict["throughput[images/s]"] = rtlsim_bs / runtime_s
        rtlsim_perf_dict["fclk[mhz]"] = fclk_mhz
        for key, val in rtlsim_perf_dict.items():
            if "max_count" in key:
                del rtlsim_perf_dict[key]
        # estimate stable-state throughput based on latency+throughput
        if rtlsim_bs == 1:
            rtlsim_perf_dict["stable_throughput[images/s]"] = rtlsim_perf_dict[
                "throughput[images/s]"
            ]
        else:
            total_cycles = rtlsim_perf_dict["cycles"]
            latency_cycles = rtlsim_perf_dict["latency_cycles"]
            stablestate_cycles = total_cycles - latency_cycles
            clk_ns = cfg.synth_clk_period_ns
            fclk_mhz = 1 / (clk_ns * 0.001)
            runtime_s = (stablestate_cycles * clk_ns) * (10**-9)
            rtlsim_perf_dict["stable_throughput[images/s]"] = rtlsim_bs / runtime_s

        with open(report_dir + "/rtlsim_performance.json", "w") as f:
            json.dump(rtlsim_perf_dict, f, indent=2)
        if cfg.verify_save_rtlsim_waveforms:
            # restore original trace depth
            os.environ["RTLSIM_TRACE_DEPTH"] = str(orig_rtlsim_trace_depth)

    return model


def step_make_driver(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Create a driver that can be used to interface the generated accelerator.
    Use DataflowBuildConfig to select PYNQ Python or C++ driver."""

    driver_dir = os.path.join(cfg.output_dir, "driver")
    if DataflowOutputType.PYNQ_DRIVER in cfg.generate_outputs:
        # generate PYNQ driver
        model = model.transform(MakePYNQDriver(cfg._resolve_driver_platform()))
        shutil.copytree(model.get_metadata_prop("pynq_driver_dir"), driver_dir, dirs_exist_ok=True)
        log.info("PYNQ Python driver written into " + driver_dir)
    elif DataflowOutputType.CPP_DRIVER in cfg.generate_outputs:
        # generate C++ Driver

        model = model.transform(
            MakeCPPDriver(
                cfg._resolve_driver_platform(),
                build_dir=cfg.output_dir,
                version=cfg.cpp_driver_version,
                driver_dir=driver_dir,
            )
        )
        log.info("C++ driver written into " + driver_dir)
    else:
        log.warning(
            "The step step_make_driver is in the build list but will not be executed"
            + " since no driver is selected in generate_outputs in your build.py file!"
        )
    return model


def step_out_of_context_synthesis(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Run out-of-context synthesis and generate reports.
    Depends on the DataflowOutputType.STITCHED_IP output product."""
    if DataflowOutputType.OOC_SYNTH in cfg.generate_outputs:
        assert DataflowOutputType.STITCHED_IP in cfg.generate_outputs, "OOC needs stitched IP"
        model = model.transform(
            SynthOutOfContext(part=cfg._resolve_fpga_part(), clk_period_ns=cfg.synth_clk_period_ns)
        )
        report_dir = cfg.output_dir + "/report"
        os.makedirs(report_dir, exist_ok=True)
        ooc_res_dict = model.get_metadata_prop("res_total_ooc_synth")
        ooc_res_dict = eval(ooc_res_dict)

        estimate_network_performance = model.analysis(dataflow_performance)
        # add some more metrics to estimated performance
        n_clock_cycles_per_sec = float(ooc_res_dict["fmax_mhz"]) * (10**6)
        est_fps = n_clock_cycles_per_sec / estimate_network_performance["max_cycles"]
        ooc_res_dict["estimated_throughput_fps"] = est_fps
        with open(report_dir + "/ooc_synth_and_timing.json", "w") as f:
            json.dump(ooc_res_dict, f, indent=2)
    return model


def step_synthesize_bitfile(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Synthesize a bitfile for the using the specified shell flow, using either
    Vivado or Vitis, to target the specified board."""

    if DataflowOutputType.BITFILE in cfg.generate_outputs:
        bitfile_dir = cfg.output_dir + "/bitfile"
        os.makedirs(bitfile_dir, exist_ok=True)
        report_dir = cfg.output_dir + "/report"
        os.makedirs(report_dir, exist_ok=True)
        partition_model_dir = cfg.output_dir + "/intermediate_models/kernel_partitions"
        if cfg.shell_flow_type == ShellFlowType.VIVADO_ZYNQ:
            model = model.transform(
                ZynqBuild(
                    cfg.board,
                    cfg.synth_clk_period_ns,
                    cfg.enable_hw_debug,
                    partition_model_dir=partition_model_dir,
                )
            )
            copy(model.get_metadata_prop("bitfile"), bitfile_dir + "/finn-accel.bit")
            copy(model.get_metadata_prop("hw_handoff"), bitfile_dir + "/finn-accel.hwh")
            copy(
                model.get_metadata_prop("vivado_synth_rpt"),
                report_dir + "/post_synth_resources.xml",
            )

            post_synth_resources = model.analysis(post_synth_res)
            with open(report_dir + "/post_synth_resources.json", "w") as f:
                json.dump(post_synth_resources, f, indent=2)

            vivado_pynq_proj_dir = model.get_metadata_prop("vivado_pynq_proj")
            timing_rpt = (
                "%s/finn_zynq_link.runs/impl_1/top_wrapper_timing_summary_routed.rpt"
                % vivado_pynq_proj_dir
            )
            copy(timing_rpt, report_dir + "/post_route_timing.rpt")

        elif cfg.shell_flow_type == ShellFlowType.VITIS_ALVEO:
            model = model.transform(
                VitisBuild(
                    cfg._resolve_fpga_part(),
                    cfg.synth_clk_period_ns,
                    cfg._resolve_vitis_platform(),
                    strategy=cfg.vitis_opt_strategy,
                    enable_debug=cfg.enable_hw_debug,
                    floorplan_file=cfg.vitis_floorplan_file,
                    partition_model_dir=partition_model_dir,
                    fpga_memory_type=cfg.fpga_memory,
                )
            )
            copy(model.get_metadata_prop("bitfile"), bitfile_dir + "/finn-accel.xclbin")
            copy(
                model.get_metadata_prop("vivado_synth_rpt"),
                report_dir + "/post_synth_resources.xml",
            )

            post_synth_resources = model.analysis(post_synth_res)
            with open(report_dir + "/post_synth_resources.json", "w") as f:
                json.dump(post_synth_resources, f, indent=2)
        else:
            raise Exception("Unrecognized shell_flow_type: " + str(cfg.shell_flow_type))
        log.info(f"Bitfile written into {bitfile_dir}")

    return model


def step_deployment_package(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Create a deployment package including the driver and bitfile."""

    if DataflowOutputType.DEPLOYMENT_PACKAGE in cfg.generate_outputs:
        deploy_dir = cfg.output_dir + "/deploy"
        bitfile_dir = cfg.output_dir + "/bitfile"
        driver_dir = cfg.output_dir + "/driver"
        os.makedirs(deploy_dir, exist_ok=True)
        shutil.copytree(bitfile_dir, deploy_dir + "/bitfile", dirs_exist_ok=True)
        shutil.copytree(driver_dir, deploy_dir + "/driver", dirs_exist_ok=True)
    return model


#: map step name strings to step functions
build_dataflow_step_lookup = {
    "step_qonnx_to_finn": step_qonnx_to_finn,
    "step_tidy_up": step_tidy_up,
    "step_streamline": step_streamline,
    "step_convert_to_hw": step_convert_to_hw,
    "step_specialize_layers": step_specialize_layers,
    "step_create_dataflow_partition": step_create_dataflow_partition,
    "step_target_fps_parallelization": step_target_fps_parallelization,
    "step_apply_folding_config": step_apply_folding_config,
    "step_minimize_bit_width": step_minimize_bit_width,
    "step_generate_estimate_reports": step_generate_estimate_reports,
    "step_hw_codegen": step_hw_codegen,
    "step_hw_ipgen": step_hw_ipgen,
    "step_set_fifo_depths": step_set_fifo_depths,
    "step_create_stitched_ip": step_create_stitched_ip,
    "step_measure_rtlsim_performance": step_measure_rtlsim_performance,
    "step_make_driver": step_make_driver,
    "step_out_of_context_synthesis": step_out_of_context_synthesis,
    "step_synthesize_bitfile": step_synthesize_bitfile,
    "step_deployment_package": step_deployment_package,
}

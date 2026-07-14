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

# ruff: noqa: SLF001
"""Collection of default build steps for building and verifying a dataflow
accelerator from an ONNX model.
"""

import json
import math
import numpy as np
import os
import shutil
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from onnx import NodeProto
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
    SortGraph,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.util.cleanup import cleanup_model
from shutil import copy, move
from typing import TYPE_CHECKING, cast

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.transformation.streamline.absorb as absorb
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.analysis.fpgadataflow.op_and_param_counts import aggregate_dict_keys, op_and_param_counts
from finn.analysis.fpgadataflow.post_synth_res import post_synth_res
from finn.analysis.fpgadataflow.res_estimation import res_estimation, res_estimation_complete
from finn.analysis.fpgadataflow.unsupported_layers import unsupported_layers
from finn.builder.build_dataflow_config import (
    AutoFIFOSizingMethod,
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType,
    VerificationStepType,
)
from finn.builder.passes import step_passes_frontend
from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.transformation.fpgadataflow.loop_rolling import LoopExtraction, LoopRolling
from finn.transformation.fpgadataflow.make_driver import (
    MakeCPPDriver,
    MakePYNQDriver,
    update_bitfile_path_after_copy,
)
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.replace_verilog_relpaths import ReplaceVerilogRelPaths
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import (
    ApplyFIFODepthsFromFile,
    ApplySimulatedFIFOSizes,
    SplitLargeFIFOs,
)
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.transformation.fpgadataflow.set_loop_boundary import SetLoopBoundary
from finn.transformation.fpgadataflow.simulation_build import BuildSimulation, SimulationType
from finn.transformation.fpgadataflow.simulation_connected import (
    NodeConnectedSimulation,
    RunLayerParallelSimulation,
)
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext
from finn.transformation.fpgadataflow.transpose_decomposition import (
    InferInnerOuterShuffles,
    ShuffleDecomposition,
)
from finn.transformation.fpgadataflow.vitis_build import VitisBuild
from finn.transformation.fpgadataflow.vivado_power_estimation import VivadoPowerEstimation
from finn.transformation.general import ApplyConfig
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.qonnx.give_unique_node_names_recursive import GiveUniqueNodeNamesRecursive
from finn.transformation.qonnx.quant_act_to_multithreshold import default_filter_function_generator
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.util.basic import get_liveness_threshold_cycles, get_rtlsim_trace_depth, getHWCustomOp
from finn.util.config import extract_model_config_to_json
from finn.util.exception import FINNUserError
from finn.util.execution import execute_parent
from finn.util.logging import log
from finn.util.mlo_sim import is_mlo, mlo_prehook_func_factory
from finn.xsi import SimEngine

if TYPE_CHECKING:
    from finn.custom_op.fpgadataflow.rtl.finn_loop import FINNLoop


BuildDataflowStep = Callable[..., ModelWrapper]
build_dataflow_step_lookup: dict[str, BuildDataflowStep] = {}


def register_build_dataflow_step(
    step_name: str | None = None,
) -> Callable[[BuildDataflowStep], BuildDataflowStep]:
    """Register a dataflow build step.

    Uses the function name by default, unless step_name is explicitly provided.
    """

    def _decorator(step_fn: BuildDataflowStep) -> BuildDataflowStep:
        """Register the build step function in the lookup table."""
        key = step_name if step_name is not None else step_fn.__name__
        if key in build_dataflow_step_lookup:
            raise ValueError(f"Duplicate build step registration: {key}")
        build_dataflow_step_lookup[key] = step_fn
        return step_fn

    return _decorator


def verify_step(
    model: ModelWrapper,
    cfg: DataflowBuildConfig,
    step_name: str,
    need_parent: bool,
    rtlsim_pre_hook: Callable[[SimEngine], None] | None = None,
) -> None:
    """Verify a build step by running simulation and comparing results.

    Args:
        model: The ONNX model to verify
        cfg: Build configuration object
        step_name: Name of the build step being verified
        need_parent: Whether parent model execution is needed for comparison
        rtlsim_pre_hook: Optional pre-hook function for RTL simulation
    """
    log.info(f"Running verification for {step_name}")
    output_dir = Path(cfg.output_dir)
    verify_out_dir = output_dir / "verification_output"
    intermediate_models_dir = output_dir / "intermediate_models"
    # Ensure tensor names are sorted and readable for easier debugging
    model = model.transform(SortGraph())
    model = model.transform(GiveUniqueNodeNamesRecursive())
    model = model.transform(GiveReadableTensorNames())
    verify_out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.verify_steps is None:
        raise FINNUserError("verify_steps is not set in config, but verification step was called")
    (in_npy_all, exp_out_npy_all) = cast(
        "tuple[np.ndarray, np.ndarray]", cfg._resolve_verification_io_pair()
    )
    bsize_in = in_npy_all.shape[0]
    bsize_out = exp_out_npy_all.shape[0]
    assert bsize_in == bsize_out, "Batch sizes don't match for verification IO pair"
    all_res = True
    out_dict: dict[str, np.ndarray] = {}
    parent_model = None
    res_to_str = {True: "SUCCESS", False: "FAIL"}
    for b in range(bsize_in):
        in_npy = np.expand_dims(in_npy_all[b], axis=0)
        exp_out_npy = np.expand_dims(exp_out_npy_all[b], axis=0)
        if need_parent:
            assert cfg.save_intermediate_models, "Enable save_intermediate_models for verification"
            parent_model_fn = intermediate_models_dir / "dataflow_parent.onnx"
            child_model_fn = intermediate_models_dir / f"verify_{step_name}.onnx"
            model.save(child_model_fn)
            parent_model = ModelWrapper(str(parent_model_fn))
            out_tensor_name = parent_model.get_first_global_out()
            exp_ishape = parent_model.get_tensor_shape(parent_model.get_first_global_in())
            if exp_ishape is None:
                raise FINNUserError(
                    f"Unable to determine expected input shape for verification. "
                    f"Shape of tensor {parent_model.get_first_global_in()} is None."
                )
            if in_npy.shape != exp_ishape:
                log.warning(
                    f"Verification input has shape {in_npy.shape} while model expects {exp_ishape}"
                )
                log.info("Attempting to force model shape on verification input")
                in_npy = in_npy.reshape(exp_ishape)
            out_dict = cast(
                "dict[str, np.ndarray]",
                execute_parent(parent_model_fn, child_model_fn, in_npy, return_full_ctx=True),
            )
            out_npy = out_dict[out_tensor_name]
        else:
            inp_tensor_name = model.get_first_global_in()
            out_tensor_name = model.get_first_global_out()
            exp_ishape = model.get_tensor_shape(inp_tensor_name)
            if exp_ishape is None:
                raise FINNUserError(
                    f"Unable to determine expected input shape for verification. "
                    f"Shape of tensor {model.get_first_global_in()} is None."
                )
            if in_npy.shape != exp_ishape:
                log.warning(
                    f"Verification input has shape {in_npy.shape} while model expects {exp_ishape}"
                )
                log.info("Attempting to force model shape on verification input")
                in_npy = in_npy.reshape(exp_ishape)
            inp_dict = {inp_tensor_name: in_npy}
            out_dict = execute_onnx(model, inp_dict, True, pre_hook=rtlsim_pre_hook)
            out_npy = out_dict[out_tensor_name]
        exp_oshape = exp_out_npy.shape
        if out_npy.shape != exp_oshape:
            log.warning(
                f"Verification input has shape {exp_oshape} while model expects {out_npy.shape}"
            )
            log.info("Attempting to force model shape on verification input")
            out_npy = out_npy.reshape(exp_oshape)

        # Check 1: Element-wise closeness between output and expected output
        res1 = np.isclose(
            out_npy, exp_out_npy, atol=cfg.verification_atol, rtol=cfg.verification_rtol
        ).all()
        # Check 2 and 3: Mean absolute and relative error over all output elements
        num_elements = out_npy.size
        abs_error = np.abs(out_npy - exp_out_npy)
        # Avoid division by zero for relative error
        exp_out_npy_safe = np.where(exp_out_npy == 0, np.finfo(float).eps, exp_out_npy)
        rel_error = np.abs((out_npy - exp_out_npy) / exp_out_npy_safe)
        res2 = np.mean(abs_error) <= cfg.verification_mean_atol
        res3 = np.mean(rel_error) <= cfg.verification_mean_rtol

        res = res1 and res2 and res3
        all_res = all_res and res
        res_str = res_to_str[bool(res)]
        if cfg.verify_save_full_context and (rtlsim_pre_hook is None):
            verification_output_fn = verify_out_dir / f"verify_{step_name}_{b}_{res_str}.npz"
            np.savez(verification_output_fn, **out_dict)

            # Log tensor statistics for debugging (only output tensors, in topological order)
            tensors_to_log = ["global_in"]
            if need_parent:
                if parent_model is None:
                    raise FINNUserError("Parent model is needed for verification but is None")
                for node in parent_model.graph.node:
                    for output in node.output:
                        tensors_to_log.append(output)  # noqa: PERF402
                sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
                sdp_prefix = sdp_node.name + "_"
            else:
                sdp_prefix = ""
            for node in model.graph.node:
                for output in node.output:
                    tensors_to_log.append(sdp_prefix + output)

            tensor_stats = []
            for key in tensors_to_log:
                if key in out_dict:
                    stat_dict = {
                        "tensor": key,
                        "shape": list(out_dict[key].shape),
                        "mean": float(np.mean(out_dict[key])),
                        "std": float(np.std(out_dict[key])),
                        "min": float(np.min(out_dict[key])),
                        "max": float(np.max(out_dict[key])),
                    }
                    tensor_stats.append(stat_dict)

            # Write tensor statistics in compact human-readable table format
            with (verify_out_dir / f"verify_{step_name}_{b}_{res_str}_stats.txt").open("w") as f:
                # Write header
                f.write(
                    f"{'Tensor':<40} {'Shape':<20} {'Mean':<12} "
                    f"{'Std':<12} {'Min':<12} {'Max':<12}\n"
                )
                f.write("-" * 108 + "\n")

                # Write data rows
                for stat in tensor_stats:
                    # Shorten/truncate long names and shapes
                    tensor_name = stat["tensor"].replace("GenericPartition", "GP")[:39]
                    shape_str = str(stat["shape"])[:19]
                    f.write(
                        f"{tensor_name:<40} {shape_str:<20} {stat['mean']:<12.6f} "
                        f"{stat['std']:<12.6f} {stat['min']:<12.6f} {stat['max']:<12.6f}\n"
                    )

                # Add output error analysis
                f.write("\n" + "=" * 108 + "\n")
                f.write("OUTPUT ERROR ANALYSIS\n")
                f.write("=" * 108 + "\n")

                f.write(f"Number of elements:           {num_elements}\n")
                f.write(f"Min absolute error:           {np.min(abs_error):.6e}\n")
                f.write(f"Max absolute error:           {np.max(abs_error):.6e}\n")
                f.write(f"Mean absolute error:          {np.mean(abs_error):.6e}\n")
                f.write(f"Min relative error:           {np.min(rel_error):.6e}\n")
                f.write(f"Max relative error:           {np.max(rel_error):.6e}\n")
                f.write(f"Mean relative error:          {np.mean(rel_error):.6e}\n")
                f.write(
                    f"Tolerance per element:        atol={cfg.verification_atol:.6e} + "
                    f"rtol={cfg.verification_rtol:.6e}\n"
                )
                f.write(f"Tolerance for mean abs. err:  {cfg.verification_mean_atol:.6e}\n")
                f.write(f"Tolerance for mean rel. err:  {cfg.verification_mean_rtol:.6e}\n")
                f.write(f"Verification result:          {res_str}\n")
        else:
            if cfg.verify_save_full_context:
                log.warning("Warning: Unable to save the full context when using MLO")
            verification_output_fn = verify_out_dir / f"verify_{step_name}_{b}_{res_str}.npy"
            np.save(verification_output_fn, out_npy)

        if cfg.verify_save_rtlsim_waveforms:
            # Handle model-level waveform (stitched IP rtlsim)
            wdb_path = model.get_metadata_prop("rtlsim_trace")
            if wdb_path is not None and Path(wdb_path).is_file():
                new_wdb_path = wdb_path.replace(".wdb", f"_{b}.wdb")
                shutil.move(wdb_path, new_wdb_path)
            # Handle node-level waveforms (only for node-by-node rtlsim)
            if step_name == "node_by_node_rtlsim":
                for node in model.graph.node:
                    node_inst = getCustomOp(node)
                    node_wdb_path = cast("str", node_inst.get_nodeattr("rtlsim_trace"))
                    if node_wdb_path is not None and Path(node_wdb_path).is_file():
                        new_node_wdb_path = node_wdb_path.replace(".wdb", f"_{b}.wdb")
                        shutil.move(node_wdb_path, new_node_wdb_path)

    log.info(f"Verification for {step_name} : {res_to_str[bool(all_res)]}")


@register_build_dataflow_step()
def step_hw_codegen(
    model: ModelWrapper, cfg: DataflowBuildConfig, parent_node: str | None = None
) -> ModelWrapper:
    """Generate Vitis HLS code to prepare HLSBackend nodes for IP generation.
    And fills RTL templates for RTLBackend nodes."""
    model = model.transform(GiveUniqueNodeNamesRecursive(prefix=parent_node))
    model = model.transform(
        PrepareIP(cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period()),
        apply_to_subgraphs=True,
        use_preorder_traversal=False,
    )
    return model


@register_build_dataflow_step()
def step_hw_ipgen(
    model: ModelWrapper, cfg: DataflowBuildConfig, parent_node: str | None = None
) -> ModelWrapper:
    """Run Vitis HLS synthesis on generated code for HLSBackend nodes,
    in order to generate IP blocks. For RTL nodes this step does not do anything."""
    model = model.transform(HLSSynthIP(cfg._resolve_fpga_part()))
    model = model.transform(ReplaceVerilogRelPaths())

    # Emit resource consumption reports
    report_dir = Path(cfg.output_dir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    estimate_layer_resources_hls = model.analysis(hls_synth_res_estimation)
    estimate_layer_resources_hls["total"] = aggregate_dict_keys(estimate_layer_resources_hls)
    filename = (
        "estimate_layer_resources_hls.json"
        if parent_node is None
        else f"estimate_layer_resources_hls_{parent_node}.json"
    )
    with (report_dir / filename).open("w") as f:
        json.dump(estimate_layer_resources_hls, f, indent=2)

    # Optional verifification step using node by node rtl simulation
    # (only supported for top level model)
    if (
        VerificationStepType.NODE_BY_NODE_RTLSIM in cfg._resolve_verification_steps()
        and parent_node is None
    ):
        if cfg.verify_save_rtlsim_waveforms:
            verify_out_dir = Path(cfg.output_dir) / "verification_output"
            waveform_dir = verify_out_dir / "node_by_node_rtlsim_waveforms"
            waveform_dir.mkdir(parents=True, exist_ok=True)
            abspath = waveform_dir.absolute()
            # Set rtlsim_trace on each node BEFORE PrepareRTLSim so compilation uses debug=True
            for node in model.graph.node:
                node_inst = getCustomOp(node)
                node_inst.set_nodeattr("rtlsim_trace", f"{abspath}/{node.name}_rtlsim.wdb")
        model = model.transform(PrepareRTLSim())
        model = model.transform(SetExecMode("rtlsim"))
        verify_step(model, cfg, "node_by_node_rtlsim", need_parent=True)

    return model


@register_build_dataflow_step()
def step_set_fifo_depths(
    model: ModelWrapper, cfg: DataflowBuildConfig, parent_node: str | None = None
) -> ModelWrapper:
    """Depending on the auto_fifo_depths setting, do one of the following:
    * if auto_fifo_depths=True:  Run the appropriate auto-sizing transformation
    to attempt to determine the FIFO sizes that provide full throughput.
    May take a long time.
    * if auto_fifo_depths=False:  Load the FIFO sizes from the folding config file and apply them.
    Coherency with config file node naming is ensured by calling
    `GiveUniqueNodeNamesRecursive`.
    """
    if cfg.auto_fifo_depths:
        if cfg.fifosim_save_waveform:
            report_dir = Path(cfg.output_dir) / "report"
            report_dir.mkdir(parents=True, exist_ok=True)
            model.set_metadata_prop("rtlsim_trace", str(report_dir.resolve() / "fifosim_trace.wdb"))
        if cfg.auto_fifo_strategy == AutoFIFOSizingMethod.DISTRIBUTED_SIMULATION:
            if cfg.fifosim_save_waveform:
                report_dir = Path(cfg.output_dir) / "report"
                report_dir.mkdir(parents=True, exist_ok=True)
                tracefile = (
                    f"{parent_node}_fifosim_trace.wdb"
                    if parent_node is not None
                    else "fifosim_trace.wdb"
                )
                model.set_metadata_prop("rtlsim_trace", str(report_dir.absolute()) + tracefile)

            model = model.transform(
                BuildSimulation(
                    cfg._resolve_fpga_part(),
                    cfg._resolve_hls_clk_period(),
                    cfg.functional_simulation,
                    performance_sim=False,
                )
            )
            model = model.transform(
                RunLayerParallelSimulation(
                    cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period(), cfg
                )
            )
            model = model.transform(ApplySimulatedFIFOSizes(cfg))
        elif cfg.auto_fifo_strategy == AutoFIFOSizingMethod.LIVE_FIFO:
            hw_attrs = [
                "PE",
                "SIMD",
                "EmbFold",
                "SeqFold",
                "parallel_window",
                "ram_style",
                "ram_style_thresholds",
                "ram_style_mask",
                "depth",
                "impl_style",
                "resType",
                "mac_resource",
                "mem_mode",
                "runtime_writeable_weights",
                "inFIFODepths",
                "outFIFODepths",
                "depth_trigger_uram",
                "depth_trigger_bram",
            ]
            # Create all DWCs and FIFOs normally
            model = model.transform(InsertDWC())
            model = model.transform(
                InsertFIFO(vivado_ram_style=cfg.large_fifo_mem_style, create_shallow_fifos=True)
            )

            # Clean up model
            model = model.transform(SortGraph())
            model = model.transform(GiveUniqueNodeNamesRecursive(prefix=parent_node))
            model = model.transform(GiveReadableTensorNames())

            # save original folding config before potentially modifying it
            cfg_path = Path(cfg.output_dir) / "report" / "folding_config_before_lfs.json"
            extract_model_config_to_json(model, cfg_path, hw_attrs)
            model.set_metadata_prop("folding_config_before_lfs", str(cfg_path))

            # Disable runtime-writable weights, external weights, and dynamic mode
            for node in model.graph.node:
                if node.domain.startswith("finn.custom_op.fpgadataflow"):
                    node_inst = getCustomOp(node)
                    try:
                        if node_inst.get_nodeattr("runtime_writeable_weights") == 1:
                            node_inst.set_nodeattr("runtime_writeable_weights", 0)
                            if node_inst.get_nodeattr("ram_style") == "ultra":
                                node_inst.set_nodeattr("ram_style", "block")
                    except AttributeError:
                        pass
                    try:
                        if node_inst.get_nodeattr("mem_mode") == "external":
                            node_inst.set_nodeattr("mem_mode", "internal_decoupled")
                    except AttributeError:
                        pass
                    try:
                        if node_inst.get_nodeattr("dynamic_mode") == 1:
                            node_inst.set_nodeattr("dynamic_mode", 0)
                    except AttributeError:
                        pass

            # Specialize FIFOs to RTL back-end
            for node in model.get_nodes_by_op_type("StreamingFIFO"):
                node_inst = getCustomOp(node)
                node_inst.set_nodeattr("preferred_impl_style", "rtl")
            model = model.transform(SpecializeLayers(cfg._resolve_fpga_part()))

            # Clean up model
            model = model.transform(SortGraph())
            model = model.transform(GiveUniqueNodeNamesRecursive(prefix=parent_node))
            model = model.transform(GiveReadableTensorNames())

            # Set impl_style + ID attributes
            # We can't infer ID from the unique node name at IP instantiation,
            # because the nodes will be wrapped in SDPs
            for node in model.get_nodes_by_op_type("StreamingFIFO_rtl"):
                node_inst = getCustomOp(node)
                idf = int(node.name.split("_")[-1])
                node_inst.set_nodeattr("impl_style", "virtual")
                node_inst.set_nodeattr("fifo_id", idf)

            return model
        else:
            raise FINNUserError("Unsupported auto_fifo_strategy: " + cfg.auto_fifo_strategy)

        # generate a dedicated report about final FIFO sizes
        # Report has to be generated before large FIFOs are split.
        fifo_info = {}
        fifo_info["fifo_depths"] = {}
        fifo_info["fifo_sizes"] = {}
        fifo_info["impl_style"] = {}
        fifo_info["ram_style"] = {}
        total_fifo_size = 0
        for node in model.get_nodes_by_op_type("StreamingFIFO_rtl"):
            node_inst = getHWCustomOp(node)
            fifo_info["fifo_depths"][node.name] = node_inst.get_nodeattr("depth")
            fifo_info["fifo_sizes"][node.name] = (
                node_inst.get_instream_width()
                * math.ceil(cast("int", node_inst.get_nodeattr("depth")) / 32)
                * 32
            )  # Round up to nearest multiple of 32 to reflect actual hardware usage
            fifo_info["impl_style"][node.name] = node_inst.get_nodeattr("impl_style")
            fifo_info["ram_style"][node.name] = node_inst.get_nodeattr("ram_style")
            total_fifo_size += fifo_info["fifo_sizes"][node.name]
        fifo_info["total_fifo_size_kiB"] = total_fifo_size / 8.0 / 1024.0

        with (Path(cfg.output_dir) / "report" / "fifo_sizing.json").open("w") as f:
            json.dump(fifo_info, f, indent=2)

        if cfg.split_large_fifos:
            model = model.transform(SplitLargeFIFOs(max_qsrl_depth=256))
        model = model.transform(GiveUniqueNodeNamesRecursive(prefix=parent_node))
        model = model.transform(GiveReadableTensorNames())
    else:
        if cfg.fifo_config_file is None:
            raise FINNUserError("auto_fifo_depths is set to False but no fifo_config_file provided")
        log.info(
            f"auto_fifo_depths is set to False, applying FIFO sizes from {cfg.fifo_config_file}"
        )
        # insert DWCs, FIFOs and run ApplyConfig once more
        model = model.transform(InsertDWC())
        # need to make sure all FIFOs are created so that their depth can be
        # set by ApplyConfig, so create_shallow_fifos=True
        model = model.transform(InsertFIFO(create_shallow_fifos=True))
        model = model.transform(SpecializeLayers(cfg._resolve_fpga_part()))
        model = model.transform(GiveUniqueNodeNamesRecursive(prefix=parent_node))
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(ApplyFIFODepthsFromFile(cfg.fifo_config_file))
        if cfg.split_large_fifos:
            model = model.transform(SplitLargeFIFOs(max_qsrl_depth=256))
            model = model.transform(GiveUniqueNodeNamesRecursive(prefix=parent_node))
            model = model.transform(GiveReadableTensorNames())

    # after FIFOs are ready to go, call PrepareIP and HLSSynthIP again
    # this will only run for the new nodes (e.g. FIFOs and DWCs)
    # Codegen for the inserted FIFOs
    model = step_hw_codegen(model, cfg, parent_node=parent_node)
    # IP Gen for the inserted FIFOs and any remaining
    # IPs that needed to be re-gen after FIFO insertion
    model = step_hw_ipgen(model, cfg, parent_node=parent_node)
    return model


@register_build_dataflow_step()
def step_generate_hardware(
    model: ModelWrapper, cfg: DataflowBuildConfig, parent_node: str | None = None
) -> ModelWrapper:
    """Generate the hardware IP of the model. This includes generating the code, IPs and sizing the
    fifos for the model and all submodels."""
    model = model.transform(GiveUniqueNodeNamesRecursive(prefix=parent_node))
    # Recursively call this step for all subgraphs
    for node in model.get_nodes_by_op_type("FINNLoop"):
        node_inst = cast("FINNLoop", getCustomOp(node))
        loop_model = cast("ModelWrapper", node_inst.get_nodeattr("body"))
        loop_model.set_metadata_prop("parent_node", node.name)
        loop_model.set_metadata_prop("is_mlo", "1")
        # Recursion here
        loop_model = step_generate_hardware(loop_model, cfg, parent_node=node.name)

        node_inst.set_nodeattr("body", loop_model.graph)

    # Codegen for the current model
    model = step_hw_codegen(model, cfg, parent_node=parent_node)

    # Stitch submodels
    for node in model.get_nodes_by_op_type("FINNLoop"):
        node_inst = cast("FINNLoop", getCustomOp(node))
        loop_model = cast("ModelWrapper", node_inst.get_nodeattr("body"))
        # Pack subgraph with IPs and FIFOs into stitched IP
        loop_model = loop_model.transform(
            CreateStitchedIP(
                cfg._resolve_fpga_part(),
                cfg.synth_clk_period_ns,
                vitis=False,
            )
        )
        node_inst.set_nodeattr("body", loop_model.graph)
    # IP Gen for the current model
    model = step_hw_ipgen(model, cfg, parent_node=parent_node)

    # FIFO sizing for the current model
    model = step_set_fifo_depths(model, cfg, parent_node=parent_node)

    return model


@register_build_dataflow_step()
def step_qonnx_to_finn(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Step will only execute if QONNX nodes are found.
    These include the following op_types: "Quant" , "Trunc" and "BinaryQuant".
    If such nodes are found the step will run the tidy-up step from QONNX
    and then convert the QONNX model to the FINN-ONNX dialect.
    """
    # Check if any QONNX nodes exist, i.e. BipolarQuant, BinaryQuant, Quant or Trunc
    q_count = 0
    for op_type in ["BipolarQuant", "BinaryQuant", "Quant", "Trunc"]:
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


@register_build_dataflow_step()
def step_tidy_up(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Run the tidy-up step on given model. This includes shape and datatype
    inference, constant folding, and giving nodes and tensors better names.
    """
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNamesRecursive())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())

    if VerificationStepType.TIDY_UP_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "initial_python", need_parent=False)

    return model


@register_build_dataflow_step()
def step_streamline(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
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


@register_build_dataflow_step()
def step_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Convert eligible nodes to `HWCustomOp` subclasses that represent HW
    layers. Which nodes and particular configurations can be converted to HW
    is limited, see the source code of the `convert_to_hw` module for more.
    In the end an empty json file is created which can be used to set user specific
    preferred implementation styles for each node."""

    # Helper function to conditionally apply transformation
    def apply_if_relevant(
        model: ModelWrapper, op_types: list[str], transform: Transformation, desc: str = ""
    ) -> ModelWrapper:
        """Apply a transform only if relevant op types exist in the model."""
        # Check if any of the relevant op types exist in the model
        if any(len(model.get_nodes_by_op_type(op_type)) > 0 for op_type in op_types):
            if desc:
                print(f"Converting {desc}...")
            model = model.transform(transform)
        return model

    # Thresholding layers (standalone mode)
    if cfg.standalone_thresholds:
        # Doing this first causes all threshold layers to be standalone
        model = apply_if_relevant(
            model,
            ["MultiThreshold"],
            to_hw.InferThresholdingLayer(),
            "threshold layers (standalone)",
        )
    else:
        log.warning(
            """standalone_thresholds are set to False.
            Please be aware that this means the MVAUs might be implemented in HLS
            because the RTL variant doesn't support the merge of
            MatMul + MultiThreshold into one layer. If you would like to have the RTL variant,
            please set standalone_thresholds to True."""
        )

    # Matrix-vector operations
    model = apply_if_relevant(
        model,
        ["XnorPopcountMatMul"],
        to_hw.InferBinaryMatrixVectorActivation(),
        "binary matmul layers",
    )
    model = apply_if_relevant(
        model, ["MatMul"], to_hw.InferQuantizedMatrixVectorActivation(), "quantized matmul layers"
    )
    model = apply_if_relevant(
        model, ["MatMul"], to_hw.InferVectorVectorActivation(), "vector-vector activation"
    )

    # Classification/output layers
    model = apply_if_relevant(model, ["TopK"], to_hw.InferLabelSelectLayer(), "label select layers")

    # Input quantization (if any) as standalone threshold
    model = apply_if_relevant(
        model, ["MultiThreshold"], to_hw.InferThresholdingLayer(), "threshold layers"
    )
    model = apply_if_relevant(model, ["Pad"], to_hw.InferFMPadding(), "padding layers")

    # Convolution-related transformations
    model = apply_if_relevant(
        model,
        ["MaxPool", "AveragePool", "MaxPoolNHWC", "QuantAvgPool2d"],
        to_hw.InferPool(),
        "pooling layers",
    )
    model = apply_if_relevant(
        model,
        ["ReduceMax", "ReduceSum", "ReduceMean"],
        to_hw.InferPoolFromReduce(),
        "reduce layers",
    )
    model = apply_if_relevant(model, ["Im2Col"], to_hw.InferConvInpGen(), "conv input generator")
    # If ConvInpGen derived, run remove cnv to fc flatten transform
    model = apply_if_relevant(
        model, ["ConvolutionInputGenerator"], RemoveCNVtoFCFlatten(), "Flatten"
    )

    # Streaming operations
    model = apply_if_relevant(model, ["Concat"], to_hw.InferConcatLayer(), "concat layers")
    model = apply_if_relevant(model, ["Split"], to_hw.InferSplitLayer(), "split layers")

    # Elementwise operations
    model = apply_if_relevant(
        model,
        [
            "Mul",
            "Div",
            "Sub",
            "Add",
            "And",
            "Or",
            "Xor",
            "Equal",
            "Less",
            "LessOrEqual",
            "Greater",
            "GreaterOrEqual",
        ],
        to_hw.InferElementwiseBinaryOperation(),
        "elementwise binary operations",
    )
    model = apply_if_relevant(
        model, ["Relu"], to_hw.InferReLUAsElementwiseMax(), "ReLU as elementwise max"
    )

    # Upsampling and resizing
    model = apply_if_relevant(model, ["Upsample"], to_hw.InferUpsample(), "upsample layers")

    # Global pooling
    model = apply_if_relevant(
        model, ["GlobalAveragePool"], to_hw.InferGlobalAccPoolLayer(), "global pooling"
    )

    # Lookup layers
    model = apply_if_relevant(model, ["Gather"], to_hw.InferLookupLayer(), "lookup layers")

    # Activation functions
    model = apply_if_relevant(model, ["Softmax"], to_hw.InferHWSoftmax(), "softmax layers")

    # Normalization layers
    model = apply_if_relevant(
        model, ["LayerNormalization"], to_hw.InferLayerNorm(), "layer normalization"
    )

    # Cropping layers
    model = apply_if_relevant(model, ["Crop"], to_hw.InferCrop(), "crop layers")

    # Quantization layers (Quant nodes with scale=1, zeropt=0 or uniform MultiThreshold)
    model = apply_if_relevant(
        model, ["Quant", "MultiThreshold"], to_hw.InferRequantLayer(), "quantization as requant"
    )

    # Graph topology transformations (always check - not based on op_type)
    # DuplicateStreams: detects forks where tensors have multiple consumers
    print("Checking for graph forks (duplicate streams)...")
    model = model.transform(to_hw.InferDuplicateStreamsLayer())

    # Cleanup and post-processing transformations
    # Get rid of Transpose -> Transpose identity sequences
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(GiveUniqueNodeNamesRecursive())
    model = model.transform(InferDataLayouts())
    model = model.transform(InferDataTypes())
    model = model.transform(InferShapes())
    model = model.transform(to_hw.InferReshape())

    # Shuffle inference (should come after InferDataLayouts and handles Transpose+Reshape patterns)
    # InferShuffle skips first Transpose by default; override to convert all if disabled
    if cfg.infer_shuffle_skip_first:
        model = apply_if_relevant(
            model, ["Transpose"], to_hw.InferShuffle(), "shuffle/transpose layers"
        )
    else:
        model = apply_if_relevant(
            model,
            ["Transpose"],
            to_hw.InferShuffle(_filter=lambda *_: True),
            "shuffle/transpose layers",
        )
    return model


@register_build_dataflow_step()
def step_create_dataflow_partition(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Separate consecutive groups of HWCustomOp nodes into StreamingDataflowPartition
    nodes, which point to a separate ONNX file. Dataflow accelerator synthesis
    can only be performed on those HWCustomOp sub-graphs."""
    unmapped_layers = [
        node.name
        for node in model.graph.node
        if not node.domain.startswith("finn.custom_op.fpgadataflow")
    ]
    # Check if there are unsupported layers somewhere between supported layers
    # This would cause a "cyclic-free graph partitioning violated" error otherwise
    results = model.analysis(unsupported_layers)
    if results[0] is False:
        raise FINNUserError(
            f"Unsupported/unmapped layer(s) found in between FINN operators, "
            f"starting with node {results[1].name}. "
            "Complete list of unmapped nodes: " + ", ".join(unmapped_layers)
        )

    # Warn if unsupported layers remain at the start or end of the graph
    if unmapped_layers:
        log.warning(
            "The following nodes at the start/end of the graph will not be mapped to the "
            "accelerator, so they will need to be implemented manually (e.g., in software): "
            + ", ".join(unmapped_layers)
        )

    parent_model = model.transform(
        CreateDataflowPartition(
            partition_model_dir=str(cfg.output_dir) + "/intermediate_models/supported_op_partitions"
        )
    )
    sdp_nodes = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")
    assert len(sdp_nodes) == 1, "Only a single StreamingDataflowPartition supported."
    sdp_node = sdp_nodes[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = cast("str", sdp_node.get_nodeattr("model"))
    if cfg.save_intermediate_models:
        parent_model.save(str(cfg.output_dir) + "/intermediate_models/dataflow_parent.onnx")
    model = ModelWrapper(dataflow_model_filename)

    # create a configuration json file that can be used to set the specialize layer config
    attrs = [
        "preferred_impl_style",
    ]
    extract_model_config_to_json(
        model, Path(cfg.output_dir) / "template_specialize_layers_config.json", attrs
    )

    return model


@register_build_dataflow_step()
def step_specialize_layers(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Convert HW nodes to either an HLS or RTL variant of the node. HW nodes
    get converted either based on pre-determined rules (details can be found
    in `specialize_layers` source code) or the user provides a configuration file
    which contains the desired setting. If the user preference cannot be fulfilled,
    a warning will be printed and the implementation style will be set to a default."""
    if cfg.specialize_layers_config_file is not None:
        model = model.transform(GiveUniqueNodeNamesRecursive())
        model = model.transform(ApplyConfig(cfg.specialize_layers_config_file))
    model = model.transform(SpecializeLayers(cfg._resolve_fpga_part()))
    model = model.transform(GiveUniqueNodeNamesRecursive())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


@register_build_dataflow_step()
def step_transpose_decomposition(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Decomposes a Shuffle into a chain of InnerShuffle and OuterShuffles that
    can be specialised into hardware operators.
    This should be executed after the folding has been configured.
    """
    # check if model contains a Shuffle node
    has_shuffle = bool(model.get_nodes_by_op_type("Shuffle"))
    loop_nodes = model.get_nodes_by_op_type("FINNLoop")
    for node in loop_nodes:
        node_inst = cast("FINNLoop", getCustomOp(node))
        loop_model = cast("ModelWrapper", node_inst.get_nodeattr("body"))
        has_shuffle = bool(loop_model.get_nodes_by_op_type("Shuffle"))

    if has_shuffle:
        model = model.transform(ShuffleDecomposition(), apply_to_subgraphs=True)
        model = model.transform(InferInnerOuterShuffles(), apply_to_subgraphs=True)
        model = model.transform(SpecializeLayers(cfg._resolve_fpga_part()), apply_to_subgraphs=True)
        model = model.transform(InferShapes(), apply_to_subgraphs=True)
        model = model.transform(InferDataTypes(), apply_to_subgraphs=True)
        model = model.transform(GiveUniqueNodeNamesRecursive())
        loop_nodes = model.get_nodes_by_op_type("FINNLoop")
        for node in loop_nodes:
            node_inst = cast("FINNLoop", getCustomOp(node))
            loop_model = cast("ModelWrapper", node_inst.get_nodeattr("body"))
            loop_model = loop_model.transform(GiveUniqueNodeNamesRecursive(prefix=node.name))
            node_inst.set_nodeattr("body", loop_model.graph)
    else:
        log.info("Model doesn't contain any Shuffle nodes, skipping step_transpose_decomposition.")
    return model


@register_build_dataflow_step()
def step_target_fps_parallelization(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
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
            ),
            apply_to_subgraphs=True,
        )
        model = model.transform(GiveUniqueNodeNamesRecursive())
        # extract the suggested configuration and save it as json
        hw_attrs = [
            "PE",
            "SIMD",
            "EmbFold",
            "SeqFold",
            "parallel_window",
            "ram_style",
            "ram_style_thresholds",
            "ram_style_mask",
            "depth",
            "impl_style",
            "resType",
            "mac_resource",
            "mem_mode",
            "runtime_writeable_weights",
            "depth_trigger_uram",
            "depth_trigger_bram",
        ]
        extract_model_config_to_json(
            model, Path(cfg.output_dir) / "report" / "auto_folding_config.json", hw_attrs
        )

    else:
        log.warning("No target_fps provided, skipping step_target_fps_parallelization.")

    return model


@register_build_dataflow_step()
def step_apply_folding_config(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Apply the folding configuration file onto the model to set folding (parallelization)
    and other attributes, if config file is specified."""
    model = model.transform(GiveUniqueNodeNamesRecursive())
    if cfg.folding_config_file is not None:
        model = model.transform(ApplyConfig(cfg.folding_config_file), apply_to_subgraphs=True)
    else:
        log.info("No folding config json provided, skipping step_apply_folding_config.")

    return model


@register_build_dataflow_step()
def step_generate_estimate_reports(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Generate per-layer resource and cycle estimates using analytical models."""
    if DataflowOutputType.ESTIMATE_REPORTS in cfg.generate_outputs:
        report_dir = Path(cfg.output_dir) / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        ops_and_params = model.analysis(op_and_param_counts)
        with (report_dir / "op_and_param_counts.json").open("w") as f:
            json.dump(ops_and_params, f, indent=2)
        estimate_layer_cycles = model.analysis(exp_cycles_per_layer)
        with (report_dir / "estimate_layer_cycles.json").open("w") as f:
            json.dump(estimate_layer_cycles, f, indent=2)
        estimate_layer_resources: dict[str, dict[str, int | float]] = model.analysis(
            partial(res_estimation, fpgapart=cfg._resolve_fpga_part())
        )
        estimate_layer_resources["total"] = aggregate_dict_keys(estimate_layer_resources)
        with (report_dir / "estimate_layer_resources.json").open("w") as f:
            json.dump(estimate_layer_resources, f, indent=2)
        estimate_layer_resources_complete = model.analysis(
            partial(res_estimation_complete, fpgapart=cfg._resolve_fpga_part())
        )
        with (report_dir / "estimate_layer_config_alternatives.json").open("w") as f:
            json.dump(estimate_layer_resources_complete, f, indent=2)

        # generate reports for MLO nodes
        loop_nodes = model.get_nodes_by_op_type("FINNLoop")
        for node in loop_nodes:
            node_inst = cast("FINNLoop", getCustomOp(node))
            loop_model = cast("ModelWrapper", node_inst.get_nodeattr("body"))
            ops_and_params = loop_model.analysis(op_and_param_counts)
            with (report_dir / f"op_and_param_counts_{node.name}.json").open("w") as f:
                json.dump(ops_and_params, f, indent=2)
            estimate_layer_cycles = loop_model.analysis(exp_cycles_per_layer)
            with (report_dir / f"estimate_layer_cycles_{node.name}.json").open("w") as f:
                json.dump(estimate_layer_cycles, f, indent=2)
            estimate_layer_resources = loop_model.analysis(
                partial(res_estimation, fpgapart=cfg._resolve_fpga_part())
            )
            estimate_layer_resources["total"] = aggregate_dict_keys(estimate_layer_resources)
            with (report_dir / f"estimate_layer_resources_{node.name}.json").open("w") as f:
                json.dump(estimate_layer_resources, f, indent=2)
            estimate_layer_resources_complete = loop_model.analysis(
                partial(res_estimation_complete, fpgapart=cfg._resolve_fpga_part())
            )
            with (report_dir / f"estimate_layer_config_alternatives_{node.name}.json").open(
                "w"
            ) as f:
                json.dump(estimate_layer_resources_complete, f, indent=2)

        if not is_mlo(model):
            # need to call AnnotateCycles before dataflow_performance
            model = model.transform(AnnotateCycles())
            estimate_network_performance: dict[str, str | int | float] = dict(
                model.analysis(dataflow_performance)
            )
            # add some more metrics to estimated performance
            n_clock_cycles_per_sec = (10**9) / cfg.synth_clk_period_ns
            est_fps = n_clock_cycles_per_sec / cast(
                "int", estimate_network_performance["max_cycles"]
            )
            estimate_network_performance["estimated_throughput_fps"] = est_fps
            est_latency_ns = (
                cast("int", estimate_network_performance["critical_path_cycles"])
                * cfg.synth_clk_period_ns
            )
            estimate_network_performance["estimated_latency_ns"] = est_latency_ns
            with (report_dir / "estimate_network_performance.json").open("w") as f:
                json.dump(estimate_network_performance, f, indent=2)
        else:
            log.warning(
                "Model contains MLO, currently network performance can't be estimated for this."
            )
    else:
        log.info(
            """DataflowOutputType.ESTIMATE_REPORTS not in requested outputs,
            skipping step_generate_estimate_reports."""
        )
    return model


@register_build_dataflow_step()
def step_minimize_bit_width(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Tighten the weight and accumulator bit widths for each layer."""
    if cfg.minimize_bit_width:
        model = model.transform(MinimizeWeightBitWidth(), apply_to_subgraphs=True)
        model = model.transform(MinimizeAccumulatorWidth(), apply_to_subgraphs=True)
        # make sure the changed datatypes are propagated through the network
        model = model.transform(InferDataTypes(), apply_to_subgraphs=True)
    else:
        log.info("minimize_bit_width set to False, only run RoundAndClipThresholds.")
    # Always run RoundAndClipThresholds after accumulator widths are determined
    model = model.transform(RoundAndClipThresholds(), apply_to_subgraphs=True)
    model = model.transform(InferDataTypes(), apply_to_subgraphs=True)
    # Run MinimizeWeightBitWidth again to minimize threshold datatypes after rounding/clipping
    if cfg.minimize_bit_width:
        model = model.transform(MinimizeWeightBitWidth(), apply_to_subgraphs=True)
        model = model.transform(InferDataTypes(), apply_to_subgraphs=True)

    if VerificationStepType.FOLDED_HLS_CPPSIM in cfg._resolve_verification_steps():
        # prepare cppsim
        model = model.transform(PrepareCppSim(), apply_to_subgraphs=True)
        model = model.transform(CompileCppSim(), apply_to_subgraphs=True)
        model = model.transform(SetExecMode("cppsim"), apply_to_subgraphs=True)
        # Set iteration context path on FINNLoop nodes if verify_save_full_context is enabled
        if cfg.verify_save_full_context:
            verify_out_dir = Path(cfg.output_dir) / "verification_output"
            verify_out_dir.mkdir(parents=True, exist_ok=True)
            for loop_node in model.get_nodes_by_op_type("FINNLoop"):
                loop_inst = getCustomOp(loop_node)
                ctx_path = (
                    verify_out_dir / f"iteration_context_{loop_node.name}_folded_hls_cppsim.npz"
                )
                loop_inst.set_nodeattr("iteration_context_path", str(ctx_path))
        verify_step(model, cfg, "folded_hls_cppsim", need_parent=True)
        # Clear iteration_context_path after verification
        if cfg.verify_save_full_context:
            for loop_node in model.get_nodes_by_op_type("FINNLoop"):
                loop_inst = getCustomOp(loop_node)
                loop_inst.set_nodeattr("iteration_context_path", "")

    return model


def step_insert_dwc(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Insert data width converters between layers where necessary."""
    model = model.transform(InsertDWC())
    return model.transform(SpecializeLayers(cfg._resolve_fpga_part()))


def verify_mlo(model: ModelWrapper, cfg: DataflowBuildConfig, step: str) -> None:  # noqa: ARG001
    """Verify a multi-layer offload model via RTL simulation."""
    finn_loop = model.get_nodes_by_op_type("FINNLoop")
    # TODO: allow for multiple FINNLoops
    mlo_prehook = mlo_prehook_func_factory(finn_loop[0])
    verify_step(model, cfg, "stitched_ip_rtlsim", need_parent=False, rtlsim_pre_hook=mlo_prehook)


@register_build_dataflow_step()
def step_create_stitched_ip(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Create stitched IP for a graph after all HLS IP blocks have been generated.
    Depends on the DataflowOutputType.STITCHED_IP output product."""
    # introduce tLAST marker, required for instrumentation
    if cfg.enable_instrumentation:
        if cfg.shell_flow_type == ShellFlowType.VITIS_ALVEO:
            raise FINNUserError("Instrumentation is not yet implemented for Alveo/Vitis flow")
        model = model.transform(
            InsertTLastMarker(
                # only insert marker on output (input TLAST is ignored for these use-cases anyway)
                both=False,
                # use ap_axiu instead of qdma_axis
                external=False,
                # static number of iterations (based on what the compiler/folding sets up)
                dynamic=False,
            )
        )
        # give a proper name to the inserted node, important for codegen
        # TODO: deal with multi-I/O accelerators?
        model.graph.node[-1].name = "TLastMarker_0"
        # re-run codegen and HLS IP gen, will affect only the new TLastMarker layer assuming
        # all other IPs have been generated already
        model = model.transform(PrepareIP(cfg._resolve_fpga_part(), cfg._resolve_hls_clk_period()))
        model = model.transform(HLSSynthIP())

    if DataflowOutputType.STITCHED_IP in cast("list[DataflowOutputType]", cfg.generate_outputs):
        report_dir = Path(cfg.output_dir) / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        stitched_ip_dir = Path(cfg.output_dir) / "stitched_ip"
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
            cast("str", model.get_metadata_prop("vivado_stitch_proj")),
            stitched_ip_dir,
            dirs_exist_ok=True,
        )
        log.info(f"Vivado stitched IP written into {stitched_ip_dir}")

        if cfg.stitched_ip_gen_dcp:
            copy(
                cast("str", model.get_metadata_prop("vivado_synth_rpt")),
                report_dir / "post_synth_resources_dcp.xml",
            )
            post_synth_resources = model.analysis(post_synth_res)
            with (report_dir / "post_synth_resources_dcp.json").open("w") as f:
                json.dump(post_synth_resources, f, indent=2)

    else:
        log.warning(
            """DataflowOutputType.STITCHED_IP not in requested outputs,
            skipping step_create_stitched_ip."""
        )
    if VerificationStepType.STITCHED_IP_RTLSIM in cfg._resolve_verification_steps():
        # prepare ip-stitched rtlsim
        verify_model = deepcopy(model)
        verify_model.set_metadata_prop("exec_mode", "rtlsim")

        # Use critical path estimate to set rtlsim liveness threshold
        # TODO: This is a heuristic which usually overestimates the maximum
        #  cycles (by a lot), but can actually also underestimate causing
        #  incorrect detection of timeouts. In these cases, this estimation can
        #  be overwritten by setting LIVENESS_THRESHOLD to a very large value.
        verify_model = verify_model.transform(AnnotateCycles())
        liveness = get_liveness_threshold_cycles()
        perf = verify_model.analysis(dataflow_performance)
        latency = cast("int", perf["critical_path_cycles"])
        max_iters = max(liveness, int(np.ceil(latency * 1.1 + 20)))
        os.environ["LIVENESS_THRESHOLD"] = str(max_iters)

        if cfg.verify_save_rtlsim_waveforms:
            verify_out_dir = Path(cfg.output_dir) / "verification_output"
            waveform_dir = verify_out_dir / "stitched_ip_rtlsim_waveforms"
            waveform_dir.mkdir(parents=True, exist_ok=True)
            abspath = waveform_dir.absolute()
            verify_model.set_metadata_prop("rtlsim_trace", str(abspath / "verify_rtlsim.wdb"))
        if is_mlo(model):
            verify_mlo(verify_model, cfg, "stitched_ip_rtlsim")
        else:
            verify_step(verify_model, cfg, "stitched_ip_rtlsim", need_parent=True)
        os.environ["LIVENESS_THRESHOLD"] = str(liveness)
    return model


@register_build_dataflow_step()
def step_measure_rtlsim_performance(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Measure performance + latency of stitched-IP model in rtlsim (xsi)."""
    report_dir = Path(cfg.output_dir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    orig_rtlsim_trace_depth = get_rtlsim_trace_depth()

    if cfg.verify_save_rtlsim_waveforms:
        # set depth to 3 for layer-by-layer visibility
        os.environ["RTLSIM_TRACE_DEPTH"] = "3"
        model.set_metadata_prop("rtlsim_trace", str(report_dir.resolve() / "rtlsim_perf_trace.wdb"))

    if not cfg.auto_fifo_depths and cfg.fifo_config_file is not None:
        # Use critical path estimate to set the timeout limit for FIFO sim
        model = model.transform(AnnotateCycles())
        perf = model.analysis(dataflow_performance)
        latency = cast("int", perf["critical_path_cycles"])
        max_iters = latency * 100
    else:
        max_iters = (
            None  # Auto FIFO depths are garanteed to prevent deadlock, no need for a timeout
        )
    # prepare simulation
    sim = NodeConnectedSimulation(
        model,
        SimulationType.NODE_BASED_CONNECTED,
        cfg._resolve_fpga_part(),
        cfg._resolve_hls_clk_period(),
        cfg.functional_simulation,
        max_qsrl_depth=256,
        performance_sim=True,
        shm_prefix=None,
    )

    nodes = [node for node in model.graph.node if "FIFO" not in node.op_type]
    num_nodes = len(nodes)
    fifo_depth: list[list[int]] = [[]] * num_nodes

    for i, node in enumerate(nodes):
        hwnode = getHWCustomOp(node)
        fifos = cast("list[int]", hwnode.get_nodeattr("outFIFODepths"))
        num_successors = len(node.output)
        if num_successors != len(fifos):
            raise FINNUserError(
                f"Number of successors ({num_successors}) doesn't match number of FIFO depths "
                f"({len(fifos)}) for node {node.name}. "
                f"Did you run the FIFO sizing step or supplied a valid FIFO config for the model?"
            )
        if fifos[0] == 2:
            log.warning(
                f"Node {node.name} has FIFO depth of 2, which is the default unconfigured depth. "
                "This might lead to deadlock in the simulation. "
                "Please run the FIFO sizing step or supply a valid FIFO config for the model."
            )
        fifo_depth[i] = fifos

    results = sim.simulate(fifo_depth, max_cycles=max_iters)
    outputs: list[dict] = []
    for res in results[0]:
        if res["samples"] != 0:
            # Cleanup of Output
            del res["fifo_utilization"]
            del res["fifo_depth"]
            del res["fifo_cycles_until_first_valid"]
            cycle_per_sec = 1e9 / cfg.synth_clk_period_ns
            res["throughput_fps"] = cycle_per_sec / res["intervals"][0]  # type: ignore
            # Attach entry to output
            outputs.append(res)

    rtl_sim_perf_dir = report_dir / "rtlsim_performance.json"
    with rtl_sim_perf_dir.open("w") as f:
        json.dump(outputs, f, indent=2)
    if cfg.verify_save_rtlsim_waveforms:
        # restore original trace depth
        os.environ["RTLSIM_TRACE_DEPTH"] = str(orig_rtlsim_trace_depth)

    log.info(f"RTLSim performance results written into {rtl_sim_perf_dir}")

    return model


@register_build_dataflow_step()
def step_make_driver(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Create a driver that can be used to interface the generated accelerator.
    Use DataflowBuildConfig to select PYNQ Python or C++ driver."""
    driver_dir = Path(cfg.output_dir) / "driver"
    if DataflowOutputType.PYNQ_DRIVER in cfg.generate_outputs:
        # determine drivertype
        if cfg.enable_instrumentation:
            driver_type = "FINNDMAInstrumentationOverlay"
            if cfg.instrumentation_no_dma:
                driver_type = "FINNInstrumentationOverlay"
            if cfg.auto_fifo_strategy == AutoFIFOSizingMethod.LIVE_FIFO and cfg.auto_fifo_depths:
                driver_type = "FINNLiveFIFOOverlay"
        else:
            driver_type = "FINNDMAOverlay"

        experiment_info = cfg.experiments_config_path

        model = model.transform(
            MakePYNQDriver(
                cfg._resolve_driver_platform(),
                driver_type,
                clk_period_ns=cfg.synth_clk_period_ns,
                validation_datset=cfg.validation_dataset,
                experiment_info=experiment_info,
                board=cfg.board,
            )
        )

        shutil.copytree(
            cast("str", model.get_metadata_prop("pynq_driver_dir")), driver_dir, dirs_exist_ok=True
        )
        log.info("PYNQ Python driver written into " + str(driver_dir))
    elif DataflowOutputType.CPP_DRIVER in cfg.generate_outputs:
        # generate C++ Driver
        model = model.transform(
            MakeCPPDriver(
                cfg._resolve_driver_platform(),
                version=cfg.cpp_driver_version,
                host_mem=cfg.fpga_memory,
            )
        )
        shutil.copytree(
            cast("str", model.get_metadata_prop("cpp_driver_dir")),
            driver_dir,
            dirs_exist_ok=True,
            copy_function=shutil.copyfile,
        )

        log.info("C++ driver written into " + str(driver_dir))
    else:
        log.warning(
            """Neither DataflowOutputType.PYNQ_DRIVER nor DataflowOutputType.CPP_DRIVER
            in requested outputs, skipping step_make_driver."""
        )
    return model


@register_build_dataflow_step()
def step_out_of_context_synthesis(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Run out-of-context synthesis and generate reports.
    Depends on the DataflowOutputType.STITCHED_IP output product."""
    if DataflowOutputType.OOC_SYNTH in cfg.generate_outputs:
        assert DataflowOutputType.STITCHED_IP in cfg.generate_outputs, "OOC needs stitched IP"
        model = model.transform(
            SynthOutOfContext(part=cfg._resolve_fpga_part(), clk_period_ns=cfg.synth_clk_period_ns)
        )
        report_dir = Path(cfg.output_dir) / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        ooc_res_dict = model.get_metadata_prop("res_total_ooc_synth")
        if ooc_res_dict is None:
            raise FINNUserError(
                "Out-of-context synthesis results not found in model metadata. "
                "Did the OOC synthesis step fail? Check the logs."
            )
        ooc_res_dict = eval(ooc_res_dict)

        estimate_network_performance = model.analysis(dataflow_performance)
        # add some more metrics to estimated performance
        n_clock_cycles_per_sec = float(ooc_res_dict["fmax_mhz"]) * (10**6)
        est_fps = n_clock_cycles_per_sec / cast("int", estimate_network_performance["max_cycles"])
        ooc_res_dict["estimated_throughput_fps"] = est_fps
        with (report_dir / "ooc_synth_and_timing.json").open("w") as f:
            json.dump(ooc_res_dict, f, indent=2)

    else:
        log.info(
            """DataflowOutputType.OOC_SYNTH not in requested outputs,
            skipping step_out_of_context_synthesis."""
        )
    return model


@register_build_dataflow_step()
def step_vivado_power_estimation(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Run Vivado power estimation on the stitched IP after OOC synthesis."""
    if DataflowOutputType.OOC_SYNTH not in cfg.generate_outputs:
        raise FINNUserError("Vivado power estimation needs OOC synth")

    report_dir = Path(cfg.output_dir) / "report"
    model.transform(
        VivadoPowerEstimation(
            str(report_dir),
            cfg.synth_clk_period_ns,
            cfg.vivado_power_simulate_activity,
            cfg.vivado_power_simulation_type,
        )
    )
    return model


@register_build_dataflow_step()
def step_synthesize_bitfile(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Synthesize a bitfile for the using the specified shell flow, using either
    Vivado or Vitis, to target the specified board."""
    if DataflowOutputType.BITFILE in cfg.generate_outputs:
        bitfile_dir = Path(cfg.output_dir) / "bitfile"
        bitfile_dir.mkdir(parents=True, exist_ok=True)
        report_dir = Path(cfg.output_dir) / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        partition_model_dir = Path(cfg.output_dir) / "intermediate_models" / "kernel_partitions"
        if cfg.shell_flow_type == ShellFlowType.VIVADO_ZYNQ:
            if cfg.instrumentation_no_dma is None:
                raise FINNUserError(
                    "instrumentation_no_dma must be set in the config for Vivado Zynq flow"
                )
            if cfg.board is None:
                raise FINNUserError("board must be set in the config for Vivado Zynq flow")
            model = model.transform(
                ZynqBuild(
                    cfg.board,
                    cfg.synth_clk_period_ns,
                    cfg.enable_hw_debug,
                    cfg.enable_instrumentation,
                    cfg.instrumentation_no_dma,
                    cfg.instrumentation_avg_n,
                    cfg.auto_fifo_depths
                    and cfg.auto_fifo_strategy == AutoFIFOSizingMethod.LIVE_FIFO,
                    partition_model_dir=str(partition_model_dir),
                )
            )

            bitfile_path = bitfile_dir / "finn-accel.bit"
            bitfile_src = model.get_metadata_prop("bitfile")
            if bitfile_src is None:
                raise FINNUserError(
                    "Bitfile path not found in model metadata. "
                    "Did the Vivado synthesis step fail? Check the logs."
                )
            hwh_src = model.get_metadata_prop("hw_handoff")
            if hwh_src is None:
                raise FINNUserError(
                    "HWH path not found in model metadata. "
                    "Did the Vivado synthesis step fail? Check the logs."
                )
            rpt_dir = model.get_metadata_prop("vivado_synth_rpt")
            if rpt_dir is None:
                raise FINNUserError(
                    "Vivado synthesis report path not found in model metadata. "
                    "Did the Vivado synthesis step fail? Check the logs."
                )
            copy(Path(bitfile_src), bitfile_path)
            copy(Path(hwh_src), bitfile_dir / "finn-accel.hwh")
            copy(
                Path(rpt_dir),
                report_dir / "post_synth_resources.xml",
            )

            model.set_metadata_prop("bitfile_output", str(bitfile_path.absolute()))

            post_synth_resources = model.analysis(post_synth_res)
            with (report_dir / "post_synth_resources.json").open("w") as f:
                json.dump(post_synth_resources, f, indent=2)

            vivado_pynq_proj_dir = model.get_metadata_prop("vivado_pynq_proj")
            timing_rpt = (
                Path(f"{vivado_pynq_proj_dir}")
                / "finn_zynq_link.runs"
                / "impl_1"
                / "top_wrapper_timing_summary_routed.rpt"
            )
            copy(timing_rpt, report_dir / "post_route_timing.rpt")

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

            bitfile_path = bitfile_dir / "finn-accel.xclbin"
            bitfile_src = model.get_metadata_prop("bitfile")
            if bitfile_src is None:
                raise FINNUserError(
                    "Bitfile path not found in model metadata. "
                    "Did the Vitis synthesis step fail? Check the logs."
                )
            rpt_dir = model.get_metadata_prop("vivado_synth_rpt")
            if rpt_dir is None:
                raise FINNUserError(
                    "Vivado synthesis report path not found in model metadata. "
                    "Did the Vitis synthesis step fail? Check the logs."
                )
            copy(Path(bitfile_src), bitfile_path)
            copy(
                Path(rpt_dir),
                report_dir / "post_synth_resources.xml",
            )

            model.set_metadata_prop("bitfile_output", str(bitfile_path.absolute()))

            post_synth_resources = model.analysis(post_synth_res)
            with (report_dir / "post_synth_resources.json").open("w") as f:
                json.dump(post_synth_resources, f, indent=2)
        else:
            raise Exception("Unrecognized shell_flow_type: " + str(cfg.shell_flow_type))
        log.info(f"Bitfile written into {bitfile_dir}")

    else:
        log.info(
            "DataflowOutputType.BITFILE not in requested outputs, skipping step_synthesize_bitfile."
        )

    return model


@register_build_dataflow_step()
def step_deployment_package(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Create a deployment package including the driver and bitfile."""
    if DataflowOutputType.DEPLOYMENT_PACKAGE in cfg.generate_outputs:
        deploy_dir = Path(cfg.output_dir) / "deploy"
        bitfile_dir = Path(cfg.output_dir) / "bitfile"
        driver_dir = Path(cfg.output_dir) / "driver"
        deploy_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bitfile_dir, deploy_dir / "bitfile", dirs_exist_ok=True)
        shutil.copytree(
            driver_dir, deploy_dir / "driver", dirs_exist_ok=True, copy_function=shutil.copyfile
        )
        if DataflowOutputType.CPP_DRIVER in cfg.generate_outputs:
            update_bitfile_path_after_copy(
                deploy_dir / "bitfile" / "finn-accel.xclbin",
                deploy_dir / "driver" / "acceleratorconfig.json",
            )

    else:
        log.info(
            """DataflowOutputType.DEPLOYMENT_PACKAGE not in requested outputs,
            skipping step_deployment_package."""
        )
    return model


@register_build_dataflow_step()
def step_loop_rolling(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Roll a repeating sequence of layers into a loop. PyTorch metadata node hierarchy
    is used to indicate the loop structure."""
    if cfg.mlo:
        if (
            cfg.loop_body_range is not None
            and isinstance(cfg.loop_body_range[0], NodeProto)
            and isinstance(cfg.loop_body_range[1], NodeProto)
        ):
            # set node metadata like loop rolling would expect
            node_metadata = {
                "pkg.torch.onnx.name_scopes": "['', 'layers.0']",
                "pkg.torch.onnx.class_hierarchy": "['TestModule', 'test']",
            }
            model = model.transform(SetLoopBoundary(node_metadata, cfg.loop_body_range))
        else:
            log.warning(
                """MLO is selected but no loop range for the subgraph is specified,
                this might cause an error during loop rolling."""
            )
        if cfg.loop_body_hierarchy is not None:
            log.info(f"Running Loop Rolling on {cfg.loop_body_hierarchy} hierarchy")
            loop_extraction = LoopExtraction(cfg.loop_body_hierarchy)
            model = model.transform(loop_extraction)
            model = model.transform(LoopRolling(loop_extraction.loop_body_template))
            move("loop-body-template.onnx", Path(cfg.output_dir) / "loop-body-template.onnx")
    else:
        log.info("MLO not selected, skipping step_loop_rolling.")

    return model


#: register imported step functions (local steps are registered via decorators)
register_build_dataflow_step()(step_passes_frontend)

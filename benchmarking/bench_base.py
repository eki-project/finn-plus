import itertools
import os
import subprocess
import copy
import json
import time
import traceback
import glob
import shutil
import numpy as np
from shutil import copy as shcopy
from shutil import copytree
import finn.core.onnx_exec as oxe
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.transformation.fpgadataflow.make_zynq_proj import collect_ip_dirs
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir, pynq_native_port_width, part_map, alveo_default_platform, alveo_part_map
from templates import template_open, template_single_test, template_sim_power, template_switching_simulation_tb, zynq_harness_template
from util import summarize_table, summarize_section, power_xml_to_dict, prepare_inputs, delete_dir_contents
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from qonnx.util.basic import (
    gen_finn_dt_tensor,
    roundup_to_integer_multiple,
)
import finn.builder.build_dataflow as build
from finn.analysis.fpgadataflow.post_synth_res import post_synth_res
from qonnx.core.modelwrapper import ModelWrapper
from finn.builder.build_dataflow_config import DataflowBuildConfig
import pandas as pd
import onnxruntime as ort


def start_test_batch_fast(results_path, project_path, run_target, pairs):
    # Prepare tcl script
    script = template_open.replace("$PROJ_PATH$", project_path)
    # script = script.replace("$PERIOD$", period)
    script = script.replace("$RUN$", run_target)
    for toggle_rate, static_prob in pairs:
        script = script + template_single_test
        script = script.replace("$TOGGLE_RATE$", str(toggle_rate))
        script = script.replace("$STATIC_PROB$", str(static_prob))
        # script = script.replace("$SWITCH_TARGET$", switch_target)
        script = script.replace("$REPORT_PATH$", results_path)
        script = script.replace("$REPORT_NAME$", f"{toggle_rate}_{static_prob}")
    with open(os.getcwd() + "/power_report.tcl", "w") as tcl_file:
        tcl_file.write(script)

    # Prepare bash script
    bash_script = os.getcwd() + "/report_power.sh"
    with open(bash_script, "w") as script:
        script.write("#!/bin/bash \n")
        script.write(f"vivado -mode batch -source {os.getcwd()}/power_report.tcl\n")

    # Run script
    sub_proc = subprocess.Popen(["bash", bash_script])
    sub_proc.communicate()

    # Parse results
    for toggle_rate, static_prob in pairs:
        power_report_dict = power_xml_to_dict(f"{results_path}/{toggle_rate}_{static_prob}.xml")
        power_report_json = f"{results_path}/{toggle_rate}_{static_prob}.json"
        with open(power_report_json, "w") as json_file:
            json_file.write(json.dumps(power_report_dict, indent=2))


def sim_power_report(results_path, project_path, in_width, out_width, dtype_width, sim_duration_ns):
    # Prepare tcl script
    script = template_open.replace("$PROJ_PATH$", project_path)
    script = script.replace("$RUN$", "impl_1")
    script = script + template_sim_power
    script = script.replace("$TB_FILE_PATH$", os.getcwd() + "/switching_simulation_tb.v")
    script = script.replace("$SAIF_FILE_PATH$", os.getcwd() + "/switching.saif")
    script = script.replace("$SIM_DURATION_NS$", str(int(sim_duration_ns)))
    script = script.replace("$REPORT_PATH$", results_path)
    script = script.replace("$REPORT_NAME$", f"sim")
    with open(os.getcwd() + "/power_report.tcl", "w") as tcl_file:
        tcl_file.write(script)

    # Prepare testbench
    testbench = template_switching_simulation_tb.replace("$INSTREAM_WIDTH$", str(in_width))
    testbench = testbench.replace("$OUTSTREAM_WIDTH$", str(out_width))
    testbench = testbench.replace("$DTYPE_WIDTH$", str(dtype_width))
    testbench = testbench.replace(
        "$RANDOM_FUNCTION$", "$urandom_range(0, {max})".format(max=2**dtype_width - 1)
    )
    with open(os.getcwd() + "/switching_simulation_tb.v", "w") as tb_file:
        tb_file.write(testbench)

    # Prepare shell script
    bash_script = os.getcwd() + "/report_power.sh"
    with open(bash_script, "w") as script:
        script.write("#!/bin/bash \n")
        script.write(f"vivado -mode batch -source {os.getcwd()}/power_report.tcl\n")

    # Run script
    sub_proc = subprocess.Popen(["bash", bash_script])
    sub_proc.communicate()

    # Parse results
    power_report_dict = power_xml_to_dict(f"{results_path}/sim.xml")
    power_report_json = f"{results_path}/sim.json"
    with open(power_report_json, "w") as json_file:
        json_file.write(json.dumps(power_report_dict, indent=2))

class bench():
    def __init__(self, params, task_id, run_id, artifacts_dir, save_dir, debug=True):
        super().__init__()
        self.params = params
        self.task_id = task_id
        self.run_id = run_id
        self.artifacts_dir = artifacts_dir
        self.save_dir = save_dir
        self.debug = debug

        #TODO: setup a logger so output can go to console (with task id prefix) and log simultaneously
        #TODO: coordinate with new builder loggin setup

        # General configuration
        # TODO: do not allow multiple targets in a single bench job due to measurement?
        if "board" in params:
            self.board = params["board"]
        else:
            self.board = "RFSoC2x2"
            self.params["board"] = self.board

        if "part" in params:
            self.part = params["part"]
        elif self.board in part_map:
            self.part = part_map[self.board]
        else:
            raise Exception("No part specified for board %s" % self.board)

        if "clock_period_ns" in params:
            self.clock_period_ns = params["clock_period_ns"]
        else:
            self.clock_period_ns = 10
            self.params["clock_period_ns"] = self.clock_period_ns

        # Clear FINN tmp build dir before every run (to avoid excessive ramdisk usage and duplicate debug artifacts)
        print("Clearing FINN BUILD DIR ahead of run")
        delete_dir_contents(os.environ["FINN_BUILD_DIR"])

        # Initialize dictionary to collect all benchmark results
        # TODO: remove completely or only use for meta data, actual results go into run-specific .json files within /report
        self.output_dict = {}

        # Inputs (e.g., ONNX model, golden I/O pair, folding config, etc.) for custom FINN build flow
        self.build_inputs = {}

        # Collect tuples of (name, source path, archive?) to save as pipeline artifacts upon run completion or fail by exception
        self.artifacts_collection = []

        # Collect tuples of (name, source path, archive?) to save as local artifacts upon run completion or fail by exception
        self.local_artifacts_collection = []
        if self.debug:
            # Save entire FINN build dir and working dir
            # TODO: add option to only save upon exception (in FINN builder or benchmarking infrastructure)
            self.local_artifacts_collection.append(("debug_finn_tmp", os.environ["FINN_BUILD_DIR"], False))
            self.local_artifacts_collection.append(("debug_finn_cwd", os.environ["FINN_ROOT"], False))

    def save_artifact(self, target_path, source_path, archive=False):
        if os.path.isdir(source_path):
            if archive:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.make_archive(target_path, "zip", source_path)
            else:
                os.makedirs(target_path, exist_ok=True)
                copytree(source_path, target_path, dirs_exist_ok=True)
        elif os.path.isfile(source_path):
            os.makedirs(target_path, exist_ok=True)
            shcopy(source_path, target_path)

    def save_artifacts_collection(self):
        # this should be called upon successful or failed completion of a run
        for (name, source_path, archive) in self.artifacts_collection:
            target_path = os.path.join(self.artifacts_dir, "runs_output", "run_%d" % (self.run_id), name)
            self.save_artifact(target_path, source_path, archive)

    def save_local_artifacts_collection(self):
        # this should be called upon successful or failed completion of a run
        for (name, source_path, archive) in self.local_artifacts_collection:
            target_path = os.path.join(self.save_dir, name, "run_%d" % (self.run_id))
            self.save_artifact(target_path, source_path, archive)
    
    # must be defined by subclass
    def step_export_onnx(self):
        pass

    # must be defined by subclass
    def step_build_setup(self):
        pass

    # defaults to normal build flow, may be overwritten by subclass
    def run(self):
        self.steps_full_build_flow()

    # def step_finn_estimate(self):
    #     # Gather FINN estimates
    #     print("Gathering FINN estimates")

    #     model = self.model_initial
    #     finn_resources_model = res_estimation(model, fpgapart=self.part)
    #     finn_cycles_model = model.analysis(exp_cycles_per_layer)
    #     if self.target_node:
    #         node = model.get_nodes_by_op_type(self.target_node)[0]
    #         finn_resources = finn_resources_model[node.name]
    #         finn_cycles = finn_cycles_model[node.name]
    #     else:
    #         finn_resources = finn_resources_model # TODO: aggregate?
    #         finn_cycles = 0 # TODO: aggregate or drop
    #     finn_estimates = finn_resources
    #     finn_estimates["CYCLES"] = finn_cycles
    #     self.output_dict["finn_estimates"] = finn_estimates

    # def step_hls(self):
    #     # Perform Vitis HLS synthesis for HLS resource/performance reports
    #     start_time = time.time()
    #     print("Performing Vitis HLS synthesis")
    #     model = self.model_initial
    #     model = model.transform(PrepareIP(self.part, self.clock_period_ns))
    #     model = model.transform(HLSSynthIP())

    #     hls_resources_model = model.analysis(hls_synth_res_estimation)
    #     if self.target_node:
    #         node = model.get_nodes_by_op_type(self.target_node)[0]
    #         hls_resources = hls_resources_model[node.name]
    #     else:
    #         hls_resources = hls_resources_model # TODO: aggregate?
    #     self.output_dict["hls_estimates"] = hls_resources
    #     self.output_dict["hls_time"] = int(time.time() - start_time)

    #     self.model_step_hls = copy.deepcopy(model)

    # def step_rtlsim(self):
    #     # Perform RTL simulation for performance measurement
    #     start_time = time.time()
    #     print("Performing Verilator RTL simulation (n=1)")
    #     # Prepare
    #     model = self.model_step_hls
    #     model = model.transform(SetExecMode("rtlsim"))
    #     model = model.transform(PrepareRTLSim())
    #     # Generate input data
    #     input_tensor = model.graph.input[0]
    #     input_shape = model.get_tensor_shape(input_tensor.name)
    #     input_dtype = model.get_tensor_datatype(input_tensor.name)
    #     x = gen_finn_dt_tensor(input_dtype, input_shape)
    #     input_dict = prepare_inputs(x, input_dtype, None) # TODO: fix Bipolar conversion case
    #     # Run
    #     oxe.execute_onnx(model, input_dict)["outp"]  # do not check output for correctness TODO: add functional verification throughout benchmarking steps
    #     # Log result
    #     node = model.get_nodes_by_op_type("MVAU_hls")[0]
    #     inst = getCustomOp(node)
    #     rtlsim_cycles = inst.get_nodeattr("cycles_rtlsim")
    #     self.output_dict["rtlsim_cycles"] = rtlsim_cycles
    #     self.output_dict["rtlsim_time"] = int(time.time() - start_time)

# TODO: re-introduce simple Vivado power estimation as new builder step
    # def step_synthesis(self):
    #     # Perform Vivado synthesis for accurate resource/timing and inaccurate power reports
    #     start_time = time.time()
    #     print("Performing Vivado (stitched-ip, out-of-context) synthesis")
    #     model = self.model_step_hls
    #     model = model.transform(ReplaceVerilogRelPaths())
    #     model = model.transform(CreateStitchedIP(self.part, self.clock_period_ns))
    #     model = model.transform(SynthOutOfContext(part=self.part, clk_period_ns=self.clock_period_ns))
    #     ooc_synth_results = eval(model.get_metadata_prop("res_total_ooc_synth"))

    #     start_test_batch_fast(
    #         results_path=self.artifacts_dir_power,
    #         project_path=os.path.join(
    #             ooc_synth_results["vivado_proj_folder"], "vivadocompile", "vivadocompile.xpr"
    #         ),
    #         run_target="impl_1",
    #         pairs=[(25, 0.5), (50, 0.5), (75, 0.5)],
    #     )

    #     # Log most important power results directly (refer to detailed logs for more)
    #     for reportname in ["25_0.5", "50_0.5", "75_0.5"]:
    #         with open(os.path.join(self.artifacts_dir_power, "%s.json" % reportname), "r") as f:
    #             report = json.load(f)
    #             power = float(report["Summary"]["tables"][0]["Total On-Chip Power (W)"][0])
    #             power_dyn = float(report["Summary"]["tables"][0]["Dynamic (W)"][0])
    #             ooc_synth_results["power_%s" % reportname] = power
    #             ooc_synth_results["power_dyn_%s" % reportname] = power_dyn

    #     self.output_dict["ooc_synth"] = ooc_synth_results
    #     self.output_dict["ooc_synth_time"] = int(time.time() - start_time)

    #     # Save model for logging purposes
    #     model.save(os.path.join(self.artifacts_dir_models, "model_%d_synthesis.onnx" % (self.run_id)))
    #     self.model_step_synthesis = copy.deepcopy(model)

# TODO: re-introduce sim-based Vivado power estimation as new builder step
    # def step_sim_power(self):
    #     # Perform Vivado simulation for accurate power report
    #     start_time = time.time()
    #     if "ooc_synth" not in self.output_dict:
    #         print("ERROR: step_sim_power requires step_synthesis")
    #     print("Performing Vivado simulation for power report")
    #     if "rtlsim_cycles" in self.output_dict:
    #         sim_duration_ns = self.output_dict["rtlsim_cycles"] * 3 * self.clock_period_ns
    #     else:
    #         sim_duration_ns = self.output_dict["finn_estimates"]["CYCLES"] * 3 * self.clock_period_ns

    #     model = self.model_step_synthesis
    #     input_tensor = model.graph.input[0]
    #     output_tensor = model.graph.output[0]
    #     input_node_inst = getCustomOp(model.find_consumer(input_tensor.name))
    #     output_node_inst = getCustomOp(model.find_producer(output_tensor.name))
    #     sim_power_report(
    #         results_path=self.artifacts_dir_power,
    #         project_path=os.path.join(
    #             self.output_dict["ooc_synth"]["vivado_proj_folder"], "vivadocompile", "vivadocompile.xpr"
    #         ),
    #         in_width=input_node_inst.get_instream_width(),
    #         out_width=output_node_inst.get_outstream_width(),
    #         dtype_width=model.get_tensor_datatype(input_tensor.name).bitwidth(),
    #         sim_duration_ns=sim_duration_ns,
    #     )

    #     # Log most important power results directly (refer to detailed logs for more)
    #     for reportname in ["sim"]:
    #         with open(os.path.join(self.artifacts_dir_power, "%s.json" % reportname), "r") as f:
    #             report = json.load(f)
    #             power = float(report["Summary"]["tables"][0]["Total On-Chip Power (W)"][0])
    #             power_dyn = float(report["Summary"]["tables"][0]["Dynamic (W)"][0])
    #             self.output_dict["power_%s" % reportname] = power
    #             self.output_dict["power_dyn%s" % reportname] = power_dyn

    #     self.output_dict["sim_power_time"] = int(time.time() - start_time)

    def step_parse_builder_output(self, build_dir):
        # TODO: output as .json or even add as new build step
        ### CHECK FOR VERIFICATION STEP SUCCESS ###
        if (os.path.exists(os.path.join(build_dir, "verification_output"))):
            # Collect all verification output filenames
            outputs = glob.glob(os.path.join(build_dir, "verification_output/*.npy"))
            # Extract the verification status for each verification output by matching
            # to the SUCCESS string contained in the filename
            status = all([
                out.split("_")[-1].split(".")[0] == "SUCCESS" for out in outputs
            ])
    
            # Construct a dictionary reporting the verification status as string
            self.output_dict["builder_verification"] = {"verification": {True: "success", False: "fail"}[status]}
            # TODO: mark job as failed if verification fails?

    def steps_full_build_flow(self):
        # Default step sequence for benchmarking a full FINN builder flow

        ### SETUP ###
        # Use a temporary dir for buildflow-related files (next to FINN_BUILD_DIR)
        # Ensure it exists but is empty (clear potential artifacts from previous runs)
        tmp_buildflow_dir = os.path.join(os.environ["PATH_WORKDIR"], "buildflow")
        os.makedirs(tmp_buildflow_dir, exist_ok=True)
        delete_dir_contents(tmp_buildflow_dir)
        self.build_inputs["build_dir"] = os.path.join(tmp_buildflow_dir, "build_output")
        os.makedirs(self.build_inputs["build_dir"], exist_ok=True)

        # Save full build dir as local artifact
        self.local_artifacts_collection.append(("build_output", self.build_inputs["build_dir"], False))
        # Save reports and deployment package as pipeline artifacts
        self.artifacts_collection.append(("reports", os.path.join(self.build_inputs["build_dir"], "report"), False))
        self.artifacts_collection.append(("reports", os.path.join(self.build_inputs["build_dir"], "build_dataflow.log"), False))
        self.artifacts_collection.append(("deploy", os.path.join(self.build_inputs["build_dir"], "deploy"), True))

        ### MODEL CREATION/IMPORT ###
        # TODO: track fixed input onnx models with DVC
        if "model_dir" in self.params:
            # input ONNX model and verification input/output pairs are provided
            model_dir = self.params["model_dir"]
            self.build_inputs["onnx_path"] = os.path.join(model_dir, "model.onnx")
            self.build_inputs["input_npy_path"] = os.path.join(model_dir, "inp.npy")
            self.build_inputs["output_npy_path"] = os.path.join(model_dir, "out.npy")
        elif "model_path" in self.params:
            self.build_inputs["onnx_path"] = self.params["model_path"]
        else:
            # input ONNX model (+ optional I/O pair for verification) will be generated
            self.build_inputs["onnx_path"] = os.path.join(tmp_buildflow_dir, "model_export.onnx")
            if self.step_export_onnx(self.build_inputs["onnx_path"]) == "skipped":
                # microbenchmarks might skip because no valid model can be generated for given params
                return
            self.save_local_artifact("model_step_export", self.build_inputs["onnx_path"])

        if "folding_path" in self.params:
            self.build_inputs["folding_path"] = self.params["folding_path"]
        if "specialize_path" in self.params:
            self.build_inputs["specialize_path"] = self.params["specialize_path"]
        if "floorplan_path" in self.params:
            self.build_inputs["floorplan_path"] = self.params["floorplan_path"]

        ### BUILD SETUP ###
        cfg = self.step_build_setup()
        cfg.generate_outputs = self.params["output_products"]
        cfg.output_dir = self.build_inputs["build_dir"]
        cfg.synth_clk_period_ns = self.clock_period_ns
        cfg.board = self.board
        if self.board in alveo_part_map:
            cfg.shell_flow_type=build_cfg.ShellFlowType.VITIS_ALVEO
            cfg.vitis_platform=alveo_default_platform[self.board]
        else:
            cfg.shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ
        # enable extra performance optimizations (physopt)
        # TODO: check OMX synth strategy again!
        cfg.vitis_opt_strategy=build_cfg.VitisOptStrategy.PERFORMANCE_BEST
        cfg.verbose = False
        cfg.enable_build_pdb_debug = False
        cfg.stitched_ip_gen_dcp = False # only needed for further manual integration
        cfg.force_python_rtlsim = False
        cfg.split_large_fifos = True
        cfg.enable_instrumentation = True # no IODMA functional correctness/accuracy test yet
        #rtlsim_use_vivado_comps # TODO ?
        #cfg.default_swg_exception
        #cfg.large_fifo_mem_style

        # "manual or "characterize" or "largefifo_rtlsim" or "live"
        if "fifo_method" in self.params:
            if self.params["fifo_method"] == "manual":
                cfg.auto_fifo_depths = False
            elif self.params["fifo_method"] == "live":
                cfg.auto_fifo_depths = False
                cfg.live_fifo_sizing = True
                cfg.enable_instrumentation = True
            else:
                cfg.auto_fifo_depths = True
                cfg.auto_fifo_strategy = self.params["fifo_method"]
        # only relevant for "characterize" method: "rtlsim" or "analytical"
        if "fifo_strategy" in self.params:
            cfg.characteristic_function_strategy = self.params["fifo_strategy"]

        # Batch size used for RTLSim performance measurement (and in-depth FIFO test here)
        # TODO: determine automatically or replace by exact instr wrapper sim
        if "rtlsim_n" in self.params:
            cfg.rtlsim_batch_size=self.params["rtlsim_n"]

        # Batch size used for FIFO sizing (largefifo_rtlsim only)
        if "fifo_rtlsim_n" in self.params:
            cfg.fifosim_n_inferences=self.params["fifo_rtlsim_n"]

        # Manual correction factor for FIFO-Sim input throttling
        if "fifo_throttle_factor" in self.params:
            cfg.fifo_throttle_factor = self.params["fifo_throttle_factor"]

        if "folding_path" in self.build_inputs:
            cfg.folding_config_file = self.build_inputs["folding_path"]
        if "specialize_path" in self.build_inputs:
            cfg.specialize_layers_config_file = self.build_inputs["specialize_path"]
        if "floorplan_path" in self.build_inputs:
            cfg.floorplan_path = self.build_inputs["floorplan_path"]

        # Default of 1M cycles is insufficient for MetaFi (6M) and RN-50 (2.5M)
        # TODO: make configurable or set on pipeline level?
        os.environ["LIVENESS_THRESHOLD"] = "10000000"

        ### BUILD ###
        build.build_dataflow_cfg(self.build_inputs["onnx_path"], cfg)

        ### ANALYSIS ###
        self.step_parse_builder_output(self.build_inputs["build_dir"])

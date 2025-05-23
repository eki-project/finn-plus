import glob
import json
import os
import shutil
import subprocess
from shutil import copy as shcopy
from shutil import copytree

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.benchmarking.templates import (
    template_open,
    template_sim_power,
    template_single_test,
    template_switching_simulation_tb,
)
from finn.benchmarking.util import delete_dir_contents, power_xml_to_dict
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.util.basic import alveo_default_platform, alveo_part_map, part_map

# TODO: merge this file into bench.py once most functionality has been moved to builder


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
    script = script.replace("$REPORT_NAME$", "sim")
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


class bench:
    def __init__(self, params, task_id, run_id, work_dir, artifacts_dir, save_dir, debug=True):
        super().__init__()
        self.params = params
        self.task_id = task_id
        self.run_id = run_id
        self.work_dir = work_dir
        self.artifacts_dir = artifacts_dir
        self.save_dir = save_dir
        self.debug = debug

        # TODO: setup a logger so output can go to console (with task id prefix)
        # TODO: coordinate with new builder loggin setup

        # Setup some basic global default configuration
        # TODO: are these class members even used anymore?
        if "synth_clk_period_ns" in params:
            self.clock_period_ns = params["synth_clk_period_ns"]
        else:
            self.clock_period_ns = 10
            self.params["synth_clk_period_ns"] = self.clock_period_ns

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

        if self.board in alveo_part_map:
            self.params["shell_flow_type"] = build_cfg.ShellFlowType.VITIS_ALVEO
            self.params["vitis_platform"] = alveo_default_platform[self.board]
        else:
            self.params["shell_flow_type"] = build_cfg.ShellFlowType.VIVADO_ZYNQ

        # Clear FINN tmp build dir before every run
        print("Clearing FINN BUILD DIR ahead of run")
        delete_dir_contents(os.environ["FINN_BUILD_DIR"])

        # Initialize dictionary to collect all benchmark results
        # TODO: remove completely or only use for meta data,
        # actual results go into run-specific .json files within /report
        self.output_dict = {}

        # Inputs (e.g., ONNX model, golden I/O pair, folding config, etc.)
        self.build_inputs = {}

        # Collect tuples of (name, source path, archive?) to save as pipeline artifacts
        self.artifacts_collection = []

        # Collect tuples of (name, source path, archive?) to save as local artifacts
        self.local_artifacts_collection = []
        if self.debug:
            # Save entire FINN_BUILD_DIR
            # TODO: add option to only save upon error/exception
            self.local_artifacts_collection.append(
                ("debug_finn_tmp", os.environ["FINN_BUILD_DIR"], True)
            )

        # SETUP
        # Use a temporary dir for buildflow-related files (next to FINN_BUILD_DIR)
        # Ensure it exists but is empty (clear potential artifacts from previous runs)
        tmp_buildflow_dir = os.path.join(self.work_dir, "buildflow")
        os.makedirs(tmp_buildflow_dir, exist_ok=True)
        delete_dir_contents(tmp_buildflow_dir)
        self.build_inputs["build_dir"] = os.path.join(
            tmp_buildflow_dir, "build_output"
        )  # TODO remove in favor of self.build_dir
        self.build_dir = os.path.join(tmp_buildflow_dir, "build_output")
        self.report_dir = os.path.join(self.build_dir, "report")
        os.makedirs(self.report_dir, exist_ok=True)

        # Save full build dir as local artifact
        self.local_artifacts_collection.append(("build_output", self.build_dir, False))
        # Save reports and deployment package as pipeline artifacts
        self.artifacts_collection.append(("reports", self.report_dir, False))
        self.artifacts_collection.append(
            ("reports", os.path.join(self.build_dir, "build_dataflow.log"), False)
        )
        self.artifacts_collection.append(("deploy", os.path.join(self.build_dir, "deploy"), True))

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
        for name, source_path, archive in self.artifacts_collection:
            target_path = os.path.join(
                self.artifacts_dir, "runs_output", "run_%d" % (self.run_id), name
            )
            self.save_artifact(target_path, source_path, archive)

    def save_local_artifacts_collection(self):
        # this should be called upon successful or failed completion of a run
        for name, source_path, archive in self.local_artifacts_collection:
            target_path = os.path.join(self.save_dir, name, "run_%d" % (self.run_id))
            self.save_artifact(target_path, source_path, archive)

    # must be defined by subclass
    def step_export_onnx(self):
        pass

    # can be overwritten by subclass if setup is too complex for YAML definition
    def step_build_setup(self):
        dut_yaml_name = self.params["dut"] + ".yml"
        dut_path = os.path.join(os.path.dirname(__file__), "dut", dut_yaml_name)
        if os.path.isfile(dut_path):
            with open(dut_path, "r") as f:
                return DataflowBuildConfig.from_yaml(f)
        else:
            raise Exception("No DUT-specific YAML build definition found")

    # defaults to normal build flow, may be overwritten by subclass
    def run(self):
        return self.steps_full_build_flow()

    def step_parse_builder_output(self, build_dir):
        # TODO: output as .json or even add as new build step
        # CHECK FOR VERIFICATION STEP SUCCESS
        if os.path.exists(os.path.join(build_dir, "verification_output")):
            # Collect all verification output filenames
            outputs = glob.glob(os.path.join(build_dir, "verification_output/*.npy"))
            # Extract the verification status for each verification output by matching
            # to the SUCCESS string contained in the filename
            status = all([out.split("_")[-1].split(".")[0] == "SUCCESS" for out in outputs])

            # Construct a dictionary reporting the verification status as string
            self.output_dict["builder_verification"] = {
                "verification": {True: "success", False: "fail"}[status]
            }
            # TODO: mark job as failed if verification fails?

    def steps_full_build_flow(self):
        # Default step sequence for benchmarking a full FINN builder flow

        # LIST OF ADDITIONAL YAML OPTIONS (beyond DataflowBuildConfig)
        custom_params = [
            "model_dir",  # used to setup onnx/npy input
            "model_path",  # used to setup onnx/npy input
            # model-gen parameters, such as seed, simd, pe, etc.
            # TODO: separate these from builder options
        ]

        # MODEL CREATION/IMPORT
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
            self.build_inputs["onnx_path"] = os.path.join(
                self.build_inputs["build_dir"], "model_export.onnx"
            )
            if self.step_export_onnx(self.build_inputs["onnx_path"]) == "skipped":
                # microbenchmarks might skip because no model can be generated for given params
                return "skipped"

        # BUILD SETUP
        # Initialize from YAML (default) or custom script (if dedicated subclass is defined)
        cfg = self.step_build_setup()

        # Set some global defaults (could still be overwritten by run-specific YAML)
        cfg.output_dir = self.build_inputs["build_dir"]
        # enable extra performance optimizations (physopt)
        # TODO: check OMX synth strategy again!
        cfg.vitis_opt_strategy = build_cfg.VitisOptStrategy.PERFORMANCE_BEST
        cfg.verbose = False
        cfg.enable_build_pdb_debug = False
        # cfg.stitched_ip_gen_dcp = False # only needed for further manual integration
        cfg.force_python_rtlsim = False
        cfg.split_large_fifos = True
        cfg.save_intermediate_models = True  # Save the intermediate model graphs
        cfg.verify_save_full_context = True  # Output full context dump for verification steps
        cfg.enable_instrumentation = True
        # rtlsim_use_vivado_comps # TODO ?
        # cfg.default_swg_exception
        # cfg.large_fifo_mem_style

        # Overwrite build config settings with run-specific YAML build definition
        for key in self.params:
            if hasattr(cfg, key):
                setattr(cfg, key, self.params[key])
            else:
                if key not in custom_params:
                    pass
                    # TODO: be more strict? support custom extra options like MetaFi uses?
                    # raise Exception("Unrecognized builder config defined in YAML: %s" % key)

        # Default of 1M cycles is insufficient for MetaFi (6M) and RN-50 (2.5M)
        # TODO: make configurable or set on pipeline level?
        os.environ["LIVENESS_THRESHOLD"] = "10000000"

        # BUILD
        build.build_dataflow_cfg(self.build_inputs["onnx_path"], cfg)

        # ANALYSIS
        self.step_parse_builder_output(self.build_inputs["build_dir"])

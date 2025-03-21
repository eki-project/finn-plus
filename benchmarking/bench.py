import itertools
import sys
import os
import json
import time
import traceback
import onnxruntime as ort
import importlib

from util import delete_dir_contents

from dut.mvau import bench_mvau
from dut.resnet50 import bench_resnet50
from dut.metafi import bench_metafi
from dut.synthetic_nonlinear import bench_synthetic_nonlinear
from dut.transformer import bench_transformer
from dut.vgg10 import bench_vgg10
from dut.mobilenetv1 import bench_mobilenetv1

dut = dict()
dut["mvau"] = bench_mvau
dut["resnet50"] = bench_resnet50
dut["metafi"] = bench_metafi
dut["synthetic_nonlinear"] = bench_synthetic_nonlinear
dut["transformer"] = bench_transformer
dut["vgg10"] = bench_vgg10
dut["mobilenetv1"] = bench_mobilenetv1

ras_module = importlib.import_module("dut.rastreamlining-fpl25.end2end.end2end")
dut["ras"] = ras_module.ras_end2end

def main(config_name):
    exit_code = 0
    # Attempt to work around onnxruntime issue on Slurm-managed clusters:
    # See https://github.com/microsoft/onnxruntime/issues/8313
    # This seems to happen only when assigned CPU cores are not contiguous
    _default_session_options = ort.capi._pybind_state.get_default_session_options()
    def get_default_session_options_new():
        _default_session_options.inter_op_num_threads = 1
        _default_session_options.intra_op_num_threads = 1
        return _default_session_options
    ort.capi._pybind_state.get_default_session_options = get_default_session_options_new

    try:
        # Launched via SLURM, expect additional CI env vars
        job_id = int(os.environ["SLURM_JOB_ID"])
        # experiment_dir = os.environ.get("EXPERIMENT_DIR") # original experiment dir (before potential copy to ramdisk)
        experiment_dir = os.environ.get("CI_PROJECT_DIR")
        save_dir = os.path.join(os.environ.get("LOCAL_ARTIFACT_DIR"),
                            "CI_" + os.environ.get("CI_PIPELINE_ID") + "_" + os.environ.get("CI_PIPELINE_NAME"))
        work_dir = os.environ["PATH_WORKDIR"]

        # Gather benchmarking configs
        if config_name == "manual":
            config_path = os.path.join(os.environ.get("LOCAL_CFG_DIR"), os.environ.get("MANUAL_CFG_PATH"))
        else:
            configs_path = os.path.join(os.path.dirname(__file__), "cfg")
            config_select = config_name + ".json"
            config_path = os.path.join(configs_path, config_select)
        print("Job launched with SLURM ID: %d" % (job_id))
    except KeyError:
        # Launched without SLURM, assume test run on local machine
        job_id = 0
        experiment_dir = "bench_output/" + time.strftime("%d_%H_%M")
        save_dir = "bench_save/" + time.strftime("%d_%H_%M")
        work_dir = "bench_work"
        os.makedirs(work_dir, exist_ok=True)
        delete_dir_contents(work_dir)
        config_path = config_name # expect caller to provide direct path to a single config file
        print("Local test job launched without SLURM")

    try:
        # Launched as SLURM job array
        array_id = int(os.environ["SLURM_ARRAY_JOB_ID"])
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        task_count = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        print(
            "Launched as job array (Array ID: %d, Task ID: %d, Task count: %d)"
            % (array_id, task_id, task_count)
        )
    except KeyError:
        # Launched as single (SLURM or non-SLURM) job
        array_id = job_id
        task_id = 0
        task_count = 1
        print("Launched as single job")

    # Prepare result directory
    artifacts_dir = os.path.join(experiment_dir, "build_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    print("Collecting results in path: %s" % artifacts_dir)

    # Prepare local save dir for large artifacts (e.g., build output, tmp dir dump for debugging)
    os.makedirs(save_dir, exist_ok=True)
    print("Saving additional artifacts in path: %s" % save_dir)

    # Load config
    print("Loading config %s" % (config_path))
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        print("ERROR: config file not found")
        return

    # Expand all specified config combinations (gridsearch)
    config_expanded = []
    for param_set in config:
        param_set_expanded = list(
            dict(zip(param_set.keys(), x)) for x in itertools.product(*param_set.values())
        )
        config_expanded.extend(param_set_expanded)

    # Save config (only first job of array) for logging purposes
    if task_id == 0:
        with open(os.path.join(artifacts_dir, "bench_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        with open(os.path.join(artifacts_dir, "bench_config_exp.json"), "w") as f:
            json.dump(config_expanded, f, indent=2)

    # Determine which runs this job will work on
    total_runs = len(config_expanded)
    if total_runs <= task_count:
        if task_id < total_runs:
            selected_runs = [task_id]
        else:
            return
    else:
        selected_runs = []
        idx = task_id
        while idx < total_runs:
            selected_runs.append(idx)
            idx = idx + task_count
    print("This job will perform %d out of %d total runs" % (len(selected_runs), total_runs))

    # Run benchmark
    # TODO: integrate this loop (especially status logging) into the bench class
    # TODO: log stdout of individual tasks of the job array into seperate files as artifacts (GitLab web interface is not readable), coordinate with new logging
    for run, run_id in enumerate(selected_runs):
        print(
            "Starting run %d/%d (id %d of %d total runs)"
            % (run + 1, len(selected_runs), run_id, total_runs)
        )

        params = config_expanded[run_id]
        print("Run parameters: %s" % (str(params)))

        log_dict = {"run_id": run_id, "task_id": task_id, "params": params}

        # Create bench object for respective DUT
        if "dut" in params:
            if params["dut"] in dut:
                bench_object = dut[params["dut"]](params, task_id, run_id, work_dir, artifacts_dir, save_dir)
            else:
                print("ERROR: unknown DUT specified")
                return 1
        else:
            print("ERROR: no DUT specified")
            return 1

        try:
            result = bench_object.run()
            if result == "skipped":
                log_dict["status"] = "skipped"
                print("Run skipped")
            else:
                log_dict["status"] = "ok"
                print("Run successfully completed")
        except Exception:
            log_dict["status"] = "failed"
            print("Run failed: " + traceback.format_exc())
            exit_code = 1

        log_dict["output"] = bench_object.output_dict

        # examine status reported by builder (which catches all exceptions before they reach us here)
        # we could also fail the pipeline if functional verification fails (TODO)
        builder_log_path = os.path.join(bench_object.report_dir, "metadata_builder.json")
        if os.path.isfile(builder_log_path):
            with open(builder_log_path, "r") as f:
                builder_log = json.load(f)
            if builder_log["status"] == "failed":
                print("Run failed (builder reported failure)")
                exit_code = 1

        # log metadata of this run to its own report directory
        log_path = os.path.join(bench_object.report_dir, "metadata_bench.json")
        with open(log_path, "w") as f:
            json.dump(log_dict, f, indent=2)

        # save GitLab artifacts of this run (e.g., reports and deployment package)
        bench_object.save_artifacts_collection()
        # save local artifacts of this run (e.g., full build dir, detailed debug info)
        bench_object.save_local_artifacts_collection()

    print("Stopping job")
    return exit_code

if __name__ == "__main__":
    exit_code = main(sys.argv[1])
    sys.exit(exit_code)

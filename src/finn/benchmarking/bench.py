import itertools
import json
import onnxruntime as ort
import os
import sys
import time
import traceback
import yaml

from finn.benchmarking.bench_base import bench
from finn.benchmarking.dut.mvau import bench_mvau
from finn.benchmarking.dut.synthetic_nonlinear import bench_synthetic_nonlinear
from finn.benchmarking.dut.transformer import bench_transformer
from finn.benchmarking.util import delete_dir_contents

# Register custom bench subclasses that offer more control than YAML-based flow
dut = dict()
dut["mvau"] = bench_mvau
dut["synthetic_nonlinear"] = bench_synthetic_nonlinear
dut["transformer"] = bench_transformer


class PrefixPrinter(object):
    """
    Create a custom stream handler that adds a prefix
    """

    def __init__(self, prefix, originalstream):
        self.console = originalstream
        self.prefix = prefix
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.console.write(f"[{self.prefix}] " + line + "\n")

    def flush(self):
        self.console.flush()


def start_bench_run(config_name):
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
        # original experiment dir (before potential copy to ramdisk):
        # experiment_dir = os.environ.get("EXPERIMENT_DIR")
        experiment_dir = os.environ.get("CI_PROJECT_DIR")
        save_dir = os.path.join(
            os.environ.get("LOCAL_ARTIFACT_DIR"),
            "CI_" + os.environ.get("CI_PIPELINE_ID") + "_" + os.environ.get("CI_PIPELINE_NAME"),
        )
        work_dir = os.environ["PATH_WORKDIR"]

        # Gather benchmarking configs
        if config_name == "manual":
            # First check if the repo contains a config with this name (in ci/cfg/*)
            config_path = os.path.join("ci", "cfg", os.environ.get("MANUAL_CFG_PATH") + ".yml")
            if not os.path.exists(config_path):
                # Otherwise look in LOCAL_CFG_DIR for the filename
                config_path = os.path.join(
                    os.environ.get("LOCAL_CFG_DIR"), os.environ.get("MANUAL_CFG_PATH")
                )
        else:
            config_path = os.path.join("ci", "cfg", config_name + ".yml")
        print("Job launched with SLURM ID: %d" % (job_id))
    except KeyError:
        # Launched without SLURM, assume test run on local machine
        job_id = 0
        experiment_dir = "bench_output/" + time.strftime("%d_%H_%M")
        save_dir = "bench_save/" + time.strftime("%d_%H_%M")
        work_dir = "bench_work"
        os.makedirs(work_dir, exist_ok=True)
        delete_dir_contents(work_dir)
        config_path = config_name  # expect caller to provide direct path to a single config file
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
            config = yaml.load(f, Loader=yaml.SafeLoader)
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
    print(
        "STARTING JOB %d. IT WILL PERFORM %d OUT OF %d TOTAL RUNS"
        % (task_id, len(selected_runs), total_runs)
    )

    # Run benchmark
    successful_runs = []
    skipped_runs = []
    failed_runs = []
    for run, run_id in enumerate(selected_runs):
        print(
            "STARTING RUN %d/%d (ID %d OF %d TOTAL RUNS)"
            % (run + 1, len(selected_runs), run_id, total_runs)
        )

        params = config_expanded[run_id]
        print("RUN %d PARAMETERS: %s" % (run_id, str(params)))

        log_dict = {"run_id": run_id, "task_id": task_id, "params": params}

        # Create bench object for respective DUT
        if "dut" in params:
            if params["dut"] in dut:
                bench_object = dut[params["dut"]](
                    params, task_id, run_id, work_dir, artifacts_dir, save_dir
                )
            else:
                # If no custom bench subclass is defined, fall back to base class,
                # expect DUT-specific YAML definition instead
                bench_object = bench(params, task_id, run_id, work_dir, artifacts_dir, save_dir)
        else:
            print("ERROR: NO DUT SPECIFIED")
            return 1

        # Wrap stdout/stderr with an additional prefix to identify the run in the live console
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = PrefixPrinter("RUN %d (%s)" % (run_id, params["dut"]), sys.stdout)
        sys.stderr = PrefixPrinter("RUN %d (%s)" % (run_id, params["dut"]), sys.stderr)
        try:
            result = bench_object.run()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            if result == "skipped":
                log_dict["status"] = "skipped"
                print("BENCH RUN %d SKIPPED" % run_id)
                skipped_runs.append(run_id)
            else:
                log_dict["status"] = "ok"
        except Exception:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_dict["status"] = "failed"
            print("BENCH RUN %d FAILED WITH EXCEPTION: %s" % (run_id, traceback.format_exc()))
            failed_runs.append(run_id)
            exit_code = 1

        log_dict["output"] = bench_object.output_dict

        # examine status reported by builder (which catches all exceptions before they reach us)
        # we could also fail the pipeline if functional verification fails (TODO)
        builder_log_path = os.path.join(bench_object.report_dir, "metadata_builder.json")
        if os.path.isfile(builder_log_path):
            with open(builder_log_path, "r") as f:
                builder_log = json.load(f)
            if builder_log["status"] == "failed":
                print("BENCH RUN %d FAILED (BUILDER REPORTED FAILURE)" % run_id)
                failed_runs.append(run_id)
                exit_code = 1
            else:
                print("BENCH RUN %d COMPLETED (BUILDER REPORTED SUCCESS)" % run_id)
                successful_runs.append(run_id)
        else:
            print("BENCH RUN %d COMPLETED" % run_id)
            successful_runs.append(run_id)

        # log metadata of this run to its own report directory
        log_path = os.path.join(bench_object.report_dir, "metadata_bench.json")
        with open(log_path, "w") as f:
            json.dump(log_dict, f, indent=2)

        # save GitLab artifacts of this run (e.g., reports and deployment package)
        bench_object.save_artifacts_collection()
        # save local artifacts of this run (e.g., full build dir, detailed debug info)
        bench_object.save_local_artifacts_collection()

    print("STOPPING JOB %d (of %d total jobs)" % (task_id, task_count))
    print("JOB %d SUCCESSFUL RUNS: %s" % (task_id, successful_runs))
    print("JOB %d SKIPPED RUNS: %s" % (task_id, skipped_runs))
    print("JOB %d FAILED RUNS: %s" % (task_id, failed_runs))
    return exit_code

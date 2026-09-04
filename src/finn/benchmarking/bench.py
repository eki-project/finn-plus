"""FINN benchmarking execution framework.

This module provides the main entry point for running FINN benchmarks, supporting
both SLURM-based cluster execution and local testing. It handles configuration
expansion, job distribution, and result collection.
"""

import itertools
import json
import onnxruntime as ort
import os
import sys
import time
import traceback
import yaml
from pathlib import Path
from typing import Any, TextIO

from finn.benchmarking.bench_base import bench
from finn.benchmarking.dut.bench_mvau_multi_dnn import bench_mvau_multi_dnn
from finn.benchmarking.dut.mvau import bench_mvau
from finn.benchmarking.dut.synthetic_nonlinear import bench_synthetic_nonlinear

# from finn.benchmarking.dut.transformer import bench_transformer
from finn.benchmarking.util import delete_dir_contents

# Register custom bench subclasses that offer more control than YAML-based flow
dut = {}
dut["mvau"] = bench_mvau
dut["mvau_multi_dnn"] = bench_mvau_multi_dnn
dut["synthetic_nonlinear"] = bench_synthetic_nonlinear


class PrefixPrinter:
    """Custom stream handler that adds a prefix to console output for run identification."""

    def __init__(self, prefix: str, originalstream: TextIO) -> None:
        """Initialize the prefix printer with a prefix string and target stream."""
        self.console = originalstream
        self.prefix = prefix
        self.linebuf = ""

    def write(self, buf: str) -> None:
        """Write buffer content with prefix to the target stream."""
        for line in buf.rstrip().splitlines():
            self.console.write(f"[{self.prefix}] " + line + "\n")

    def flush(self) -> None:
        """Flush the target stream."""
        self.console.flush()


def start_bench_run(config_name: str) -> int | None:
    """Start a benchmarking run with the specified configuration.

    This function handles both SLURM cluster execution and local testing,
    loading configuration files, expanding parameter combinations, and
    distributing work across available tasks.

    Args:
        config_name (str): Name of configuration file or path to config file

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    exit_code = 0
    is_followup = False
    # Attempt to work around onnxruntime issue on Slurm-managed clusters:
    # See https://github.com/microsoft/onnxruntime/issues/8313
    # This seems to happen only when assigned CPU cores are not contiguous
    _default_session_options = ort.capi._pybind_state.get_default_session_options()  # noqa: SLF001

    def get_default_session_options_new() -> Any:
        """Return specific default session options for onnxruntime."""
        _default_session_options.inter_op_num_threads = 1
        _default_session_options.intra_op_num_threads = 1
        return _default_session_options

    ort.capi._pybind_state.get_default_session_options = (  # noqa: SLF001
        get_default_session_options_new
    )

    try:
        # Launched via SLURM, expect additional CI env vars
        job_id = int(os.environ["SLURM_JOB_ID"])
        # original experiment dir (before potential copy to ramdisk):
        # experiment_dir = os.environ.get("EXPERIMENT_DIR")
        experiment_dir = os.environ.get("CI_PROJECT_DIR")
        save_dir = str(
            Path(os.environ.get("LOCAL_ARTIFACT_DIR"))
            / ("CI_" + os.environ.get("CI_PIPELINE_ID") + "_" + os.environ.get("CI_PIPELINE_NAME"))
        )
        work_dir = os.environ["PATH_WORKDIR"]

        # Gather benchmarking configs
        if config_name == "manual":
            # First check if the repo contains a config with this name (in ci/cfg/*)
            config_path = str(Path("ci") / "cfg" / (os.environ.get("MANUAL_CFG_PATH") + ".yml"))
            if not Path(config_path).exists():
                # Otherwise look in LOCAL_CFG_DIR for the filename
                config_path = str(
                    Path(os.environ.get("LOCAL_CFG_DIR")) / os.environ.get("MANUAL_CFG_PATH")
                )
        elif config_name == "followup":
            config_path = str(Path("followup_bench_config.json"))
            is_followup = True
            save_dir = save_dir + "_followup"
        else:
            config_path = (
                config_name
                if config_name.endswith((".yaml", ".yml"))
                else str(Path("ci") / "cfg" / (config_name + ".yml"))
            )
        print(f"Job launched with SLURM ID: {job_id}")
    except KeyError:
        # Launched without SLURM, assume test run on local machine
        job_id = 0
        experiment_dir = "bench_output/" + time.strftime("%d_%H_%M")
        save_dir = "bench_save/" + time.strftime("%d_%H_%M")
        work_dir = "bench_work"
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        delete_dir_contents(work_dir)
        config_path = config_name  # expect caller to provide direct path to a single config file
        print("Local test job launched without SLURM")

    try:
        # Launched as SLURM job array
        array_id = int(os.environ["SLURM_ARRAY_JOB_ID"])
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        task_count = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        print(
            f"Launched as job array (Array ID: {array_id}, Task ID: {task_id}, "
            f"Task count: {task_count})"
        )
    except KeyError:
        # Launched as single (SLURM or non-SLURM) job
        array_id = job_id
        task_id = 0
        task_count = 1
        print("Launched as single job")

    # Prepare result directory
    artifacts_dir = str(Path(experiment_dir) / "build_artifacts")
    if is_followup:
        artifacts_dir = artifacts_dir + "_followup"
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    print(f"Collecting results in path: {artifacts_dir}")

    # Prepare local save dir for large artifacts (e.g., build output, tmp dir dump for debugging)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving additional artifacts in path: {save_dir}")

    # Load config
    print(f"Loading config {config_path}")
    if Path(config_path).exists():
        with Path(config_path).open() as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        print("ERROR: config file not found")
        return None

    # Expand all specified config combinations (gridsearch)
    config_expanded = []
    for param_set in config:
        param_set_expanded = [
            dict(zip(param_set.keys(), x, strict=True))
            for x in itertools.product(*param_set.values())
        ]
        config_expanded.extend(param_set_expanded)

    # Save config (only first job of array) for logging purposes
    if task_id == 0:
        with (Path(artifacts_dir) / "bench_config.json").open("w") as f:
            json.dump(config, f, indent=2)
        with (Path(artifacts_dir) / "bench_config_exp.json").open("w") as f:
            json.dump(config_expanded, f, indent=2)

    # Determine which runs this job will work on
    total_runs = len(config_expanded)
    if total_runs <= task_count:
        if task_id < total_runs:
            selected_runs = [task_id]
        else:
            return None
    else:
        selected_runs = []
        idx = task_id
        while idx < total_runs:
            selected_runs.append(idx)
            idx = idx + task_count
    print(
        f"STARTING JOB {task_id}. IT WILL PERFORM {len(selected_runs)} "
        f"OUT OF {total_runs} TOTAL RUNS"
    )

    # Run benchmark
    successful_runs = []
    skipped_runs = []
    failed_runs = []
    for run, run_id in enumerate(selected_runs):
        print(
            f"STARTING RUN {run + 1}/{len(selected_runs)} "
            f"(ID {run_id} OF {total_runs} TOTAL RUNS)"
        )

        params = config_expanded[run_id]
        print(f"RUN {run_id} PARAMETERS: {params!s}")

        log_dict = {"run_id": run_id, "task_id": task_id, "params": params}

        # Make experiments_config path relative to config file path if not absolute
        if "experiments_config" in params and not Path(params["experiments_config"]).is_absolute():
            cfg_path = Path(config_path).parent.resolve()
            params["experiments_config"] = str(cfg_path / params["experiments_config"])

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
        run_prefix = f"RUN {run_id} ({params['dut']})"
        sys.stdout = PrefixPrinter(run_prefix, sys.stdout)
        sys.stderr = PrefixPrinter(run_prefix, sys.stderr)
        try:
            result = bench_object.run()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            if result == "skipped":
                log_dict["status"] = "skipped"
                print(f"BENCH RUN {run_id} SKIPPED")
                skipped_runs.append(run_id)
            else:
                log_dict["status"] = "ok"
        except Exception:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_dict["status"] = "failed"
            print(f"BENCH RUN {run_id} FAILED WITH EXCEPTION: {traceback.format_exc()}")
            failed_runs.append(run_id)
            exit_code = 1

        log_dict["output"] = bench_object.output_dict

        # examine status reported by builder (which catches all exceptions before they reach us)
        # we could also fail the pipeline if functional verification fails (TODO)
        builder_log_path = Path(bench_object.report_dir) / "metadata_builder.json"
        if builder_log_path.is_file():
            with builder_log_path.open() as f:
                builder_log = json.load(f)
            if builder_log["status"] == "failed":
                print(f"BENCH RUN {run_id} FAILED (BUILDER REPORTED FAILURE)")
                failed_runs.append(run_id)
                exit_code = 1
            else:
                print(f"BENCH RUN {run_id} COMPLETED (BUILDER REPORTED SUCCESS)")
                successful_runs.append(run_id)
        else:
            print(f"BENCH RUN {run_id} COMPLETED")
            successful_runs.append(run_id)

        # log metadata of this run to its own report directory
        log_path = Path(bench_object.report_dir) / "metadata_bench.json"
        with log_path.open("w") as f:
            json.dump(log_dict, f, indent=2)

        # save GitLab artifacts of this run (e.g., reports and deployment package)
        bench_object.save_artifacts_collection()
        # save local artifacts of this run (e.g., full build dir, detailed debug info)
        bench_object.save_local_artifacts_collection()

    print(f"STOPPING JOB {task_id} (of {task_count} total jobs)")
    print(f"JOB {task_id} SUCCESSFUL RUNS: {successful_runs}")
    print(f"JOB {task_id} SKIPPED RUNS: {skipped_runs}")
    print(f"JOB {task_id} FAILED RUNS: {failed_runs}")
    return exit_code

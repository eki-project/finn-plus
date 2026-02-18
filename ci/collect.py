"""Collect and log benchmark results to DVC."""

import argparse
import json
import os
import shutil
import sys
from datetime import date
from dvclive import Live


def delete_dir_contents(dir):
    """Delete all contents of a directory."""
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def open_json_report(id, report_name, is_followup=False):
    """Open JSON report from build or measurement artifacts."""
    # TODO: handle followup setting better
    # look in both, build & measurement, artifacts
    if is_followup:
        path1 = os.path.join(
            "build_artifacts_followup", "runs_output", "run_%d" % (id), "reports", report_name
        )
        path2 = os.path.join(
            "measurement_artifacts_followup", "runs_output", "run_%d" % (id), "reports", report_name
        )
    else:
        path1 = os.path.join(
            "build_artifacts", "runs_output", "run_%d" % (id), "reports", report_name
        )
        path2 = os.path.join(
            "measurement_artifacts", "runs_output", "run_%d" % (id), "reports", report_name
        )
    if os.path.isfile(path1):
        with open(path1, "r") as f:
            report = json.load(f)
        return report
    elif os.path.isfile(path2):
        with open(path2, "r") as f:
            report = json.load(f)
        return report
    else:
        return None


# Wrapper around DVC Live object
class DVCLoggerHelper:
    """Wrapper around DVC Live for logging experiments."""

    def __init__(self, experiment_name, experiment_msg, id, params, is_followup=False):
        """Initialize DVC logger with experiment details."""
        self.id = id
        self.is_followup = is_followup

        # extract logging settings from params
        self.store_as_experiment = params["params"].get("store_results_in_dvc_experiment", True)
        self.store_as_data = params["params"].get("store_results_in_dvc_data", False)

        if self.store_as_experiment:
            # Start DVC Live experiment session
            # TODO: cache images once we switch to a cache provider that works with DVC Studio
            self.live = Live(
                exp_name=experiment_name, exp_message=experiment_msg, cache_images=False
            )
        else:
            self.live = None

        if self.store_as_data:
            self.data_dict = dict()
        else:
            self.data_dict = None

        self.log_params(params)

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context manager and end DVC session."""
        if self.store_as_experiment:
            # End DVC Live experiment session
            self.live.end()

    def log_params(self, params):
        """Log parameters to DVC."""
        if self.store_as_experiment:
            self.live.log_params(params)
        if self.store_as_data:
            self.data_dict.update(params)

    def log_metric(self, prefix, name, value):
        """Log a single metric to DVC."""
        # sanitize '/' in name because DVC uses it to nest metrics (which we do via prefix)
        name = name.replace("/", "-")

        if self.store_as_experiment:
            self.live.log_metric(prefix + name, value, plot=False)
        if self.store_as_data:
            # store in nested dictionary structure based on prefix
            if "metrics" not in self.data_dict:
                self.data_dict["metrics"] = dict()
            _dict = self.data_dict["metrics"]

            for key in prefix.split("/"):
                if key:
                    if key not in _dict:
                        _dict[key] = dict()
                    _dict = _dict[key]
            _dict[name] = value

    def log_image(self, image_name, image_path):
        """Log an image artifact to DVC."""
        if self.store_as_experiment:
            self.live.log_image(image_name, image_path)

    def log_artifact(self, artifact_path):
        """Log an artifact to DVC."""
        if self.store_as_experiment:
            self.live.log_artifact(artifact_path)

    def log_all_metrics_from_report(self, report_name, prefix=""):
        """Log all metrics from a JSON report."""
        report = open_json_report(self.id, report_name, self.is_followup)
        if report:
            for key in report:
                self.log_metric(prefix, key, report[key])

    def log_metrics_from_report(self, report_name, keys, prefix=""):
        """Log specific metrics from a JSON report."""
        report = open_json_report(self.id, report_name, self.is_followup)
        if report:
            for key in keys:
                if key in report:
                    self.log_metric(prefix, key, report[key])

    def log_nested_metrics_from_report(self, report_name, key_top, keys, prefix=""):
        """Log nested metrics from a JSON report."""
        report = open_json_report(self.id, report_name, self.is_followup)
        if report:
            if key_top in report:
                for key in keys:
                    if key in report[key_top]:
                        self.log_metric(prefix, key, report[key_top][key])


if __name__ == "__main__":
    """Go through all runs found in the artifacts and log their results to DVC."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Collect and log benchmark results to DVC.")
    parser.add_argument(
        "--followup",
        action="store_true",
        help="Indicate this is a follow-up run (prevents generating new follow-up configs)",
    )
    args = parser.parse_args()

    if args.followup:
        run_dir_list = os.listdir(os.path.join("build_artifacts_followup", "runs_output"))
    else:
        run_dir_list = os.listdir(os.path.join("build_artifacts", "runs_output"))
    print("Looking for runs in build artifacts")
    run_ids = []
    for run_dir in run_dir_list:
        if run_dir.startswith("run_"):
            run_id = int(run_dir[4:])
            run_ids.append(run_id)
    run_ids.sort()
    print("Found %d runs" % len(run_ids))

    follow_up_bench_cfg = list()
    microbench_result_data = dict()

    for id in run_ids:
        print("Processing run %d" % id)
        if args.followup:
            experiment_name = "CI_" + os.environ.get("CI_PIPELINE_ID") + "_followup_" + str(id)
        else:
            experiment_name = "CI_" + os.environ.get("CI_PIPELINE_ID") + "_" + str(id)
        experiment_msg = (
            "[CI] "
            + os.environ.get("CI_PIPELINE_NAME")
            + " ("
            + os.environ.get("CI_PIPELINE_ID")
            + "_"
            + str(id)
            + ")"
        )

        # initialize logging wrapper with input parameters logged by benchmarking infrastructure
        metadata_bench = open_json_report(id, "metadata_bench.json", args.followup)
        params = {"params": metadata_bench["params"]}
        with DVCLoggerHelper(
            experiment_name, experiment_msg, id, params, is_followup=args.followup
        ) as dvc_logger:
            # optional metadata logged by builder
            metadata_builder = open_json_report(id, "metadata_builder.json", args.followup)
            if metadata_builder:
                metadata = {
                    "metadata": {
                        "tool_version": metadata_builder["tool_version"],
                    }
                }
                dvc_logger.log_params(metadata)

            # optional dut_info.json (additional information generated during model generation)
            dut_info_report = open_json_report(id, "dut_info.json", args.followup)
            if dut_info_report:
                dut_info = {"dut_info": dut_info_report}
                dvc_logger.log_params(dut_info)

            # METRICS
            # TODO: make all logs consistent (at generation), e.g., BRAM vs BRAM18 vs BRAM36)

            # status
            status = metadata_bench["status"]
            if status == "ok":
                # mark as failed if either bench or builder indicates failure
                if metadata_builder:
                    status_builder = metadata_builder["status"]
                    if status_builder == "failed":
                        status = "failed"
            dvc_logger.log_metric("", "status", status)

            # verification steps
            if "output" in metadata_bench:
                if "builder_verification" in metadata_bench["output"]:
                    dvc_logger.log_metric(
                        "",
                        "verification",
                        metadata_bench["output"]["builder_verification"]["verification"],
                    )

            # estimate_layer_resources.json
            dvc_logger.log_nested_metrics_from_report(
                "estimate_layer_resources.json",
                "total",
                [
                    "LUT",
                    "DSP",
                    "BRAM_18K",
                    "URAM",
                ],
                prefix="estimate/resources/",
            )

            # estimate_layer_resources_hls.json
            dvc_logger.log_nested_metrics_from_report(
                "estimate_layer_resources_hls.json",
                "total",
                [
                    "LUT",
                    "FF",
                    "DSP",
                    "DSP48E",
                    "DSP58E",  # TODO: aggregate/unify DSP reporting
                    "BRAM_18K",
                    "URAM",
                ],
                prefix="hls_estimate/resources/",
            )

            # estimate_network_performance.json
            dvc_logger.log_metrics_from_report(
                "estimate_network_performance.json",
                [
                    "critical_path_cycles",
                    "max_cycles",
                    "max_cycles_node_name",
                    "estimated_throughput_fps",
                    "estimated_latency_ns",
                ],
                prefix="estimate/performance/",
            )

            # rtlsim_performance.json
            dvc_logger.log_metrics_from_report(
                "rtlsim_performance.json",
                [
                    "N",
                    "TIMEOUT",
                    "latency_cycles",
                    "cycles",
                    "fclk[mhz]",
                    "throughput[images/s]",
                    "stable_throughput[images/s]",
                    # add INPUT_DONE, OUTPUT_DONE, number transactions?
                ],
                prefix="rtlsim/performance/",
            )

            # fifo_sizing.json
            dvc_logger.log_metrics_from_report(
                "fifo_sizing.json", ["total_fifo_size_kB"], prefix="fifosizing/"
            )

            # stitched IP DCP synth resource report
            dvc_logger.log_nested_metrics_from_report(
                "post_synth_resources_dcp.json",
                "(top)",
                [
                    "LUT",
                    "FF",
                    "SRL",
                    "DSP",
                    "BRAM_18K",
                    "BRAM_36K",
                    "URAM",
                ],
                prefix="synth(dcp)/resources/",
            )

            # stitched IP DCP synth resource breakdown
            # TODO: generalize to all build flows and bitfile synth
            layer_categories = ["MAC", "Eltwise", "Thresholding", "FIFO", "DWC", "SWG", "Other"]
            for category in layer_categories:
                dvc_logger.log_nested_metrics_from_report(
                    "res_breakdown_build_output.json",
                    category,
                    [
                        "LUT",
                        "FF",
                        "SRL",
                        "DSP",
                        "BRAM_18K",
                        "BRAM_36K",
                        "URAM",
                    ],
                    prefix="synth(dcp)/resources(breakdown)/" + category + "/",
                )

            # ooc_synth_and_timing.json (OOC synth / step_out_of_context_synthesis)
            dvc_logger.log_metrics_from_report(
                "ooc_synth_and_timing.json",
                [
                    "LUT",
                    "LUTRAM",
                    "FF",
                    "DSP",
                    "BRAM",
                    "BRAM_18K",
                    "BRAM_36K",
                    "URAM",
                ],
                prefix="synth(ooc)/resources/",
            )
            dvc_logger.log_metrics_from_report(
                "ooc_synth_and_timing.json",
                [
                    "WNS",
                    "fmax_mhz",
                    # add TNS? what is "delay"?
                ],
                prefix="synth(ooc)/timing/",
            )

            # post_synth_resources.json (shell synth / step_synthesize_bitfile)
            # special handling for microbenchmarks to extract only the relevant layer
            report_hierarchy_level = "(top)"
            if metadata_bench["params"]["dut"] == "mvau":
                resource_report = open_json_report(id, "post_synth_resources.json", args.followup)
                if resource_report:
                    for key in resource_report:
                        if "MVAU" in key:
                            report_hierarchy_level = key
                            break
                    if report_hierarchy_level == "(top)":
                        print("ERROR: No MVAU found in post_synth_resources.json")
                        sys.exit(1)
            # TODO: also do this for other reports or make it optional/configurable

            dvc_logger.log_nested_metrics_from_report(
                "post_synth_resources.json",
                report_hierarchy_level,
                [
                    "LUT",
                    "FF",
                    "SRL",
                    "DSP",
                    "BRAM_18K",
                    "BRAM_36K",
                    "URAM",
                ],
                prefix="synth/resources/",
            )

            # post synth timing report
            # TODO: only exported as post_route_timing.rpt, not .json

            # TODO: update collection of all measurement reports after recent driver changes
            # instrumentation measurement
            # dvc_logger.log_all_metrics_from_report(
            #     "measured_performance.json", prefix="measurement/performance/"
            # )

            # IODMA validation accuracy
            # dvc_logger.log_metrics_from_report(
            #     "validation.json",
            #     [
            #         "top-1_accuracy",
            #     ],
            #     prefix="measurement/validation/",
            # )

            # power estimation
            dvc_logger.log_all_metrics_from_report(
                "power_estimate_summary.json", prefix="vivado_estimate/power/"
            )

            # power measurement
            # dvc_logger.log_all_metrics_from_report(
            #     "measured_power.json", prefix="measurement/power/"
            # )

            # live fifosizing report + graph png
            # dvc_logger.log_metrics_from_report(
            #     "fifo_sizing_report.json",
            #     [
            #         "error",
            #         "fifo_size_total_kB",
            #     ],
            #     prefix="fifosizing/live/",
            # )

            # image = os.path.join(
            #     "measurement_artifacts",
            #     "runs_output",
            #     "run_%d" % (id),
            #     "reports",
            #     "fifo_sizing_graph.png",
            # )
            # if os.path.isfile(image):
            #     dvc_logger.log_image("fifosizing_pass_1", image)

            # time_per_step.json
            dvc_logger.log_all_metrics_from_report("time_per_step.json", prefix="time/")

            # ARTIFACTS
            # Log build reports as they come from GitLab artifacts,
            # but copy them to a central dir first so all runs share the same path
            if args.followup:
                run_report_dir1 = os.path.join(
                    "build_artifacts_followup", "runs_output", "run_%d" % (id), "reports"
                )
                run_report_dir2 = os.path.join(
                    "measurement_artifacts_followup", "runs_output", "run_%d" % (id), "reports"
                )
            else:
                run_report_dir1 = os.path.join(
                    "build_artifacts", "runs_output", "run_%d" % (id), "reports"
                )
                run_report_dir2 = os.path.join(
                    "measurement_artifacts", "runs_output", "run_%d" % (id), "reports"
                )
            dvc_report_dir = "reports"
            os.makedirs(dvc_report_dir, exist_ok=True)
            delete_dir_contents(dvc_report_dir)
            if os.path.isdir(run_report_dir1):
                shutil.copytree(run_report_dir1, dvc_report_dir, dirs_exist_ok=True)
            if os.path.isdir(run_report_dir2):
                shutil.copytree(run_report_dir2, dvc_report_dir, dirs_exist_ok=True)
            dvc_logger.log_artifact(dvc_report_dir)

            # Save microbenchmark results in a list per DUT for later aggregation
            dut = params["params"]["dut"]
            if dut not in microbench_result_data:
                # Initialize data dict for this DUT
                microbench_result_data[dut] = list()
            microbench_result_data[dut].append(dvc_logger.data_dict)

        # Prepare benchmarking config for follow-up runs after live FIFO-sizing
        # Only generate follow-up config if this is not already a follow-up run
        if not args.followup:
            # Choose the search order with the lowest fifo_size_total_kB
            lfs_base_dir = os.path.join(
                "measurement_artifacts",
                "runs_output",
                "run_%d" % (id),
                "reports",
                "experiment_fifosizing",
                "exp_itr_1",
            )
            best_search_order = None
            best_fifo_size = float("inf")
            if os.path.isdir(lfs_base_dir):
                for search_order in os.listdir(lfs_base_dir):
                    sizing_report_path = os.path.join(
                        lfs_base_dir, search_order, "both", "fifo_sizing_report.json"
                    )
                    if os.path.isfile(sizing_report_path):
                        with open(sizing_report_path, "r") as f:
                            sizing_report = json.load(f)
                        fifo_size = sizing_report.get("fifo_size_total_kB", float("inf"))
                        if fifo_size < best_fifo_size:
                            best_fifo_size = fifo_size
                            best_search_order = search_order
            if best_search_order is not None:
                print(
                    "Selecting search order '%s' with fifo_size_total_kB=%.2f"
                    % (best_search_order, best_fifo_size)
                )
                folding_config_lfs_path = os.path.join(
                    lfs_base_dir, best_search_order, "both", "folding_config_lfs.json"
                )
            else:
                print(
                    "No valid search order with fifo_sizing_report.json found in %s." % lfs_base_dir
                )
                folding_config_lfs_path = None

            if folding_config_lfs_path is not None and os.path.isfile(folding_config_lfs_path):
                print(
                    "Creating follow-up experiment config based on lfs folding config: %s"
                    % folding_config_lfs_path
                )

                # Create benchmarking config
                metadata_bench = open_json_report(id, "metadata_bench.json", args.followup)
                configuration = dict()
                for key in metadata_bench["params"]:
                    # wrap in list
                    configuration[key] = [metadata_bench["params"][key]]
                # overwrite FIFO-related params
                configuration["live_fifo_sizing"] = [False]
                configuration["auto_fifo_depths"] = [False]
                configuration["target_fps"] = ["None"]
                configuration["folding_config_file"] = [folding_config_lfs_path]

                # Exception for ResNet-50: Final model doesn't fit board used for FIFO-sizing
                if "dut" in metadata_bench["params"]:
                    if metadata_bench["params"]["dut"] == "resnet50":
                        configuration["board"] = ["U250"]
                        configuration["enable_instrumentation"] = [False]
                        configuration["rtlsim_batch_size"] = [3]
                        configuration["generate_outputs"] = [
                            ["stitched_ip", "rtlsim_performance", "bitfile"]
                        ]

                follow_up_bench_cfg.append(configuration)

    # Save microbenchmark results as (DVC-tracked? TODO) JSON for each DUT
    for dut in microbench_result_data:
        if None not in microbench_result_data[dut]:
            # dut_dir = os.path.join("ci", "benchmark_data", dut) TODO
            dut_dir = os.path.join(os.environ.get("LOCAL_BENCHMARK_DIR_STORE"), dut)
            os.makedirs(dut_dir, exist_ok=True)
            dut_json_path = os.path.join(
                dut_dir,
                date.today().strftime("%Y-%m-%d")
                + "_"
                + os.environ.get("CI_COMMIT_SHORT_SHA")
                + "_"
                + os.environ.get("CI_PIPELINE_ID")
                + "_"
                + str(len(microbench_result_data[dut]))
                + ".json",
            )
            dut_json = {
                "dut": dut,
                "date": date.today().strftime("%Y-%m-%d"),
                "commit": os.environ.get("CI_COMMIT_SHA"),
                "pipeline_id": os.environ.get("CI_PIPELINE_ID"),
                "pipeline_name": os.environ.get("CI_PIPELINE_NAME"),
                "runs": microbench_result_data[dut],
            }
            print("Saving microbenchmark results for %s to %s" % (dut, dut_json_path))
            with open(dut_json_path, "w") as f:
                json.dump(dut_json, f, indent=2)

    # Save aggregated benchmarking config for follow-up job to working dir
    # It is forwarded to the follow-up job via GitLab CI artifact
    if follow_up_bench_cfg:
        followup_artifact_path = "followup_bench_config.json"
        print("Saving follow-up bench config as artifact: %s" % followup_artifact_path)
        with open(followup_artifact_path, "w") as f:
            json.dump(follow_up_bench_cfg, f, indent=2)

    print("Done")

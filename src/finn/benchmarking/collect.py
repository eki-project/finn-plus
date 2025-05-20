import json
import os
import shutil
from dvclive.live import Live

from finn.benchmarking.util import delete_dir_contents


def log_dvc_metric(live, prefix, name, value):
    # sanitize '/' in name because DVC uses it to nest metrics (which we do via prefix)
    live.log_metric(prefix + name.replace("/", "-"), value, plot=False)

def open_json_report(id, report_name):
    # look in both, build & measurement, artifacts
    path1 = os.path.join("build_artifacts", "runs_output", "run_%d" % (id), "reports", report_name)
    path2 = os.path.join("measurement_artifacts", "runs_output", "run_%d" % (id), "reports", report_name)
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

def log_all_metrics_from_report(id, live, report_name, prefix=""):
    report = open_json_report(id, report_name)
    if report:
        for key in report:
            log_dvc_metric(live, prefix, key, report[key])

def log_metrics_from_report(id, live, report_name, keys, prefix=""):
    report = open_json_report(id, report_name)
    if report:
        for key in keys:
            if key in report:
                log_dvc_metric(live, prefix, key, report[key])

def log_nested_metrics_from_report(id, live, report_name, key_top, keys, prefix=""):
    report = open_json_report(id, report_name)
    if report:
        if key_top in report:
            for key in keys:
                if key in report[key_top]:
                    log_dvc_metric(live, prefix, key, report[key_top][key])

if __name__ == "__main__":
    # Go through all runs found in the artifacts and log their results to DVC
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
    # Prepare (local) output directory where follow-up bench configs will be stored
    output_cfg_dir = os.path.join(os.environ.get("LOCAL_CFG_DIR_STORE"), "lfs", "CI_" + os.environ.get("CI_PIPELINE_ID"))
    output_folding_dir = os.path.join(output_cfg_dir, "folding")
    output_cfg_path = os.path.join(output_cfg_dir, "follow-up.json")

    for id in run_ids:
        print("Processing run %d" % id)
        experiment_name = "CI_" + os.environ.get("CI_PIPELINE_ID") + "_" + str(id)
        experiment_msg = "[CI] " + os.environ.get("CI_PIPELINE_NAME") + " (" + os.environ.get("CI_PIPELINE_ID") + "_" + str(id) + ")"
        #TODO: cache images once we switch to a cache provider that works with DVC Studio
        with Live(exp_name = experiment_name, exp_message=experiment_msg, cache_images=False) as live:
            ### PARAMS ###
            # input parameters logged by benchmarking infrastructure
            metadata_bench = open_json_report(id, "metadata_bench.json")   
            params = {"params": metadata_bench["params"]}
            live.log_params(params)

            # optional metadata logged by builder
            metadata_builder = open_json_report(id, "metadata_builder.json")
            if metadata_builder:
                metadata = {
                    "metadata": {
                        "tool_version": metadata_builder["tool_version"],
                    }
                }
                live.log_params(metadata)

            # optional dut_info.json (additional information about DUT generated during model generation)
            dut_info_report = open_json_report(id, "dut_info.json")
            if dut_info_report:
                dut_info = {"dut_info": dut_info_report}
                live.log_params(dut_info)

            ### METRICS ###
            # TODO: for microbenchmarks, only summarize results for target node (or surrounding SDP?) (see old step_finn_estimate etc.)
            # TODO: make all logs consistent at the point of generation (e.g. BRAM vs BRAM18 vs BRAM36)

            # status
            status = metadata_bench["status"]
            if status == "ok":
                # mark as failed if either bench or builder indicates failure
                if metadata_builder:
                    status_builder = metadata_builder["status"]
                    if status_builder == "failed":
                        status = "failed"
            log_dvc_metric(live, "", "status", status)

            # verification steps
            if "output" in metadata_bench:
                if "builder_verification" in metadata_bench["output"]:
                    log_dvc_metric(live, "", "verification", metadata_bench["output"]["builder_verification"]["verification"])

            # estimate_layer_resources.json
            log_nested_metrics_from_report(id, live, "estimate_layer_resources.json", "total", [
                "LUT",
                "DSP",
                "BRAM_18K",
                "URAM",
                ], prefix="estimate/resources/")

            # estimate_layer_resources_hls.json
            log_nested_metrics_from_report(id, live, "estimate_layer_resources_hls.json", "total", [
                "LUT",
                "FF",
                "DSP",
                "DSP48E",
                "DSP58E", # TODO: aggregate/unify DSP reporting
                "BRAM_18K",
                "URAM",
                ], prefix="hls_estimate/resources/")

            # estimate_network_performance.json
            log_metrics_from_report(id, live, "estimate_network_performance.json", [
                "critical_path_cycles",
                "max_cycles",
                "max_cycles_node_name",
                "estimated_throughput_fps",
                "estimated_latency_ns",
                ], prefix="estimate/performance/")

            # rtlsim_performance.json
            log_metrics_from_report(id, live, "rtlsim_performance.json", [
                "N",
                "TIMEOUT",
                "latency_cycles",
                "cycles",
                "fclk[mhz]",
                "throughput[images/s]",
                "stable_throughput[images/s]",
                # add INPUT_DONE, OUTPUT_DONE, number transactions?
                ], prefix="rtlsim/performance/")

            # fifo_sizing.json
            log_metrics_from_report(id, live, "fifo_sizing.json", ["total_fifo_size_kB"], prefix="fifosizing/")

            # stitched IP DCP synth resource report
            log_nested_metrics_from_report(id, live, "post_synth_resources_dcp.json", "(top)", [
                "LUT",
                "FF",
                "SRL",
                "DSP",
                "BRAM_18K",
                "BRAM_36K",
                "URAM",
                ], prefix="synth(dcp)/resources/")

            # stitched IP DCP synth resource breakdown
            # TODO: generalize to all build flows and bitfile synth
            layer_categories = ["MAC", "Eltwise", "Thresholding", "FIFO", "DWC", "SWG", "Other"]
            for category in layer_categories:
                log_nested_metrics_from_report(id, live, "res_breakdown_build_output.json", category, [
                    "LUT",
                    "FF",
                    "SRL",
                    "DSP",
                    "BRAM_18K",
                    "BRAM_36K",
                    "URAM",
                    ], prefix="synth(dcp)/resources(breakdown)/" + category + "/")

            # ooc_synth_and_timing.json (OOC synth / step_out_of_context_synthesis)
            log_metrics_from_report(id, live, "ooc_synth_and_timing.json", [
                "LUT",
                "LUTRAM",
                "FF",
                "DSP",
                "BRAM",
                "BRAM_18K",
                "BRAM_36K",
                "URAM",
                ], prefix="synth(ooc)/resources/")
            log_metrics_from_report(id, live, "ooc_synth_and_timing.json", [
                "WNS",
                "fmax_mhz",
                # add TNS? what is "delay"?
                ], prefix="synth(ooc)/timing/")

            # post_synth_resources.json (shell synth / step_synthesize_bitfile)
            log_nested_metrics_from_report(id, live, "post_synth_resources.json", "(top)", [
                "LUT",
                "FF",
                "SRL",
                "DSP",
                "BRAM_18K",
                "BRAM_36K",
                "URAM",
                ], prefix="synth/resources/")

            # post synth timing report 
            # TODO: only exported as post_route_timing.rpt, not .json

            # instrumentation measurement
            log_all_metrics_from_report(id, live, "measured_performance.json", prefix="measurement/performance/")

            # IODMA validation accuracy
            log_metrics_from_report(id, live, "validation.json", [
                "top-1_accuracy",
                ], prefix="measurement/validation/")

            # power measurement
            # TODO

            # live fifosizing report + graph png
            log_metrics_from_report(id, live, "fifo_sizing_report.json", [
                "error",
                "fifo_size_total_kB",
                ], prefix="fifosizing/live/")

            image = os.path.join("measurement_artifacts", "runs_output", "run_%d" % (id), "reports", "fifo_sizing_graph.png")
            if os.path.isfile(image):
                live.log_image("fifosizing_pass_1", image)

            # time_per_step.json
            log_metrics_from_report(id, live, "time_per_step.json", ["total_build_time"])

            ### ARTIFACTS ###
            # Log build reports as they come from GitLab artifacts,
            # but copy them to a central dir first so all runs share the same path
            run_report_dir1 = os.path.join("build_artifacts", "runs_output", "run_%d" % (id), "reports")
            run_report_dir2 = os.path.join("measurement_artifacts", "runs_output", "run_%d" % (id), "reports")
            dvc_report_dir = "reports"
            os.makedirs(dvc_report_dir, exist_ok=True)
            delete_dir_contents(dvc_report_dir)
            if os.path.isdir(run_report_dir1):
                shutil.copytree(run_report_dir1, dvc_report_dir, dirs_exist_ok=True)
            if os.path.isdir(run_report_dir2):
                shutil.copytree(run_report_dir2, dvc_report_dir, dirs_exist_ok=True)
            live.log_artifact(dvc_report_dir)

        # Prepare benchmarking config for follow-up runs after live FIFO-sizing
        folding_config_lfs_path = os.path.join("measurement_artifacts", "runs_output", "run_%d" % (id), "reports", "folding_config_lfs.json")
        if os.path.isfile(folding_config_lfs_path):
            # Copy folding config produced by live FIFO-sizing
            output_folding_path = os.path.join(output_folding_dir, experiment_name + ".json")
            os.makedirs(output_folding_dir, exist_ok=True)
            print("Saving lfs-generated folding config of this run to use in a future follow-up run: %s" % output_folding_path)
            shutil.copy(folding_config_lfs_path, output_folding_path)

            # Create benchmarking config
            metadata_bench = open_json_report(id, "metadata_bench.json")   
            configuration = dict()
            for key in metadata_bench["params"]:
                # wrap in list
                configuration[key] = [metadata_bench["params"][key]]
            # overwrite FIFO-related params
            import_folding_path = os.path.join(os.environ.get("LOCAL_CFG_DIR"), "lfs", "CI_" + os.environ.get("CI_PIPELINE_ID"), "folding", experiment_name + ".json")
            configuration["fifo_method"] = ["manual"]
            configuration["target_fps"] = ["None"]
            configuration["folding_path"] = [import_folding_path]

            follow_up_bench_cfg.append(configuration)

    # Save aggregated benchmarking config for follow-up job
    if follow_up_bench_cfg:
        print("Saving follow-up bench config for lfs: %s" % output_cfg_path)
        with open(output_cfg_path, "w") as f:
            json.dump(follow_up_bench_cfg, f, indent=2)

    print("Done")

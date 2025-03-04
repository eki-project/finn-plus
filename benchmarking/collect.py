import itertools
import json
import os
import sys
import time
import shutil
from dvclive import Live

from util import delete_dir_contents

def merge_dicts(a: dict, b: dict):
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key])
            elif a[key] != b[key]:
                raise Exception("ERROR: Dict merge conflict")
        else:
            a[key] = b[key]
    return a

def consolidate_logs(path, output_filepath):
    log = []
    i = 0
    while (i < 1024):
        if (os.path.isfile(os.path.join(path,"task_%d.json"%(i)))):
            with open(os.path.join(path,"task_%d.json"%(i)), "r") as f:
                log_task = json.load(f)
            log.extend(log_task)
        i = i + 1
    
    with open(output_filepath, "w") as f:
        json.dump(log, f, indent=2)

def merge_logs(log_a, log_b, log_out):
    # merges json log (list of nested dicts) b into a, not vice versa (TODO)

    with open(log_a, "r") as f:
        a = json.load(f)
    with open(log_b, "r") as f:
        b = json.load(f)

    for idx, run_a in enumerate(a):
        for run_b in b:
            if run_a["run_id"] == run_b["run_id"]:
                #a[idx] |= run_b # requires Python >= 3.9
                #a[idx] = {**run_a, **run_b}
                a[idx] = merge_dicts(run_a, run_b)
                break

    # also sort by run id
    out = sorted(a, key=lambda x: x["run_id"])

    with open(log_out, "w") as f:
        json.dump(out, f, indent=2)

def wait_for_power_measurements():
    # TODO: detect when no bitstreams are to be measured (e.g. for fifosizing) and skip
    # TODO: make configurable, relative to some env variable due to different mountint points
    bitstreams_path = os.path.join("/mnt/pfs/hpc-prf-radioml/felix/jobs/", 
                            "CI_" + os.environ.get("CI_PIPELINE_IID") + "_" + os.environ.get("CI_PIPELINE_NAME"), 
                            "bitstreams")
    
    power_log_path = os.path.join("/mnt/pfs/hpc-prf-radioml/felix/jobs/", 
                            "CI_" + os.environ.get("CI_PIPELINE_IID") + "_" + os.environ.get("CI_PIPELINE_NAME"), 
                            "power_measure.json")

    # count bitstreams to measure (can't rely on total number of runs since some of them could've failed)
    files = os.listdir(bitstreams_path)
    bitstream_count = len(list(filter(lambda x : ".bit" in x, files)))

    log = []
    print("Checking if all bitstreams of pipeline have been measured..")
    while(len(log) < bitstream_count):
        if os.path.isfile(power_log_path):
            with open(power_log_path, "r") as f:
                log = json.load(f)
        print("Found measurements for %d/%d bitstreams"%(len(log),bitstream_count))
        time.sleep(60)
    print("Power measurement complete")

def log_dvc_metric(live, prefix, name, value):
    # sanitize '/' in name because DVC uses it to nest metrics (which we do via prefix)
    live.log_metric(prefix + name.replace("/", "-"), value, plot=False)

def open_json_report(id, report_name):
    path = os.path.join("bench_artifacts", "runs_output", "run_%d" % (id), "reports", report_name)
    if os.path.isfile(path):
        with open(path, "r") as f:
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
    run_dir_list = os.listdir(os.path.join("bench_artifacts", "runs_output"))
    print("Looking for runs in %s" % run_dir_list)
    run_ids = []
    for run_dir in run_dir_list:
        if run_dir.startswith("run_"):
            run_id = int(run_dir[4:])
            run_ids.append(run_id)
    run_ids.sort()
    print("Found %d runs" % len(run_ids))

    for id in run_ids:
        print("Processing run %d" % id)
        experiment_name = "CI_" + os.environ.get("CI_PIPELINE_ID") + "_" + str(id)
        experiment_msg = "[CI] " + os.environ.get("CI_PIPELINE_NAME")
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

            # power measurement
            # TODO

            # live fifosizing report + graph png
            log_metrics_from_report(id, live, "fifo_sizing_report.json", [
                "error",
                "fifo_size_total_kB",
                ], prefix="fifosizing/live/")

            image = os.path.join("bench_artifacts", "runs_output", "run_%d" % (id), "reports", "fifo_sizing_graph.png")
            if os.path.isfile(image):
                live.log_image("fifosizing_pass_1", image)

            # time_per_step.json
            log_metrics_from_report(id, live, "time_per_step.json", ["total_build_time"])

            ### ARTIFACTS ###
            # Log build reports as they come from GitLab artifacts,
            # but copy them to a central dir first so all runs share the same path
            run_report_dir = os.path.join("bench_artifacts", "runs_output", "run_%d" % (id), "reports")
            dvc_report_dir = "reports"
            os.makedirs(dvc_report_dir, exist_ok=True)
            delete_dir_contents(dvc_report_dir)
            shutil.copytree(run_report_dir, dvc_report_dir, dirs_exist_ok=True)
            live.log_artifact(dvc_report_dir)

    print("Done")

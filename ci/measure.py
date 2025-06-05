import json
import os
import pandas as pd
import shutil
import subprocess
import sys


def delete_dir_contents(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


if __name__ == "__main__":
    exit_code = 0
    print("Looking for deployment packages in artifacts..")
    # Find deployment packages from artifacts
    artifacts_in_dir = os.path.join("build_artifacts", "runs_output")
    artifacts_out_dir = os.path.join("measurement_artifacts", "runs_output")
    for run in os.listdir(artifacts_in_dir):
        run_in_dir = os.path.join(artifacts_in_dir, run)
        run_out_dir = os.path.join(artifacts_out_dir, run)
        reports_dir = os.path.join(run_out_dir, "reports")
        deploy_archive = os.path.join(run_in_dir, "deploy.zip")
        extract_dir = "measurement"
        if os.path.isfile(deploy_archive):
            print("Found deployment package in %s, extracting.." % run_in_dir)

            # Extract to temporary dir
            os.makedirs(extract_dir, exist_ok=True)
            delete_dir_contents(extract_dir)
            shutil.unpack_archive(deploy_archive, extract_dir)

            # Run driver
            print("Running measurement manager..")
            # run validate.py (from IODMA driver) if present, otherwise driver.py (instrumentation)
            # TODO: unify IODMA/instrumentation shell & driver
            if os.path.isfile(f"{extract_dir}/driver/validate.py"):
                driver_file = f"{extract_dir}/driver/validate.py"
                driver_args = {
                    "settingsfile": f"{extract_dir}/driver/settings.json",
                    "reportfile": f"{extract_dir}/validation.json",
                    "dataset_root": "/home/xilinx/datasets",  # TODO: env var
                    "batchsize": 100,
                    "platform": "zynq-iodma",
                    "runtime": 30,  # only relevant for idle baseline run
                    "frequency": 100.0,  # will be overwritten by settingsfile (TODO)
                    "device": 0,  # TODO: unnecessary?
                }
            else:
                driver_file = f"{extract_dir}/driver/driver.py"
                driver_args = {
                    "settingsfile": f"{extract_dir}/driver/settings.json",
                    "reportfile": f"{extract_dir}/measured_performance.json",
                    "runtime": 30,  # not relevant for live FIFO-Sizing
                    "frequency": 100.0,  # will be overwritten by settingsfile (TODO)
                    "seed": 1,
                    "device": 0,
                }

            # Generate (static) measurement config
            # TODO: make this configurable (e.g., to run accelerator at multiple frequencies)
            # TODO: make base configuration board-specific
            measurement_cfg = {
                "experimentDesc": "Automatically triggered flow",
                "global": {
                    "FINN": {
                        "driver": f"{extract_dir}/driver",
                        "bitstream": f"{extract_dir}/bitfile/finn-accel.bit",
                    },
                    "PAF": {
                        "rails": [
                            "0V85",
                            "1V2_PL",
                            "1V8",
                            "3V3",
                            "2V5_DC",
                            "1V2_PS",
                            "1V1_DC",
                            "3V5_DC",
                            "5V0_DC",
                        ],
                        "sensors": [],
                        "board": "rfsoc2x2",
                    },
                    "experiment_path": driver_file,
                    "import_paths": [],
                    "report_path": f"{extract_dir}",
                },
                "experiments": [
                    {
                        "title": "idle",
                        "functions": [{"name": "run_idle", "args": [], "kwargs": driver_args}],
                        "num_runs": 1,
                        "warmup": 10,
                    },
                    {
                        "title": "load",
                        "functions": [{"name": "main", "args": [], "kwargs": driver_args}],
                        "num_runs": 1,
                        "warmup": 10,
                    },
                ],
            }
            with open(f"{extract_dir}/measurement_config.json", "w") as f:
                json.dump(measurement_cfg, f, indent=2)

            # Launch experiment manager with generated config
            sys.stdout.flush()
            result = subprocess.run(
                [
                    sys.executable,
                    "ci/power_measurement/measurement_manager.py",
                    f"{extract_dir}/measurement_config.json",
                ]
            )

            if result.returncode != 0:
                print("Measurement manager reported error!")
                exit_code = 1
            else:
                print("Measurement finished successfully.")

            # parse power measurement results into a compact report
            # TODO: aggregate results from multiple runs
            # TODO: make aggregation board-specific
            df = pd.read_excel(os.path.join(extract_dir, "idle_run_1.xlsx"))
            power_pl_ps_idle = round(df["0V85_power"].mean() * 0.001, 3)
            power_total_idle = round(df["total_power"].mean() * 0.001, 3)
            df = pd.read_excel(os.path.join(extract_dir, "load_run_1.xlsx"))
            power_pl_ps_load = round(df["0V85_power"].mean() * 0.001, 3)
            power_total_load = round(df["total_power"].mean() * 0.001, 3)

            power_pl_ps_dyn_load_comp = round(power_pl_ps_load - power_pl_ps_idle, 3)
            power_total_dyn_load_comp = round(power_total_load - power_total_idle, 3)

            power_report = {
                "power_pl_ps_load": power_pl_ps_load,
                "power_pl_ps_idle": power_pl_ps_idle,
                "power_pl_ps_dyn": power_pl_ps_dyn_load_comp,
                "power_total_load": power_total_load,
                "power_total_idle": power_total_idle,
                "power_total_dyn": power_total_dyn_load_comp,
            }
            power_log_path = os.path.join(extract_dir, "measured_power.json")
            with open(power_log_path, "w") as f:
                json.dump(power_report, f, indent=2)

            # Copy results back to artifact directory
            for report in [
                "measured_performance.json",
                "measured_power.json",
                "fifo_sizing_report.json",
                "fifo_depth_export.json",
                "fifo_sizing_graph.png",
                "folding_config_lfs.json",
                "validation.json",
            ]:
                report_path = os.path.join(extract_dir, report)
                if os.path.isfile(report_path):
                    print("Copying %s to %s" % (report_path, reports_dir))
                    os.makedirs(reports_dir, exist_ok=True)
                    shutil.copy(report_path, reports_dir)

            print("Clearing temporary directory..")
            # Clear temporary dir
            delete_dir_contents(extract_dir)
            print("Done.")
    print("Processed all deployment packages.")
    sys.exit(exit_code)

import os
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
            shutil.unpack_archive(deploy_archive, extract_dir)

            # Run driver
            print("Running driver..")
            # run validate.py (from IODMA driver) if present, otherwise driver.py (instrumentation)
            # TODO: unify IODMA/instrumentation shell & driver
            if os.path.isfile(f"{extract_dir}/driver/validate.py"):
                result = subprocess.run(
                    [
                        "python",
                        f"{extract_dir}/driver/validate.py",
                        "--bitfile",
                        f"{extract_dir}/bitfile/finn-accel.bit",
                        "--settingsfile",
                        f"{extract_dir}/driver/settings.json",
                        "--reportfile",
                        f"{extract_dir}/validation.json",
                        "--dataset_root",
                        "/home/xilinx/datasets",  # TODO: env var
                    ]
                )
            else:
                result = subprocess.run(
                    [
                        "python",
                        f"{extract_dir}/driver/driver.py",
                        "--bitfile",
                        f"{extract_dir}/bitfile/finn-accel.bit",
                        "--settingsfile",
                        f"{extract_dir}/driver/settings.json",
                        "--reportfile",
                        f"{extract_dir}/measured_performance.json",
                    ]
                )
            if result.returncode != 0:
                print("Driver reported error!")
                exit_code = 1
            else:
                print("Driver finished successfully.")

            # Copy results back to artifact directory
            for report in [
                "measured_performance.json",
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

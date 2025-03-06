import os
import subprocess
import shutil

from util import delete_dir_contents


if __name__ == "__main__":
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
            subprocess.run(["python", f"{extract_dir}/driver/driver.py",
                            "--bitfile",  f"{extract_dir}/bitfile/finn-accel.bit",
                            "--settingsfile", f"{extract_dir}/driver/settings.json",
                            "--reportfile", f"{extract_dir}/measured_performance.json",
                            ]) 
            print("Driver finished.")

            # Copy results back to artifact directory
            for report in ["measured_performance.json", 
                           "fifo_sizing_report.json",
                           "fifo_depth_export.json",
                           "fifo_sizing_graph.png",
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

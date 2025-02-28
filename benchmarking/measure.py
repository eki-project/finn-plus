import os
import subprocess
import shutil


def delete_dir_contents(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

if __name__ == "__main__":
    print("Looking for deployment packages in artifacts..")
    # Find deployment packages from artifacts
    artifacts_dir = os.path.join("bench_artifacts", "runs_output")
    for run in os.listdir(artifacts_dir):
        run_dir = os.path.join(artifacts_dir, run)
        reports_dir = os.path.join(run_dir, "reports")
        deploy_archive = os.path.join(run_dir, "deploy.zip")
        extract_dir = "measurement"
        if os.path.isfile(deploy_archive):
            print("Found deployment package in %s, extracting.." % run_dir)

            # Extract to temporary dir
            shutil.unpack_archive(deploy_archive, extract_dir)

            # Run driver
            print("Running driver..")
            subprocess.run(["sudo", "python", f"{extract_dir}/driver/driver.py",
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
                    shutil.copy(report_path, reports_dir)

            print("Clearing temporary directory..")
            # Clear temporary dir
            delete_dir_contents(extract_dir)
            print("Done.")

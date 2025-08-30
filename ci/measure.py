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
            print("ERROR: Failed to delete %s. Reason: %s" % (file_path, e))


if __name__ == "__main__":
    exit_code = 0
    print(f"SCANNING DEPLOYMENT PACKAGES IN BUILD ARTIFACTS {os.getcwd()}..")
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
            print("FOUND DEPLOYMENT PACKAGE IN %s, EXTRACTING.." % run_in_dir)

            # Extract to temporary dir
            os.makedirs(extract_dir, exist_ok=True)
            delete_dir_contents(extract_dir)
            shutil.unpack_archive(deploy_archive, extract_dir)

            # Prefix stdout to make it easier to identify the run in the console output
            print(
                "LAUNCHING MEASUREMENT MANAGER FOR DEPLOY PACKAGE: %s"
                % os.path.basename(run_in_dir)
            )
            sys.stdout.flush()

            # Launch experiment manager with generated config
            result = subprocess.run(
                [
                    sys.executable,
                    "ci/power_measurement/expirment_manager.py",
                    os.path.join(extract_dir, "driver/settings.json"),
                    extract_dir,
                ],
                capture_output=True,
                text=True,
            )

            for line in result.stdout.splitlines():
                print(f"{line}")
            for line in result.stderr.splitlines():
                print(f"{line}")
            if result.returncode != 0:
                print("ERROR: MEASUREMENT MANAGER NON-ZERO EXIT CODE!")
                exit_code = 1
            else:
                print("MEASUREMENT MANAGER COMPLETED SUCCESSFULLY.")

            report_path = os.path.join(extract_dir, "report")
            shutil.copytree(report_path, reports_dir, dirs_exist_ok=True)

            delete_dir_contents(extract_dir)

    print("PROCESSED ALL DEPLOYMENT PACKAGES. EXITING..")
    sys.exit(exit_code)

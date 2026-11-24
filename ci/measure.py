"""CI measurement script for FINN deployment packages."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def delete_dir_contents(directory: str | Path) -> None:
    """Delete all contents of a directory."""
    for file_path in Path(directory).iterdir():
        try:
            if file_path.is_file() or file_path.is_symlink():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)
        except Exception as e:  # noqa: PERF203  (one failure must not stop the cleanup)
            print(f"ERROR: Failed to delete {file_path}. Reason: {e}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run measurements on FINN deployment packages.")
    parser.add_argument(
        "--followup",
        action="store_true",
        help="Indicate this is a follow-up run (uses different artifact directories)",
    )
    args = parser.parse_args()

    exit_code = 0
    print("SCANNING DEPLOYMENT PACKAGES IN BUILD ARTIFACTS..")
    # Find deployment packages from artifacts
    if args.followup:
        artifacts_in_dir = Path("build_artifacts_followup") / "runs_output"
        artifacts_out_dir = Path("measurement_artifacts_followup") / "runs_output"
    else:
        artifacts_in_dir = Path("build_artifacts") / "runs_output"
        artifacts_out_dir = Path("measurement_artifacts") / "runs_output"
    for run_in_dir in artifacts_in_dir.iterdir():
        run = run_in_dir.name
        run_out_dir = artifacts_out_dir / run
        reports_dir = run_out_dir / "reports"
        deploy_archive = run_in_dir / "deploy.zip"
        extract_dir = "measurement"
        if deploy_archive.is_file():
            print(f"FOUND DEPLOYMENT PACKAGE IN {run_in_dir}, EXTRACTING..")

            # Extract to temporary dir
            Path(extract_dir).mkdir(parents=True, exist_ok=True)
            delete_dir_contents(extract_dir)
            shutil.unpack_archive(deploy_archive, extract_dir)

            # Prefix stdout to make it easier to identify the run in the console output
            print(f"LAUNCHING MEASUREMENT MANAGER FOR DEPLOY PACKAGE: {run_in_dir.name}")
            sys.stdout.flush()

            # Launch experiment manager with generated config
            result = subprocess.run(
                [
                    sys.executable,
                    "ci/power_measurement/experiment_manager.py",
                    str(Path(extract_dir) / "driver/settings.json"),
                    extract_dir,
                ],
                capture_output=True,
                text=True,
            )

            for line in result.stdout.splitlines():
                print(f"[{run_in_dir.name}] {line}")
            for line in result.stderr.splitlines():
                print(f"[{run_in_dir.name}] {line}")
            if result.returncode != 0:
                print(f"ERROR: MEASUREMENT MANAGER NON-ZERO EXIT CODE ({result.returncode})!")
                exit_code = 1
            else:
                print("MEASUREMENT MANAGER COMPLETED SUCCESSFULLY.")

            # Collect whatever reports were produced. A failed measurement may not have
            # written any, which must not abort the remaining runs or discard the
            # artifacts of the runs that did succeed.
            report_path = Path(extract_dir) / "report"
            if report_path.is_dir():
                reports_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(report_path, reports_dir, dirs_exist_ok=True)
            else:
                print(f"WARNING: No report directory found in {run}, nothing to collect.")
                exit_code = 1

            delete_dir_contents(extract_dir)

    print("PROCESSED ALL DEPLOYMENT PACKAGES. EXITING..")
    sys.exit(exit_code)

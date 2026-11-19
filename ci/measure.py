"""CI measurement script for FINN deployment packages.

Runs on the FPGA board runner. Deployment packages (bitfile + driver) are read from the
build artifacts and the measurement reports are written next to them. The location is
resolved by ``finn.benchmarking.exchange``: the shared exchange directory if
``FINN_BENCH_EXCHANGE_DIR`` is set (CI), else ``build_artifacts/`` in the working directory
(GitLab artifacts / local use). A small ``measurement_summary.json`` is always written to
the working directory for the GitLab artifact.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from finn.benchmarking import exchange  # noqa: E402


def _previous_measurement_ok(run_out_dir):
    """True if ``run_out_dir`` carries a DONE marker of an earlier measurement with status ok."""
    marker = run_out_dir / exchange.DONE_MARKER
    if not marker.is_file():
        return False
    try:
        with open(marker) as f:
            return json.load(f).get("status") == "ok"
    except (OSError, ValueError):
        return False


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
            print("ERROR: Failed to delete %s. Reason: %s" % (file_path, e))


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
    print(exchange.describe())
    exchange.set_shared_umask()
    artifacts_in_dir = exchange.artifacts_dir("build", args.followup)
    artifacts_out_dir = exchange.artifacts_dir("measurement", args.followup)
    summary_name = "measurement_summary%s.json" % ("_followup" if args.followup else "")
    summary = {}

    # An absent build directory is legitimate only if the pipeline directory itself is there
    # (every build succeeded or failed before writing artifacts, or a follow-up without
    # runs). A missing pipeline directory means the share is not visible from the board
    # (e.g. not mounted after a reboot) and must fail the job instead of passing silently.
    pipeline_dir = exchange.pipeline_exchange_dir()
    if pipeline_dir is not None and not (pipeline_dir / exchange.CREATED_MARKER).is_file():
        print(
            "ERROR: pipeline directory %s not found on the exchange share (is %s mounted?)"
            % (pipeline_dir, exchange.exchange_root())
        )
        sys.exit(1)
    if not artifacts_in_dir.is_dir():
        print("No build artifacts found in %s, nothing to measure" % artifacts_in_dir)
        with open(summary_name, "w") as f:
            json.dump(summary, f, indent=2)
        sys.exit(0)

    exchange.ensure_dir(artifacts_out_dir)
    exchange.check_writable(artifacts_out_dir)

    print("SCANNING DEPLOYMENT PACKAGES IN BUILD ARTIFACTS (%s).." % artifacts_in_dir)
    # Only runs marked as complete by the build job are measured (see exchange.mark_done)
    for run_id in exchange.list_run_ids("build", args.followup):
        run_in_dir = exchange.run_dir("build", run_id, args.followup)
        run_out_dir = exchange.run_dir("measurement", run_id, args.followup)
        reports_dir = run_out_dir / "reports"
        deploy_archive = run_in_dir / "deploy.zip"
        extract_dir = "measurement"
        if not deploy_archive.is_file():
            continue
        # A retried job resumes: runs whose previous measurement completed (DONE marker with
        # status ok) keep their reports, so an infrastructure failure (share outage, board
        # reboot) after hundreds of runs does not cost a full re-measurement
        if _previous_measurement_ok(run_out_dir):
            print("ALREADY MEASURED %s (DONE marker), SKIPPING.." % run_in_dir.name)
            summary[f"run_{run_id}"] = "ok"
            continue
        print("FOUND DEPLOYMENT PACKAGE IN %s, EXTRACTING.." % run_in_dir)

        # Extract to a temporary dir on the local disk (never unpack onto the share)
        os.makedirs(extract_dir, exist_ok=True)
        delete_dir_contents(extract_dir)
        t_start = time.monotonic()
        shutil.unpack_archive(str(deploy_archive), extract_dir)
        t_unpacked = time.monotonic()

        # Prefix stdout to make it easier to identify the run in the console output
        print("LAUNCHING MEASUREMENT MANAGER FOR DEPLOY PACKAGE: %s" % run_in_dir.name)
        sys.stdout.flush()

        # Launch experiment manager with generated config
        result = subprocess.run(
            [
                sys.executable,
                "ci/power_measurement/experiment_manager.py",
                os.path.join(extract_dir, "driver/settings.json"),
                extract_dir,
            ],
            capture_output=True,
            text=True,
        )

        t_measured = time.monotonic()
        for line in result.stdout.splitlines():
            print(f"[{run_in_dir.name}] {line}")
        for line in result.stderr.splitlines():
            print(f"[{run_in_dir.name}] {line}")
        status = "ok"
        if result.returncode != 0:
            print("ERROR: MEASUREMENT MANAGER NON-ZERO EXIT CODE (%d)!" % result.returncode)
            exit_code = 1
            status = "failed"
        else:
            print("MEASUREMENT MANAGER COMPLETED SUCCESSFULLY.")

        # Collect whatever reports were produced. A failed measurement may not have
        # written any, which must not abort the remaining runs or discard the
        # artifacts of the runs that did succeed.
        report_path = os.path.join(extract_dir, "report")
        if os.path.isdir(report_path):
            os.makedirs(reports_dir, exist_ok=True)
            shutil.copytree(report_path, reports_dir, dirs_exist_ok=True)
        else:
            print("WARNING: No report directory found in run_%d, nothing to collect." % run_id)
            exit_code = 1
            status = "failed"

        exchange.mark_done(run_out_dir, status, returncode=result.returncode)
        # Per-phase wall-clock of this run (the board trace carries no timestamps): where the
        # time goes between the experiments themselves and the share/board overhead
        print(
            "TIMING %s: unpack %.1f s, measure %.1f s, collect %.1f s (%s)"
            % (
                run_in_dir.name,
                t_unpacked - t_start,
                t_measured - t_unpacked,
                time.monotonic() - t_measured,
                time.strftime("%Y-%m-%d %H:%M:%S"),
            )
        )
        exchange.chown_to_sudo_user(run_out_dir)
        summary[f"run_{run_id}"] = status
        delete_dir_contents(extract_dir)

    with open(summary_name, "w") as f:
        json.dump(summary, f, indent=2)
    print("PROCESSED ALL DEPLOYMENT PACKAGES (%d). EXITING.." % len(summary))
    sys.exit(exit_code)

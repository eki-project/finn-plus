import os
import shlex
import subprocess
import sys
from pathlib import Path

from interface import IS_POSIX


def run_test(variant: str, num_workers: str) -> None:
    """Run a given test variant with the given number of workers"""
    original_dir = Path.cwd()

    # TODO: Make this optional
    ci_project_dir = os.environ["CI_PROJECT_DIR"]

    os.chdir(Path(__file__).parent.parent)
    python_prefix = str(Path(os.environ["VIRTUAL_ENV"]) / "bin" / "python3")
    match variant:
        case "quick":
            subprocess.run(
                shlex.split(
                    f"{python_prefix} -m pytest -m 'not "
                    f"(vivado or slow or vitis or board or notebooks or bnn_pynq)' "
                    f"--dist=loadfile -n {num_workers}",
                    posix=IS_POSIX,
                )
            )
        case "main":
            subprocess.run(
                shlex.split(
                    f"{python_prefix} -m pytest -k 'not (rtlsim or end2end)' "
                    f"--dist=loadfile -n {num_workers}",
                    posix=IS_POSIX,
                )
            )
        case "rtlsim":
            subprocess.run(
                shlex.split(
                    f"{python_prefix} -m pytest -k rtlsim --workers {num_workers}", posix=IS_POSIX
                )
            )
        case "end2end":
            subprocess.run("pytest -k end2end", shell=True)
        case "full":
            test_1_process = subprocess.Popen(
                shlex.split(
                    (
                        f"{python_prefix} -m pytest -m 'not (end2end or sanity_bnn or notebooks)' "
                        f"--junitxml={ci_project_dir}/reports/main.xml "
                        f"--html={ci_project_dir}/reports/main.html "
                        f"--reruns 1 --dist worksteal -n {num_workers}"
                    ),
                    posix=IS_POSIX,
                )
            )
            test_2_process = subprocess.Popen(
                shlex.split(
                    (
                        f"{python_prefix} -m pytest -m 'end2end or sanity_bnn or notebooks' "
                        f"--junitxml={ci_project_dir}/reports/end2end.xml "
                        f"--html={ci_project_dir}/reports/end2end.html "
                        f"--reruns 1 --dist loadgroup -n {num_workers}"
                    ),
                    posix=IS_POSIX,
                )
            )
            test_1_process.communicate()
            test_1_returncode = test_1_process.returncode
            test_2_process.communicate()
            test_2_returncode = test_2_process.returncode

            subprocess.run(
                shlex.split(
                    (
                        f"{python_prefix} -m pytest_html_merger -i {ci_project_dir}/reports/ "
                        f"-o {ci_project_dir}/reports/full_test_suite.html"
                    ),
                    posix=IS_POSIX,
                )
            )

            if test_1_returncode or test_2_returncode:
                sys.exit(1)

        case _:
            subprocess.run(shlex.split(f"{python_prefix} -m pytest -k '{variant}'", posix=IS_POSIX))
    os.chdir(original_dir)

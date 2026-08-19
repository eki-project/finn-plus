"""Manage FINNs testsuite."""
import os
import shlex
import subprocess
import sys
from pathlib import Path

from finn.interface import IS_POSIX
from finn.interface.interface_utils import status
from finn.util.exception import FINNUserError
from finn.util.settings import get_settings


def run_doctests(num_workers: int) -> bool:
    """Run all doctests in FINN and report if any failed."""
    returncodes = []
    tests = []
    for submodule in [
        "analysis",
        "builder",
        "core",
        "custom_op",
        "interface",
        "transformation",
        "util",
    ]:
        status(f"Running doctest on submodule finn.{submodule}")
        tests.append(
            subprocess.Popen(
                shlex.split(
                    f"{sys.executable} -m pytest --doctest-modules "
                    f"--doctest-continue-on-failure -n {num_workers} "
                    f"--pyargs finn." + submodule,
                    posix=IS_POSIX,
                )
            )
        )
        tests[-1].communicate()
    returncodes = [test.returncode for test in tests]
    return any(returncodes)


def run_test(variant: str, num_workers: str, name: str = "") -> None:
    """Run a given test variant with the given number of workers."""
    original_dir = Path.cwd()

    # TODO: Make this optional
    if "CI_PROJECT_DIR" in os.environ.keys():
        ci_project_dir = os.environ["CI_PROJECT_DIR"]
    else:
        ci_project_dir = str(get_settings().finn_build_dir)
    status(f"Putting test reports into {ci_project_dir}")

    os.chdir(os.environ["FINN_TESTS"])
    match variant:
        case "custom":
            if name == "":
                raise FINNUserError(
                    "--variant custom was specified, but no test was "
                    "given (please additionally pass --name "
                    "<test-name> in pytest syntax)"
                )
            subprocess.run(
                shlex.split(f"{sys.executable} -m pytest -n {num_workers} {name}", posix=IS_POSIX)
            )
        case "doctest":
            if name == "":
                status(
                    "No test name was specified, running "
                    "doctests on all relevant FINN submodules."
                )
                run_doctests(int(num_workers))
                return
            if name.endswith(".py"):
                raise FINNUserError(
                    "To run doctests, specify the name as a python module path. "
                    "Instead of src/finn/interface/manage_tests.py do "
                    "finn.interface.manage_tests!"
                )
            subprocess.run(
                shlex.split(
                    f"{sys.executable} -m pytest --doctest-modules "
                    f"--doctest-continue-on-failure -n {num_workers} "
                    f"--pyargs {name}",
                    posix=IS_POSIX,
                )
            )
        case "quick":
            subprocess.run(
                shlex.split(
                    f"{sys.executable} -m pytest -v -m 'not "
                    f"(vivado or slow or vitis or board or notebooks or bnn_pynq or end2end)' "
                    f"--dist=loadfile -n {num_workers}",
                    posix=IS_POSIX,
                )
            )
        case "quicktest_ci":
            subprocess.run(
                shlex.split(
                    f"{sys.executable} -m pytest -v -m 'not "
                    f"(vivado or slow or vitis or board or notebooks or bnn_pynq or end2end)' "
                    f"--junitxml={ci_project_dir}/reports/quick.xml "
                    f"--html={ci_project_dir}/reports/quick.html "
                    f"--reruns 1 --dist worksteal -n {num_workers}",
                    posix=IS_POSIX,
                )
            )
        case "full_ci":
            test_1_process = subprocess.Popen(
                shlex.split(
                    (
                        f"{sys.executable} -m pytest -v -m 'not "
                        f"(end2end or sanity_bnn or notebooks)' "
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
                        f"{sys.executable} -m pytest -v -m 'end2end or sanity_bnn or notebooks' "
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

            # Run doctests for all FINN submodules
            test_3_returncode = run_doctests(int(num_workers))

            subprocess.run(
                shlex.split(
                    (
                        f"{sys.executable} -m pytest_html_merger -i {ci_project_dir}/reports/ "
                        f"-o {ci_project_dir}/reports/full_test_suite.html"
                    ),
                    posix=IS_POSIX,
                )
            )

            if test_1_returncode or test_2_returncode or test_3_returncode:
                sys.exit(1)

        case _:
            subprocess.run(
                shlex.split(f"{sys.executable} -m pytest -k '{variant}'", posix=IS_POSIX)
            )
    os.chdir(original_dir)

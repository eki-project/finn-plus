"""Manage FINNs testsuite."""

import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path

from finn.interface import IS_POSIX
from finn.interface.interface_utils import status
from finn.util.exception import FINNUserError
from finn.util.settings import get_settings

# Python modules the test runner needs beyond the finn-plus package itself. They come with
# the "test" extra only, so a plain "pip install finn-plus" must not import any of them at
# module level, or the whole CLI breaks on start-up. They are checked here, when "finn test"
# is actually invoked (tests/util/test_import_hygiene.py guards the package against this).
_TEST_SUITE_MODULES = ("pytest", "xdist", "junitparser")


def resolve_tests_dir(tests_dir: Path | None) -> Path:
    """Locate the test suite and check that its tooling is installed.

    The suite is the tests/ directory of a repository checkout and is not part of the pip
    package. It is taken from the argument, or else from ./tests of the current directory.
    The environment is deliberately not consulted: a FINN_TESTS variable exported by a shell
    profile would silently point every other checkout or worktree at the wrong suite.
    Raises a FINNUserError explaining what is missing otherwise.
    """
    if tests_dir is None:
        tests_dir = Path.cwd() / "tests"
    tests_dir = tests_dir.expanduser().absolute()
    problems = []
    if not (tests_dir / "conftest.py").is_file():
        problems.append(
            f"no test suite found at {tests_dir}. The suite is the tests/ directory of "
            "a finn-plus repository checkout, pass it with --tests-path or run 'finn test' "
            "from the checkout root"
        )
    missing = [m for m in _TEST_SUITE_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        problems.append(
            "missing Python modules: " + ", ".join(missing) + ". Install the test tooling "
            "with 'poetry install' in a checkout or with 'pip install finn-plus[test]'"
        )
    if problems:
        raise FINNUserError("Cannot run the FINN+ test suite: " + "; ".join(problems))
    return tests_dir


def run_doctests(num_workers: int) -> bool:
    """Run all doctests in FINN and report if any failed."""
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
    # pytest exit code 5 means "no tests collected", not a failure
    return any(rc not in (0, 5) for rc in returncodes)


# Per-test wall-clock limit (setup + call + teardown) for the CI variants. The slowest
# tests of the suite take a bit over an hour, so this only ever fires for a test that
# is genuinely stuck (e.g. a deadlocked simulation), instead of the whole job idling
# until the Slurm time limit kills it without a report.
CI_TEST_TIMEOUT_S = 3 * 3600

# The "thread" method is required, not a preference: pytest-timeout's default ("signal")
# raises the timeout from a SIGALRM handler, which the interpreter only runs between
# bytecodes. A test blocked inside a C extension - which is where our hangs are, xsim
# simulation via XSI - never returns to the interpreter, so the alarm is never handled
# and the test hangs anyway. The timer thread dumps the stacks of all threads and kills
# the process; under pytest-xdist that surfaces as a crashed worker, whose test
# pytest-rerunfailures retries like any other failure.
# The stack dump pytest-timeout prints on the way out goes to the worker's captured
# stderr and never reaches the job log under xdist, so pytest's own faulthandler
# plugin dumps all threads a minute earlier through the C-level faulthandler on the
# real stderr fd, which does show up in the log (and works inside C extensions too).
CI_TEST_TIMEOUT_ARGS = (
    f"-o faulthandler_timeout={CI_TEST_TIMEOUT_S - 60} "
    f"--timeout {CI_TEST_TIMEOUT_S} --timeout-method=thread"
)


def run_test(variant: str, num_workers: str, tests_dir: Path, args: str = "") -> None:
    """Run a given test variant with the given number of workers.

    The test suite at tests_dir (see resolve_tests_dir) is run against the installed finn
    package: pytest picks up the pyproject.toml of the checkout as its configuration and
    puts the checkout root on sys.path for the tests package, while "import finn" resolves
    like in any other process of this interpreter.
    """
    original_dir = Path.cwd()

    # TODO: Make this optional
    if "CI_PROJECT_DIR" in os.environ.keys():
        ci_project_dir = os.environ["CI_PROJECT_DIR"]
    else:
        ci_project_dir = str(get_settings().finn_build_dir)
    status(f"Putting test reports into {ci_project_dir}")

    # Export the suite that is actually run, replacing any stale value from the shell
    os.environ["FINN_TESTS"] = str(tests_dir)
    os.chdir(tests_dir)
    match variant:
        case "custom":
            if args == "":
                raise FINNUserError(
                    "--variant custom was specified, but no test was "
                    "given (please additionally pass --args "
                    "<test-name> in pytest syntax)"
                )
            workers_arg = f"-n {num_workers}" if num_workers not in ["1", ""] else ""
            subprocess.run(
                shlex.split(f"{sys.executable} -m  pytest {workers_arg} {args}", posix=IS_POSIX)
            )
        case "doctest":
            if args == "":
                status(
                    "No test name was specified, running doctests on all relevant FINN submodules."
                )
                run_doctests(int(num_workers))
                return
            if args.endswith(".py"):
                raise FINNUserError(
                    "To run doctests, specify the name as a python module path. "
                    "Instead of src/finn/interface/manage_tests.py do "
                    "finn.interface.manage_tests!"
                )
            subprocess.run(
                shlex.split(
                    f"{sys.executable} -m pytest --doctest-modules "
                    f"--doctest-continue-on-failure -n {num_workers} "
                    f"--pyargs {args}",
                    posix=IS_POSIX,
                )
            )
        case "quick":
            subprocess.run(
                shlex.split(
                    f"{sys.executable} -m pytest -v -m 'not "
                    f"(vivado or slow or vitis or board or bnn_pynq or end2end or xslow)' "
                    f"--dist=loadfile -n {num_workers}",
                    posix=IS_POSIX,
                )
            )
        case "quicktest_ci":
            subprocess.run(
                shlex.split(
                    f"{sys.executable} -m pytest -q -rf --tb=short -m 'not "
                    f"(vivado or slow or vitis or board or bnn_pynq or end2end or xslow)' "
                    f"--junitxml={ci_project_dir}/reports/quick.xml "
                    f"--html={ci_project_dir}/reports/quick.html "
                    f"{CI_TEST_TIMEOUT_ARGS} "
                    f"--reruns 1 --dist worksteal -n {num_workers}",
                    posix=IS_POSIX,
                )
            )
        case "full_ci":
            main_xml = f"{ci_project_dir}/reports/main.xml"
            main_html = f"{ci_project_dir}/reports/main.html"

            # --------------------------
            # 1) Main suite
            # --------------------------

            test_1_process = subprocess.Popen(
                shlex.split(
                    (
                        f"{sys.executable} -m pytest -q -rf --tb=short -m 'not xslow' "
                        f"--junitxml={main_xml} "
                        f"--html={main_html} "
                        f"{CI_TEST_TIMEOUT_ARGS} "
                        f"--reruns 1 --dist worksteal -n {num_workers}"
                    ),
                    posix=IS_POSIX,
                )
            )
            test_1_process.communicate()
            test_1_returncode = test_1_process.returncode

            main_returncode = test_1_returncode

            # --------------------------
            # 2) Run doctests and merge all reports into a single HTML and XML report
            # --------------------------
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
            script_dir = tests_dir / "testing_util" / "merge_xml_reports.py"
            success = subprocess.run(
                shlex.split(
                    (
                        f"{sys.executable} {script_dir} "
                        f"-o {ci_project_dir}/reports/full_test_suite.xml "
                        f"{main_xml}"
                    ),
                    posix=IS_POSIX,
                )
            ).returncode
            if success != 0:
                print(f"[WARN] Merging XML reports failed with exit code {success}.")
            else:
                # Remove individual XML reports to avoid confusion with the merged report
                if Path(main_xml).exists():
                    Path(main_xml).unlink()

            if main_returncode or test_3_returncode:
                sys.exit(1)

        case _:
            subprocess.run(
                shlex.split(f"{sys.executable} -m pytest -k '{variant}'", posix=IS_POSIX)
            )
    os.chdir(original_dir)

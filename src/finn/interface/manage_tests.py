"""Manage FINNs testsuite."""

import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from junitparser import JUnitXml, TestCase
from pathlib import Path
from re import Pattern

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
                    "No test name was specified, running doctests on all relevant FINN submodules."
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
            main_xml = f"{ci_project_dir}/reports/main.xml"
            main_html = f"{ci_project_dir}/reports/main.html"
            crash_xml = f"{ci_project_dir}/reports/crash_rerun.xml"
            crash_html = f"{ci_project_dir}/reports/crash_rerun.html"
            end2end_xml = f"{ci_project_dir}/reports/end2end.xml"
            end2end_html = f"{ci_project_dir}/reports/end2end.html"

            crash_re: Pattern[str] = re.compile(
                r"(worker.*crash|worker.*terminated|segmentation fault|sigsegv|signal 11|fatal python error)",  # noqa
                re.IGNORECASE,
            )

            @dataclass(frozen=True)
            class CrashDetectionResult:
                """Result of scanning a JUnit XML report for crash-like and non-crash failures."""

                crashed_nodeids: list[str]
                has_non_crash_failures: bool

            def make_nodeid(case: TestCase) -> str:
                """Return pytest-like nodeid from junit testcase."""
                classname: str = getattr(case, "classname", "") or ""
                name: str = getattr(case, "name", "") or ""
                return f"{classname}::{name}" if classname else name

            def detect_crashed_nodeids(junit_xml_path: str) -> CrashDetectionResult:
                """Parse a JUnit XML file and classify failing testcases.

                A testcase is considered:
                - crash-like failure: if any failure/error message matches CRASH_RE
                - non-crash failure: failure/error exists but no crash marker matched

                Skipped/passed cases are ignored for failure classification.

                Args:
                    junit_xml_path: Path to JUnit XML file.

                Returns:
                    CrashDetectionResult with:
                    - crashed_nodeids: deduplicated list of crash-like testcase nodeids
                    - has_non_crash_failures: True if any failure/error is non-crash-like
                """
                xml = JUnitXml.fromfile(junit_xml_path)

                crashed_nodeids: list[str] = []
                seen: set[str] = set()
                has_non_crash_failures = False

                for suite in xml:
                    for case in suite:
                        # Collect failure/error blocks only (ignore skipped)
                        failure_blocks: list[str] = []
                        for res in case.result:
                            tag = getattr(res, "_tag", "")
                            if tag not in {"failure", "error"}:
                                continue
                            msg: str = getattr(res, "message", "") or ""
                            txt: str = getattr(res, "text", "") or ""
                            failure_blocks.append(f"{msg}\n{txt}".strip())

                        if not failure_blocks:
                            continue  # passed or skipped-only

                        combined_failure_text = "\n".join(failure_blocks)
                        is_crash_like = bool(crash_re.search(combined_failure_text))

                        if is_crash_like:
                            nodeid = make_nodeid(case)
                            if nodeid and nodeid not in seen:
                                seen.add(nodeid)
                                crashed_nodeids.append(nodeid)
                        else:
                            has_non_crash_failures = True

                return CrashDetectionResult(
                    crashed_nodeids=crashed_nodeids,
                    has_non_crash_failures=has_non_crash_failures,
                )

            # --------------------------
            # 1) Main suite
            # --------------------------

            test_1_process = subprocess.Popen(
                shlex.split(
                    (
                        f"{sys.executable} -m pytest -v -m 'not "
                        f"(end2end or sanity_bnn or notebooks)' "
                        f"--junitxml={main_xml} "
                        f"--html={main_html} "
                        f"--reruns 1 --dist worksteal -n {num_workers}"
                    ),
                    posix=IS_POSIX,
                )
            )
            test_1_process.communicate()
            test_1_returncode = test_1_process.returncode

            # --------------------------
            # 2) Detect crashed tests
            # --------------------------
            crashed_tests: list[str] = []
            if Path(main_xml).exists():
                try:
                    result = detect_crashed_nodeids(main_xml)
                    crashed_tests = result.crashed_nodeids
                    has_non_crash_failures = result.has_non_crash_failures
                    test_1_returncode = 1 if has_non_crash_failures else 0
                except Exception as exc:
                    print(f"[WARN] Failed to parse {main_xml}: {exc}")
            else:
                print(f"[WARN] Main XML not found: {main_xml}")

            print(f"[INFO] Crashed tests detected: {len(crashed_tests)}")

            # --------------------------
            # 3) Rerun only crashed tests
            # --------------------------
            test_2_returncode = 0
            if crashed_tests:
                nodeids = " ".join(shlex.quote(t) for t in crashed_tests)
                rerun_cmd = (
                    f"{sys.executable} -m pytest -v "
                    f"--junitxml={shlex.quote(crash_xml)} "
                    f"--html={shlex.quote(crash_html)} "
                    f"--reruns 3 -n 1 "
                    f"{nodeids}"
                )
                test_2_process = subprocess.Popen(
                    shlex.split(
                        rerun_cmd,
                        posix=IS_POSIX,
                    )
                )
                test_2_process.communicate()
                test_2_returncode = test_2_process.returncode
            else:
                print("[INFO] No crash-like tests to rerun.")
            main_returncode = test_1_returncode or test_2_returncode

            # --------------------------
            # 4) Run end2end tests
            # --------------------------

            test_3_process = subprocess.Popen(
                shlex.split(
                    (
                        f"{sys.executable} -m pytest -v -m 'end2end or sanity_bnn or notebooks' "
                        f"--junitxml={end2end_xml} "
                        f"--html={end2end_html} "
                        f"--reruns 1 --dist loadgroup -n {num_workers}"
                    ),
                    posix=IS_POSIX,
                )
            )
            test_3_process.communicate()
            test_3_returncode = test_3_process.returncode

            # --------------------------
            # 5) Run doctests and merge all reports into a single HTML and XML report
            # --------------------------
            test_4_returncode = run_doctests(int(num_workers))

            subprocess.run(
                shlex.split(
                    (
                        f"{sys.executable} -m pytest_html_merger -i {ci_project_dir}/reports/ "
                        f"-o {ci_project_dir}/reports/full_test_suite.html"
                    ),
                    posix=IS_POSIX,
                )
            )
            script_dir = Path(get_settings().finn_tests) / "testing_util" / "merge_xml_reports.py"
            success = subprocess.run(
                shlex.split(
                    (
                        f"{sys.executable} {script_dir} "
                        f"-o {ci_project_dir}/reports/full_test_suite.xml "
                        f"{main_xml} {crash_xml} {end2end_xml}"
                    ),
                    posix=IS_POSIX,
                )
            ).returncode
            if success != 0:
                print(f"[WARN] Merging XML reports failed with exit code {success}.")
            else:
                # Remove individual XML reports to avoid confusion with the merged report
                for xml_file in [main_xml, crash_xml, end2end_xml]:
                    if Path(xml_file).exists():
                        Path(xml_file).unlink()

            if main_returncode or test_2_returncode or test_3_returncode or test_4_returncode:
                sys.exit(1)

        case _:
            subprocess.run(
                shlex.split(f"{sys.executable} -m pytest -k '{variant}'", posix=IS_POSIX)
            )
    os.chdir(original_dir)

"""Tests for the JUnit report merger used by the CI test suite."""

import pytest

from pathlib import Path
from testing_util.merge_xml_reports import merge_reports
from xml.etree import ElementTree as ET

RERUN_CASE = """
<testcase classname="pkg.TestX" name="test_a" time="1.0">
  <rerun message="mip.exceptions.InterfacingError: Gurobi environment could not be loaded">
  traceback of the first attempt
  </rerun>
</testcase>
"""
FAILED_CASE = """
<testcase classname="pkg.TestX" name="test_a" time="2.0">
  <failure message="mip.exceptions.InterfacingError: Gurobi environment could not be loaded">
  traceback of the final attempt
  </failure>
</testcase>
"""
PASSED_CASE = '<testcase classname="pkg.TestX" name="test_a" time="3.0"/>'
OTHER_CASE = '<testcase classname="pkg.TestX" name="test_b" time="0.1"/>'


def write_report(path: Path, *cases: str) -> str:
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" '
        f'tests="{len(cases)}">{"".join(cases)}</testsuite></testsuites>'
    )
    return str(path)


def merged_status(output: Path, name: str) -> str:
    for tc in ET.parse(output).getroot().iter("testcase"):
        if tc.get("name") == name:
            tags = {child.tag for child in tc}
            return next(iter(tags)) if tags else "passed"
    raise AssertionError(f"{name} not in merged report")


@pytest.mark.infrastructure
def test_rerun_attempt_does_not_hide_final_failure(tmp_path: Path) -> None:
    """pytest-rerunfailures writes the retried attempt as its own testcase carrying a <rerun>
    child. That entry must not be taken for a passed test and must not win over the final
    outcome of the test, regardless of the order in which the entries appear.
    """
    out = tmp_path / "merged.xml"
    merge_reports([write_report(tmp_path / "main.xml", RERUN_CASE, FAILED_CASE, OTHER_CASE)], out)
    assert merged_status(out, "test_a") == "failure"
    assert merged_status(out, "test_b") == "passed"

    merge_reports([write_report(tmp_path / "main2.xml", FAILED_CASE, RERUN_CASE)], out)
    assert merged_status(out, "test_a") == "failure"


@pytest.mark.infrastructure
def test_rerun_attempt_then_pass(tmp_path: Path) -> None:
    out = tmp_path / "merged.xml"
    merge_reports([write_report(tmp_path / "main.xml", RERUN_CASE, PASSED_CASE)], out)
    assert merged_status(out, "test_a") == "passed"


@pytest.mark.infrastructure
def test_later_pass_replaces_earlier_failure(tmp_path: Path) -> None:
    """A crash rerun in a later report file turns an earlier failure into a pass."""
    out = tmp_path / "merged.xml"
    merge_reports(
        [
            write_report(tmp_path / "main.xml", FAILED_CASE),
            write_report(tmp_path / "rerun.xml", PASSED_CASE),
        ],
        out,
    )
    assert merged_status(out, "test_a") == "passed"

    # ... but a failure in a later file never replaces a pass.
    merge_reports(
        [
            write_report(tmp_path / "main3.xml", PASSED_CASE),
            write_report(tmp_path / "rerun3.xml", FAILED_CASE),
        ],
        out,
    )
    assert merged_status(out, "test_a") == "passed"

#!/usr/bin/env python3
"""Merge arbitrary many pytest JUnit XML files with this rule:
- Identity key: (classname, name)
- Within one file, a testcase that appears several times was retried (pytest-rerunfailures):
  the attempts come first and the final outcome last, so the last entry wins. An attempt is
  either marked <rerun> or, with pytest-rerunfailures 16 and pytest 9, a testcase without any
  result element that merely looks like a pass; a marked attempt never wins over a final
  outcome.
- Across files (main run, crash rerun), for the same testcase:
  - PASSED always wins over FAILED/ERROR/SKIPPED
  - A retried attempt (<rerun>) never wins over a final outcome
  - Otherwise, latest file wins (input order).

Usage:
    python merge_junit_prefer_pass.py -o merged.xml main.xml rerun1.xml rerun2.xml
"""

import argparse
import sys
from collections import OrderedDict
from collections.abc import Iterable
from junitparser import JUnitXml, TestSuite
from junitparser.junitparser import TestCase
from pathlib import Path
from typing import Literal

TestStatus = Literal["passed", "failed", "skipped", "rerun", "unknown"]

# Result children that pytest-rerunfailures writes for every attempt that was retried. Such an
# entry describes an attempt that did not count, never the final outcome of the test.
RERUN_TAGS = {"rerun", "flakyFailure", "flakyError"}
TestKey = tuple[str, str]


def testcase_key(tc: TestCase) -> TestKey:
    """Build the identity key for a testcase.

    Args:
        tc: TestCase object parsed from JUnit XML.

    Returns:
        Tuple of (classname, name). Missing attributes default to empty strings.
    """
    classname: str = getattr(tc, "classname", "") or ""
    name: str = getattr(tc, "name", "") or ""
    return (classname, name)


def testcase_status(tc: TestCase) -> TestStatus:
    """Derive a normalized status from a JUnit testcase.

    Status rules (pytest JUnit convention):
    - passed: no result children
    - failed: contains <failure> or <error>
    - skipped: contains <skipped>
    - rerun: contains only <rerun>/<flakyFailure>/<flakyError> (pytest-rerunfailures writes one
      such testcase per retried attempt, next to the testcase holding the final outcome)
    - unknown: any other non-empty result shape

    Args:
        tc: TestCase object.

    Returns:
        One of: "passed", "failed", "skipped", "rerun", "unknown".
    """
    # Look at the raw children: junitparser's `result` only knows failure/error/skipped and
    # would report a rerun attempt as having no result, i.e. as passed.
    tags = {child.tag for child in tc._elem}  # noqa: SLF001
    tags.discard("system-out")
    tags.discard("system-err")
    tags.discard("properties")
    if not tags:
        return "passed"

    if "failure" in tags or "error" in tags:
        return "failed"
    if "skipped" in tags:
        return "skipped"
    if tags <= RERUN_TAGS:
        return "rerun"
    return "unknown"


def should_replace(existing_tc: TestCase, new_tc: TestCase) -> bool:
    """Decide whether a newly seen testcase should replace the currently stored one.

    Priority:
    1) PASSED always wins over non-passed.
    2) If existing is passed, never replace with non-passed.
    3) A rerun attempt never replaces a final outcome, and any final outcome replaces it.
    4) If neither side is passed, latest file wins (replace with new).

    Args:
        existing_tc: Previously stored testcase for the same key.
        new_tc: Newly encountered testcase for the same key.

    Returns:
        True if new_tc should replace existing_tc, else False.
    """
    old_status = testcase_status(existing_tc)
    new_status = testcase_status(new_tc)

    if old_status == "passed":
        return False
    if new_status == "passed":
        return True
    if new_status == "rerun":
        return old_status == "rerun"  # keep the final outcome, refresh a mere attempt
    return True  # latest wins if no pass involved


def should_replace_within_file(existing_tc: TestCase, new_tc: TestCase) -> bool:
    """Decide whether a later entry of the same testcase in the same file replaces an earlier one.

    Entries of one testcase in one file are the attempts of pytest-rerunfailures in
    chronological order, so the later entry is the more final one. Only an entry that is
    marked as a mere attempt (<rerun>) does not replace an earlier final outcome.

    Args:
        existing_tc: Earlier entry of the testcase in the file.
        new_tc: Later entry of the testcase in the file.

    Returns:
        True if new_tc should replace existing_tc, else False.
    """
    if testcase_status(new_tc) == "rerun":
        return testcase_status(existing_tc) == "rerun"
    return True


def merge_reports(inputs: Iterable[str], output: str) -> None:
    """Merge multiple JUnit XML reports into one output report.

    Input order is chronological (earlier -> later).
    For duplicate test keys, replacement follows `should_replace`.

    Args:
        inputs: Iterable of input XML file paths.
        output: Output XML file path.

    Returns:
        None
    """
    by_key: OrderedDict[TestKey, TestCase] = OrderedDict()

    for path in inputs:
        if not Path(path).is_file():
            continue
        xml = JUnitXml.fromfile(path)
        # Reruns within one file: the final attempt comes last and is the outcome of the
        # test, whatever the earlier attempts look like (a result-less attempt is not a pass)
        in_file: OrderedDict[TestKey, TestCase] = OrderedDict()
        for suite in xml:
            for tc in suite:
                key = testcase_key(tc)
                if key not in in_file or should_replace_within_file(in_file[key], tc):
                    in_file[key] = tc
        for key, tc in in_file.items():
            if key not in by_key or should_replace(by_key[key], tc):
                by_key[key] = tc

    merged = JUnitXml()
    out_suite = TestSuite("merged")

    for tc in by_key.values():
        out_suite.add_testcase(tc)

    merged.add_testsuite(out_suite)
    merged.write(output)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Merge JUnit XML files, always preferring PASSED results."
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output merged XML path.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input XML files in chronological order (earlier -> later).",
    )
    return parser


def main() -> int:
    """CLI entrypoint.

    Returns:
        Process exit code (0 on success, non-zero on error).
    """
    parser = build_parser()
    args = parser.parse_args()

    try:
        merge_reports(args.inputs, args.output)
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"Merged {len(args.inputs)} files into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

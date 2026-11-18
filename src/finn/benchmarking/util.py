"""Utility functions for benchmarking and report processing."""

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from finn.util.exception import FINNInternalError


def _find_rows_and_headers(
    table: ET.Element,
) -> tuple[list[ET.Element], list[ET.Element]]:
    """Find table rows and headers in XML table structure.

    Searches through table rows to find the first row that contains
    table headers, which are used to identify column structure.

    Args:
        table: XML table element to parse

    Returns:
        tuple: (list of all table rows, list of header elements)
    """
    rows = table.findall("tablerow")
    headers = []

    for row in rows:
        headers = row.findall("tableheader")
        if len(headers) > 0:
            break
    return (rows, headers)


def summarize_table(table: ET.Element) -> dict[str, Any]:
    """Summarize table data into a structured dictionary format.

    Parses XML table structure to extract headers and row data,
    organizing the information into a summary dictionary for easier
    processing and analysis of benchmarking results.

    Args:
        table: XML table element to summarize

    Returns:
        dict: Summary containing headers and processed row data
    """
    table_summary: dict[str, Any] = {}
    table_summary["headers"] = []
    rows, headers = _find_rows_and_headers(table)

    if len(headers) > 0:
        for header in headers:
            table_summary["headers"].append(header.attrib["contents"])

    for row in rows:
        cells = row.findall("tablecell")
        if len(cells) > 0:
            cell_name = cells[0].attrib["contents"]
            table_summary[cell_name] = []
            for cell in cells[1:]:
                table_summary[cell_name].append(cell.attrib["contents"])

    return table_summary


def summarize_section(section: ET.Element) -> dict[str, Any]:
    """Summarize report section."""
    section_summary: dict[str, Any] = {}
    section_summary["tables"] = []
    section_summary["subsections"] = {}

    tables = section.findall("table")
    sub_sections = section.findall("section")
    for table in tables:
        section_summary["tables"].append(summarize_table(table))
    for sub_section in sub_sections:
        section_summary["subsections"][sub_section.attrib["title"]] = summarize_section(sub_section)

    return section_summary


def power_xml_to_dict(xml_path: str | Path) -> dict[str, Any]:
    """Convert power XML to dictionary."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sections = root.findall("section")
    result = {}

    for section in sections:
        result[section.attrib["title"]] = summarize_section(section)

    return result


def delete_dir_contents(dir_path: str | Path) -> None:
    """Delete directory contents."""
    dir_path = Path(dir_path)
    for entry in dir_path.iterdir():
        try:
            if entry.is_file() or entry.is_symlink():
                entry.unlink()
            elif entry.is_dir():
                shutil.rmtree(entry)
        except Exception as e:  # noqa: PERF203
            print(f"Failed to delete {entry}. Reason: {e}")


def merge_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple dictionaries."""
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key])
            elif a[key] != b[key]:
                raise FINNInternalError("ERROR: Dict merge conflict")
        else:
            a[key] = b[key]
    return a


def merge_logs(log_a: str | Path, log_b: str | Path, log_out: str | Path) -> None:
    """Merge log files."""
    # merges json log (list of nested dicts) b into a, not vice versa (TODO)

    with Path(log_a).open() as f:
        a = json.load(f)
    with Path(log_b).open() as f:
        b = json.load(f)

    for idx, run_a in enumerate(a):
        for run_b in b:
            if run_a["run_id"] == run_b["run_id"]:
                a[idx] = merge_dicts(run_a, run_b)
                break

    # also sort by run id
    out = sorted(a, key=lambda x: x["run_id"])

    with Path(log_out).open("w") as f:
        json.dump(out, f, indent=2)

"""Reporting for the automatic floorplanner.

The Vivado-side engine writes what it allocated; this module joins that with what
the Python side decided -- which nodes are in each unit, and which sizing pass the
numbers came from -- and renders a floorplan diagram.

The renderer handles both this feature's report and the DFX flow's
`pr_region_resources.json`, since the two describe the same thing at different
granularities: named regions with SLICE-coordinate footprints on one device.
"""

import json
import re
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from typing import Any, cast

from finn.transformation.fpgadataflow.floorplanning.binding import (
    UNASSIGNED_SLR,
    iter_floorplanned_models,
)
from finn.transformation.fpgadataflow.floorplanning.grouping import is_placeable_node

from finn.util.logging import log

#: Matches one SLICE range inside a Vivado GRID_RANGES string.
SLICE_RANGE_PATTERN: re.Pattern[str] = re.compile(r"SLICE_X(\d+)Y(\d+):SLICE_X(\d+)Y(\d+)")

#: Pixels per SLICE coordinate unit in the rendered diagram.
SVG_SCALE: int = 3

#: Border padding, in pixels.
SVG_PAD: int = 30

#: Base font size, in points.
SVG_FONT: int = 11

#: Per-SLR fill colours, cycled if a device somehow has more SLRs than entries.
SLR_COLOURS: tuple[str, ...] = ("#4c9be8", "#e8864c", "#4ce87c", "#e84c9b", "#9b4ce8")

#: Fallback palette for reports that carry no SLR information (the DFX report).
REGION_COLOURS: tuple[str, ...] = (
    "#4c9be8",
    "#e8864c",
    "#4ce87c",
    "#e84c9b",
    "#9b4ce8",
    "#e8e04c",
    "#4ce8d8",
    "#e84c4c",
)

#: A region is only labelled when its rectangle is at least this many pixels across
#: and tall. At per-layer granularity most are not, and unreadable overlapping text
#: makes the diagram worse, not better.
MIN_LABEL_W: int = 46
MIN_LABEL_H: int = 14


def enrich_floorplan_report(model: ModelWrapper, report: dict[str, Any]) -> dict[str, Any]:
    """Join the Vivado engine's report with what the model knows about each unit.

    Args:
        model: The floorplanned model, or the parent of the SDP children that carry
            the floorplan attrs.
        report: The JSON the Vivado-side engine wrote.

    Returns:
        The same report with each unit gaining the node names it holds.

        The engine's own `resource_source` is left alone: it records how the pblock
        geometry was sized, which only the engine knows, and it is not the same fact
        as how SLR assignment was decided (always from estimates, since kernels are
        pinned to SLRs before anything is synthesized).
    """
    by_pblock: dict[str, dict[str, Any]] = {}
    for submodel in iter_floorplanned_models(model):
        for node in submodel.graph.node:
            if not is_placeable_node(node):
                continue
            inst = getCustomOp(node)
            pblock = cast("str", inst.get_nodeattr("floorplan_pblock"))
            if not pblock:
                continue
            entry = by_pblock.setdefault(pblock, {"nodes": []})
            entry["nodes"].append(node.name)
            slr = cast("int", inst.get_nodeattr("floorplan_slr"))
            if slr != UNASSIGNED_SLR:
                entry["slr"] = slr

    enriched = dict(report)
    units = dict(enriched.get("units", {}))
    for name, extra in by_pblock.items():
        unit = dict(units.get(name, {}))
        unit["nodes"] = extra["nodes"]
        unit.setdefault("slr", extra.get("slr", UNASSIGNED_SLR))
        units[name] = unit
    enriched["units"] = units
    return enriched


def _regions_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract drawable regions from either report shape.

    Returns one entry per region with its name, SLR (or -1) and the list of SLICE
    rectangles it occupies. Vivado splits a pblock's SLICE range wherever the span
    crosses a column that holds no SLICE sites, so a region routinely has several.
    """
    raw = report.get("units") or report.get("pr_regions") or {}
    regions = []
    for name, data in raw.items():
        rects = [
            (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
            for m in SLICE_RANGE_PATTERN.finditer(data.get("grid_ranges", "") or "")
        ]
        if not rects:
            continue
        regions.append({"name": name, "slr": data.get("slr", UNASSIGNED_SLR), "rects": rects})
    return regions


def _escape(text: str) -> str:
    """Escape the characters that would break out of SVG text content."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_floorplan_svg(report: dict[str, Any]) -> str | None:
    """Render a floorplan report as an SVG diagram.

    Regions are drawn to scale in SLICE coordinates against the full device array,
    with the Y axis flipped so that FPGA Y=0 (bottom of die) is at the bottom of the
    image. Where the report carries SLR assignments, regions are coloured by SLR and
    the SLR boundaries are drawn, since at per-layer granularity which SLR a layer
    landed in is the thing worth seeing at a glance.

    Args:
        report: A floorplanner report or a DFX `pr_region_resources` report.

    Returns:
        The SVG document, or None when the report holds nothing drawable.
    """
    regions = _regions_from_report(report)
    if not regions:
        return None

    dev_max_x = int(report.get("device_slice_max_x", 0))
    dev_max_y = int(report.get("device_slice_max_y", 0))
    for region in regions:
        for _, _, x1, y1 in region["rects"]:
            dev_max_x = max(dev_max_x, x1)
            dev_max_y = max(dev_max_y, y1)

    canvas_w = (dev_max_x + 1) * SVG_SCALE + 2 * SVG_PAD
    canvas_h = (dev_max_y + 1) * SVG_SCALE + 2 * SVG_PAD
    by_slr = sorted({int(r["slr"]) for r in regions if int(r["slr"]) != UNASSIGNED_SLR})

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}">',
        (
            f'<rect x="{SVG_PAD}" y="{SVG_PAD}"'
            f' width="{(dev_max_x + 1) * SVG_SCALE}" height="{(dev_max_y + 1) * SVG_SCALE}"'
            f' fill="#d8d8d8" stroke="#555" stroke-width="1"/>'
        ),
        (
            f'<text transform="rotate(-90)" x="-{canvas_h // 2}" y="{SVG_PAD - 6}"'
            f' text-anchor="middle" font-size="{SVG_FONT}"'
            f' font-family="sans-serif">SLICE Y</text>'
        ),
        (
            f'<text x="{canvas_w // 2}" y="{canvas_h - 4}"'
            f' text-anchor="middle" font-size="{SVG_FONT}"'
            f' font-family="sans-serif">SLICE X</text>'
        ),
    ]

    def flip(y: int) -> int:
        """Convert an FPGA Y coordinate to an SVG one."""
        return SVG_PAD + (dev_max_y - y) * SVG_SCALE

    for index, region in enumerate(regions):
        slr = int(region["slr"])
        if by_slr and slr != UNASSIGNED_SLR:
            colour = SLR_COLOURS[by_slr.index(slr) % len(SLR_COLOURS)]
        else:
            colour = REGION_COLOURS[index % len(REGION_COLOURS)]

        for x0, y0, x1, y1 in region["rects"]:
            left = SVG_PAD + x0 * SVG_SCALE
            top = flip(y1)
            width = (x1 - x0 + 1) * SVG_SCALE
            height = (y1 - y0 + 1) * SVG_SCALE
            lines.append(
                f'<rect x="{left}" y="{top}" width="{width}" height="{height}"'
                f' fill="{colour}" fill-opacity="0.55" stroke="#222" stroke-width="1"/>'
            )

        # Label the widest rectangle of the region, and only if the text will fit.
        x0, y0, x1, y1 = max(region["rects"], key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
        width = (x1 - x0 + 1) * SVG_SCALE
        height = (y1 - y0 + 1) * SVG_SCALE
        if width >= MIN_LABEL_W and height >= MIN_LABEL_H:
            label = _escape(
                str(region["name"]).replace("pblock_finn_unit_", "").replace("pblock_Hier_", "")
            )
            cx = SVG_PAD + x0 * SVG_SCALE + width // 2
            cy = flip(y1) + height // 2
            lines.append(
                f'<text x="{cx}" y="{cy + SVG_FONT // 3}" text-anchor="middle"'
                f' font-size="{SVG_FONT}" font-family="sans-serif"'
                f' font-weight="bold">{label}</text>'
            )

    if by_slr:
        legend_y = SVG_PAD - 10
        for position, slr in enumerate(by_slr):
            swatch_x = SVG_PAD + position * 78
            colour = SLR_COLOURS[position % len(SLR_COLOURS)]
            lines.append(
                f'<rect x="{swatch_x}" y="{legend_y - SVG_FONT}" width="{SVG_FONT}"'
                f' height="{SVG_FONT}" fill="{colour}" fill-opacity="0.55"'
                f' stroke="#222" stroke-width="1"/>'
            )
            lines.append(
                f'<text x="{swatch_x + SVG_FONT + 4}" y="{legend_y}"'
                f' font-size="{SVG_FONT}" font-family="sans-serif">SLR{slr}</text>'
            )

    lines.append("</svg>")
    return "\n".join(lines)


def write_floorplan_report(
    model: ModelWrapper,
    tcl_report_path: str | Path,
    json_path: str | Path,
    svg_path: str | Path | None = None,
) -> bool:
    """Write the enriched floorplan report and its diagram into the report directory.

    Args:
        model: The floorplanned model, used to enrich the report.
        tcl_report_path: The JSON the Vivado-side engine wrote.
        json_path: Where to write the enriched report.
        svg_path: Where to write the diagram, if anywhere.

    Returns:
        Whether a report was written. False when the engine produced nothing, which
        is the normal case for a build that did not floorplan.
    """
    source = Path(tcl_report_path)
    if not source.is_file():
        return False
    try:
        report = json.loads(source.read_text())
    except json.JSONDecodeError as error:
        log.warning(f"Floorplanner: report at {source} is not valid JSON ({error}); skipping.")
        return False

    enriched = enrich_floorplan_report(model, report)
    Path(json_path).write_text(json.dumps(enriched, indent=2))

    if svg_path is not None:
        svg = render_floorplan_svg(enriched)
        if svg is not None:
            Path(svg_path).write_text(svg)
    return True

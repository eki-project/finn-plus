"""Analysis pass that reads the per-node out-of-context resource utilization."""

from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from typing import cast

from finn.util.basic import getHWCustomOp
from finn.util.exception import FINNUserError
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.util.vivado import parse_vivado_utilization_report


def node_ooc_res_estimation(
    model: ModelWrapper, place_and_route: bool
) -> dict[str, dict[str, int]]:
    """Read the resource utilization of every node from its out-of-context Vivado run.
    BuildNodeOOC must have run first; it sets the ``ooc_proj_path`` node attribute.
    Nodes without that attribute (e.g. skipped FINNLoop nodes) are left out.

    Args:
        model: Model whose nodes were implemented by BuildNodeOOC.
        place_and_route: Read the utilization after routing if True, otherwise the
            utilization after opt_design.

    Returns:
        Maps node names to the Vivado resource counts (LUT, FF, DSP, BRAM_18K, BRAM_36K,
        URAM, ...). Note that BRAM_18K counts only RAMB18 primitives here, a RAMB36 is
        counted in BRAM_36K.
    """
    report_name = "ooc_utilization.rpt" if place_and_route else "ooc_opt_utilization.rpt"
    res_dict = {}
    for node in model.graph.node:
        if not (is_hls_node(node) or is_rtl_node(node)):
            continue
        proj_path = cast("str", getHWCustomOp(node).get_nodeattr("ooc_proj_path"))
        if proj_path == "":
            continue
        report = Path(proj_path) / report_name
        if not report.is_file():
            log_file = Path(proj_path) / "vivado.log"
            if place_and_route:
                cause = (
                    "BuildNodeOOC probably ran with place_and_route=False, or its Vivado run "
                    f"did not finish (see {log_file})"
                )
            else:
                cause = f"Its Vivado run probably did not finish (see {log_file})"
            raise FINNUserError(f"Node {node.name}: no utilization report at {report}. {cause}.")
        res_dict[node.name] = parse_vivado_utilization_report(report)
    return res_dict

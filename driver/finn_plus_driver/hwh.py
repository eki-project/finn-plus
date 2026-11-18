"""Helpers for reading metadata out of the HWH file accompanying a bitfile."""

import xml.etree.ElementTree as ET
from pathlib import Path


def get_clk_wiz_params_from_hwh(bitfile_name: str) -> dict[str, str]:
    """Parse the HWH file to get clk_wiz_0 parameters.

    PYNQ's ip_dict only contains IPs with an AXI-Lite slave interface, so the
    Clocking Wizard (when configured without AXI-Lite) will not appear there.
    This function reads the HWH XML directly to retrieve its parameters.

    Returns a dict of parameter name -> value strings. The dict is empty when the HWH file
    is missing, unparsable, or contains no Clocking Wizard, so that callers can fall back to
    their configured frequency via ``dict.get(..., default)`` without a None check.
    """
    hwh_path = Path(bitfile_name).with_suffix(".hwh")
    if not hwh_path.exists():
        return {}
    try:
        tree = ET.parse(hwh_path)
        root = tree.getroot()
        for module in root.iter("MODULE"):
            if module.get("INSTANCE") == "clk_wiz_0":
                params = {}
                for param in module.iter("PARAMETER"):
                    params[param.get("NAME")] = param.get("VALUE")
                return params
    except ET.ParseError:
        return {}
    return {}

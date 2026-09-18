import builtins


def max(values):
    return builtins.max(values)


def min(values):
    return builtins.min(values)


def mean(values):
    return builtins.sum(values) / len(values)


def sum(values):
    return builtins.sum(values)


def alltrue(values):
    return builtins.all(values)


def allfalse(values):
    return not builtins.any(values)


def anytrue(values):
    return builtins.any(values)


def anyfalse(values):
    return not builtins.all(values)


# --- microbenchmark helpers -------------------------------------------------------------

#: Hierarchy entries of post_synth_resources.json that belong to the shell, not to a node
SHELL_COMPONENTS = (
    "(top)",
    "top_instrumentation_wrap_0_0",
    "top_axi_interconnect_0_0",
    "top_smartconnect_0_0",
)

#: DUTs whose artifacts predate dut_info.json's dut_node_name: op_type substring to match
LEGACY_DUT_OPTYPE = {"mvau": "MVAU"}


def find_dut_hierarchy(report, node_name=None, op_type=None):
    """Hierarchy key of the DUT node in a post_synth_resources.json report.

    Tries the exact node name, then a substring match on the node name, then a substring
    match on the op_type. Returns None if nothing matches.
    """
    keys = [k for k in report if k not in SHELL_COMPONENTS]
    if node_name:
        if node_name in report:
            return node_name
        for key in keys:
            if node_name in key:
                return key
    if op_type:
        for key in keys:
            if op_type in key:
                return key
    return None


def sum_extra_hierarchies(report, dut_key, resources):
    """Sum the resources of all non-shell hierarchy entries except the DUT (e.g. inserted
    TLastMarker/FIFO/DWC nodes), plus their count under ``num_nodes``."""
    extra = {res: 0 for res in resources}
    num_nodes = 0
    for key, entry in report.items():
        if key in SHELL_COMPONENTS or key == dut_key:
            continue
        num_nodes += 1
        for res in resources:
            value = entry.get(res, 0)
            if isinstance(value, (int, float)):
                extra[res] += value
    extra["num_nodes"] = num_nodes
    return extra

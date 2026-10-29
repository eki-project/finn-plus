import builtins
import json
import os
import re


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

    The node is named ``<op_type>_<index>`` at export; the Zynq shell flow prefixes it with
    the dataflow partition name (``StreamingDataflowPartition_0_MVAU_hls_0``). So the exact
    name is tried first, then a key ending in ``_<node_name>``; legacy artifacts without
    dut_info.json fall back to a substring match on the op_type. A match must be unique:
    None is returned if nothing or more than one key matches (the caller reports an error
    rather than guessing between e.g. the DUT and an inserted node of the same type).
    """
    keys = [k for k in report if k not in SHELL_COMPONENTS]
    if node_name:
        if node_name in report:
            return node_name
        anchored = [key for key in keys if key.endswith("_" + node_name)]
        if len(anchored) == 1:
            return anchored[0]
        if anchored:
            return None
    if op_type:
        hits = [key for key in keys if op_type in key]
        if len(hits) == 1:
            return hits[0]
    return None


def _first_number(line):
    """Float value of the first token of a table line (``|`` separators ignored), or None."""
    tokens = line.replace("|", " ").split()
    if not tokens:
        return None
    try:
        return float(tokens[0])
    except ValueError:
        return None


def parse_wns(report_path):
    """Worst negative slack (ns) of a Vivado timing summary report, None if unavailable.

    The design timing summary is a table whose header starts with ``WNS(ns)  TNS(ns)``,
    followed by a dashed separator line and the value line (WNS first). Kept free of FINN
    imports: the collection runner has no FINN installation.
    """
    try:
        with open(report_path, "r") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    for i, line in enumerate(lines):
        if re.search(r"WNS\(ns\)\s+TNS\(ns\)", line):
            for candidate in lines[i + 1 : i + 6]:
                value = _first_number(candidate)
                if value is not None:
                    return value
            return None
    for line in lines:
        match = re.search(r"Worst Negative Slack:\s*(-?[\d.]+)\s*ns", line)
        if match:
            return float(match.group(1))
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


# --- power measurement reports ---------------------------------------------------------

IDLE_EXPERIMENT = "idle_experiment"
LOAD_EXPERIMENT = "instrumentation_experiment"


def generate_power_report(rails_names, experiment_reports_path):
    """Average the power recordings of every experiment directory of a run.

    ``rails_names`` are the recorded columns (``<rail>_power``, ``total_power``); None
    takes whatever the recordings contain. Returns ``{experiment: {avg_<rail>, min_<rail>,
    max_<rail>}}`` (avg/min/max over the experiment's run files, each file averaged over
    its samples), None if the directory does not exist. Previously the first file of the
    first listed directory was returned, so the logged value was the idle or the load
    experiment depending on the directory order of the filesystem.
    """
    if not os.path.isdir(experiment_reports_path):
        return None
    result = {}
    explicit_rails = rails_names
    for experiment in sorted(os.listdir(experiment_reports_path)):
        exp_dir = os.path.join(experiment_reports_path, experiment)
        if not os.path.isdir(exp_dir):
            continue
        power_measurements = sorted(
            m for m in os.listdir(exp_dir) if m.startswith(experiment) and m.endswith(".json")
        )
        per_file_averages = []
        for file in power_measurements:
            with open(os.path.join(exp_dir, file), "r") as f:
                report = json.load(f)
            rails = report.get("rails", [])
            if not rails:
                continue
            rails_names = explicit_rails or sorted(
                k
                for k in rails[0]
                if k.endswith("_power") and isinstance(rails[0][k], (int, float))
            )
            per_file_averages.append(
                {key: sum(r[key] for r in rails if key in r) / len(rails) for key in rails_names}
            )
        if not per_file_averages:
            continue
        result[experiment] = {
            **{
                "avg_" + key: sum(a[key] for a in per_file_averages) / len(per_file_averages)
                for key in rails_names
            },
            **{"min_" + key: min(a[key] for a in per_file_averages) for key in rails_names},
            **{"max_" + key: max(a[key] for a in per_file_averages) for key in rails_names},
        }
    return result


def power_metrics(per_experiment, rails_names=None):
    """Flatten :func:`generate_power_report` output into metric names.

    ``measurement/power/<experiment>/avg_<rail>`` per experiment, the load experiment
    (``instrumentation_experiment``, else the first non-idle one) additionally under the
    legacy flat names ``measurement/power/avg_<rail>`` that the QoR database reads, and
    ``measurement/power/dyn_<rail>`` = load - idle (the design's dynamic power; the idle
    experiment has the bitstream loaded but the accelerator not running).
    """
    metrics = {}
    for experiment, values in per_experiment.items():
        for name, value in values.items():
            metrics[experiment + "/" + name] = value
    load = LOAD_EXPERIMENT if LOAD_EXPERIMENT in per_experiment else None
    if load is None:
        candidates = [e for e in per_experiment if e != IDLE_EXPERIMENT]
        load = candidates[0] if candidates else None
    if load is not None:
        metrics.update(per_experiment[load])
        if IDLE_EXPERIMENT in per_experiment:
            idle = per_experiment[IDLE_EXPERIMENT]
            if rails_names is None:
                rails_names = [
                    k[len("avg_") :] for k in per_experiment[load] if k.startswith("avg_")
                ]
            for key in rails_names:
                if "avg_" + key in idle:
                    metrics["dyn_" + key] = per_experiment[load]["avg_" + key] - idle["avg_" + key]
    return metrics
